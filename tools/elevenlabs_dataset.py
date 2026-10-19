#!/usr/bin/env python3
"""Make a Piper/LJSpeech-style Hebrew TTS training set with the ElevenLabs API.

    ELEVENLABS_API_KEY=... python tools/elevenlabs_dataset.py data/voice_he/sentences_*.txt \\
        --voice-id VOICE_ID --out dataset_he [--model eleven_v3] [--limit N]
        [--language-code he] [--stability 0.5] [--similarity 0.75] [--style 0.0]
        [--concurrency 2] [--dry-run | --metadata-only | --check [--stt-url URL]]

Reads the sentence files (one per line; blank lines and lines starting with # are
skipped), dedupes them in order and gives each an id (he_0001, he_0002...). The ids
live in OUT/sentences.tsv: it is written on the first run and reused after, so ids
never shift; sentences that are new on a later run get the next free ids.

--dry-run makes no API call: it prints the sentence count, characters, an estimated
credit cost (per-model assumption, check your plan) and a rough audio duration.

A real run POSTs each sentence to /v1/text-to-speech/{voice_id}?output_format=pcm_22050
and saves OUT/wavs/{id}.wav (16-bit mono, 22050 Hz, Piper's rate). It is resume-safe
(non-empty wavs are skipped), retries 429/5xx with backoff and stops cleanly on an
auth or quota error. Then it writes OUT/metadata.csv (LJSpeech: id|text) for every
sentence with a wav; --metadata-only does only that.

--check measures every wav (duration, chars/s of the speech, leading/trailing silence, clipping),
writes OUT/check.tsv and prints a summary of flagged clips. With --stt-url it also
transcribes each clip with whisper-server forced to Hebrew and flags CER > 0.25.

The API key is read from ELEVENLABS_API_KEY and is never printed or stored.
"""
import argparse
import csv
import glob
import io
import os
import re
import sys
import threading
import time
import unicodedata
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.insert(0, REPO)

API_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
OUTPUT_FORMAT = "pcm_22050"
SAMPLE_RATE = 22050
DEFAULT_MODEL = "eleven_v3"  # Hebrew: eleven_multilingual_v2 does not support it
ID_PREFIX = "he_"
MAX_TRIES = 5
TIMEOUT_S = 60.0
# Credits per character, as published when this was written; plans differ.
CREDITS_PER_CHAR = {
    "eleven_multilingual_v2": 1.0,
    "eleven_v3": 1.0,
    "eleven_turbo_v2_5": 0.5,
    "eleven_flash_v2_5": 0.5,
}
HEBREW_CHARS_PER_S = 14.0  # rough speaking rate, for the duration estimate

# --check thresholds
CPS_RANGE = (6.0, 25.0)  # the oracle speaks slowly: ~8-10 characters per second
MIN_DURATION_S = 0.5
MAX_EDGE_SILENCE_S = 1.0
SILENCE_DBFS = -40.0
CLIP_LEVEL = 32700
MIN_CLIPPED_SAMPLES = 10
MAX_CER = 0.25


class FatalAPIError(Exception):
    """Auth or quota problem: no point in sending more requests."""


class SentenceError(Exception):
    """This sentence failed (bad request, retries exhausted); the others go on."""


# --- sentences and ids ---------------------------------------------------------

def clean_line(line: str) -> str:
    """Collapse whitespace (tabs too, for the TSV) and drop '|' (LJSpeech separator)."""
    return " ".join(line.replace("|", " ").split())


def load_sentences(paths: list[str]) -> list[str]:
    """Sentences of the files in order, deduped, without blanks and # comments."""
    seen, out = set(), []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for raw in f:
                if raw.strip().startswith("#"):
                    continue
                line = clean_line(raw)
                if line and line not in seen:
                    seen.add(line)
                    out.append(line)
    return out


def format_id(n: int) -> str:
    return f"{ID_PREFIX}{n:04d}"


def read_manifest(path: str) -> list[tuple[str, str]]:
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line:
                sid, text = line.split("\t", 1)
                entries.append((sid, text))
    return entries


def write_atomic(path: str, data: bytes) -> None:
    tmp = f"{path}.tmp{os.getpid()}.{threading.get_ident()}"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def merge_manifest(entries: list[tuple[str, str]],
                   sentences: list[str]) -> list[tuple[str, str]]:
    """entries plus the sentences not in them, with the next free ids."""
    merged = list(entries)
    known = {text for _, text in merged}
    next_n = max((int(sid[len(ID_PREFIX):]) for sid, _ in merged), default=0) + 1
    for text in sentences:
        if text not in known:
            merged.append((format_id(next_n), text))
            known.add(text)
            next_n += 1
    return merged


def manifest_path(out_dir: str) -> str:
    return os.path.join(out_dir, "sentences.tsv")


def load_manifest(out_dir: str, sentences: list[str]) -> list[tuple[str, str]]:
    """OUT/sentences.tsv merged with the sentences, without writing anything."""
    path = manifest_path(out_dir)
    return merge_manifest(read_manifest(path) if os.path.exists(path) else [], sentences)


def update_manifest(out_dir: str, sentences: list[str]) -> list[tuple[str, str]]:
    """All (id, text) of OUT/sentences.tsv, after appending the new sentences with
    the next ids. Existing ids never change."""
    path = manifest_path(out_dir)
    old = read_manifest(path) if os.path.exists(path) else None
    entries = merge_manifest(old or [], sentences)
    if entries != old:
        os.makedirs(out_dir, exist_ok=True)
        body = "".join(f"{sid}\t{text}\n" for sid, text in entries)
        write_atomic(path, body.encode("utf-8"))
    return entries


def select(entries: list[tuple[str, str]], sentences: list[str],
           limit: "int | None") -> list[tuple[str, str]]:
    """The manifest entries of the current sentences, in id order, at most limit."""
    current = set(sentences)
    chosen = [e for e in entries if e[1] in current]
    return chosen[:limit] if limit else chosen


# --- dry run -------------------------------------------------------------------

def estimate(entries: list[tuple[str, str]], model: str) -> dict:
    chars = sum(len(text) for _, text in entries)
    per_char = CREDITS_PER_CHAR.get(model, 1.0)
    return {"sentences": len(entries), "chars": chars, "credits_per_char": per_char,
            "known_model": model in CREDITS_PER_CHAR, "credits": chars * per_char,
            "audio_s": chars / HEBREW_CHARS_PER_S}


def print_estimate(est: dict, model: str) -> None:
    note = "" if est["known_model"] else " (unknown model, assumed)"
    print(f"sentences: {est['sentences']}")
    print(f"characters: {est['chars']}")
    print(f"credits: ~{est['credits']:.0f} (assuming {est['credits_per_char']:g} credit"
          f" per character for {model}{note}; check your plan)")
    print(f"audio: ~{est['audio_s'] / 60:.1f} min (rough estimate at"
          f" {HEBREW_CHARS_PER_S:g} characters per second of Hebrew speech)")


# --- synthesis -----------------------------------------------------------------

def build_request(voice_id: str, text: str, model: str, settings: dict,
                  language_code: "str | None") -> tuple[str, dict, dict]:
    """(url, params, json body) of one text-to-speech request."""
    body = {"text": text, "model_id": model, "voice_settings": dict(settings)}
    if language_code:
        body["language_code"] = language_code
    return API_URL.format(voice_id=voice_id), {"output_format": OUTPUT_FORMAT}, body


def _is_quota(resp) -> bool:
    return "quota" in (resp.text or "").lower()


def synthesize(api_key: str, voice_id: str, text: str, model: str, settings: dict,
               language_code: "str | None" = None, sleep=time.sleep) -> bytes:
    """Raw 16-bit PCM of one sentence. Retries 429/5xx and network errors with
    exponential backoff; FatalAPIError on auth/quota, SentenceError otherwise."""
    url, params, body = build_request(voice_id, text, model, settings, language_code)
    headers = {"xi-api-key": api_key, "Content-Type": "application/json",
               "Accept": "audio/pcm"}
    reason = ""
    for attempt in range(MAX_TRIES):
        try:
            resp = requests.post(url, params=params, json=body, headers=headers,
                                 timeout=TIMEOUT_S)
        except requests.exceptions.RequestException as e:
            reason = f"network error: {type(e).__name__}"
        else:
            if resp.status_code == 200:
                return resp.content
            if resp.status_code in (401, 402, 403) or _is_quota(resp):
                raise FatalAPIError(
                    f"HTTP {resp.status_code}: {resp.text[:300]}\n"
                    "Check ELEVENLABS_API_KEY, the voice id and your remaining credits.")
            if resp.status_code != 429 and resp.status_code < 500:
                raise SentenceError(f"HTTP {resp.status_code}: {resp.text[:300]}")
            reason = f"HTTP {resp.status_code}"
            retry_after = resp.headers.get("Retry-After", "")
            if retry_after.isdigit() and attempt < MAX_TRIES - 1:
                sleep(min(float(retry_after), 60.0))
                continue
        if attempt < MAX_TRIES - 1:
            sleep(2.0 ** attempt)
    raise SentenceError(f"gave up after {MAX_TRIES} tries ({reason})")


def pcm_to_wav(pcm: bytes, sr: int = SAMPLE_RATE) -> bytes:
    pcm = pcm[: len(pcm) - len(pcm) % 2]
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)
    return buf.getvalue()


def wav_path(out_dir: str, sid: str) -> str:
    return os.path.join(out_dir, "wavs", f"{sid}.wav")


def has_wav(out_dir: str, sid: str) -> bool:
    path = wav_path(out_dir, sid)
    return os.path.exists(path) and os.path.getsize(path) > 44  # more than a header


def generate(entries: list[tuple[str, str]], out_dir: str, api_key: str, voice_id: str,
             model: str, settings: dict, language_code: "str | None" = None,
             concurrency: int = 2, sleep=time.sleep) -> dict:
    """Synthesize every entry without a wav. Returns counts; raises FatalAPIError
    (after the in-flight requests finish) on an auth or quota error."""
    os.makedirs(os.path.join(out_dir, "wavs"), exist_ok=True)
    todo = [(sid, text) for sid, text in entries if not has_wav(out_dir, sid)]
    stats = {"skipped": len(entries) - len(todo), "made": 0, "failed": 0}
    stop = threading.Event()
    fatal: list[FatalAPIError] = []

    def one(sid, text):
        if stop.is_set():
            return None
        try:
            pcm = synthesize(api_key, voice_id, text, model, settings, language_code,
                             sleep=sleep)
        except FatalAPIError as e:
            stop.set()
            fatal.append(e)
            return None
        if not pcm:
            raise SentenceError("empty audio")
        write_atomic(wav_path(out_dir, sid), pcm_to_wav(pcm))
        return sid

    total = len(todo)
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = {pool.submit(one, sid, text): sid for sid, text in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            sid = futures[fut]
            try:
                if fut.result():
                    stats["made"] += 1
                    print(f"[{i}/{total}] {sid}", flush=True)
            except SentenceError as e:
                stats["failed"] += 1
                print(f"[{i}/{total}] {sid} FAILED: {e}", file=sys.stderr, flush=True)
    if fatal:
        raise fatal[0]
    return stats


# --- metadata ------------------------------------------------------------------

def wav_duration(path: str) -> float:
    with wave.open(path, "rb") as w:
        return w.getnframes() / float(w.getframerate())


def write_metadata(entries: list[tuple[str, str]], out_dir: str) -> tuple[int, float]:
    """OUT/metadata.csv (id|text) for the entries with a wav; (clips, seconds)."""
    lines, seconds = [], 0.0
    for sid, text in entries:
        if has_wav(out_dir, sid):
            lines.append(f"{sid}|{text}\n")
            seconds += wav_duration(wav_path(out_dir, sid))
    write_atomic(os.path.join(out_dir, "metadata.csv"), "".join(lines).encode("utf-8"))
    return len(lines), seconds


# --- check ---------------------------------------------------------------------

def read_wav(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return pcm, sr


def edge_silence(pcm: np.ndarray, sr: int) -> tuple[float, float]:
    """Seconds of leading and trailing silence (10 ms frames below SILENCE_DBFS)."""
    frame = max(1, sr // 100)
    n = len(pcm) // frame
    if n == 0:
        return len(pcm) / sr, 0.0
    frames = pcm[: n * frame].astype(np.float64).reshape(n, frame) / 32768.0
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    loud = np.nonzero(rms > 10 ** (SILENCE_DBFS / 20))[0]
    if len(loud) == 0:
        return len(pcm) / sr, 0.0
    return loud[0] * frame / sr, (n - 1 - loud[-1]) * frame / sr


def normalize_for_cer(text: str) -> str:
    """Letters and digits only (no niqqud, punctuation), single spaces."""
    text = "".join(ch for ch in unicodedata.normalize("NFD", text)
                   if not unicodedata.category(ch).startswith("M"))
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return " ".join(text.split())


def cer(ref: str, hyp: str) -> float:
    """Character error rate: Levenshtein(ref, hyp) / len(ref), on normalized text."""
    ref, hyp = normalize_for_cer(ref), normalize_for_cer(hyp)
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1] / len(ref)


def resample(pcm: np.ndarray, sr: int, target: int = 16000) -> np.ndarray:
    """Float samples in [-1, 1] at target rate (linear interpolation; enough for STT)."""
    x = pcm.astype(np.float32) / 32768.0
    if sr == target or len(x) == 0:
        return x
    n = int(round(len(x) * target / sr))
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


def check_clip(text: str, pcm: np.ndarray, sr: int, stt=None) -> dict:
    duration = len(pcm) / sr
    lead, trail = edge_silence(pcm, sr)
    speech = duration - lead - trail  # the rate of the speech, not of the silences
    cps = len(text) / speech if speech > 0 else float("inf")
    clipped = int(np.count_nonzero(np.abs(pcm.astype(np.int32)) >= CLIP_LEVEL))
    flags = []
    if duration < MIN_DURATION_S:
        flags.append("short")
    if not CPS_RANGE[0] <= cps <= CPS_RANGE[1]:
        flags.append("slow" if cps < CPS_RANGE[0] else "fast")
    if lead > MAX_EDGE_SILENCE_S:
        flags.append("lead_silence")
    if trail > MAX_EDGE_SILENCE_S:
        flags.append("trail_silence")
    if clipped >= MIN_CLIPPED_SAMPLES:
        flags.append("clipping")
    row = {"duration_s": duration, "cps": cps, "lead_s": lead, "trail_s": trail,
           "clipped": clipped, "cer": None, "heard": ""}
    if stt is not None:
        heard = stt.transcribe(resample(pcm, sr), sr=16000, language="he").text
        row["heard"], row["cer"] = heard, cer(text, heard)
        # whisper writes numbers as digits ("2026") where the text spells them out:
        # no misread check on those clips
        if row["cer"] > MAX_CER and not re.search(r"\d", heard):
            flags.append("cer")
    row["flags"] = flags
    return row


def run_check(entries: list[tuple[str, str]], out_dir: str, stt=None) -> list[dict]:
    """Check every entry with a wav, write OUT/check.tsv, return the rows."""
    rows = []
    for sid, text in entries:
        if not has_wav(out_dir, sid):
            continue
        pcm, sr = read_wav(wav_path(out_dir, sid))
        try:
            row = check_clip(text, pcm, sr, stt)
        except Exception as e:  # STT down for one clip: still report the rest
            row = check_clip(text, pcm, sr)
            row["flags"].append(f"stt_error:{type(e).__name__}")
        rows.append({"id": sid, "text": text, **row})
    path = os.path.join(out_dir, "check.tsv")
    cols = ["id", "duration_s", "cps", "lead_s", "trail_s", "clipped", "cer", "flags",
            "text", "heard"]
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(cols)
        for r in rows:
            w.writerow([r["id"], f"{r['duration_s']:.2f}", f"{r['cps']:.1f}",
                        f"{r['lead_s']:.2f}", f"{r['trail_s']:.2f}", r["clipped"],
                        "" if r["cer"] is None else f"{r['cer']:.2f}",
                        ",".join(r["flags"]), r["text"], r["heard"]])
    os.replace(tmp, path)
    return rows


def print_check_summary(rows: list[dict], out_dir: str) -> None:
    flagged = [r for r in rows if r["flags"]]
    total = sum(r["duration_s"] for r in rows)
    print(f"checked {len(rows)} clips, {total / 60:.1f} min; {len(flagged)} flagged"
          f" (see {os.path.join(out_dir, 'check.tsv')})")
    counts: dict[str, int] = {}
    for r in flagged:
        for flag in r["flags"]:
            counts[flag] = counts.get(flag, 0) + 1
    for flag, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {flag}: {n}")
    for r in flagged[:20]:
        print(f"  {r['id']} [{','.join(r['flags'])}] {r['text']}")


# --- main ----------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("sentence_files", nargs="+", help="text files, one sentence per line")
    p.add_argument("--voice-id", help="ElevenLabs voice id (needed for a real run)")
    p.add_argument("--out", required=True, help="dataset directory")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--limit", type=int, help="only the first N sentences (by id)")
    p.add_argument("--language-code", default="he",
                   help="ISO code sent as language_code (default he; pass '' to omit it)")
    p.add_argument("--stability", type=float, default=0.5)
    p.add_argument("--similarity", type=float, default=0.75, help="similarity_boost")
    p.add_argument("--style", type=float, default=0.0)
    p.add_argument("--no-speaker-boost", action="store_true")
    p.add_argument("--concurrency", type=int, default=2, help="parallel requests")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="estimate only, no API calls")
    mode.add_argument("--metadata-only", action="store_true")
    mode.add_argument("--check", action="store_true", help="flag suspicious clips")
    p.add_argument("--stt-url", help="with --check: whisper-server URL to measure CER")
    return p.parse_args(argv)


def expand(patterns: list[str]) -> list[str]:
    paths = []
    for pat in patterns:
        hits = sorted(glob.glob(pat))
        paths.extend(hits if hits else [pat])
    return paths


def main(argv=None) -> int:
    args = parse_args(argv)
    sentences = load_sentences(expand(args.sentence_files))
    if args.dry_run:  # touch nothing on disk
        entries = load_manifest(args.out, sentences)
        todo = [e for e in select(entries, sentences, args.limit)
                if not has_wav(args.out, e[0])]
        print_estimate(estimate(todo, args.model), args.model)
        return 0

    manifest = update_manifest(args.out, sentences)
    entries = select(manifest, sentences, args.limit)
    if args.check:
        stt = None
        if args.stt_url:
            from mirror.stt import WhisperServerSTT
            stt = WhisperServerSTT(args.stt_url, timeout_s=30.0, language="he")
        print_check_summary(run_check(entries, args.out, stt), args.out)
        return 0

    code = 0
    if not args.metadata_only:
        api_key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
        if not api_key:
            print("ELEVENLABS_API_KEY is not set", file=sys.stderr)
            return 2
        if not args.voice_id:
            print("--voice-id is required", file=sys.stderr)
            return 2
        settings = {"stability": args.stability, "similarity_boost": args.similarity,
                    "style": args.style, "use_speaker_boost": not args.no_speaker_boost}
        try:
            stats = generate(entries, args.out, api_key, args.voice_id, args.model,
                             settings, args.language_code, args.concurrency)
            print(f"made {stats['made']}, skipped {stats['skipped']} (already there),"
                  f" failed {stats['failed']}")
            code = 1 if stats["failed"] else 0
        except FatalAPIError as e:
            print(f"stopping: {e}", file=sys.stderr)
            code = 3
    clips, seconds = write_metadata(select(manifest, sentences, None), args.out)
    print(f"metadata.csv: {clips} clips, {seconds / 60:.1f} min of audio")
    return code


if __name__ == "__main__":
    sys.exit(main())
