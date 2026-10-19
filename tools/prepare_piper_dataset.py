#!/usr/bin/env python3
"""Turn the ElevenLabs Hebrew dataset into a Piper fine-tuning set whose phonemes are
exactly what the mirror feeds the Hebrew voice at runtime.

    .venv/bin/python tools/prepare_piper_dataset.py [--dataset dataset_he] [--out train_he]
        [--voice-config models/phonikud-tts-he.onnx.json] [--g2p renikud | phonikud]
        [--keep-flagged] [--no-copy-wavs]

Reads DATASET/metadata.csv (LJSpeech: id|text) and DATASET/wavs/{id}.wav (16-bit mono,
at the voice's sample rate). Each text goes through the same G2P as at runtime
(tts.he_g2p: RenikudPlus by default, through RenikudG2P.phonemize) and then the same
filter (mirror.tts.keep_voice_symbols against the voice's phoneme_id_map), and the
phonemes become ids exactly as piper does at runtime (BOS, PAD, each phoneme then PAD,
EOS). Training on these ids with the checkpoint's own phoneme_id_map means the new voice
takes the very same input as today's.

A clip is dropped (and listed in OUT/dropped.tsv) when: its wav is missing or not
16-bit mono at the voice's rate; it is flagged in DATASET/check.tsv (non-empty flags
column; --keep-flagged keeps those); the G2P output is empty; the G2P emitted symbols
outside the phoneme_id_map (the runtime filter would silently drop them, so the
transcript would no longer match the audio); or RenikudG2P dropped a whole word (a
Latin word, say) that is still spoken in the audio.

Writes to OUT (nothing in DATASET is changed):
  metadata_ids.csv       id|text|phoneme ids   piper1-gpl `--data.dataset_type phoneme_ids`
  metadata_phonemes.csv  id|phonemes           for reading / `--data.phoneme_type text`
  phoneme_id_map.json    the checkpoint's map  piper1-gpl `--data.phonemes_path`
  config.json            the voice config      becomes the exported voice's .onnx.json
  samples_ids.csv        a few sentences not in the dataset, for listening mid-training
  wavs/{id}.wav          kept clips (copied; --no-copy-wavs to skip)
  dropped.tsv            id, reason, detail

The training side is deploy/train_piper_hebrew.sh; see docs/HEBREW_VOICE_TRAINING.md.
"""
import argparse
import csv
import json
import os
import shutil
import sys
import wave
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

from mirror.tts import keep_voice_symbols  # noqa: E402

BOS, EOS, PAD = "^", "$", "_"  # piper's phoneme id conventions

# Held-out lines (not in data/voice_he) the training box synthesizes from its checkpoints,
# so progress is judged on sentences the model has never heard.
SAMPLE_SENTENCES = [
    "שלום, אני המראה. מה תרצה לדעת היום?",
    "העתיד שלך כתוב בכוכבים, אבל גם בהחלטות הקטנות של כל בוקר.",
    "כמה זמן לוקח לנסוע מתל אביב לירושלים ברכבת?",
    "אל תפחד מהשאלה. תפחד מהתשובה שאתה כבר יודע.",
    "בשעה שבע וחצי בערב, השמש שוקעת מעל הים.",
]


def phoneme_ids(phonemes: str, id_map: dict) -> list[int]:
    """Piper's phonemes_to_ids (piper/phoneme_ids.py), for a string of single-character
    phonemes that are all in the map: BOS PAD, then each phoneme's ids and PAD, then EOS."""
    ids = list(id_map[BOS]) + list(id_map[PAD])
    for ph in phonemes:
        ids.extend(id_map[ph])
        ids.extend(id_map[PAD])
    ids.extend(id_map[EOS])
    return ids


def read_metadata(path: str) -> list[tuple[str, str]]:
    rows = []
    with open(path, encoding="utf-8", newline="") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if not line.strip():
                continue
            sid, sep, text = line.partition("|")
            if sep:
                rows.append((sid.strip(), text.strip()))
    return rows


def read_flagged(path: str) -> dict[str, str]:
    """{id: flags} for the check.tsv rows whose flags column isn't empty."""
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        return {row["id"]: row["flags"].strip() for row in reader
                if (row.get("flags") or "").strip()}


def wav_problem(path: str, sample_rate: int) -> tuple[str | None, float]:
    """(None, seconds) for a usable wav, else (reason, 0)."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return "missing wav", 0.0
    try:
        with wave.open(path, "rb") as w:
            fmt = (w.getnchannels(), w.getsampwidth(), w.getframerate())
            frames = w.getnframes()
    except (wave.Error, EOFError) as e:
        return f"unreadable wav ({e})", 0.0
    if fmt != (1, 2, sample_rate):
        return (f"wav is {fmt[0]} ch, {8 * fmt[1]}-bit, {fmt[2]} Hz "
                f"(want mono 16-bit {sample_rate} Hz)"), 0.0
    if frames == 0:
        return "empty wav", 0.0
    return None, frames / sample_rate


def phonemize_clip(g2p, text: str, known) -> tuple[str | None, str, str]:
    """(phonemes, reason, detail): phonemes is None when the clip must be dropped."""
    raw = g2p.phonemize(text)
    phonemes = keep_voice_symbols(raw, known)
    unknown = sorted(set(raw) - set(known))
    if unknown:
        return None, "symbols not in phoneme_id_map", " ".join(repr(c) for c in unknown)
    if not phonemes:
        return None, "empty phonemes", raw
    ipa = getattr(g2p, "ipa", None)  # RenikudG2P: every token, voice symbols or not
    if callable(ipa):
        full = ipa(text)
        if len(full.split()) != len(phonemes.split()):
            return None, "G2P dropped word(s)", full
    return phonemes, "", ""


def make_g2p(name: str, renikud_model: str, phonikud_model: str):
    from mirror import tts  # lazy: onnxruntime and the G2P packages

    if name == "renikud":
        return tts.RenikudG2P(renikud_model)
    if name == "phonikud":
        return tts.PhonikudG2P(phonikud_model)
    raise ValueError(f"unknown --g2p {name!r} (renikud | phonikud)")


def prepare(dataset: str, out: str, voice_config: str, g2p, keep_flagged: bool = False,
            copy_wavs: bool = True, log=print) -> dict:
    with open(voice_config, encoding="utf-8") as f:
        config = json.load(f)
    id_map = config["phoneme_id_map"]
    rate = int(config["audio"]["sample_rate"])
    for special in (BOS, EOS, PAD):
        if special not in id_map:
            raise ValueError(f"{voice_config}: phoneme_id_map has no {special!r}")
    # The specials are framing ids, never a phoneme of the transcript.
    known = set(id_map) - {BOS, EOS, PAD}

    rows = read_metadata(os.path.join(dataset, "metadata.csv"))
    flagged = {} if keep_flagged else read_flagged(os.path.join(dataset, "check.tsv"))
    os.makedirs(out, exist_ok=True)
    if copy_wavs:
        os.makedirs(os.path.join(out, "wavs"), exist_ok=True)

    kept, dropped, seconds, used = [], [], 0.0, Counter()
    for sid, text in rows:
        if sid in flagged:
            dropped.append((sid, "flagged in check.tsv", flagged[sid]))
            continue
        wav = os.path.join(dataset, "wavs", f"{sid}.wav")
        problem, dur = wav_problem(wav, rate)
        if problem:
            dropped.append((sid, problem, ""))
            continue
        phonemes, reason, detail = phonemize_clip(g2p, text, known)
        if phonemes is None:
            dropped.append((sid, reason, detail))
            continue
        kept.append((sid, text, phonemes, phoneme_ids(phonemes, id_map)))
        seconds += dur
        used.update(phonemes)
        if copy_wavs:
            shutil.copyfile(wav, os.path.join(out, "wavs", f"{sid}.wav"))

    # csv.writer quotes a field only when it must (a text starting with '"', say) and
    # piper1-gpl reads these files with csv.reader(delimiter="|"), so they round-trip.
    with open(os.path.join(out, "metadata_ids.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="|", lineterminator="\n")
        for sid, text, _ph, ids in kept:
            w.writerow([sid, text, " ".join(map(str, ids))])
    with open(os.path.join(out, "metadata_phonemes.csv"), "w", encoding="utf-8",
              newline="") as f:
        w = csv.writer(f, delimiter="|", lineterminator="\n")
        for sid, _text, ph, _ids in kept:
            w.writerow([sid, ph])
    with open(os.path.join(out, "phoneme_id_map.json"), "w", encoding="utf-8") as f:
        json.dump(id_map, f, ensure_ascii=False, indent=2)
    voice = dict(config, dataset=os.path.basename(os.path.normpath(dataset)))
    with open(os.path.join(out, "config.json"), "w", encoding="utf-8") as f:
        json.dump(voice, f, ensure_ascii=False, indent=4)
    with open(os.path.join(out, "samples_ids.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="|", lineterminator="\n")
        for n, text in enumerate(SAMPLE_SENTENCES, start=1):
            phonemes, _reason, _detail = phonemize_clip(g2p, text, known)
            if phonemes:
                w.writerow([f"sample_{n}", text, " ".join(map(str, phoneme_ids(phonemes, id_map)))])
    with open(os.path.join(out, "dropped.tsv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["id", "reason", "detail"])
        w.writerows(dropped)

    summary = {
        "total": len(rows), "kept": len(kept), "dropped": len(dropped),
        "minutes": round(seconds / 60, 1),
        "reasons": dict(Counter(r for _id, r, _d in dropped)),
        "symbols_used": "".join(sorted(used)),
        "max_id": max((max(ids) for *_x, ids in kept), default=0),
    }
    log(f"{summary['kept']} of {summary['total']} clips kept "
        f"({summary['minutes']} min of audio), {summary['dropped']} dropped")
    for reason, n in Counter(r for _id, r, _d in dropped).most_common():
        ids = [sid for sid, r, _d in dropped if r == reason]
        more = f" (+{len(ids) - 8} more)" if len(ids) > 8 else ""
        log(f"  {n:4d}  {reason}: {' '.join(ids[:8])}{more}")
    log(f"phonemes used ({len(used)}): {summary['symbols_used']}")
    log(f"wrote {out}/metadata_ids.csv, metadata_phonemes.csv, phoneme_id_map.json, "
        f"config.json, samples_ids.csv, dropped.tsv{', wavs/' if copy_wavs else ''}")
    if summary["minutes"] < 30:
        log("WARNING: under 30 minutes of audio; expect a weaker fine-tune")
    return summary


def main(argv=None) -> int:
    from mirror.config import TtsConfig

    d = TtsConfig()
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dataset", default="dataset_he")
    p.add_argument("--out", default="train_he")
    p.add_argument("--voice-config", default=f"{d.phonikud_voice_he}.json",
                   help="the checkpoint's voice config (its phoneme_id_map and rate)")
    p.add_argument("--g2p", default=d.he_g2p, choices=["renikud", "phonikud"],
                   help="must match tts.he_g2p of the mirror that will use the voice")
    p.add_argument("--renikud-model", default=d.renikud_model)
    p.add_argument("--phonikud-model", default=d.phonikud_model)
    p.add_argument("--keep-flagged", action="store_true",
                   help="keep clips flagged in check.tsv")
    p.add_argument("--no-copy-wavs", dest="copy_wavs", action="store_false",
                   help="don't copy the kept wavs into OUT/wavs")
    args = p.parse_args(argv)
    if not os.path.exists(os.path.join(args.dataset, "metadata.csv")):
        print(f"no {args.dataset}/metadata.csv", file=sys.stderr)
        return 2
    g2p = make_g2p(args.g2p, args.renikud_model, args.phonikud_model)
    summary = prepare(args.dataset, args.out, args.voice_config, g2p,
                      keep_flagged=args.keep_flagged, copy_wavs=args.copy_wavs)
    return 0 if summary["kept"] else 1


if __name__ == "__main__":
    sys.exit(main())
