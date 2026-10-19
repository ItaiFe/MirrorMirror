#!/usr/bin/env python3
"""Record an offline, full demo of the mirror to MP4: no real audio devices.

    # 1. synthesize the question WAVs with the repo's own voices (no servers needed)
    python tools/record_demo.py questions OUT_DIR
    # 2. with whisper-server :8081 and llama-server :8080 up (make servers-mac)
    python tools/record_demo.py record WORK_DIR Q1.wav [Q2.wav ...]
    # 3. servers may be stopped again
    python tools/record_demo.py render WORK_DIR OUT.mp4

`record` runs the REAL pipeline on each question WAV, one conversation: Silero VAD +
Segmenter -> WhisperServerSTT -> wake matching -> Controller with LlamaServerLLM
(streaming), FillerBank and TTSRouter.synth_visemes. Only the speaker and the face
bus are replaced: TimelinePlayer appends each clip to an audio timeline and calls
on_level at the clip's own 20 ms hops without sleeping, and TimelineFace records
(time, state, mouth, viseme) against that timeline. While the controller works
(STT, waiting for the LLM's first sentence), the real wall-clock time passes on the
timeline as silence, so the video shows the real wait (the thinking face); a single
gap is capped at GAP_CAP_S (noted in the log and in events.json). WORK_DIR gets
timeline.wav and events.json.

`render` replays events.json like mirror-face does (latest message per frame,
step_mouth, SpriteFace) at 30 fps, 504x504 (2x), burns subtitles (what was heard
at the top, what the oracle said at the bottom) and muxes timeline.wav with ffmpeg.
English subtitles use cv2.putText. Hebrew ones need python-bidi (for RTL order) and
a Hebrew font (ArialHB); without python-bidi they are skipped, with a warning.

`record` and `render` are separate so the servers can be stopped before rendering
(this matters on an 8 GB Mac).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mirror.audio import CHUNK, mouth_envelope, resample  # noqa: E402
from mirror.conversation import HOP_S  # noqa: E402
from mirror.states import State  # noqa: E402

log = logging.getLogger("record_demo")

GAP_CAP_S = 6.0      # longest silence one measured wait may put on the timeline
TIMELINE_SR = 22050  # the timeline's audio rate (clips are resampled to it)
LEAD_IN_S = 1.0      # idle face before the first question
FOLLOWUP_PAUSE_S = 1.0  # the visitor's pause before a follow-up question
TAIL_S = 2.5         # after the last reply
FPS = 30
SCALE = 2

QUESTIONS = {
    "en_wake": ("en", "Mirror, mirror on the wall, what does the future hold for me?"),
    "en_followup": ("en", "And what should I carry forward?"),
    "he_wake": ("he", "מראה, מראה שעל הקיר, מה צופן לי העתיד?"),
}


# --------------------------------------------------------------------------
# timeline, fake player, fake face sink (no servers, unit-tested)
# --------------------------------------------------------------------------
class Timeline:
    """Mono audio timeline plus a clock that maps wall time onto it.

    `cursor` is where the next audio goes. Outside `anchor()` the clock is frozen at
    the cursor. Inside it (while the controller handles one segment), wall time
    since the anchor passes on the timeline, but never before the cursor (audio
    already "playing") and never more than `cap_s` past it; a longer wait is cut
    to cap_s and remembered in `capped`.
    """

    def __init__(self, sr: int = TIMELINE_SR, cap_s: float = GAP_CAP_S,
                 wall=time.monotonic):
        self.sr = sr
        self.cap_s = cap_s
        self._wall = wall
        self._chunks: list[np.ndarray] = []
        self._n = 0  # samples so far
        self._anchor = None  # (timeline s, wall s)
        self._shift = 0.0  # timeline seconds removed by capping
        self._play_t = None  # set while a clip's on_level callbacks run
        self.gaps: list[dict] = []
        self.capped: list[dict] = []

    @property
    def cursor(self) -> float:
        return self._n / self.sr

    def append(self, samples: np.ndarray, sr: int) -> float:
        """Append audio at the cursor; returns its start time."""
        start = self.cursor
        x = resample(np.asarray(samples, np.float32), sr, self.sr)
        self._chunks.append(x)
        self._n += len(x)
        return start

    def silence(self, seconds: float) -> None:
        n = int(round(max(0.0, seconds) * self.sr))
        if n:
            self._chunks.append(np.zeros(n, np.float32))
            self._n += n

    def anchor(self) -> None:
        """From now on, wall time passes on the timeline (from the cursor)."""
        self._anchor = (self.cursor, self._wall())
        self._shift = 0.0

    def release(self) -> None:
        self._anchor = None

    def _mapped(self) -> float:
        tl, w = self._anchor
        return tl + (self._wall() - w) - self._shift

    def now(self) -> float:
        if self._play_t is not None:
            return self._play_t
        if self._anchor is None:
            return self.cursor
        return min(max(self.cursor, self._mapped()), self.cursor + self.cap_s)

    def catch_up(self, label: str) -> float:
        """Fill the cursor up to now() with silence (the measured wait); returns the
        silence inserted. A wait over cap_s is cut to cap_s and noted."""
        if self._anchor is None:
            return 0.0
        want = self._mapped() - self.cursor
        if want > self.cap_s:
            self._shift += want - self.cap_s
            self.capped.append({"at": round(self.cursor, 3), "label": label,
                                "measured_s": round(want, 3), "kept_s": self.cap_s})
            log.warning("gap before %s: %.2f s measured, capped at %.1f s",
                        label, want, self.cap_s)
            want = self.cap_s
        if want <= 0:
            return 0.0
        self.gaps.append({"at": round(self.cursor, 3), "label": label,
                          "seconds": round(want, 3)})
        self.silence(want)
        return want

    def play_clock(self, t) -> None:
        self._play_t = t

    def audio(self) -> np.ndarray:
        return np.concatenate(self._chunks) if self._chunks else np.zeros(0, np.float32)


class TimelinePlayer:
    """Player stand-in: puts the clip on the timeline (after the measured wait) and
    calls on_level once per 20 ms hop on the clip's own timeline, then 0.0, without
    sleeping. `texts` maps id(samples) -> what the clip says (for subtitles)."""

    def __init__(self, timeline: Timeline, texts: dict | None = None):
        self.tl = timeline
        self.texts = texts if texts is not None else {}
        self.clips: list[dict] = []

    def play(self, samples, sr, on_level) -> None:
        kind, text = self.texts.get(id(samples), ("clip", ""))
        self.tl.catch_up(f"{kind} {len(self.clips) + 1}")
        env = mouth_envelope(samples, sr, HOP_S)
        start = self.tl.append(samples, sr)
        end = self.tl.cursor
        try:
            for i, level in enumerate(env):
                self.tl.play_clock(start + i * HOP_S)
                on_level(float(level))
            self.tl.play_clock(end)
            on_level(0.0)
        finally:
            self.tl.play_clock(None)
        self.clips.append({"start": round(start, 3), "end": round(end, 3),
                           "kind": kind, "text": text})


class TimelineFace:
    """FaceBus stand-in: records (time, state, mouth, viseme) at the timeline clock."""

    def __init__(self, timeline: Timeline):
        self.tl = timeline
        self.events: list[tuple[float, str, float, str]] = []

    def send(self, state: State, mouth: float, viseme: str = "X") -> None:
        self.events.append((round(self.tl.now(), 4), State(state).value,
                            round(float(mouth), 4), str(viseme)))

    def close(self) -> None:
        pass


class _NullLights:
    def set_state(self, state) -> None:
        pass


def face_message_at(events, t, start=0):
    """Like FaceBus.poll: the latest event in events[start:] with time <= t (None if
    no new one), and the index to resume from (events sorted by time)."""
    i, latest = start, None
    while i < len(events) and events[i][0] <= t:
        latest = events[i]
        i += 1
    return latest, i


# --------------------------------------------------------------------------
# timing wrappers around the real components
# --------------------------------------------------------------------------
class TimedSTT:
    def __init__(self, stt):
        self._stt = stt
        self.calls: list[dict] = []

    def transcribe(self, samples, sr: int = 16000):
        t0 = time.monotonic()
        tr = self._stt.transcribe(samples, sr)
        dt = time.monotonic() - t0
        self.calls.append({"seconds": round(dt, 3), "text": tr.text, "lang": tr.lang})
        log.info("STT %.2f s: (%s) %r", dt, tr.lang, tr.text)
        return tr


class TimedLLM:
    def __init__(self, llm):
        self._llm = llm
        self.calls: list[dict] = []

    def warm(self, lang):
        return self._llm.warm(lang)

    def reply_stream(self, history, user_text, lang, cancel=None):
        t0 = time.monotonic()
        call = {"user": user_text, "lang": lang, "first_sentence_s": None,
                "sentences": [], "sentence_s": []}
        self.calls.append(call)
        for s in self._llm.reply_stream(history, user_text, lang, cancel=cancel):
            dt = time.monotonic() - t0
            if call["first_sentence_s"] is None:
                call["first_sentence_s"] = round(dt, 3)
                log.info("LLM first sentence after %.2f s", dt)
            call["sentences"].append(s)
            call["sentence_s"].append(round(dt, 3))
            yield s
        call["total_s"] = round(time.monotonic() - t0, 3)


class TextTTS:
    """TTSRouter wrapper that remembers which text each returned clip says."""

    def __init__(self, tts, texts: dict, kind: str = "filler"):
        self._tts = tts
        self.texts = texts
        self.kind = kind  # "filler" while FillerBank.prepare runs, then "reply"
        self._keep: list = []  # keep the arrays alive so their ids stay unique
        self.synth_s: list[float] = []

    def synth_visemes(self, text, lang):
        t0 = time.monotonic()
        samples, sr, track = self._tts.synth_visemes(text, lang)
        if self.kind == "reply":
            self.synth_s.append(round(time.monotonic() - t0, 3))
        self.texts[id(samples)] = (self.kind, text)
        self._keep.append(samples)
        return samples, sr, track

    def synth(self, text, lang):
        return self._tts.synth(text, lang)

    def warmup(self):
        self._tts.warmup()


# --------------------------------------------------------------------------
# audio files
# --------------------------------------------------------------------------
def read_wav(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, "rb") as w:
        n_ch, width, rate = w.getnchannels(), w.getsampwidth(), w.getframerate()
        raw = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError(f"{path}: only 16-bit PCM WAV is supported")
    audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    return audio.reshape(-1, n_ch).mean(axis=1), rate


def write_wav(path: str, samples: np.ndarray, sr: int) -> None:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def _setup_env(cfg) -> None:
    models_dir = os.path.abspath(cfg.models_dir)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HOME"] = os.path.join(models_dir, "hf")


# --------------------------------------------------------------------------
# questions
# --------------------------------------------------------------------------
def cmd_questions(args) -> None:
    from mirror.config import load_config
    from mirror.voice_main import load_tts

    cfg = load_config(args.config)
    _setup_env(cfg)
    tts, _, _ = load_tts(cfg.tts)
    os.makedirs(args.out_dir, exist_ok=True)
    for name, (lang, text) in QUESTIONS.items():
        samples, sr, _ = tts.synth_visemes(text, lang)
        pad = np.zeros(int(0.2 * sr), np.float32)
        path = os.path.join(args.out_dir, f"{name}.wav")
        write_wav(path, np.concatenate([pad, samples, pad]), sr)
        print(f"{path}: {len(samples) / sr:.2f} s  {text}")


# --------------------------------------------------------------------------
# record
# --------------------------------------------------------------------------
def feed_question(timeline: Timeline, segmenter, vad, audio: np.ndarray, sr: int,
                  vad_sr: int = 16000):
    """Put the question on the timeline and run it through the VAD in CHUNK reads
    (unpaced). Returns (question start, segment or None). The cursor ends where the
    segmenter cut the utterance (its end-of-speech silence included)."""
    segmenter.reset()
    vad.reset()
    start = timeline.cursor
    tail = np.zeros(int((segmenter.min_silence_s + 0.5) * sr), np.float32)
    full = np.concatenate([audio, tail])
    x16 = resample(full, sr, vad_sr)
    for pos in range(0, len(x16), CHUNK):
        chunk = x16[pos:pos + CHUNK]
        if len(chunk) < CHUNK:
            chunk = np.concatenate([chunk, np.zeros(CHUNK - len(chunk), np.float32)])
        segs = segmenter.feed(chunk)
        if segs:
            cut = (pos + CHUNK) / vad_sr
            timeline.append(full[:int(round(cut * sr))], sr)
            return start, segs[0]
    timeline.append(full, sr)
    return start, None


def cmd_record(args) -> None:
    from mirror.config import load_config, resolve_repo_path
    from mirror.conversation import Controller
    from mirror.fillers import FillerBank
    from mirror.llm import LlamaServerLLM
    from mirror.persona import FILLERS, load_world
    from mirror.stt import WhisperServerSTT
    from mirror.vad import Segmenter, SileroVad
    from mirror.voice_main import FILLER_BUDGET_S, load_tts, warm_llm

    cfg = load_config(args.config)
    _setup_env(cfg)
    os.makedirs(args.work_dir, exist_ok=True)
    tl = Timeline(cap_s=args.gap_cap)
    texts: dict = {}
    player = TimelinePlayer(tl, texts)
    face = TimelineFace(tl)

    vad = SileroVad(os.path.join(os.path.abspath(cfg.models_dir), "silero_vad.onnx"))
    segmenter = Segmenter(vad, frame_samples=CHUNK, sample_rate=cfg.audio.sample_rate)
    stt = TimedSTT(WhisperServerSTT(cfg.servers.stt_url, cfg.servers.timeout_s))
    world = load_world(resolve_repo_path(cfg.conversation.world_file, args.config))
    llm = TimedLLM(LlamaServerLLM(cfg.servers.llm_url, cfg.servers.timeout_s,
                                  cfg.conversation.max_reply_sentences,
                                  cfg.conversation.max_reply_chars, world=world))
    router, _, engine = load_tts(cfg.tts)
    tts = TextTTS(router, texts, kind="filler")
    fillers = FillerBank(tts, FILLERS,
                         max_clip_s=FILLER_BUDGET_S if engine == "pocket" else None)
    fillers.prepare()
    tts.kind = "reply"
    unwarmed = warm_llm(llm)
    controller = Controller(stt, llm, tts, player, _NullLights(), face, cfg.conversation,
                            clock=tl.now, fillers=fillers,
                            first_sentence_timeout_s=cfg.servers.timeout_s)
    for lang in unwarmed:
        controller.mark_unwarmed(lang)

    face.send(State.IDLE, 0.0)
    tl.silence(LEAD_IN_S)
    turns = []
    for n, path in enumerate(args.wavs):
        if n:
            tl.silence(FOLLOWUP_PAUSE_S)
        audio, sr = read_wav(path)
        q_start, seg = feed_question(tl, segmenter, vad, audio, sr, cfg.audio.sample_rate)
        turn = {"wav": path, "start": round(q_start, 3), "cut": round(tl.cursor, 3),
                "state_before": controller.state.value}
        turns.append(turn)
        if seg is None:
            log.warning("%s: the VAD found no utterance", path)
            continue
        controller.tick()  # a listening window that ran out goes idle first
        turn["state_before"] = controller.state.value
        n_stt, n_llm, n_clips = len(stt.calls), len(llm.calls), len(player.clips)
        tl.anchor()
        controller.on_segment(seg)
        tl.release()
        if len(stt.calls) > n_stt:
            turn.update(heard=stt.calls[-1]["text"], lang=stt.calls[-1]["lang"],
                        stt_s=stt.calls[-1]["seconds"])
        if len(llm.calls) > n_llm:
            c = llm.calls[-1]
            turn.update(llm_first_sentence_s=c["first_sentence_s"],
                        llm_total_s=c.get("total_s"), llm_sentence_s=c["sentence_s"])
        turn["clips"] = player.clips[n_clips:]
        turn["reply"] = " ".join(c["text"] for c in turn["clips"] if c["kind"] == "reply")
        turn["end"] = round(tl.cursor, 3)
        turn["state_after"] = controller.state.value
        log.info("turn %d: heard %r -> %r", n + 1, turn.get("heard"), turn["reply"])

    # the tail: tick the controller so a listening window can run out on screen
    end = tl.cursor + args.tail
    while tl.cursor < end:
        tl.silence(0.1)
        controller.tick()
    controller.close()

    write_wav(os.path.join(args.work_dir, "timeline.wav"), tl.audio(), tl.sr)
    doc = {"sr": tl.sr, "duration": round(tl.cursor, 3), "gap_cap_s": tl.cap_s,
           "turns": turns, "gaps": tl.gaps, "capped": tl.capped,
           "tts_reply_synth_s": tts.synth_s, "events": face.events}
    with open(os.path.join(args.work_dir, "events.json"), "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in doc.items() if k != "events"},
                     ensure_ascii=False, indent=1))


# --------------------------------------------------------------------------
# render
# --------------------------------------------------------------------------
HEBREW_FONTS = ("/System/Library/Fonts/ArialHB.ttc",
                "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
                "/System/Library/Fonts/SFHebrew.ttf")


def _has_hebrew(text: str) -> bool:
    return any("֐" <= ch <= "׿" for ch in text)


class Subtitler:
    """Burns wrapped subtitle lines into a BGR frame (top or bottom band)."""

    def __init__(self, width: int):
        import cv2
        self.cv2 = cv2
        self.width = width
        self.margin = 8
        self.scale = 0.45
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.bidi = None
        self.he_font = None
        try:
            from bidi.algorithm import get_display
            self.bidi = get_display
        except ImportError:
            try:
                from bidi import get_display
                self.bidi = get_display
            except ImportError:
                pass
        if self.bidi is not None:
            from PIL import ImageFont
            for p in HEBREW_FONTS:
                if os.path.exists(p):
                    self.he_font = ImageFont.truetype(p, 17)
                    break
        self.hebrew_ok = self.bidi is not None and self.he_font is not None

    def _wrap(self, text, measure):
        words, lines, cur = text.split(), [], ""
        for w in words:
            cand = (cur + " " + w).strip()
            if cur and measure(cand) > self.width - 2 * self.margin:
                lines.append(cur)
                cur = w
            else:
                cur = cand
        if cur:
            lines.append(cur)
        return lines

    def draw(self, bgr, text: str, where: str, color=(255, 255, 255)) -> None:
        if not text:
            return
        if _has_hebrew(text):
            if self.hebrew_ok:
                self._draw_pil(bgr, text, where, color)
            return
        cv2 = self.cv2

        def measure(s):
            return cv2.getTextSize(s, self.font, self.scale, 1)[0][0]

        lines = self._wrap(text, measure)
        lh = 18
        y0 = self.margin + 12 if where == "top" else bgr.shape[0] - self.margin - lh * (len(lines) - 1) - 4
        band_top = y0 - 14
        band_bot = y0 + lh * (len(lines) - 1) + 6
        sub = bgr[max(0, band_top):band_bot]
        sub[:] = (sub * 0.35).astype(bgr.dtype)
        for i, line in enumerate(lines):
            w = measure(line)
            x = (self.width - w) // 2
            y = y0 + i * lh
            cv2.putText(bgr, line, (x, y), self.font, self.scale, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(bgr, line, (x, y), self.font, self.scale, color, 1, cv2.LINE_AA)

    def _draw_pil(self, bgr, text, where, color) -> None:
        from PIL import Image, ImageDraw
        font = self.he_font

        def measure(s):
            return font.getlength(self.bidi(s))

        # wrap in logical order, then reorder each line for display
        lines = self._wrap(text, measure)
        lh = 21
        n = len(lines)
        y0 = self.margin if where == "top" else bgr.shape[0] - self.margin - lh * n
        band = bgr[max(0, y0 - 2):y0 + lh * n + 2]
        band[:] = (band * 0.35).astype(bgr.dtype)
        img = Image.fromarray(bgr[..., ::-1].copy())
        d = ImageDraw.Draw(img)
        rgb = tuple(int(c) for c in color[::-1])
        for i, line in enumerate(lines):
            vis = self.bidi(line)
            x = (self.width - font.getlength(vis)) / 2
            d.text((x, y0 + i * lh), vis, font=font, fill=rgb,
                   stroke_width=2, stroke_fill=(0, 0, 0))
        bgr[:] = np.asarray(img)[..., ::-1]


def subtitles_at(doc, t):
    """(heard text, oracle text) to show at timeline time t."""
    heard = said = ""
    for turn in doc["turns"]:
        if turn.get("heard") and turn["start"] <= t < turn.get("end", turn["cut"]):
            heard = turn["heard"]
        for c in turn.get("clips", []):
            if c["start"] <= t < c["end"] + 0.3:
                said = c["text"]
    return heard, said


def cmd_render(args) -> None:
    import cv2

    from mirror.config import load_config, resolve_repo_path
    from mirror.face.app import step_mouth
    from mirror.face.bank import load_bank
    from mirror.face.sprite import SpriteFace

    cfg = load_config(args.config)
    with open(os.path.join(args.work_dir, "events.json"), encoding="utf-8") as f:
        doc = json.load(f)
    wav = os.path.join(args.work_dir, "timeline.wav")
    bank = load_bank(args.bank or resolve_repo_path(cfg.face.bank, args.config))
    import random
    face = SpriteFace(bank, rng=random.Random(7))
    size = bank.size * SCALE
    subs = Subtitler(size)
    if not subs.hebrew_ok and any(_has_hebrew(t.get("heard", "") + t.get("reply", ""))
                                  for t in doc["turns"]):
        log.warning("Hebrew subtitles skipped: %s",
                    "python-bidi is not installed" if subs.bidi is None
                    else "no Hebrew font found")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SystemExit("ffmpeg not found")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    proc = subprocess.Popen(
        [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{size}x{size}", "-r", str(FPS), "-i", "-", "-i", wav,
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-c:a", "aac",
         "-b:a", "128k", "-shortest", args.out], stdin=subprocess.PIPE)
    events = [tuple(e) for e in doc["events"]]
    n_frames = int(round(doc["duration"] * FPS))
    state, target, viseme, mouth, idx = State.IDLE, 0.0, "X", 0.0, 0
    dt = 1.0 / FPS
    try:
        for i in range(n_frames):
            t = i / FPS
            msg, idx = face_message_at(events, t, idx)
            if msg is not None:
                state, target, viseme = State(msg[1]), msg[2], msg[3]
            speaking = state is State.SPEAKING
            mouth = step_mouth(mouth, target if speaking else 0.0, dt)
            face.update(dt, state, mouth, viseme if speaking else "X")
            rgb = face.render()
            big = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_NEAREST)
            bgr = np.ascontiguousarray(cv2.cvtColor(big, cv2.COLOR_RGB2BGR))
            heard, said = subtitles_at(doc, t)
            subs.draw(bgr, heard, "top", (200, 230, 255))
            subs.draw(bgr, said, "bottom", (255, 255, 255))
            if args.state_label:
                cv2.putText(bgr, state.value, (6, size // 2), cv2.FONT_HERSHEY_SIMPLEX,
                            0.32, (140, 140, 140), 1, cv2.LINE_AA)
            proc.stdin.write(bgr.tobytes())
    finally:
        proc.stdin.close()
        rc = proc.wait()
    if rc:
        raise SystemExit(f"ffmpeg failed ({rc})")
    print(f"wrote {args.out}: {n_frames} frames, {doc['duration']:.1f} s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="config/mac.toml")
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("questions", help="synthesize the question WAVs")
    q.add_argument("out_dir")
    r = sub.add_parser("record", help="run the pipeline on question WAVs (servers up)")
    r.add_argument("work_dir")
    r.add_argument("wavs", nargs="+")
    r.add_argument("--gap-cap", type=float, default=GAP_CAP_S)
    r.add_argument("--tail", type=float, default=TAIL_S)
    v = sub.add_parser("render", help="render WORK_DIR to an MP4 (no servers)")
    v.add_argument("work_dir")
    v.add_argument("out")
    v.add_argument("--bank", default=None)
    v.add_argument("--state-label", action="store_true", help="print the state name")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    {"questions": cmd_questions, "record": cmd_record, "render": cmd_render}[args.cmd](args)


if __name__ == "__main__":
    main()
