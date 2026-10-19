#!/usr/bin/env python3
"""Render the sprite face offline to MP4, lip-synced to a real TTS reply.

    python tools/demo_face.py OUT_DIR [--config config/mac.toml] [--bank DIR] [--lang en he]
                              [--idle S] [--seconds S] [--text LINE]

For each language: synthesize an oracle line with TTSRouter.synth_visemes, then
replay what the live system does frame by frame at 30 fps: the voice process
sends (SPEAKING, mouth level, viseme) once per 20 ms hop (mouth_envelope, the
viseme LEAD_S ahead), the face app keeps the latest message, smooths the mouth
with step_mouth and feeds SpriteFace (floating per the config's face.float_*, the
background mist per face.mist_*).
--idle s of idle before the line (default 1.5), idle after it until --seconds
in all (default: --idle after it too), and one blink in the first idle stretch.
--text replaces the language's oracle line.

Writes <lang>.wav (with the idle silence, so it lines up), <lang>.mp4 (504x504,
mp4v) and, when ffmpeg is on PATH, <lang>_with_audio.mp4. --stills also saves
PNG stills (idle, blink, D viseme, A viseme) for the first language.
"""
import argparse
import math
import os
import shutil
import subprocess
import sys
import wave

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mirror.audio import mouth_envelope  # noqa: E402
from mirror.config import load_config, resolve_repo_path  # noqa: E402
from mirror.conversation import HOP_S, LEAD_S  # noqa: E402
from mirror.face.app import FPS, step_mouth  # noqa: E402
from mirror.face.bank import load_bank  # noqa: E402
from mirror.face.mist import MistParams  # noqa: E402
from mirror.face.sprite import FloatParams, SpriteFace  # noqa: E402
from mirror.states import State  # noqa: E402
from mirror.visemes import rms_shape  # noqa: E402

LINES = {
    "en": "The mists part for you, traveller. What you carry forward, the desert will remember.",
    "he": "הערפל נפתח בשבילכם, נוסעים. מה שתיקחו קדימה, המדבר יזכור.",
}
IDLE_S = 1.5
BLINK_AT_S = 0.8
SCALE = 2


def make_tts(cfg, langs):
    models_dir = os.path.abspath(cfg.models_dir)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HOME"] = os.path.join(models_dir, "hf")
    from mirror.tts import PiperTTS, TTSRouter
    from mirror.voice_main import build_hebrew_tts

    he = build_hebrew_tts(cfg.tts) if "he" in langs else None  # only load what is used
    return TTSRouter(PiperTTS(cfg.tts.piper_voice_en), he,
                     voice_speed=cfg.tts.voice_speed, voice_room=cfg.tts.voice_room)


def write_wav(path, samples, sr):
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def bus_message(t, speech_start, env, track):
    """The latest (state, mouth, viseme) the face would have received by time t."""
    hop = math.floor((t - speech_start) / HOP_S)
    if hop < 0:
        return State.IDLE, 0.0, "X"
    if hop >= len(env):  # the player's final on_level(0.0), then idle
        return (State.SPEAKING, 0.0, "X") if hop == len(env) else (State.IDLE, 0.0, "X")
    level = float(env[hop])
    shape = track.at(hop * HOP_S + LEAD_S) if track is not None else rms_shape(level)
    return State.SPEAKING, level, shape


def render_demo(bank, samples, sr, track, idle_before=IDLE_S, idle_after=IDLE_S,
                float_params=None, mist_params=None):
    """Yield (frame_rgb, viseme, mouth, eye_name) for the whole timeline."""
    env = mouth_envelope(samples, sr, HOP_S)
    total = idle_before + len(samples) / sr + idle_after
    face = SpriteFace(bank, float_params=float_params, mist_params=mist_params)
    face.auto_blink = False
    mouth, dt = 0.0, 1.0 / FPS
    blinked = False
    for i in range(int(round(total * FPS))):
        t = i / FPS
        if not blinked and t >= BLINK_AT_S:
            face.blink()
            blinked = True
        state, target, viseme = bus_message(t, idle_before, env, track)
        speaking = state is State.SPEAKING
        mouth = step_mouth(mouth, target if speaking else 0.0, dt)
        face.update(dt, state, mouth, viseme if speaking else "X")
        yield face.render(), viseme, mouth, face.eye_name


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out_dir")
    ap.add_argument("--config", default="config/mac.toml")
    ap.add_argument("--bank", default=None, help="face bank dir (default: face.bank)")
    ap.add_argument("--lang", nargs="+", default=["en", "he"], choices=sorted(LINES))
    ap.add_argument("--stills", action="store_true", help="save idle/blink/D/A PNG stills")
    ap.add_argument("--idle", type=float, default=IDLE_S, help="idle s before the line")
    ap.add_argument("--seconds", type=float, default=None,
                    help="total length (default: --idle, the line, --idle)")
    ap.add_argument("--text", default=None, help="speak this instead of the oracle line")
    args = ap.parse_args()

    cfg = load_config(args.config)
    bank = load_bank(args.bank or resolve_repo_path(cfg.face.bank, args.config))
    tts = make_tts(cfg, args.lang)
    float_params = FloatParams.from_config(cfg.face)
    mist_params = MistParams.from_config(cfg.face)
    os.makedirs(args.out_dir, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    size = bank.size * SCALE

    for n, lang in enumerate(args.lang):
        samples, sr, track = tts.synth_visemes(args.text or LINES[lang], lang)
        print(f"{lang}: {len(samples) / sr:.2f} s, track={'yes' if track else 'no (rms shapes)'}")
        after = args.idle if args.seconds is None else max(
            0.0, args.seconds - args.idle - len(samples) / sr)
        wav_path = os.path.join(args.out_dir, f"{lang}.wav")
        write_wav(wav_path, np.concatenate([np.zeros(int(args.idle * sr), np.float32), samples,
                                            np.zeros(int(after * sr), np.float32)]), sr)

        mp4_path = os.path.join(args.out_dir, f"{lang}.mp4")
        writer = cv2.VideoWriter(mp4_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (size, size))
        stills, best, run, prev = {}, {}, 0, None
        for i, (rgb, viseme, mouth, eye) in enumerate(render_demo(
                bank, samples, sr, track, args.idle, after, float_params, mist_params)):
            big = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_NEAREST)
            bgr = cv2.cvtColor(big, cv2.COLOR_RGB2BGR)
            writer.write(bgr)
            run = run + 1 if viseme == prev else 1
            prev = viseme
            if args.stills and n == 0:
                if i == 10:
                    stills["idle"] = bgr
                if eye == "closed":
                    stills.setdefault("blink", bgr)
                # D: the widest settled D; A: the longest-held A
                score = {"D": (run >= 3, mouth), "A": (run, 0.0)}.get(viseme)
                if score is not None and score > best.get(viseme, ((-1, -1.0),))[0]:
                    best[viseme] = (score, bgr, i)
        writer.release()
        print(f"  wrote {wav_path}, {mp4_path}")
        for shape, (score, img, i) in best.items():
            stills[f"viseme_{shape}"] = img
            print(f"  viseme_{shape}: frame {i}, score {score}")
        for name, img in stills.items():
            p = os.path.join(args.out_dir, f"still_{name}.png")
            cv2.imwrite(p, img)
            print(f"  still {p}")

        if ffmpeg:
            out = os.path.join(args.out_dir, f"{lang}_with_audio.mp4")
            subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", mp4_path, "-i", wav_path,
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                            "-shortest", out], check=True)
            print(f"  wrote {out}")


if __name__ == "__main__":
    main()
