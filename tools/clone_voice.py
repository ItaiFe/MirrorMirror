#!/usr/bin/env python3
"""Clone a voice for Pocket TTS, offline: a reference recording -> a voice .npy.

    python tools/clone_voice.py REF.wav OUT.npy [--model models/pocket-tts-english-ipa.onnx]
                                                [--max-seconds 20]

Runs Pocket's voice encoder (bundled in the model file) on REF.wav, 10-20 s of one
clear speaker, and saves the conditioning (about 0.8 MB). Point tts.pocket_voice_file
at OUT.npy; at runtime only the .npy is loaded and the encoder never runs.

The encoder needs ~1.5 GB of RAM: run this on the Mac, not the Orin, and copy the .npy.
Kyutai's terms forbid cloning a voice without its owner's explicit consent.
"""
import argparse
import os
import sys
import time

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
DEFAULT_MODEL = os.path.join(REPO, "models", "pocket-tts-english-ipa.onnx")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("reference", help="reference recording (.wav, one speaker)")
    ap.add_argument("out", help="where to save the voice (.npy)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Pocket TTS ONNX file")
    ap.add_argument("--max-seconds", type=float, default=20.0,
                    help="use at most this much of the reference")
    args = ap.parse_args(argv)
    os.environ["HF_HUB_OFFLINE"] = "1"  # everything is local

    import numpy as np
    from pocket_tts_onnx import PocketTTS

    tts = PocketTTS(args.model, num_threads=2)
    t0 = time.perf_counter()
    cond = tts.clone_voice(args.reference, max_seconds=args.max_seconds)
    np.save(args.out, cond)
    print(f"{args.out}: conditioning {tuple(cond.shape)} in {time.perf_counter() - t0:.1f} s; "
          f"set tts.pocket_voice_file to it")
    return 0


if __name__ == "__main__":
    sys.exit(main())
