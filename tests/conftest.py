"""Keep the test run offline, like the app at runtime (see mirror/voice_main.py).

Runs at import, before any test module imports mirror.tts / transformers, so the
Hugging Face libraries never try the network; slow real-model tests skip instead.
"""
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent.parent / "models" / "hf"))
