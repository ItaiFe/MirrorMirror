from pathlib import Path

import numpy as np
import pytest

from mirror.tts import TTSRouter, prepare_hebrew

PIPER_MODEL = Path("models/en_US-lessac-medium.onnx")
PHONIKUD_MODEL = Path("models/phonikud-1.0.int8.onnx")
PHONIKUD_VOICE = Path("models/phonikud-tts-he.onnx")


class FakeSynth:
    def __init__(self, audio, rate=16000):
        self.audio = audio
        self.rate = rate
        self.calls = []

    def synth(self, text):
        self.calls.append(text)
        return self.audio, self.rate


def test_router_picks_language():
    en = FakeSynth(np.full(10, 0.1, dtype=np.float32), 22050)
    he = FakeSynth(np.full(10, 0.2, dtype=np.float32), 16000)
    router = TTSRouter(en=en, he=he)
    a, r = router.synth("shalom", "he")
    assert r == 16000 and np.allclose(a, 0.2)
    a, r = router.synth("hello", "en")
    assert r == 22050 and np.allclose(a, 0.1)
    a, r = router.synth("bonjour", "fr")
    assert r == 22050 and np.allclose(a, 0.1)
    assert he.calls == ["shalom"] and en.calls == ["hello", "bonjour"]


def test_router_routes_he_to_configured_engine():
    he = FakeSynth(np.full(4, 0.3, dtype=np.float32), 22050)
    router = TTSRouter(en=FakeSynth(np.zeros(4)), he=he)
    a, r = router.synth("מראה", "he")
    assert r == 22050 and np.allclose(a, 0.3) and he.calls == ["מראה"]


def test_prepare_hebrew_drops_hyphen_before_number():
    # Phonikud reads "-25" as "minus 25"; a prefix letter + hyphen + digits is just a prefix.
    assert prepare_hebrew("ו-25 שקלים") == "ו25 שקלים"
    assert prepare_hebrew("ב־3 בלילה") == "ב3 בלילה"  # maqaf
    assert prepare_hebrew("בשנת 1948, 3-4 ימים") == "בשנת 1948, 3-4 ימים"


def test_prepare_hebrew_collapses_whitespace():
    assert prepare_hebrew("  שלום\n\nעולם  ") == "שלום עולם"


def test_router_output_float32_mono():
    pcm = np.array([[0, 16384], [-32768, 32767]], dtype=np.int16)
    router = TTSRouter(en=FakeSynth(pcm), he=FakeSynth(pcm))
    a, _ = router.synth("x", "en")
    assert a.dtype == np.float32 and a.ndim == 1 and a.shape == (4,)
    assert a.min() >= -1.0 and a.max() <= 1.0
    assert a[1] == pytest.approx(0.5) and a[2] == pytest.approx(-1.0)


def test_router_float64_becomes_float32():
    router = TTSRouter(en=FakeSynth(np.zeros(5)), he=FakeSynth(np.zeros(5)))
    a, _ = router.synth("x", "he")
    assert a.dtype == np.float32


def test_warmup_synthesizes_each_language():
    en, he = FakeSynth(np.zeros(5)), FakeSynth(np.zeros(5))
    TTSRouter(en=en, he=he).warmup()
    assert len(en.calls) == 1 and len(he.calls) == 1


@pytest.mark.slow
def test_piper_real():
    pytest.importorskip("piper")
    if not PIPER_MODEL.exists():
        pytest.skip("piper model missing")
    from mirror.tts import PiperTTS

    audio, rate = PiperTTS(str(PIPER_MODEL)).synth("hello")
    assert len(audio) / rate > 0.2


@pytest.mark.slow
def test_mms_real():
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    from mirror.tts import MmsHebrewTTS

    try:
        tts = MmsHebrewTTS("facebook/mms-tts-heb")
    except OSError:
        pytest.skip("MMS model not in HF cache")
    audio, rate = tts.synth("שלום")
    assert len(audio) / rate > 0.2


@pytest.mark.slow
def test_phonikud_real():
    pytest.importorskip("phonikud")
    pytest.importorskip("phonikud_onnx")
    pytest.importorskip("piper")
    if not (PHONIKUD_MODEL.exists() and PHONIKUD_VOICE.exists()):
        pytest.skip("phonikud models missing")
    from mirror.tts import PhonikudHebrewTTS

    tts = PhonikudHebrewTTS(str(PHONIKUD_MODEL), str(PHONIKUD_VOICE))
    audio, rate = tts.synth("מראה מראה שעל הקיר, מי הכי יפה?")
    assert rate == 22050
    assert audio.dtype == np.float32 and audio.ndim == 1
    assert len(audio) / rate > 0.5
    assert 0.01 < np.abs(audio).max() <= 1.0
