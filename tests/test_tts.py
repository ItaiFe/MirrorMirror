import logging
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from mirror.tts import TTSRouter, group_phoneme_samples, prepare_hebrew
from mirror.visemes import VisemeTrack

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


class FakeAligningSynth(FakeSynth):
    def __init__(self, audio, rate, track):
        super().__init__(audio, rate)
        self.track = track

    def synth_visemes(self, text):
        self.calls.append(text)
        return self.audio, self.rate, self.track


def test_router_synth_visemes_returns_engine_track():
    track = VisemeTrack(np.array([0.0]), ["D"])
    pcm = np.array([0, 16384], dtype=np.int16)
    router = TTSRouter(en=FakeAligningSynth(pcm, 22050, track), he=FakeSynth(np.zeros(4)))
    a, r, t = router.synth_visemes("hello", "en")
    assert t is track and r == 22050
    assert a.dtype == np.float32 and a[1] == pytest.approx(0.5)


def test_router_synth_visemes_none_for_engine_without_alignments():
    he = FakeSynth(np.full(4, 0.2, dtype=np.float32), 16000)  # e.g. MMS
    router = TTSRouter(en=FakeSynth(np.zeros(4)), he=he)
    a, r, t = router.synth_visemes("שלום", "he")
    assert t is None and r == 16000 and np.allclose(a, 0.2) and he.calls == ["שלום"]


def _fake_piper(monkeypatch, chunks):
    """Install a fake `piper` module whose voice yields the given chunks."""
    loads = []

    class Voice:
        config = types.SimpleNamespace(sample_rate=22050)

        def synthesize(self, text, include_alignments=False):
            assert include_alignments
            return iter(chunks)

    def load(path, include_alignments=False):
        loads.append((path, include_alignments))
        return Voice()

    module = types.ModuleType("piper")
    module.PiperVoice = types.SimpleNamespace(load=load)
    monkeypatch.setitem(sys.modules, "piper", module)
    return loads


def _chunk(n, alignments):
    al = None if alignments is None else [
        types.SimpleNamespace(phoneme=p, num_samples=k) for p, k in alignments]
    return types.SimpleNamespace(audio_int16_array=np.zeros(n, np.int16),
                                 phoneme_alignments=al)


def test_piper_synth_visemes_concatenates_chunks(monkeypatch):
    loads = _fake_piper(monkeypatch, [_chunk(4410, [("m", 2205), ("a", 2205)]),
                                      _chunk(4410, [(".", 4410)])])
    from mirror.tts import PiperTTS

    audio, sr, track = PiperTTS("voice.onnx").synth_visemes("hi")
    assert loads == [("voice.onnx", True)]
    assert len(audio) == 8820 and sr == 22050
    assert track.shapes == ["A", "B", "D", "X"]
    assert track.end == pytest.approx(0.4)


def test_piper_synth_visemes_none_when_alignment_fails(monkeypatch, caplog):
    _fake_piper(monkeypatch, [_chunk(4410, [("m", 4410)]), _chunk(4410, None)])
    from mirror.tts import PiperTTS

    tts = PiperTTS("voice.onnx")
    with caplog.at_level(logging.WARNING, logger="mirror.tts"):
        for _ in range(2):
            audio, sr, track = tts.synth_visemes("hi")
            assert track is None and len(audio) == 8820
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1


def test_group_phoneme_samples():
    id_map = {"_": [0], "^": [1], "$": [2], "a": [3], "m": [4, 5]}
    # ids: ^ _ m(2) _ a _ $
    samples = np.array([10, 1, 20, 21, 2, 30, 3, 40])
    assert group_phoneme_samples(["m", "a"], id_map, samples) == [
        ("^", 11), ("m", 43), ("a", 33), ("$", 40)]
    assert group_phoneme_samples(["m", "a"], id_map, samples[:-1]) is None


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


@pytest.mark.slow
def test_phonikud_alignments_real():
    pytest.importorskip("phonikud")
    pytest.importorskip("phonikud_onnx")
    pytest.importorskip("piper")
    pytest.importorskip("onnx")
    if not (PHONIKUD_MODEL.exists() and PHONIKUD_VOICE.exists()):
        pytest.skip("phonikud models missing")
    from mirror.tts import PhonikudHebrewTTS

    tts = PhonikudHebrewTTS(str(PHONIKUD_MODEL), str(PHONIKUD_VOICE))
    audio, sr, track = tts.synth_visemes("מראה מראה שעל הקיר, מי הכי יפה?")
    assert track is not None
    assert track.end == pytest.approx(len(audio) / sr)
    assert track.end - track.times[0] >= 0.9 * len(audio) / sr
    # Piper's durations are stochastic: a given shape (e.g. "D") may be held too
    # briefly to survive the min-hold, so only ask for a varied, plausible track.
    shapes = set(track.shapes)
    assert shapes <= set("ABCDEFGHX")
    assert len(shapes) >= 3 and shapes & {"B", "C"}


# --- fast stubs for the Hebrew engines ---------------------------------------------

def _stub_phonikud(ipa):
    """A PhonikudHebrewTTS with its models replaced by stubs (no model files)."""
    from mirror.tts import PhonikudHebrewTTS, _Tracks

    tts = object.__new__(PhonikudHebrewTTS)
    calls = {}

    class Diacritizer:
        def add_diacritics(self, text):
            calls["diacritize"] = text
            return text + "!"

    class Voice:
        config = types.SimpleNamespace(phoneme_id_map={"_": [0], "^": [1], "$": [2],
                                                       "a": [3], "b": [4]},
                                       sample_rate=22050)

        def phonemes_to_ids(self, phonemes):
            calls["phonemes"] = phonemes
            return [3] * len(phonemes)

        def phoneme_ids_to_audio(self, ids, include_alignments=False):
            calls["ids"] = ids
            return np.full(100 * len(ids), 16384, np.int16)

    tts._phonemize = lambda text: calls.setdefault("phonemize", text) and ipa
    tts._diacritizer = Diacritizer()
    tts._voice = Voice()
    tts._known = set(Voice.config.phoneme_id_map)
    tts._rate = 22050
    tts._tracks = _Tracks("stub")
    return tts, calls


def test_phonikud_phonemes_prepares_and_drops_unknown_characters():
    tts, calls = _stub_phonikud(" a?b ")
    assert tts.phonemes("ו-25  שלום") == "ab"
    assert calls["diacritize"] == "ו25 שלום"  # prepare_hebrew ran first
    assert calls["phonemize"] == "ו25 שלום!"


def test_phonikud_synth_returns_float_audio():
    tts, calls = _stub_phonikud("ab")
    audio, sr = tts.synth("שלום")
    assert sr == 22050 and audio.dtype == np.float32 and len(audio) == 200
    assert np.allclose(audio, 0.5) and calls["phonemes"] == ["a", "b"]


def test_phonikud_synth_empty_when_nothing_pronounceable():
    tts, calls = _stub_phonikud("?!")
    audio, sr = tts.synth("...")
    assert len(audio) == 0 and sr == 22050 and "ids" not in calls


def test_mms_empty_text_returns_empty_audio():
    from mirror.tts import MmsHebrewTTS

    tts = object.__new__(MmsHebrewTTS)
    tts._rate = 16000

    def boom(*args, **kwargs):
        raise AssertionError("the model must not run on empty text")

    tts._tokenizer = boom
    audio, sr = tts.synth("   ")
    assert len(audio) == 0 and audio.dtype == np.float32 and sr == 16000
