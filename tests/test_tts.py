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


def test_normalize_scales_other_integer_dtypes():
    from mirror.tts import _normalize

    i32, _ = _normalize(np.array([0, 2**30], dtype=np.int32), 16000)
    u8, _ = _normalize(np.array([128, 255, 0], dtype=np.uint8), 16000)
    assert i32.dtype == np.float32 and i32[1] == pytest.approx(0.5)
    assert u8.dtype == np.float32
    assert u8[0] == pytest.approx(0.0) and u8[2] == pytest.approx(-1.0)


def test_router_float64_becomes_float32():
    router = TTSRouter(en=FakeSynth(np.zeros(5)), he=FakeSynth(np.zeros(5)))
    a, _ = router.synth("x", "he")
    assert a.dtype == np.float32


def test_warmup_synthesizes_each_language():
    en, he = FakeSynth(np.zeros(5)), FakeSynth(np.zeros(5))
    TTSRouter(en=en, he=he).warmup()
    assert len(en.calls) == 1 and len(he.calls) == 1


def _load_or_skip(factory, *args, **kwargs):
    """Build a real model offline; skip (not error) when its HF files aren't cached."""
    try:
        return factory(*args, **kwargs)
    except OSError as e:  # includes huggingface_hub LocalEntryNotFoundError
        pytest.skip(f"model files not in local HF cache: {type(e).__name__}")


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

    tts = _load_or_skip(MmsHebrewTTS, "facebook/mms-tts-heb")
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

    tts = _load_or_skip(PhonikudHebrewTTS, str(PHONIKUD_MODEL), str(PHONIKUD_VOICE))
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

    tts = _load_or_skip(PhonikudHebrewTTS, str(PHONIKUD_MODEL), str(PHONIKUD_VOICE))
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
    from mirror.tts import PhonikudG2P, PhonikudHebrewTTS, _Tracks

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

    tts._g2p = PhonikudG2P.__new__(PhonikudG2P)
    tts._g2p._phonemize = lambda text: calls.setdefault("phonemize", text) and ipa
    tts._g2p._diacritizer = Diacritizer()
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


# --- RenikudPlus G2P ---------------------------------------------------------------

RENIKUD_MODEL = Path("models/renikud-plus.int8.onnx")


def _fake_renikud(monkeypatch, ipa="ʃalˈom"):
    """Install a fake `renikud_onnx` whose G2P records how it is built and called."""
    record = {"inits": [], "calls": []}

    class G2P:
        def __init__(self, model_path=None, **kwargs):
            record["inits"].append((model_path, kwargs))

        def phonemize(self, text, speaker=0, **kwargs):
            record["calls"].append((text, speaker))
            return ipa

    module = types.ModuleType("renikud_onnx")
    module.G2P = G2P
    monkeypatch.setitem(sys.modules, "renikud_onnx", module)
    return record


def test_renikud_g2p_loads_lazily_once_offline(monkeypatch):
    record = _fake_renikud(monkeypatch)
    from mirror.tts import RenikudG2P

    g2p = RenikudG2P("models/r.onnx")
    assert record["inits"] == []  # nothing loads until the first sentence
    assert g2p.phonemize("שלום") == "ʃalˈom"
    assert g2p.phonemize("  שלום\n עולם ") == "ʃalˈom"
    (path, kwargs), = record["inits"]
    # an explicit model path and no datastore: nothing is ever fetched at runtime
    assert path == "models/r.onnx" and kwargs["datastore"] is None
    # the oracle is masculine; whitespace is collapsed before the model sees it
    assert record["calls"] == [("שלום", 1), ("שלום עולם", 1)]


def test_renikud_g2p_speaker_override(monkeypatch):
    record = _fake_renikud(monkeypatch)
    from mirror.tts import RenikudG2P

    RenikudG2P("m.onnx", speaker=2).phonemize("שלום")
    assert record["calls"] == [("שלום", 2)]


def test_renikud_g2p_reads_interjections_as_sounds(monkeypatch):
    record = _fake_renikud(monkeypatch)
    from mirror.tts import RenikudG2P

    RenikudG2P("m.onnx").phonemize("הממ")
    lexicon = record["inits"][0][1]["lexicon"]
    # the fillers' "hmm"/"mmm" are sounds, not words to vowelize ("hamamˈa")
    assert lexicon["הממ"] == "hmm" and lexicon["מממ"] == "mmm"


@pytest.mark.parametrize("raw, mapped", [
    ("mmm… zˈo", "mmm, zˈo"),                     # ellipsis: a pause
    ("talˈuj balˈev – bivχiʁˈa", "talˈuj balˈev, bivχiʁˈa"),  # dashes: a pause
    ("lahovˈe - ʔˈaχ", "lahovˈe, ʔˈaχ"),
    ("ziχʁˈu: χˈut; ʔadˈom", "ziχʁˈu, χˈut, ʔadˈom"),  # colon/semicolon: a pause
    ("be־ʃalˈoʃ ve-ʔaχˈat", "beʃalˈoʃ veʔaχˈat"),      # maqaf/hyphen inside a word: joined
    ("kmˈo *meχajˈeχ* (Hi) \"tʃuvˈa\"", "kmˈo meχajˈeχ tʃuvˈa"),  # markup, Latin dropped
    ("mˈa?! tˈov.", "mˈa?! tˈov."),
    ("tʃuvˈa:.", "tʃuvˈa."),                       # no doubled punctuation
])
def test_renikud_g2p_maps_symbols_to_the_voice_inventory(monkeypatch, raw, mapped):
    _fake_renikud(monkeypatch, ipa=raw)
    from mirror.tts import RenikudG2P

    assert RenikudG2P("m.onnx").phonemize("x") == mapped


def test_hebrew_voice_uses_the_given_g2p(monkeypatch):
    tts, calls = _stub_phonikud("unused")

    class G2P:
        def phonemize(self, text):
            calls["g2p"] = text
            return "ba?"

    tts._g2p = G2P()
    assert tts.phonemes("שלום") == "ba"
    assert calls["g2p"] == "שלום" and "diacritize" not in calls


@pytest.mark.slow
def test_renikud_real_ipa_uses_only_voice_symbols():
    pytest.importorskip("renikud_onnx")
    if not RENIKUD_MODEL.exists():
        pytest.skip("renikud model missing")
    import json

    from mirror.persona import FALLBACK, FILLERS
    from mirror.tts import HEBREW_VOICE_SYMBOLS, RenikudG2P

    g2p = RenikudG2P(str(RENIKUD_MODEL))
    voice_symbols = set(json.loads(Path("models/phonikud-tts-he.onnx.json").read_text())
                        ["phoneme_id_map"]) if PHONIKUD_VOICE.exists() else set()
    for line in [*FILLERS["he"], FALLBACK["he"], "ראיתי 15 כוכבים, ב־3 בלילה – *מחייך*."]:
        ipa = g2p.phonemize(line)
        assert ipa and "ˈ" in ipa, line
        assert set(ipa) <= HEBREW_VOICE_SYMBOLS, (line, ipa)
        assert not voice_symbols or set(ipa) <= voice_symbols


# --- Pocket TTS (optional engine) -----------------------------------------------------

POCKET_MODEL = Path("models/pocket-tts-english-ipa.onnx")
POCKET_SR = 24000


def _speech(seconds):
    t = np.arange(int(seconds * POCKET_SR)) / POCKET_SR
    return (0.3 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)


def _fake_pocket(monkeypatch, durations=None):
    """Install a fake `pocket_tts_onnx`. durations(call_index, text) -> seconds of audio
    (default: 0.08 s per character, never truncated)."""
    record = {"inits": [], "calls": []}

    class Model:
        sample_rate = POCKET_SR

        def __init__(self, path, providers=None, num_threads=2):
            record["inits"].append((path, providers, num_threads))

        def voices(self):
            return ["alba", "javert", "michael"]

        def create(self, text, voice="alba", phonemes=False, seed=None, **kwargs):
            i = len(record["calls"])
            record["calls"].append(dict(text=text, voice=voice, phonemes=phonemes, seed=seed))
            s = durations(i, text) if durations else 0.08 * len(text)
            return _speech(s), POCKET_SR

    module = types.ModuleType("pocket_tts_onnx")
    module.PocketTTS = Model
    monkeypatch.setitem(sys.modules, "pocket_tts_onnx", module)
    return record


class FakeIpaG2P:
    def __init__(self, ipa="ʃalˈom ʔolˈam."):
        self.ipa_calls = []
        self._ipa = ipa

    def ipa(self, text):
        self.ipa_calls.append(text)
        return self._ipa


def test_pocket_loads_the_model_on_cpu_with_threads(monkeypatch):
    record = _fake_pocket(monkeypatch)
    from mirror.tts import PocketTTS

    tts = PocketTTS("m.onnx", voice="michael", threads=3, g2p=FakeIpaG2P(), en_ipa=None)
    (path, providers, threads), = record["inits"]
    assert path == "m.onnx" and providers == ["CPUExecutionProvider"] and threads == 3
    assert tts.sample_rate == POCKET_SR


def test_pocket_unknown_voice_fails_at_load(monkeypatch):
    _fake_pocket(monkeypatch)
    from mirror.tts import PocketTTS

    with pytest.raises(ValueError, match="male1"):
        PocketTTS("m.onnx", voice="male1", g2p=FakeIpaG2P())


def test_pocket_voice_file_is_a_cloned_conditioning(monkeypatch, tmp_path):
    record = _fake_pocket(monkeypatch)
    from mirror.tts import PocketTTS

    cond = np.zeros((1, 4, 8), np.float32)
    np.save(tmp_path / "v.npy", cond)
    tts = PocketTTS("m.onnx", voice="nope", voice_file=str(tmp_path / "v.npy"),
                    g2p=FakeIpaG2P(), en_ipa=None)
    tts.synth("Hello there.", "en")
    voice = record["calls"][0]["voice"]
    assert isinstance(voice, np.ndarray) and voice.shape == (1, 4, 8)


def test_pocket_english_text_hebrew_ipa(monkeypatch):
    record = _fake_pocket(monkeypatch)
    from mirror.tts import PocketTTS

    g2p = FakeIpaG2P()
    tts = PocketTTS("m.onnx", voice="javert", g2p=g2p, en_ipa=lambda text: None)
    audio, sr = tts.synth("The path you seek.", "en")
    assert sr == POCKET_SR and audio.dtype == np.float32 and len(audio) > 0
    audio, sr = tts.synth("שלום עולם", "he")
    en, he = record["calls"]
    assert en["text"] == "The path you seek." and en["phonemes"] is False
    assert he["text"] == "ʃalˈom ʔolˈam." and he["phonemes"] is True
    assert en["voice"] == he["voice"] == "javert"
    assert g2p.ipa_calls == ["שלום עולם"]


def test_pocket_rerolls_a_truncated_clip(monkeypatch, caplog):
    # the first call stops after the first word; the re-roll speaks it all
    record = _fake_pocket(monkeypatch, lambda i, text: 0.2 if i == 0 else 2.0)
    from mirror.tts import PocketTTS

    tts = PocketTTS("m.onnx", g2p=FakeIpaG2P(), en_ipa=None)
    with caplog.at_level("INFO", logger="mirror.tts"):
        audio, sr = tts.synth("Hmm. The mists are parting.", "en")
    assert len(audio) / sr == pytest.approx(2.0)
    seeds = [c["seed"] for c in record["calls"]]
    assert len(seeds) == 2 and seeds[0] != seeds[1]


def test_pocket_reroll_is_bounded_and_keeps_the_longest(monkeypatch, caplog):
    lengths = [0.2, 0.4, 0.3]
    record = _fake_pocket(monkeypatch, lambda i, text: lengths[i])
    from mirror.tts import PocketTTS

    tts = PocketTTS("m.onnx", g2p=FakeIpaG2P(), en_ipa=None)
    with caplog.at_level("WARNING", logger="mirror.tts"):
        audio, sr = tts.synth("Hmm. The mists are parting.", "en")
    assert len(record["calls"]) == 3  # the first try and at most 2 re-rolls
    assert len(set(c["seed"] for c in record["calls"])) == 3
    assert len(audio) / sr == pytest.approx(0.4)
    assert any(r.levelname == "WARNING" for r in caplog.records)


def test_pocket_no_reroll_when_long_enough(monkeypatch):
    # 27 characters x 0.055 s = 1.5 s expected; 45 % of that is 0.67 s
    record = _fake_pocket(monkeypatch, lambda i, text: 0.7)
    from mirror.tts import PocketTTS

    PocketTTS("m.onnx", g2p=FakeIpaG2P(), en_ipa=None).synth(
        "Hmm. The mists are parting.", "en")
    assert len(record["calls"]) == 1


def test_pocket_expected_duration_from_phonemes(monkeypatch):
    # Hebrew: 10 phonemes (marks, spaces, punctuation not counted) x 0.06 s = 0.6 s
    record = _fake_pocket(monkeypatch, lambda i, text: 0.25 if i == 0 else 0.6)
    from mirror.tts import PocketTTS

    tts = PocketTTS("m.onnx", g2p=FakeIpaG2P("ʃalˈom ʔolˈam."), en_ipa=None)
    tts.synth("שלום עולם", "he")
    assert len(record["calls"]) == 2  # 0.25 s < 45 % of 0.6 s


def test_pocket_empty_text_is_silent(monkeypatch):
    record = _fake_pocket(monkeypatch)
    from mirror.tts import PocketTTS

    tts = PocketTTS("m.onnx", g2p=FakeIpaG2P(""), en_ipa=None)
    audio, sr, track = tts.synth_visemes("  ", "en")
    assert len(audio) == 0 and track is None
    audio, sr = tts.synth("שלום", "he")
    assert len(audio) == 0 and record["calls"] == []


def test_pocket_synth_visemes_estimates_a_track(monkeypatch):
    _fake_pocket(monkeypatch)
    from mirror.tts import PocketTTS

    tts = PocketTTS("m.onnx", g2p=FakeIpaG2P("mˈama bˈaba."), en_ipa=lambda t: "mˈama.")
    audio, sr, track = tts.synth_visemes("Mama.", "en")
    assert track is not None and "A" in track.shapes
    assert track.end == pytest.approx(len(audio) / sr)
    audio, sr, track = tts.synth_visemes("אמא", "he")
    assert track is not None and "A" in track.shapes


def test_pocket_english_without_espeak_falls_back_to_loudness(monkeypatch):
    _fake_pocket(monkeypatch)
    from mirror.tts import PocketTTS

    tts = PocketTTS("m.onnx", g2p=FakeIpaG2P(), en_ipa=lambda t: None)
    audio, sr, track = tts.synth_visemes("Hello there.", "en")
    assert len(audio) > 0 and track is None


def test_pocket_language_views_plug_into_the_router(monkeypatch):
    record = _fake_pocket(monkeypatch)
    from mirror.tts import PocketTTS

    tts = PocketTTS("m.onnx", g2p=FakeIpaG2P("mˈa."), en_ipa=lambda t: "hˈɛloʊ.")
    router = TTSRouter(tts.for_lang("en"), tts.for_lang("he"))
    _, sr, track = router.synth_visemes("Hello.", "en")
    assert sr == POCKET_SR and track is not None
    router.synth("מה", "he")
    assert [c["phonemes"] for c in record["calls"]] == [False, True]
    assert tts.for_lang("he").sample_rate == POCKET_SR


def test_renikud_g2p_ipa_keeps_symbols_outside_the_voice(monkeypatch):
    # Pocket reads any IPA (and Latin words); only pauses and markup are cleaned
    _fake_renikud(monkeypatch, ipa="kmˈo *meχajˈeχ* (Hi) mmm… zˈo")
    from mirror.tts import RenikudG2P

    assert RenikudG2P("m.onnx").ipa("x") == "kmˈo meχajˈeχ Hi mmm, zˈo"


@pytest.mark.slow
def test_pocket_real():
    pytest.importorskip("pocket_tts_onnx")
    pytest.importorskip("sentencepiece")
    if not POCKET_MODEL.exists():
        pytest.skip("pocket model missing")
    from mirror.tts import PocketTTS

    tts = _load_or_skip(PocketTTS, str(POCKET_MODEL), voice="javert", g2p=FakeIpaG2P("ʃalˈom."))
    audio, sr, track = tts.synth_visemes("Mirror, mirror on the wall.", "en")
    assert sr == 24000 and len(audio) / sr > 1.0
    assert np.abs(audio).max() > 0.01


def _padded(lead_s, speech_s, tail_s):
    z = lambda s: np.zeros(int(s * POCKET_SR), np.float32)  # noqa: E731
    return np.concatenate([z(lead_s), _speech(speech_s), z(tail_s)])


def test_pocket_trims_long_leading_and_trailing_silence(monkeypatch):
    # Pocket's English voices often open with ~1 s of silence: dead air before every sentence
    _fake_pocket(monkeypatch)
    import mirror.tts as T

    tts = T.PocketTTS("m.onnx", g2p=FakeIpaG2P(), en_ipa=None)
    monkeypatch.setattr(tts._model, "create",
                        lambda *a, **k: (_padded(1.0, 2.0, 0.8), POCKET_SR))
    audio, sr = tts.synth("The path you seek.", "en")
    assert len(audio) / sr == pytest.approx(T.POCKET_LEAD_S + 2.0 + T.POCKET_TAIL_S, abs=0.05)  # 20 ms envelope
    assert np.abs(audio[: int(0.05 * sr)]).max() == 0  # a short lead-in is kept


def test_pocket_truncation_check_ignores_silence(monkeypatch):
    # a stop after "Hmm" padded by a second of silence is still a truncated clip
    record = _fake_pocket(monkeypatch)
    from mirror.tts import PocketTTS

    tts = PocketTTS("m.onnx", g2p=FakeIpaG2P(), en_ipa=None)
    takes = [_padded(1.0, 0.3, 0.1), _padded(0.3, 1.5, 0.1)]
    monkeypatch.setattr(tts._model, "create", lambda *a, **k: (takes.pop(0), POCKET_SR))
    audio, sr = tts.synth("Hmm. The mists are parting.", "en")
    assert takes == [] and len(audio) / sr > 1.5


def test_pocket_hebrew_without_g2p_is_a_clear_error(monkeypatch):
    _fake_pocket(monkeypatch)
    from mirror.tts import PocketTTS

    tts = PocketTTS("m.onnx", g2p=None, en_ipa=None)
    assert len(tts.synth("Hello there.", "en")[0]) > 0  # English needs no G2P
    with pytest.raises(ValueError, match="G2P"):
        tts.synth("שלום", "he")


def test_renikud_g2p_caps_its_onnxruntime_threads(monkeypatch):
    """RenikudPlus gets 2 intra-op threads, not one per core (it shares the Orin's
    6 cores with whisper, llama and the face)."""
    record = _fake_renikud(monkeypatch)
    from mirror.tts import RENIKUD_THREADS, RenikudG2P

    assert RENIKUD_THREADS == 2
    RenikudG2P("models/r.onnx").phonemize("שלום")
    (_, kwargs), = record["inits"]
    opts = kwargs["session_options"]
    assert opts.intra_op_num_threads == 2


# --- the "villain" voice effect ---

def _sine(freq, seconds, sr=22050):
    t = np.arange(int(sr * seconds)) / sr
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _peak_hz(audio, sr):
    spectrum = np.abs(np.fft.rfft(audio))
    return np.fft.rfftfreq(len(audio), 1 / sr)[int(np.argmax(spectrum))]


def test_voice_effect_off_returns_the_input_arrays():
    from mirror.tts import voice_effect

    a = _sine(220, 0.5)
    track = VisemeTrack(np.array([0.0, 0.2]), ["A", "D"], end=0.4)
    out, t = voice_effect(a, 22050, track, speed=1.0, room=0.0)
    assert out is a and t is track


def test_voice_effect_slows_by_the_speed_factor():
    from mirror.tts import voice_effect

    a = _sine(220, 1.0)
    out, _ = voice_effect(a, 22050, None, speed=0.86, room=0.0)
    assert len(out) == int(len(a) / 0.86)
    assert out.dtype == np.float32


def test_voice_effect_lowers_the_pitch_by_the_speed_factor():
    from mirror.tts import voice_effect

    sr = 22050
    a = _sine(440, 1.0, sr)
    out, _ = voice_effect(a, sr, None, speed=0.86, room=1.0)
    assert _peak_hz(out, sr) == pytest.approx(440 * 0.86, abs=3)


def test_voice_effect_peak_normalizes():
    from mirror.tts import voice_effect

    out, _ = voice_effect(_sine(220, 0.5), 22050, None, speed=0.86, room=1.0)
    assert np.abs(out).max() == pytest.approx(0.9, abs=1e-4)


def test_voice_effect_room_only_keeps_the_length_and_adds_echo():
    from mirror.tts import voice_effect

    a = np.zeros(22050, np.float32)
    a[0] = 1.0  # an impulse: the echoes show up at their delays
    out, _ = voice_effect(a, 22050, None, speed=1.0, room=0.5)
    assert len(out) == len(a)
    d = int(22050 * 37 / 1000)
    assert out[d] / out[0] == pytest.approx(0.28 * 0.5, rel=1e-3)


def test_voice_effect_scales_the_viseme_track():
    from mirror.tts import voice_effect

    track = VisemeTrack(np.array([0.0, 0.43, 0.86]), ["A", "D", "X"], end=1.72)
    _, t = voice_effect(_sine(220, 2.0), 22050, track, speed=0.86, room=1.0)
    assert np.allclose(t.times, [0.0, 0.5, 1.0])
    assert t.end == pytest.approx(2.0) and t.shapes == ["A", "D", "X"]
    assert np.allclose(track.times, [0.0, 0.43, 0.86])  # the input is untouched


def test_voice_effect_keeps_an_open_track_end_infinite():
    from mirror.tts import voice_effect

    track = VisemeTrack(np.array([0.0, 0.43]), ["A", "D"])
    _, t = voice_effect(_sine(220, 1.0), 22050, track, speed=0.86, room=0.0)
    assert t.end == float("inf") and np.allclose(t.times, [0.0, 0.5])


def test_voice_effect_none_track_stays_none():
    from mirror.tts import voice_effect

    _, t = voice_effect(_sine(220, 0.5), 22050, None, speed=0.86, room=1.0)
    assert t is None


@pytest.mark.parametrize("audio", [np.zeros(0, np.float32), np.zeros(1000, np.float32)])
def test_voice_effect_empty_or_silent_audio(audio):
    from mirror.tts import voice_effect

    out, _ = voice_effect(audio, 22050, None, speed=0.86, room=1.0)
    assert np.all(out == 0) and np.all(np.isfinite(out))


def test_voice_effect_is_cheap():
    import time

    from mirror.tts import voice_effect

    a = (np.random.default_rng(0).standard_normal(22050 * 10) * 0.1).astype(np.float32)
    voice_effect(a, 22050, None, speed=0.86, room=1.0)  # warm the caches
    t0 = time.perf_counter()
    voice_effect(a, 22050, None, speed=0.86, room=1.0)
    assert time.perf_counter() - t0 < 0.1


def test_router_applies_the_voice_effect_to_both_languages():
    sr = 22050
    track = VisemeTrack(np.array([0.0, 0.43]), ["A", "D"], end=0.86)
    en = FakeAligningSynth(_sine(440, 1.0, sr), sr, track)
    he = FakeSynth(_sine(440, 1.0, sr), sr)
    router = TTSRouter(en=en, he=he, voice_speed=0.86, voice_room=1.0)
    a, r, t = router.synth_visemes("hello", "en")
    assert r == sr and len(a) == int(sr / 0.86)
    assert np.allclose(t.times, [0.0, 0.5]) and t.end == pytest.approx(1.0)
    a, _, t = router.synth_visemes("שלום", "he")
    assert len(a) == int(sr / 0.86) and t is None
    a, _ = router.synth("שלום", "he")
    assert _peak_hz(a, sr) == pytest.approx(440 * 0.86, abs=3)


def test_router_without_the_effect_is_unchanged():
    pcm = _sine(440, 0.5)
    router = TTSRouter(en=FakeSynth(pcm, 22050), he=FakeSynth(pcm, 22050))
    a, _ = router.synth("hello", "en")
    assert np.array_equal(a, pcm)


def test_shorten_pauses_caps_long_gaps_and_keeps_speech():
    from mirror.tts import shorten_pauses

    sr = 16000
    tone = (0.5 * np.sin(np.arange(int(0.5 * sr)) * 0.3)).astype(np.float32)
    gap = np.zeros(int(1.5 * sr), np.float32)
    short = np.zeros(int(0.2 * sr), np.float32)
    audio = np.concatenate([tone, gap, tone, short, tone])
    out = shorten_pauses(audio, sr, max_gap_s=0.4)
    # the 1.5 s pause becomes ~0.4 s; the 0.2 s one is left alone
    assert abs(len(out) / sr - (1.5 + 0.4 + 0.2)) < 0.1
    assert float(np.abs(out).sum()) == pytest.approx(float(np.abs(audio).sum()), rel=0.02)
    assert shorten_pauses(tone, sr) is tone
    assert len(shorten_pauses(np.zeros(0, np.float32), sr)) == 0
