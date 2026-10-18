import numpy as np
import pytest

import mirror.tts
from mirror.config import TtsConfig
from mirror.llm import LLMError
from mirror.voice_main import build_hebrew_tts, open_output, resolve_repo_path, warm_llm


class Recorder:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


def test_build_hebrew_tts_phonikud(monkeypatch):
    monkeypatch.setattr(mirror.tts, "PhonikudHebrewTTS", Recorder)
    cfg = TtsConfig(he_engine="phonikud", he_g2p="phonikud", phonikud_model="g2p.onnx",
                    phonikud_voice_he="voice.onnx", phonikud_voice_config_he="voice.json")
    he = build_hebrew_tts(cfg)
    assert isinstance(he, Recorder) and he.args == ("g2p.onnx", "voice.onnx", "voice.json")
    assert he.kwargs == {"g2p": None}  # the voice builds its own Phonikud G2P


class G2PRecorder(Recorder):
    def phonemize(self, text):
        self.probed = text
        return "ʃalˈom"


def _renikud_cfg():
    return TtsConfig(he_engine="phonikud", he_g2p="renikud", renikud_model="renikud.onnx",
                     phonikud_model="g2p.onnx", phonikud_voice_he="voice.onnx")


def test_build_hebrew_tts_renikud_g2p(monkeypatch):
    monkeypatch.setattr(mirror.tts, "PhonikudHebrewTTS", Recorder)
    monkeypatch.setattr(mirror.tts, "RenikudG2P", G2PRecorder)
    he = build_hebrew_tts(_renikud_cfg())
    assert he.args[1:] == ("voice.onnx", None)
    g2p = he.kwargs["g2p"]
    assert isinstance(g2p, G2PRecorder) and g2p.args == ("renikud.onnx",)
    assert g2p.probed  # loaded and tried at build time, not first at warmup


class BrokenG2P:
    def __init__(self, *args):
        pass

    def phonemize(self, text):
        raise ImportError("No module named 'renikud_onnx'")


def test_build_hebrew_tts_renikud_failure_falls_back_to_phonikud(monkeypatch, caplog):
    monkeypatch.setattr(mirror.tts, "PhonikudHebrewTTS", Recorder)
    monkeypatch.setattr(mirror.tts, "RenikudG2P", BrokenG2P)
    with caplog.at_level("WARNING", logger="mirror.voice"):
        he = build_hebrew_tts(_renikud_cfg())
    assert he.args == ("g2p.onnx", "voice.onnx", None) and he.kwargs == {"g2p": None}
    assert any("Phonikud" in r.getMessage() and r.levelname == "WARNING"
               for r in caplog.records)


def test_build_hebrew_tts_both_g2ps_fail_raises_clearly(monkeypatch):
    def no_phonikud(*args, **kwargs):
        raise FileNotFoundError("g2p.onnx")

    monkeypatch.setattr(mirror.tts, "PhonikudHebrewTTS", no_phonikud)
    monkeypatch.setattr(mirror.tts, "RenikudG2P", BrokenG2P)
    with pytest.raises(RuntimeError, match="RenikudPlus.*Phonikud") as e:
        build_hebrew_tts(_renikud_cfg())
    assert isinstance(e.value.__cause__, FileNotFoundError)


def test_build_hebrew_tts_unknown_g2p():
    with pytest.raises(ValueError, match="he_g2p"):
        build_hebrew_tts(TtsConfig(he_engine="phonikud", he_g2p="espeak"))


def test_build_hebrew_tts_mms(monkeypatch):
    monkeypatch.setattr(mirror.tts, "MmsHebrewTTS", Recorder)
    he = build_hebrew_tts(TtsConfig(he_engine="mms", mms_model_he="facebook/mms-tts-heb"))
    assert isinstance(he, Recorder) and he.args == ("facebook/mms-tts-heb",)


def test_build_hebrew_tts_unknown_engine():
    with pytest.raises(ValueError, match="he_engine"):
        build_hebrew_tts(TtsConfig(he_engine="espeak"))


def test_open_output_failure_is_not_fatal(caplog):
    class NoSpeaker:
        def open(self, sr):
            raise OSError("no such device")

    with caplog.at_level("WARNING", logger="mirror.voice"):
        open_output(NoSpeaker(), 22050)  # must not raise
    assert "no such device" in caplog.text


def test_open_output_opens_at_rate():
    class Speaker:
        rate = None

        def open(self, sr):
            self.rate = sr

    out = Speaker()
    open_output(out, 22050)
    assert out.rate == 22050


def test_world_file_resolves_against_repo_root(tmp_path):
    cfg_path = tmp_path / "config" / "orin.toml"
    assert resolve_repo_path("config/oracle_world.md", str(cfg_path)) == str(
        tmp_path / "config" / "oracle_world.md")
    assert resolve_repo_path("/abs/world.md", str(cfg_path)) == "/abs/world.md"
    assert resolve_repo_path("config/w.md", None) == "config/w.md"


class FakeLights:
    def __init__(self, *args, fail=False):
        self.states = []
        self.fail = fail

    def set_state(self, state):
        self.states.append(state)
        if self.fail:
            raise OSError("serial gone")


def _run_until_vad(monkeypatch, tmp_path, lights, vad_error):
    """Run voice_main.run with every device stubbed; loading the VAD raises vad_error,
    so run() goes straight to its shutdown path."""
    import mirror.voice_main as vm
    from mirror.config import Config

    class Nothing:
        def __init__(self, *args, **kwargs):
            pass

        def send(self, *args):
            pass

        def close(self):
            pass

    def no_vad(*args):
        raise vad_error

    monkeypatch.setattr(vm.signal, "signal", lambda *args: None)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.setattr(vm.FaceBus, "sender", classmethod(lambda cls, port: Nothing()))
    monkeypatch.setattr(vm, "find_device", lambda *args: None)
    monkeypatch.setattr(vm, "Player", Nothing)
    monkeypatch.setattr(vm, "LogLights", lambda: lights)
    monkeypatch.setattr(vm, "SileroVad", no_vad)
    cfg = Config()
    cfg.models_dir = str(tmp_path)
    return vm.run(cfg)


def test_shutdown_returns_lights_to_idle(monkeypatch, tmp_path):
    from mirror.states import State
    from mirror.voice_main import _Shutdown

    lights = FakeLights()
    assert _run_until_vad(monkeypatch, tmp_path, lights, _Shutdown()) == 0
    assert lights.states == [State.IDLE, State.IDLE]  # startup push, then shutdown push


def test_crash_still_returns_lights_to_idle(monkeypatch, tmp_path):
    from mirror.states import State

    lights = FakeLights()
    with pytest.raises(RuntimeError, match="boom"):
        _run_until_vad(monkeypatch, tmp_path, lights, RuntimeError("boom"))
    assert lights.states[-1] == State.IDLE and len(lights.states) == 2


def test_shutdown_lights_failure_is_not_fatal(monkeypatch, tmp_path, caplog):
    from mirror.voice_main import _Shutdown

    lights = FakeLights(fail=True)
    with caplog.at_level("ERROR", logger="mirror.voice"):
        assert _run_until_vad(monkeypatch, tmp_path, lights, _Shutdown()) == 0
    assert len(lights.states) == 2 and "shutdown lights push failed" in caplog.text


def test_warm_llm_warms_both_languages_and_survives_failure(caplog):
    class LLM:
        def __init__(self):
            self.langs = []

        def warm(self, lang):
            self.langs.append(lang)
            if lang == "en":
                raise LLMError("down")
            return 5.0

    llm = LLM()
    assert warm_llm(llm) == ["en"]
    assert llm.langs == ["en", "he"]
    assert "down" in caplog.text


class _Resettable:
    def __init__(self):
        self.resets = 0

    def reset(self):
        self.resets += 1


class _Segmenter(_Resettable):
    def feed(self, chunk):
        return [chunk]


class _Gate:
    def __init__(self):
        import threading

        self.answering = threading.Event()


class _StallingMic:
    """Yields `chunks` reads, then raises MicStalled (like Mic.read's timeout)."""

    def __init__(self, chunks, log):
        self.chunks, self.reads, self.log = chunks, 0, log

    def __enter__(self):
        self.log.append("open")
        return self

    def __exit__(self, *exc):
        self.log.append("close")

    def read(self):
        from mirror.audio import MicStalled
        import numpy as np

        if self.reads >= self.chunks:
            raise MicStalled("no audio")
        self.reads += 1
        return np.zeros(512, np.float32)


def _run_worker(open_source, max_failures=3):
    import queue
    import threading

    from mirror.voice_main import audio_worker

    segments, stop, failed = queue.Queue(maxsize=100), threading.Event(), threading.Event()
    seg, vad = _Segmenter(), _Resettable()
    audio_worker(open_source, seg, vad, _Gate(), segments, stop, failed,
                 max_failures=max_failures, retry_s=0.0)
    return failed, segments, seg, vad


def test_audio_worker_reopens_a_stalled_mic_then_gives_up(caplog):
    events, mics = [], []

    def open_mic():
        mics.append(_StallingMic(0, events))
        return mics[-1]

    with caplog.at_level("WARNING", logger="mirror.voice"):
        failed, *_ = _run_worker(open_mic)
    assert failed.is_set()
    assert events == ["open", "close"] * 3
    assert "reopening" in caplog.text and "giving up" in caplog.text


def test_audio_worker_counts_only_consecutive_failures():
    events, opened = [], []
    script = [2, 0, 5, 0, 0]  # chunks each mic yields before stalling

    def open_mic():
        opened.append(1)
        return _StallingMic(script[len(opened) - 1], events)

    failed, segments, seg, vad = _run_worker(open_mic)
    assert failed.is_set()
    assert len(opened) == 5  # failures 1, 2; audio in between resets it: 1, 2, 3
    assert segments.qsize() == 7
    assert seg.resets == vad.resets == 5  # fresh VAD state on each reopen


def test_audio_worker_retries_a_failing_open():
    calls = []

    def open_mic():
        calls.append(1)
        raise RuntimeError("Error querying device -1")

    failed, *_ = _run_worker(open_mic)
    assert failed.is_set() and len(calls) == 3


def test_audio_worker_fails_at_once_on_a_pipeline_bug(caplog):
    events = []

    class BadSegmenter(_Segmenter):
        def feed(self, chunk):
            raise ValueError("bug")

    import queue
    import threading

    from mirror.voice_main import audio_worker

    failed = threading.Event()
    audio_worker(lambda: _StallingMic(5, events), BadSegmenter(), _Resettable(), _Gate(),
                 queue.Queue(), threading.Event(), failed, retry_s=0.0)
    assert failed.is_set() and events == ["open", "close"]
    assert "audio thread failed" in caplog.text


def test_segment_queue_drops_the_oldest_when_full():
    import queue

    from mirror.voice_main import _put_dropping_oldest

    q = queue.Queue(maxsize=3)
    for i in range(5):
        _put_dropping_oldest(q, i)
    assert [q.get_nowait() for _ in range(3)] == [2, 3, 4]


def test_signal_during_startup_shuts_down_cleanly(monkeypatch, tmp_path):
    """A SIGTERM while the face bus / lights / devices are still being set up is a
    clean shutdown (exit 0), not a traceback."""
    import mirror.voice_main as vm
    from mirror.voice_main import _Shutdown

    def interrupted(cls, port):
        raise _Shutdown()

    monkeypatch.setattr(vm.FaceBus, "sender", classmethod(interrupted))
    monkeypatch.setattr(vm.signal, "signal", lambda *args: None)
    monkeypatch.setattr(vm, "LogLights", lambda: FakeLights())
    from mirror.config import Config
    cfg = Config()
    cfg.models_dir = str(tmp_path)
    assert vm.run(cfg) == 0


def test_second_signal_is_ignored():
    """The first SIGINT/SIGTERM raises _Shutdown; a second one (impatient Ctrl-C
    during teardown) is ignored, so the cleanup in run()'s finally completes."""
    import threading

    from mirror.voice_main import _Shutdown, _signal_handler

    stop = threading.Event()
    handler = _signal_handler(stop)
    with pytest.raises(_Shutdown):
        handler(2, None)
    assert stop.is_set()
    assert handler(2, None) is None


def test_wav_source_tail_is_configurable(tmp_path):
    import wave

    from mirror.voice_main import WavSource

    path = tmp_path / "a.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(np.zeros(1600, "<i2").tobytes())
    src = WavSource(str(path), 16000, tail_s=0.5)
    assert len(src._audio) == 1600 + 8000


def test_wav_tail_covers_the_segmenter_silence():
    from mirror.vad import Segmenter
    from mirror.voice_main import WAV_TAIL_MARGIN_S, wav_tail_s

    seg = Segmenter(lambda f: False, frame_samples=512, sample_rate=16000,
                    min_silence_ms=1500)
    assert seg.min_silence_s == pytest.approx(46 * 0.032)  # whole frames, floored
    assert wav_tail_s(seg) == pytest.approx(seg.min_silence_s + WAV_TAIL_MARGIN_S)
    assert wav_tail_s(seg) > 1.5


# --- tts.engine: piper (default) | pocket ---------------------------------------------

class FakePiper(Recorder):
    sample_rate = 22050


class FakePocket(Recorder):
    sample_rate = 24000

    def for_lang(self, lang):
        return ("view", lang)


def _stub_piper_stack(monkeypatch):
    monkeypatch.setattr(mirror.tts, "PiperTTS", FakePiper)
    monkeypatch.setattr(mirror.tts, "PhonikudHebrewTTS", Recorder)
    monkeypatch.setattr(mirror.tts, "RenikudG2P", G2PRecorder)


def test_build_tts_piper_by_default(monkeypatch):
    from mirror.voice_main import build_tts

    _stub_piper_stack(monkeypatch)
    monkeypatch.setattr(mirror.tts, "PocketTTS", lambda *a, **k: pytest.fail("loaded pocket"))
    router, rate, engine = build_tts(TtsConfig(piper_voice_en="en.onnx"))
    assert engine == "piper" and rate == 22050
    assert isinstance(router._en, FakePiper) and router._en.args == ("en.onnx",)
    assert isinstance(router._he, Recorder)


def test_build_tts_pocket(monkeypatch):
    from mirror.voice_main import build_tts

    _stub_piper_stack(monkeypatch)
    monkeypatch.setattr(mirror.tts, "PocketTTS", FakePocket)
    cfg = TtsConfig(engine="pocket", pocket_model="p.onnx", pocket_voice="michael",
                    pocket_voice_file="v.npy", pocket_threads=3, renikud_model="r.onnx")
    router, rate, engine = build_tts(cfg)
    assert engine == "pocket" and rate == 24000
    assert router._en == ("view", "en") and router._he == ("view", "he")


def test_build_tts_pocket_passes_its_settings(monkeypatch):
    from mirror.voice_main import build_tts

    _stub_piper_stack(monkeypatch)
    made = []

    class Pocket(FakePocket):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            made.append(self)

    monkeypatch.setattr(mirror.tts, "PocketTTS", Pocket)
    build_tts(TtsConfig(engine="pocket", pocket_model="p.onnx", pocket_voice="michael",
                        pocket_voice_file="v.npy", pocket_threads=3, renikud_model="r.onnx"))
    (p,) = made
    assert p.args == ("p.onnx",)
    assert p.kwargs["voice"] == "michael" and p.kwargs["voice_file"] == "v.npy"
    assert p.kwargs["threads"] == 3
    g2p = p.kwargs["g2p"]
    assert isinstance(g2p, G2PRecorder) and g2p.args == ("r.onnx",) and g2p.probed


def test_build_tts_pocket_failure_falls_back_to_piper(monkeypatch, caplog):
    from mirror.voice_main import build_tts

    _stub_piper_stack(monkeypatch)

    def broken(*args, **kwargs):
        raise ImportError("No module named 'pocket_tts_onnx'")

    monkeypatch.setattr(mirror.tts, "PocketTTS", broken)
    with caplog.at_level("WARNING", logger="mirror.voice"):
        router, rate, engine = build_tts(TtsConfig(engine="pocket"))
    assert engine == "piper" and rate == 22050 and isinstance(router._en, FakePiper)
    assert any(r.levelname == "WARNING" and "pocket" in r.getMessage().lower()
               for r in caplog.records)


def test_build_tts_pocket_without_hebrew_g2p_falls_back_to_piper(monkeypatch, caplog):
    from mirror.voice_main import build_tts

    _stub_piper_stack(monkeypatch)
    monkeypatch.setattr(mirror.tts, "PocketTTS", FakePocket)
    monkeypatch.setattr(mirror.tts, "RenikudG2P", BrokenG2P)  # the piper stack falls back too
    with caplog.at_level("WARNING", logger="mirror.voice"):
        router, rate, engine = build_tts(TtsConfig(engine="pocket"))
    assert engine == "piper" and isinstance(router._en, FakePiper)


def test_build_tts_unknown_engine():
    from mirror.voice_main import build_tts

    with pytest.raises(ValueError, match="engine"):
        build_tts(TtsConfig(engine="espeak"))
