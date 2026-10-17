import pytest

import mirror.tts
from mirror.config import TtsConfig
from mirror.voice_main import build_hebrew_tts, open_output, resolve_repo_path


class Recorder:
    def __init__(self, *args):
        self.args = args


def test_build_hebrew_tts_phonikud(monkeypatch):
    monkeypatch.setattr(mirror.tts, "PhonikudHebrewTTS", Recorder)
    cfg = TtsConfig(he_engine="phonikud", phonikud_model="g2p.onnx",
                    phonikud_voice_he="voice.onnx", phonikud_voice_config_he="voice.json")
    he = build_hebrew_tts(cfg)
    assert isinstance(he, Recorder) and he.args == ("g2p.onnx", "voice.onnx", "voice.json")


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
