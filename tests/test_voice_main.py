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
