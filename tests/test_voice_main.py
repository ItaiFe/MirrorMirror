import pytest

import mirror.tts
from mirror.config import TtsConfig
from mirror.voice_main import build_hebrew_tts


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
