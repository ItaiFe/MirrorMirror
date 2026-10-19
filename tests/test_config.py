import pytest


from mirror.config import TtsConfig, load_config

DEFAULT_G2P = "renikud"  # tts.he_g2p: Task 20 Step 5 (A/B listening set)


def test_load_mac_config():
    c = load_config("config/mac.toml")
    assert c.face.size == 252 and c.face.fullscreen is False and c.face.window_scale == 3
    assert c.lights.kind == "log"
    assert c.servers.llm_url.startswith("http://127.0.0.1")
    assert c.models_dir == "models"
    assert c.tts.he_engine == "phonikud"
    assert c.tts.phonikud_model == "models/phonikud-1.0.int8.onnx"
    assert c.tts.he_g2p == DEFAULT_G2P
    assert c.tts.renikud_model == "models/renikud-plus.int8.onnx"
    assert c.tts.phonikud_voice_he == "models/phonikud-tts-he.onnx"
    assert c.face.renderer == "sprite" and c.face.bank == "assets/face/stub"


def test_load_orin_config():
    c = load_config("config/orin.toml")
    assert (c.face.screen_w, c.face.screen_h) == (1920, 1080) and c.face.fullscreen is True
    assert c.lights.kind == "wled" and c.lights.serial_port == "/dev/ttyUSB0"
    assert c.models_dir == "/home/mirror/mirror-models"
    assert c.tts.he_engine == "phonikud"
    assert c.tts.phonikud_voice_he == "/home/mirror/mirror-models/phonikud-tts-he.onnx"
    assert c.tts.he_g2p == DEFAULT_G2P
    assert c.tts.renikud_model == "/home/mirror/mirror-models/renikud-plus.int8.onnx"
    assert c.face.renderer == "sprite" and c.face.bank == "assets/face/stub"


def test_default_g2p_matches_the_shipped_configs():
    assert TtsConfig().he_g2p == DEFAULT_G2P
    assert TtsConfig().renikud_model == "models/renikud-plus.int8.onnx"


def test_tts_engine_defaults_to_piper_with_pocket_off():
    t = TtsConfig()
    assert t.engine == "piper"
    assert t.pocket_model == "models/pocket-tts-english-ipa.onnx"
    assert t.pocket_voice == "javert" and t.pocket_voice_file is None
    assert t.pocket_threads == 2


def test_shipped_configs_name_the_pocket_model():
    mac, orin = load_config("config/mac.toml").tts, load_config("config/orin.toml").tts
    # the Mac speaks in the oracle's cloned voice; the Orin keeps Piper until Pocket
    # is measured there
    assert mac.engine == "pocket" and mac.pocket_voice_file == "models/voices/mirror.npy"
    assert orin.engine == "piper"
    assert mac.pocket_model == "models/pocket-tts-english-ipa.onnx"
    assert orin.pocket_model == "/home/mirror/mirror-models/pocket-tts-english-ipa.onnx"
    assert mac.pocket_voice == orin.pocket_voice == "javert"
    assert mac.pocket_threads == orin.pocket_threads == 2


def test_shipped_configs_set_the_lures():
    for path in ("config/mac.toml", "config/orin.toml"):
        c = load_config(path).conversation
        assert c.lures_enabled is True, path
        assert c.lure_min_idle_s == 120 and list(c.lure_interval_s) == [180, 420], path
        assert c.lure_quiet_hours == [], path


def test_voice_effect_defaults_off():
    cfg = TtsConfig()
    assert cfg.voice_speed == 1.0 and cfg.voice_room == 0.0


def test_shipped_configs_keep_the_voice_effect_off():
    for name in ("mac", "orin"):
        cfg = load_config(f"config/{name}.toml")
        assert cfg.tts.voice_speed == 1.0 and cfg.tts.voice_room == 0.0


def test_voice_effect_is_read_from_toml(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text("[tts]\nvoice_speed = 0.86\nvoice_room = 1.0\n")
    cfg = load_config(str(p))
    assert cfg.tts.voice_speed == 0.86 and cfg.tts.voice_room == 1.0


@pytest.mark.parametrize("kwargs", [{"voice_speed": 0.5}, {"voice_speed": 1.3},
                                    {"voice_room": -0.1}, {"voice_room": 1.5}])
def test_voice_effect_out_of_range(kwargs):
    with pytest.raises(ValueError, match="voice_"):
        TtsConfig(**kwargs)
