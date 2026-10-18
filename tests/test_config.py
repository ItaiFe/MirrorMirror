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
