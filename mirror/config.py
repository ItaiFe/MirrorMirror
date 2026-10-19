import os
from dataclasses import MISSING, dataclass, field, fields
from typing import Optional

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib


@dataclass
class AudioConfig:
    input_device: Optional[str] = None
    output_device: Optional[str] = None
    sample_rate: int = 16000
    # several devices match input_device/output_device: prefer one that converts
    # formats (ALSA plughw, PulseAudio/PipeWire, "default") over a raw hw: device
    prefer_alsa_plug: bool = True


@dataclass
class ServersConfig:
    stt_url: str = "http://127.0.0.1:8081"
    llm_url: str = "http://127.0.0.1:8080"
    timeout_s: float = 20


@dataclass
class TtsConfig:
    piper_voice_en: str = "models/en_US-lessac-medium.onnx"
    he_engine: str = "phonikud"  # "phonikud" (G2P + Phonikud Piper voice) | "mms"
    he_g2p: str = "renikud"  # the voice's text->IPA stage: "renikud" | "phonikud"
    renikud_model: str = "models/renikud-plus.int8.onnx"
    phonikud_model: str = "models/phonikud-1.0.int8.onnx"
    phonikud_voice_he: str = "models/phonikud-tts-he.onnx"
    phonikud_voice_config_he: Optional[str] = None  # default: <phonikud_voice_he>.json
    mms_model_he: str = "facebook/mms-tts-heb"
    # "piper" (Piper English + the Hebrew engine above) | "pocket" (Pocket TTS for both
    # languages, Hebrew via RenikudPlus IPA; falls back to piper if it fails to load)
    engine: str = "piper"
    pocket_model: str = "models/pocket-tts-english-ipa.onnx"
    pocket_voice: str = "javert"  # a built-in voice; ignored when pocket_voice_file is set
    pocket_voice_file: Optional[str] = None  # a cloned voice (.npy, tools/clone_voice.py)
    pocket_threads: int = 2


@dataclass
class FaceConfig:
    size: int = 252
    screen_w: int = 1920
    screen_h: int = 1080
    fullscreen: bool = False
    window_scale: int = 3
    bus_port: int = 7777
    renderer: str = "sprite"  # "sprite" (face bank) | "procedural" (drawn lines)
    bank: str = "assets/face/stub"  # face bank directory; relative to the repo root


@dataclass
class LightsConfig:
    kind: str = "log"  # "wled" | "log"
    serial_port: Optional[str] = None
    baud: int = 115200


@dataclass
class ConversationConfig:
    listen_timeout_s: float = 8.0
    followup_s: float = 6.0
    max_reply_sentences: int = 3
    max_reply_chars: int = 300
    filler_enabled: bool = True     # speak a short filler line while the LLM thinks
    filler_min_wait_s: float = 0.0  # ...only if no sentence is ready this soon (0: always)
    world_file: str = "config/oracle_world.md"  # the oracle's background; relative to the repo
    # idle lures: a short line, now and then, while no one is talking to the mirror
    lures_enabled: bool = True
    lure_min_idle_s: float = 120.0  # no lure until this long after the last interaction
    # [min, max]: each wait between lures (or since the last interaction) is drawn from it
    lure_interval_s: list = field(default_factory=lambda: [180.0, 420.0])
    # "HH:MM-HH:MM" local-time windows with no lures; a window may cross midnight
    lure_quiet_hours: list = field(default_factory=list)


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    servers: ServersConfig = field(default_factory=ServersConfig)
    tts: TtsConfig = field(default_factory=TtsConfig)
    face: FaceConfig = field(default_factory=FaceConfig)
    lights: LightsConfig = field(default_factory=LightsConfig)
    conversation: ConversationConfig = field(default_factory=ConversationConfig)
    models_dir: str = "models"  # silero_vad.onnx and the Hugging Face cache (hf/)


def load_config(path: str) -> Config:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    kwargs = {}
    for fld in fields(Config):
        if fld.default_factory is MISSING:  # top-level scalar
            if fld.name in raw:
                kwargs[fld.name] = raw[fld.name]
            continue
        kwargs[fld.name] = fld.default_factory(**raw.get(fld.name, {}))
    return Config(**kwargs)


def resolve_repo_path(path: str, config_path) -> str:
    """A relative path from the config, resolved against the repo root (the parent
    of the config file's directory); against the working directory without one."""
    if os.path.isabs(path) or config_path is None:
        return path
    repo = os.path.dirname(os.path.dirname(os.path.abspath(config_path)))
    return os.path.join(repo, path)
