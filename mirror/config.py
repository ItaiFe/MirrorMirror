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
    # the "villain" character effect on every spoken line (answers, fillers, lures):
    # voice_speed < 1 resamples deeper and slower (1.0 = off), voice_room scales a
    # small-room echo (0 = off). The sample the user picked: 0.86 and 1.0.
    voice_speed: float = 1.0
    voice_room: float = 0.0

    def __post_init__(self):
        if not 0.6 <= self.voice_speed <= 1.2:
            raise ValueError(f"tts.voice_speed must be 0.6..1.2, got {self.voice_speed!r}")
        if not 0.0 <= self.voice_room <= 1.0:
            raise ValueError(f"tts.voice_room must be 0..1, got {self.voice_room!r}")


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
    # the floating face (Task 27): a spirit hovering in the glass
    float_enabled: bool = True
    float_amp_px: float = 5.0  # vertical bob, +/- px at 252 (idle; less in other states)
    float_period_s: float = 6.5  # bob period; sway x1.37, tilt x0.83
    float_tilt_deg: float = 1.2  # +/- rotation
    float_scale: float = 0.93  # the whole face shrunk about the centre: margin for the motion
    float_edge_feather_px: float = 6.0  # the outline fades into black over this many px


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
class LlmConfig:
    """Sampling sent with every llama-server request: a high temperature and
    repetition penalties keep the oracle's answers from all sounding alike."""
    temperature: float = 0.95
    top_p: float = 0.95
    min_p: float = 0.05
    repeat_penalty: float = 1.12
    presence_penalty: float = 0.4
    frequency_penalty: float = 0.3

    def sampling(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    servers: ServersConfig = field(default_factory=ServersConfig)
    tts: TtsConfig = field(default_factory=TtsConfig)
    face: FaceConfig = field(default_factory=FaceConfig)
    lights: LightsConfig = field(default_factory=LightsConfig)
    conversation: ConversationConfig = field(default_factory=ConversationConfig)
    llm: LlmConfig = field(default_factory=LlmConfig)
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
