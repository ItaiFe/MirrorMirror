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


@dataclass
class ServersConfig:
    stt_url: str = "http://127.0.0.1:8081"
    llm_url: str = "http://127.0.0.1:8080"
    timeout_s: float = 20


@dataclass
class TtsConfig:
    piper_voice_en: str = "models/en_US-lessac-medium.onnx"
    mms_model_he: str = "facebook/mms-tts-heb"


@dataclass
class FaceConfig:
    size: int = 252
    screen_w: int = 1920
    screen_h: int = 1080
    fullscreen: bool = False
    window_scale: int = 3
    bus_port: int = 7777


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
