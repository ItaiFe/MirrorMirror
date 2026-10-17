"""Speech-to-text client for whisper.cpp's whisper-server."""
import io
import wave
from dataclasses import dataclass

import numpy as np
import requests

# Name of the detected-language field in the verbose_json response.
# To be pinned against a real whisper-server in Task 11.
LANGUAGE_FIELD = "language"
_LANGUAGE_FALLBACK_FIELD = "detected_language"


class STTError(Exception):
    """Raised when the whisper-server request fails or returns garbage."""


@dataclass
class Transcript:
    text: str
    lang: str  # "he" or "en"


def _to_wav_bytes(samples: np.ndarray, sr: int) -> bytes:
    pcm = (np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0) * 32767.0)
    pcm = np.rint(pcm).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


class WhisperServerSTT:
    def __init__(self, url: str, timeout_s: float):
        self._endpoint = url.rstrip("/") + "/inference"
        self._timeout_s = timeout_s

    def transcribe(self, samples: np.ndarray, sr: int = 16000) -> Transcript:
        wav = _to_wav_bytes(samples, sr)
        try:
            resp = requests.post(
                self._endpoint,
                files={"file": ("speech.wav", wav, "audio/wav")},
                data={"response_format": "verbose_json", "language": "auto",
                      "temperature": "0"},
                timeout=self._timeout_s,
            )
            resp.raise_for_status()
            payload = resp.json()
            text = payload["text"]
        except requests.exceptions.RequestException as e:  # incl. Timeout, HTTPError
            raise STTError(f"whisper-server request failed: {e}") from e
        except (ValueError, KeyError, TypeError) as e:  # bad JSON / missing text
            raise STTError(f"malformed whisper-server response: {e}") from e
        lang = payload.get(LANGUAGE_FIELD) or payload.get(_LANGUAGE_FALLBACK_FIELD)
        return Transcript(str(text).strip(), "he" if lang == "he" else "en")
