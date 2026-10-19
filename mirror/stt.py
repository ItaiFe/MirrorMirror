"""Speech-to-text client for whisper.cpp's whisper-server."""
import io
import wave
from dataclasses import dataclass

import numpy as np
import requests

# The transcript's language comes from its script (script_language) first.
# Detected-language field of the verbose_json response, pinned against a real
# whisper-server (whisper.cpp 1.9.5). Its value is the full language name
# ("hebrew", "english"), not the ISO code; both forms are accepted.
LANGUAGE_FIELD = "language"
_LANGUAGE_FALLBACK_FIELD = "detected_language"
_HEBREW = {"he", "hebrew", "iw"}


def script_language(text: str) -> "str | None":
    """Language by script: "he" if Hebrew letters (U+05D0-U+05EA) outnumber Latin
    letters, "en" if Latin letters do, None if there are no letters (or a tie)."""
    hebrew = sum("\u05d0" <= ch <= "\u05ea" for ch in text)
    latin = sum(ch.isascii() and ch.isalpha() for ch in text)
    if hebrew > latin:
        return "he"
    if latin > hebrew:
        return "en"
    return None


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
    def __init__(self, url: str, timeout_s: float, language: str = "auto"):
        self._endpoint = url.rstrip("/") + "/inference"
        self._timeout_s = timeout_s
        self._language = language  # the default for transcribe()

    def transcribe(self, samples: np.ndarray, sr: int = 16000,
                   language: str | None = None) -> Transcript:
        """language: "auto", or a code ("he", "en") to force; None: the constructor's.
        Forcing skips language detection; "en" re-reads English that the Hebrew
        fine-tune wrote in Hebrew letters."""
        language = self._language if language is None else language
        wav = _to_wav_bytes(samples, sr)
        try:
            resp = requests.post(
                self._endpoint,
                files={"file": ("speech.wav", wav, "audio/wav")},
                data={"response_format": "verbose_json", "language": language,
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
        text = "" if text is None else str(text)  # {"text": null}: heard nothing
        text = text.strip()
        # The script decides first. ivrit-ai's Hebrew fine-tune reports "hebrew" for
        # every clip, English included, so the language field alone is useless with it.
        # When it transliterates English into Hebrew letters ("מיר מיר"), the script
        # says he even if a server reported "english": the text is Hebrew letters,
        # so the reply goes out in Hebrew. The field is only the fallback for
        # transcripts with no letters at all.
        lang = script_language(text)
        if lang is None:
            server = payload.get(LANGUAGE_FIELD) or payload.get(_LANGUAGE_FALLBACK_FIELD)
            # case- and region-blind: "HE", "he-IL" and "he_IL" are all Hebrew
            code = str(server).strip().lower().replace("_", "-").split("-")[0]
            lang = "he" if code in _HEBREW else "en"
        return Transcript(text, lang)
