"""Bilingual text-to-speech: Piper (English), Phonikud + Piper or Meta MMS (Hebrew),
and a router."""
from __future__ import annotations

import json
import re
from typing import Protocol

import numpy as np


class Synth(Protocol):
    def synth(self, text: str) -> tuple[np.ndarray, int]:
        """Return (float32 mono audio, sample rate)."""


def _normalize(audio, rate: int) -> tuple[np.ndarray, int]:
    """int16 -> float32 in -1..1, flatten to mono 1-D, float64 -> float32."""
    a = np.asarray(audio)
    if a.dtype == np.int16:
        a = a.astype(np.float32) / 32768.0
    else:
        a = a.astype(np.float32, copy=False)
    return a.reshape(-1), int(rate)


class PiperTTS:
    def __init__(self, voice_path: str):
        from piper import PiperVoice  # lazy: heavy optional dep

        self._voice = PiperVoice.load(voice_path)
        self._rate = int(self._voice.config.sample_rate)

    def synth(self, text: str) -> tuple[np.ndarray, int]:
        voice = self._voice
        if hasattr(voice, "synthesize_stream_raw"):
            # piper-tts 1.2: yields raw int16 bytes
            pcm = b"".join(voice.synthesize_stream_raw(text))
            audio = np.frombuffer(pcm, dtype=np.int16)
        else:
            # newer piper-tts: synthesize() yields AudioChunk objects
            parts = []
            for chunk in voice.synthesize(text):
                if hasattr(chunk, "audio_int16_array"):
                    parts.append(np.asarray(chunk.audio_int16_array))
                else:
                    parts.append(np.asarray(chunk.audio_float_array))
            audio = np.concatenate(parts) if parts else np.zeros(0, np.float32)
        return _normalize(audio, self._rate)


# A Hebrew prefix letter joined to a number by a hyphen or maqaf ("ו-25", "ב־3").
# Phonikud reads that hyphen as "minus", so it is dropped.
_PREFIX_HYPHEN_NUMBER = re.compile(r"(?<=[\u05d0-\u05ea])[-\u05be](?=\d)")


def prepare_hebrew(text: str) -> str:
    """Normalize plain Hebrew text before Phonikud diacritizes and phonemizes it."""
    text = _PREFIX_HYPHEN_NUMBER.sub("", text)
    return " ".join(text.split())


class PhonikudHebrewTTS:
    """Plain Hebrew text -> Phonikud (niqqud, stress) -> IPA -> Piper VITS voice (ONNX, CPU).

    Phonikud loads its character tokenizer (dicta-il/dictabert-large-char-menaked)
    through the Hugging Face cache, so HF_HOME must point at a cache that has it
    (deploy/fetch_models.sh puts it in MODELS_DIR/hf).
    """

    def __init__(self, phonikud_model: str, voice_model: str, voice_config: str | None = None):
        import onnxruntime  # lazy: heavy optional deps
        from phonikud import phonemize
        from phonikud_onnx import Phonikud
        from piper.config import PiperConfig
        from piper.voice import PiperVoice

        cpu = ["CPUExecutionProvider"]  # keep the GPU free for the LLM
        self._phonemize = phonemize
        self._diacritizer = Phonikud.from_session(
            onnxruntime.InferenceSession(phonikud_model, providers=cpu))
        with open(voice_config or f"{voice_model}.json", encoding="utf-8") as f:
            config = json.load(f)
        # The voice takes raw IPA characters ("phoneme_type": "raw"), which piper's
        # config enum doesn't know. Phonemes are always passed in, so piper's own
        # phonemizer is never used and "text" is a safe stand-in.
        config["phoneme_type"] = "text"
        self._voice = PiperVoice(
            session=onnxruntime.InferenceSession(voice_model, providers=cpu),
            config=PiperConfig.from_dict(config))
        self._known = set(self._voice.config.phoneme_id_map)
        self._rate = int(self._voice.config.sample_rate)

    def phonemes(self, text: str) -> str:
        """IPA for plain Hebrew text (characters the voice doesn't know are dropped)."""
        ipa = self._phonemize(self._diacritizer.add_diacritics(prepare_hebrew(text)))
        return "".join(c for c in ipa if c in self._known).strip()

    def synth(self, text: str) -> tuple[np.ndarray, int]:
        ipa = self.phonemes(text)
        if not ipa:
            return np.zeros(0, np.float32), self._rate
        ids = self._voice.phonemes_to_ids(list(ipa))
        return _normalize(self._voice.phoneme_ids_to_audio(ids), self._rate)


class MmsHebrewTTS:
    def __init__(self, model_name: str):
        import torch  # lazy: heavy optional deps
        from transformers import AutoTokenizer, VitsModel

        # CUDA if available, else CPU. MPS is deliberately skipped: VITS ops
        # are unreliable on MPS.
        self._torch = torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = VitsModel.from_pretrained(model_name).to(self._device).eval()
        self._rate = int(self._model.config.sampling_rate)

    def synth(self, text: str) -> tuple[np.ndarray, int]:
        inputs = self._tokenizer(text, return_tensors="pt").to(self._device)
        with self._torch.no_grad():
            wave = self._model(**inputs).waveform[0]
        return _normalize(wave.cpu().numpy(), self._rate)


class TTSRouter:
    def __init__(self, en: Synth, he: Synth):
        self._en = en
        self._he = he

    def synth(self, text: str, lang: str) -> tuple[np.ndarray, int]:
        engine = self._he if lang == "he" else self._en
        audio, rate = engine.synth(text)
        return _normalize(audio, rate)

    def warmup(self) -> None:
        self.synth("hello", "en")
        self.synth("שלום", "he")
