"""Bilingual text-to-speech: Piper (English), Phonikud + Piper or Meta MMS (Hebrew),
and a router."""
from __future__ import annotations

import json
import logging
import re
from typing import Protocol

import numpy as np

from mirror.visemes import VisemeTrack, track_from_alignments

log = logging.getLogger("mirror.tts")


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


class _Tracks:
    """Builds viseme tracks for one voice; warns once if its alignments fail."""

    def __init__(self, voice_name: str):
        self._voice_name = voice_name
        self._warned = False

    def build(self, alignments: list[tuple[str, int]] | None, sr: int,
              total_samples: int) -> VisemeTrack | None:
        if alignments is None:
            if not self._warned:
                log.warning("%s: no phoneme alignments; the mouth falls back to loudness",
                            self._voice_name)
                self._warned = True
            return None
        return track_from_alignments(alignments, sr, total_samples)


class PiperTTS:
    def __init__(self, voice_path: str):
        from piper import PiperVoice  # lazy: heavy optional dep

        # include_alignments patches the model in memory (needs the `onnx` package,
        # the piper-tts[alignment] extra); without it, synth_visemes returns no track.
        self._voice = PiperVoice.load(voice_path, include_alignments=True)
        self._rate = int(self._voice.config.sample_rate)
        self._tracks = _Tracks(f"piper {voice_path}")

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

    def synth_visemes(self, text: str) -> tuple[np.ndarray, int, VisemeTrack | None]:
        """Like synth, plus the mouth-shape track from Piper's phoneme alignments."""
        parts, alignments = [], []
        for chunk in self._voice.synthesize(text, include_alignments=True):
            parts.append(np.asarray(chunk.audio_int16_array))
            if alignments is not None:
                if chunk.phoneme_alignments is None:
                    alignments = None
                else:
                    alignments.extend((a.phoneme, int(a.num_samples))
                                      for a in chunk.phoneme_alignments)
        audio, rate = _normalize(
            np.concatenate(parts) if parts else np.zeros(0, np.int16), self._rate)
        return audio, rate, self._tracks.build(alignments, rate, len(audio))


def group_phoneme_samples(phonemes: list[str], id_map: dict[str, list[int]],
                          id_samples) -> list[tuple[str, int]] | None:
    """Per-phoneme-id sample counts -> [(phoneme, samples)], or None if they don't line up.

    Piper's ids are: BOS, PAD, then each phoneme's ids followed by PAD, then EOS;
    each PAD is counted with the phoneme before it (as piper itself does).
    """
    pad = len(id_map.get("_", []))
    out, i = [], 0
    for ph in ["^", *phonemes, "$"]:
        n = len(id_map.get(ph, [])) + (pad if ph != "$" else 0)
        out.append((ph, int(np.sum(id_samples[i:i + n]))))
        i += n
    return out if i == len(id_samples) else None


def _alignment_session(model_path: str, providers: list[str]):
    """onnxruntime session for a Piper voice, patched in memory to also output the
    per-phoneme-id durations. Falls back to the plain model if `onnx` is missing."""
    import onnxruntime  # lazy: heavy optional dep

    model = model_path
    try:
        import onnx
        from piper.patch_voice_with_alignment import add_alignment_output

        proto = onnx.load(model_path)
        add_alignment_output(proto)
        model = proto.SerializeToString()
    except ImportError:
        log.warning("onnx not installed (piper-tts[alignment]); no alignments for %s",
                    model_path)
    except ValueError as e:  # already patched, or no Ceil tensor
        log.debug("not patching %s for alignments: %s", model_path, e)
    return onnxruntime.InferenceSession(model, providers=providers)


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
        self._voice = PiperVoice(session=_alignment_session(voice_model, cpu),
                                 config=PiperConfig.from_dict(config))
        self._known = set(self._voice.config.phoneme_id_map)
        self._rate = int(self._voice.config.sample_rate)
        self._tracks = _Tracks(f"phonikud {voice_model}")

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

    def synth_visemes(self, text: str) -> tuple[np.ndarray, int, VisemeTrack | None]:
        """Like synth, plus the mouth-shape track from the voice's phoneme alignments."""
        ipa = self.phonemes(text)
        if not ipa:
            return np.zeros(0, np.float32), self._rate, None
        phonemes = list(ipa)
        ids = self._voice.phonemes_to_ids(phonemes)
        raw, id_samples = self._voice.phoneme_ids_to_audio(ids, include_alignments=True)
        audio, rate = _normalize(raw, self._rate)
        alignments = None
        if id_samples is not None:
            alignments = group_phoneme_samples(
                phonemes, self._voice.config.phoneme_id_map, id_samples)
        return audio, rate, self._tracks.build(alignments, rate, len(audio))


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

    def synth_visemes(self, text: str, lang: str) -> tuple[np.ndarray, int, VisemeTrack | None]:
        """Audio plus a mouth-shape track; the track is None for engines that can't
        align phonemes (MMS), and the face then shapes the mouth from loudness."""
        engine = self._he if lang == "he" else self._en
        if not hasattr(engine, "synth_visemes"):
            audio, rate = engine.synth(text)
            return (*_normalize(audio, rate), None)
        audio, rate, track = engine.synth_visemes(text)
        return (*_normalize(audio, rate), track)

    def warmup(self) -> None:
        self.synth("hello", "en")
        self.synth("שלום", "he")
