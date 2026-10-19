"""Bilingual text-to-speech: Piper (English), a Hebrew Piper voice fed by a
grapheme-to-phoneme stage (RenikudPlus or Phonikud) or Meta MMS (Hebrew), and a router.
Optionally Pocket TTS, one model for both languages (Hebrew read as RenikudPlus IPA)."""
from __future__ import annotations

import json
import logging
import re
import threading
from typing import Protocol

import numpy as np

from mirror.visemes import VisemeTrack, estimate_track, track_from_alignments, voiced_segments

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

    @property
    def sample_rate(self) -> int:
        return self._rate

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


class PhonikudG2P:
    """Plain Hebrew text -> Phonikud (niqqud, stress) -> IPA.

    Phonikud loads its character tokenizer (dicta-il/dictabert-large-char-menaked)
    through the Hugging Face cache, so HF_HOME must point at a cache that has it
    (deploy/fetch_models.sh puts it in MODELS_DIR/hf).
    """

    def __init__(self, model_path: str):
        import onnxruntime  # lazy: heavy optional deps
        from phonikud import phonemize
        from phonikud_onnx import Phonikud

        self._phonemize = phonemize
        self._diacritizer = Phonikud.from_session(
            onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"]))

    def phonemize(self, text: str) -> str:
        return self._phonemize(self._diacritizer.add_diacritics(prepare_hebrew(text)))


# What the Hebrew Piper voice was trained on: Phonikud's output inventory (its
# consonants, a e i o u, the stress mark) and the punctuation Phonikud keeps.
# RenikudPlus emits the same phonemes, with the same stress notation (ˈ before the
# stressed vowel), but passes other punctuation, markup and Latin words through.
HEBREW_VOICE_SYMBOLS = frozenset("abdefhijklmnopstuvwzɡʁʃʒʔχˈ") | frozenset(" ,.?!")

# Filler interjections are sounds, not words: RenikudPlus would vowelize them
# ("הממ" -> "hamamˈa"). Its force lexicon always wins where a token matches.
RENIKUD_LEXICON = {"הממ": "hmm", "מממ": "mmm", "ממ": "mm", "אהה": "ʔahˈa"}

_JOINER = re.compile(r"(?<=\S)[-\u05be](?=\S)")  # hyphen/maqaf inside a word
_PAUSE = re.compile(r"[-\u05be\u2013\u2014\u2026:;]")  # dashes, ellipsis, : ; -> a comma
_MARKUP = str.maketrans("", "", "*()[]{}\"'«»")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+(?=[,.?!])")
_COMMA_BEFORE_PUNCT = re.compile(r",+(?=[,.?!])")


def map_renikud_ipa(ipa: str, voice_symbols_only: bool = True) -> str:
    """RenikudPlus IPA -> only symbols the Hebrew voice was trained on: pauses become
    commas, a hyphen inside a word joins it, markup goes, and a token with any other
    symbol left (a Latin word passed through) is dropped. With voice_symbols_only=False
    every token is kept (Pocket TTS reads any IPA, and Latin words as text)."""
    ipa = _PAUSE.sub(",", _JOINER.sub("", ipa))
    words = [w.translate(_MARKUP) for w in ipa.split()]
    out = " ".join(w for w in words
                   if w and (not voice_symbols_only or set(w) <= HEBREW_VOICE_SYMBOLS))
    out = _COMMA_BEFORE_PUNCT.sub("", _SPACE_BEFORE_PUNCT.sub("", out))
    return out.lstrip(" ,.?!")


# RenikudPlus's onnxruntime intra-op threads: ORT's default is one per core, and on
# the Orin the 6 cores are shared with whisper, llama and the face.
RENIKUD_THREADS = 2


class RenikudG2P:
    """Plain Hebrew text -> IPA with RenikudPlus (ONNX, CPU), speaker-gender aware
    (0 unknown, 1 male, 2 female; the oracle is masculine). The model loads on the
    first sentence, once; it is always given a local path and no rescoring
    datastore, so nothing is fetched at runtime."""

    def __init__(self, model_path: str, speaker: int = 1):
        self._model_path = model_path
        self._speaker = speaker
        self._g2p = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._g2p is None:
                import onnxruntime as ort  # lazy: heavy optional deps
                from renikud_onnx import G2P

                opts = ort.SessionOptions()
                opts.intra_op_num_threads = RENIKUD_THREADS
                self._g2p = G2P(self._model_path, datastore=None, lexicon=RENIKUD_LEXICON,
                                session_options=opts)
        return self._g2p

    def _raw(self, text: str) -> str:
        g2p = self._g2p or self._load()
        # Numbers, hyphenated prefixes ("ו-25") and maqaf are handled by RenikudPlus.
        return g2p.phonemize(" ".join(text.split()), speaker=self._speaker)

    def phonemize(self, text: str) -> str:
        """IPA in the Hebrew Piper voice's symbol inventory."""
        return map_renikud_ipa(self._raw(text))

    def ipa(self, text: str) -> str:
        """IPA with pauses and markup cleaned but every symbol kept (for Pocket TTS)."""
        return map_renikud_ipa(self._raw(text), voice_symbols_only=False)


class PhonikudHebrewTTS:
    """Plain Hebrew text -> G2P (Phonikud by default, or the given one, e.g.
    RenikudG2P) -> IPA -> the Phonikud Piper VITS voice (ONNX, CPU)."""

    def __init__(self, phonikud_model: str, voice_model: str, voice_config: str | None = None,
                 g2p=None):
        from piper.config import PiperConfig  # lazy: heavy optional deps
        from piper.voice import PiperVoice

        cpu = ["CPUExecutionProvider"]  # keep the GPU free for the LLM
        self._g2p = g2p if g2p is not None else PhonikudG2P(phonikud_model)
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
        self._tracks = _Tracks(f"hebrew voice {voice_model}")

    def phonemes(self, text: str) -> str:
        """IPA for plain Hebrew text (characters the voice doesn't know are dropped)."""
        ipa = self._g2p.phonemize(text)
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
        if not text.strip():  # the tokenizer/VITS model can't take an empty input
            return np.zeros(0, np.float32), self._rate
        inputs = self._tokenizer(text, return_tensors="pt").to(self._device)
        with self._torch.no_grad():
            wave = self._model(**inputs).waveform[0]
        return _normalize(wave.cpu().numpy(), self._rate)


# Pocket TTS sometimes ends a short sentence after its first word (a premature
# end-of-speech that depends on the seed). A clip shorter than this share of the
# expected length is re-rolled with a new seed, at most POCKET_MAX_REROLLS times,
# and the longest take is kept.
POCKET_TRUNCATED = 0.45
POCKET_MAX_REROLLS = 2
POCKET_S_PER_PHONEME = 0.06
POCKET_S_PER_CHAR = 0.055
# Pocket's English voices often open with ~1 s of silence (Piper: ~0.05 s). Silence
# beyond these margins is cut, so it neither delays the reply nor hides a truncation.
POCKET_LEAD_S = 0.1
POCKET_TAIL_S = 0.2
_NOT_PHONEMES = frozenset(" ,.?!…;:-—–\"'ˈˌːˑ")


def expected_duration(text: str, ipa: str | None) -> float:
    """Rough spoken length: phonemes x 0.06 s from the IPA, else characters x 0.055 s."""
    if ipa:
        return POCKET_S_PER_PHONEME * sum(1 for c in ipa if c not in _NOT_PHONEMES)
    return POCKET_S_PER_CHAR * len(text.strip())


def trim_silence(audio: np.ndarray, sr: int, lead_s: float = POCKET_LEAD_S,
                 tail_s: float = POCKET_TAIL_S) -> np.ndarray:
    """Audio with at most lead_s of silence before the speech and tail_s after it."""
    segs = voiced_segments(audio, sr)
    if not segs:
        return audio
    start = max(0, int((segs[0][0] - lead_s) * sr))
    end = min(len(audio), int(round((segs[-1][1] + tail_s) * sr)))
    return audio[start:end]


class _EspeakIPA:
    """English text -> espeak IPA through piper's bundled espeak-ng (for the viseme
    estimate only); None, with one warning, if piper's espeak is not available."""

    def __init__(self):
        self._phonemizer = None
        self._failed = False
        self._lock = threading.Lock()

    def __call__(self, text: str) -> str | None:
        if self._failed:
            return None
        try:
            with self._lock:
                if self._phonemizer is None:
                    from piper.phonemize_espeak import EspeakPhonemizer  # lazy

                    self._phonemizer = EspeakPhonemizer()
                sentences = self._phonemizer.phonemize("en-us", text)
        except Exception as e:  # noqa: BLE001 - visemes fall back to loudness
            log.warning("no espeak for English visemes (%s: %s); the mouth follows "
                        "loudness", type(e).__name__, e)
            self._failed = True
            return None
        return " ".join("".join(s) for s in sentences)


class PocketTTS:
    """Pocket TTS (ONNX, CPU): one voice for both languages. English goes in as text,
    Hebrew as RenikudPlus IPA through the model's IPA adapter. It gives no phoneme
    timings, so the viseme track is estimated from the IPA and the audio's loudness
    (English IPA from espeak when available, else no track: the mouth follows loudness).
    """

    def __init__(self, model_path: str, voice: str = "javert", voice_file: str | None = None,
                 threads: int = 2, g2p=None, en_ipa="espeak", seed: int = 0):
        from pocket_tts_onnx import PocketTTS as Model  # lazy: heavy optional dep

        self._model = Model(model_path, providers=["CPUExecutionProvider"],
                            num_threads=threads)
        if voice_file:
            # A conditioning cloned offline (tools/clone_voice.py); the encoder never runs here.
            self._voice = np.load(voice_file)
        else:
            if voice not in self._model.voices():
                raise ValueError(f"unknown Pocket voice {voice!r}; built in: "
                                 f"{', '.join(self._model.voices())}")
            self._voice = voice
        self._g2p = g2p
        self._en_ipa = _EspeakIPA() if en_ipa == "espeak" else en_ipa
        self._seed = seed
        self._rate = int(self._model.sample_rate)
        self._lock = threading.Lock()

    @property
    def sample_rate(self) -> int:
        return self._rate

    def for_lang(self, lang: str) -> "_PocketLang":
        """A one-language Synth (synth(text), synth_visemes(text)) for TTSRouter."""
        return _PocketLang(self, lang)

    def _generate(self, text: str, phonemes: bool, expected_s: float) -> np.ndarray:
        best = np.zeros(0, np.float32)
        for attempt in range(POCKET_MAX_REROLLS + 1):
            with self._lock:
                raw, rate = self._model.create(text, voice=self._voice, phonemes=phonemes,
                                               seed=self._seed + attempt)
            audio = trim_silence(_normalize(raw, rate)[0], self._rate)
            if len(audio) > len(best):
                best = audio
            if len(audio) / self._rate >= POCKET_TRUNCATED * expected_s:
                if attempt:
                    log.info("pocket: re-roll %d gave a full clip for %r", attempt, text)
                return audio
            log.info("pocket: %.2f s clip, %.2f s expected: re-rolling %r",
                     len(audio) / self._rate, expected_s, text)
        log.warning("pocket: still short after %d re-rolls; keeping the longest "
                    "(%.2f s of %.2f s expected) for %r", POCKET_MAX_REROLLS,
                    len(best) / self._rate, expected_s, text)
        return best

    def synth_visemes(self, text: str, lang: str = "en"
                      ) -> tuple[np.ndarray, int, VisemeTrack | None]:
        text = " ".join(text.split())
        if lang == "he":
            if self._g2p is None:
                raise ValueError("PocketTTS: Hebrew needs a G2P (g2p=RenikudG2P(...))")
            ipa = self._g2p.ipa(text) if text else ""
            model_input, phonemes = ipa, True
        else:
            ipa = self._en_ipa(text) if text and self._en_ipa is not None else None
            model_input, phonemes = text, False
        if not model_input.strip(" ,.?!"):
            return np.zeros(0, np.float32), self._rate, None
        audio = self._generate(model_input, phonemes, expected_duration(text, ipa))
        track = estimate_track(ipa, audio, self._rate) if ipa else None
        return audio, self._rate, track

    def synth(self, text: str, lang: str = "en") -> tuple[np.ndarray, int]:
        audio, rate, _ = self.synth_visemes(text, lang)
        return audio, rate


class _PocketLang:
    def __init__(self, tts: PocketTTS, lang: str):
        self._tts = tts
        self._lang = lang

    @property
    def sample_rate(self) -> int:
        return self._tts.sample_rate

    def synth(self, text: str) -> tuple[np.ndarray, int]:
        return self._tts.synth(text, self._lang)

    def synth_visemes(self, text: str) -> tuple[np.ndarray, int, VisemeTrack | None]:
        return self._tts.synth_visemes(text, self._lang)


# the small room: (delay ms, gain) echoes of the voice, gains scaled by `room`
_ROOM_ECHOES = ((37, 0.28), (53, 0.22), (71, 0.16), (97, 0.11))
_EFFECT_PEAK = 0.9


def voice_effect(audio: np.ndarray, sr: int, track: VisemeTrack | None,
                 speed: float = 1.0, room: float = 0.0
                 ) -> tuple[np.ndarray, VisemeTrack | None]:
    """The "villain" voice: resample so playback runs at `speed` (deeper and slower
    below 1), add a few decaying room echoes, peak-normalize. The viseme track is
    stretched by 1/speed to stay in sync. Off (speed 1, room 0) returns the inputs."""
    if speed == 1.0 and room == 0.0:
        return audio, track
    out = audio
    if speed != 1.0:
        n = int(len(audio) / speed)
        out = np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)),
                        audio).astype(np.float32) if len(audio) else audio[:0]
        if track is not None:
            track = VisemeTrack(track.times / speed, list(track.shapes), track.end / speed)
    if room > 0.0:
        dry = out
        out = dry.copy()
        for ms, gain in _ROOM_ECHOES:
            d = int(sr * ms / 1000)
            if 0 < d < len(dry):
                out[d:] += room * gain * dry[:-d]
    peak = float(np.abs(out).max()) if len(out) else 0.0
    if peak > 1e-6:
        out = (out * (_EFFECT_PEAK / peak)).astype(np.float32, copy=False)
    return out, track


class TTSRouter:
    def __init__(self, en: Synth, he: Synth, voice_speed: float = 1.0,
                 voice_room: float = 0.0):
        self._en = en
        self._he = he
        self._voice_speed = voice_speed  # voice_effect, on every line in both languages
        self._voice_room = voice_room

    def _effect(self, audio, rate, track):
        return voice_effect(audio, rate, track, self._voice_speed, self._voice_room)

    def synth(self, text: str, lang: str) -> tuple[np.ndarray, int]:
        engine = self._he if lang == "he" else self._en
        audio, rate = _normalize(*engine.synth(text))
        return self._effect(audio, rate, None)[0], rate

    def synth_visemes(self, text: str, lang: str) -> tuple[np.ndarray, int, VisemeTrack | None]:
        """Audio plus a mouth-shape track; the track is None for engines that can't
        align phonemes (MMS), and the face then shapes the mouth from loudness."""
        engine = self._he if lang == "he" else self._en
        if not hasattr(engine, "synth_visemes"):
            audio, rate = engine.synth(text)
            track = None
        else:
            audio, rate, track = engine.synth_visemes(text)
        audio, rate = _normalize(audio, rate)
        audio, track = self._effect(audio, rate, track)
        return audio, rate, track

    def warmup(self) -> None:
        self.synth("hello", "en")
        self.synth("שלום", "he")
