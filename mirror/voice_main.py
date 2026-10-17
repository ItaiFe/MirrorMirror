"""mirror-voice: mic -> VAD -> conversation controller -> speaker, lights and face.

The audio thread reads the mic (or, with --wav, a file paced in real time),
cuts utterances with Silero VAD + Segmenter and queues them. The main thread
feeds queued segments to the controller and ticks it every 100 ms. While the
mirror is thinking or speaking the audio thread discards what it hears; once
the answer is over, the VAD state is reset and the segment queue drained, so
the mirror never reacts to stale audio or to its own voice.
"""
import argparse
import contextlib
import logging
import os
import queue
import signal
import threading
import time
import wave

import numpy as np

from mirror.audio import CHUNK, Mic, Player, find_device
from mirror.bus import FaceBus
from mirror.config import Config, TtsConfig, load_config
from mirror.conversation import Controller
from mirror.lights import LogLights, WledSerial
from mirror.llm import LlamaServerLLM
from mirror.states import State
from mirror.stt import WhisperServerSTT
from mirror.vad import Segmenter, SileroVad

log = logging.getLogger("mirror.voice")

TICK_S = 0.1
WAV_TAIL_SILENCE_S = 1.0
_ANSWERING = (State.THINKING, State.SPEAKING)


class _Shutdown(BaseException):
    """Raised from the signal handler; BaseException so the controller won't swallow it."""


class AnswerGate:
    """Lights wrapper that tells the audio side when the mirror is answering."""

    def __init__(self, lights):
        self._lights = lights
        self.answering = threading.Event()  # set while THINKING/SPEAKING
        self.answered = threading.Event()   # set once an answer started; main clears it

    def set_state(self, state: State) -> None:
        if state in _ANSWERING:
            self.answering.set()
            self.answered.set()
        else:
            self.answering.clear()
        self._lights.set_state(state)


class LatencyPlayer:
    """Player wrapper that logs end-of-speech -> first-audio once per utterance."""

    def __init__(self, player):
        self._player = player
        self.speech_end = None  # time.monotonic() of the end of the user's speech

    def play(self, samples, sr, on_level) -> None:
        if self.speech_end is not None:
            log.info("latency: end of speech -> first audio %.2f s",
                     time.monotonic() - self.speech_end)
            self.speech_end = None
        self._player.play(samples, sr, on_level)


class WavSource:
    """Mic stand-in: a WAV file (mono-mixed, resampled to sr) then WAV_TAIL_SILENCE_S
    of silence, delivered in CHUNK-sized reads paced in real time; silence after that."""

    def __init__(self, path: str, sr: int = 16000):
        with wave.open(path, "rb") as w:
            n_ch, width, rate = w.getnchannels(), w.getsampwidth(), w.getframerate()
            raw = w.readframes(w.getnframes())
        if width != 2:
            raise ValueError(f"{path}: only 16-bit PCM WAV is supported")
        audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        audio = audio.reshape(-1, n_ch).mean(axis=1)
        if rate != sr:
            t_out = np.arange(int(len(audio) * sr / rate)) / sr
            audio = np.interp(t_out, np.arange(len(audio)) / rate, audio).astype(np.float32)
        self._speech_samples = len(audio)
        self._audio = np.concatenate(
            [audio, np.zeros(int(WAV_TAIL_SILENCE_S * sr), np.float32)])
        self._sr = sr
        self._pos = 0
        self._t0 = None
        self.speech_end = None  # monotonic time the WAV's audio finished "playing in"
        self.done = threading.Event()  # set once WAV + tail silence have been read

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def read(self) -> np.ndarray:
        if self._t0 is None:
            self._t0 = time.monotonic()
        delay = self._t0 + self._pos / self._sr - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        if self._pos >= len(self._audio):
            self.done.set()
        chunk = self._audio[self._pos:self._pos + CHUNK]
        if len(chunk) < CHUNK:
            chunk = np.concatenate([chunk, np.zeros(CHUNK - len(chunk), np.float32)])
        self._pos += CHUNK
        if self.speech_end is None and self._pos >= self._speech_samples:
            self.speech_end = self._t0 + self._speech_samples / self._sr
        return chunk


def audio_loop(source, segmenter: Segmenter, vad: SileroVad, gate: AnswerGate,
               segments: "queue.Queue", stop: threading.Event) -> None:
    deaf = False
    while not stop.is_set():
        chunk = source.read()
        if gate.answering.is_set():
            deaf = True  # discard everything heard while thinking/speaking
            continue
        if deaf:
            segmenter.reset()
            vad.reset()
            deaf = False
        for seg in segmenter.feed(chunk):
            segments.put(seg)


def _drain(q: "queue.Queue") -> int:
    n = 0
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            return n
        n += 1


def _push_idle(lights, face) -> None:
    for sink, args in ((lights.set_state, (State.IDLE,)), (face.send, (State.IDLE, 0.0))):
        try:
            sink(*args)
        except Exception:
            log.exception("startup idle push failed")


def build_hebrew_tts(tts_cfg: TtsConfig):
    """The Hebrew engine named by tts.he_engine."""
    from mirror import tts

    if tts_cfg.he_engine == "phonikud":
        return tts.PhonikudHebrewTTS(tts_cfg.phonikud_model, tts_cfg.phonikud_voice_he,
                                     tts_cfg.phonikud_voice_config_he)
    if tts_cfg.he_engine == "mms":
        return tts.MmsHebrewTTS(tts_cfg.mms_model_he)
    raise ValueError(f"unknown tts.he_engine {tts_cfg.he_engine!r} (phonikud | mms)")


def run(cfg: Config, wav_path=None) -> int:
    models_dir = os.path.abspath(cfg.models_dir)
    # MMS (transformers) and Phonikud's tokenizer load through the Hugging Face hub;
    # they must come from the local cache, offline.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HOME"] = os.path.join(models_dir, "hf")
    from mirror.tts import PiperTTS, TTSRouter  # after the env is set

    stop = threading.Event()

    def on_signal(signum, _frame):
        stop.set()
        raise _Shutdown()

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    face = FaceBus.sender(cfg.face.bus_port)
    raw_lights = (WledSerial(cfg.lights.serial_port, cfg.lights.baud)
                  if cfg.lights.kind == "wled" else LogLights())
    output = Player(find_device(cfg.audio.output_device, "output"))
    try:
        _push_idle(raw_lights, face)
        gate = AnswerGate(raw_lights)

        vad = SileroVad(os.path.join(models_dir, "silero_vad.onnx"))
        segmenter = Segmenter(vad, frame_samples=CHUNK, sample_rate=cfg.audio.sample_rate)
        stt = WhisperServerSTT(cfg.servers.stt_url, cfg.servers.timeout_s)
        llm = LlamaServerLLM(cfg.servers.llm_url, cfg.servers.timeout_s,
                             cfg.conversation.max_reply_sentences,
                             cfg.conversation.max_reply_chars)
        log.info("loading TTS voices (Hebrew: %s)", cfg.tts.he_engine)
        en_tts = PiperTTS(cfg.tts.piper_voice_en)
        tts = TTSRouter(en_tts, build_hebrew_tts(cfg.tts))
        tts.warmup()
        # Open the speaker now: the first open of a cold output device can take seconds.
        output.open(en_tts.sample_rate)
        player = LatencyPlayer(output)
        controller = Controller(stt, llm, tts, player, gate, face, cfg.conversation)

        if wav_path:
            source = WavSource(wav_path, cfg.audio.sample_rate)
            source_cm = contextlib.nullcontext(source)
        else:
            source = None
            source_cm = Mic(find_device(cfg.audio.input_device, "input"),
                            cfg.audio.sample_rate)

        segments: "queue.Queue[np.ndarray]" = queue.Queue()
        failed = threading.Event()
        with source_cm as src:
            def audio_thread():
                try:
                    audio_loop(src, segmenter, vad, gate, segments, stop)
                except Exception:
                    log.exception("audio thread failed")
                    failed.set()

            threading.Thread(target=audio_thread, name="audio", daemon=True).start()
            log.info("ready")
            while not stop.is_set() and not failed.is_set():
                try:
                    seg = segments.get(timeout=TICK_S)
                except queue.Empty:
                    seg = None
                if seg is not None:
                    log.info("segment: %.1f s", len(seg) / cfg.audio.sample_rate)
                    if source is not None:
                        player.speech_end = source.speech_end
                    else:  # the segment ends with ~0.7 s of VAD hang-over silence
                        player.speech_end = time.monotonic() - 0.7
                    controller.on_segment(seg)
                    player.speech_end = None
                    if gate.answered.is_set():
                        gate.answered.clear()
                        dropped = _drain(segments)
                        if dropped:
                            log.info("dropped %d stale segment(s)", dropped)
                controller.tick()
                if source is not None and source.done.is_set() and segments.empty():
                    log.info("wav finished; state %s", controller.state.value)
                    break
        return 1 if failed.is_set() else 0
    except _Shutdown:
        log.info("shutting down")
        return 0
    finally:
        stop.set()
        face.close()
        with contextlib.suppress(Exception):
            output.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Mirror, Mirror voice loop")
    ap.add_argument("--config", required=True)
    ap.add_argument("--wav", help="feed this WAV through the pipeline instead of the mic, "
                                  "then exit once the reply has been spoken")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    raise SystemExit(run(load_config(args.config), args.wav))


if __name__ == "__main__":
    main()
