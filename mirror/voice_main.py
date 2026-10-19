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

from mirror.audio import CHUNK, Mic, MicStalled, Player, find_device
from mirror.bus import FaceBus
from mirror.config import Config, TtsConfig, load_config, resolve_repo_path
from mirror.conversation import Controller
from mirror.fillers import FillerBank
from mirror.lights import LogLights, WledSerial
from mirror.llm import LANG_SLOTS, LlamaServerLLM, LLMError
from mirror.persona import FILLERS, load_world
from mirror.states import State
from mirror.stt import WhisperServerSTT
from mirror.vad import Segmenter, SileroVad

log = logging.getLogger("mirror.voice")

TICK_S = 0.1
# --wav: after the file, feed enough silence for the segmenter to end the utterance
WAV_TAIL_MARGIN_S = 0.3
SEGMENTS_MAX = 3  # queued utterances; under constant noise the oldest are dropped
# Waits between consecutive failed mic attempts (~31 s in all); the attempt after
# the last wait is the final one. CoreAudio can stop a whole process for several
# seconds under heavy swap, so ride that out before letting systemd restart us.
MIC_RETRY_DELAYS_S = (1, 2, 4, 8, 16)
_ANSWERING = (State.THINKING, State.SPEAKING)


class _Shutdown(BaseException):
    """Raised from the signal handler; BaseException so the controller won't swallow it."""


def _signal_handler(stop: threading.Event):
    """SIGINT/SIGTERM handler: the first signal raises _Shutdown; any later one (an
    impatient second Ctrl-C during teardown) is ignored so the cleanup completes."""

    def on_signal(signum, _frame):
        if stop.is_set():
            return
        stop.set()
        raise _Shutdown()

    return on_signal


def wav_tail_s(segmenter: Segmenter) -> float:
    """Silence fed after a --wav file: the segmenter's end-of-utterance silence plus
    a margin, so the last utterance is always cut."""
    return segmenter.min_silence_s + WAV_TAIL_MARGIN_S


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
    """Player wrapper that logs end-of-speech -> start of each clip of an answer
    (the filler, then each reply sentence)."""

    def __init__(self, player):
        self._player = player
        self._speech_end = None  # time.monotonic() of the end of the user's speech
        self._clip = 0

    @property
    def speech_end(self):
        return self._speech_end

    @speech_end.setter
    def speech_end(self, t) -> None:
        self._speech_end, self._clip = t, 0

    def play(self, samples, sr, on_level) -> None:
        if self._speech_end is not None:
            self._clip += 1
            log.info("latency: end of speech -> audio clip %d %.2f s", self._clip,
                     time.monotonic() - self._speech_end)
        self._player.play(samples, sr, on_level)


class WavSource:
    """Mic stand-in: a WAV file (mono-mixed, resampled to sr) then tail_s of silence, delivered in CHUNK-sized reads paced in real time; silence after that."""

    def __init__(self, path: str, sr: int = 16000, tail_s: float = 1.0):
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
            [audio, np.zeros(int(round(tail_s * sr)), np.float32)])
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


def open_output(output, sr: int) -> None:
    """Open the speaker now: the first open of a cold output device can take seconds.
    A failure (missing or busy speaker) is not fatal; the first reply retries it."""
    try:
        output.open(sr)
    except Exception as e:
        log.warning("could not open the audio output yet: %s", e)


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
            _put_dropping_oldest(segments, seg)


def _put_dropping_oldest(q: "queue.Queue", item) -> None:
    """Queue item; if q is full, drop its oldest entry first (never block the mic)."""
    while True:
        try:
            q.put_nowait(item)
            return
        except queue.Full:
            try:
                q.get_nowait()
                log.info("segment backlog: dropped the oldest segment")
            except queue.Empty:
                pass


def audio_worker(open_source, segmenter: Segmenter, vad: SileroVad, gate: AnswerGate,
                 segments: "queue.Queue", stop: threading.Event, failed: threading.Event,
                 delays=MIC_RETRY_DELAYS_S) -> None:
    """Run audio_loop on open_source(), reopening it when the mic stalls or fails
    to open, waiting delays[i] before retry i+1 (interruptible by `stop`). After
    len(delays)+1 failures in a row (no audio in between) set `failed`, so run()
    exits non-zero and systemd restarts the service."""
    failures = 0
    max_failures = len(delays) + 1
    while not stop.is_set():
        src, opened = None, False
        try:
            src = open_source()
            with src:
                opened = True
                segmenter.reset()  # drop any half-cut utterance from before
                vad.reset()
                audio_loop(src, segmenter, vad, gate, segments, stop)
            return
        except MicStalled as e:
            err = e
        except Exception as e:  # noqa: BLE001 - logged; the main thread sees `failed`
            if opened:  # not a device problem: a bug in the VAD/segmenter path
                log.exception("audio thread failed")
                failed.set()
                return
            err = e  # the device would not open (gone, busy): retry like a stall
        failures = 1 if getattr(src, "reads", 0) else failures + 1
        if failures >= max_failures:
            log.error("mic failed %d times in a row (%s); giving up", failures, err)
            failed.set()
            return
        delay = delays[failures - 1]
        log.warning("mic failed (attempt %d/%d): %s; waiting %g s before reopening",
                    failures, max_failures, err, delay)
        stop.wait(delay)


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


def _lights_to_idle(lights) -> None:
    """On the way out: leave the frame on the idle preset, not on "speaking"."""
    try:
        lights.set_state(State.IDLE)
    except Exception:
        log.exception("shutdown lights push failed")


def build_hebrew_tts(tts_cfg: TtsConfig):
    """The Hebrew engine named by tts.he_engine (with the G2P named by tts.he_g2p)."""
    from mirror import tts

    if tts_cfg.he_engine == "phonikud":
        renikud_error = None
        if tts_cfg.he_g2p == "renikud":
            g2p = tts.RenikudG2P(tts_cfg.renikud_model)
            try:
                g2p.phonemize("שלום")  # load it now: a broken install falls back, not dies
            except Exception as e:
                log.warning("RenikudPlus G2P (%s) failed to load (%s: %s); "
                            "falling back to Phonikud", tts_cfg.renikud_model,
                            type(e).__name__, e)
                g2p, renikud_error = None, e
        elif tts_cfg.he_g2p == "phonikud":
            g2p = None  # the voice builds Phonikud from phonikud_model
        else:
            raise ValueError(f"unknown tts.he_g2p {tts_cfg.he_g2p!r} (renikud | phonikud)")
        try:
            return tts.PhonikudHebrewTTS(tts_cfg.phonikud_model, tts_cfg.phonikud_voice_he,
                                         tts_cfg.phonikud_voice_config_he, g2p=g2p)
        except Exception as e:
            if renikud_error is None:
                raise
            raise RuntimeError(
                f"no Hebrew G2P: RenikudPlus failed ({type(renikud_error).__name__}: "
                f"{renikud_error}) and the Phonikud fallback failed too "
                f"({type(e).__name__}: {e})") from e
    if tts_cfg.he_engine == "mms":
        return tts.MmsHebrewTTS(tts_cfg.mms_model_he)
    raise ValueError(f"unknown tts.he_engine {tts_cfg.he_engine!r} (phonikud | mms)")


FILLER_BUDGET_S = 2.5  # persona.py's filler length; Pocket voices can run past it


def build_tts(tts_cfg: TtsConfig):
    """(router, sample rate, engine) for tts.engine. "pocket" loads Pocket TTS for both
    languages (Hebrew through RenikudPlus IPA) and warms it up; if loading or the
    warmup fails it logs a WARNING and falls back to the piper stack."""
    from mirror import tts

    if tts_cfg.engine not in ("piper", "pocket"):
        raise ValueError(f"unknown tts.engine {tts_cfg.engine!r} (piper | pocket)")
    if tts_cfg.engine == "pocket":
        try:
            g2p = tts.RenikudG2P(tts_cfg.renikud_model)
            g2p.phonemize("שלום")  # load it now: Pocket's Hebrew needs it
            pocket = tts.PocketTTS(tts_cfg.pocket_model, voice=tts_cfg.pocket_voice,
                                   voice_file=tts_cfg.pocket_voice_file,
                                   threads=tts_cfg.pocket_threads, g2p=g2p)
            router = tts.TTSRouter(pocket.for_lang("en"), pocket.for_lang("he"))
            # one short line per language: a model that loads but can't speak falls back too
            router.warmup()
            log.info("TTS: pocket (voice %s, %d threads)",
                     tts_cfg.pocket_voice_file or tts_cfg.pocket_voice, tts_cfg.pocket_threads)
            return router, pocket.sample_rate, "pocket"
        except Exception as e:
            log.warning("Pocket TTS (%s) failed to load (%s: %s); falling back to piper",
                        tts_cfg.pocket_model, type(e).__name__, e)
    log.info("loading TTS voices (Hebrew: %s, G2P %s)", tts_cfg.he_engine, tts_cfg.he_g2p)
    en_tts = tts.PiperTTS(tts_cfg.piper_voice_en)
    return tts.TTSRouter(en_tts, build_hebrew_tts(tts_cfg)), en_tts.sample_rate, "piper"


def load_tts(tts_cfg: TtsConfig):
    """build_tts, warmed up once. Pocket is already warmed inside build_tts's try
    block, where a warmup failure triggers the piper fallback, so only the piper
    stack is warmed here."""
    router, rate, engine = build_tts(tts_cfg)
    if engine != "pocket":
        router.warmup()
    return router, rate, engine


def warm_llm(llm) -> list:
    """Load each language's system prompt into its llama-server slot, so the first
    question is not slowed by a cold prompt. A failure is logged, not fatal; returns
    the languages that failed (the controller re-warms them when idle)."""
    failed = []
    for lang in LANG_SLOTS:
        try:
            llm.warm(lang)
        except LLMError as e:
            log.warning("llm warmup (%s) failed: %s", lang, e)
            failed.append(lang)
    return failed


def run(cfg: Config, wav_path=None, config_path=None) -> int:
    models_dir = os.path.abspath(cfg.models_dir)
    # MMS (transformers) and Phonikud's tokenizer load through the Hugging Face hub;
    # they must come from the local cache, offline.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HOME"] = os.path.join(models_dir, "hf")

    stop = threading.Event()
    face = raw_lights = output = None
    controller = None
    audio_thread = None
    try:
        # inside the try: a signal during device setup is a clean shutdown too
        on_signal = _signal_handler(stop)
        signal.signal(signal.SIGTERM, on_signal)
        signal.signal(signal.SIGINT, on_signal)

        face = FaceBus.sender(cfg.face.bus_port)
        raw_lights = (WledSerial(cfg.lights.serial_port, cfg.lights.baud)
                      if cfg.lights.kind == "wled" else LogLights())
        output = Player(find_device(cfg.audio.output_device, "output",
                                    cfg.audio.prefer_alsa_plug))
        _push_idle(raw_lights, face)
        gate = AnswerGate(raw_lights)

        vad = SileroVad(os.path.join(models_dir, "silero_vad.onnx"))
        segmenter = Segmenter(vad, frame_samples=CHUNK, sample_rate=cfg.audio.sample_rate)
        stt = WhisperServerSTT(cfg.servers.stt_url, cfg.servers.timeout_s)
        world = load_world(resolve_repo_path(cfg.conversation.world_file, config_path))
        llm = LlamaServerLLM(cfg.servers.llm_url, cfg.servers.timeout_s,
                             cfg.conversation.max_reply_sentences,
                             cfg.conversation.max_reply_chars, world=world)
        tts, tts_rate, engine = load_tts(cfg.tts)  # after the env is set
        fillers = FillerBank(tts, FILLERS,
                             max_clip_s=FILLER_BUDGET_S if engine == "pocket" else None)
        t0 = time.monotonic()
        fillers.prepare()
        log.info("filler lines ready in %.1f s", time.monotonic() - t0)
        unwarmed = warm_llm(llm)
        open_output(output, tts_rate)
        player = LatencyPlayer(output)
        controller = Controller(stt, llm, tts, player, gate, face, cfg.conversation,
                                fillers=fillers,
                                first_sentence_timeout_s=cfg.servers.timeout_s)
        for lang in unwarmed:
            controller.mark_unwarmed(lang)

        if wav_path:
            source = WavSource(wav_path, cfg.audio.sample_rate, wav_tail_s(segmenter))
            open_source = lambda: contextlib.nullcontext(source)  # noqa: E731
        else:
            source = None
            mic_device = find_device(cfg.audio.input_device, "input",
                                     cfg.audio.prefer_alsa_plug)
            open_source = lambda: Mic(mic_device, cfg.audio.sample_rate)  # noqa: E731

        segments: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=SEGMENTS_MAX)
        failed = threading.Event()
        audio_thread = threading.Thread(
            target=audio_worker, name="audio", daemon=True,
            args=(open_source, segmenter, vad, gate, segments, stop, failed))
        audio_thread.start()
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
                else:  # the segment ends with the segmenter's hang-over silence
                    player.speech_end = time.monotonic() - segmenter.min_silence_s
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
        if audio_thread is not None:
            audio_thread.join(timeout=1.0)  # closes the mic within one chunk
        if controller is not None:
            with contextlib.suppress(Exception):
                controller.close()  # waits (bounded) for a background re-warm
        if raw_lights is not None:
            _lights_to_idle(raw_lights)
        if face is not None:
            face.close()
        if output is not None:
            with contextlib.suppress(Exception):
                output.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Mirror, Mirror voice loop")
    ap.add_argument("--config", required=True)
    ap.add_argument("--wav", help="feed this WAV through the pipeline instead of the mic, "
                                  "then exit once the reply has been spoken")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    raise SystemExit(run(load_config(args.config), args.wav, args.config))


if __name__ == "__main__":
    main()
