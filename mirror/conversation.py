"""Conversation controller: the IDLE/LISTENING/THINKING/SPEAKING state machine.

An answer is a small pipeline: a worker streams the LLM reply sentence by sentence,
a second worker synthesizes each sentence while the one before it plays, and the
calling thread plays a filler line and then each sentence. Both workers are stopped
and joined before _answer returns, whatever way it returns.
"""
import logging
import queue
import threading
import time
from typing import Callable

import numpy as np

from mirror.config import ConversationConfig
from mirror.llm import LLMError
from mirror.persona import FALLBACK
from mirror.states import State
from mirror.stt import STTError
from mirror.visemes import rms_shape
from mirror.wake import is_hallucination, match_wake

log = logging.getLogger("mirror.conversation")

_HISTORY_MAX = 6
HOP_S = 0.02   # the player reports one mouth level per 20 ms hop
LEAD_S = 0.05  # show each mouth shape slightly before its sound
_POLL_S = 0.1  # how often a blocked worker checks whether it was cancelled
_END = object()  # end of the sentence / clip stream


class Controller:
    def __init__(self, stt, llm, tts, player, lights, face, cfg: ConversationConfig,
                 clock: Callable[[], float] = time.monotonic, fillers=None,
                 first_sentence_timeout_s: float = 20.0):
        self._stt = stt
        self._llm = llm
        self._tts = tts
        self._player = player
        self._lights = lights
        self._face = face
        self._cfg = cfg
        self._clock = clock
        self._fillers = fillers  # FillerBank or None
        self._first_sentence_timeout_s = first_sentence_timeout_s
        self._state = State.IDLE
        self._deadline = 0.0
        self._history: list[dict] = []
        self._busy = False
        self._lock = threading.Lock()

    @property
    def state(self) -> State:
        return self._state

    def on_segment(self, samples: np.ndarray) -> None:
        with self._lock:
            if self._busy:
                log.debug("dropping segment while %s", self._state.value)
                return
            self._busy = True
        try:
            self._handle(samples)
        except Exception:
            log.exception("conversation step failed")
            self._go_idle()
        finally:
            self._busy = False

    def tick(self) -> None:
        with self._lock:
            if self._busy or self._state != State.LISTENING:
                return
            if self._clock() >= self._deadline:
                log.info("listening timed out")
                self._go_idle()

    def _handle(self, samples: np.ndarray) -> None:
        try:
            transcript = self._stt.transcribe(samples)
        except STTError as e:
            log.warning("stt failed: %s", e)
            return  # stay in the resting state we were in
        log.info("heard (%s, in %s): %r", transcript.lang, self._state.value, transcript.text)
        if is_hallucination(transcript.text):
            return
        wake = match_wake(transcript.text)
        if self._state == State.IDLE:
            if wake is None:
                return
            text = wake.remainder
        else:  # LISTENING
            text = transcript.text if wake is None else wake.remainder
        if text:
            self._answer(text, transcript.lang)
        else:
            self._listen(self._cfg.listen_timeout_s)

    def _answer(self, text: str, lang: str) -> None:
        self._set_state(State.THINKING)
        cancel = threading.Event()
        sentences: "queue.Queue" = queue.Queue()
        clips: "queue.Queue" = queue.Queue(maxsize=1)  # synthesize one sentence ahead
        workers = [
            threading.Thread(target=self._stream_worker, name="llm-stream", daemon=True,
                             args=(list(self._history), text, lang, sentences, cancel)),
            threading.Thread(target=self._synth_worker, name="tts-ahead", daemon=True,
                             args=(lang, sentences, clips, cancel)),
        ]
        for w in workers:
            w.start()
        try:
            try:
                reply = self._play_reply(lang, clips)
            except LLMError as e:  # nothing to say: no sentence, or not in time
                cancel.set()
                log.warning("llm failed: %s", e)
                self._speak(FALLBACK[lang], lang)
                self._go_idle()
                return
        finally:
            # Also on BaseException (shutdown): no worker outlives the answer. The
            # stream worker stops at its next streamed token, or at the latest when
            # its request's read timeout (servers.timeout_s) expires.
            cancel.set()
            for w in workers:
                w.join()
        log.info("reply (%s): %r", lang, reply)
        self._history.extend([{"role": "user", "content": text},
                              {"role": "assistant", "content": reply}])
        self._history = self._history[-_HISTORY_MAX:]
        self._listen(self._cfg.followup_s)

    def _play_reply(self, lang: str, clips: "queue.Queue") -> str:
        """Play the filler, then each synthesized sentence as it is ready; return the
        spoken reply. Raises LLMError if no sentence comes (in time)."""
        deadline = time.monotonic() + self._first_sentence_timeout_s
        item = None
        if self._fillers is not None and self._cfg.filler_enabled:
            if self._cfg.filler_min_wait_s > 0:
                item = _get(clips, self._cfg.filler_min_wait_s)
            if item is None:
                clip = self._fillers.pick(lang)
                if clip is not None:
                    log.info("filler (%s)", lang)
                    self._play_clip(*clip)
                    if clips.empty():
                        self._set_state(State.THINKING)
        if item is None:
            item = _get(clips, max(0.0, deadline - time.monotonic()))
            if item is None:
                raise LLMError(f"no reply sentence within {self._first_sentence_timeout_s} s")
        spoken: list[str] = []
        while item is not _END:
            if isinstance(item, BaseException):
                if spoken and isinstance(item, LLMError):
                    log.warning("llm stream failed mid-reply: %s", item)
                    break
                raise item
            sentence, samples, sr, track = item
            log.info("reply sentence %d (%s): %r", len(spoken) + 1, lang, sentence)
            self._play_clip(samples, sr, track)
            spoken.append(sentence)
            item = clips.get()
        if not spoken:
            raise LLMError("llama-server returned an empty reply")
        return " ".join(spoken)

    def _stream_worker(self, history: list[dict], text: str, lang: str,
                       out: "queue.Queue", cancel: threading.Event) -> None:
        """LLM sentences -> out; then an exception (if the stream failed) and _END."""
        stream = None
        try:
            stream = self._llm.reply_stream(history, text, lang, cancel=cancel)
            for sentence in stream:
                if cancel.is_set():
                    break
                out.put(sentence)
        except Exception as e:  # noqa: BLE001 - handed to the controller thread
            out.put(e)
        finally:
            close = getattr(stream, "close", None)
            if close is not None:
                try:
                    close()  # closes the HTTP response
                except Exception as e:  # noqa: BLE001 - best effort
                    log.warning("closing the llm stream failed: %s", e)
            out.put(_END)

    def _synth_worker(self, lang: str, sentences: "queue.Queue", clips: "queue.Queue",
                      cancel: threading.Event) -> None:
        """Sentences -> (sentence, samples, sr, track) clips; forwards the end marker
        and exceptions (its own included). Stops when cancelled."""
        try:
            while not cancel.is_set():
                try:
                    item = sentences.get(timeout=_POLL_S)
                except queue.Empty:
                    continue
                if item is _END or isinstance(item, BaseException):
                    _put(clips, item, cancel)
                    return
                samples, sr, track = self._tts.synth_visemes(item, lang)
                if not _put(clips, (item, samples, sr, track), cancel):
                    return
        except Exception as e:  # noqa: BLE001 - handed to the controller thread
            _put(clips, e, cancel)

    def _speak(self, text: str, lang: str) -> None:
        samples, sr, track = self._tts.synth_visemes(text, lang)
        self._play_clip(samples, sr, track)

    def _play_clip(self, samples: np.ndarray, sr: int, track) -> None:
        self._set_state(State.SPEAKING)
        hop = 0

        def send_mouth(level: float) -> None:
            nonlocal hop
            t = hop * HOP_S + LEAD_S
            hop += 1
            shape = track.at(t) if track is not None else rms_shape(level)
            _guarded(self._face.send, State.SPEAKING, level, shape)

        self._player.play(samples, sr, on_level=send_mouth)

    def _listen(self, timeout_s: float) -> None:
        self._deadline = self._clock() + timeout_s
        self._set_state(State.LISTENING)

    def _go_idle(self) -> None:
        self._history = []
        self._set_state(State.IDLE)

    def _set_state(self, state: State) -> None:
        if state == self._state:
            return
        self._state = state
        _guarded(self._lights.set_state, state)
        _guarded(self._face.send, state, 0.0)


def _get(q: "queue.Queue", timeout: float):
    """The next item, or None if none arrives within timeout seconds."""
    try:
        return q.get(timeout=timeout) if timeout > 0 else q.get_nowait()
    except queue.Empty:
        return None


def _put(q: "queue.Queue", item, cancel: threading.Event) -> bool:
    """Put item, waiting for room unless cancelled; False if it was dropped."""
    while True:
        try:
            q.put(item, timeout=_POLL_S)
            return True
        except queue.Full:
            if cancel.is_set():
                return False


def _guarded(sink: Callable, *args) -> None:
    """Call a lights/face sink; a failing sink is logged, never raised."""
    try:
        sink(*args)
    except Exception:
        log.exception("sink %s failed", getattr(sink, "__qualname__", sink))
