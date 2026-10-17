"""Conversation controller: the IDLE/LISTENING/THINKING/SPEAKING state machine."""
import logging
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


class Controller:
    def __init__(self, stt, llm, tts, player, lights, face, cfg: ConversationConfig,
                 clock: Callable[[], float] = time.monotonic):
        self._stt = stt
        self._llm = llm
        self._tts = tts
        self._player = player
        self._lights = lights
        self._face = face
        self._cfg = cfg
        self._clock = clock
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
        try:
            reply = self._llm.reply(self._history, text, lang)
        except LLMError as e:
            log.warning("llm failed: %s", e)
            self._speak(FALLBACK[lang], lang)
            self._go_idle()
            return
        log.info("reply (%s): %r", lang, reply)
        self._speak(reply, lang)
        self._history.extend([{"role": "user", "content": text},
                              {"role": "assistant", "content": reply}])
        self._history = self._history[-_HISTORY_MAX:]
        self._listen(self._cfg.followup_s)

    def _speak(self, text: str, lang: str) -> None:
        self._set_state(State.SPEAKING)
        samples, sr, track = self._tts.synth_visemes(text, lang)
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


def _guarded(sink: Callable, *args) -> None:
    """Call a lights/face sink; a failing sink is logged, never raised."""
    try:
        sink(*args)
    except Exception:
        log.exception("sink %s failed", getattr(sink, "__qualname__", sink))
