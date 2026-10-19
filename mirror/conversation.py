"""Conversation controller: the IDLE/LISTENING/THINKING/SPEAKING state machine.

An answer is a small pipeline: a worker streams the LLM reply sentence by sentence,
a second worker synthesizes each sentence while the one before it plays, and the
calling thread plays a filler line and then each sentence. Both workers are stopped
and joined before _answer returns, whatever way it returns.

While no one talks to it, tick() now and then plays an idle lure: a short
pre-synthesized line, through the same play path as a filler, never touching the
conversation history.
"""
import logging
import queue
import random
import re
import threading
import time
from datetime import datetime
from datetime import time as dtime
from typing import Callable, Optional

import numpy as np

from mirror.config import ConversationConfig
from mirror.llm import LLMError, server_unreachable
from mirror.persona import FALLBACK, SYSTEM_PROMPT, pick_angle
from mirror.states import State
from mirror.stt import STTError
from mirror.visemes import rms_shape
from mirror.wake import is_hallucination, match_wake

log = logging.getLogger("mirror.conversation")

_HISTORY_MAX = 6
HOP_S = 0.02   # the player reports one mouth level per 20 ms hop
LEAD_S = 0.05  # show each mouth shape slightly before its sound
_POLL_S = 0.1  # how often a blocked worker checks whether it was cancelled
STALL_S = 0.6  # no next sentence this long after one finished -> show THINKING
REWARM_AFTER_S = 600.0  # re-warm a language's prompt cache after this much idle
REWARM_RETRY_S = 30.0  # after a failed re-warm, try again this much later
REWARM_WAIT_S = 0.5  # how long an answer waits for an in-flight re-warm
# recent answer angles are remembered this long, across conversations too, so the
# next visitor does not hear the same kind of answer
ANGLE_MEMORY_S = 900.0
_ANGLES_KEPT = 8  # more than any no-repeat window
_END = object()  # end of the sentence / clip stream


class Controller:
    def __init__(self, stt, llm, tts, player, lights, face, cfg: ConversationConfig,
                 clock: Callable[[], float] = time.monotonic, fillers=None,
                 first_sentence_timeout_s: float = 20.0, lures=None,
                 wall_clock: Callable[[], datetime] = datetime.now,
                 rng: Optional[random.Random] = None,
                 angle_rng: Optional[random.Random] = None):
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
        # Last LLM use per language; voice_main warms both slots before this exists
        # and marks a language whose warmup failed with mark_unwarmed().
        self._last_llm = {lang: clock() for lang in SYSTEM_PROMPT}
        self._rewarm_thread: "threading.Thread | None" = None
        self._closed = threading.Event()
        # idle lures; quiet hours are checked against the local wall clock
        self._lures = lures  # FillerBank of LURES or None
        self._wall_clock = wall_clock
        self._rng = rng or random.Random()
        self._quiet_hours = parse_quiet_hours(cfg.lure_quiet_hours)
        lo, hi = (float(x) for x in cfg.lure_interval_s)
        if not 0 <= lo <= hi:
            raise ValueError(f"lure_interval_s must be [min, max] with 0 <= min <= max, "
                             f"got {cfg.lure_interval_s!r}")
        self._lure_interval = (lo, hi)
        self._last_interaction = clock()
        self._next_lure_at = self._last_interaction + self._draw_lure_wait()
        # answer angles: (time, Angle), oldest first; own rng, so lure draws stay put
        self._angle_rng = angle_rng or random.Random()
        self._recent_angles: list = []

    def mark_unwarmed(self, lang: str) -> None:
        """The language's prompt cache is cold: re-warm it at the next idle tick."""
        self._last_llm[lang] = float("-inf")

    def close(self, timeout_s: float = 1.0) -> None:
        """Start no more re-warms and wait (bounded) for one in flight."""
        self._closed.set()
        t = self._rewarm_thread
        if t is not None:
            t.join(timeout_s)
            if t.is_alive():
                log.warning("prompt-cache re-warm still running at shutdown")

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
            if self._busy:
                return
            if self._state == State.LISTENING and self._clock() >= self._deadline:
                log.info("listening timed out")
                self._go_idle()
            self._maybe_rewarm()
            lure = self._due_lure()
            if lure is None:
                return
            self._busy = True  # segments are dropped while the lure plays
        try:
            self._play_lure(*lure)
        finally:
            self._busy = False

    # --- idle lures -------------------------------------------------------------

    def _draw_lure_wait(self) -> float:
        return self._rng.uniform(*self._lure_interval)

    def _mark_interaction(self) -> None:
        """Someone talked to the mirror: restart the idle timer."""
        self._last_interaction = self._clock()
        self._next_lure_at = self._last_interaction + self._draw_lure_wait()

    def _due_lure(self):
        """(lang, clip) if a lure should play now, else None. Never raises."""
        try:
            if (self._lures is None or not self._cfg.lures_enabled
                    or self._state != State.IDLE or self._closed.is_set()):
                return None
            now = self._clock()
            if (now - self._last_interaction < self._cfg.lure_min_idle_s
                    or now < self._next_lure_at):
                return None
            if in_quiet_hours(self._quiet_hours, self._wall_clock().time()):
                return None
            langs = self._lures.languages()
            if not langs:
                return None
            # a lure is due: whatever happens next, wait a fresh interval for the next
            self._next_lure_at = now + self._draw_lure_wait()
            lang = self._rng.choice(sorted(langs))
            clip = self._lures.pick(lang)
            return None if clip is None else (lang, clip)
        except Exception:
            log.exception("choosing a lure failed")
            return None

    def _play_lure(self, lang: str, clip) -> None:
        """Speak the lure (SPEAKING with visemes), then back to IDLE. The history
        is not touched. Never raises an Exception."""
        log.info("lure (%s)", lang)
        try:
            self._play_clip(*clip)
        except Exception:
            log.exception("lure failed")
        finally:
            self._set_state(State.IDLE)
            # the wait for the next lure starts once this one is over
            self._next_lure_at = self._clock() + self._draw_lure_wait()

    def _maybe_rewarm(self) -> None:
        """Re-warm, in the background, the prompt cache of the language idle longest,
        if that is more than REWARM_AFTER_S. One language at a time, so a question
        never waits behind more than one short warm request. Called only while not
        busy."""
        warm = getattr(self._llm, "warm", None)
        if (warm is None or self._closed.is_set()
                or (self._rewarm_thread is not None and self._rewarm_thread.is_alive())):
            return
        now = self._clock()
        lang = min(self._last_llm, key=self._last_llm.get)
        if now - self._last_llm[lang] <= REWARM_AFTER_S:
            return
        self._last_llm[lang] = now
        self._rewarm_thread = threading.Thread(target=self._rewarm, args=(warm, lang),
                                               name="llm-rewarm", daemon=True)
        self._rewarm_thread.start()

    def _rewarm(self, warm: Callable, lang: str) -> None:
        if self._closed.is_set():
            return
        try:
            warm(lang)
        except Exception as e:  # noqa: BLE001 - background best effort
            log.warning("re-warming the %s prompt cache failed: %s", lang, e)
            self._last_llm[lang] = self._clock() - REWARM_AFTER_S + REWARM_RETRY_S

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
        self._mark_interaction()
        if text:
            self._answer(text, transcript.lang)
        else:
            self._listen(self._cfg.listen_timeout_s)

    def _answer(self, text: str, lang: str) -> None:
        self._set_state(State.THINKING)
        self._last_llm[lang] = self._clock()
        self._wait_for_rewarm()
        angle = self._next_angle()
        cancel = threading.Event()
        sentences: "queue.Queue" = queue.Queue()
        clips: "queue.Queue" = queue.Queue(maxsize=1)  # synthesize one sentence ahead
        workers = [
            threading.Thread(target=self._stream_worker, name="llm-stream", daemon=True,
                             args=(list(self._history), text, lang, sentences, cancel,
                                   angle)),
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
                if server_unreachable(e):
                    # llama-server is down or restarting: its slots will come back
                    # cold, so re-warm both at the next idle ticks
                    for slot_lang in self._last_llm:
                        self.mark_unwarmed(slot_lang)
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

    def _next_angle(self):
        """A fresh angle for this answer, unlike the recent ones (forgotten after
        ANGLE_MEMORY_S); None if picking fails. Never raises an Exception."""
        try:
            now = self._clock()
            self._recent_angles = [(t, a) for t, a in self._recent_angles
                                   if now - t <= ANGLE_MEMORY_S][-_ANGLES_KEPT:]
            angle = pick_angle(self._angle_rng, [a for _, a in self._recent_angles])
            self._recent_angles.append((now, angle))
            log.info("angle: %s, %s", angle.mode, angle.imagery)
            return angle
        except Exception:
            log.exception("choosing an answer angle failed")
            return None

    def _wait_for_rewarm(self) -> None:
        """Let an in-flight re-warm finish (bounded) so the question does not queue
        behind it on the server."""
        t = self._rewarm_thread
        if t is None or not t.is_alive():
            return
        t.join(REWARM_WAIT_S)
        if t.is_alive():
            log.info("prompt-cache re-warm still running; asking anyway")

    def _play_reply(self, lang: str, clips: "queue.Queue") -> str:
        """Play the filler, then each synthesized sentence as it is ready; return the
        spoken reply. Raises LLMError if no sentence comes (in time). A failure after
        the first sentence was spoken ends the reply with what was spoken."""
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
        try:
            while item is not _END:
                if isinstance(item, BaseException):
                    raise item
                sentence, samples, sr, track = item
                log.info("reply sentence %d (%s): %r", len(spoken) + 1, lang, sentence)
                self._play_clip(samples, sr, track)
                spoken.append(sentence)
                item = _get(clips, STALL_S)
                if item is None:  # the next sentence is late: look thoughtful
                    self._set_state(State.THINKING)
                    item = clips.get()
        except Exception as e:
            if not spoken:
                raise
            log.warning("reply broke off after %d sentence(s): %s", len(spoken), e,
                        exc_info=not isinstance(e, LLMError))
        if not spoken:
            raise LLMError("llama-server returned an empty reply")
        return " ".join(spoken)

    def _stream_worker(self, history: list[dict], text: str, lang: str,
                       out: "queue.Queue", cancel: threading.Event, angle=None) -> None:
        """LLM sentences -> out; then an exception (if the stream failed) and _END."""
        stream = None
        try:
            stream = self._llm.reply_stream(history, text, lang, cancel=cancel,
                                            angle=angle)
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
        """End of a conversation (a lure never comes here)."""
        self._history = []
        self._set_state(State.IDLE)
        self._mark_interaction()  # the idle timer starts now

    def _set_state(self, state: State) -> None:
        if state == self._state:
            return
        self._state = state
        _guarded(self._lights.set_state, state)
        _guarded(self._face.send, state, 0.0)


_QUIET = re.compile(r"^\s*(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})\s*$")


def parse_quiet_hours(windows) -> list:
    """["HH:MM-HH:MM", ...] -> [(start, end) datetime.time pairs]; ValueError if a
    window is malformed or empty (start == end)."""
    parsed = []
    for w in windows:
        m = _QUIET.match(w) if isinstance(w, str) else None
        if m is None:
            raise ValueError(f"lure_quiet_hours: {w!r} is not \"HH:MM-HH:MM\"")
        h1, m1, h2, m2 = map(int, m.groups())
        try:
            start, end = dtime(h1, m1), dtime(h2, m2)
        except ValueError as e:
            raise ValueError(f"lure_quiet_hours: {w!r}: {e}") from None
        if start == end:
            raise ValueError(f"lure_quiet_hours: {w!r} is empty (start == end)")
        parsed.append((start, end))
    return parsed


def in_quiet_hours(windows, now: dtime) -> bool:
    """True if now falls in a [start, end) window; a window with end < start
    crosses midnight."""
    for start, end in windows:
        if start < end:
            if start <= now < end:
                return True
        elif now >= start or now < end:
            return True
    return False


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
