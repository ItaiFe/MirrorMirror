"""The ambient bed: a looping background track under the oracle's voice.

The Player mixes it in (mirror/audio.py): between clips its own thread keeps
the output fed with the bed, and while a clip plays the clip's writer adds the
bed under it. The level follows the mirror's state (clearly present while idle,
lower while listening, barely there while it thinks and speaks) with a smooth
fade, and the bed is silent during quiet hours.

render() is called by one writer at a time (the Player's I/O lock);
set_state() may come from any thread.
"""
from __future__ import annotations

import datetime
import logging
import os
import time
import wave
from typing import Callable, Optional

import numpy as np

from mirror.audio import resample
from mirror.conversation import in_quiet_hours, parse_quiet_hours
from mirror.states import State

log = logging.getLogger("mirror.ambient")

QUIET_CHECK_S = 10.0  # how often render() re-reads the wall clock for quiet hours


def db_to_gain(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def load_loop(path: str) -> tuple[np.ndarray, int]:
    """A 16-bit PCM WAV as mono float32 (channels averaged) and its rate."""
    with wave.open(path, "rb") as w:
        n_ch, width, rate = w.getnchannels(), w.getsampwidth(), w.getframerate()
        raw = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError(f"{path}: only 16-bit PCM WAV is supported")
    audio = np.frombuffer(raw, dtype="<i2").reshape(-1, n_ch).astype(np.float32)
    mono = audio.mean(axis=1) / 32768.0
    if len(mono) == 0:
        raise ValueError(f"{path}: no audio")
    return mono.astype(np.float32), rate


class AmbientBed:
    """The loop, its read position and its current level.

    volume_db is the idle level (dB on the file as mastered); listening_db and
    answering_db are relative to it. A level change ramps linearly over fade_s.
    """

    def __init__(self, loop: np.ndarray, sr: int, volume_db: float = -12.0,
                 listening_db: float = -10.0, answering_db: float = -24.0,
                 fade_s: float = 1.5, quiet_hours=(),
                 wall_clock: Callable[[], datetime.datetime] = datetime.datetime.now,
                 clock: Callable[[], float] = time.monotonic):
        if len(loop) == 0:
            raise ValueError("ambient loop is empty")
        if fade_s <= 0:
            raise ValueError("ambient fade_s must be > 0")
        self._loop = np.asarray(loop, dtype=np.float32)
        self._sr = sr
        self._levels = {
            State.IDLE: db_to_gain(volume_db),
            State.LISTENING: db_to_gain(volume_db + listening_db),
            State.THINKING: db_to_gain(volume_db + answering_db),
            State.SPEAKING: db_to_gain(volume_db + answering_db),
        }
        self._fade_s = fade_s
        self._quiet = parse_quiet_hours(quiet_hours)
        self._wall_clock, self._clock = wall_clock, clock
        self._state_gain = self._levels[State.IDLE]
        self._gain = 0.0  # fades in from silence at startup
        self._quiet_now = False
        self._quiet_checked: Optional[float] = None
        self._rate: Optional[int] = None  # the rate _data is at
        self._data = self._loop
        self._pos = 0

    def set_state(self, state: State) -> None:
        self._state_gain = self._levels.get(state, self._levels[State.IDLE])

    def prepare(self, rate: int) -> None:
        """Resample the loop for an output at rate (slow for a long loop: call it
        before the stream starts, not from a writer)."""
        if rate == self._rate:
            return
        if rate == self._sr:
            data = self._loop
        else:
            t0 = time.monotonic()
            data = resample(self._loop, self._sr, rate)
            log.info("ambient loop resampled %d -> %d Hz in %.1f s", self._sr, rate,
                     time.monotonic() - t0)
        if self._rate is not None:  # keep the place in the music
            self._pos = int(self._pos * len(data) / len(self._data)) % len(data)
        self._data, self._rate = data, rate

    def _target(self) -> float:
        now = self._clock()
        if self._quiet and (self._quiet_checked is None
                            or now - self._quiet_checked >= QUIET_CHECK_S):
            self._quiet_checked = now
            quiet = in_quiet_hours(self._quiet, self._wall_clock().time())
            if quiet != self._quiet_now:
                log.info("ambient %s (quiet hours)", "off" if quiet else "on")
            self._quiet_now = quiet
        return 0.0 if self._quiet_now else self._state_gain

    def render(self, n: int, rate: int) -> np.ndarray:
        """The next n samples at rate, at the current (fading) level."""
        self.prepare(rate)
        data = self._data
        idx = (self._pos + np.arange(n)) % len(data)
        self._pos = int((self._pos + n) % len(data))
        target = self._target()
        step = max(self._levels.values()) / (self._fade_s * rate)  # full range per fade_s
        delta = target - self._gain
        if abs(delta) <= step * n:  # reaches the target within this block
            reach = int(abs(delta) / step) if step > 0 else 0
            ramp = np.full(n, target, dtype=np.float32)
            if reach:
                ramp[:reach] = self._gain + np.sign(delta) * step * np.arange(1, reach + 1)
            self._gain = target
        else:
            ramp = (self._gain + np.sign(delta) * step * np.arange(1, n + 1)).astype(np.float32)
            self._gain = float(ramp[-1])
        return (data[idx] * ramp).astype(np.float32)


def build_ambient(cfg, path: str) -> Optional[AmbientBed]:
    """The bed from [ambient], or None if it is off or its file can't be loaded
    (a WARNING; the mirror runs without it)."""
    if not cfg.enabled:
        return None
    if not os.path.exists(path):
        log.warning("ambient file %s not found; running without background music", path)
        return None
    try:
        loop, sr = load_loop(path)
        bed = AmbientBed(loop, sr, cfg.volume_db, cfg.listening_db, cfg.answering_db,
                         cfg.fade_s, cfg.quiet_hours)
    except Exception as e:  # noqa: BLE001 - background music is never fatal
        log.warning("ambient file %s could not be loaded (%s: %s); running without it",
                    path, type(e).__name__, e)
        return None
    log.info("ambient: %s (%.0f s loop, idle %.0f dB)", path, len(loop) / sr, cfg.volume_db)
    return bed
