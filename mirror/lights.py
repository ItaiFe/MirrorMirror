"""Frame lights: WLED presets over USB serial."""
import logging
import time
from typing import Callable, Optional, Protocol

from mirror.states import State

log = logging.getLogger("mirror.lights")

PRESETS = {State.IDLE: 1, State.LISTENING: 2, State.THINKING: 3, State.SPEAKING: 4}
_REOPEN_INTERVAL_S = 10.0


class Lights(Protocol):
    def set_state(self, state: State) -> None: ...


class LogLights:
    """Lights stand-in that just logs each state change."""

    def set_state(self, state: State) -> None:
        log.info("lights: %s", state.value)


class WledSerial:
    def __init__(
        self,
        port: str,
        baud: int,
        write: Optional[Callable[[bytes], None]] = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._port = port
        self._baud = baud
        self._write = write
        self._clock = clock
        self._serial = None
        self._last_open_attempt: Optional[float] = None
        self._last_sent: Optional[State] = None
        self._failing = False

    def set_state(self, state: State) -> None:
        if state == self._last_sent:
            return
        data = b'{"ps":%d}\n' % PRESETS[state]
        try:
            self._send(data)
        except Exception as exc:  # keep the assistant running without lights
            self._last_sent = None
            if not self._failing:
                self._failing = True
                log.warning("WLED lights unavailable (%s): %s", self._port, exc)
            return
        self._failing = False
        self._last_sent = state

    def _send(self, data: bytes) -> None:
        if self._write is not None:
            self._write(data)
            return
        if self._serial is None:
            now = self._clock()
            if (
                self._last_open_attempt is not None
                and now - self._last_open_attempt < _REOPEN_INTERVAL_S
            ):
                raise OSError("serial port closed; waiting to retry open")
            self._last_open_attempt = now
            import serial  # lazy: hardware dep

            # write_timeout: a wedged ESP32 must not block the controller thread; a
            # SerialTimeoutException then takes the close-and-retry path below
            self._serial = serial.Serial(self._port, self._baud, timeout=1, write_timeout=1)
        try:
            self._serial.write(data)
            self._serial.flush()
        except Exception:  # incl. serial.SerialTimeoutException
            try:
                self._serial.close()
            except Exception:
                pass
            self._serial = None
            raise
