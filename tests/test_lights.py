from mirror.lights import PRESETS, LogLights, WledSerial
from mirror.states import State


def test_sends_preset_json():
    sent = []
    WledSerial("/dev/null", 115200, write=sent.append).set_state(State.SPEAKING)
    assert sent == [b'{"ps":4}\n']


def test_dedupes_same_state():
    sent = []
    lights = WledSerial("/dev/null", 115200, write=sent.append)
    lights.set_state(State.IDLE)
    lights.set_state(State.IDLE)
    assert len(sent) == 1


def test_write_failure_does_not_raise():
    def boom(_data):
        raise OSError("gone")

    WledSerial("/dev/null", 115200, write=boom).set_state(State.LISTENING)


def test_failed_write_retries_and_logs_once(caplog):
    calls = []
    fail = [True]

    def write(data):
        calls.append(data)
        if fail[0]:
            raise OSError("gone")

    lights = WledSerial("/dev/null", 115200, write=write)
    with caplog.at_level("WARNING", logger="mirror.lights"):
        lights.set_state(State.THINKING)
        lights.set_state(State.THINKING)
    assert len(calls) == 2
    assert len([r for r in caplog.records if r.levelname == "WARNING"]) == 1
    fail[0] = False
    lights.set_state(State.THINKING)
    lights.set_state(State.THINKING)
    assert len(calls) == 3


def test_open_retry_throttled_to_10s(monkeypatch):
    import sys
    import types

    opens = []

    class FakeSerial:
        def __init__(self, port, baud, **kwargs):
            opens.append(port)
            raise OSError("no port")

    monkeypatch.setitem(sys.modules, "serial", types.SimpleNamespace(Serial=FakeSerial))
    now = [100.0]
    lights = WledSerial("/dev/ttyX", 115200, clock=lambda: now[0])
    lights.set_state(State.IDLE)
    now[0] += 5
    lights.set_state(State.SPEAKING)
    assert len(opens) == 1
    now[0] += 6
    lights.set_state(State.LISTENING)
    assert len(opens) == 2


def test_presets_and_loglights(caplog):
    assert PRESETS == {State.IDLE: 1, State.LISTENING: 2, State.THINKING: 3, State.SPEAKING: 4}
    with caplog.at_level("INFO", logger="mirror.lights"):
        LogLights().set_state(State.IDLE)
    assert "idle" in caplog.text


def test_serial_write_timeout_closes_and_retries(monkeypatch, caplog):
    import sys
    import types

    class SerialTimeoutException(OSError):
        pass

    made = []

    class FakeSerial:
        def __init__(self, port, baud, **kwargs):
            self.kwargs = kwargs
            self.closed = False
            made.append(self)

        def write(self, data):
            raise SerialTimeoutException("Write timeout")

        def flush(self):
            pass

        def close(self):
            self.closed = True

    monkeypatch.setitem(sys.modules, "serial", types.SimpleNamespace(
        Serial=FakeSerial, SerialTimeoutException=SerialTimeoutException))
    now = [0.0]
    lights = WledSerial("/dev/ttyX", 115200, clock=lambda: now[0])
    with caplog.at_level("WARNING", logger="mirror.lights"):
        lights.set_state(State.SPEAKING)  # must return, not raise or block
    (ser,) = made
    assert ser.kwargs["write_timeout"] == 1 and ser.kwargs["timeout"] == 1
    assert ser.closed
    assert "Write timeout" in caplog.text
    now[0] += 11
    lights.set_state(State.SPEAKING)
    assert len(made) == 2  # reopened after the retry interval


def test_reopens_at_once_after_a_healthy_port_fails(monkeypatch):
    """The open-retry throttle is for a port that won't open: once a port has worked,
    a later failure is retried straight away (once), not up to 10 s later."""
    import sys
    import types

    made = []

    class FakeSerial:
        fail = False

        def __init__(self, port, baud, **kwargs):
            made.append(self)

        def write(self, data):
            if FakeSerial.fail:
                raise OSError("unplugged")

        def flush(self):
            pass

        def close(self):
            pass

    monkeypatch.setitem(sys.modules, "serial", types.SimpleNamespace(Serial=FakeSerial))
    now = [0.0]
    lights = WledSerial("/dev/ttyX", 115200, clock=lambda: now[0])
    lights.set_state(State.LISTENING)  # opens at t=0, works
    now[0] = 5.0
    FakeSerial.fail = True
    lights.set_state(State.THINKING)  # fails, port closed
    now[0] = 6.0
    FakeSerial.fail = False
    lights.set_state(State.SPEAKING)
    assert len(made) == 2  # reopened at once
    assert lights._last_sent == State.SPEAKING
    FakeSerial.fail = True
    now[0] = 7.0
    lights.set_state(State.IDLE)  # fails right after a (re)open...
    now[0] = 8.0
    lights.set_state(State.IDLE)
    assert len(made) == 3  # ...the first retry is immediate
    now[0] = 9.0
    lights.set_state(State.IDLE)
    assert len(made) == 3  # but a port failing every write is throttled again
