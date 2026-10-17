import math
import sys
import threading
import types

import numpy as np
import pytest

from mirror import audio


def test_envelope_silence():
    env = audio.mouth_envelope(np.zeros(16000, dtype=np.float32), 16000)
    assert len(env) == math.ceil(16000 / 320)
    assert np.all(env == 0)


def test_envelope_tracks_loudness():
    sr = 16000
    x = np.concatenate([np.full(int(0.2 * sr), 0.05), np.full(int(0.2 * sr), 0.5)])
    env = audio.mouth_envelope(x.astype(np.float32), sr)
    half = len(env) // 2
    assert env[half:].mean() > 3 * env[:half].mean()
    assert env.max() <= 1.0


class FakeOutputStream:
    opened = []

    def __init__(self, samplerate, device, channels, dtype):
        self.samplerate = samplerate
        self.latency = 0.0
        self.writes = []
        self.events = []
        FakeOutputStream.opened.append(self)

    def start(self):
        self.events.append("start")

    def write(self, data):
        assert data.dtype == np.float32 and data.shape[1] == 1
        self.writes.append(data)

    def stop(self):
        self.events.append("stop")

    def close(self):
        self.events.append("close")


def _fake_sd(devices=None):
    sd = types.ModuleType("sounddevice")
    sd.OutputStream = FakeOutputStream
    sd.query_devices = lambda: devices or []
    FakeOutputStream.opened = []
    return sd


def _fake_clock(monkeypatch):
    now = [0.0]

    def fake_sleep(s):
        now[0] += s

    monkeypatch.setattr(audio.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(audio.time, "sleep", fake_sleep)
    return now


def test_player_reports_levels(monkeypatch):
    monkeypatch.setitem(sys.modules, "sounddevice", _fake_sd())
    now = _fake_clock(monkeypatch)
    sr = 16000
    dur = 0.5
    x = (np.random.default_rng(0).standard_normal(int(dur * sr)) * 0.1).astype(np.float32)
    levels = []
    audio.Player(None).play(x, sr, levels.append)
    assert len(levels) == math.ceil(dur / 0.02) + 1
    assert levels[-1] == 0.0
    assert dur <= now[0] < dur + 0.05  # paced against the clock, no drift
    (stream,) = FakeOutputStream.opened
    assert np.array_equal(np.concatenate(stream.writes)[:, 0], x)


def test_player_reuses_stream_until_rate_changes(monkeypatch):
    monkeypatch.setitem(sys.modules, "sounddevice", _fake_sd())
    _fake_clock(monkeypatch)
    player = audio.Player(None)
    player.open(22050)
    x = np.zeros(2205, dtype=np.float32)
    player.play(x, 22050, lambda _: None)
    player.play(x, 22050, lambda _: None)
    assert len(FakeOutputStream.opened) == 1
    first = FakeOutputStream.opened[0]
    assert first.events == ["start"] and sum(len(w) for w in first.writes) == 2 * 2205
    player.play(np.zeros(1600, dtype=np.float32), 16000, lambda _: None)
    assert [s.samplerate for s in FakeOutputStream.opened] == [22050, 16000]
    assert first.events == ["start", "stop", "close"]
    player.close()
    assert FakeOutputStream.opened[1].events == ["start", "stop", "close"]


def test_player_write_error_raises_and_reopens(monkeypatch):
    monkeypatch.setitem(sys.modules, "sounddevice", _fake_sd())
    _fake_clock(monkeypatch)
    good_write = FakeOutputStream.write

    def broken_write(self, data):
        raise OSError("device gone")

    player = audio.Player(None)
    monkeypatch.setattr(FakeOutputStream, "write", broken_write)
    with pytest.raises(OSError):
        player.play(np.zeros(320, dtype=np.float32), 16000, lambda _: None)
    assert FakeOutputStream.opened[0].events == ["start", "stop", "close"]
    monkeypatch.setattr(FakeOutputStream, "write", good_write)
    player.play(np.zeros(320, dtype=np.float32), 16000, lambda _: None)
    assert len(FakeOutputStream.opened) == 2  # a fresh stream after the failure


class BlockingOutputStream(FakeOutputStream):
    """write() blocks like a real device: each block takes real time to drain."""

    def write(self, data):
        self.events.append("write")
        threading.Event().wait(0.01)

    def abort(self):
        self.events.append("abort")


class Stop(BaseException):
    pass


def test_player_interrupted_mid_play_aborts_and_joins_writer(monkeypatch):
    sd = _fake_sd()
    sd.OutputStream = BlockingOutputStream
    monkeypatch.setitem(sys.modules, "sounddevice", sd)
    _fake_clock(monkeypatch)
    calls = []

    def on_level(level):
        calls.append(level)
        if len(calls) == 3:
            raise Stop()

    player = audio.Player(None)
    with pytest.raises(Stop):
        player.play(np.zeros(16000, dtype=np.float32), 16000, on_level)
    (stream,) = FakeOutputStream.opened
    # the writer stopped early (1 s = 20 blocks) and the stream was only aborted
    # and closed after it had exited
    assert stream.events[-2:] == ["abort", "close"]
    assert stream.events.count("write") < 20
    player.close()  # already discarded: a no-op
    assert stream.events.count("close") == 1


def test_player_stops_animating_on_write_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "sounddevice", _fake_sd())
    _fake_clock(monkeypatch)
    failed = threading.Event()

    def broken_write(self, data):
        failed.set()
        raise OSError("device gone")

    monkeypatch.setattr(FakeOutputStream, "write", broken_write)
    levels = []

    def on_level(level):
        levels.append(level)
        failed.wait(5)  # make sure the writer has failed before the next hop

    with pytest.raises(OSError):
        audio.Player(None).play(np.zeros(16000, dtype=np.float32), 16000, on_level)
    assert len(levels) <= 2  # stopped early instead of animating 1 s of silence


def test_find_device_matches_direction_case_insensitively(monkeypatch):
    devs = [
        {"name": "HDMI Out", "max_input_channels": 0, "max_output_channels": 2},
        {"name": "USB Mic Array", "max_input_channels": 4, "max_output_channels": 0},
        {"name": "usb speaker", "max_input_channels": 0, "max_output_channels": 2},
    ]
    monkeypatch.setitem(sys.modules, "sounddevice", _fake_sd(devs))
    assert audio.find_device("USB", "input") == 1
    assert audio.find_device("usb", "output") == 2
    assert audio.find_device("nothing", "input") is None
    assert audio.find_device(None, "input") is None
