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
        self.channels = channels
        self.latency = 0.0
        self.writes = []
        self.events = []
        FakeOutputStream.opened.append(self)

    def start(self):
        self.events.append("start")

    def write(self, data):
        assert data.dtype == np.float32 and data.shape[1] == self.channels
        self.writes.append(data)

    def stop(self):
        self.events.append("stop")

    def close(self):
        self.events.append("close")


def _fake_sd(devices=None, accepts=None, default=None):
    """accepts: the (rate, channels) the device takes (None: anything);
    default: what query_devices(device, kind) reports for it."""
    sd = types.ModuleType("sounddevice")
    sd.OutputStream = FakeOutputStream
    sd.checked = []

    def query_devices(device=None, kind=None):
        if kind is None:
            return devices or []
        return default or {"name": "fake", "default_samplerate": 44100.0,
                           "max_input_channels": 2, "max_output_channels": 2}

    def check(device=None, samplerate=None, channels=None, dtype=None):
        sd.checked.append((samplerate, channels))
        if accepts is not None and (samplerate, channels) not in accepts:
            raise ValueError("Invalid sample rate")

    sd.query_devices = query_devices
    sd.check_output_settings = check
    sd.check_input_settings = check
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


def test_find_device_prefers_a_converting_device(monkeypatch, caplog):
    devs = [
        {"name": "XVF3800: USB Audio (hw:2,0)", "max_input_channels": 2,
         "max_output_channels": 2},
        {"name": "XVF3800 plughw:2,0", "max_input_channels": 2, "max_output_channels": 2},
    ]
    monkeypatch.setitem(sys.modules, "sounddevice", _fake_sd(devs))
    assert audio.find_device("xvf3800", "output") == 0
    with caplog.at_level("INFO", logger="mirror.audio"):
        assert audio.find_device("xvf3800", "output", prefer_plug=True) == 1
    assert "plughw" in caplog.text
    one = [devs[0]]
    monkeypatch.setitem(sys.modules, "sounddevice", _fake_sd(one))
    assert audio.find_device("xvf3800", "input", prefer_plug=True) == 0


def test_resample_lengths_and_identity():
    x = np.sin(np.arange(22050) / 10).astype(np.float32)
    assert np.array_equal(audio.resample(x, 22050, 22050), x)
    y = audio.resample(x, 22050, 48000)
    assert y.dtype == np.float32 and abs(len(y) - 48000) <= 1


@pytest.mark.parametrize("sr_in", [48000, 44100, 16000])
def test_stream_resampler_has_no_seams(sr_in):
    rng = np.random.default_rng(1)
    x = rng.standard_normal(sr_in).astype(np.float32)  # 1 s
    whole = audio.StreamResampler(sr_in, 16000).feed(x)
    r = audio.StreamResampler(sr_in, 16000)
    parts = np.concatenate([r.feed(x[i:i + 1000]) for i in range(0, len(x), 1000)])
    assert abs(len(parts) - 16000) <= 2
    n = min(len(whole), len(parts))
    np.testing.assert_allclose(parts[:n], whole[:n], atol=1e-5)


def test_stream_resampler_integer_factor_averages():
    r = audio.StreamResampler(48000, 16000)
    out = r.feed(np.array([1, 2, 3, 4, 5], dtype=np.float32))
    assert np.allclose(out, [2.0])
    assert np.allclose(r.feed(np.array([6], dtype=np.float32)), [5.0])


class FakeInputStream:
    opened = []

    def __init__(self, samplerate, blocksize, device, channels, dtype, callback):
        self.samplerate, self.blocksize, self.channels = samplerate, blocksize, channels
        self.callback = callback
        self.events = []
        FakeInputStream.opened.append(self)

    def start(self):
        self.events.append("start")

    def abort(self):
        self.events.append("abort")

    def close(self):
        self.events.append("close")


def test_mic_opens_a_supported_format_and_yields_16k_mono(monkeypatch, caplog):
    sd = _fake_sd(accepts=[(48000, 2)])
    sd.InputStream = FakeInputStream
    FakeInputStream.opened = []
    monkeypatch.setitem(sys.modules, "sounddevice", sd)
    with caplog.at_level("INFO", logger="mirror.audio"):
        mic = audio.Mic(None, 16000)
        mic.__enter__()
    assert sd.checked[:3] == [(16000, 1), (16000, 2), (48000, 2)]
    (stream,) = FakeInputStream.opened
    assert (stream.samplerate, stream.channels, stream.blocksize) == (48000, 2, 1536)
    assert "48000 Hz x 2 ch" in caplog.text
    left = np.arange(1536, dtype=np.float32).reshape(-1, 1)
    block = np.hstack([left, np.full_like(left, 99.0)])  # channel 1 must be ignored
    stream.callback(block, 1536, None, None)
    chunk = mic.read()
    assert chunk.shape == (audio.CHUNK,) and chunk.dtype == np.float32
    assert np.allclose(chunk[:3], [1.0, 4.0, 7.0])  # means of channel-0 triples
    assert mic.reads == 1
    mic.__exit__(None, None, None)
    assert stream.events == ["start", "abort", "close"]


def test_mic_uses_device_default_when_nothing_else_fits(monkeypatch):
    sd = _fake_sd(accepts=[(44100, 2)])
    sd.InputStream = FakeInputStream
    FakeInputStream.opened = []
    monkeypatch.setitem(sys.modules, "sounddevice", sd)
    with audio.Mic(None, 16000) as mic:
        (stream,) = FakeInputStream.opened
        assert (stream.samplerate, stream.channels) == (44100, 2)
        stream.callback(np.zeros((stream.blocksize, 2), np.float32), stream.blocksize,
                        None, None)
        stream.callback(np.zeros((stream.blocksize, 2), np.float32), stream.blocksize,
                        None, None)
        assert len(mic.read()) == audio.CHUNK


def test_mic_read_times_out_when_the_device_stalls(monkeypatch):
    sd = _fake_sd()
    sd.InputStream = FakeInputStream
    monkeypatch.setitem(sys.modules, "sounddevice", sd)
    with audio.Mic(None, 16000, read_timeout_s=0.05) as mic:
        with pytest.raises(audio.MicStalled):
            mic.read()


def test_player_resamples_and_duplicates_to_stereo(monkeypatch, caplog):
    sd = _fake_sd(accepts=[(48000, 2)])
    monkeypatch.setitem(sys.modules, "sounddevice", sd)
    now = _fake_clock(monkeypatch)
    sr, dur = 22050, 0.5
    x = (np.random.default_rng(0).standard_normal(int(dur * sr)) * 0.1).astype(np.float32)
    levels = []
    with caplog.at_level("INFO", logger="mirror.audio"):
        audio.Player(None).play(x, sr, levels.append)
    (stream,) = FakeOutputStream.opened
    assert (stream.samplerate, stream.channels) == (48000, 2)
    assert "48000 Hz x 2 ch" in caplog.text
    out = np.concatenate(stream.writes)
    assert abs(len(out) - int(dur * 48000)) <= 1
    assert np.array_equal(out[:, 0], out[:, 1])
    # levels stay on the clip's own timeline
    assert len(levels) == math.ceil(len(x) / round(sr * 0.02)) + 1
    assert dur <= now[0] < dur + 0.05


def test_player_keeps_clip_rate_when_device_accepts_it(monkeypatch):
    sd = _fake_sd()
    monkeypatch.setitem(sys.modules, "sounddevice", sd)
    _fake_clock(monkeypatch)
    audio.Player(None).play(np.zeros(2205, np.float32), 22050, lambda _: None)
    (stream,) = FakeOutputStream.opened
    assert (stream.samplerate, stream.channels) == (22050, 1)
    assert sd.checked == [(22050, 1)]


class WedgedOutputStream(FakeOutputStream):
    """The first write() hangs (a wedged device) until released."""

    release = None

    def write(self, data):
        self.events.append("write")
        WedgedOutputStream.release.wait(0.5)

    def abort(self):
        self.events.append("abort")


def test_player_raises_when_a_write_wedges(monkeypatch):
    sd = _fake_sd()
    sd.OutputStream = WedgedOutputStream
    WedgedOutputStream.release = threading.Event()
    monkeypatch.setitem(sys.modules, "sounddevice", sd)
    monkeypatch.setattr(audio, "PLAY_JOIN_GRACE_S", 0.05)
    _fake_clock(monkeypatch)
    player = audio.Player(None)
    with pytest.raises(audio.PlayerStalled):
        player.play(np.zeros(1600, np.float32), 16000, lambda _: None)
    (stream,) = FakeOutputStream.opened
    assert stream.events[-2:] == ["abort", "close"]
    assert player._stream is None  # the next play opens a fresh stream


class FailingStartInputStream(FakeInputStream):
    def start(self):
        self.events.append("start")
        raise OSError("device busy")


def test_mic_closes_the_stream_when_start_fails(monkeypatch):
    sd = _fake_sd()
    sd.InputStream = FailingStartInputStream
    FakeInputStream.opened = []
    monkeypatch.setitem(sys.modules, "sounddevice", sd)
    mic = audio.Mic(None, 16000)
    with pytest.raises(OSError, match="busy"):
        mic.__enter__()
    (stream,) = FakeInputStream.opened
    assert stream.events[-1] == "close"  # not leaked
    assert mic._stream is None
    mic.__exit__(None, None, None)  # a no-op now
    assert stream.events.count("close") == 1
