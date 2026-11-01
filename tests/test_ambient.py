import datetime
import logging
import sys
import threading
import wave

import numpy as np
import pytest

from mirror import audio
from mirror.ambient import AmbientBed, build_ambient, db_to_gain, load_loop
from mirror.config import AmbientConfig, load_config
from mirror.states import State
from mirror.voice_main import AnswerGate
from tests.test_audio import FakeOutputStream, _fake_sd

SR = 1000  # a tiny rate keeps the arithmetic readable


def _bed(loop=None, **kw):
    loop = np.ones(SR, dtype=np.float32) if loop is None else loop
    kw.setdefault("volume_db", 0.0)
    kw.setdefault("fade_s", 0.1)
    return AmbientBed(loop, SR, **kw)


def _write_wav(path, frames: np.ndarray, rate: int, channels: int):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((frames * 32767).astype("<i2").tobytes())


def test_load_loop_mixes_channels_to_mono(tmp_path):
    stereo = np.array([[0.5, -0.5], [0.25, 0.25]] * 10, dtype=np.float32)
    _write_wav(tmp_path / "a.wav", stereo, 8000, 2)
    loop, sr = load_loop(str(tmp_path / "a.wav"))
    assert sr == 8000 and loop.shape == (20,)
    assert loop[0] == pytest.approx(0.0, abs=1e-4)
    assert loop[1] == pytest.approx(0.25, abs=1e-3)


def test_render_wraps_around_the_loop_and_keeps_its_place():
    loop = np.arange(10, dtype=np.float32) / 10
    bed = _bed(loop, fade_s=1e-6)  # no fade: the samples come out as they are
    first = bed.render(7, SR)
    second = bed.render(7, SR)
    assert np.allclose(first[1:], loop[1:7])  # (sample 0 is the fade's first step)
    assert np.allclose(second, loop[[7, 8, 9, 0, 1, 2, 3]])


def test_fades_in_from_silence_then_holds_the_idle_level():
    bed = _bed(volume_db=-6.0)
    out = bed.render(SR, SR)  # 1 s; the fade is 0.1 s
    assert out[0] < 0.05
    assert np.all(np.diff(out[:100]) >= 0)  # rises smoothly, no jump
    assert np.allclose(out[150:], db_to_gain(-6.0))


def test_level_follows_the_state():
    bed = _bed(listening_db=-10.0, answering_db=-24.0)
    bed.render(SR, SR)
    for state, db in ((State.LISTENING, -10.0), (State.SPEAKING, -24.0),
                      (State.THINKING, -24.0), (State.IDLE, 0.0)):
        bed.set_state(state)
        out = bed.render(SR, SR)
        assert out[-1] == pytest.approx(db_to_gain(db), rel=1e-3), state
        assert np.max(np.abs(np.diff(out))) < 0.02  # a fade, not a cut


def test_quiet_hours_silence_the_bed():
    now = [datetime.datetime(2026, 11, 3, 2, 0)]
    t = [0.0]
    bed = _bed(quiet_hours=["01:00-05:00"], wall_clock=lambda: now[0], clock=lambda: t[0])
    assert np.allclose(bed.render(SR, SR)[-1], 0.0)
    now[0] = datetime.datetime(2026, 11, 3, 6, 0)
    t[0] += 60  # the clock is re-read only every QUIET_CHECK_S
    assert bed.render(SR, SR)[-1] == pytest.approx(1.0)


def test_prepare_resamples_for_the_output_rate():
    bed = _bed(np.ones(SR, dtype=np.float32), fade_s=1e-6)
    out = bed.render(4 * SR, 2 * SR)  # 2 s of a 1 s loop at twice the rate
    assert len(out) == 4 * SR and np.allclose(out[10:-10], 1.0, atol=0.02)


def test_bad_settings_are_rejected():
    with pytest.raises(ValueError):
        _bed(fade_s=0)
    with pytest.raises(ValueError):
        AmbientBed(np.zeros(0, np.float32), SR)
    with pytest.raises(ValueError):
        AmbientConfig(fade_s=0)
    with pytest.raises(ValueError):
        AmbientConfig(volume_db=3.0)


def test_build_ambient_off_missing_or_broken_runs_without_it(tmp_path, caplog):
    assert build_ambient(AmbientConfig(enabled=False), str(tmp_path / "x.wav")) is None
    with caplog.at_level(logging.WARNING):
        assert build_ambient(AmbientConfig(enabled=True), str(tmp_path / "x.wav")) is None
        (tmp_path / "bad.wav").write_bytes(b"not a wav")
        assert build_ambient(AmbientConfig(enabled=True), str(tmp_path / "bad.wav")) is None
    assert "not found" in caplog.text and "could not be loaded" in caplog.text


def test_build_ambient_loads_the_file(tmp_path):
    _write_wav(tmp_path / "a.wav", np.full(800, 0.5, np.float32), 8000, 1)
    bed = build_ambient(AmbientConfig(enabled=True), str(tmp_path / "a.wav"))
    assert isinstance(bed, AmbientBed)


def test_answer_gate_tells_the_bed_the_state():
    states = []

    class Lights:
        def set_state(self, s):
            pass

    class Bed:
        def set_state(self, s):
            states.append(s)

    gate = AnswerGate(Lights(), ambient=Bed())
    gate.set_state(State.LISTENING)
    gate.set_state(State.SPEAKING)
    assert states == [State.LISTENING, State.SPEAKING]


def test_shipped_configs_turn_the_ambient_on():
    mac, orin = load_config("config/mac.toml").ambient, load_config("config/orin.toml").ambient
    assert mac.enabled and orin.enabled
    assert mac.file == "models/ambient_loop.wav"
    assert orin.file == "/home/mirror/mirror-models/ambient_loop.wav"
    assert mac.answering_db < mac.listening_db < 0  # barely there while answering


class PacedOutputStream(FakeOutputStream):
    """Each write takes real time, like a device draining its buffer."""
    lock = threading.Lock()
    active = 0
    overlap = False

    def write(self, data):
        with PacedOutputStream.lock:
            PacedOutputStream.active += 1
            PacedOutputStream.overlap |= PacedOutputStream.active > 1
        threading.Event().wait(0.002)
        self.writes.append(np.array(data))
        with PacedOutputStream.lock:
            PacedOutputStream.active -= 1


def test_player_plays_the_bed_between_clips_and_mixes_it_under_a_clip(monkeypatch):
    sd = _fake_sd()
    sd.OutputStream = PacedOutputStream
    monkeypatch.setitem(sys.modules, "sounddevice", sd)
    rate = 16000
    bed = AmbientBed(np.full(rate, 0.4, np.float32), rate, volume_db=0.0, fade_s=0.001)
    player = audio.Player(None, bed=bed)
    player.open(rate)
    (stream,) = FakeOutputStream.opened
    deadline = threading.Event()
    while len(stream.writes) < 3 and not deadline.wait(0.01):
        pass
    assert len(stream.writes) >= 3  # the bed plays with no clip
    clip = np.full(int(0.2 * rate), 0.2, np.float32)
    player.play(clip, rate, lambda _: None)
    player.close()
    out = np.concatenate([w[:, 0] for w in stream.writes])
    assert np.isclose(out, 0.6).sum() >= len(clip)  # the clip on top of the bed
    assert np.isclose(out, 0.4).sum() > 0  # bed alone before and after it
    assert not PacedOutputStream.overlap  # never two writers at once
    assert stream.events[-2:] == ["stop", "close"]
    assert player._bed_thread is None


def test_player_without_a_bed_writes_only_clips(monkeypatch):
    monkeypatch.setitem(sys.modules, "sounddevice", _fake_sd())
    player = audio.Player(None)
    player.open(16000)
    (stream,) = FakeOutputStream.opened
    assert stream.writes == [] and player._bed_thread is None
