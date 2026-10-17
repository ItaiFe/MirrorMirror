import math
import sys
import types

import numpy as np

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


def _fake_sd(devices=None):
    sd = types.ModuleType("sounddevice")
    sd.play = lambda *a, **k: None
    sd.wait = lambda: None
    sd.query_devices = lambda: devices or []
    return sd


def test_player_reports_levels(monkeypatch):
    monkeypatch.setitem(sys.modules, "sounddevice", _fake_sd())
    now = [0.0]
    sleeps = []

    def fake_sleep(s):
        sleeps.append(s)
        now[0] += s

    monkeypatch.setattr(audio.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(audio.time, "sleep", fake_sleep)
    sr = 16000
    dur = 0.5
    x = (np.random.default_rng(0).standard_normal(int(dur * sr)) * 0.1).astype(np.float32)
    levels = []
    audio.Player(None).play(x, sr, levels.append)
    assert len(levels) == math.ceil(dur / 0.02) + 1
    assert levels[-1] == 0.0
    assert now[0] < dur + 0.05  # paced against the clock, no drift


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
