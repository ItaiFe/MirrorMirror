"""tools/record_demo.py: the timeline clock, the fake player and the face sink."""
import os
import sys

import numpy as np

from mirror.states import State

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "tools"))

import record_demo as rd  # noqa: E402

SR = 1000


class FakeWall:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def _timeline(cap_s=6.0):
    wall = FakeWall()
    return rd.Timeline(sr=SR, cap_s=cap_s, wall=wall), wall


def test_clock_is_frozen_outside_an_anchor():
    tl, wall = _timeline()
    tl.silence(1.0)
    wall.t += 50.0
    assert tl.now() == 1.0


def test_wait_becomes_silence_before_the_clip_and_levels_follow_the_clip():
    tl, wall = _timeline()
    tl.silence(1.0)
    face = rd.TimelineFace(tl)
    clip = np.ones(int(0.1 * SR), np.float32) * 0.5
    player = rd.TimelinePlayer(tl, {id(clip): ("reply", "hello")})
    tl.anchor()
    wall.t += 2.0  # STT + LLM
    face.send(State.THINKING, 0.0)
    player.play(clip, SR, lambda level: face.send(State.SPEAKING, level, "B"))
    tl.release()

    assert tl.gaps == [{"at": 1.0, "label": "reply 1", "seconds": 2.0}]
    assert tl.capped == []
    assert player.clips == [{"start": 3.0, "end": 3.1, "kind": "reply", "text": "hello"}]
    assert tl.cursor == 3.1
    assert face.events[0] == (3.0, "thinking", 0.0, "X")
    speaking = face.events[1:]
    assert [e[0] for e in speaking[:3]] == [3.0, 3.02, 3.04]  # 20 ms hops
    assert speaking[-1] == (3.1, "speaking", 0.0, "B")  # the final on_level(0.0)
    audio = tl.audio()
    assert len(audio) == int(3.1 * SR)
    assert not audio[:3000].any() and audio[3000:].min() > 0.4


def test_a_clip_ready_during_the_previous_one_follows_it_without_a_gap():
    tl, wall = _timeline()
    player = rd.TimelinePlayer(tl)
    tl.anchor()
    player.play(np.zeros(2 * SR, np.float32), SR, lambda _: None)  # a 2 s filler
    wall.t += 0.5  # the next sentence was ready long before the filler ended
    assert tl.now() == 2.0
    player.play(np.zeros(SR, np.float32), SR, lambda _: None)
    assert [c["start"] for c in player.clips] == [0.0, 2.0]
    assert tl.gaps == []


def test_long_waits_are_capped_and_later_waits_measured_from_the_cap():
    tl, wall = _timeline(cap_s=6.0)
    player = rd.TimelinePlayer(tl)
    tl.anchor()
    wall.t += 10.0
    assert tl.now() == 6.0  # the clock never runs past the cap either
    player.play(np.zeros(SR, np.float32), SR, lambda _: None)
    assert tl.capped == [{"at": 0.0, "label": "clip 1", "measured_s": 10.0, "kept_s": 6.0}]
    assert player.clips[0]["start"] == 6.0
    wall.t += 3.0  # 1 s of clip, then 2 s more waiting
    player.play(np.zeros(SR, np.float32), SR, lambda _: None)
    assert player.clips[1]["start"] == 9.0


def test_face_message_at_returns_only_new_events():
    events = [(0.0, "idle", 0.0, "X"), (1.0, "thinking", 0.0, "X"),
              (1.01, "speaking", 0.3, "B")]
    msg, i = rd.face_message_at(events, 0.5)
    assert msg[1] == "idle" and i == 1
    msg, i = rd.face_message_at(events, 0.9, i)
    assert msg is None and i == 1
    msg, i = rd.face_message_at(events, 1.5, i)
    assert msg[1] == "speaking" and i == 3


def test_demo_face_uses_the_floating_face_params_from_config():
    from mirror.config import load_config
    from mirror.face.bank import load_bank
    from mirror.face.sprite import FloatParams

    cfg = load_config(os.path.join(os.path.dirname(__file__), os.pardir, "config", "orin.toml"))
    bank_dir = os.path.join(os.path.dirname(__file__), os.pardir, cfg.face.bank)
    face = rd.make_face(load_bank(bank_dir), cfg)
    assert face.float_params == FloatParams.from_config(cfg.face)
