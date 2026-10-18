import dataclasses
import os
import random
import statistics
import time
from pathlib import Path

import numpy as np
import pytest

from mirror.face.bank import load_bank
from mirror.face.sprite import SpriteFace
from mirror.states import State

STUB = Path(__file__).resolve().parent.parent / "assets" / "face" / "stub"
DT = 1.0 / 30


@pytest.fixture(scope="module")
def bank():
    return load_bank(str(STUB))


def _face(bank):
    return SpriteFace(bank, rng=random.Random(1))


def _settle(face, state=State.IDLE, mouth=0.0, viseme="X", frames=15):
    for _ in range(frames):
        face.update(DT, state, mouth, viseme)


def _crop(img, box):
    x, y, w, h = box
    return img[y:y + h, x:x + w].astype(np.float32)


def test_render_shape_dtype(bank):
    face = _face(bank)
    face.update(DT, State.IDLE, 0.0, "X")
    img = face.render()
    assert img.shape == (252, 252, 3) and img.dtype == np.uint8


def test_background_stays_black(bank):
    face = _face(bank)
    for state, mouth, vis in ((State.IDLE, 0.0, "X"), (State.LISTENING, 0.0, "X"),
                              (State.THINKING, 0.0, "X"), (State.SPEAKING, 1.0, "D")):
        _settle(face, state, mouth, vis)
        img = face.render()
        for corner in (img[:4, :4], img[:4, -4:], img[-4:, :4], img[-4:, -4:]):
            assert corner.max() == 0


def test_mouth_patch_changes_mouth_region(bank):
    face_x, face_d = _face(bank), _face(bank)
    _settle(face_x, State.SPEAKING, 0.3, "X")
    _settle(face_d, State.SPEAKING, 0.3, "D")
    diff = np.abs(_crop(face_d.render(), bank.mouth_box) - _crop(face_x.render(), bank.mouth_box))
    assert diff.mean() > 10


def test_blink_closes_eyes(bank):
    face = _face(bank)
    _settle(face, State.SPEAKING)  # speaking: gaze centred, no wander
    open_eye = _crop(face.render(), bank.eye_box_left).mean()
    face.blink()
    face.update(0.075, State.SPEAKING, 0.0, "X")  # mid-blink: closed
    closed_eye = _crop(face.render(), bank.eye_box_left).mean()
    assert closed_eye < open_eye - 1.0


def _lower_centroid_y(img):
    lum = img.astype(np.float32).mean(axis=2)
    lum[: 2 * 252 // 3] = 0
    ys = np.arange(252, dtype=np.float32)[:, None]
    return float((lum * ys).sum() / lum.sum())


def test_jaw_warp_moves_chin(bank):
    # same mouth level and viseme (so the same patch) with and without the jaw
    # field: any difference in the chin is the warp's alone
    no_jaw = dataclasses.replace(bank, warps={**bank.warps,
                                              "jaw": np.zeros_like(bank.warps["jaw"])})
    still, moved = _face(no_jaw), _face(bank)
    _settle(still, State.SPEAKING, 1.0, "C")
    _settle(moved, State.SPEAKING, 1.0, "C")
    assert _lower_centroid_y(moved.render()) - _lower_centroid_y(still.render()) >= 2.0


def test_crossfade_two_frames(bank):
    face = _face(bank)
    _settle(face, State.SPEAKING, 0.3, "X")
    x_img = _crop(face.render(), bank.mouth_box)
    face.update(DT, State.SPEAKING, 0.3, "D")
    mid = _crop(face.render(), bank.mouth_box)
    face.update(DT, State.SPEAKING, 0.3, "D")
    mid2 = _crop(face.render(), bank.mouth_box)
    face.update(DT, State.SPEAKING, 0.3, "D")
    d_img = _crop(face.render(), bank.mouth_box)
    for m in (mid, mid2):
        assert np.abs(m - x_img).mean() > 1.0
        assert np.abs(m - d_img).mean() > 1.0
    # the crossfade is over after two in-between frames
    face.update(DT, State.SPEAKING, 0.3, "D")
    assert np.abs(_crop(face.render(), bank.mouth_box) - d_img).mean() < 1.0


@pytest.mark.skipif(bool(os.environ.get("MIRROR_SKIP_PERF")), reason="MIRROR_SKIP_PERF set")
def test_render_under_5ms(bank):
    face = _face(bank)
    _settle(face, State.SPEAKING, 0.8, "D")
    times = []
    for i in range(100):
        face.update(DT, State.SPEAKING, 0.8, "CD"[i % 2])
        t0 = time.perf_counter()
        face.render()
        times.append(time.perf_counter() - t0)
    assert statistics.median(times) < 0.005


def test_smirk_rescheduled_while_not_idle(bank):
    f = SpriteFace(bank, random.Random(3))
    f.auto_blink = False
    for _ in range(600):  # 20 s of conversation at 30 fps
        f.update(1 / 30, State.SPEAKING, 0.0, "X")
    assert f._next_smirk > f._t + 15.0
