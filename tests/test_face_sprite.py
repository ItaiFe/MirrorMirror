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


# -- Task 27: floating face ------------------------------------------------------

from mirror.face.sprite import FloatParams  # noqa: E402


def _float_face(bank, **kw):
    f = SpriteFace(bank, rng=random.Random(1), float_params=FloatParams(**kw))
    f.auto_blink = False
    return f


def _centroid(img):
    lum = img.astype(np.float32).mean(axis=2)
    ys, xs = np.mgrid[0:252, 0:252].astype(np.float32)
    return float((lum * xs).sum() / lum.sum()), float((lum * ys).sum() / lum.sum())


def test_float_disabled_reproduces_todays_frames(bank):
    seq = ([(State.IDLE, 0.0, "X")] * 40 + [(State.THINKING, 0.0, "X")] * 15
           + [(State.SPEAKING, 0.9, "D")] * 15 + [(State.IDLE, 0.0, "X")] * 10)
    plain = SpriteFace(bank, rng=random.Random(4))
    off = SpriteFace(bank, rng=random.Random(4), float_params=FloatParams(enabled=False))
    for s, m, v in seq:
        plain.update(DT, s, m, v)
        off.update(DT, s, m, v)
        assert np.array_equal(plain.render(), off.render())


def test_float_bobs_sways_and_is_smooth(bank):
    face = _float_face(bank)
    cs = []
    for _ in range(int(6.5 * 1.37 * 30) + 1):  # one full sway period
        face.update(DT, State.SPEAKING, 0.0, "X")  # gaze centred: motion is the float's
        cs.append(_centroid(face.render()))
    cs = np.array(cs)
    span_x, span_y = np.ptp(cs[:, 0]), np.ptp(cs[:, 1])
    assert span_y > 2.0  # 2 x 5 px x 0.45 (speaking) x 0.93, minus breathe/blur
    assert span_x > 0.5
    assert np.abs(np.diff(cs, axis=0)).max() < 0.5  # no jumps between frames


def test_float_amplitude_follows_state_without_jumps(bank):
    def bob_span(state):
        face = _float_face(bank)
        for _ in range(60):  # let the 600 ms spring settle
            face.update(DT, state, 0.0, "X")
        ys = []
        for _ in range(int(6.5 * 30)):
            face.update(DT, state, 0.0, "X")
            ys.append(face.float_offset()[1])
        return np.ptp(ys)

    idle, speaking = bob_span(State.IDLE), bob_span(State.SPEAKING)
    assert idle == pytest.approx(2 * 5.0, rel=0.05)
    assert speaking == pytest.approx(0.45 * idle, rel=0.08)
    # a state change only moves the target amplitude: the offset stays continuous
    face = _float_face(bank)
    prev = None
    for i in range(240):
        face.update(DT, State.IDLE if (i // 40) % 2 == 0 else State.SPEAKING, 0.0, "X")
        off = np.array(face.float_offset())
        if prev is not None:
            assert np.abs(off - prev).max() < 0.4
        prev = off


def test_float_never_clips_at_max_jaw(bank):
    face = _float_face(bank)
    worst = 252
    # one full bob period, sampled densely, IDLE amplitude (the largest) with the jaw
    # fully open, at several phases of the sway and tilt periods
    for _ in range(int(3 * 6.5 * 1.37 * 10)):
        face.update(0.1, State.IDLE, 1.0, "D")
        img = face.render()
        ys, xs = np.nonzero(img.max(axis=2) > 0)
        worst = min(worst, ys.min(), xs.min(), 251 - ys.max(), 251 - xs.max())
    assert worst >= 2


def test_float_edge_feather_is_monotonic(bank):
    plain, feathered = _face(bank)._mask, _float_face(bank)._mask
    assert feathered.max() == pytest.approx(255.0)
    assert np.all(feathered <= plain + 1e-3)  # only fades the edge, never adds light
    # going inward from the outside: down the crown's centre line, up from the chin,
    # in from the left cheek
    for prof_f, prof_p in ((feathered[:60, 126], plain[:60, 126]),
                           (feathered[:150:-1, 126], plain[:150:-1, 126]),
                           (feathered[150, :120], plain[150, :120])):
        assert np.all(np.diff(prof_f) >= -1e-3)  # alpha rises monotonically inward
        assert np.count_nonzero((prof_f > 2) & (prof_f < 250)) >= 5  # a soft ramp, no step
        # full alpha only several px further in than in the bank's mask
        assert np.argmax(prof_f >= 250) >= np.argmax(prof_p >= 250) + 3


@pytest.mark.skipif(bool(os.environ.get("MIRROR_SKIP_PERF")), reason="MIRROR_SKIP_PERF set")
def test_render_under_5ms_floating(bank):
    face = _float_face(bank)
    _settle(face, State.SPEAKING, 0.8, "D")
    times = []
    for i in range(100):
        face.update(DT, State.SPEAKING, 0.8, "CD"[i % 2])
        t0 = time.perf_counter()
        face.render()
        times.append(time.perf_counter() - t0)
    assert statistics.median(times) < 0.005


def test_float_params_from_config():
    from mirror.config import load_config

    root = Path(__file__).resolve().parent.parent
    for name in ("mac.toml", "orin.toml"):
        p = FloatParams.from_config(load_config(str(root / "config" / name)).face)
        assert p == FloatParams(enabled=True, amp_px=5.0, period_s=6.5, tilt_deg=1.2,
                                scale=0.93, edge_feather_px=6.0)
