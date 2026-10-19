import os
import random
import statistics
import time
from pathlib import Path

import numpy as np
import pytest

from mirror.face.bank import load_bank
from mirror.face.mist import Mist, MistParams, tileable_noise
from mirror.face.sprite import FloatParams, SpriteFace
from mirror.states import State

ROOT = Path(__file__).resolve().parent.parent
STUB = ROOT / "assets" / "face" / "stub"
DT = 1.0 / 30
ON = MistParams(enabled=True)


@pytest.fixture(scope="module")
def bank():
    return load_bank(str(STUB))


def _face(bank, mist=None, seed=4):
    f = SpriteFace(bank, rng=random.Random(seed), float_params=FloatParams(),
                   mist_params=mist)
    f.auto_blink = False
    return f


def test_mist_off_reproduces_todays_frames(bank):
    seq = ([(State.IDLE, 0.0, "X")] * 30 + [(State.SPEAKING, 0.9, "D")] * 15
           + [(State.IDLE, 0.0, "X")] * 10)
    plain, off = _face(bank), _face(bank, MistParams(enabled=False))
    assert off._mist is None
    for s, m, v in seq:
        plain.update(DT, s, m, v)
        off.update(DT, s, m, v)
        assert np.array_equal(plain.render(), off.render())


def test_noise_is_tileable_and_normalised():
    n = tileable_noise(252, np.random.default_rng(0))
    assert n.shape == (252, 252) and n.dtype == np.float32
    assert abs(float(n.mean())) < 1e-3 and float(n.std()) == pytest.approx(1.0, rel=1e-3)
    # the wrap-around seam is no steeper than the steps inside the field
    inner = np.abs(np.diff(n, axis=1)).max()
    assert np.abs(n[:, 0] - n[:, -1]).max() <= inner * 1.01
    assert np.abs(n[0] - n[-1]).max() <= np.abs(np.diff(n, axis=0)).max() * 1.01


def test_mist_lights_the_background_faintly(bank):
    plain, misty = _face(bank), _face(bank, ON)
    for _ in range(30):
        plain.update(DT, State.IDLE, 0.0, "X")
        misty.update(DT, State.IDLE, 0.0, "X")
    a, b = plain.render(), misty.render()
    bg = a.max(axis=2) == 0  # black without the mist: outside the face
    lit = b[bg].max(axis=1) > 0
    assert lit.mean() > 0.15  # a real haze, not a few specks...
    assert lit.mean() < 0.9   # ...with black gaps (and black corners)
    peak = 0.10 * 255
    assert b[bg].max() <= peak + 1
    # old gold: red over green over blue
    r, g, bl = (b[bg][:, i].astype(float).sum() for i in range(3))
    assert r > g > bl > 0
    for corner in (b[:4, :4], b[:4, -4:], b[-4:, :4], b[-4:, -4:]):
        assert corner.max() == 0


def test_mist_leaves_the_face_opaque(bank):
    # unfloated, so the mask lines up with the output (the warps move it only a few px)
    plain = SpriteFace(bank, rng=random.Random(4), float_params=FloatParams(enabled=False))
    misty = SpriteFace(bank, rng=random.Random(4), float_params=FloatParams(enabled=False),
                       mist_params=MistParams(enabled=True, intensity=1.0))
    plain.update(DT, State.IDLE, 0.0, "X")
    misty.update(DT, State.IDLE, 0.0, "X")
    a, b = plain.render().astype(int), misty.render().astype(int)
    assert np.abs(a - b).max() > 100  # the full-brightness mist is really there...
    inner = misty._mask >= 255.0  # ...but not inside the face (full alpha)
    inner = inner & np.roll(inner, 8, 0) & np.roll(inner, -8, 0) & np.roll(inner, 8, 1) \
        & np.roll(inner, -8, 1)
    assert inner.sum() > 10000
    assert np.abs(a[inner] - b[inner]).max() <= 1
    # the mist only adds light
    assert np.all(b >= a - 1)


def test_feathered_edge_blends_into_the_mist(bank):
    plain, misty = _face(bank), _face(bank, MistParams(enabled=True, intensity=1.0))
    plain.update(DT, State.IDLE, 0.0, "X")
    misty.update(DT, State.IDLE, 0.0, "X")
    a, b = plain.render().astype(int), misty.render().astype(int)
    level = misty._mist.level(misty._t)
    rgb = misty._mist.rgb
    # where the face is partly transparent (the dissolving edge), face + mist x (1 - a):
    # never brighter than the face plus the full mist there
    assert np.all(b <= a + level[..., None] * rgb + 1)
    assert np.all(b >= a - 1)


def test_mist_peak_follows_intensity_and_colour():
    for intensity in (0.05, 0.10, 0.3):
        m = Mist(MistParams(enabled=True, intensity=intensity), 252)
        assert m.rgb.max() == pytest.approx(intensity * 255.0)
        levels = np.stack([m.level(t) for t in range(0, 120, 5)])
        assert levels.min() >= 0.0 and levels.max() <= 1.0
        assert levels.max() > 0.9  # the densest wisps do reach the peak now and then
    m = Mist(MistParams(enabled=True, color=(150, 110, 45)), 252)
    assert m.rgb / m.rgb.max() == pytest.approx(np.array([150, 110, 45]) / 150.0)


def test_mist_drifts_slowly_and_smoothly():
    m = Mist(ON, 252)
    prev, deltas = m.level(0.0), []
    for i in range(1, 31):  # one second at 30 fps
        cur = m.level(i * DT)
        deltas.append(np.abs(cur - prev).max())
        prev = cur
    assert max(deltas) < 0.06  # never flickers: < 1.5 levels of 255 at 10 %
    assert max(deltas) > 0.0
    # ...but over tens of seconds it really moves
    a, b = m.level(0.0), m.level(20.0)
    assert np.abs(a - b).mean() > 0.5 * (a.mean() + b.mean()) / 2


def test_mist_speed_scales_time():
    slow, fast = Mist(MistParams(enabled=True, speed=0.5), 252), Mist(ON, 252)
    assert np.allclose(slow.level(20.0), fast.level(10.0), atol=1e-5)
    still = Mist(MistParams(enabled=True, speed=0.0), 252)
    assert np.array_equal(still.level(0.0), still.level(30.0))


def test_mist_is_deterministic_per_seed():
    a, b = Mist(ON, 252), Mist(ON, 252)
    assert np.array_equal(a.level(7.3), b.level(7.3))
    c = Mist(MistParams(enabled=True, seed=12), 252)
    assert not np.array_equal(a.level(7.3), c.level(7.3))


def test_mist_follows_the_face_clock(bank):
    f1, f2 = _face(bank, ON, seed=1), _face(bank, ON, seed=1)
    for _ in range(60):
        f1.update(DT, State.IDLE, 0.0, "X")
    f2.update(60 * DT, State.IDLE, 0.0, "X")
    assert np.array_equal(f1._mist.level(f1._t), f2._mist.level(f2._t))


@pytest.mark.skipif(bool(os.environ.get("MIRROR_SKIP_PERF")), reason="MIRROR_SKIP_PERF set")
def test_mist_costs_under_a_millisecond(bank):
    plain, misty = _face(bank), _face(bank, ON)
    times = {id(plain): [], id(misty): []}
    for i in range(150):
        for f in (plain, misty):
            f.update(DT, State.SPEAKING, 0.8, "CD"[i % 2])
            t0 = time.perf_counter()
            f.render()
            times[id(f)].append(time.perf_counter() - t0)
    extra = statistics.median(times[id(misty)]) - statistics.median(times[id(plain)])
    assert extra < 0.003  # ~0.5 ms on an M-series Mac; generous for CI noise


def test_mist_params_from_config():
    from mirror.config import load_config

    for name in ("mac.toml", "orin.toml"):
        p = MistParams.from_config(load_config(str(ROOT / "config" / name)).face)
        assert p == MistParams(enabled=True, intensity=0.10, color=(150, 110, 45), speed=1.0)
