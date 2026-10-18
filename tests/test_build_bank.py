"""Face-bank builder tools: tools/stub_bank.py and tools/build_bank.py."""
import os
import sys

import cv2
import numpy as np
import pytest

from mirror.face.bank import SIZE, load_bank

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "tools"))

import build_bank  # noqa: E402
import stub_bank  # noqa: E402

LADDER = [f"{s}_{w}" for s in "ABCD" for w in ("mild", "wide")]


def _textured_face(n, seed=0):
    """An n x n RGB 'face': smooth bronze texture inside an ellipse, black outside."""
    rng = np.random.default_rng(seed)
    noise = rng.random((n, n)).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), n / 120)
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    lum = 90 + 130 * noise  # 90..220: nothing in the face is near-black
    img = np.stack([lum, lum * 0.75, lum * 0.4], axis=-1)
    face = np.zeros((n, n), np.uint8)
    cv2.ellipse(face, (n // 2, n // 2), (int(n * 0.36), int(n * 0.46)), 0, 0, 360, 1, -1)
    return (img * face[..., None]).astype(np.uint8)


def _save(path, rgb):
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def _shift(rgb, dx, dy):
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(rgb, m, rgb.shape[1::-1], flags=cv2.INTER_LINEAR)


def _raw_dir(root, hub, frames=None):
    """RAW_DIR with every frame a copy of the hub unless overridden in `frames`."""
    frames = frames or {}
    root.mkdir(parents=True, exist_ok=True)
    names = ["hub"] + [f"lip_{i:02d}" for i in range(5)] + [f"eye_{i:02d}" for i in range(4)]
    for name in names + [n for n in frames if n not in names]:
        _save(root / f"{name}.png", frames.get(name, hub))
    return root


def test_stub_bank_roundtrip(tmp_path):
    src = tmp_path / "face.png"
    _save(src, _textured_face(400))
    out = tmp_path / "bank"
    stub_bank.build_stub_bank(str(src), str(out))

    bank = load_bank(str(out))
    for key in LADDER:
        assert key in bank.mouth_patches
    assert {"E_mild", "F_mild"} <= set(bank.mouth_patches)
    assert {"open", "half", "closed"} <= set(bank.eye_patches)
    assert {"jaw", "brows", "breathe", "gaze_x", "gaze_y"} <= set(bank.warps)
    for field in bank.warps.values():
        assert field.shape == (SIZE, SIZE, 2) and field.dtype == np.float32
    # the jaw opens downwards: the chin samples from above (remap grid = id + field)
    assert bank.warps["jaw"][..., 1].min() < -5
    # background stays exactly black
    assert bank.hub[0, 0].tolist() == [0, 0, 0] and bank.mask[0, 0] == 0
    # open shapes have a pure-black cavity
    assert (bank.mouth_patches["D_wide"][..., :3].max(axis=-1) == 0).sum() > 20


def test_build_bank_aligns(tmp_path):
    hub = _textured_face(504, seed=1)
    raw = _raw_dir(tmp_path / "raw", hub, {"lip_00": _shift(hub, 7.0, -5.0)})
    out = tmp_path / "bank"
    build_bank.build_bank(str(raw), str(out))

    bank = load_bank(str(out))
    x, y, w, h = bank.mouth_box
    ref = cv2.cvtColor(bank.hub[y:y + h, x:x + w], cv2.COLOR_RGB2GRAY).astype(np.float64)
    patch = (bank.mouth_patches["A_mild"][..., :3] * 255).astype(np.uint8)
    got = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY).astype(np.float64)
    (dx, dy), _ = cv2.phaseCorrelate(ref, got)
    assert abs(dx) < 1 and abs(dy) < 1
    # without alignment the patch would be off by (3.5, -2.5) px at 252
    assert np.abs(got - ref).mean() < 3


def test_black_cavity(tmp_path):
    hub = _textured_face(504, seed=2)
    pts = stub_bank.load_layout(None).points
    k = 504 / SIZE
    open_mouth = hub.copy()
    cx, cy = pts["lip_lower"][0] * k, (pts["lip_upper"][1] + 3) * k
    cv2.ellipse(open_mouth, (int(cx), int(cy)), (40, 14), 0, 0, 360, (18, 12, 8), -1)
    raw = _raw_dir(tmp_path / "raw", hub, {"lip_04": open_mouth})
    out = tmp_path / "bank"
    build_bank.build_bank(str(raw), str(out))

    bank = load_bank(str(out))
    patch = bank.mouth_patches["D_wide"][..., :3]
    lum = patch.max(axis=-1)
    darkest = lum <= np.percentile(lum, 5)
    assert darkest.sum() > 0
    assert lum[darkest].max() == 0.0


def test_build_bank_extra_shapes_and_expressions(tmp_path):
    hub = _textured_face(504, seed=3)
    smirk = hub.copy()
    cv2.circle(smirk, (330, 330), 30, (250, 200, 120), -1)
    raw = _raw_dir(tmp_path / "raw", hub, {"shape_E": hub, "expr_smirk": smirk})
    out = tmp_path / "bank"
    build_bank.build_bank(str(raw), str(out))

    bank = load_bank(str(out))
    assert "E_mild" in bank.mouth_patches
    smirk_rgba = bank.expressions["smirk"]
    assert smirk_rgba.shape == (SIZE, SIZE, 4)
    assert smirk_rgba[165, 165, 3] > 0.9  # the changed spot is covered...
    assert smirk_rgba[60, 126, 3] == 0.0  # ...the unchanged forehead is not


def test_build_bank_lists_all_missing_frames(tmp_path):
    raw = _raw_dir(tmp_path / "raw", _textured_face(504))
    (raw / "lip_03.png").unlink()
    (raw / "eye_01.png").unlink()
    with pytest.raises(SystemExit) as e:
        build_bank.build_bank(str(raw), str(tmp_path / "bank"))
    assert "lip_03.png" in str(e.value) and "eye_01.png" in str(e.value)
    assert not (tmp_path / "bank").exists()


def test_align_warns_on_large_shift(capsys):
    hub = _textured_face(504, seed=4)
    build_bank.align(hub, _shift(hub, 30.0, 0.0), name="lip_02.png")
    err = capsys.readouterr().err
    assert "lip_02.png" in err and "px" in err


def test_parse_box_rejects_non_square():
    import argparse
    assert stub_bank.parse_box("1,2,30,30") == (1, 2, 30, 30)
    with pytest.raises(argparse.ArgumentTypeError):
        stub_bank.parse_box("0,0,40,30")


def test_write_bank_clears_stale_files(tmp_path):
    src = tmp_path / "face.png"
    _save(src, _textured_face(400))
    out = tmp_path / "bank"
    stub_bank.build_stub_bank(str(src), str(out))
    stale = out / "mouth" / "G_mild.png"
    stale.write_bytes((out / "mouth" / "A_mild.png").read_bytes())
    (out / "expr").mkdir(exist_ok=True)
    (out / "expr" / "old.png").write_bytes(b"x")
    stub_bank.build_stub_bank(str(src), str(out))
    assert not stale.exists() and not (out / "expr").exists()
    load_bank(str(out))


def test_align_warns_when_ecc_fails(monkeypatch, capsys):
    def no_convergence(*args, **kwargs):
        raise cv2.error("findTransformECC: the algorithm stopped before convergence")

    monkeypatch.setattr(build_bank.cv2, "findTransformECC", no_convergence)
    hub = _textured_face(504, seed=5)
    out = build_bank.align(hub, _shift(hub, 2.0, 1.0), name="lip_04.png")
    err = capsys.readouterr().err
    assert "lip_04.png" in err and "ECC" in err and "coarse" in err
    assert out.shape == hub.shape  # still aligned, with the phase-correlation estimate
