import json

import numpy as np
import pytest
from PIL import Image

from mirror.face.bank import BankError, load_bank

S = 252


def _png(path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def make_bank(root, patches=("C_mild",), size=S, eyes=True):
    _png(root / "hub.png", np.full((size, size, 3), 100, np.uint8))
    _png(root / "mask.png", np.full((size, size), 255, np.uint8))
    manifest = {
        "size": S,
        "hub": "hub.png",
        "mask": "mask.png",
        "mouth": {"box": [90, 150, 40, 20], "patches": {}},
    }
    for name in patches:
        _png(root / "mouth" / f"{name}.png", np.full((20, 40, 4), 200, np.uint8))
        manifest["mouth"]["patches"][name] = f"mouth/{name}.png"
    if eyes:
        manifest["eyes"] = {
            "box_left": [60, 90, 30, 16],
            "box_right": [160, 90, 30, 16],
            "patches": {},
        }
        _png(root / "eyes/open.png", np.full((16, 30, 4), 255, np.uint8))
        manifest["eyes"]["patches"]["open"] = "eyes/open.png"
    (root / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_load_minimal_bank(tmp_path):
    make_bank(tmp_path)
    bank = load_bank(str(tmp_path))
    assert bank.size == S
    assert bank.hub.shape == (S, S, 3) and bank.hub.dtype == np.uint8
    assert bank.mask.shape == (S, S) and bank.mask.dtype == np.float32
    assert bank.mask.max() == pytest.approx(1.0)
    assert bank.mouth_box == (90, 150, 40, 20)
    p = bank.mouth_patches["C_mild"]
    assert p.shape == (20, 40, 4) and p.dtype == np.float32
    assert 0.0 <= p.min() and p.max() <= 1.0
    assert bank.eye_patch("closed") is bank.eye_patches["open"]


def test_mouth_fallbacks(tmp_path):
    make_bank(tmp_path)
    bank = load_bank(str(tmp_path))
    c = bank.mouth_patches["C_mild"]
    assert bank.mouth_patch("E", True) is c
    assert bank.mouth_patch("H", False) is c
    assert bank.mouth_patch("C", True) is c
    assert bank.mouth_patch("X", False) is None
    assert bank.mouth_patch("A", False) is None


def test_bad_size_raises(tmp_path):
    make_bank(tmp_path, size=100)
    with pytest.raises(BankError, match="hub.png"):
        load_bank(str(tmp_path))


def test_missing_file_names_file(tmp_path):
    make_bank(tmp_path)
    (tmp_path / "mouth" / "C_mild.png").unlink()
    with pytest.raises(BankError, match="C_mild.png"):
        load_bank(str(tmp_path))


def test_optional_sections_and_warps(tmp_path):
    m = make_bank(tmp_path, eyes=False)
    np.save(tmp_path / "jaw.npy", np.zeros((S, S, 2), np.float32))
    np.save(tmp_path / "idle.npy", np.zeros((2, S, S, 3), np.uint8))
    m["warps"] = {"jaw": "jaw.npy"}
    m["idle_loops"] = ["idle.npy"]
    (tmp_path / "manifest.json").write_text(json.dumps(m))
    bank = load_bank(str(tmp_path))
    assert bank.warps["jaw"].shape == (S, S, 2)
    assert bank.idle_loops[0].shape == (2, S, S, 3)
    assert bank.eye_patches == {}

    np.save(tmp_path / "jaw.npy", np.zeros((S, S, 3), np.float32))
    with pytest.raises(BankError, match="jaw.npy"):
        load_bank(str(tmp_path))


@pytest.mark.parametrize("mutate", [
    lambda m: m["eyes"].pop("box_left"),
    lambda m: m.update(eyes={"box_left": [60, 90, 30, 16], "box_right": [160, 90, 30, 16],
                             "patches": ["eyes/open.png"]}),
    lambda m: m.update(expressions=["smirk.png"]),
    lambda m: m.update(warps="jaw.npy"),
    lambda m: m.update(idle_loops="loop.npy"),
])
def test_malformed_optional_sections_raise_bank_error(tmp_path, mutate):
    m = make_bank(tmp_path)
    mutate(m)
    (tmp_path / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(BankError, match="manifest.json"):
        load_bank(str(tmp_path))


@pytest.mark.parametrize("section,box", [
    ("mouth", [230, 150, 40, 20]),   # runs off the right edge
    ("mouth", [-1, 150, 40, 20]),    # negative origin
    ("mouth", [90, 150, 0, 20]),     # empty
    ("eyes", [60, 240, 30, 16]),     # runs off the bottom
])
def test_box_outside_the_face_raises_bank_error(tmp_path, section, box):
    m = make_bank(tmp_path)
    if section == "mouth":
        m["mouth"]["box"] = box
    else:
        m["eyes"]["box_left"] = box
        m["eyes"]["box_right"] = [160] + box[1:]
    (tmp_path / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(BankError, match="box"):
        load_bank(str(tmp_path))
