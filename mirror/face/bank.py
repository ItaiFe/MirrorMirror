"""Face asset bank: format and loader (spec: assets/face/README.md).

Array representation (all coordinates in a SIZE x SIZE = 252 px square):
  hub          uint8   (S, S, 3)  RGB base face with the mouth/eyes neutral
  mask         float32 (S, S)     0..1 mirror-disc mask
  patches      float32 (h, w, 4)  RGBA in 0..1, straight (non-premultiplied) alpha
  expressions  float32 (S, S, 4)  RGBA in 0..1
  warps        float32 (S, S, 2)  displacement in px at weight 1.0
  idle_loops   uint8   (N, S, S, 3)
"""
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

SIZE = 252
Box = Tuple[int, int, int, int]


class BankError(Exception):
    """A bank file is missing, unreadable, or has the wrong size."""


def _read_image(path: str, mode: str) -> np.ndarray:
    from PIL import Image

    if not os.path.isfile(path):
        raise BankError(f"missing file: {os.path.basename(path)} ({path})")
    try:
        with Image.open(path) as im:
            return np.asarray(im.convert(mode))
    except Exception as e:  # corrupt / unsupported image
        raise BankError(f"cannot read {os.path.basename(path)}: {e}") from e


def _read_npy(path: str) -> np.ndarray:
    if not os.path.isfile(path):
        raise BankError(f"missing file: {os.path.basename(path)} ({path})")
    try:
        return np.load(path)
    except Exception as e:
        raise BankError(f"cannot read {os.path.basename(path)}: {e}") from e


def _check(name: str, arr: np.ndarray, shape: tuple) -> None:
    if arr.shape != shape:
        raise BankError(f"{name}: expected shape {shape}, got {arr.shape}")


def _box(manifest_box, what: str) -> Box:
    try:
        x, y, w, h = (int(v) for v in manifest_box)
    except Exception as e:
        raise BankError(f"manifest.json: bad {what} box {manifest_box!r}") from e
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > SIZE or y + h > SIZE:
        raise BankError(f"manifest.json: {what} box {manifest_box!r} is outside the "
                        f"{SIZE} x {SIZE} face")
    return (x, y, w, h)


def _section(m: dict, key: str, kind: type):
    """An optional manifest section (empty if absent), checked to be of `kind`."""
    value = m.get(key)
    if value is None:
        return kind()
    if not isinstance(value, kind):
        raise BankError(f"manifest.json: {key} must be a {kind.__name__}, "
                        f"got {type(value).__name__}")
    return value


def _load_patches(root: str, files: dict, box: Box) -> Dict[str, np.ndarray]:
    if not isinstance(files, dict):
        raise BankError(f"manifest.json: patches must be a dict, got {type(files).__name__}")
    out = {}
    for key, rel in files.items():
        arr = _read_image(os.path.join(root, rel), "RGBA")
        _check(os.path.basename(rel), arr, (box[3], box[2], 4))
        out[key] = arr.astype(np.float32) / 255.0
    return out


@dataclass
class FaceBank:
    """Loaded face bank; see the module docstring for array dtypes/shapes."""

    size: int
    hub: np.ndarray
    mask: np.ndarray
    mouth_box: Box
    mouth_patches: Dict[str, np.ndarray]  # "A_mild", "C_wide", ...
    eye_box_left: Optional[Box] = None
    eye_box_right: Optional[Box] = None
    eye_patches: Dict[str, np.ndarray] = field(default_factory=dict)
    expressions: Dict[str, np.ndarray] = field(default_factory=dict)
    warps: Dict[str, np.ndarray] = field(default_factory=dict)
    idle_loops: List[np.ndarray] = field(default_factory=list)

    def mouth_patch(self, shape: str, wide: bool) -> Optional[np.ndarray]:
        """Patch for a viseme shape letter, applying fallbacks.

        Order: exact width, other width, then (for E/F/G/H) the same two for C.
        None means "draw nothing" (the hub's own mouth shows), which is also
        the answer for X.
        """
        widths = ("wide", "mild") if wide else ("mild", "wide")
        shapes = [shape]
        if shape in ("E", "F", "G", "H"):
            shapes.append("C")
        if shape != "X":
            for s in shapes:
                for w in widths:
                    p = self.mouth_patches.get(f"{s}_{w}")
                    if p is not None:
                        return p
        return None

    def eye_patch(self, name: str) -> Optional[np.ndarray]:
        """Eye patch by name; missing names fall back to 'open' (else None)."""
        p = self.eye_patches.get(name)
        return p if p is not None else self.eye_patches.get("open")


def load_bank(path: str) -> FaceBank:
    manifest_path = os.path.join(path, "manifest.json")
    if not os.path.isfile(manifest_path):
        raise BankError(f"missing file: manifest.json ({manifest_path})")
    try:
        with open(manifest_path) as f:
            m = json.load(f)
    except ValueError as e:
        raise BankError(f"manifest.json: invalid JSON: {e}") from e

    size = m.get("size", SIZE)
    if size != SIZE:
        raise BankError(f"manifest.json: size {size} != {SIZE}")
    try:
        hub_rel, mask_rel = m["hub"], m["mask"]
        mouth = m["mouth"]
        mouth_box = _box(mouth["box"], "mouth")
        mouth_files = mouth["patches"]
    except KeyError as e:
        raise BankError(f"manifest.json: missing key {e}") from e

    hub = _read_image(os.path.join(path, hub_rel), "RGB")
    _check(hub_rel, hub, (SIZE, SIZE, 3))
    mask = _read_image(os.path.join(path, mask_rel), "L").astype(np.float32) / 255.0
    _check(mask_rel, mask, (SIZE, SIZE))

    bank = FaceBank(
        size=SIZE,
        hub=hub,
        mask=mask,
        mouth_box=mouth_box,
        mouth_patches=_load_patches(path, mouth_files, mouth_box),
    )

    eyes = _section(m, "eyes", dict)
    if eyes:
        try:
            box_left, box_right = eyes["box_left"], eyes["box_right"]
        except KeyError as e:
            raise BankError(f"manifest.json: eyes: missing key {e}") from e
        bank.eye_box_left = _box(box_left, "eyes.box_left")
        bank.eye_box_right = _box(box_right, "eyes.box_right")
        # one patch per name, used for both eyes: both boxes share a size
        if bank.eye_box_left[2:] != bank.eye_box_right[2:]:
            raise BankError("manifest.json: eye boxes differ in size")
        bank.eye_patches = _load_patches(
            path, eyes.get("patches", {}), bank.eye_box_left
        )

    for key, rel in _section(m, "expressions", dict).items():
        arr = _read_image(os.path.join(path, rel), "RGBA")
        _check(rel, arr, (SIZE, SIZE, 4))
        bank.expressions[key] = arr.astype(np.float32) / 255.0

    for key, rel in _section(m, "warps", dict).items():
        arr = _read_npy(os.path.join(path, rel))
        _check(rel, arr, (SIZE, SIZE, 2))
        bank.warps[key] = arr.astype(np.float32, copy=False)

    for rel in _section(m, "idle_loops", list):
        arr = _read_npy(os.path.join(path, rel))
        if arr.ndim != 4 or arr.shape[1:] != (SIZE, SIZE, 3) or arr.dtype != np.uint8:
            raise BankError(
                f"{rel}: expected uint8 (N, {SIZE}, {SIZE}, 3), got {arr.dtype} {arr.shape}"
            )
        bank.idle_loops.append(arr)

    return bank
