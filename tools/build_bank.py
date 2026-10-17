#!/usr/bin/env python3
"""Build the real face bank from LivePortrait exports (see docs/FACE_ASSETS.md).

    python tools/build_bank.py RAW_DIR OUT_DIR [--box x,y,w,h] [--points points.json]

RAW_DIR holds full frames, all rendered from the same source image:
  hub.png                 neutral face (closed mouth, eyes open)
  lip_00.png .. lip_04.png  lip-open ladder, closed .. widest
  eye_00.png .. eye_03.png  eyes open, half, closed, looking up
  shape_E.png .. shape_H.png  optional: rounded, puckered, f/v, l mouths
  expr_<name>.png         optional: whole-face expressions (smirk, squint, brows_up)

Every frame is aligned to the hub (phase correlation, refined with ECC to
sub-pixel), cropped like the hub and scaled to 252. Mouth/eye boxes come
from the points layout; each patch is colour-matched to the hub and the
mouth interior (pixels darker than CAVITY_LUM) is painted pure black.
"""
import argparse
import glob
import os
from typing import Optional

import cv2
import numpy as np

import stub_bank as sb

CAVITY_LUM = 24  # 0..255: darker than this around the lips = cavity -> pure black (lip shadows on the bronze reach ~30)
EXPR_DIFF = 12  # 0..255: expression alpha covers pixels that changed more than this

# lip ladder frame for each mouth patch
LIP_LADDER = {"A_mild": 0, "A_wide": 0, "B_mild": 1, "B_wide": 2,
              "C_mild": 2, "C_wide": 3, "D_mild": 3, "D_wide": 4}
EYE_FRAMES = {"open": 0, "half": 1, "closed": 2, "look_up": 3}
EXTRA_SHAPES = ("E", "F", "G", "H")


def align(ref: np.ndarray, img: np.ndarray, ignore: Optional[np.ndarray] = None) -> np.ndarray:
    """Translate `img` (RGB) onto `ref` (RGB, same size), sub-pixel.

    `ignore` (uint8 0/1, ref-sized) marks pixels left out of the match, e.g.
    the mouth, which legitimately differs between frames.
    """
    if img.shape != ref.shape:
        img = cv2.resize(img, ref.shape[1::-1], interpolation=cv2.INTER_AREA)
    a = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY).astype(np.float32)
    b = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    win = cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F)
    (dx, dy), _ = cv2.phaseCorrelate(a, b, win)
    warp = np.float32([[1, 0, dx], [0, 1, dy]])
    mask = None if ignore is None else (1 - ignore).astype(np.uint8)
    try:
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5)
        _, warp = cv2.findTransformECC(a, b, warp, cv2.MOTION_TRANSLATION, crit, mask, 5)
    except cv2.error:
        pass  # keep the phase-correlation estimate
    return cv2.warpAffine(img, warp, ref.shape[1::-1],
                          flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def match_colour(patch: np.ndarray, ref: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Per-channel mean/std transfer (float 0..1 RGB) over pixels lit in both."""
    lit = (weight > 0.5) & (patch.max(-1) > CAVITY_LUM / 255) & (ref.max(-1) > CAVITY_LUM / 255)
    if lit.sum() < 10:
        return patch
    out = patch.copy()
    for c in range(3):
        p, r = patch[..., c][lit], ref[..., c][lit]
        out[..., c] = (patch[..., c] - p.mean()) / max(p.std(), 1e-3) * r.std() + r.mean()
    return np.clip(out, 0, 1)


def _cavity_zone(layout: sb.Layout, box: sb.Box) -> np.ndarray:
    """Ellipse around the lips (box-sized bool) where dark pixels count as cavity."""
    p = layout.points
    x, y, w, h = box
    cx = (p["mouth_l"][0] + p["mouth_r"][0]) / 2
    cy = (p["lip_upper"][1] + p["lip_lower"][1]) / 2 + 2
    rx = (p["mouth_r"][0] - p["mouth_l"][0]) / 2 + 3
    ry = max(p["chin"][1] - p["lip_upper"][1], 8)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32) + 0.5
    return ((xs + x - cx) / rx) ** 2 + ((ys + y - cy) / ry) ** 2 <= 1


def cut_patch(face: np.ndarray, hub: np.ndarray, mask: np.ndarray, box: sb.Box,
              cavity: Optional[np.ndarray] = None) -> np.ndarray:
    """RGBA patch from an aligned 252 frame: feathered, colour-matched, cavity black."""
    x, y, w, h = box
    rgb = face[y:y + h, x:x + w].astype(np.float32) / 255
    ref = hub[y:y + h, x:x + w].astype(np.float32) / 255
    alpha = sb._box_feather(w, h, 5) * mask[y:y + h, x:x + w]
    # the cavity is judged on the frame as rendered, before colour matching
    dark = None if cavity is None else cavity & (rgb.max(-1) * 255 < CAVITY_LUM)
    rgb = match_colour(rgb, ref, alpha)
    if dark is not None:
        rgb[dark] = 0.0
        alpha = np.where(dark, 1.0, alpha)
    return sb._rgba(rgb, alpha)


def expression(face: np.ndarray, hub: np.ndarray, mask: np.ndarray) -> np.ndarray:
    diff = np.abs(face.astype(np.int16) - hub.astype(np.int16)).max(-1)
    changed = (diff > EXPR_DIFF).astype(np.uint8)
    changed = cv2.dilate(changed, np.ones((5, 5), np.uint8))
    alpha = cv2.GaussianBlur(changed.astype(np.float32), (0, 0), 2.0) * mask
    return sb._rgba(face.astype(np.float32) / 255, alpha)


def build_bank(raw_dir: str, out_dir: str, box: Optional[sb.Box] = None,
               points: Optional[str] = None) -> str:
    def raw(name):
        return sb.read_rgb(os.path.join(raw_dir, name))

    hub_full = raw("hub.png")
    box = box or sb.centre_box(hub_full.shape)
    layout = sb.load_layout(points)
    hub, mask = sb.hub_and_mask(sb.crop_to_size(hub_full, box))
    m_box = sb.mouth_box(layout)
    e_boxes = sb.eye_boxes(layout)

    # leave the mouth out of the alignment match (it moves from frame to frame)
    k = box[2] / sb.SIZE
    ignore = np.zeros(hub_full.shape[:2], np.uint8)
    mx, my, mw, mh = m_box
    ignore[int(box[1] + my * k):int(box[1] + (my + mh) * k),
           int(box[0] + mx * k):int(box[0] + (mx + mw) * k)] = 1

    def frame(name):
        return sb.crop_to_size(align(hub_full, raw(name), ignore), box)

    zone = _cavity_zone(layout, m_box)
    lips = {i: frame(f"lip_{i:02d}.png") for i in sorted(set(LIP_LADDER.values()))}
    mouth = {key: cut_patch(lips[i], hub, mask, m_box, zone) for key, i in LIP_LADDER.items()}
    for s in EXTRA_SHAPES:
        if os.path.isfile(os.path.join(raw_dir, f"shape_{s}.png")):
            mouth[f"{s}_mild"] = cut_patch(frame(f"shape_{s}.png"), hub, mask, m_box, zone)

    # one patch serves both eyes; it is cut from the left eye
    eyes = {name: cut_patch(frame(f"eye_{i:02d}.png"), hub, mask, e_boxes[0])
            for name, i in EYE_FRAMES.items()}

    expressions = {}
    for path in sorted(glob.glob(os.path.join(raw_dir, "expr_*.png"))):
        name = os.path.basename(path)[len("expr_"):-len(".png")]
        expressions[name] = expression(frame(os.path.basename(path)), hub, mask)

    return sb.write_bank(out_dir, hub, mask, m_box, mouth, e_boxes, eyes,
                         expressions, sb.warp_fields(layout))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("raw_dir", help="directory of LivePortrait exports (see docstring)")
    ap.add_argument("out_dir", help="bank directory to write")
    ap.add_argument("--box", type=sb.parse_box, help="crop x,y,w,h in source pixels (default: centre square)")
    ap.add_argument("--points", help="control points JSON (default: generic layout)")
    a = ap.parse_args(argv)
    print(build_bank(a.raw_dir, a.out_dir, a.box, a.points))


if __name__ == "__main__":
    main()
