#!/usr/bin/env python3
"""Build a working face bank from ONE front-facing image (no GPU, no AI).

    python tools/stub_bank.py SOURCE.png OUT_DIR [--box x,y,w,h] [--points points.json]

The face is cropped (centre square unless --box), scaled to 252x252 and
masked by luminance (the source must be on black). Warp fields come from
moving-least-squares (affine) over a ~25-point face layout; mouth shapes
A-D are made by tearing the mouth open along the lip line, dropping the
lower lip/chin and painting the cavity pure black (E/F squeeze C). Eye
half/closed pull the upper lid down over the eye and darken it.

Coordinates in a points file are in the 252x252 face (see
assets/face/source/trickster_points.json). The format is the one
mirror/face/bank.py loads (assets/face/README.md). build_bank.py reuses the
helpers here.

Warp fields are *backward* displacements, matching the renderer's
``cv2.remap(grid = identity + sum(weight * field))``: output pixel v shows
the source pixel at v + field(v). Weight 1.0 means: jaw open by JAW_PX at
the chin, brows lifted BROW_PX, gaze moved GAZE_PX right/down, breathe
grown BREATHE_PX at the crown.
"""
import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

SIZE = 252
LUM_THRESHOLD = 8  # 0..255; below this (outside the face) is background
JAW_PX = 16.0
BROW_PX = 4.0
GAZE_PX = 3.0
BREATHE_PX = 1.0
MLS_ALPHA = 2.0  # MLS weight falloff 1/d^(2a): higher keeps each point's pull local
SUPERSAMPLE = 4  # mouth/eye patches are drawn at 4x and downscaled (anti-aliasing)

Box = Tuple[int, int, int, int]

# A generic front face framed like the reference image (face fills the frame,
# eyes ~38 % down, mouth ~65 % down). Used when no --points file is given.
DEFAULT_POINTS = {
    "forehead_top": (126, 10), "forehead_l": (96, 52), "forehead_r": (156, 52),
    "ear_l": (38, 64), "ear_r": (214, 64), "earlobe_l": (58, 134), "earlobe_r": (194, 134),
    "brow_l": (90, 80), "brow_r": (162, 80), "eye_l": (91, 97), "eye_r": (161, 97),
    "cheek_l": (72, 120), "cheek_r": (180, 120), "nose_tip": (126, 144),
    "moustache_tip_l": (82, 148), "moustache_tip_r": (170, 148),
    "moustache_l": (102, 154), "moustache_c": (126, 157), "moustache_r": (150, 154),
    "mouth_l": (101, 162), "mouth_r": (151, 162),
    "lip_upper": (126, 168), "lip_lower": (126, 176), "chin": (126, 186),
    "jaw_l": (98, 190), "jaw_r": (154, 190), "beard_tip": (126, 206), "ornament_tip": (126, 236),
}
DEFAULT_EYE_HALF = (14, 7)

# Which points each field moves (fraction of the field's full displacement);
# every other point is pinned, so the moustache, nose, cheeks, mouth corners,
# ears and forehead stay put under the jaw, and only the brow ridges lift.
JAW_MOVERS = {"lip_line_lower": 1.0, "lip_lower": 1.0, "chin": 1.0, "beard_tip": 1.0, "ornament_tip": 1.0,
              "jaw_l": 0.6, "jaw_r": 0.6}
BROW_MOVERS = {"brow_l": 1.0, "brow_r": 1.0}
EYE_MOVERS = {"eye_l": 1.0, "eye_r": 1.0}


@dataclass
class Layout:
    points: Dict[str, Tuple[float, float]]  # name -> (x, y) in the 252 face
    eye_half: Tuple[float, float]  # half width, half height of each eye opening


def load_layout(path: Optional[str]) -> Layout:
    if path is None:
        return Layout(dict(DEFAULT_POINTS), DEFAULT_EYE_HALF)
    with open(path) as f:
        d = json.load(f)
    k = SIZE / float(d.get("size", SIZE))
    pts = {n: (float(x) * k, float(y) * k) for n, (x, y) in d["points"].items()}
    missing = set(DEFAULT_POINTS) - set(pts)
    if missing:
        raise SystemExit(f"{path}: missing points {sorted(missing)}")
    ew, eh = d.get("eye_half_size", DEFAULT_EYE_HALF)
    return Layout(pts, (ew * k, eh * k))


# ---------------------------------------------------------------- image + mask

def read_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"cannot read image {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def write_rgb(path: str, rgb: np.ndarray) -> None:
    cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def centre_box(shape) -> Box:
    h, w = shape[:2]
    n = min(h, w)
    return ((w - n) // 2, (h - n) // 2, n, n)


def crop_to_size(rgb: np.ndarray, box: Box) -> np.ndarray:
    x, y, w, h = box
    return cv2.resize(rgb[y:y + h, x:x + w], (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def face_region(rgb: np.ndarray) -> np.ndarray:
    """Binary (uint8 0/1) face silhouette: bright pixels, small specks dropped, holes filled."""
    lum = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    lit = (lum > LUM_THRESHOLD).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(lit, connectivity=8)
    keep = np.zeros_like(lit)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= 30:
            keep[labels == i] = 1
    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    # fill interior holes (pupils, deep shadows): flood the background from a border
    flood = np.pad(keep, 1)
    ff = flood.copy() * 255
    cv2.floodFill(ff, None, (0, 0), 128)
    return ((ff != 128)[1:-1, 1:-1]).astype(np.uint8)


def hub_and_mask(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Hub with the background forced to exactly 0, and a feathered 0..1 mask."""
    region = face_region(rgb)
    hub = rgb * region[..., None]
    mask = cv2.GaussianBlur(region.astype(np.float32), (0, 0), 0.8)
    return hub.astype(np.uint8), np.clip(mask, 0.0, 1.0)


# ---------------------------------------------------------------- warp fields

def mls_field(src: np.ndarray, dst: np.ndarray, size: int = SIZE, alpha: float = 1.0) -> np.ndarray:
    """Backward displacement field of the affine MLS deformation taking src -> dst.

    Output pixel v samples the source at v + field(v); built by solving MLS
    from dst to src (Schaefer et al. 2006, affine variant).
    """
    p = np.asarray(dst, np.float64)  # control points where the content ends up
    q = np.asarray(src, np.float64)  # ...and where it comes from
    ys, xs = np.mgrid[0:size, 0:size]
    v = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64)  # (P, 2)
    d2 = ((v[:, None, :] - p[None, :, :]) ** 2).sum(-1)  # (P, N)
    w = 1.0 / (d2 ** alpha + 1e-3)
    sw = w.sum(1, keepdims=True)
    ps = (w @ p) / sw
    qs = (w @ q) / sw
    ph = p[None] - ps[:, None]  # (P, N, 2)
    qh = q[None] - qs[:, None]
    a = np.einsum("pn,pni,pnj->pij", w, ph, ph)
    b = np.einsum("pn,pni,pnj->pij", w, ph, qh)
    det = a[:, 0, 0] * a[:, 1, 1] - a[:, 0, 1] * a[:, 1, 0]
    inv = np.empty_like(a)
    inv[:, 0, 0], inv[:, 1, 1] = a[:, 1, 1], a[:, 0, 0]
    inv[:, 0, 1], inv[:, 1, 0] = -a[:, 0, 1], -a[:, 1, 0]
    inv /= det[:, None, None]
    m = inv @ b
    f = np.einsum("pi,pij->pj", v - ps, m) + qs
    return (f - v).reshape(size, size, 2).astype(np.float32)


def _pins(layout: Layout) -> Dict[str, Tuple[float, float]]:
    """Layout points plus eye corners/lids and the frame corners."""
    pins = dict(layout.points)
    ew, eh = layout.eye_half
    for side in ("l", "r"):
        cx, cy = layout.points[f"eye_{side}"]
        pins.update({f"eye_{side}_outer": (cx - ew, cy), f"eye_{side}_inner": (cx + ew, cy),
                     f"eye_{side}_top": (cx, cy - eh), f"eye_{side}_bottom": (cx, cy + eh)})
    # the lower lip's top edge, just under the (pinned) lip line: moving it with the
    # jaw makes the warp open a gap that stretches only the dark lip line
    lx, ly = layout.points["lip_upper"]
    pins["lip_line_lower"] = (lx, ly + 2)
    s = SIZE - 1
    pins.update({"frame_tl": (0, 0), "frame_tr": (s, 0), "frame_bl": (0, s), "frame_br": (s, s),
                 "frame_l": (0, s / 2), "frame_r": (s, s / 2), "frame_t": (s / 2, 0)})
    return pins


def _moving_field(layout: Layout, movers: Dict[str, float], delta: Tuple[float, float]) -> np.ndarray:
    pins = _pins(layout)
    names = list(pins)
    src = np.array([pins[n] for n in names])
    dst = src.copy()
    for i, n in enumerate(names):
        if n in movers:
            dst[i] += np.array(delta) * movers[n]
    return mls_field(src, dst, alpha=MLS_ALPHA)


def warp_fields(layout: Layout) -> Dict[str, np.ndarray]:
    ys, xs = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    # breathe: grow about the bottom centre so the crown rises BREATHE_PX
    ax, ay = (SIZE - 1) / 2.0, SIZE - 1.0
    k = BREATHE_PX / ay
    breathe = np.stack([-(xs - ax) * k, -(ys - ay) * k], axis=-1).astype(np.float32)
    return {
        "jaw": _moving_field(layout, JAW_MOVERS, (0.0, JAW_PX)),
        "brows": _moving_field(layout, BROW_MOVERS, (0.0, -BROW_PX)),
        "breathe": breathe,
        "gaze_x": _moving_field(layout, EYE_MOVERS, (GAZE_PX, 0.0)),
        "gaze_y": _moving_field(layout, EYE_MOVERS, (0.0, GAZE_PX)),
    }


# ---------------------------------------------------------------- boxes + patches

def mouth_box(layout: Layout) -> Box:
    p = layout.points
    x0 = int(round(p["mouth_l"][0] - 10))
    x1 = int(round(p["mouth_r"][0] + 10))
    y0 = int(round(min(p["mouth_l"][1], p["mouth_r"][1]) - 6))
    y1 = int(round(p["chin"][1] + 10))
    return (x0, y0, x1 - x0, y1 - y0)


def eye_boxes(layout: Layout) -> Tuple[Box, Box]:
    ew, eh = layout.eye_half
    w, h = int(round(2 * ew + 12)), int(round(2 * eh + 10))
    out = []
    for side in ("l", "r"):
        cx, cy = layout.points[f"eye_{side}"]
        out.append((int(round(cx - w / 2)), int(round(cy - h / 2)), w, h))
    return out[0], out[1]


def _ramp(x: np.ndarray, x0: float, x1: float) -> np.ndarray:
    """0 at x0, 1 at x1, smoothstep between."""
    t = np.clip((x - x0) / (x1 - x0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _box_feather(w: int, h: int, edge: float) -> np.ndarray:
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32) + 0.5
    return (_ramp(xs, 0, edge) * _ramp(w - xs, 0, edge)
            * _ramp(ys, 0, edge) * _ramp(h - ys, 0, edge))


def _sample(img: np.ndarray, mapx: np.ndarray, mapy: np.ndarray) -> np.ndarray:
    return cv2.remap(img, mapx.astype(np.float32), mapy.astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _hi_grid(box: Box):
    """Pixel-centre coordinates (in 252 space) of the box at SUPERSAMPLE x."""
    x, y, w, h = box
    n = SUPERSAMPLE
    ys, xs = np.mgrid[0:h * n, 0:w * n].astype(np.float32)
    return x + (xs + 0.5) / n - 0.5, y + (ys + 0.5) / n - 0.5


def _down(img: np.ndarray, box: Box) -> np.ndarray:
    return cv2.resize(img, (box[2], box[3]), interpolation=cv2.INTER_AREA)


def _rgba(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    return np.concatenate([np.clip(rgb, 0, 1), np.clip(alpha, 0, 1)[..., None]], axis=-1).astype(np.float32)


def mouth_patch(hub: np.ndarray, mask: np.ndarray, layout: Layout, box: Box, opening: float,
                extra_w: float = 0.0, stretch: float = 1.0, squeeze: float = 1.0) -> np.ndarray:
    """Tear the mouth open along the lip line: upper lip/moustache stay, lower lip and
    chin drop by `opening` px (less towards the corners), the gap is pure black.

    `extra_w`/`stretch` widen the cavity/lower lip (wide shapes); `squeeze` narrows only
    the cavity (rounded shapes). The corners stay anchored so the grin survives.
    """
    p = layout.points
    (lx, ly), (rx, ry) = p["mouth_l"], p["mouth_r"]
    cx, half_w = (lx + rx) / 2, (rx - lx) / 2
    X, Y = _hi_grid(box)
    u = (X - cx) / half_w
    corner_y = ly + (ry - ly) * (X - lx) / (rx - lx)
    centre_corner_y = (ly + ry) / 2
    y_top = corner_y + (p["lip_upper"][1] - centre_corner_y) * np.clip(1 - u * u, 0, 1)
    inside = np.abs(u) < 1
    lower = inside & (Y >= y_top)

    cav_hw = (half_w + extra_w) * squeeze
    uc = (X - cx) / cav_hw
    prof = np.clip(1 - uc * uc, 0, 1) ** 0.5
    drop = opening * prof  # how far the lower lip drops in this column

    # horizontal scale of the lower lip, fading out past the corners and below the lip
    # (rounded shapes only narrow the cavity: squeezing the lip would pull in background)
    sx = stretch
    lip_bottom = p["lip_lower"][1] + 5
    hx = (1 - _ramp(np.abs(X - cx), half_w, half_w + 8)) * (1 - _ramp(Y, lip_bottom, lip_bottom + 10))
    src_x = X - (X - cx) * (1 - 1 / sx) * hx
    src_y = Y - drop

    hub_f = hub.astype(np.float32) / 255.0
    up = _sample(hub_f, X, Y)
    dn = _sample(hub_f, np.where(lower, src_x, X), np.where(lower, src_y, Y))
    rgb = np.where(lower[..., None], dn, up)
    cavity = lower & (Y < y_top + drop)
    rgb[cavity] = 0.0
    rgb = _down(rgb, box)
    cav = _down(cavity.astype(np.float32), box)
    rgb[cav > 0.5] = 0.0  # pure black inside, anti-aliased only at the lip edges

    # alpha: nothing above the lip line (moustache stays the hub's), feathered sides/bottom
    x, y, w, h = box
    top = _down(np.where(inside, y_top, p["lip_upper"][1]).astype(np.float32), box)
    ys = np.arange(h, dtype=np.float32)[:, None] + y + 0.5
    alpha = _ramp(ys, top - 4, top - 1) * _box_feather(w, h, 5)
    alpha = np.maximum(alpha, cav)  # the cavity is always fully covered
    alpha *= np.maximum(mask[y:y + h, x:x + w], cav)
    return _rgba(rgb, alpha)


def _eye_curves(X, layout: Layout, side: str):
    cx, cy = layout.points[f"eye_{side}"]
    ew, eh = layout.eye_half
    u = np.clip((X - cx) / ew, -1, 1)
    bump = 1 - u * u
    return cy - eh * bump, cy + eh * bump, bump


def eye_patch(hub: np.ndarray, mask: np.ndarray, layout: Layout, box: Box, side: str,
              closed: float = 0.0, look_up: float = 0.0) -> np.ndarray:
    """closed: 0 open .. 1 shut (upper lid pulled down to the lower lid, darkened).
    look_up: px the iris moves up inside the eye opening."""
    X, Y = _hi_grid(box)
    yt, yb, bump = _eye_curves(X, layout, side)
    ew, eh = layout.eye_half
    yl = yt + closed * (yb - yt)  # new lid edge
    skin = yt - eh * 1.2  # lid skin above the eye that gets stretched down
    src_y = Y.copy()
    in_skin = (Y >= skin) & (Y < yl)
    src_y[in_skin] = (skin + (Y - skin) * (yt - skin) / np.maximum(yl - skin, 1e-3))[in_skin]
    in_eye = (Y >= yl) & (Y < yb)
    src_y[in_eye] = (yb - (yb - Y) * (yb - yt) / np.maximum(yb - yl, 1e-3))[in_eye]
    if look_up:
        src_y = np.where(in_eye, np.minimum(src_y + look_up * bump, yb - 1), src_y)
    rgb = _sample(hub.astype(np.float32) / 255.0, X, src_y)

    shade = np.ones_like(X)
    if closed:
        covered = (Y >= yt) & (Y < yl)
        shade[covered] = 0.8  # the lid over the eye is in shadow
        shade[in_eye] = 0.85
        shade *= 1 - 0.55 * closed * np.exp(-((Y - yl) / 1.2) ** 2) * (bump > 0.05)  # lash line
    rgb *= shade[..., None]
    rgb = _down(rgb, box)

    x, y, w, h = box
    cx, cy = layout.points[f"eye_{side}"]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32) + 0.5
    r = np.sqrt(((xs + x - cx) / (ew + 4)) ** 2 + ((ys + y - cy) / (eh * 1.2 + 4)) ** 2)
    alpha = 1 - _ramp(r, 0.8, 1.0)
    alpha *= mask[y:y + h, x:x + w]
    return _rgba(rgb, alpha)


# ---------------------------------------------------------------- writing

def _png_rgba(path: str, rgba: np.ndarray) -> None:
    u8 = np.round(np.clip(rgba, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(path, cv2.cvtColor(u8, cv2.COLOR_RGBA2BGRA))


def write_bank(out_dir: str, hub: np.ndarray, mask: np.ndarray, m_box: Box,
               mouth: Dict[str, np.ndarray], e_boxes: Optional[Tuple[Box, Box]] = None,
               eyes: Optional[Dict[str, np.ndarray]] = None,
               expressions: Optional[Dict[str, np.ndarray]] = None,
               warps: Optional[Dict[str, np.ndarray]] = None) -> str:
    """Write a bank directory in the format of assets/face/README.md; returns manifest path.

    Bank subfolders already in out_dir are emptied first, so no stale patch survives.
    """
    for sub in ("mouth", "eyes", "expr", "warps"):
        d = os.path.join(out_dir, sub)
        if os.path.isdir(d):
            for name in os.listdir(d):
                if name.endswith((".png", ".npy")):
                    os.remove(os.path.join(d, name))
        os.makedirs(d, exist_ok=True)
    write_rgb(os.path.join(out_dir, "hub.png"), hub)
    cv2.imwrite(os.path.join(out_dir, "mask.png"), np.round(mask * 255).astype(np.uint8))
    manifest = {"size": SIZE, "hub": "hub.png", "mask": "mask.png",
                "mouth": {"box": list(m_box), "patches": {}}}
    for key, rgba in sorted(mouth.items()):
        rel = f"mouth/{key}.png"
        _png_rgba(os.path.join(out_dir, rel), rgba)
        manifest["mouth"]["patches"][key] = rel
    if eyes:
        manifest["eyes"] = {"box_left": list(e_boxes[0]), "box_right": list(e_boxes[1]),
                            "patches": {}}
        for key, rgba in sorted(eyes.items()):
            rel = f"eyes/{key}.png"
            _png_rgba(os.path.join(out_dir, rel), rgba)
            manifest["eyes"]["patches"][key] = rel
    if expressions:
        manifest["expressions"] = {}
        for key, rgba in sorted(expressions.items()):
            rel = f"expr/{key}.png"
            _png_rgba(os.path.join(out_dir, rel), rgba)
            manifest["expressions"][key] = rel
    if warps:
        manifest["warps"] = {}
        for key, field in sorted(warps.items()):
            rel = f"warps/{key}.npy"
            np.save(os.path.join(out_dir, rel), field.astype(np.float32))
            manifest["warps"][key] = rel
    for sub in ("mouth", "eyes", "expr", "warps"):
        d = os.path.join(out_dir, sub)
        if not os.listdir(d):
            os.rmdir(d)
    path = os.path.join(out_dir, "manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=1)
        f.write("\n")
    return path


# mouth shapes: (opening px, extra cavity half-width px, lower-lip stretch, squeeze)
MOUTH_SHAPES = {
    "A_mild": (-1.0, 0, 1.0, 1.0), "A_wide": (-1.0, 0, 1.0, 1.0),  # lips pressed
    "B_mild": (3.0, 0, 1.0, 1.0), "B_wide": (4.0, 2, 1.04, 1.0),
    "C_mild": (6.0, 0, 1.0, 1.0), "C_wide": (8.0, 2, 1.06, 1.0),
    "D_mild": (10.0, 0, 1.0, 1.0), "D_wide": (13.0, 2, 1.06, 1.0),
    "E_mild": (6.0, 0, 1.0, 0.8),  # rounded: C squeezed
    "F_mild": (5.0, 0, 1.0, 0.6),  # puckered: C squeezed harder
}


def build_stub_bank(source: str, out_dir: str, box: Optional[Box] = None,
                    points: Optional[str] = None) -> str:
    rgb = read_rgb(source)
    face = crop_to_size(rgb, box or centre_box(rgb.shape))
    hub, mask = hub_and_mask(face)
    layout = load_layout(points)
    m_box = mouth_box(layout)
    mouth = {k: mouth_patch(hub, mask, layout, m_box, *args) for k, args in MOUTH_SHAPES.items()}
    e_boxes = eye_boxes(layout)
    # one patch serves both eyes, so each patch is cut from the left eye
    eyes = {
        "open": eye_patch(hub, mask, layout, e_boxes[0], "l"),
        "half": eye_patch(hub, mask, layout, e_boxes[0], "l", closed=0.5),
        "closed": eye_patch(hub, mask, layout, e_boxes[0], "l", closed=1.0),
        "look_up": eye_patch(hub, mask, layout, e_boxes[0], "l", look_up=2.5),
    }
    return write_bank(out_dir, hub, mask, m_box, mouth, e_boxes, eyes, warps=warp_fields(layout))


def parse_box(text: str) -> Box:
    try:
        x, y, w, h = (int(v) for v in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError("expected x,y,w,h")
    if w != h or w <= 0:
        raise argparse.ArgumentTypeError(f"the box must be square (w == h), got {w}x{h}")
    return (x, y, w, h)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("source", help="front-facing face image on a black background")
    ap.add_argument("out_dir", help="bank directory to write")
    ap.add_argument("--box", type=parse_box, help="crop x,y,w,h in source pixels (default: centre square)")
    ap.add_argument("--points", help="control points JSON (default: generic layout)")
    a = ap.parse_args(argv)
    print(build_stub_bank(a.source, a.out_dir, a.box, a.points))


if __name__ == "__main__":
    main()
