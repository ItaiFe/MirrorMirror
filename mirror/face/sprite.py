"""Sprite compositor face: a layered face built from a FaceBank (Task 15 format).

Per frame:
  1. base = the hub, or the current idle-loop frame (ping-pong at 15 fps);
  2. alpha-blend, in hub coordinates, the expression overlay, the eye patches
     and the mouth patch (crossfaded over 2 frames when the shape changes);
  3. ONE cv2.remap of the composited image together with the mask (stacked as
     a 4th channel), grid = identity + sum(weight * warp field), borderValue 0.

Order: the patches are composited before the warp so that breathe and gaze
move face and patches together (no seams at the patch edges). The jaw field
then also carries the mouth patch: its upper part (alpha 0 above the lip line,
moustache pinned) stays put while its dropped lower lip and cavity move down
with the chin, so the patch's opening and the jaw's add up (a clearly visible
black mouth at 252 px) and the patch's lower edge stays aligned with the warped
chin. Task 17's "+jaw" contact-sheet tiles paste the patch after the jaw warp;
compared side by side, that order gives a smaller opening and a seam where the
patch's lower edge meets the dropped chin, so it was not used.

The jaw field alone smears the hub's closed lips (and the A patch's pressed
lips) into a vertical streak, so the closed shapes X and A damp the jaw
(JAW_SHAPE_FACTOR, crossfaded along with the patch).

The bank has one eye patch for both eyes (ruling R20); it is mirrored
horizontally on the right eye box. The "open" eye is not pasted: the hub's own
eyes show, so the right eye keeps its real (unmirrored) shape when not blinking.

Floating (Task 27, FloatParams): the whole face hovers like a spirit in the
glass: a sine bob, a slower sway (x1.37) and a tilt (x0.83), their amplitude
sprung per state (FLOAT_STATE_AMP, FLOAT_TAU_S), the face shrunk by `scale`
about the centre so the motion never reaches the frame edge, and the mask's
outline faded into black (feathered once, at construction). The float is a
similarity transform A of the warped face; it is folded into the same single
image remap: grid(p) = A^-1(p) + D(A^-1(p)), where D is the summed warp
displacement, resampled at A^-1(p) with one cheap 2-channel cv2.warpAffine
(the fields are steep around the jaw, so D(p) would misplace the chin).
With the float disabled, render() runs exactly the pre-float code path.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from mirror.face.bank import FaceBank
from mirror.states import State

SPRING_TAU_S = 0.120
JAW_SCALE = 0.85  # the brief's 0.6 read too subtly at 252 px
# Cap: the stub's beard ornament ends at row 241 and the jaw moves it ~15.6 px
# at weight 1, so much beyond 0.6 its tip would leave the 252 px frame.
JAW_MAX = 0.6
BREATHE_PERIOD_S = 4.5
GAZE_WANDER_S = (2.0, 5.0)
GAZE_WANDER_RANGE = (0.8, 0.5)  # max |gaze_x|, |gaze_y| when wandering
THINKING_GAZE = (0.6, -0.7)
BROWS = {State.LISTENING: 0.8, State.THINKING: 0.4}
BLINK_S = 0.150
BLINK_GAP_S = (2.0, 6.0)
SMIRK_GAP_S = (20.0, 40.0)
SMIRK_HOLD_S = 2.0
LOOP_FPS = 15.0
CROSSFADE_FRAMES = 2  # in-between frames when the mouth shape changes
WIDE_ABOVE = 0.5  # mouth level that switches a shape to its wide patch...
MILD_BELOW = 0.4  # ...and back (hysteresis, so the patch doesn't flicker)
# Closed-lip shapes damp the jaw: the jaw field would otherwise stretch the
# pressed lips of the A patch into a long smear. Crossfaded with the patch.
JAW_SHAPE_FACTOR = {"A": 0.2, "X": 0.25}
# floating face: amplitude per state, sprung slower than the expressions
FLOAT_STATE_AMP = {State.IDLE: 1.0, State.LISTENING: 0.7, State.THINKING: 0.8,
                   State.SPEAKING: 0.45}
FLOAT_TAU_S = 0.600
FLOAT_SWAY = (0.4, 1.37)  # (amplitude x amp_px, period x period_s)
FLOAT_TILT_PERIOD = 0.83  # x period_s


@dataclass(frozen=True)
class FloatParams:
    """The floating-face settings (config face.float_*)."""

    enabled: bool = True
    amp_px: float = 5.0
    period_s: float = 6.5
    tilt_deg: float = 1.2
    scale: float = 0.93
    edge_feather_px: float = 6.0

    @classmethod
    def from_config(cls, face_cfg) -> "FloatParams":
        return cls(enabled=bool(face_cfg.float_enabled), amp_px=float(face_cfg.float_amp_px),
                   period_s=float(face_cfg.float_period_s),
                   tilt_deg=float(face_cfg.float_tilt_deg), scale=float(face_cfg.float_scale),
                   edge_feather_px=float(face_cfg.float_edge_feather_px))


def feather_mask(mask: np.ndarray, px: float) -> np.ndarray:
    """mask (0..255) faded to 0 over its outer `px` pixels: x smoothstep(distance
    inside the mask / px). Monotonic across the boundary, never brighter."""
    if px <= 0:
        return mask
    inside = (mask >= 127.5).astype(np.uint8)
    dist = cv2.distanceTransform(inside, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    u = np.clip(dist / px, 0.0, 1.0)
    return (mask * (u * u * (3.0 - 2.0 * u))).astype(np.float32)


class SpriteFace:
    def __init__(self, bank: FaceBank, rng: Optional[random.Random] = None,
                 float_params: Optional[FloatParams] = None):
        self.bank = bank
        self.rng = rng or random.Random()
        self.float_params = float_params or FloatParams(enabled=False)
        s = bank.size
        self._hub = bank.hub.astype(np.float32)
        self._mask = bank.mask.astype(np.float32) * 255.0
        if self.float_params.enabled:
            self._mask = feather_mask(self._mask, self.float_params.edge_feather_px)
        self._float_amp = FLOAT_STATE_AMP[State.IDLE]
        self._loop = (np.concatenate(bank.idle_loops).astype(np.float32)
                      if bank.idle_loops else np.zeros((0, s, s, 3), np.float32))
        ys, xs = np.mgrid[0:s, 0:s].astype(np.float32)
        self._ident = np.stack([xs, ys], axis=-1)
        self._t = 0.0
        self._state = State.IDLE
        self._mouth = 0.0
        # sprung weights (warp fields and expression overlays)
        self._w: Dict[str, float] = {"gaze_x": 0.0, "gaze_y": 0.0, "brows": 0.0,
                                     "smirk": 0.0}
        self._wander = (0.0, 0.0)
        self._next_wander = self.rng.uniform(*GAZE_WANDER_S)
        self.auto_blink = True  # random blinks every BLINK_GAP_S; blink() works either way
        self._blink_at = -1.0
        self._next_blink = self.rng.uniform(*BLINK_GAP_S)
        self._smirk_until = -1.0
        self._next_smirk = self.rng.uniform(*SMIRK_GAP_S)
        self._viseme = "X"
        self._wide = False
        self._patch: Optional[np.ndarray] = None
        self._prev_patch: Optional[np.ndarray] = None
        self._fade_step = CROSSFADE_FRAMES + 1  # > CROSSFADE_FRAMES: no fade running
        self._prev_jaw_factor = 1.0

    # -- animation ---------------------------------------------------------
    def blink(self) -> None:
        """Start a blink now."""
        self._blink_at = self._t
        self._next_blink = self._t + BLINK_S + self.rng.uniform(*BLINK_GAP_S)

    def update(self, dt: float, state: State, mouth: float, viseme: str) -> None:
        dt = max(0.0, dt)
        self._t += dt
        t = self._t
        self._state = state
        self._mouth = min(1.0, max(0.0, mouth))

        if self.auto_blink and t >= self._next_blink:
            self.blink()
        if t >= self._next_wander:
            gx, gy = GAZE_WANDER_RANGE
            self._wander = (self.rng.uniform(-gx, gx), self.rng.uniform(-gy, gy))
            self._next_wander = t + self.rng.uniform(*GAZE_WANDER_S)
        if state is not State.IDLE:
            self._next_smirk = max(self._next_smirk, t + self.rng.uniform(*SMIRK_GAP_S))
        elif t >= self._next_smirk:
            self._smirk_until = t + SMIRK_HOLD_S
            self._next_smirk = t + SMIRK_HOLD_S + self.rng.uniform(*SMIRK_GAP_S)

        if state is State.IDLE:
            gaze = self._wander
        elif state is State.THINKING:
            gaze = THINKING_GAZE
        else:  # listening and speaking look straight at the visitor
            gaze = (0.0, 0.0)
        target = {
            "gaze_x": gaze[0],
            "gaze_y": gaze[1],
            "brows": BROWS.get(state, 0.0),
            "smirk": 1.0 if state is State.IDLE and t < self._smirk_until else 0.0,
        }
        k = 1.0 - math.exp(-dt / SPRING_TAU_S)
        for name, goal in target.items():
            self._w[name] += (goal - self._w[name]) * k
        goal = FLOAT_STATE_AMP.get(state, 1.0)
        self._float_amp += (goal - self._float_amp) * (1.0 - math.exp(-dt / FLOAT_TAU_S))

        self._update_mouth(viseme)

    def _update_mouth(self, viseme: str) -> None:
        # intentional hysteresis: wide above WIDE_ABOVE, back to mild only below MILD_BELOW
        wide = self._mouth > WIDE_ABOVE or (self._wide and self._mouth >= MILD_BELOW)
        if viseme == self._viseme and wide == self._wide:
            # unchanged shape and width: keep the patch; a running crossfade advances
            self._fade_step += 1
            return
        self._prev_jaw_factor = self._jaw_factor()
        self._prev_patch = self._current_patch()
        self._viseme, self._wide = viseme, wide
        self._patch = self.bank.mouth_patch(viseme, wide)
        if self._patch is self._prev_patch:  # e.g. a bank with only one width
            self._fade_step += 1
            return
        self._fade_step = 1

    def _current_patch(self) -> Optional[np.ndarray]:
        """What the mouth shows right now (a crossfade freezes into one patch)."""
        k = self._fade_weight()
        if k >= 1.0 or self._prev_patch is self._patch:
            return self._patch
        p0, p1 = self._prev_patch, self._patch
        if p0 is None or p1 is None:  # fading to/from the hub's own mouth
            p = p1 if p0 is None else p0
            kk = k if p0 is None else 1.0 - k
            out = p.copy()
            out[..., 3] *= kk
            return out
        return p0 * (1.0 - k) + p1 * k

    def _jaw_factor(self) -> float:
        k = self._fade_weight()
        cur = JAW_SHAPE_FACTOR.get(self._viseme, 1.0)
        return self._prev_jaw_factor * (1.0 - k) + cur * k

    def _fade_weight(self) -> float:
        return min(1.0, self._fade_step / (CROSSFADE_FRAMES + 1))

    @property
    def eye_name(self) -> str:
        return self._eye_name()

    def _eye_name(self) -> str:
        if self._blink_at >= 0.0:
            u = (self._t - self._blink_at) / BLINK_S
            if 0.0 <= u < 1.0:
                closure = 1.0 - abs(2.0 * u - 1.0)
                if closure > 0.66:
                    return "closed"
                if closure > 0.2:
                    return "half"
        if self._state is State.THINKING and "squint" in self.bank.eye_patches:
            return "squint"
        return "open"

    def _base(self) -> np.ndarray:
        """The hub, or the idle-loop frame for now: the loops played back to back,
        ping-ponged (0..n-1..1) at LOOP_FPS."""
        n = len(self._loop)
        if n == 0:
            return self._hub.copy()
        if n == 1:
            return self._loop[0].copy()
        i = int(self._t * LOOP_FPS) % (2 * n - 2)
        return self._loop[i if i < n else 2 * n - 2 - i].copy()

    def float_offset(self) -> Tuple[float, float, float]:
        """The float right now: (dx px, dy px, tilt degrees), before the scale."""
        p = self.float_params
        if not p.enabled:
            return 0.0, 0.0, 0.0
        a, tau = self._float_amp, 2.0 * math.pi * self._t / p.period_s
        sway_amp, sway_period = FLOAT_SWAY
        return (a * sway_amp * p.amp_px * math.sin(tau / sway_period),
                a * p.amp_px * math.sin(tau),
                a * p.tilt_deg * math.sin(tau / FLOAT_TILT_PERIOD))

    def _float_inverse(self) -> np.ndarray:
        """2x3 affine taking an output pixel to the unfloated face: the inverse of
        p = c + scale * R(tilt) (q - c) + (dx, dy)."""
        dx, dy, deg = self.float_offset()
        c = (self.bank.size - 1) / 2.0
        th = math.radians(deg)
        k = 1.0 / self.float_params.scale
        cos, sin = math.cos(th) * k, math.sin(th) * k
        # q = c + R(-th) (p - c - d) / scale
        ux, uy = c + dx, c + dy
        return np.array([[cos, sin, c - cos * ux - sin * uy],
                         [-sin, cos, c + sin * ux - cos * uy]], np.float32)

    # -- rendering ---------------------------------------------------------
    @staticmethod
    def _blend(img: np.ndarray, patch: np.ndarray, x: int, y: int, k: float = 1.0) -> None:
        h, w = patch.shape[:2]
        a = patch[..., 3:4] * k
        region = img[y:y + h, x:x + w]
        region += (patch[..., :3] * 255.0 - region) * a

    def render(self) -> np.ndarray:
        bank = self.bank
        img = self._base()

        for name in ("smirk",):  # squint is the eye patch (_eye_name), not a whole-face overlay
            w = self._w[name]
            expr = bank.expressions.get(name)
            if expr is not None and w > 0.01:
                self._blend(img, expr, 0, 0, w)

        eye = self._eye_name()
        if eye != "open" and bank.eye_box_left is not None:
            patch = bank.eye_patch(eye)
            if patch is not None:
                self._blend(img, patch, *bank.eye_box_left[:2])
                self._blend(img, patch[:, ::-1], *bank.eye_box_right[:2])

        patch = self._current_patch()
        if patch is not None:
            self._blend(img, patch, *bank.mouth_box[:2])

        weights = {
            # the jaw is not sprung: it follows the already-smoothed mouth (app.step_mouth)
            "jaw": min(JAW_MAX, self._mouth * JAW_SCALE) * self._jaw_factor(),
            "breathe": 0.5 + 0.5 * math.sin(2.0 * math.pi * self._t / BREATHE_PERIOD_S),
            "gaze_x": self._w["gaze_x"],
            "gaze_y": self._w["gaze_y"],
            "brows": self._w["brows"],
        }
        if self.float_params.enabled:
            grid = self._float_grid(weights)
        else:
            grid = self._ident.copy()
            for name, w in weights.items():
                field = bank.warps.get(name)
                if field is not None and abs(w) > 1e-3:
                    grid += w * field

        rgba = np.concatenate([img, self._mask[..., None]], axis=2)
        out = cv2.remap(rgba, grid, None, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        rgb = out[..., :3] * (out[..., 3:4] * (1.0 / 255.0))
        return np.clip(rgb, 0, 255).astype(np.uint8)

    def _float_grid(self, weights: Dict[str, float]) -> np.ndarray:
        """The remap grid with the float folded in: A^-1(p) + D(A^-1(p))."""
        m = self._float_inverse()
        s = self.bank.size
        q = (self._ident @ m[:, :2].T) + m[:, 2]
        disp = None
        for name, w in weights.items():
            field = self.bank.warps.get(name)
            if field is not None and abs(w) > 1e-3:
                disp = w * field if disp is None else disp + w * field
        if disp is not None:
            q += cv2.warpAffine(disp, m, (s, s), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return q
