"""The background mist: a faint, slowly drifting old-gold haze on the black glass.

Only lit pixels show through the two-way mirror, so the mist stays very dim (its
brightest pixel is `intensity` of full scale) and has pure black gaps between its
wisps: the glass still reads as a mirror.

At construction (numpy only, seeded): two tileable fractal noise fields, white
noise shaped in the Fourier domain (amplitude ~ 1/f, a Gaussian cutoff at
FEATURE_PX), so they are periodic at the frame size by construction.

Per frame, at half resolution (DOWNSCALE: the mist has no fine detail, and a
quarter of the pixels: the mist adds ~0.4 ms a frame at 252 px on an M-series Mac,
compositing included): each field
scrolled by its own slow drift (sub-pixel, one cv2.warpAffine with BORDER_WRAP
each), the two summed (two layers sliding
across each other: the wisps grow, merge and thin out instead of just
translating; the sum / sqrt 2 is unit-std noise again), a smoothstep over SHAPE
(below SHAPE[0] std exactly 0: the black gaps; 1 only above SHAPE[1] std, the
rare densest wisps), upsampled (bilinear), times a static vignette (densest behind the face, black from
VIGNETTE_OUTER of the half-size outwards, so black corners). SpriteFace adds
it, in the gold colour, under the face: rgb += mist * (1 - face alpha).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

FEATURE_PX = 18.0  # high-frequency cutoff: no detail finer than this (soft, never grainy)
SPECTRUM_SLOPE = 1.1  # amplitude ~ 1/f^slope: big billows with some smaller curls
SHAPE = (-0.3, 2.2)  # smoothstep over (lo, hi) std of the summed noise: < lo is black
# drift of the two layers in px/s at speed 1.0 (rising slowly, crossing each other)
DRIFT = ((2.6, -1.3), (-1.7, -2.4))
VIGNETTE_INNER = 0.5  # fraction of the half-size: full density inside...
VIGNETTE_OUTER = 1.38  # ...black from here out: the corners (1.41)
DOWNSCALE = 2  # the noise runs at 1/DOWNSCALE of the frame size


@dataclass(frozen=True)
class MistParams:
    """The background mist settings (config face.mist_*)."""

    enabled: bool = False
    intensity: float = 0.10  # brightest mist pixel, 0..1 of full scale
    color: Tuple[int, int, int] = (150, 110, 45)  # hue only: rescaled to `intensity`
    speed: float = 1.0  # drift rate multiplier
    seed: int = 11

    @classmethod
    def from_config(cls, face_cfg) -> "MistParams":
        return cls(enabled=bool(face_cfg.mist), intensity=float(face_cfg.mist_intensity),
                   color=tuple(int(c) for c in face_cfg.mist_color),
                   speed=float(face_cfg.mist_speed))


def tileable_noise(size: int, rng: np.random.Generator,
                   feature_px: float = FEATURE_PX) -> np.ndarray:
    """(size, size) float32 noise, zero mean and unit std, periodic at `size`."""
    f = np.fft.fftfreq(size)
    kr = np.hypot(f[:, None], f[None, :])
    kr[0, 0] = 1.0
    amp = kr ** -SPECTRUM_SLOPE * np.exp(-(kr * feature_px) ** 2)
    amp[0, 0] = 0.0
    n = np.fft.ifft2(np.fft.fft2(rng.standard_normal((size, size))) * amp).real
    return ((n - n.mean()) / n.std()).astype(np.float32)


def _smoothstep(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    u = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def vignette(size: int) -> np.ndarray:
    c = (size - 1) / 2.0
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
    r = np.hypot(xs - c, ys - c) / (size / 2.0)
    return (1.0 - _smoothstep(r, VIGNETTE_INNER, VIGNETTE_OUTER)).astype(np.float32)


class Mist:
    def __init__(self, params: MistParams, size: int):
        self.params = params
        self.size = size
        self._n = n = max(1, size // DOWNSCALE)
        self._px = n / size  # noise px per frame px
        rng = np.random.default_rng(params.seed)
        # pre-scaled by 1/sqrt(2) / (hi - lo) so the frame only adds, offsets and clips
        lo, hi = SHAPE
        k = 1.0 / (math.sqrt(len(DRIFT)) * (hi - lo))
        self._fields = [tileable_noise(n, rng, FEATURE_PX * self._px) * np.float32(k)
                        for _ in DRIFT]
        self._offset = np.float32(-lo / (hi - lo))
        self._vignette = vignette(size)
        # the per-pixel colour at mist level 1: the hue scaled so its max is the peak
        color = np.asarray(params.color, np.float32)
        self.rgb = color * (params.intensity * 255.0 / max(1.0, float(color.max())))

    def level(self, t: float) -> np.ndarray:
        """(size, size) float32 mist level 0..1 at time t s."""
        n, k = self._n, t * self.params.speed * self._px
        u = None
        for field, (vx, vy) in zip(self._fields, DRIFT):
            # BORDER_WRAP needs offsets within the tile: wrap them first
            m = np.float32([[1, 0, math.fmod(vx * k, n)], [0, 1, math.fmod(vy * k, n)]])
            moved = cv2.warpAffine(field, m, (n, n), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_WRAP)
            u = moved if u is None else cv2.add(u, moved)
        u += self._offset
        np.clip(u, 0.0, 1.0, out=u)
        # smoothstep u^2 (3 - 2u), up to the frame size, then the vignette
        u = u * u * (3.0 - 2.0 * u)
        if n != self.size:
            u = cv2.resize(u, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        return u * self._vignette
