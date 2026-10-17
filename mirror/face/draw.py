"""Draw the mirror's face on a square pygame surface.

Designed at 252x252 (one LED panel); every coordinate is scaled by
``surf.get_width() / 252``. Only lit pixels show through the two-way acrylic,
so the background is pure black and the palette is the state accent colour
plus dimmed versions of it (no white fills).
"""
import math

import pygame

from mirror.states import State

DESIGN = 252.0

ACCENT = {
    State.IDLE: (0xFF, 0x7A, 0x8A),
    State.LISTENING: (0x6F, 0xD5, 0xD0),
    State.THINKING: (0xE8, 0xB4, 0x57),
    State.SPEAKING: (0xFF, 0x7A, 0x8A),
}

EYE_CX = (76.0, 176.0)
EYE_CY = 100.0
EYE_HALF_W = 32.0
EYE_HALF_H = 22.0
IRIS_R = 13.0
LINE_W = 3.0
MOUTH_CX = 126.0
MOUTH_CY = 200.0
MOUTH_HALF_W = 40.0
MOUTH_MAX_H = 44.0


def _dim(colour, k):
    return tuple(int(c * k) for c in colour)


def _curve(xs, f):
    return [(x, f(x)) for x in xs]


def _line(surf, colour, pts, s):
    pygame.draw.lines(surf, colour, False, [(x * s, y * s) for x, y in pts], max(3, round(LINE_W * s)))


def _draw_eye(surf, colour, cx, cy, openness, iris_dx, iris_dy, s):
    half_h = EYE_HALF_H * openness
    n = 24
    xs = [cx - EYE_HALF_W + 2 * EYE_HALF_W * i / n for i in range(n + 1)]

    def lid(x):
        u = (x - cx) / EYE_HALF_W
        return half_h * (1 - u * u)

    upper = _curve(xs, lambda x: cy - lid(x))
    lower = _curve(xs, lambda x: cy + 0.8 * lid(x))

    if half_h > 1.5:
        # Render glow + iris on a scratch surface, mask it to the almond, add.
        rect = pygame.Rect(
            int((cx - EYE_HALF_W - 2) * s), int((cy - EYE_HALF_H - 2) * s),
            int((2 * EYE_HALF_W + 4) * s), int((2 * EYE_HALF_H + 4) * s),
        )
        scratch = pygame.Surface(rect.size)
        mask = pygame.Surface(rect.size)
        ox, oy = rect.topleft
        ix, iy = (cx + iris_dx) * s - ox, (cy + iris_dy) * s - oy
        for r, k in ((IRIS_R * 1.7, 0.15), (IRIS_R * 1.35, 0.3), (IRIS_R, 1.0)):
            pygame.draw.circle(scratch, _dim(colour, k), (ix, iy), r * s)
        pygame.draw.circle(scratch, (0, 0, 0), (ix, iy), IRIS_R * 0.4 * s)
        pygame.draw.polygon(
            mask, (255, 255, 255),
            [(x * s - ox, y * s - oy) for x, y in upper + lower[::-1]],
        )
        scratch.blit(mask, (0, 0), special_flags=pygame.BLEND_RGB_MULT)
        surf.blit(scratch, rect.topleft, special_flags=pygame.BLEND_RGB_ADD)

    _line(surf, colour, upper, s)
    _line(surf, colour, lower, s)


def _draw_brow(surf, colour, cx, outer_sign, lift, s):
    pts = []
    for i in range(13):
        u = -1 + i / 6
        x = cx + u * 30
        y = 56 - lift - 8 * (1 - u * u) + outer_sign * u * 4
        pts.append((x, y))
    _line(surf, _dim(colour, 0.7), pts, s)


def _draw_mouth(surf, colour, mouth, s):
    h = max(0.0, min(1.0, mouth)) * MOUTH_MAX_H
    xs = [MOUTH_CX - MOUTH_HALF_W + 2 * MOUTH_HALF_W * i / 20 for i in range(21)]

    def base(x):
        u = (x - MOUTH_CX) / MOUTH_HALF_W
        return MOUTH_CY - 6 * u * u

    def span(x):
        u = (x - MOUTH_CX) / MOUTH_HALF_W
        return h / 2 * (1 - u * u)

    top = _curve(xs, lambda x: base(x) - span(x))
    bottom = _curve(xs, lambda x: base(x) + span(x))
    if h > 2:
        poly = [(x * s, y * s) for x, y in top + bottom[::-1]]
        pygame.draw.polygon(surf, _dim(colour, 0.3), poly)
    _line(surf, colour, top, s)
    _line(surf, colour, bottom, s)


def draw_face(surf: pygame.Surface, state: State, mouth: float, t: float, blink: float) -> None:
    s = surf.get_width() / DESIGN
    colour = ACCENT[state]
    surf.fill((0, 0, 0))

    openness = 1.0 - max(0.0, min(1.0, blink))
    drift_x = drift_y = 0.0
    iris_dx = iris_dy = 0.0
    lift = 0.0
    if state is State.IDLE:
        drift_x = 3.0 * math.sin(t * 0.7)
        drift_y = 2.0 * math.sin(t * 0.43 + 1.0)
    elif state is State.THINKING:
        # glance up-left, then up-right
        iris_dx = -8.0 if math.sin(t * 1.2) < 0 else 8.0
        iris_dy = -8.0
        lift = 3.0
    elif state is State.LISTENING:
        lift = 2.0

    for cx, sign in zip(EYE_CX, (-1, 1)):
        _draw_eye(surf, colour, cx + drift_x, EYE_CY + drift_y, openness,
                  iris_dx, iris_dy, s)
        _draw_brow(surf, colour, cx + drift_x, sign, lift, s)
    _draw_mouth(surf, colour, mouth, s)
