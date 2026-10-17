import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame

from mirror.face.draw import draw_face
from mirror.states import State

TEAL = (0x6F, 0xD5, 0xD0)
PINK = (0xFF, 0x7A, 0x8A)


def _render(state=State.IDLE, mouth=0.0, t=0.0, blink=0.0, size=252):
    surf = pygame.Surface((size, size))
    surf.fill((9, 9, 9))  # prove draw_face clears the background itself
    draw_face(surf, state, mouth, t, blink)
    return surf


def _lit(surf, rows, cols):
    n = 0
    for y in range(*rows):
        for x in range(*cols):
            if sum(surf.get_at((x, y))[:3]) > 60:
                n += 1
    return n


def test_background_black():
    assert tuple(_render().get_at((0, 0)))[:3] == (0, 0, 0)


def test_mouth_opens():
    closed = _lit(_render(mouth=0.0), (170, 230), (76, 176))
    opened = _lit(_render(mouth=1.0), (170, 230), (76, 176))
    assert opened >= 1.5 * closed


def test_blink_closes_eyes():
    eyes_open = _lit(_render(blink=0.0), (70, 130), (0, 252))
    eyes_shut = _lit(_render(blink=1.0), (70, 130), (0, 252))
    assert eyes_shut < 0.4 * eyes_open


def _dominant(surf):
    r = g = b = n = 0
    for y in range(surf.get_height()):
        for x in range(surf.get_width()):
            c = surf.get_at((x, y))
            if c.r + c.g + c.b > 150:
                r, g, b, n = r + c.r, g + c.g, b + c.b, n + 1
    return (r / n, g / n, b / n)


def _dist(a, b):
    return sum((p - q) ** 2 for p, q in zip(a, b)) ** 0.5


def test_state_colour():
    dom = _dominant(_render(State.LISTENING))
    assert _dist(dom, TEAL) < _dist(dom, PINK)


def _bbox(surf):
    xs, ys = [], []
    for y in range(surf.get_height()):
        for x in range(surf.get_width()):
            if sum(surf.get_at((x, y))[:3]) > 60:
                xs.append(x)
                ys.append(y)
    return min(xs), min(ys), max(xs), max(ys)


def test_eye_corners_lit():
    surf = _render(State.LISTENING)
    for cx in (76, 176):
        assert _lit(surf, (97, 104), (cx - 34, cx - 29)) > 0
        assert _lit(surf, (97, 104), (cx + 29, cx + 34)) > 0


def test_scales():
    small = _bbox(_render(State.SPEAKING, mouth=0.6))
    big = _bbox(_render(State.SPEAKING, mouth=0.6, size=756))
    for a, b in zip(small, big):
        assert abs(b - 3 * a) <= 0.1 * 3 * 252 / 2 or a == 0
    w_small, w_big = small[2] - small[0], big[2] - big[0]
    h_small, h_big = small[3] - small[1], big[3] - big[1]
    assert 2.7 <= w_big / w_small <= 3.3
    assert 2.7 <= h_big / h_small <= 3.3
