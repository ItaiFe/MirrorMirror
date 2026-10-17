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


def test_scales():
    surf = _render(State.SPEAKING, mouth=0.6, size=756)
    lit = sum(
        1
        for y in range(0, 756, 3)
        for x in range(0, 756, 3)
        if sum(surf.get_at((x, y))[:3]) > 60
    )
    assert lit / (252 * 252) > 0.01
