"""mirror-face: the face renderer process.

Polls the FaceBus for (state, mouth, viseme) from mirror-voice and draws the
face at 30 fps with a smoothed mouth: either the sprite face composited from a
face bank (face.renderer = "sprite") or the procedural line face
("procedural", with its own random blinks). Exits on Esc or SIGTERM.
"""
import argparse
import logging
import math
import random
import signal
import time

import numpy as np
import pygame

from mirror import watchdog
from mirror.bus import FaceBus
from mirror.config import Config, load_config, resolve_repo_path
from mirror.face.draw import draw_face
from mirror.states import State

log = logging.getLogger("mirror.face")

FPS = 30
BLINK_S = 0.150
BLINK_GAP_S = (3.0, 6.0)
MOUTH_ATTACK_S = 0.060
MOUTH_RELEASE_S = 0.120


def step_mouth(current: float, target: float, dt: float) -> float:
    """One-pole smoothing towards target: fast opening, slower closing."""
    target = min(1.0, max(0.0, target))
    tau = MOUTH_ATTACK_S if target > current else MOUTH_RELEASE_S
    k = 1.0 - math.exp(-max(0.0, dt) / tau)
    return min(1.0, max(0.0, current + (target - current) * k))


def blink_amount(since_blink_start: float) -> float:
    """Eyelid closure 0..1 over a BLINK_S blink: close, then reopen."""
    u = since_blink_start / BLINK_S
    if u < 0.0 or u >= 1.0:
        return 0.0
    return 1.0 - abs(2.0 * u - 1.0)


def make_sprite_face(cfg: Config, config_path: str):
    """The SpriteFace for face.renderer = "sprite", None for "procedural"."""
    renderer = cfg.face.renderer
    if renderer == "procedural":
        return None
    if renderer != "sprite":
        raise ValueError(f"unknown face.renderer {renderer!r} (sprite | procedural)")
    from mirror.face.bank import load_bank
    from mirror.face.mist import MistParams
    from mirror.face.sprite import FloatParams, SpriteFace

    bank_dir = resolve_repo_path(cfg.face.bank, config_path)
    log.info("face bank %s", bank_dir)
    return SpriteFace(load_bank(bank_dir), float_params=FloatParams.from_config(cfg.face),
                      mist_params=MistParams.from_config(cfg.face))


def _open_display(cfg):
    size = cfg.face.size
    pygame.display.set_caption("Mirror, Mirror")
    if cfg.face.fullscreen:
        screen = pygame.display.set_mode((cfg.face.screen_w, cfg.face.screen_h),
                                         pygame.FULLSCREEN | pygame.NOFRAME)
        pygame.mouse.set_visible(False)
    else:
        scale = cfg.face.window_scale
        screen = pygame.display.set_mode((size * scale, size * scale))
    return screen


def run(config_path: str) -> None:
    cfg = load_config(config_path)
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))

    sprite = make_sprite_face(cfg, config_path)
    pygame.init()
    bus = FaceBus.receiver(cfg.face.bus_port)
    try:
        screen = _open_display(cfg)
        log.info("display %dx%d", *screen.get_size())
        face = pygame.Surface((cfg.face.size, cfg.face.size))
        sprite_surf = pygame.Surface((sprite.bank.size,) * 2) if sprite else None
        clock = pygame.time.Clock()
        state, target, mouth, viseme = State.IDLE, 0.0, 0.0, "X"
        t0 = last = time.monotonic()
        next_blink = t0 + random.uniform(*BLINK_GAP_S)
        blink_start = -1.0
        reported_ready = False
        while not stop["flag"]:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN
                                              and ev.key == pygame.K_ESCAPE):
                    stop["flag"] = True
            msg = bus.poll()
            if msg is not None:
                if msg[0] != state:
                    log.info("face: %s", msg[0].value)
                state, target, viseme = msg  # the procedural mouth ignores the shape
            now = time.monotonic()
            dt, last = now - last, now
            speaking = state is State.SPEAKING
            mouth = step_mouth(mouth, target if speaking else 0.0, dt)
            if sprite is not None:
                sprite.update(dt, state, mouth, viseme if speaking else "X")
                rgb = sprite.render()  # (h, w, 3); surfarray is (w, h, 3)
                pygame.surfarray.blit_array(sprite_surf, np.ascontiguousarray(rgb.transpose(1, 0, 2)))
                if sprite_surf.get_size() == face.get_size():
                    face.blit(sprite_surf, (0, 0))
                else:
                    pygame.transform.smoothscale(sprite_surf, face.get_size(), face)
            else:
                if now >= next_blink:
                    blink_start = now
                    next_blink = now + BLINK_S + random.uniform(*BLINK_GAP_S)
                draw_face(face, state, mouth, now - t0, blink_amount(now - blink_start))
            if cfg.face.fullscreen:
                screen.fill((0, 0, 0))
                screen.blit(face, (0, 0))
            else:
                pygame.transform.scale(face, screen.get_size(), screen)
            pygame.display.flip()
            if not reported_ready:
                reported_ready = watchdog.ready() or True  # after the first frame
            watchdog.ping()  # rate-limited
            clock.tick(FPS)
    finally:
        bus.close()
        pygame.quit()


def main() -> None:
    ap = argparse.ArgumentParser(description="Mirror, Mirror face renderer")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    run(args.config)


if __name__ == "__main__":
    main()
