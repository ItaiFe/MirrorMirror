import itertools
import os

from mirror.face.app import step_mouth


def test_step_mouth():
    assert step_mouth(0.0, 1.0, 0.06) >= 0.6   # 60 ms attack
    assert step_mouth(1.0, 0.0, 0.06) >= 0.3   # 120 ms release is slower
    for cur, tgt, dt in itertools.product((0.0, 0.3, 1.0), (-0.5, 0.0, 0.7, 1.0, 2.0),
                                          (0.0, 0.033, 0.06, 1.0, 10.0)):
        assert 0.0 <= step_mouth(cur, tgt, dt) <= 1.0


def test_make_sprite_face_from_config(tmp_path):
    import pytest

    from mirror.config import load_config
    from mirror.face.app import make_sprite_face
    from mirror.face.sprite import SpriteFace

    cfg = load_config("config/mac.toml")
    # bank = "assets/face/stub" resolves against the repo root, not the cwd
    face = make_sprite_face(cfg, os.path.abspath("config/mac.toml"))
    assert isinstance(face, SpriteFace)
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        assert isinstance(make_sprite_face(cfg, os.path.join(cwd, "config/mac.toml")), SpriteFace)
    finally:
        os.chdir(cwd)
    cfg.face.renderer = "procedural"
    assert make_sprite_face(cfg, "config/mac.toml") is None
    cfg.face.renderer = "bogus"
    with pytest.raises(ValueError):
        make_sprite_face(cfg, "config/mac.toml")
