import itertools

from mirror.face.app import step_mouth


def test_step_mouth():
    assert step_mouth(0.0, 1.0, 0.06) >= 0.6   # 60 ms attack
    assert step_mouth(1.0, 0.0, 0.06) >= 0.3   # 120 ms release is slower
    for cur, tgt, dt in itertools.product((0.0, 0.3, 1.0), (-0.5, 0.0, 0.7, 1.0, 2.0),
                                          (0.0, 0.033, 0.06, 1.0, 10.0)):
        assert 0.0 <= step_mouth(cur, tgt, dt) <= 1.0
