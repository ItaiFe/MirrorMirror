import random

import numpy as np

from mirror.fillers import FillerBank


class FakeTTS:
    def __init__(self):
        self.calls = []

    def synth_visemes(self, text, lang):
        self.calls.append((text, lang))
        return np.full(100, len(self.calls), dtype=np.float32), 22050, None


def test_prepare_synthesizes_every_line():
    tts = FakeTTS()
    bank = FillerBank(tts, {"en": ["a", "b"], "he": ["א"]})
    bank.prepare()
    assert sorted(tts.calls) == [("a", "en"), ("b", "en"), ("א", "he")]
    samples, sr, track = bank.pick("he")
    assert sr == 22050 and len(samples) == 100 and track is None


def test_pick_never_repeats():
    tts = FakeTTS()
    bank = FillerBank(tts, {"en": ["a", "b", "c"]}, rng=random.Random(1))
    bank.prepare()
    picks = [int(bank.pick("en")[0][0]) for _ in range(50)]
    assert all(a != b for a, b in zip(picks, picks[1:]))
    assert set(picks) == {1, 2, 3}


def test_pick_unknown_lang_none():
    bank = FillerBank(FakeTTS(), {"en": ["a"]})
    bank.prepare()
    assert bank.pick("fr") is None
    assert FillerBank(FakeTTS(), {"en": ["a"]}).pick("en") is None  # not prepared


def test_failing_line_is_skipped():
    class Flaky(FakeTTS):
        def synth_visemes(self, text, lang):
            if text == "bad":
                raise RuntimeError("no voice")
            return super().synth_visemes(text, lang)

    bank = FillerBank(Flaky(), {"en": ["bad", "good"]})
    bank.prepare()
    assert bank.pick("en") is not None
