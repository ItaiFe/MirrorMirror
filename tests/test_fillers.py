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


class LongTTS(FakeTTS):
    def synth_visemes(self, text, lang):
        self.calls.append((text, lang))
        return np.zeros(int(len(text) * 22050), np.float32), 22050, None  # 1 s per char


def test_long_filler_warns_when_a_budget_is_set(caplog):
    bank = FillerBank(LongTTS(), {"en": ["ab", "abc"]}, max_clip_s=2.5)
    with caplog.at_level("WARNING", logger="mirror.fillers"):
        bank.prepare()
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1 and "abc" in warnings[0] and "3.0" in warnings[0]
    assert bank.pick("en") is not None  # still used: a warning, not a drop


def test_no_filler_budget_by_default(caplog):
    with caplog.at_level("WARNING", logger="mirror.fillers"):
        FillerBank(LongTTS(), {"en": ["abcdef"]}).prepare()
    assert not [r for r in caplog.records if r.levelname == "WARNING"]


def test_languages_lists_only_languages_with_clips():
    class Flaky(FakeTTS):
        def synth_visemes(self, text, lang):
            if lang == "he":
                raise RuntimeError("no hebrew voice")
            return super().synth_visemes(text, lang)

    bank = FillerBank(Flaky(), {"en": ["a"], "he": ["א"]})
    assert bank.languages() == []  # not prepared
    bank.prepare()
    assert bank.languages() == ["en"]


def test_lure_copy():
    from mirror.persona import LURES, singular_address

    assert LURES["en"] == [
        "Come closer, traveller… the glass has something to show you.",
        "I see you there. Ask, and the mirror will answer.",
        "Mirror, mirror… say the words, and I will wake.",
        "The dust remembers you. Come, ask me anything.",
    ]
    assert LURES["he"] == [
        "התקרבו, נוסעים… לזכוכית יש משהו להראות לכם.",
        "אני רואה אתכם. שאלו, והמראה תענה.",
        "מראה, מראה… אמרו את המילים, ואתעורר.",
        "האבק זוכר אתכם. בואו, שאלו אותי הכול.",
    ]
    for line in LURES["he"]:
        assert singular_address(line) == [], line  # the crowd, in plural
