import pytest

from mirror.wake import is_hallucination, match_wake


@pytest.mark.parametrize("text,rem", [
    ("Mirror, mirror on the wall, who is the fairest?", "who is the fairest"),
    ("mirror mirror", ""),
    ("Mirrors mirror, tell me a joke", "tell me a joke"),
    ("מראה מראה שעל הקיר, מי הכי יפה?", "מי הכי יפה"),
    ("מַרְאָה מַרְאָה", ""),
    ("מרא מראה ספרי לי בדיחה", "ספרי לי בדיחה"),
])
def test_matches(text, rem):
    assert match_wake(text).remainder == rem


@pytest.mark.parametrize("text", ["the mirror is nice", "hello there", "מה קורה", "", "mirror"])
def test_no_match(text):
    assert match_wake(text) is None


@pytest.mark.parametrize("text", ["", "  ", "Thank you.", "תודה רבה.", "[BLANK_AUDIO]", "(wind blowing)", "you"])
def test_hallucinations_ignored(text):
    assert is_hallucination(text)
    assert match_wake(text) is None
