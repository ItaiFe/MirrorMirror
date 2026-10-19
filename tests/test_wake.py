import pytest

from mirror.wake import is_hallucination, match_wake


@pytest.mark.parametrize("text,rem", [
    ("Mirror, mirror on the wall, who is the fairest?", "who is the fairest"),
    ("mirror mirror", ""),
    ("Mirrors mirror, tell me a joke", "tell me a joke"),
    ("מראה מראה שעל הקיר, מי הכי יפה?", "מי הכי יפה"),
    ("מַרְאָה מַרְאָה", ""),
    ("מרא מראה ספרי לי בדיחה", "ספרי לי בדיחה"),
    # ivrit-ai Whisper turbo: near-misses, transliterations and a tripled wake word
    ("mirer mirror, what is my fate?", "what is my fate"),
    ("מיר מיר, מה אני אמור לעשות?", "מה אני אמור לעשות"),
    ("מירור מירר tell me", "tell me"),
    ("mirror mirror mirror who is fairest", "who is fairest"),
    ("Mirror mirror, mirror, Shall I change my job?", "shall i change my job"),
    ("Mirror, mirror, mirror on the wall, who is the fairest?", "who is the fairest"),
    ("מראה מראה מראה שעל הקיר מי הכי יפה", "מי הכי יפה"),
    ("mirror mirror mirror mirror will i find love", "will i find love"),
    ("Mirror mirror mirror mirror  mirror  around the wall, who", "around the wall who"),
    ("מראה מראה מראה מראה האם אמצא אהבה", "האם אמצא אהבה"),
    # only strong matches count as repeats: a fuzzy near-miss is a real first word
    ("mirror mirror, error codes?", "error codes"),
    ("mirror mirror minor key", "minor key"),
    ("מראה מראה מורה אמרה לי", "מורה אמרה לי"),
    ("מראה מראה מראש", "מראש"),
    ("מראה מראה מירה", "מירה"),
])
def test_matches(text, rem):
    assert match_wake(text).remainder == rem


@pytest.mark.parametrize("text", ["the mirror is nice", "hello there", "מה קורה", "", "mirror", "מיר",
                                  "a mirror mirror", "mirror is mirror"])
def test_no_match(text):
    assert match_wake(text) is None


@pytest.mark.parametrize("text", ["", "  ", "Thank you.", "תודה רבה.", "[BLANK_AUDIO]", "(wind blowing)", "you"])
def test_hallucinations_ignored(text):
    assert is_hallucination(text)
    assert match_wake(text) is None


def test_transliterated_on_the_wall_tail_is_stripped():
    # live on the Mac: English "Mirror, mirror on the wall" came out in Hebrew letters
    assert match_wake("מירר מירר על הוול").remainder == ""
    assert match_wake("מירר מירר און דה וול, מה צופן העתיד?").remainder == "מה צופן העתיד"
