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
    ("מראה מראה מירה", ""),  # מירה is her name now, a wake word
    # one wake word is enough
    ("mirror", ""),
    ("מיר", ""),
    ("Mirror, tell me a joke", "tell me a joke"),
    ("mirror is mirror", "is mirror"),
    ("mirror, error codes?", "error codes"),
    ("מראה, מה העתיד שלי?", "מה העתיד שלי"),
    # live on the Orin: the first "מראה" of two was lost
    ("המראה שעל הקיר", ""),
    ("מירו על הוול, מה צופן העתיד?", "מה צופן העתיד"),
    # live on the Orin: a cut start or a scrap of noise before the wake word
    ("רא, מרא, על פרסום.", "על פרסום"),
    ("אירוד מראה מראה שעל הקיר", ""),
    ("a mirror mirror, who is fairest?", "who is fairest"),
    # her name, and the wake word after background talk (noise at a festival)
    ("Mira, what's my future?", "what s my future"),
    ("מירה, מה העתיד שלי?", "מה העתיד שלי"),
    ("yeah so Mira what's my future", "what s my future"),
    ("כן אז אמרתי לו מראה מה שלומך", "מה שלומך"),
    ("the mirror is nice", "is nice"),
    ("מה יש במראה הזאת", "הזאת"),
    ("תגידי למירה שלום", "שלום"),
])
def test_matches(text, rem):
    assert match_wake(text).remainder == rem


@pytest.mark.parametrize("text", ["hello there", "מה קורה", "", "אני בשמירה הלילה",
                                  "אמר אשר על הקיר", "I like your outfit",
                                  # near-misses that are real Hebrew words don't wake it
                                  "תראה בגדול אפשר להביא עוד שניים", "מראש", "מורה"])
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
