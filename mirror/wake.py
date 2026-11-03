"""Bilingual (English/Hebrew) wake-phrase matching: "mirror" / "מראה", once or more."""
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher

_THRESHOLD = 0.75
# Looser for "mirror" itself: ivrit-ai's Whisper turbo writes "mirer" (0.73).
_MIRROR_THRESHOLD = 0.7
# Hebrew only by these exact spellings: a near-miss is a real word ("תראה", look!)
_HEBREW_VARIANTS = {"מראה", "המראה", "ומראה", "מראא", "מרא", "מירא",
                    "מירה", "ומירה", "מיירה"}  # מירה: her name, Mira
# English "mirror" transliterated into Hebrew letters by the Hebrew fine-tune.
_TRANSLITERATED_MIRROR = {"מיר", "מירור", "מירר", "מירו"}
# Repeats after the first wake token are stripped only on these exact spellings.
# Her name in Latin letters, as whisper writes it
_ENGLISH_NAMES = {"mira", "meera", "mirah", "miera"}
_REPEAT_WORDS = {"mirror", "mirrors"} | _ENGLISH_NAMES | _TRANSLITERATED_MIRROR | _HEBREW_VARIANTS
# Spelled exactly like this, the wake word counts anywhere in what was heard: in noise
# a scrap of background talk often comes before it ("…yeah Mira what's my future")
_EXACT_WAKE = _REPEAT_WORDS
_TAILS = (("on", "the", "wall"), ("שעל", "הקיר"), ("על", "הקיר"),
          # English "on the wall" in Hebrew letters (heard live: "מירר מירר על הוול")
          ("און", "דה", "וול"), ("על", "הוול"), ("על", "דה", "וול"))
_HALLUCINATIONS = {"thank you", "thanks for watching", "you", "תודה", "תודה רבה", "תודה שצפיתם"}
_TAG_RE = re.compile(r"^\s*(\[[^\]]*\]|\([^)]*\))\s*$")
_NIQQUD_RE = re.compile("[֑-ׇ]")


@dataclass
class WakeMatch:
    remainder: str
    english: bool = False  # "mirror" in Latin or in Hebrew letters ("מירר"), not "מראה"


def normalize(text: str) -> str:
    """Lowercase, strip Hebrew niqqud, turn punctuation into spaces, collapse whitespace."""
    text = _NIQQUD_RE.sub("", text.lower())
    text = "".join(
        ch if unicodedata.category(ch)[0] in "LN" else " " for ch in text
    )
    return " ".join(text.split())


def _close(a: str, b: str, threshold: float = _THRESHOLD) -> bool:
    return SequenceMatcher(None, a, b).ratio() >= threshold


def _is_wake_word(token: str) -> bool:
    return (_close(token, "mirror", _MIRROR_THRESHOLD) or token in _EXACT_WAKE)


def is_hallucination(text: str) -> bool:
    if _TAG_RE.match(text):
        return True
    norm = normalize(text)
    return not norm or norm in _HALLUCINATIONS


def _wake_start(tokens: list) -> "int | None":
    """Where the wake word is: the first word, matched loosely ("mirer"; one "mirror" /
    "מראה" / "Mira" is enough), or the first exact spelling anywhere after it: in the
    noise of a festival, background talk often comes first ("רא, מרא", "אירוד מראה",
    "yeah so Mira what's my future"). The trade-off: "the mirror is nice" wakes it too."""
    if tokens and _is_wake_word(tokens[0]):
        return 0
    for i, token in enumerate(tokens[1:], start=1):
        if token in _EXACT_WAKE:
            return i
    return None


def match_wake(text: str) -> "WakeMatch | None":
    if is_hallucination(text):
        return None
    tokens = normalize(text).split()
    start = _wake_start(tokens)
    if start is None:
        return None
    tokens = tokens[start:]
    rest = tokens[1:]
    # ivrit-ai writes "mirror" up to 5 times. Only exact spellings count as repeats,
    # so a fuzzy near-miss ("error", "מורה") stays as the question's first word.
    while rest and rest[0] in _REPEAT_WORDS:
        rest = rest[1:]
    for tail in _TAILS:
        n = len(tail)
        if len(rest) >= n and all(_close(t, w) for t, w in zip(rest[:n], tail)):
            rest = rest[n:]
            break
    english = tokens[0] in _TRANSLITERATED_MIRROR or tokens[0].isascii()
    return WakeMatch(remainder=" ".join(rest), english=english)
