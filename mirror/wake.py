"""Bilingual (English/Hebrew) wake-phrase matching: "mirror" / "מראה", once or more."""
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher

_THRESHOLD = 0.75
# Looser for "mirror" itself: ivrit-ai's Whisper turbo writes "mirer" (0.73).
_MIRROR_THRESHOLD = 0.7
# Hebrew only by these exact spellings: a near-miss is a real word ("תראה", look!)
_HEBREW_VARIANTS = {"מראה", "המראה", "ומראה", "מראא", "מרא", "מירא"}
# English "mirror" transliterated into Hebrew letters by the Hebrew fine-tune.
_TRANSLITERATED_MIRROR = {"מיר", "מירור", "מירר", "מירו"}
# Repeats after the first wake token are stripped only on these exact spellings.
_REPEAT_WORDS = {"mirror", "mirrors"} | _TRANSLITERATED_MIRROR | _HEBREW_VARIANTS
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
    return (_close(token, "mirror", _MIRROR_THRESHOLD) or token in _TRANSLITERATED_MIRROR
            or token in _HEBREW_VARIANTS)


def is_hallucination(text: str) -> bool:
    if _TAG_RE.match(text):
        return True
    norm = normalize(text)
    return not norm or norm in _HALLUCINATIONS


def _wake_start(tokens: list) -> "int | None":
    """Where the wake word is: the first word (one "mirror" / "מראה" is enough; live,
    the first of two was often lost), or the second after one scrap of noise or a cut
    start ("רא, מרא", "אירוד מראה"), if it is an exact Hebrew spelling or a doubled
    "mirror" ("a mirror mirror", not "the mirror is nice")."""
    if tokens and _is_wake_word(tokens[0]):
        return 0
    if len(tokens) >= 2 and (tokens[1] in _HEBREW_VARIANTS
                             or (_is_wake_word(tokens[1]) and len(tokens) >= 3
                                 and _is_wake_word(tokens[2]))):
        return 1
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
