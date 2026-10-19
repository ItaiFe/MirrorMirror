"""Bilingual (English/Hebrew) wake-phrase matching for "mirror, mirror"."""
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher

_THRESHOLD = 0.75
# Looser for "mirror" itself: ivrit-ai's Whisper turbo writes "mirer" (0.73).
_MIRROR_THRESHOLD = 0.7
_HEBREW_VARIANTS = {"מראה", "מראא", "מרא", "מירא"}
# English "mirror" transliterated into Hebrew letters by the Hebrew fine-tune.
_TRANSLITERATED_MIRROR = {"מיר", "מירור", "מירר"}
# Repeats after the first two wake tokens are stripped only on these exact spellings.
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
            or token in _HEBREW_VARIANTS or _close(token, "מראה"))


def is_hallucination(text: str) -> bool:
    if _TAG_RE.match(text):
        return True
    norm = normalize(text)
    return not norm or norm in _HALLUCINATIONS


def match_wake(text: str) -> "WakeMatch | None":
    if is_hallucination(text):
        return None
    tokens = normalize(text).split()
    if len(tokens) < 2 or not (_is_wake_word(tokens[0]) and _is_wake_word(tokens[1])):
        return None
    rest = tokens[2:]
    # ivrit-ai writes "mirror" up to 5 times. Only exact spellings count as repeats,
    # so a fuzzy near-miss ("error", "מורה") stays as the question's first word.
    while rest and rest[0] in _REPEAT_WORDS:
        rest = rest[1:]
    for tail in _TAILS:
        n = len(tail)
        if len(rest) >= n and all(_close(t, w) for t, w in zip(rest[:n], tail)):
            rest = rest[n:]
            break
    return WakeMatch(remainder=" ".join(rest))
