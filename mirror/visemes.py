"""Mouth shapes (visemes) from Piper's phoneme alignments.

Shape letters follow the Preston Blair / Rhubarb set:
  A closed (m b p)        B slightly open, most consonants, "ee"
  C open ("eh", "ae")     D wide open ("aa")
  E slightly rounded ("o", "er")    F puckered ("oo", "w")
  G teeth on lip (f v)    H tongue up (l)    X rest / silence
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

FPS = 30
FRAME_S = 1.0 / FPS
MIN_HOLD_S = 2 * FRAME_S  # shorter unprotected segments are absorbed into the previous one
# Brief consonant shapes that must stay visible (closed lips, teeth on lip, tongue up,
# pucker): never absorbed, and lengthened to at least one frame.
PROTECTED = frozenset("AFGH")

_SHAPES = {
    "A": "m b p ɱ",
    "B": ("t d k ɡ g n ŋ ɲ s z ʃ ʒ ɕ ʑ ʔ h ɦ j ç x χ ʁ ʀ ɹ r ɾ ɻ θ ð c ɟ q ɢ ħ ʕ ɣ "
          "i ɪ ɨ ᵻ"),
    "C": "e ɛ æ ə ɐ ʌ ɘ ɵ ɜ",
    "D": "a ɑ ɶ",
    "E": "o ɔ ɒ ɚ ɝ ø œ ɞ",
    "F": "u ʊ ʉ w ɯ y ʏ ʍ ɥ",
    "G": "f v ʋ ɸ β",
    "H": "l ɫ ɬ ɮ ɭ ʎ ʟ",
    # Piper's sentence start/end/pad, word gaps and punctuation
    "X": "^ $ _ , . ! ? ; : - … ( ) \" ' « » „ “ ” —",
}
PHONEME_TO_SHAPE: dict[str, str] = {
    p: shape for shape, phonemes in _SHAPES.items() for p in phonemes.split()}
PHONEME_TO_SHAPE[" "] = "X"
UNKNOWN_SHAPE = "B"

# Piper gives these marks their own durations. Stress marks belong to the
# phoneme that follows them; length marks and modifiers to the one before.
_JOIN_NEXT = {"ˈ", "ˌ"}
_JOIN_PREVIOUS = {"ː", "ˑ", "˞", "ʰ", "ʲ", "ʷ", "̃", "̩", "̯", "͡"}


def rms_shape(level: float) -> str:
    """Fallback shape from the mouth level alone (no alignments)."""
    if level < 0.08:
        return "X"
    if level < 0.35:
        return "B"
    if level < 0.65:
        return "C"
    return "D"


@dataclass
class VisemeTrack:
    times: np.ndarray  # shape start times in seconds, sorted
    shapes: list[str]
    end: float = field(default=math.inf)  # end of the last shape

    def at(self, t: float) -> str:
        """The shape in effect at t seconds, or "X" outside the track."""
        if not self.shapes or t < self.times[0] or t >= self.end:
            return "X"
        return self.shapes[int(np.searchsorted(self.times, t, side="right")) - 1]


def _shape_segments(alignments: list[tuple[str, int]]) -> list[list]:
    """[(phoneme, samples)] -> [[shape, samples]] with marks folded into neighbours."""
    segs: list[list] = []
    carry = 0
    for phoneme, n in alignments:
        if phoneme in _JOIN_NEXT:
            carry += n
        elif phoneme in _JOIN_PREVIOUS and segs:
            segs[-1][1] += n
        else:
            shape = "X" if phoneme in _JOIN_PREVIOUS else PHONEME_TO_SHAPE.get(
                phoneme, UNKNOWN_SHAPE)
            segs.append([shape, n + carry])
            carry = 0
    if carry and segs:
        segs[-1][1] += carry
    return segs


def track_from_alignments(alignments: list[tuple[str, int]], sr: int,
                          total_samples: int) -> VisemeTrack:
    """Build a viseme track from per-phoneme sample counts.

    The counts are rescaled to total_samples (the real audio length) and runs of
    the same shape are merged. PROTECTED shapes shorter than a frame are lengthened
    to one frame, taking the time from the following segment (or the previous one).
    Any other segment but the first that is shorter than MIN_HOLD_S is absorbed into
    the previous one. A one-frame B is put between adjacent A and D when both last
    at least MIN_HOLD_S, so each can give up half a frame.
    """
    total = sum(n for _, n in alignments)
    end = total_samples / sr
    if total <= 0 or total_samples <= 0:
        return VisemeTrack(np.zeros(0), [], end)
    scale = total_samples / (total * sr)  # samples -> seconds of real audio

    raw: list[list] = []  # [shape, duration_s], identical neighbours merged
    for shape, n in _shape_segments(alignments):
        if raw and shape == raw[-1][0]:
            raw[-1][1] += n * scale
        else:
            raw.append([shape, n * scale])

    # Protected shapes last at least one frame, borrowing from the next segment
    # (or the previous one when the next can't spare it).
    for i, seg in enumerate(raw):
        need = FRAME_S - seg[1]
        if seg[0] not in PROTECTED or need <= 0:
            continue
        for j in (i + 1, i - 1):
            if 0 <= j < len(raw) and raw[j][1] >= need:
                raw[j][1] -= need
                seg[1] = FRAME_S
                break

    segs: list[list] = []
    for shape, dur in raw:
        if dur <= 0:
            continue
        if segs and (shape == segs[-1][0] or (shape not in PROTECTED and dur < MIN_HOLD_S)):
            segs[-1][1] += dur
        else:
            segs.append([shape, dur])

    times: list[float] = []
    out: list[str] = []
    start, shift = 0.0, 0.0  # shift: a B before this segment took its first half frame
    for i, (shape, dur) in enumerate(segs):
        times.append(start + shift)
        out.append(shape)
        start += dur
        shift = 0.0
        if i + 1 < len(segs) and {shape, segs[i + 1][0]} == {"A", "D"} \
                and min(dur, segs[i + 1][1]) >= MIN_HOLD_S:
            times.append(start - FRAME_S / 2)
            out.append("B")
            shift = FRAME_S / 2
    return VisemeTrack(np.array(times), out, end)
