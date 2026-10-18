"""Mouth shapes (visemes) from Piper's phoneme alignments, or estimated from IPA and
the audio's loudness for engines that give no phoneme timings (Pocket TTS).

Shape letters follow the Preston Blair / Rhubarb set:
  A closed (m b p)        B slightly open, most consonants, "ee"
  C open ("eh", "ae")     D wide open ("aa")
  E slightly rounded ("o", "er")    F puckered ("oo", "w")
  G teeth on lip (f v)    H tongue up (l)    X rest / silence
"""
from __future__ import annotations

import math
import re
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


def _can_give(seg: list, need: float) -> bool:
    left = seg[1] - need
    return left >= FRAME_S if seg[0] in PROTECTED else left > 0


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
    # (or the previous one when the next can't spare it). A donor must stay legal:
    # a protected one keeps at least a frame, an unprotected one stays above 0.
    # With no legal donor the segment is kept as it is.
    for i, seg in enumerate(raw):
        need = FRAME_S - seg[1]
        if seg[0] not in PROTECTED or need <= 0:
            continue
        for j in (i + 1, i - 1):
            if 0 <= j < len(raw) and _can_give(raw[j], need):
                raw[j][1] -= need
                seg[1] = FRAME_S
                break

    segs: list[list] = []
    for shape, dur in raw:
        if dur <= 0:
            continue  # a phoneme Piper gave no samples
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


# --- Estimated tracks: IPA spread over the voiced parts of the audio -----------------

ENVELOPE_HOP_S = 0.02     # loudness envelope resolution
VOICED_THRESHOLD = 0.08   # of the envelope's 95th percentile
VOICED_FLOOR = 1e-4       # absolute: below this RMS is silence whatever the percentile
MIN_GAP_S = 0.07          # shorter silences inside speech are closed
MIN_VOICED_S = 0.04       # shorter voiced islands are dropped
_PAUSE_PUNCT = ",.?!…;:—–"
_IPA_TOKENS = re.compile(rf"[^\s{_PAUSE_PUNCT}]+|[{_PAUSE_PUNCT}]")
_VOWELS = frozenset("aeiouəɛɔɪʊæɑʌɐɜøœyɒɚɘɵɤɯɨʉᵻ")
_VOWEL_WEIGHT, _CONSONANT_WEIGHT, _WORD_GAP_WEIGHT = 1.4, 0.9, 0.15
_MARKS = _JOIN_NEXT | _JOIN_PREVIOUS


def voiced_segments(samples: np.ndarray, sr: int) -> list[list[float]]:
    """[[start_s, end_s]] of the audio's voiced stretches, from a 20 ms RMS envelope."""
    a = np.asarray(samples, dtype=np.float32).reshape(-1)
    hop = max(1, int(round(sr * ENVELOPE_HOP_S)))
    n = len(a) // hop
    if n == 0:
        return []
    rms = np.sqrt((a[: n * hop].reshape(n, hop) ** 2).mean(axis=1))
    rms = np.convolve(rms, np.ones(3) / 3, "same")
    on = rms > max(VOICED_THRESHOLD * float(np.percentile(rms, 95)), VOICED_FLOOR)
    segs: list[list[float]] = []
    i = 0
    while i < n:
        if not on[i]:
            i += 1
            continue
        j = i
        while j < n and on[j]:
            j += 1
        start, end = i * hop / sr, j * hop / sr
        if segs and start - segs[-1][1] < MIN_GAP_S:
            segs[-1][1] = end
        else:
            segs.append([start, end])
        i = j
    return [s for s in segs if s[1] - s[0] >= MIN_VOICED_S]


def _ipa_groups(ipa: str) -> list[list[tuple[str, float]]]:
    """IPA -> punctuation-delimited groups of weighted phonemes (word gaps included)."""
    groups, words = [], []
    for token in _IPA_TOKENS.findall(ipa) + ["."]:
        if token in _PAUSE_PUNCT:
            units: list[tuple[str, float]] = []
            for word in words:
                phonemes = [(c, _VOWEL_WEIGHT if c in _VOWELS else _CONSONANT_WEIGHT)
                            for c in word if c not in _MARKS]
                if phonemes:
                    if units:
                        units.append((" ", _WORD_GAP_WEIGHT))
                    units.extend(phonemes)
            if units:
                groups.append(units)
            words = []
        else:
            words.append(token)
    return groups


def estimate_track(ipa: str, samples: np.ndarray, sr: int) -> VisemeTrack | None:
    """A viseme track for audio with no phoneme timings: each punctuation group of the
    IPA goes to the voiced audio between the longest pauses, its phonemes spread over
    that voiced time (vowels a little longer than consonants), then the usual
    track_from_alignments hold rules. Silence is rest (X). None when the IPA has no
    phonemes (the face then shapes the mouth from loudness)."""
    total = len(samples)
    end = total / sr if sr else 0.0
    groups = _ipa_groups(ipa)
    if not groups:
        return None
    segs = voiced_segments(samples, sr)
    if not segs:
        return VisemeTrack(np.array([0.0]), ["X"], end)

    # The k-1 longest pauses split the voiced stretches among the first k groups;
    # with more groups than stretches the extra groups join the last one.
    k = min(len(groups), len(segs))
    gaps = sorted(range(len(segs) - 1), key=lambda i: segs[i + 1][0] - segs[i][1],
                  reverse=True)[: k - 1]
    seg_groups, first = [], 0
    for cut in sorted(gaps) + [len(segs) - 1]:
        seg_groups.append(segs[first: cut + 1])
        first = cut + 1
    if len(groups) > k:
        groups = groups[: k - 1] + [[u for g in groups[k - 1:] for u in g]]

    spans: list[tuple[str, float, float]] = []  # (phoneme, start_s, end_s), in order
    for units, stretch in zip(groups, seg_groups):
        voiced = sum(e - b for b, e in stretch)
        weight = sum(w for _, w in units)
        bounds = np.concatenate([[0.0], np.cumsum([w for _, w in units])]) * voiced / weight
        offset = 0.0  # voiced time before this stretch, within the group
        for b, e in stretch:
            lo, hi = offset, offset + (e - b)
            for (phoneme, _), p0, p1 in zip(units, bounds[:-1], bounds[1:]):
                s0, s1 = max(p0, lo), min(p1, hi)
                if s1 > s0:
                    spans.append((phoneme, b + s0 - lo, b + s1 - lo))
            offset = hi

    alignments: list[tuple[str, int]] = []
    t = 0.0
    for phoneme, b, e in spans:
        if b > t:
            alignments.append(("_", int(round((b - t) * sr))))
        alignments.append((phoneme, int(round((e - b) * sr))))
        t = e
    if end > t:
        alignments.append(("$", int(round((end - t) * sr))))
    return track_from_alignments(alignments, sr, total)
