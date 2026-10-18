"""Pre-synthesized filler lines the oracle speaks while the LLM is thinking."""
from __future__ import annotations

import logging
import random

import numpy as np

from mirror.visemes import VisemeTrack

log = logging.getLogger("mirror.fillers")

Clip = tuple[np.ndarray, int, "VisemeTrack | None"]


class FillerBank:
    def __init__(self, tts, lines: dict[str, list[str]], rng: random.Random | None = None,
                 max_clip_s: float | None = None):
        self._tts = tts
        self._max_clip_s = max_clip_s  # warn about longer clips (None: no check)
        self._lines = lines
        self._rng = rng or random.Random()
        self._clips: dict[str, list[Clip]] = {}
        self._last: dict[str, int] = {}

    def prepare(self) -> None:
        """Synthesize every line (once, at startup). A line that fails is skipped."""
        for lang, lines in self._lines.items():
            clips = []
            for text in lines:
                try:
                    samples, sr, track = self._tts.synth_visemes(text, lang)
                except Exception:
                    log.exception("could not synthesize filler %r", text)
                    continue
                seconds = len(samples) / sr
                log.debug("filler (%s, %.2f s): %r", lang, seconds, text)
                if self._max_clip_s is not None and seconds > self._max_clip_s:
                    log.warning("filler (%s) is %.1f s, over the %.1f s budget: %r",
                                lang, seconds, self._max_clip_s, text)
                clips.append((samples, sr, track))
            self._clips[lang] = clips

    def pick(self, lang: str) -> Clip | None:
        """A random cached clip for lang, never the same one twice in a row;
        None if there is none."""
        clips = self._clips.get(lang)
        if not clips:
            return None
        choices = [i for i in range(len(clips)) if i != self._last.get(lang)] or [0]
        i = self._rng.choice(choices)
        self._last[lang] = i
        return clips[i]
