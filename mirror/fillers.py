"""Pre-synthesized lines: the fillers the oracle speaks while the LLM is thinking,
and (a second bank) the idle lures."""
from __future__ import annotations

import logging
import random

import numpy as np

from mirror.visemes import VisemeTrack

log = logging.getLogger("mirror.fillers")

Clip = tuple[np.ndarray, int, "VisemeTrack | None"]


class FillerBank:
    def __init__(self, tts, lines: dict[str, list[str]], rng: random.Random | None = None,
                 max_clip_s: float | None = None, kind: str = "filler"):
        self._tts = tts
        self._kind = kind  # "filler" | "lure": names the lines in the log
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
                    log.exception("could not synthesize %s %r", self._kind, text)
                    continue
                seconds = len(samples) / sr
                log.debug("%s (%s, %.2f s): %r", self._kind, lang, seconds, text)
                if self._max_clip_s is not None and seconds > self._max_clip_s:
                    log.warning("%s (%s) is %.1f s, over the %.1f s budget: %r",
                                self._kind, lang, seconds, self._max_clip_s, text)
                clips.append((samples, sr, track))
            self._clips[lang] = clips

    def languages(self) -> list[str]:
        """The languages that have at least one prepared clip."""
        return [lang for lang, clips in self._clips.items() if clips]

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
