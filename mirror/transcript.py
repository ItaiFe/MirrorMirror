"""The conversation log: one JSON line per exchange (what the visitor said, as
transcribed, and the mirror's answer), in one file per day. Text only, no audio.
Each line is flushed and fsynced, so a power cut loses at most the line being
written."""
from __future__ import annotations

import datetime
import json
import logging
import os
from typing import Callable

log = logging.getLogger("mirror.transcript")


class ConversationLog:
    def __init__(self, directory: str,
                 wall_clock: Callable[[], datetime.datetime] = datetime.datetime.now):
        self._dir = directory
        self._wall_clock = wall_clock
        self._warned = False

    def write(self, heard: str, reply: str, lang: str, **extra) -> None:
        """Append one exchange. Never raises: a full or missing card only logs."""
        now = self._wall_clock()
        record = {"time": now.isoformat(timespec="seconds"), "heard": heard,
                  "reply": reply, "lang": lang, **extra}
        path = os.path.join(self._dir, f"{now:%Y-%m-%d}.jsonl")
        try:
            os.makedirs(self._dir, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
        except OSError as e:
            if not self._warned:  # once: a broken card would repeat it every exchange
                log.warning("could not save the conversation to %s: %s", path, e)
                self._warned = True
