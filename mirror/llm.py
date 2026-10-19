"""LLM client for llama.cpp's llama-server (OpenAI-compatible API)."""
from __future__ import annotations

import json
import logging
import re
import threading
import time
import unicodedata
from datetime import datetime
from typing import Callable, Iterator

import requests

from mirror.config import LlmConfig
from mirror.persona import Angle, angle_note, system_prompt

log = logging.getLogger("mirror.llm")

_STAGE_DIRECTIONS = re.compile(r"\*[^*]*\*|\[[^\]]*\]")
_PARENTHETICAL = re.compile(r"\(([^()]*)\)")
_SENTENCE_END = ".!?…"
# words that make a parenthetical at the start or end of a sentence a stage
# direction: "(laughs)", "(leans in)", "(בשקט)". Only these: "(July)", "(only)" stay.
_DIRECTION_WORDS = frozenset({
    "softly", "quietly", "slowly", "gently", "warmly", "sadly", "tenderly",
    "playfully", "mysteriously", "dramatically", "thoughtfully", "solemnly",
    "knowingly", "wistfully", "lightly", "suddenly", "mischievously",
    "laughs", "laugh", "laughing", "chuckles", "chuckle", "chuckling", "giggles",
    "smiles", "smile", "smiling", "grins", "grin", "grinning", "sighs", "sigh",
    "winks", "wink", "winking", "nods", "nod", "pause", "pauses", "beat", "silence",
    "whispers", "whispering", "leans", "leaning", "murmurs", "murmuring", "hums",
    "shrugs", "beams", "ahem", "gasps", "laughter", "smirks",
    "בשקט", "בלחש", "בלחישה", "בחיוך", "בקריצה", "בעדינות", "ברכות", "לוחש",
    "מחייך", "צוחק", "מצחקק", "קורץ", "נאנח", "רוכן", "מהסס", "ממלמל", "הפסקה",
    "שתיקה", "צחוק", "אנחה", "חיוך", "קריצה",
})
# A reply that opens with a label: the few-shot format ("תשובה: ...", "Answer - ...")
# or a speaker name ("האורקל: ...", "Oracle: ...", "Mirror - ...")
_ANSWER_LABEL = re.compile(
    r"^\s*(?:תשובה|answer|האורקל|אורקל|oracle|המראה|mirror)\s*[:\-–—]\s*",
    re.IGNORECASE)
_STRAY_MARKS = re.compile(r"[*\[\]()]")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+")
_JOINERS = {"‍", "︎", "️"}  # ZWJ and variation selectors
# llama-server runs with -np 2: each language keeps its own slot, so its long
# system prompt stays cached there between requests.
LANG_SLOTS = {"en": 0, "he": 1}
_MAX_TOKENS = 120
_SAMPLING = LlmConfig().sampling()
# Errors that mean the request or its streamed response failed (incl. bad JSON shape).
_REQUEST_ERRORS = (requests.exceptions.RequestException, ValueError, KeyError,
                   IndexError, TypeError, AttributeError)


class LLMError(Exception):
    """Raised when the llama-server request fails or returns an empty reply."""


def server_unreachable(e: BaseException) -> bool:
    """True if the LLMError came from a failed connection (llama-server down or
    restarting, so its prompt caches are gone)."""
    return isinstance(e.__cause__, requests.exceptions.ConnectionError)


def _strip_emoji(text: str) -> str:
    return "".join(
        ch for ch in text
        if ch not in _JOINERS and unicodedata.category(ch) not in ("So", "Sk")
    )


def _clean(text: str) -> str:
    """Drop an "Answer:" label, stage directions (*smiles*, [winks], (sighs)), stray
    *[]() marks and emoji; collapse whitespace. A parenthetical that is not a stage
    direction keeps its words: "it doesn't (hold) a pattern" -> "it doesn't hold a
    pattern"."""
    text = _ANSWER_LABEL.sub("", text)
    text = _STAGE_DIRECTIONS.sub(" ", text)
    text = _PARENTHETICAL.sub(_parenthetical, text)
    text = _STRAY_MARKS.sub("", text)
    text = _strip_emoji(text)
    text = " ".join(text.split())
    return _SPACE_BEFORE_MARK.sub(r"\1", text)


_SPACE_BEFORE_MARK = re.compile(r"(?<=\S) +([.,!?;:…])")


def _parenthetical(m: "re.Match") -> str:
    """A stage direction goes: a parenthetical that is a sentence of its own, or
    one at the start or end of a sentence that reads like a direction. Any other
    keeps its words, without the brackets."""
    inner = m.group(1).strip()
    before = m.string[:m.start()].rstrip()
    after = m.string[m.end():].lstrip()
    at_start = not before or before[-1] in _SENTENCE_END
    at_end = not after or after[0] in _SENTENCE_END
    if not inner or (at_start and (at_end or inner[-1] in _SENTENCE_END)):
        return " "
    if (at_start or at_end) and _looks_like_direction(inner):
        return " "
    return " " + inner + " "


def _looks_like_direction(inner: str) -> bool:
    words = [w.strip(".,!?;:…") for w in inner.split()]
    if not words or len(words) > 4:
        return False
    return any(w.lower() in _DIRECTION_WORDS for w in words)


# The hidden angle note (persona.angle_note) echoed back, whole, cut short by
# max_tokens or without its brackets: never spoken.
_ECHO_START = ("(answer", "answer as", "(ענה", "ענה ב")
_ECHO_ANYWHERE = ("answering their actual question", "פתח בתשובה לשאלה")


_BRACKETED_ECHO = re.compile(r"\(\s*(?:answer|ענה)\b[^()]*\)", re.IGNORECASE)


def _is_angle_echo(sentence: str) -> bool:
    s = sentence.lstrip(" \t\n*[").lower()
    return s.startswith(_ECHO_START) or any(e in s for e in _ECHO_ANYWHERE)


def _cut(text: str, max_chars: int) -> str:
    """Shorten text to at most max_chars at a word boundary, ending in "…"."""
    if max_chars < 2:
        return ""
    cut = text[: max_chars - 1]
    space = cut.rfind(" ")
    if space > 0:
        cut = cut[:space]
    return cut.rstrip(" ,;:.!?") + "…"


def _unclosed(text: str) -> bool:
    """True while text has an open *stage direction*, [bracket] or (parenthesis)."""
    return (text.count("*") % 2 == 1 or text.count("[") > text.count("]")
            or text.count("(") > text.count(")"))


def _pop_sentences(buf: str) -> tuple[list[str], str]:
    """Complete sentences at the start of buf, and the rest. A sentence end inside an
    unclosed stage direction or bracket is not a split point."""
    done, start = [], 0
    for m in _SENTENCE_SPLIT.finditer(buf):
        piece = buf[start:m.start()]
        if not _unclosed(piece):
            done.append(piece)
            start = m.end()
    return done, buf[start:]


def _clean_sentence(raw: str) -> str:
    """A cleaned sentence, or "" if it is an echo of the angle note."""
    raw = _BRACKETED_ECHO.sub(" ", raw)  # a whole note, with what follows it
    if _is_angle_echo(raw):
        return ""
    sentence = _clean(raw)
    return "" if _is_angle_echo(sentence) else sentence


def _finished(sentence: str) -> bool:
    """True if the sentence ends like one (not a fragment left by max_tokens)."""
    return sentence.rstrip("\"'”’»)").endswith(tuple(_SENTENCE_END))


def trim_reply(text: str, max_sentences: int, max_chars: int) -> str:
    # sentence by sentence, as the stream does: an unclosed (note stays one piece
    done, rest = _pop_sentences(text)
    text = " ".join(filter(None, map(_clean_sentence, done + [rest])))
    out: list[str] = []
    for sentence in (_SENTENCE_SPLIT.split(text) if text else [])[:max_sentences]:
        # past the first sentence, an unfinished one or one over the limit is dropped
        # rather than spoken cut off
        if out and (not _finished(sentence)
                    or len(" ".join(out + [sentence])) > max_chars):
            break
        out.append(sentence)
    joined = " ".join(out)
    return _cut(joined, max_chars) if len(joined) > max_chars else joined


class LlamaServerLLM:
    def __init__(self, url: str, timeout_s: float,
                 max_sentences: int = 3, max_chars: int = 300, world: str = "",
                 now: Callable[[], datetime] = datetime.now,
                 sampling: "dict | None" = None):
        self._endpoint = url.rstrip("/") + "/v1/chat/completions"
        self._timeout_s = timeout_s
        self._max_sentences = max_sentences
        self._max_chars = max_chars
        self._world = world
        self._now = now
        self._sampling = dict(_SAMPLING if sampling is None else sampling)

    def _body(self, history: list[dict], user_text: str, lang: str,
              max_tokens: int = _MAX_TOKENS, angle: "Angle | None" = None) -> dict:
        # the angle goes at the end of the user message, never into the system
        # prompt: that stays byte-identical, so its cached prefix is reused
        content = user_text if angle is None else f"{user_text} {angle_note(angle, lang)}"
        messages = ([{"role": "system",
                      "content": system_prompt(lang, self._world, self._now())}]
                    + list(history)
                    + [{"role": "user", "content": content}])
        # cache_prompt + a fixed slot per language: llama-server reuses the shared
        # prefix (persona + examples + world) between turns
        return {"messages": messages, "max_tokens": max_tokens, **self._sampling,
                "cache_prompt": True, "id_slot": LANG_SLOTS[lang]}

    def warm(self, lang: str) -> float | None:
        """Load the language's full system prompt into its slot's cache (one token
        generated). Returns llama-server's prompt-eval time in ms, if it reports one.
        Raises LLMError on failure."""
        try:
            resp = requests.post(self._endpoint,
                                 json=self._body([], ".", lang, max_tokens=1),
                                 timeout=self._timeout_s)
            resp.raise_for_status()
            timings = resp.json().get("timings") or {}
        except _REQUEST_ERRORS as e:
            raise LLMError(f"llama-server warmup failed: {e}") from e
        ms = timings.get("prompt_ms")
        log.info("llm warm (%s): prompt eval %s ms for %s tokens", lang,
                 f"{ms:.0f}" if isinstance(ms, (int, float)) else "?",
                 timings.get("prompt_n", "?"))
        return ms

    def reply(self, history: list[dict], user_text: str, lang: str,
              angle: "Angle | None" = None) -> str:
        try:
            resp = requests.post(
                self._endpoint,
                json=self._body(history, user_text, lang, angle=angle),
                timeout=self._timeout_s,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            if content is None:
                raise ValueError("null content")
        except requests.exceptions.RequestException as e:  # incl. Timeout, HTTPError
            raise LLMError(f"llama-server request failed: {e}") from e
        except (ValueError, KeyError, IndexError, TypeError) as e:  # bad JSON shape
            raise LLMError(f"malformed llama-server response: {e}") from e
        text = trim_reply(str(content), self._max_sentences, self._max_chars)
        if not text:
            raise LLMError("llama-server returned an empty reply")
        return text

    def reply_stream(self, history: list[dict], user_text: str, lang: str,
                     cancel: threading.Event | None = None,
                     angle: "Angle | None" = None) -> Iterator[str]:
        """Stream the reply, yielding complete, cleaned sentences as they arrive.

        At most max_sentences are yielded and max_chars in total. Past the first
        sentence, one that would not fit or is unfinished ends the reply; only a first
        sentence is ever cut (with "…"). A failure before the first sentence raises LLMError; after it, the
        stream just ends (logged). Setting `cancel` ends the stream at the next
        streamed token; closing the iterator closes the connection either way.
        An `angle` is appended to the user message as a short hidden instruction.
        """
        yielded, total = 0, 0
        raw = self._raw_sentences(history, user_text, lang, cancel, angle)
        try:
            for sentence in raw:
                sentence = _clean_sentence(sentence)
                if not sentence:
                    continue
                room = self._max_chars - total - (1 if yielded else 0)
                last = yielded + 1 >= self._max_sentences
                # past the first sentence, an unfinished one (max_tokens) or one
                # over the limit is dropped rather than spoken cut off
                if yielded and (not _finished(sentence) or len(sentence) > room):
                    return
                if len(sentence) > room:
                    sentence, last = _cut(sentence, room), True
                    if not sentence:
                        break
                total += len(sentence) + (1 if yielded else 0)
                yielded += 1
                yield sentence
                if last or (cancel is not None and cancel.is_set()):
                    return
        except _REQUEST_ERRORS as e:
            if yielded == 0:
                raise LLMError(f"llama-server stream failed: {e}") from e
            log.warning("llama-server stream broke after %d sentence(s): %s", yielded, e)
            return
        finally:
            raw.close()  # closes the HTTP response, so llama-server stops generating
        if yielded == 0 and not (cancel is not None and cancel.is_set()):
            raise LLMError("llama-server returned an empty reply")

    def _raw_sentences(self, history, user_text, lang,
                       cancel: threading.Event | None, angle=None) -> Iterator[str]:
        """Uncleaned sentences from the SSE stream; the remainder is flushed at the end."""
        body = dict(self._body(history, user_text, lang, angle=angle), stream=True)
        # llama-server's timings arrive in the final chunk, which a reply cut at
        # max_sentences never reads: log the client-side time to first token too
        t0, first = time.monotonic(), True
        with requests.post(self._endpoint, json=body, timeout=self._timeout_s,
                           stream=True) as resp:
            resp.raise_for_status()
            buf = ""
            for line in resp.iter_lines():
                if cancel is not None and cancel.is_set():
                    return
                line = line.decode("utf-8") if isinstance(line, bytes) else line
                if not line.startswith("data:"):
                    continue  # blank separators, ": keep-alive" comments
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if chunk.get("error"):
                    raise ValueError(f"llama-server error: {chunk['error']}")
                timings = chunk.get("timings")
                if timings:
                    log.info("llm prompt eval (%s): %s ms for %s tokens", lang,
                             timings.get("prompt_ms"), timings.get("prompt_n"))
                if not chunk.get("choices"):
                    continue
                delta = chunk["choices"][0].get("delta", {}).get("content")
                if not delta:
                    continue
                if first:
                    first = False
                    log.info("llm first token (%s): %.0f ms", lang,
                             1000.0 * (time.monotonic() - t0))
                done, buf = _pop_sentences(buf + delta)
                yield from done
            if buf.strip():
                yield buf
