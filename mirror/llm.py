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

from mirror.persona import system_prompt

log = logging.getLogger("mirror.llm")

_STAGE_DIRECTIONS = re.compile(r"\*[^*]*\*|\[[^\]]*\]|\([^)]*\)")
# A reply that copies the few-shot format ("תשובה: ...", "Answer: ...")
_ANSWER_LABEL = re.compile(r"^\s*(?:תשובה|answer)\s*:\s*", re.IGNORECASE)
_STRAY_MARKS = re.compile(r"[*\[\]()]")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+")
_JOINERS = {"‍", "︎", "️"}  # ZWJ and variation selectors
# llama-server runs with -np 2: each language keeps its own slot, so its long
# system prompt stays cached there between requests.
LANG_SLOTS = {"en": 0, "he": 1}
_MAX_TOKENS = 120
_TEMPERATURE = 0.8
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
    *[]() marks and emoji; collapse whitespace."""
    text = _ANSWER_LABEL.sub("", text)
    text = _STAGE_DIRECTIONS.sub(" ", text)
    text = _STRAY_MARKS.sub("", text)
    text = _strip_emoji(text)
    return " ".join(text.split())


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


def trim_reply(text: str, max_sentences: int, max_chars: int) -> str:
    text = _clean(text)
    sentences = _SENTENCE_SPLIT.split(text) if text else []
    out = " ".join(sentences[:max_sentences])
    if len(out) > max_chars:
        out = _cut(out, max_chars)
    return out


class LlamaServerLLM:
    def __init__(self, url: str, timeout_s: float,
                 max_sentences: int = 3, max_chars: int = 300, world: str = "",
                 now: Callable[[], datetime] = datetime.now):
        self._endpoint = url.rstrip("/") + "/v1/chat/completions"
        self._timeout_s = timeout_s
        self._max_sentences = max_sentences
        self._max_chars = max_chars
        self._world = world
        self._now = now

    def _body(self, history: list[dict], user_text: str, lang: str,
              max_tokens: int = _MAX_TOKENS) -> dict:
        messages = ([{"role": "system",
                      "content": system_prompt(lang, self._world, self._now())}]
                    + list(history)
                    + [{"role": "user", "content": user_text}])
        # cache_prompt + a fixed slot per language: llama-server reuses the shared
        # prefix (persona + examples + world) between turns
        return {"messages": messages, "max_tokens": max_tokens,
                "temperature": _TEMPERATURE, "cache_prompt": True,
                "id_slot": LANG_SLOTS[lang]}

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

    def reply(self, history: list[dict], user_text: str, lang: str) -> str:
        try:
            resp = requests.post(
                self._endpoint,
                json=self._body(history, user_text, lang),
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
                     cancel: threading.Event | None = None) -> Iterator[str]:
        """Stream the reply, yielding complete, cleaned sentences as they arrive.

        At most max_sentences are yielded and max_chars in total (the last one cut
        with "…"). A failure before the first sentence raises LLMError; after it, the
        stream just ends (logged). Setting `cancel` ends the stream at the next
        streamed token; closing the iterator closes the connection either way.
        """
        yielded, total = 0, 0
        raw = self._raw_sentences(history, user_text, lang, cancel)
        try:
            for sentence in raw:
                sentence = _clean(sentence)
                if not sentence:
                    continue
                room = self._max_chars - total - (1 if yielded else 0)
                last = yielded + 1 >= self._max_sentences
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
                       cancel: threading.Event | None) -> Iterator[str]:
        """Uncleaned sentences from the SSE stream; the remainder is flushed at the end."""
        body = dict(self._body(history, user_text, lang), stream=True)
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
