"""LLM client for llama.cpp's llama-server (OpenAI-compatible API)."""
from __future__ import annotations

import json
import logging
import re
import threading
import unicodedata
from datetime import datetime
from typing import Callable, Iterator

import requests

from mirror.persona import system_prompt

log = logging.getLogger("mirror.llm")

_STAGE_DIRECTIONS = re.compile(r"\*[^*]*\*|\[[^\]]*\]")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+")
_JOINERS = {"‍", "︎", "️"}  # ZWJ and variation selectors
_MAX_TOKENS = 120
_TEMPERATURE = 0.8
# Errors that mean the request or its streamed response failed (incl. bad JSON shape).
_REQUEST_ERRORS = (requests.exceptions.RequestException, ValueError, KeyError,
                   IndexError, TypeError, AttributeError)


class LLMError(Exception):
    """Raised when the llama-server request fails or returns an empty reply."""


def _strip_emoji(text: str) -> str:
    return "".join(
        ch for ch in text
        if ch not in _JOINERS and unicodedata.category(ch) not in ("So", "Sk")
    )


def _clean(text: str) -> str:
    """Drop stage directions (*smiles*, [winks]) and emoji; collapse whitespace."""
    text = _STAGE_DIRECTIONS.sub(" ", text)
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

    def _body(self, history: list[dict], user_text: str, lang: str) -> dict:
        messages = ([{"role": "system",
                      "content": system_prompt(lang, self._world, self._now())}]
                    + list(history)
                    + [{"role": "user", "content": user_text}])
        # cache_prompt: llama-server reuses the shared prefix (persona + world) between turns
        return {"messages": messages, "max_tokens": _MAX_TOKENS,
                "temperature": _TEMPERATURE, "cache_prompt": True}

    def reply(self, history: list[dict], user_text: str, lang: str) -> str:
        try:
            resp = requests.post(
                self._endpoint,
                json=self._body(history, user_text, lang),
                timeout=self._timeout_s,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
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
                if not chunk.get("choices"):
                    continue
                delta = chunk["choices"][0].get("delta", {}).get("content")
                if not delta:
                    continue
                *done, buf = _SENTENCE_SPLIT.split(buf + delta)
                yield from done
            if buf.strip():
                yield buf
