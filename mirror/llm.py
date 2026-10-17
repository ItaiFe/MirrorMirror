"""LLM client for llama.cpp's llama-server (OpenAI-compatible API)."""
import re
import unicodedata

import requests

from mirror.persona import SYSTEM_PROMPT

_STAGE_DIRECTIONS = re.compile(r"\*[^*]*\*|\[[^\]]*\]")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+")
_JOINERS = {"‍", "︎", "️"}  # ZWJ and variation selectors


class LLMError(Exception):
    """Raised when the llama-server request fails or returns an empty reply."""


def _strip_emoji(text: str) -> str:
    return "".join(
        ch for ch in text
        if ch not in _JOINERS and unicodedata.category(ch) not in ("So", "Sk")
    )


def trim_reply(text: str, max_sentences: int, max_chars: int) -> str:
    text = _STAGE_DIRECTIONS.sub(" ", text)
    text = _strip_emoji(text)
    text = " ".join(text.split())
    sentences = _SENTENCE_SPLIT.split(text) if text else []
    out = " ".join(sentences[:max_sentences])
    if len(out) > max_chars:
        cut = out[: max_chars - 1]
        space = cut.rfind(" ")
        if space > 0:
            cut = cut[:space]
        out = cut.rstrip(" ,;:.!?") + "…"
    return out


class LlamaServerLLM:
    def __init__(self, url: str, timeout_s: float,
                 max_sentences: int = 3, max_chars: int = 300):
        self._endpoint = url.rstrip("/") + "/v1/chat/completions"
        self._timeout_s = timeout_s
        self._max_sentences = max_sentences
        self._max_chars = max_chars

    def reply(self, history: list[dict], user_text: str, lang: str) -> str:
        messages = ([{"role": "system", "content": SYSTEM_PROMPT[lang]}]
                    + list(history)
                    + [{"role": "user", "content": user_text}])
        try:
            resp = requests.post(
                self._endpoint,
                json={"messages": messages, "max_tokens": 120, "temperature": 0.8},
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
