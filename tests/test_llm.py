from unittest.mock import MagicMock, patch

import pytest
import requests

from mirror.llm import LLMError, LlamaServerLLM, trim_reply
from mirror.persona import FALLBACK, SYSTEM_PROMPT


def test_reply_trimmed():
    assert trim_reply("One. Two! Three? Four. Five.", 3, 300) == "One. Two! Three?"
    long = trim_reply("word " * 100, 3, 300)
    assert len(long) <= 300
    assert long.endswith("…")
    assert "word…" in long or long[-2] != " "
    assert trim_reply("*smiles* Hello \U0001F600 there [winks] friend.", 3, 300) == "Hello there friend."


def test_hebrew_sentence_split():
    assert trim_reply("אחת. שתיים. שלוש. ארבע.", 3, 300) == "אחת. שתיים. שלוש."


def _ok(content):
    resp = MagicMock()
    resp.json.return_value = {"choices": [{"message": {"content": content}}]}
    return resp


def test_request_shape():
    history = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    with patch("mirror.llm.requests.post", return_value=_ok("Hi there.")) as post:
        out = LlamaServerLLM("http://x:8080/", 5).reply(history, "שלום", "he")
    assert out == "Hi there."
    args, kwargs = post.call_args
    assert args[0] == "http://x:8080/v1/chat/completions"
    body = kwargs["json"]
    assert body["max_tokens"] == 120 and body["temperature"] == 0.8
    assert body["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT["he"]}
    assert body["messages"][1:3] == history
    assert body["messages"][-1] == {"role": "user", "content": "שלום"}
    assert kwargs["timeout"] == 5


def test_llm_error_on_timeout_and_empty():
    llm = LlamaServerLLM("http://x", 1)
    with patch("mirror.llm.requests.post", side_effect=requests.exceptions.Timeout()):
        with pytest.raises(LLMError):
            llm.reply([], "hi", "en")
    with patch("mirror.llm.requests.post", return_value=_ok("  ")):
        with pytest.raises(LLMError):
            llm.reply([], "hi", "en")


def test_fallback_texts():
    assert FALLBACK["en"] == "The mirror's mist is too thick right now. Ask me again in a moment."
    assert FALLBACK["he"] == "הערפל על המראה סמיך מדי כרגע. שאלו אותי שוב עוד רגע."
