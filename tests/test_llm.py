import json
import logging
import re
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
import requests

from mirror.llm import LLMError, LlamaServerLLM, trim_reply
from mirror.persona import FALLBACK, SYSTEM_PROMPT, load_world, system_prompt

NOW = datetime(2026, 11, 5, 2, 30)  # a Thursday, 02:30


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
        out = LlamaServerLLM("http://x:8080/", 5, world="W", now=lambda: NOW).reply(
            history, "שלום", "he")
    assert out == "Hi there."
    args, kwargs = post.call_args
    assert args[0] == "http://x:8080/v1/chat/completions"
    body = kwargs["json"]
    assert body["max_tokens"] == 120 and body["temperature"] == 0.8
    assert body["messages"][0] == {"role": "system",
                                   "content": system_prompt("he", "W", NOW)}
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
    assert FALLBACK["en"] == "The mists are too thick right now. Ask me again in a moment."
    assert FALLBACK["he"] == "הערפל סמיך מדי כרגע. שאלו אותי שוב עוד רגע."


# --- oracle world background -------------------------------------------------

def test_world_strips_header_and_comments(tmp_path):
    f = tmp_path / "world.md"
    f.write_text("# Title\n<!-- editor note\n  more -->\nignored\n---\n"
                 "You are at Midburn. <!-- hidden -->\n\nDrink water.\n", encoding="utf-8")
    assert load_world(str(f)) == "You are at Midburn.\n\nDrink water."


def test_missing_world_file_empty(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="mirror.persona"):
        assert load_world(str(tmp_path / "nope.md")) == ""
    assert "nope.md" in caplog.text


def test_system_prompt_contains_world_and_time():
    en = system_prompt("en", "WORLD TEXT", NOW)
    assert en.startswith(SYSTEM_PROMPT["en"] + "\n\nWORLD TEXT\n\n")
    assert en.endswith("It is now Thursday night at the burn.")
    he = system_prompt("he", "WORLD TEXT", NOW)
    assert he.startswith(SYSTEM_PROMPT["he"] + "\n\n")
    assert "\n\nWORLD TEXT\n\n" in he
    assert he.endswith("עכשיו חמישי לילה בברן.")
    assert "ענה תמיד בעברית" in SYSTEM_PROMPT["he"]


def test_time_of_day_boundaries():
    def part(h):
        return system_prompt("en", "", datetime(2026, 11, 5, h, 0)).split()[-4]
    assert [part(h) for h in (4, 5, 11, 12, 17, 18, 21, 22)] == [
        "night", "morning", "morning", "afternoon", "afternoon", "evening", "evening", "night"]


# --- streaming ------------------------------------------------------------------

def _sse(*chunks, error=None):
    """A fake streamed response: one SSE data line per content chunk, then [DONE]
    (or `error` raised after the chunks)."""
    def lines():
        yield b": keep-alive"
        for c in chunks:
            yield b"data: " + json.dumps(
                {"choices": [{"delta": {"content": c}}]}).encode()
            yield b""
        if error is not None:
            raise error
        yield b"data: " + json.dumps({"choices": [{"delta": {}}]}).encode()
        yield b"data: [DONE]"

    resp = MagicMock()
    resp.__enter__.return_value = resp
    resp.iter_lines.return_value = lines()
    return resp


def test_reply_stream_yields_sentences():
    with patch("mirror.llm.requests.post",
               return_value=_sse("Mirror", ", mirror. Who", " is fair? You", " are.")):
        out = list(LlamaServerLLM("http://x", 5).reply_stream([], "hi", "en"))
    assert out == ["Mirror, mirror.", "Who is fair?", "You are."]


def test_reply_stream_cleans_each_sentence():
    with patch("mirror.llm.requests.post",
               return_value=_sse("*smiles* Hello \U0001F600 there. ", "[winks] Bye…")):
        out = list(LlamaServerLLM("http://x", 5).reply_stream([], "hi", "en"))
    assert out == ["Hello there.", "Bye…"]


def test_reply_stream_limits():
    with patch("mirror.llm.requests.post",
               return_value=_sse("One. Two! Three? Four. Five.")):
        out = list(LlamaServerLLM("http://x", 5).reply_stream([], "hi", "en"))
    assert out == ["One.", "Two!", "Three?"]
    with patch("mirror.llm.requests.post",
               return_value=_sse("Short one. " + "word " * 40 + "end. Never.")):
        out = list(LlamaServerLLM("http://x", 5, max_chars=60).reply_stream([], "hi", "en"))
    assert out[0] == "Short one."
    assert len(out) == 2 and out[1].endswith("…")
    assert len(" ".join(out)) <= 60


def test_request_sends_cache_prompt():
    with patch("mirror.llm.requests.post", return_value=_sse("Hi.")) as post:
        list(LlamaServerLLM("http://x:8080", 5, world="W", now=lambda: NOW)
             .reply_stream([{"role": "user", "content": "a"}], "hi", "en"))
    args, kwargs = post.call_args
    assert args[0] == "http://x:8080/v1/chat/completions"
    body = kwargs["json"]
    assert body["stream"] is True and body["cache_prompt"] is True
    assert body["messages"][0] == {"role": "system", "content": system_prompt("en", "W", NOW)}
    assert body["messages"][-1] == {"role": "user", "content": "hi"}
    assert kwargs["stream"] is True and kwargs["timeout"] == 5
    with patch("mirror.llm.requests.post", return_value=_ok("Hi.")) as post:
        LlamaServerLLM("http://x", 5).reply([], "hi", "en")
    assert post.call_args.kwargs["json"]["cache_prompt"] is True


def test_reply_stream_error_before_first_raises():
    llm = LlamaServerLLM("http://x", 1)
    with patch("mirror.llm.requests.post", side_effect=requests.exceptions.Timeout()):
        with pytest.raises(LLMError):
            list(llm.reply_stream([], "hi", "en"))
    bad = _sse()
    bad.raise_for_status.side_effect = requests.exceptions.HTTPError("500")
    with patch("mirror.llm.requests.post", return_value=bad):
        with pytest.raises(LLMError):
            list(llm.reply_stream([], "hi", "en"))
    with patch("mirror.llm.requests.post", return_value=_sse(
            "No full stop yet", error=requests.exceptions.ConnectionError("reset"))):
        with pytest.raises(LLMError):
            list(llm.reply_stream([], "hi", "en"))
    with patch("mirror.llm.requests.post", return_value=_sse("  ")):
        with pytest.raises(LLMError):  # nothing to say
            list(llm.reply_stream([], "hi", "en"))


def test_reply_stream_error_after_first_ends_quietly(caplog):
    with patch("mirror.llm.requests.post", return_value=_sse(
            "First. Second", error=requests.exceptions.ConnectionError("reset"))):
        with caplog.at_level(logging.WARNING, logger="mirror.llm"):
            out = list(LlamaServerLLM("http://x", 5).reply_stream([], "hi", "en"))
    assert out == ["First."]
    assert "reset" in caplog.text


def test_reply_stream_stops_when_cancelled():
    import threading

    cancel = threading.Event()
    resp = _sse("One. Two. Three.")
    with patch("mirror.llm.requests.post", return_value=resp):
        stream = LlamaServerLLM("http://x", 5).reply_stream([], "hi", "en", cancel=cancel)
        assert next(stream) == "One."
        cancel.set()
        assert list(stream) == []
    resp.__exit__.assert_called()  # the connection was closed


# --- prompt cache: slot pinning and warmup ---------------------------------------

def test_requests_pin_a_slot_per_language():
    from mirror.llm import LANG_SLOTS

    assert LANG_SLOTS == {"en": 0, "he": 1}
    for lang, slot in LANG_SLOTS.items():
        with patch("mirror.llm.requests.post", return_value=_sse("Hi.")) as post:
            list(LlamaServerLLM("http://x", 5).reply_stream([], "hi", lang))
        body = post.call_args.kwargs["json"]
        assert body["id_slot"] == slot and body["cache_prompt"] is True
        with patch("mirror.llm.requests.post", return_value=_ok("Hi.")) as post:
            LlamaServerLLM("http://x", 5).reply([], "hi", lang)
        assert post.call_args.kwargs["json"]["id_slot"] == slot


def test_warm_sends_full_prompt_and_one_token(caplog):
    resp = _ok("x")
    resp.json.return_value["timings"] = {"prompt_ms": 812.5, "prompt_n": 900}
    with patch("mirror.llm.requests.post", return_value=resp) as post:
        with caplog.at_level(logging.INFO, logger="mirror.llm"):
            ms = LlamaServerLLM("http://x", 5, world="W", now=lambda: NOW).warm("he")
    assert ms == 812.5
    body = post.call_args.kwargs["json"]
    assert body["max_tokens"] == 1 and body["id_slot"] == 1 and body["cache_prompt"] is True
    assert body["messages"] == [
        {"role": "system", "content": system_prompt("he", "W", NOW)},
        {"role": "user", "content": "."}]
    assert "stream" not in body
    assert "812" in caplog.text


def test_warm_failure_raises_llm_error():
    with patch("mirror.llm.requests.post", side_effect=requests.exceptions.ConnectionError()):
        with pytest.raises(LLMError):
            LlamaServerLLM("http://x", 5).warm("en")


def test_stream_logs_prompt_timings(caplog):
    resp = _sse("Hi.")
    lines = list(resp.iter_lines.return_value)
    lines.insert(-1, b"data: " + json.dumps(
        {"choices": [{"delta": {}, "finish_reason": "stop"}],
         "timings": {"prompt_ms": 123.4, "prompt_n": 7}}).encode())
    resp.iter_lines.return_value = iter(lines)
    with patch("mirror.llm.requests.post", return_value=resp):
        with caplog.at_level(logging.INFO, logger="mirror.llm"):
            assert list(LlamaServerLLM("http://x", 5).reply_stream([], "hi", "en")) == ["Hi."]
    assert "123" in caplog.text


# --- stage directions across sentences -------------------------------------------

def test_stream_keeps_stage_directions_whole():
    text = "*leans in. whispers* The stars know. [a long pause. then] Yes."
    with patch("mirror.llm.requests.post", return_value=_sse(*re.findall(r"\S+\s*", text))):
        out = list(LlamaServerLLM("http://x", 5).reply_stream([], "hi", "en"))
    assert out == ["The stars know.", "Yes."]
    with patch("mirror.llm.requests.post",
               return_value=_sse("(softly. very softly) Look up. ", "Done.")):
        out = list(LlamaServerLLM("http://x", 5).reply_stream([], "hi", "en"))
    assert out == ["Look up.", "Done."]


def test_clean_strips_lone_brackets_and_stars():
    assert trim_reply("*The stars] know (yes.", 3, 300) == "The stars know yes."
    assert trim_reply("(sighs) The stars (and the moon) know.", 3, 300) == "The stars know."


def test_clean_strips_copied_answer_label():
    assert trim_reply("תשובה: אני רואה אתכם.", 3, 300) == "אני רואה אתכם."
    assert trim_reply("Answer: The stars know.", 3, 300) == "The stars know."
    with patch("mirror.llm.requests.post", return_value=_sse("תשובה:", " שאלו שוב.")):
        assert list(LlamaServerLLM("http://x", 5).reply_stream([], "hi", "he")) == ["שאלו שוב."]


# --- Hebrew few-shot examples ------------------------------------------------------

def test_hebrew_prompt_has_plural_examples_before_world():
    from mirror.persona import HEBREW_EXAMPLES

    assert len(HEBREW_EXAMPLES) == 3
    he = system_prompt("he", "WORLD", NOW)
    en = system_prompt("en", "WORLD", NOW)
    for question, answer in HEBREW_EXAMPLES:
        assert question in he and answer in he
        assert answer not in en
        assert any(w in answer for w in ("אתם", "שאלו", "לכם"))
        assert he.index(answer) < he.index("WORLD")
    assert len({a.split()[0] for _, a in HEBREW_EXAMPLES}) == 3  # varied openings
    assert "רק את הטון" in he  # the examples are not to be copied


def test_null_content_is_an_error_not_none():
    with patch("mirror.llm.requests.post", return_value=_ok(None)):
        with pytest.raises(LLMError):
            LlamaServerLLM("http://x", 1).reply([], "hi", "en")


def test_system_prompt_unknown_language_falls_back_to_english():
    assert system_prompt("fr", "W", NOW) == system_prompt("en", "W", NOW)


def test_stream_logs_first_token_time_even_when_cut(caplog):
    """llama-server's timings come in the last chunk, which a reply cut at
    max_sentences never reads; the client-side time to first token is logged."""
    with patch("mirror.llm.requests.post",
               return_value=_sse("One. Two. Three. Four.", " Five.")):
        with caplog.at_level(logging.INFO, logger="mirror.llm"):
            out = list(LlamaServerLLM("http://x", 5, max_sentences=2)
                       .reply_stream([], "hi", "en"))
    assert out == ["One.", "Two."]
    assert "first token" in caplog.text
