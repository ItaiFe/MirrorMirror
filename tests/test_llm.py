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
    assert body["max_tokens"] == 120 and body["temperature"] == 0.95
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
    assert trim_reply("(sighs) The stars (and the moon) know.", 3, 300) == \
        "The stars and the moon know."


def test_clean_strips_copied_answer_label():
    assert trim_reply("תשובה: אני רואה אתכם.", 3, 300) == "אני רואה אתכם."
    assert trim_reply("Answer: The stars know.", 3, 300) == "The stars know."
    with patch("mirror.llm.requests.post", return_value=_sse("תשובה:", " שאלו שוב.")):
        assert list(LlamaServerLLM("http://x", 5).reply_stream([], "hi", "he")) == ["שאלו שוב."]


@pytest.mark.parametrize("label", [
    "האורקל:", "אורקל :", "המראה -", "תשובה -", "Oracle:", "ORACLE -", "oracle :",
    "Mirror:", "mirror -", "answer-",
])
def test_clean_strips_speaker_labels(label):
    """A reply opening with a speaker label ("האורקל:", "Oracle -") is not spoken."""
    assert trim_reply(f"{label} שאלו את הרוח.", 3, 300) == "שאלו את הרוח."
    assert trim_reply(f"{label}The stars know.", 3, 300) == "The stars know."
    with patch("mirror.llm.requests.post",
               return_value=_sse(label, " שאלו את הרוח. ", "Then go.")):
        out = list(LlamaServerLLM("http://x", 5).reply_stream([], "hi", "he"))
    assert out == ["שאלו את הרוח.", "Then go."]


def test_clean_keeps_label_words_inside_a_sentence():
    assert trim_reply("Mirror, mirror on the wall: look.", 3, 300) == \
        "Mirror, mirror on the wall: look."
    assert trim_reply("המראה רואה הכל.", 3, 300) == "המראה רואה הכל."


# --- Hebrew few-shot examples ------------------------------------------------------

def test_hebrew_prompt_has_plural_examples_before_world():
    from mirror.persona import HEBREW_EXAMPLES

    assert len(HEBREW_EXAMPLES) == 5
    he = system_prompt("he", "WORLD", NOW)
    en = system_prompt("en", "WORLD", NOW)
    for question, answer in HEBREW_EXAMPLES:
        assert question in he and answer in he
        assert answer not in en
        assert any(w in answer for w in ("אתם", "אתכם", "שאלו", "לכם"))
        assert he.index(answer) < he.index("WORLD")
    assert len({a.split()[0] for _, a in HEBREW_EXAMPLES}) == 5  # varied openings
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


# --- Hebrew address: singular questions, plural answers -----------------------------

def test_singular_address_markers():
    from mirror.persona import singular_address

    assert singular_address("אני רואה אתכם, ומה שלכם יאיר לכם את הדרך.") == []
    assert singular_address("ברגע שאומרים את שמו, הוא נעלם.") == []  # accusative את
    assert singular_address("הכוכבים שלך מחכים לך.") == ["שלך", "לך"]
    assert singular_address("אתה מוכן? זכרי את הלילה.") == ["אתה", "זכרי"]
    assert singular_address("את לא לבד, והאור בעינייך.") == ["את", "בעינייך"]
    assert singular_address("צריך ללכת בדרך הארוכה, איך שהוא.") == []  # not address
    assert singular_address("שאלה: מה שאתה רואה?") == ["שאתה"]
    assert singular_address("השאר את החששות שלכם במקדש.") == ["השאר"]


def test_hebrew_examples_include_singular_questions_with_plural_answers():
    from mirror.persona import HEBREW_EXAMPLES, singular_address

    questions = [q for q, _ in HEBREW_EXAMPLES]
    assert any(q.startswith("תגיד") for q in questions)  # masculine singular
    assert any("מפחדת" in q for q in questions)  # feminine singular
    he = system_prompt("he", "WORLD", NOW)
    for question, answer in HEBREW_EXAMPLES:
        assert answer in he
        assert singular_address(answer) == [], (question, answer)
    assert any("אני יודע" in a for _, a in HEBREW_EXAMPLES)  # masculine first person


# --- llama-server flags --------------------------------------------------------------

def _repo_file(*parts):
    import os
    with open(os.path.join(os.path.dirname(__file__), os.pardir, *parts),
              encoding="utf-8") as f:
        return f.read()


def test_llama_server_keeps_full_swa_cache_in_dev_and_deploy():
    """Gemma 3's sliding-window cache otherwise disables cache_reuse, and the
    English slot re-processes ~500 prompt tokens per turn."""
    make = next(l for l in _repo_file("Makefile").splitlines()
                if l.startswith("LLAMA :="))
    unit = next(l for l in _repo_file("deploy", "systemd", "mirror-llm.service")
                .splitlines() if l.startswith("ExecStart="))
    for cmd in (make, unit):
        assert "--swa-full" in cmd.split()
        assert re.search(r"-c 8192\b", cmd) and re.search(r"-np 2\b", cmd)
        assert "--cache-reuse 256" in cmd
        # 8-bit KV cache (V quantization needs flash attention): ~-0.55 GB on the Orin
        assert re.search(r"-fa on\b", cmd)
        assert re.search(r"-ctk q8_0\b", cmd) and re.search(r"-ctv q8_0\b", cmd)
        # no host-RAM prompt cache: the per-language slots already keep the prompt
        assert re.search(r"-cram 0\b", cmd)


def test_llm_model_file_is_one_setting():
    """The model file is set in one place per launcher (Gemma 4 E2B is the bake-off
    runner-up), not buried in the command line."""
    assert re.search(r"^LLM_MODEL \?= \$\(MODELS\)/gemma-3-4b-it-Q4_K_M\.gguf$",
                     _repo_file("Makefile"), re.MULTILINE)
    unit = _repo_file("deploy", "systemd", "mirror-llm.service")
    assert "Environment=LLM_MODEL=gemma-3-4b-it-Q4_K_M.gguf" in unit
    exec_line = next(l for l in unit.splitlines() if l.startswith("ExecStart="))
    assert "-m %h/mirror-models/${LLM_MODEL} " in exec_line


def test_hebrew_address_check_tool_reports_rate(capsys):
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "tools"))
    import hebrew_address_check

    replies = iter(["אני רואה אתכם.", "הכוכבים שלך זורחים.", "שאלו את הרוח."])
    with patch("mirror.llm.requests.post",
               side_effect=lambda *a, **kw: _ok(next(replies))):
        assert hebrew_address_check.main(["--n", "3", "--url", "http://x"]) == 0
    out = capsys.readouterr().out
    assert "SINGULAR שלך" in out
    assert "singular address in 1/3 replies (33%)" in out


# --- varied answers: angles, sampling, persona ---------------------------------------

def test_pick_angle_is_deterministic_with_a_seeded_rng():
    import random

    from mirror.persona import IMAGERY, MODES, pick_angle

    assert len(MODES) == 8 and len(IMAGERY) >= 25
    a = [pick_angle(random.Random(3), []) for _ in range(2)]
    assert a[0] == a[1]
    assert a[0].mode in MODES and a[0].imagery in IMAGERY


def test_pick_angle_respects_the_no_repeat_windows():
    import random

    from mirror.persona import IMAGERY, MODES, pick_angle

    rng, recent = random.Random(11), []
    for _ in range(200):
        angle = pick_angle(rng, recent)
        assert angle.mode not in [r.mode for r in recent[-3:]]
        assert angle.imagery not in [r.imagery for r in recent[-5:]]
        recent.append(angle)
    assert {r.mode for r in recent} == set(MODES)  # every mode gets its turn
    assert len({r.imagery for r in recent}) == len(IMAGERY)


def test_angle_note_in_both_languages():
    from mirror.persona import Angle, angle_note

    en = angle_note(Angle("playful dare", "bicycles"), "en")
    assert en == ("(Answer as a playful dare, using images of bicycles. "
                  "Start by answering their actual question.)")
    he = angle_note(Angle("playful dare", "bicycles"), "he")
    assert he.startswith("(") and he.endswith(")") and "אופניים" in he
    assert not re.search(r"[A-Za-z]", he)
    assert angle_note(Angle("riddle", "fire"), "fr") == angle_note(Angle("riddle", "fire"), "en")


def test_every_mode_and_image_world_has_hebrew():
    from mirror.persona import IMAGERY, MODES, Angle, angle_note

    for mode in MODES:
        for imagery in IMAGERY:
            assert not re.search(r"[A-Za-z]", angle_note(Angle(mode, imagery), "he"))


def test_angle_goes_at_the_end_of_the_user_message_only():
    from mirror.persona import Angle, angle_note

    llm = LlamaServerLLM("http://x", 5, world="W", now=lambda: NOW)
    bodies = []
    for angle in (Angle("riddle", "fire"), Angle("tiny story", "kites")):
        with patch("mirror.llm.requests.post", return_value=_sse("Hi.")) as post:
            list(llm.reply_stream([{"role": "user", "content": "a"}], "hi", "he",
                                  angle=angle))
        bodies.append(post.call_args.kwargs["json"])
        assert bodies[-1]["messages"][-1] == {
            "role": "user", "content": "hi " + angle_note(angle, "he")}
        assert bodies[-1]["messages"][1] == {"role": "user", "content": "a"}
    # the system prompt is byte-identical, so the prompt cache stays warm
    assert bodies[0]["messages"][0] == bodies[1]["messages"][0]
    with patch("mirror.llm.requests.post", return_value=_ok("Hi.")) as post:
        llm.reply([], "hi", "en", angle=Angle("riddle", "fire"))
    assert post.call_args.kwargs["json"]["messages"][-1]["content"].startswith("hi (Answer as")
    with patch("mirror.llm.requests.post", return_value=_ok("Hi.")) as post:
        llm.reply([], "hi", "en")  # no angle: the question as asked
    assert post.call_args.kwargs["json"]["messages"][-1]["content"] == "hi"


SAMPLING = {"temperature": 0.95, "top_p": 0.95, "min_p": 0.05, "repeat_penalty": 1.12,
            "presence_penalty": 0.4, "frequency_penalty": 0.3}


def test_sampling_fields_in_every_request():
    llm = LlamaServerLLM("http://x", 5)
    with patch("mirror.llm.requests.post", return_value=_sse("Hi.")) as post:
        list(llm.reply_stream([], "hi", "en"))
    stream_body = post.call_args.kwargs["json"]
    with patch("mirror.llm.requests.post", return_value=_ok("Hi.")) as post:
        llm.reply([], "hi", "en")
    for body in (stream_body, post.call_args.kwargs["json"]):
        for k, v in SAMPLING.items():
            assert body[k] == v, k
    custom = LlamaServerLLM("http://x", 5, sampling=dict(SAMPLING, temperature=0.5))
    with patch("mirror.llm.requests.post", return_value=_ok("Hi.")) as post:
        custom.reply([], "hi", "en")
    assert post.call_args.kwargs["json"]["temperature"] == 0.5


@pytest.mark.parametrize("name", ["mac.toml", "orin.toml"])
def test_sampling_is_set_in_both_configs(name):
    import os

    from mirror.config import load_config

    cfg = load_config(os.path.join(os.path.dirname(__file__), os.pardir, "config", name))
    assert cfg.llm.sampling() == SAMPLING
    assert "[llm]" in _repo_file("config", name)


def test_persona_drops_the_fixed_imagery_and_bans_stock_words():
    en, he = SYSTEM_PROMPT["en"], SYSTEM_PROMPT["he"]
    assert "stars, mists, threads" not in en
    assert "כוכבים, ערפל, חוטים" not in he
    assert "first sentence" in en and "previous replies" in en
    for word in ("shimmer", "thread", "whisper", "ochre", "violet", "tapestry", "echo",
                 "dust remembers"):
        assert word in en
    assert "במשפט הראשון" in he and "אוקר" in he and "נצנוץ" in he
    assert "never cruel" in en and "one to three short" in en


def test_hebrew_examples_avoid_the_overused_images():
    from mirror.persona import HEBREW_EXAMPLES

    for _, answer in HEBREW_EXAMPLES:
        for stock in ("חוט", "כוכב", "ערפל", "לוחש", "נצנוץ", "השתקפות"):
            assert stock not in answer, (stock, answer)


# --- parentheses: stage directions go, mid-sentence words stay -------------------------

@pytest.mark.parametrize("raw, clean", [
    ("The desert doesn't (hold) a pattern. It simply (is).",
     "The desert doesn't hold a pattern. It simply is."),
    ("it doesn't (hold) a pattern.", "it doesn't hold a pattern."),
    ("(softly) Look up.", "Look up."),
    ("Look up. (pause)", "Look up."),
    ("Look up (laughs).", "Look up."),
    ("(leans in) The Temple knows.", "The Temple knows."),
    ("Look up. (The mirror goes quiet.) Then go.", "Look up. Then go."),
    ("The stars (and the moon) know.", "The stars and the moon know."),
    ("המדבר לא (מחזיק) דפוס.", "המדבר לא מחזיק דפוס."),
    ("(בשקט) הביטו למעלה.", "הביטו למעלה."),
    ("הביטו למעלה. (הפסקה)", "הביטו למעלה."),
    ("הוא פשוט (קיים).", "הוא פשוט קיים."),
])
def test_parentheses_keep_words_unless_a_stage_direction(raw, clean):
    assert trim_reply(raw, 3, 300) == clean
    with patch("mirror.llm.requests.post", return_value=_sse(*re.findall(r"\S+\s*", raw))):
        out = list(LlamaServerLLM("http://x", 5).reply_stream([], "hi", "en"))
    assert " ".join(out) == clean


def test_tools_and_voice_send_the_configured_sampling():
    import os
    import sys

    from mirror.config import Config, LlmConfig
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "tools"))
    import hebrew_address_check

    cfg = Config(llm=LlmConfig(temperature=0.3))
    llm = hebrew_address_check.build_llm(cfg, None, "http://x")
    with patch("mirror.llm.requests.post", return_value=_ok("Hi.")) as post:
        llm.reply([], "hi", "he")
    assert post.call_args.kwargs["json"]["temperature"] == 0.3
    for src in (_repo_file("mirror", "voice_main.py"), _repo_file("tools", "record_demo.py")):
        assert "sampling=cfg.llm.sampling()" in src
