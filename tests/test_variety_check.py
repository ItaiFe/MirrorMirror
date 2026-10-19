import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "tools"))
import variety_check as vc  # noqa: E402


def test_distinct_ratios():
    replies = ["the sun is up", "the sun is down"]
    # unigrams: the sun is up the sun is down -> 5 distinct of 8
    assert vc.distinct_n(replies, 1) == pytest.approx(5 / 8)
    # bigrams do not cross replies: 6 in all, 4 distinct
    assert vc.distinct_n(replies, 2) == pytest.approx(4 / 6)
    assert vc.distinct_n([], 1) == 0.0
    assert vc.distinct_n(["Fire!", "fire."], 1) == pytest.approx(1 / 2)  # case, punctuation


def test_top_content_words_skip_stopwords():
    replies = ["The thread of silver, a thread.", "I see a thread and the moon."]
    assert vc.top_content_words(replies, "en", 2) == [("thread", 3), ("silver", 1)]
    he = ["אני רואה את הירח, והירח מביט בכם.", "הירח שלכם"]
    assert vc.top_content_words(he, "he", 1) == [("הירח", 2)]


def test_stock_share():
    replies = ["A shimmer of light.", "Bicycles know the way.", "The dust remembers you.",
               "Threads everywhere."]
    assert vc.stock_share(replies, "en") == pytest.approx(3 / 4)
    assert vc.stock_share(["אני רואה נצנוץ קטן.", "שאלו את הרוח."], "he") == 0.5
    assert vc.stock_share([], "en") == 0.0


def _ok(content):
    resp = MagicMock()
    resp.json.return_value = {"choices": [{"message": {"content": content}}]}
    return resp


@pytest.mark.parametrize("no_angle", [False, True])
def test_run_sends_questions_with_or_without_angles(capsys, no_angle):
    bodies = []

    def post(url, json, timeout):
        bodies.append(json)
        return _ok(f"Reply number {len(bodies)} shimmers.")

    argv = ["--url", "http://x", "--lang", "en", "--seed", "1"]
    with patch("mirror.llm.requests.post", side_effect=post):
        assert vc.main(argv + (["--no-angle"] if no_angle else [])) == 0
    assert len(bodies) == 25  # 10 questions x 2, then 5 repeats of one
    users = [b["messages"][-1]["content"] for b in bodies]
    assert len(set(users[-5:])) == (1 if no_angle else 5)
    assert all(("(Answer as" in u) != no_angle for u in users)
    assert len({b["messages"][0]["content"] for b in bodies}) == 1
    out = capsys.readouterr().out
    assert "distinct-1" in out and "distinct-2" in out and "stock words in 25/25" in out
