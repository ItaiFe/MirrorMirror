#!/usr/bin/env python3
"""How varied are the oracle's answers from the running llama-server?

    python tools/variety_check.py [--lang en|he|both] [--no-angle] [--seed N]
                                  [--url http://127.0.0.1:8080] [--config config/mac.toml]

Per language, sends 10 different burner questions twice, then one question 5 more
times, each as a fresh conversation, through LlamaServerLLM with the real oracle
prompt, world and sampling, plus an answer angle picked the way the Controller picks
it (unlike the recent ones). --no-angle sends the bare questions, for a before/after
comparison. Prints every reply, then distinct-1 and distinct-2 (distinct n-grams /
all n-grams, across replies), the most frequent content words and the share of
replies with a stock word (shimmer, thread, ochre...).
"""
import argparse
import os
import random
import re
import sys
from collections import Counter

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.insert(0, REPO)

from mirror.config import load_config, resolve_repo_path  # noqa: E402
from mirror.llm import LLMError, LlamaServerLLM  # noqa: E402
from mirror.persona import load_world, pick_angle  # noqa: E402

QUESTIONS = {
    "en": [
        "What does my future hold?",
        "Will I find love this year?",
        "What should I leave at the Temple?",
        "Tell me a riddle.",
        "Should I quit my job?",
        "Where is the best party tonight?",
        "What am I afraid of?",
        "Who is the fairest of them all?",
        "What gift should I give my camp?",
        "Will it rain at the burn?",
    ],
    "he": [
        "מה מחכה לי בעתיד?",
        "האם אמצא אהבה השנה?",
        "מה כדאי לי להשאיר במקדש?",
        "ספר לי חידה.",
        "כדאי לי לעזוב את העבודה?",
        "איפה המסיבה הכי טובה הלילה?",
        "ממה אני מפחד?",
        "מי הכי יפה בממלכה?",
        "איזו מתנה לתת למחנה שלי?",
        "ירד גשם בברן?",
    ],
}
REPEATS = 5  # extra asks of the first question

STOCK = {
    "en": ["shimmer", "thread", "whisper", "ochre", "violet", "tapestry", "echo",
           "dust remembers", "star", "mist", "reflection"],
    "he": ["נצנוץ", "מנצנץ", "חוט", "לחישה", "לוחש", "אוקר", "סגול", "מארג", "הד ",
           "האבק זוכר", "כוכב", "ערפל", "השתקפות"],
}
STOPWORDS = {
    "en": set("""a an the and or but of to in on at for with from by as is are was be
        been am i you your yours me my mine we us our it its this that these those
        there here what who how why when where which not no do does did so if than
        then they them their he she his her will would can could shall should may
        might just very too also all any some one into out up down over about like
        have has had let yes oh ah""".split()),
    "he": set("""אני אתה את אתם אתן הוא היא הם הן אנחנו של שלי שלך שלכם שלו שלה זה זו
        זאת אלה כי גם רק אם אבל או על אל עם לא כן יש אין מה מי איך כל כך עוד כבר
        אתכם לכם בכם שלכם אותם אותו אותה הזה הזאת היה היו יהיה עכשיו""".split()),
}
_WORD = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?")


def _words(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


def distinct_n(replies: list[str], n: int) -> float:
    """Distinct n-grams / all n-grams, counted within each reply."""
    grams = [tuple(ws[i:i + n]) for ws in map(_words, replies)
             for i in range(len(ws) - n + 1)]
    return len(set(grams)) / len(grams) if grams else 0.0


def top_content_words(replies: list[str], lang: str, k: int = 10) -> list:
    stop = STOPWORDS.get(lang, STOPWORDS["en"])
    counts = Counter(w for r in replies for w in _words(r) if w not in stop and len(w) > 1)
    return counts.most_common(k)


def _has_stock(reply: str, lang: str) -> bool:
    if lang == "he":
        return any(s in reply + " " for s in STOCK["he"])
    low = reply.lower()
    return any(re.search(r"\b" + re.escape(s), low) for s in STOCK["en"])


def stock_share(replies: list[str], lang: str) -> float:
    return sum(_has_stock(r, lang) for r in replies) / len(replies) if replies else 0.0


def build_llm(cfg, config_path, url=None) -> LlamaServerLLM:
    world = load_world(resolve_repo_path(cfg.conversation.world_file, config_path))
    return LlamaServerLLM(url or cfg.servers.llm_url, max(cfg.servers.timeout_s, 60),
                          max_sentences=cfg.conversation.max_reply_sentences,
                          max_chars=cfg.conversation.max_reply_chars, world=world,
                          sampling=cfg.llm.sampling())


def run(llm, lang: str, use_angles: bool, rng: random.Random, show=print) -> list[str]:
    questions = QUESTIONS[lang] * 2 + [QUESTIONS[lang][0]] * REPEATS
    recent, replies = [], []
    for i, q in enumerate(questions):
        angle = pick_angle(rng, recent) if use_angles else None
        if angle is not None:
            recent.append(angle)
        try:
            reply = llm.reply([], q, lang, angle=angle)
        except LLMError as e:
            show(f"[{lang} {i + 1}] {q}\n    ERROR: {e}")
            continue
        replies.append(reply)
        tag = f" [{angle.mode} / {angle.imagery}]" if angle else ""
        show(f"[{lang} {i + 1}] {q}{tag}\n    {reply}")
    return replies


def report(replies: list[str], lang: str, show=print) -> None:
    n = len(replies)
    stock = sum(_has_stock(r, lang) for r in replies)
    rep = replies[-REPEATS:]
    show(f"\n== {lang}: {n} replies ==")
    show(f"distinct-1 {distinct_n(replies, 1):.3f}   distinct-2 {distinct_n(replies, 2):.3f}"
         f"   (the {len(rep)} repeats: {distinct_n(rep, 1):.3f} / {distinct_n(rep, 2):.3f})")
    show("top words: " + ", ".join(f"{w} {c}" for w, c in top_content_words(replies, lang)))
    show(f"stock words in {stock}/{n} replies ({100.0 * stock / n if n else 0:.0f}%)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lang", choices=["en", "he", "both"], default="both")
    ap.add_argument("--no-angle", action="store_true", help="send the bare questions")
    ap.add_argument("--seed", type=int, default=None, help="angle rng seed")
    ap.add_argument("--url", default=None, help="llama-server (default: the config's)")
    ap.add_argument("--config", default=os.path.join(REPO, "config", "mac.toml"))
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    llm = build_llm(cfg, args.config, args.url)
    rng = random.Random(args.seed)
    langs = ["en", "he"] if args.lang == "both" else [args.lang]
    results = {lang: run(llm, lang, not args.no_angle, rng) for lang in langs}
    for lang, replies in results.items():
        report(replies, lang)
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
