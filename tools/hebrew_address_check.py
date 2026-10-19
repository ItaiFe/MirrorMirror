#!/usr/bin/env python3
"""How often does the running llama-server address a Hebrew visitor in the singular?

    python tools/hebrew_address_check.py [--n 20] [--url http://127.0.0.1:8080]
                                         [--config config/orin.toml] [--no-angle] [--seed N]

Sends N Hebrew questions (cycling through a fixed list: neutral, masculine and
feminine singular, plural) through LlamaServerLLM with the real oracle prompt and
world, one fresh conversation each, each with an answer angle as in the mirror
(--no-angle: bare questions), and prints every reply with the singular-address
markers mirror.persona.singular_address finds, then the rate. The check is a
heuristic: read the replies too (it misses past-tense forms such as "ששכחת").
"""
import argparse
import os
import random
import sys
import time

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.insert(0, REPO)

from mirror.config import load_config, resolve_repo_path  # noqa: E402
from mirror.llm import LLMError, LlamaServerLLM  # noqa: E402
from mirror.persona import load_world, pick_angle, singular_address  # noqa: E402

QUESTIONS = [
    "מה מחכה לי בעתיד?",
    "תגיד לי, מה איבדתי השנה ואיך אמצא את זה?",
    "אני שואלת אותך: האם הוא אוהב אותי?",
    "מה כדאי לי להשאיר במקדש לפני שהוא נשרף?",
    "אני מרגישה קצת לבד פה, מה לעשות?",
    "תן לי עצה לפני השקיעה.",
    "אנחנו שלושה חברים שהגענו יחד, מה מחכה לנו?",
    "חם לי נורא והראש שלי מסתובב, מה לעשות?",
    "מה אני צריכה לשמוע היום?",
    "אתה באמת יודע הכל?",
]


def build_llm(cfg, config_path, url=None) -> LlamaServerLLM:
    """The oracle's LlamaServerLLM (real prompt and world) for this config."""
    world = load_world(resolve_repo_path(cfg.conversation.world_file, config_path))
    return LlamaServerLLM(url or cfg.servers.llm_url, max(cfg.servers.timeout_s, 60),
                          max_sentences=cfg.conversation.max_reply_sentences,
                          max_chars=cfg.conversation.max_reply_chars, world=world,
                          sampling=cfg.llm.sampling())


def run_check(llm, n: int, show=print, budget_s=None, max_consecutive_errors=None,
              clock=time.monotonic, rng=None) -> dict:
    """Send n Hebrew questions, one fresh conversation each; show() each reply.
    Stops early once budget_s has passed (checked between questions, so one request
    can overrun it by its timeout) or after max_consecutive_errors failed requests in
    a row. With rng, each question gets an answer angle picked the way the
    Controller picks it (unlike the recent ones). Returns {"n", "singular", "answered", "failed", "aborted" (None or why),
    "replies": [...]}."""
    singular = failed = in_a_row = 0
    replies = []
    aborted = None
    recent = []
    deadline = None if budget_s is None else clock() + budget_s
    for i in range(n):
        if deadline is not None and clock() >= deadline:
            aborted = f"the budget of {budget_s:g} s is used up"
            break
        if max_consecutive_errors and in_a_row >= max_consecutive_errors:
            aborted = f"{in_a_row} errors in a row (llama-server stalled?)"
            break
        question = QUESTIONS[i % len(QUESTIONS)]
        kwargs = {}
        if rng is not None:
            kwargs["angle"] = pick_angle(rng, recent)
            recent.append(kwargs["angle"])
        try:
            reply = llm.reply([], question, "he", **kwargs)
        except LLMError as e:
            failed += 1
            in_a_row += 1
            replies.append({"question": question, "error": str(e)})
            show(f"[{i + 1}] {question}\n    ERROR: {e}\n")
            continue
        in_a_row = 0
        markers = singular_address(reply)
        singular += bool(markers)
        replies.append({"question": question, "reply": reply, "markers": markers})
        flag = f"SINGULAR {' '.join(markers)}" if markers else "plural"
        show(f"[{i + 1}] {question}\n    {reply}\n    -> {flag}\n")
    answered = sum("reply" in r for r in replies)
    return {"n": n, "singular": singular, "answered": answered, "failed": failed,
            "aborted": aborted, "replies": replies}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=len(QUESTIONS), help="questions to send")
    ap.add_argument("--url", default=None, help="llama-server (default: the config's)")
    ap.add_argument("--config", default=os.path.join(REPO, "config", "mac.toml"))
    ap.add_argument("--no-angle", action="store_true",
                    help="bare questions, without the Controller's answer angles")
    ap.add_argument("--seed", type=int, default=None, help="angle picks (default: random)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    rng = None if args.no_angle else random.Random(args.seed)
    res = run_check(build_llm(cfg, args.config, args.url), args.n, rng=rng)
    singular, answered, failed = res["singular"], res["answered"], res["failed"]
    if res["aborted"]:
        print(f"stopped early: {res['aborted']}")
    rate = f"{100.0 * singular / answered:.0f}%" if answered else "n/a"
    print(f"singular address in {singular}/{answered} replies ({rate})"
          + (f"; {failed} request(s) failed" if failed else ""))
    return 1 if answered == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
