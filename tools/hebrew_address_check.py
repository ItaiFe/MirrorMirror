#!/usr/bin/env python3
"""How often does the running llama-server address a Hebrew visitor in the singular?

    python tools/hebrew_address_check.py [--n 20] [--url http://127.0.0.1:8080]
                                         [--config config/orin.toml]

Sends N Hebrew questions (cycling through a fixed list: neutral, masculine and
feminine singular, plural) through LlamaServerLLM with the real oracle prompt and
world, one fresh conversation each, and prints every reply with the singular-address
markers mirror.persona.singular_address finds, then the rate. The check is a
heuristic: read the replies too (it misses past-tense forms such as "ששכחת").
"""
import argparse
import os
import sys

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.insert(0, REPO)

from mirror.config import load_config, resolve_repo_path  # noqa: E402
from mirror.llm import LLMError, LlamaServerLLM  # noqa: E402
from mirror.persona import load_world, singular_address  # noqa: E402

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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=len(QUESTIONS), help="questions to send")
    ap.add_argument("--url", default=None, help="llama-server (default: the config's)")
    ap.add_argument("--config", default=os.path.join(REPO, "config", "mac.toml"))
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    url = args.url or cfg.servers.llm_url
    world = load_world(resolve_repo_path(cfg.conversation.world_file, args.config))
    llm = LlamaServerLLM(url, max(cfg.servers.timeout_s, 60),
                         max_sentences=cfg.conversation.max_reply_sentences,
                         max_chars=cfg.conversation.max_reply_chars, world=world)

    singular = failed = 0
    for i in range(args.n):
        question = QUESTIONS[i % len(QUESTIONS)]
        try:
            reply = llm.reply([], question, "he")
        except LLMError as e:
            failed += 1
            print(f"[{i + 1}] {question}\n    ERROR: {e}\n")
            continue
        markers = singular_address(reply)
        singular += bool(markers)
        flag = f"SINGULAR {' '.join(markers)}" if markers else "plural"
        print(f"[{i + 1}] {question}\n    {reply}\n    -> {flag}\n")
    answered = args.n - failed
    rate = f"{100.0 * singular / answered:.0f}%" if answered else "n/a"
    print(f"singular address in {singular}/{answered} replies ({rate})"
          + (f"; {failed} request(s) failed" if failed else ""))
    return 1 if failed == args.n else 0


if __name__ == "__main__":
    sys.exit(main())
