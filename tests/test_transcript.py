import datetime
import json
import logging

from mirror.transcript import ConversationLog


def test_one_line_per_exchange_one_file_per_day(tmp_path):
    now = [datetime.datetime(2026, 11, 3, 23, 59, 30)]
    log = ConversationLog(str(tmp_path / "conv"), wall_clock=lambda: now[0])
    log.write("מה העתיד שלי", "Bright lights, darling.", "en")
    now[0] = datetime.datetime(2026, 11, 4, 0, 1, 0)
    log.write("Who are you?", "A mirror, honey.", "en", failed=False)
    day1 = (tmp_path / "conv" / "2026-11-03.jsonl").read_text(encoding="utf-8").splitlines()
    day2 = (tmp_path / "conv" / "2026-11-04.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(day1[0]) == {"time": "2026-11-03T23:59:30", "heard": "מה העתיד שלי",
                                   "reply": "Bright lights, darling.", "lang": "en"}
    assert json.loads(day2[0])["failed"] is False and len(day1) == len(day2) == 1
    assert "מה העתיד שלי" in day1[0]  # Hebrew kept readable, not \\u-escaped


def test_an_unwritable_directory_only_warns_once(tmp_path, caplog):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    log = ConversationLog(str(blocker / "conv"))
    with caplog.at_level(logging.WARNING):
        log.write("a", "b", "en")
        log.write("a", "b", "en")
    assert caplog.text.count("could not save") == 1
