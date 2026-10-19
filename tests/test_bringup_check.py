"""tools/bringup_check.py: the runner, status aggregation, thresholds and the report,
with fakes only (no devices, no servers)."""
import json
import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "tools"))

import bringup_check as bc  # noqa: E402
import record_demo as rd  # noqa: E402

from mirror.config import Config  # noqa: E402
from mirror.stt import Transcript  # noqa: E402


# --- runner and aggregation -----------------------------------------------------
def test_runner_runs_in_order_and_turns_a_crash_into_a_fail():
    seen = []

    def ok(ctx):
        seen.append("ok")
        return bc.Result("ok", bc.PASS, "fine")

    def boom(ctx):
        seen.append("boom")
        raise RuntimeError("no device")

    def two(ctx):
        # a later check can see earlier results
        assert ctx.status_of("boom") == bc.FAIL
        return [bc.Result("a", bc.WARN, "meh"), bc.Result("b", bc.PASS, "good")]

    reported = []
    ctx = bc.Context()
    res = bc.run_checks([bc.Check("ok", ok), bc.Check("boom", boom),
                         bc.Check("never", ok, skip="--quick"), bc.Check("two", two)],
                        ctx, reported.append)
    assert seen == ["ok", "boom"]  # the skipped check never ran
    assert [(r.name, r.status) for r in res] == [
        ("ok", "PASS"), ("boom", "FAIL"), ("never", "SKIP"), ("a", "WARN"), ("b", "PASS")]
    assert "RuntimeError: no device" in res[1].reason
    assert "traceback" in res[1].details
    assert res[2].reason == "--quick"
    assert reported == res


@pytest.mark.parametrize("statuses,want,code", [
    ([], "PASS", 0),
    (["SKIP", "SKIP"], "PASS", 0),
    (["PASS", "SKIP"], "PASS", 0),
    (["PASS", "WARN", "SKIP"], "WARN", 0),
    (["WARN", "FAIL", "PASS"], "FAIL", 1),
])
def test_overall_and_exit_code(statuses, want, code):
    results = [bc.Result(f"c{i}", s, "") for i, s in enumerate(statuses)]
    assert bc.overall(results) == want
    assert bc.exit_code(results) == code


def test_summary_table_lists_every_check_and_the_totals():
    results = [bc.Result("config", bc.PASS, "loaded"),
               bc.Result("memory", bc.FAIL, "x" * 200)]
    table = bc.summary_table(results)
    assert "config" in table and "loaded" in table
    assert "x" * 200 not in table and "..." in table  # long reasons are cut
    assert table.splitlines()[-1] == "overall FAIL: 1 pass, 0 warn, 1 fail, 0 skipped"


def test_json_report(tmp_path):
    results = [bc.Result("config", bc.PASS, "loaded", details={"files": {"vad": "x"}}),
               bc.Result("e2e Hebrew", bc.WARN, "slow: 8.1 s", "README", {"heard": "מראה"})]
    path = tmp_path / "sub" / "report.json"
    bc.write_json(str(path), results, {"config": "c.toml"})
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["overall"] == "WARN" and doc["exit_code"] == 0
    assert doc["counts"] == {"PASS": 1, "WARN": 1, "FAIL": 0, "SKIP": 0}
    assert doc["meta"] == {"config": "c.toml"}
    assert doc["results"][1]["details"]["heard"] == "מראה"  # not \u-escaped garbage
    assert doc["results"][1]["fix"] == "README"
    assert set(doc["results"][0]) == {"name", "status", "reason", "fix", "details", "seconds"}


# --- memory ------------------------------------------------------------------------
@pytest.mark.parametrize("mb,want", [(0, "FAIL"), (299, "FAIL"), (300, "WARN"),
                                     (499, "WARN"), (500, "PASS"), (4000, "PASS")])
def test_memory_thresholds(mb, want):
    assert bc.memory_status(mb) == want


def test_memory_result_notes_when_servers_are_down():
    r = bc.memory_result(800, "free -m", servers_up=False, details={})
    assert r.status == "PASS" and "servers down" in r.reason and r.fix
    r = bc.memory_result(450, "free -m", servers_up=True, details={})
    assert r.status == "WARN" and "Memory on the Orin" in r.fix
    assert r.details["available_mb"] == 450


def test_parse_free():
    text = ("               total        used        free      shared  buff/cache   available\n"
            "Mem:            7620        6900         120          30         600         512\n"
            "Swap:           3810         100        3710\n")
    assert bc.parse_free(text) == {"total": 7620, "used": 6900, "available": 512}
    assert bc.parse_free("") is None


def test_parse_tegrastats():
    line = "10-24-2026 12:00:00 RAM 7012/7620MB (lfb 4x4MB) SWAP 120/3810MB CPU [5%@1510]"
    assert bc.parse_tegrastats(line) == (7012, 7620)
    assert bc.parse_tegrastats("nonsense") is None


def test_parse_vm_stat():
    text = ("Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
            "Pages free:                               10000.\n"
            "Pages active:                            200000.\n"
            "Pages inactive:                           20000.\n"
            "Pages speculative:                         2000.\n"
            "Pages purgeable:                            768.\n")
    vm = bc.parse_vm_stat(text)
    assert vm == {"page": 16384, "free": 10000, "inactive": 20000, "speculative": 2000,
                  "purgeable": 768}
    assert bc.mac_available_mb(vm) == pytest.approx(32768 * 16384 / 1048576)


# --- display, servers, config ------------------------------------------------------------
def test_parse_xrandr():
    text = ("Screen 0: minimum 8 x 8, current 1920 x 1080, maximum 32767 x 32767\n"
            "DP-0 disconnected (normal left inverted right x axis y axis)\n"
            "DP-1 connected primary 1920x1080+0+0 (normal left) 530mm x 300mm\n"
            "   1920x1080     60.00*+  50.00\n")
    assert bc.parse_xrandr(text) == ("DP-1", 1920, 1080)
    assert bc.parse_xrandr("HDMI-0 connected 1280x720+0+0 (normal)") == ("HDMI-0", 1280, 720)
    assert bc.parse_xrandr("DP-1 connected (normal left)") is None  # connected, no mode


def test_unit_env_takes_the_last_assignment():
    unit = "[Service]\nEnvironment=STT_MODEL=a.bin\nExecStart=x\nEnvironment=STT_MODEL=b.bin\n"
    assert bc._unit_env(unit, "STT_MODEL") == "b.bin"
    assert bc._unit_env(unit, "LLM_MODEL") is None


def test_check_files_reports_missing_and_truncated(tmp_path):
    good = tmp_path / "en_US-lessac-medium.onnx.json"
    good.write_text("{" + " " * 300 + "}")
    small = tmp_path / "silero_vad.onnx"
    small.write_bytes(b"x" * 10)
    missing, too_small, sizes = bc.check_files(
        [("cfg", str(good)), ("vad", str(small)), ("llm", str(tmp_path / "nope.gguf"))])
    assert len(missing) == 1 and "nope.gguf" in missing[0]
    assert len(too_small) == 1 and "silero_vad.onnx" in too_small[0]
    assert sizes[str(good)] > 200


def test_model_files_follow_the_tts_engine():
    cfg = Config()
    labels = {label for label, _ in bc.model_files(cfg)}
    assert {"vad", "piper en voice", "phonikud he voice", "renikud g2p"} <= labels
    assert "pocket model" not in labels
    cfg.tts.engine, cfg.tts.he_g2p = "pocket", "phonikud"
    labels = {label for label, _ in bc.model_files(cfg)}
    assert {"pocket model", "renikud g2p"} <= labels


def test_servers_fail_with_the_start_command(monkeypatch):
    monkeypatch.setattr(bc, "health", lambda url: (url.endswith("8081"), "x"))
    monkeypatch.setattr(bc, "is_orin", lambda: True)
    ctx = bc.Context(cfg=Config())
    stt, llm = bc.check_servers(ctx)
    assert stt.status == "PASS" and llm.status == "FAIL"
    assert "systemctl --user start mirror-stt mirror-llm" in llm.fix


def test_serial_is_a_warn_with_log_lights():
    r = bc.check_serial(bc.Context(cfg=Config()))
    assert r.status == "WARN" and "log" in r.reason


def test_audio_result():
    base = {"devices": [{"index": 0, "name": "MacBook Mic", "channels": 1}],
            "wanted": "XVF3800", "index": None, "name": "MacBook Mic"}
    r = bc.audio_result("input", dict(base, matched=False))
    assert r.status == "FAIL" and "XVF3800" in r.reason and "MacBook Mic" in r.reason
    ok = dict(base, wanted=None, matched=True, rate=48000, channels=2, ok=True)
    r = bc.audio_result("input", dict(ok, reads=30, peak=0.02))
    assert r.status == "PASS" and "48000 Hz x 2 ch" in r.reason
    assert bc.audio_result("input", dict(ok, reads=30, peak=0.0)).status == "WARN"
    r = bc.audio_result("output", dict(ok, error="hung after stage 'opened'"))
    assert r.status == "FAIL" and "48000 Hz x 2 ch" in r.reason


def test_parse_probe_merges_lines_and_survives_noise():
    out = 'log line\nPROBE {"name": "X", "matched": true}\nPROBE {"rate": 16000}\n'
    assert bc.parse_probe(out.encode()) == {"name": "X", "matched": True, "rate": 16000}
    assert bc.parse_probe(None) == {}


# --- end to end and Hebrew --------------------------------------------------------------
@pytest.mark.parametrize("wake,first,want", [
    (True, 3.0, "PASS"), (True, 6.0, "PASS"), (True, 9.9, "WARN"), (True, 10.5, "FAIL"),
    (False, 2.0, "FAIL"), (True, None, "FAIL")])
def test_latency_status(wake, first, want):
    assert bc.latency_status(wake, first) == want


def test_turn_result_warns_on_the_wrong_language():
    turn = {"heard": "Mirror, mirror", "wake": True, "lang": "he", "filler_s": 1.0,
            "first_sentence_s": 4.0}
    r = bc.turn_result("en", turn)
    assert r.status == "WARN" and "expected en" in r.reason
    assert bc.turn_result("he", turn).status == "PASS"
    assert bc.turn_result("en", {}).status == "FAIL"


class FakeVad:
    def reset(self):
        pass


class FakeSegmenter:
    """Cuts one utterance at the first chunk past `cut_s` of audio."""
    min_silence_s = 0.5

    def __init__(self, cut_s):
        self.cut_s, self.n = cut_s, 0

    def reset(self):
        self.n = 0

    def feed(self, chunk):
        self.n += len(chunk)
        return [np.zeros(16000, np.float32)] if self.n >= self.cut_s * 16000 else []


class FakeSTT:
    def transcribe(self, samples, sr=16000):
        return Transcript("Mirror, mirror on the wall, what do you see?", "en")


class FakeLLM:
    def warm(self, lang):
        return 0.0

    def reply_stream(self, history, text, lang, cancel=None, angle=None):
        time.sleep(1.2)  # the first sentence takes 1.2 s (real time: the timeline's clock)
        yield "I see a traveller."
        yield "Dust follows you."


class FakeTTS:
    def synth_visemes(self, text, lang):
        return np.full(1000, 0.3, np.float32), 1000, None  # 1 s clips

    def synth(self, text, lang):
        return self.synth_visemes(text, lang)[:2]

    def warmup(self):
        pass


class FakeFillers:
    def __init__(self, tts):
        self.clip = tts.synth_visemes("hmm", "en")

    def pick(self, lang):
        return self.clip


def test_run_turn_measures_filler_and_first_sentence_from_end_of_speech(tmp_path):
    from mirror.config import ConversationConfig
    from mirror.conversation import Controller

    wav = tmp_path / "q.wav"
    rd.write_wav(str(wav), np.full(16000, 0.2, np.float32), 16000)  # 1 s "question"
    tl = rd.Timeline(cap_s=1e9)
    texts: dict = {}
    player = rd.TimelinePlayer(tl, texts)
    face = rd.TimelineFace(tl)
    stt = rd.TimedSTT(FakeSTT())
    llm = rd.TimedLLM(FakeLLM())
    tts = rd.TextTTS(FakeTTS(), texts, kind="filler")
    fillers = FakeFillers(tts)
    tts.kind = "reply"
    ctl = Controller(stt, llm, tts, player, rd._NullLights(), face, ConversationConfig(),
                     clock=tl.now, fillers=fillers, first_sentence_timeout_s=5.0)
    seg = FakeSegmenter(cut_s=1.5)  # speech ends at 1.0, the VAD cuts at 1.5
    turn = bc.run_turn(ctl, seg, FakeVad(), tl, player, stt, llm, str(wav), 16000)
    ctl.close()
    assert turn["wake"] is True and turn["lang"] == "en"
    assert turn["filler_s"] == pytest.approx(0.5, abs=0.15)  # just the hang-over
    # the 1 s filler is over at 1.5 s; the sentence is ready 1.2 s after the cut: 1.7 s
    assert turn["first_sentence_s"] == pytest.approx(1.7, abs=0.15)
    assert turn["reply"] == "I see a traveller. Dust follows you."
    assert bc.turn_result("en", turn).status == "PASS"


def test_question_wavs_are_cached(tmp_path):
    calls = []

    class CountingTTS(FakeTTS):
        def synth_visemes(self, text, lang):
            calls.append(lang)
            return super().synth_visemes(text, lang)

    cfg = Config()
    first = bc.question_wavs(cfg, str(tmp_path), CountingTTS())
    assert calls == ["en", "he"]
    assert [v[0] for v in first.values()] == ["en", "he"]
    assert all(os.path.exists(v[2]) for v in first.values())
    again = bc.question_wavs(cfg, str(tmp_path), CountingTTS())
    assert again == first and calls == ["en", "he"]  # no new synthesis
    cfg.tts.pocket_voice, cfg.tts.engine = "michael", "pocket"
    assert bc.question_wavs(cfg, str(tmp_path), CountingTTS()) != first  # new voice, new key


def test_hebrew_result():
    r = bc.hebrew_result({"n": 10, "singular": 2, "answered": 10, "failed": 0})
    assert r.status == "PASS" and "2/10" in r.reason
    assert bc.hebrew_result({"n": 10, "singular": 3, "answered": 10, "failed": 0}).status == "WARN"
    assert bc.hebrew_result({"n": 10, "singular": 0, "answered": 9, "failed": 1}).status == "WARN"
    assert bc.hebrew_result({"n": 10, "singular": 0, "answered": 0, "failed": 10}).status == "FAIL"


def test_hebrew_address_run_check_counts_singular_replies():
    import hebrew_address_check as hac
    from mirror.llm import LLMError

    replies = iter(["אתם תמצאו את הדרך.", "אתה תמצא את הדרך.", LLMError("down")])

    class LLM:
        def reply(self, history, q, lang):
            r = next(replies)
            if isinstance(r, Exception):
                raise r
            return r

    res = hac.run_check(LLM(), 3, show=lambda s: None)
    assert res["failed"] == 1 and res["answered"] == 2
    assert res["singular"] == 1


def test_hebrew_address_run_check_sends_angles_like_the_controller():
    import random

    import hebrew_address_check as hac

    angles = []

    class LLM:
        def reply(self, history, q, lang, angle=None):
            angles.append(angle)
            return "אתם תמצאו את הדרך."

    hac.run_check(LLM(), 4, show=lambda s: None, rng=random.Random(1))
    assert all(a is not None for a in angles) and len(set(angles)) > 1


def test_e2e_and_hebrew_fail_fast_without_servers():
    ctx = bc.Context(cfg=Config())
    ctx.results = [bc.Result("stt server", bc.FAIL, ""), bc.Result("llm server", bc.FAIL, "")]
    assert [r.status for r in bc.check_e2e(ctx)] == ["FAIL", "FAIL"]
    assert bc.check_hebrew(ctx).status == "FAIL"


# --- review fixes: hard deadlines -----------------------------------------------------
SLEEPER = [sys.executable, "-c", "import time; time.sleep(30)"]


def test_tegrastats_that_prints_nothing_returns_none_within_the_deadline():
    t0 = time.monotonic()
    assert bc._tegrastats_line(timeout_s=1.0, cmd=SLEEPER) is None
    assert time.monotonic() - t0 < 4.0


def test_tegrastats_returns_its_first_line_early():
    cmd = [sys.executable, "-c",
           "import sys, time; print('RAM 7000/7620MB SWAP 0/3810MB'); sys.stdout.flush(); "
           "time.sleep(30)"]
    t0 = time.monotonic()
    assert bc._tegrastats_line(timeout_s=3.0, cmd=cmd) == "RAM 7000/7620MB SWAP 0/3810MB"
    assert time.monotonic() - t0 < 2.5


def test_run_probe_kills_a_hung_child_and_keeps_its_partial_report():
    cmd = [sys.executable, "-c",
           "import sys, time; print('PROBE {\"name\": \"XVF3800\", \"stage\": \"opened\", "
           "\"rate\": 16000, \"channels\": 2}'); sys.stdout.flush(); time.sleep(30)"]
    t0 = time.monotonic()
    probe = bc._run_probe("input", "unused.toml", timeout_s=1.0, cmd=cmd)
    assert time.monotonic() - t0 < 4.0
    assert probe["name"] == "XVF3800" and "hung after stage 'opened'" in probe["error"]
    r = bc.audio_result("input", probe)
    assert r.status == "FAIL" and "16000 Hz x 2 ch" in r.reason


def test_run_child_reports_a_timeout():
    t0 = time.monotonic()
    out = bc._run_child(SLEEPER, 0.5)
    assert out["timed_out"] and time.monotonic() - t0 < 4.0


@pytest.mark.parametrize("turns,statuses,abort", [
    # 2 LLM errors in a row: the third turn never runs
    ([{"llm_error": True}, {"llm_error": True}, {"ok": True}], ["FAIL", "FAIL", "FAIL"],
     "2 LLM errors in a row"),
    ([{"ok": True}, {"llm_error": True}, {"ok": True}], ["PASS", "FAIL", "PASS"], None),
])
def test_run_turns_aborts_after_consecutive_llm_errors(turns, statuses, abort):
    ran = []

    def make(spec, i):
        def fn():
            ran.append(i)
            if spec.get("ok"):
                return {"heard": "mirror mirror", "wake": True, "lang": "en",
                        "filler_s": 0.5, "first_sentence_s": 2.0}
            return {"heard": "mirror mirror", "wake": True, "lang": "en",
                    "first_sentence_s": None, "llm_error": True}
        return fn

    specs = [(f"t{i}", "en", make(s, i)) for i, s in enumerate(turns)]
    res = bc.run_turns(specs, budget_s=10.0)
    assert [r.status for r in res] == statuses
    if abort:
        assert ran == [0, 1] and abort in res[2].reason
    else:
        assert ran == [0, 1, 2]


def test_run_turns_stops_a_stalled_turn_at_the_budget():
    def stalled():
        time.sleep(5)
        return {}

    def never():
        raise AssertionError("must not run after the budget is spent")

    t0 = time.monotonic()
    res = bc.run_turns([("e2e English", "en", stalled), ("e2e Hebrew", "he", never)],
                       budget_s=0.5)
    assert time.monotonic() - t0 < 2.0
    assert [r.status for r in res] == ["FAIL", "FAIL"]
    assert "budget" in res[0].reason and "budget" in res[1].reason


def test_hebrew_run_check_aborts_after_two_errors_in_a_row():
    import hebrew_address_check as hac
    from mirror.llm import LLMError

    calls = []

    class Down:
        def reply(self, history, q, lang):
            calls.append(q)
            raise LLMError("read timeout")

    res = hac.run_check(Down(), 10, show=lambda s: None, max_consecutive_errors=2)
    assert len(calls) == 2 and res["failed"] == 2
    assert "2 errors in a row" in res["aborted"]
    r = bc.hebrew_result(res)
    assert r.status == "FAIL" and "2 errors in a row" in r.reason


def test_hebrew_run_check_stops_at_its_budget():
    import hebrew_address_check as hac

    now = [0.0]

    class Slow:
        def reply(self, history, q, lang):
            now[0] += 50.0
            return "אתם תמצאו."

    res = hac.run_check(Slow(), 10, show=lambda s: None, budget_s=120.0,
                        clock=lambda: now[0])
    assert res["answered"] == 3 and "budget" in res["aborted"]
    r = bc.hebrew_result(res)
    assert r.status == "WARN" and "3/3" not in r.reason and "budget" in r.reason


# --- review fixes: systemd Environment= parsing and precedence ---------------------------------
@pytest.mark.parametrize("text,want", [
    ("Environment=STT_MODEL=a.bin", "a.bin"),
    ("Environment=FOO=1 STT_MODEL=b.bin BAR=2", "b.bin"),
    ('Environment="STT_MODEL=c d.bin" FOO=1', "c d.bin"),
    ("Environment='FOO=x' 'STT_MODEL=e.bin'", "e.bin"),
    ("Environment=STT_MODEL=a.bin\nEnvironment=STT_MODEL=f.bin", "f.bin"),
    ("Environment=STT_MODEL=a.bin\nEnvironment=\n", None),  # an empty assignment resets
    ("Environment=STT_MODEL=a.bin\nEnvironment=\nEnvironment=STT_MODEL=g.bin", "g.bin"),
    ("# Environment=STT_MODEL=commented.bin\nEnvironment=STT_MODEL=h.bin", "h.bin"),
    ("Environment=STT_MODELX=no.bin", None),
])
def test_unit_env_forms(text, want):
    assert bc._unit_env(text, "STT_MODEL") == want


def _units(tmp_path, repo_stt="ivrit.bin", override=None, installed=None):
    repo = tmp_path / "repo"
    user = tmp_path / "user"
    repo.mkdir()
    user.mkdir()
    (repo / "mirror-stt.service").write_text(f"[Service]\nEnvironment=STT_MODEL={repo_stt}\n")
    (repo / "mirror-llm.service").write_text("[Service]\nEnvironment=LLM_MODEL=gemma.gguf\n")
    if installed:
        (user / "mirror-stt.service").write_text(f"[Service]\nEnvironment=STT_MODEL={installed}\n")
    if override:
        d = user / "mirror-stt.service.d"
        d.mkdir()
        (d / "override.conf").write_text(f"[Service]\nEnvironment=STT_MODEL={override}\n")
    return str(repo), str(user)


def test_server_model_names_dropin_beats_the_unit(tmp_path):
    repo, user = _units(tmp_path, override="ggml-small.bin")
    names = bc.server_model_names(prefer_units=True, env={}, units_dir=user, repo_units_dir=repo)
    assert names["stt"]["name"] == "ggml-small.bin"
    assert "override.conf" in names["stt"]["source"]
    assert names["llm"] == {"name": "gemma.gguf", "source": os.path.join(repo, "mirror-llm.service")}


def test_server_model_names_installed_unit_replaces_the_repo_copy(tmp_path):
    repo, user = _units(tmp_path, installed="installed.bin")
    names = bc.server_model_names(prefer_units=True, env={}, units_dir=user, repo_units_dir=repo)
    assert names["stt"]["name"] == "installed.bin"


def test_server_model_names_env_precedence(tmp_path):
    repo, user = _units(tmp_path, override="ggml-small.bin")
    env = {"STT_MODEL": "models/from-env.bin"}
    # the Orin: what systemd will actually run wins over a stray shell variable
    on_orin = bc.server_model_names(prefer_units=True, env=env, units_dir=user,
                                    repo_units_dir=repo)
    assert on_orin["stt"]["name"] == "ggml-small.bin"
    # the Mac: make servers-mac takes STT_MODEL from the environment
    on_mac = bc.server_model_names(prefer_units=False, env=env, units_dir=user,
                                   repo_units_dir=repo)
    assert on_mac["stt"] == {"name": "from-env.bin", "source": "$STT_MODEL"}
    # without an env value the units decide on the Mac too
    assert bc.server_model_names(prefer_units=False, env={}, units_dir=user,
                                 repo_units_dir=repo)["stt"]["name"] == "ggml-small.bin"


# --- review fixes: display, serial, memory dispatch -----------------------------------------
XRANDR = ("Screen 0: minimum 8 x 8, current 1280 x 720\n"
          "DP-1 connected primary 1280x720+0+0 (normal left) 0mm x 0mm\n")


def _orin_display(monkeypatch, stdout, rc=0):
    monkeypatch.setattr(bc, "is_orin", lambda: True)
    monkeypatch.setattr(bc.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setenv("DISPLAY", ":0")
    seen = {}

    def fake_child(cmd, timeout_s, env=None):
        seen["cmd"], seen["env"] = cmd, env
        return {"rc": rc, "stdout": stdout, "stderr": "", "timed_out": False}

    monkeypatch.setattr(bc, "_run_child", fake_child)
    return seen


def test_check_display_warns_on_the_wrong_mode(monkeypatch):
    seen = _orin_display(monkeypatch, XRANDR)
    r = bc.check_display(bc.Context(cfg=Config()))
    assert seen["cmd"] == ["xrandr"]
    assert r.status == "WARN" and "1280x720" in r.reason and "1920x1080" in r.reason


def test_check_display_passes_at_1920x1080(monkeypatch):
    _orin_display(monkeypatch, XRANDR.replace("1280x720", "1920x1080"))
    r = bc.check_display(bc.Context(cfg=Config()))
    assert r.status == "PASS" and r.details["output"] == "DP-1"


def test_check_display_skips_off_the_orin(monkeypatch):
    monkeypatch.setattr(bc, "is_orin", lambda: False)
    assert bc.check_display(bc.Context(cfg=Config())).status == "SKIP"


def test_check_serial_sends_the_idle_preset(monkeypatch, tmp_path):
    import types

    port = tmp_path / "ttyUSB0"
    port.write_text("")
    sent = []

    class FakeSerial:
        def __init__(self, path, baud, timeout=None, write_timeout=None):
            sent.append(("open", path, baud, write_timeout))

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            sent.append("close")

        def write(self, data):
            sent.append(data)

        def flush(self):
            pass

    monkeypatch.setitem(sys.modules, "serial", types.SimpleNamespace(Serial=FakeSerial))
    cfg = Config()
    cfg.lights.kind, cfg.lights.serial_port = "wled", str(port)
    r = bc.check_serial(bc.Context(cfg=cfg))
    assert r.status == "PASS"
    assert sent == [("open", str(port), 115200, 1), b'{"ps":1}\n', "close"]


def test_check_serial_fails_on_a_missing_port():
    cfg = Config()
    cfg.lights.kind, cfg.lights.serial_port = "wled", "/dev/definitely-not-a-port"
    assert bc.check_serial(bc.Context(cfg=cfg)).status == "FAIL"


VM_STAT = ("Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
           "Pages free:                               10000.\n"
           "Pages inactive:                           10000.\n")
FREE = ("               total        used        free      shared  buff/cache   available\n"
        "Mem:            7620        7300         100          30         200         280\n")


def _fake_cmds(monkeypatch, outputs, calls):
    def fake(cmd, timeout_s=10.0):
        calls.append(cmd[0])
        return outputs[cmd[0]]
    monkeypatch.setattr(bc, "_cmd_output", fake)


def test_check_memory_on_the_mac_uses_vm_stat(monkeypatch):
    calls = []
    monkeypatch.setattr(bc, "_platform", lambda: "darwin")
    _fake_cmds(monkeypatch, {"vm_stat": VM_STAT,
                             "sysctl": "total = 2048.00M  used = 100.00M  free = 1948.00M"},
               calls)
    r = bc.check_memory(bc.Context(cfg=Config()))
    assert calls == ["vm_stat", "sysctl"]
    assert r.status == "WARN"  # 312 MB
    assert r.details["available_mb"] == round(20000 * 16384 / 1048576)
    assert "swap used 100 MB" in r.reason


def test_check_memory_on_the_orin_uses_free_and_tegrastats(monkeypatch):
    calls = []
    monkeypatch.setattr(bc, "_platform", lambda: "linux")
    monkeypatch.setattr(bc, "is_orin", lambda: True)
    monkeypatch.setattr(bc.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(bc, "_tegrastats_line", lambda: "RAM 7300/7620MB SWAP 0/3810MB")
    _fake_cmds(monkeypatch, {"free": FREE}, calls)
    r = bc.check_memory(bc.Context(cfg=Config()))
    assert calls == ["free"]
    assert r.status == "FAIL" and "280 MB" in r.reason and "7300/7620MB" in r.reason


def test_default_wav_cache_is_in_the_repo_and_gitignored():
    assert bc.DEFAULT_CACHE == os.path.join(bc.REPO, ".cache", "bringup")
    with open(os.path.join(bc.REPO, ".gitignore"), encoding="utf-8") as f:
        assert ".cache/" in f.read().split()


# --- re-review: abandoned turns -----------------------------------------------------------
import threading  # noqa: E402


def test_run_turns_marks_a_stalled_turn_abandoned():
    abandoned = threading.Event()
    release = threading.Event()
    seen = {}

    def stalled():
        release.wait(5)
        seen["abandoned_after_release"] = abandoned.is_set()
        return {}

    res = bc.run_turns([("e2e English", "en", stalled)], budget_s=0.3, abandoned=abandoned)
    assert res[0].status == "FAIL" and abandoned.is_set()
    release.set()
    time.sleep(0.2)
    assert seen["abandoned_after_release"] is True


def test_run_turns_leaves_abandoned_clear_when_all_finish():
    abandoned = threading.Event()
    ok = {"heard": "mirror mirror", "wake": True, "lang": "en", "first_sentence_s": 1.0}
    bc.run_turns([("t", "en", lambda: ok)], budget_s=5.0, abandoned=abandoned)
    assert not abandoned.is_set()


def test_guarded_components_refuse_work_once_abandoned():
    from mirror.llm import LLMError
    from mirror.stt import STTError

    abandoned = threading.Event()

    class LLM:
        calls = ["x"]

        def warm(self, lang):
            return 0.1

        def reply_stream(self, history, text, lang, cancel=None, angle=None):
            yield "one."
            abandoned.set()  # abandoned mid-reply
            yield "two."

    stt = bc.GuardedSTT(FakeSTT(), abandoned)
    llm = bc.GuardedLLM(LLM(), abandoned)
    tts = bc.GuardedTTS(FakeTTS(), abandoned)
    assert stt.transcribe(np.zeros(10)).lang == "en"
    assert llm.calls == ["x"]  # pass-through attributes (TimedLLM.calls)
    assert llm.warm("en") == 0.1
    assert tts.synth_visemes("hi", "en")[1] == 1000
    got = []
    with pytest.raises(LLMError):
        for s in llm.reply_stream([], "q", "en"):
            got.append(s)
    assert got == ["one."]  # stopped before the second sentence
    with pytest.raises(STTError):
        stt.transcribe(np.zeros(10))
    with pytest.raises(LLMError):
        llm.warm("he")
    with pytest.raises(RuntimeError):
        tts.synth_visemes("hi", "en")


def test_turn_fn_does_nothing_and_prints_nothing_once_abandoned():
    abandoned = threading.Event()
    shown, ran = [], []

    def run(wav):
        ran.append(wav)
        abandoned.set()  # the budget ran out while this turn was busy
        return {"heard": "h", "reply": "r"}

    fn = bc.make_turn_fn(run, "a.wav", abandoned, show=shown.append)
    fn()
    assert ran == ["a.wav"] and shown == []  # finished late: no print
    fn()
    assert ran == ["a.wav"]  # never starts once abandoned
    abandoned.clear()
    bc.make_turn_fn(lambda wav: {"heard": "h", "reply": "r"}, "b.wav", abandoned,
                    show=shown.append)()
    assert len(shown) == 1 and "'h'" in shown[0]


def test_hebrew_is_skipped_after_an_abandoned_turn():
    ctx = bc.Context(cfg=Config())
    ctx.results = [bc.Result("llm server", bc.PASS, "")]
    ctx.abandoned.set()
    r = bc.check_hebrew(ctx)
    assert r.status == "WARN"
    assert r.reason == "skipped: an end-to-end turn was abandoned (LLM busy)"


def test_finish_hard_exits_only_when_a_thread_was_abandoned():
    exits = []
    abandoned = threading.Event()
    assert bc.finish(1, abandoned, _exit=exits.append) == 1 and exits == []
    abandoned.set()
    bc.finish(0, abandoned, _exit=exits.append)
    assert exits == [0]


def test_warmup_outcome():
    names = ("e2e English", "e2e Hebrew")
    assert bc.warmup_failure(names, True, [], None) is None
    assert bc.warmup_failure(names, True, ["he"], None) is None  # one cold slot is fine
    res = bc.warmup_failure(names, True, None, RuntimeError("boom"))
    assert [r.status for r in res] == ["FAIL", "FAIL"]
    assert "warmup crashed: RuntimeError: boom" in res[0].reason
    res = bc.warmup_failure(names, True, ["en", "he"], None)
    assert "2 LLM warmup errors" in res[0].reason
    res = bc.warmup_failure(names, False, None, None)
    assert "did not finish" in res[0].reason
