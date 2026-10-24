#!/usr/bin/env python3
"""One-command bring-up check: what works on this box, and what to fix.

    .venv/bin/python tools/bringup_check.py [--config config/orin.toml]
        [--quick] [--skip-audio] [--json REPORT.json] [--cache DIR]

Runs, in order, each check PASS / WARN / FAIL (or SKIP) with a one-line reason and
a pointer into the README:

 1. config      the config loads; every model file it (and the server units) use
                exists with a plausible size
 2. audio       the configured input/output device matches; Mic/Player negotiate a
                format and run for ~1 s (each in a subprocess, killed after 15 s)
 3. serial      the WLED port exists and takes the idle preset (WARN if kind="log")
 4. display     (Orin) xrandr's current mode is 1920x1080
 5. servers     whisper-server and llama-server answer /health (never started here)
 6. memory      available RAM (free -m + one tegrastats line; vm_stat on the Mac)
 7. end-to-end  an English and a Hebrew wake+question WAV (synthesized with the
                repo's voices, cached in .cache/bringup) through the real VAD -> STT -> wake -> LLM
                (streaming) -> TTS path, with tools/record_demo.py's fake player
 8. hebrew      tools/hebrew_address_check.py, 10 questions: singular-address rate

7 and 8 have total budgets (120 s / 180 s) and stop after 2 LLM errors in a row; every
child process (audio probes, xrandr, tegrastats, free) has a hard deadline.

--quick skips 7-8; --skip-audio skips 2. Exit code 0 when nothing FAILed.
Stdlib + the repo's own dependencies only; Python 3.10-compatible.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import re
import shutil
import shlex
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tools"))

log = logging.getLogger("bringup")

PASS, WARN, FAIL, SKIP = "PASS", "WARN", "FAIL", "SKIP"
_RANK = {SKIP: 0, PASS: 1, WARN: 2, FAIL: 3}

AUDIO_TIMEOUT_S = 15.0
AUDIO_RUN_S = 1.0
MEM_FAIL_MB = 300
MEM_WARN_MB = 500
E2E_PASS_S = 6.0
E2E_WARN_S = 10.0
HEBREW_N = 10
E2E_BUDGET_S = 120.0     # the whole e2e check, warmups included
HEBREW_BUDGET_S = 180.0  # checked between questions; one request may overrun by its timeout
MAX_LLM_ERRORS = 2       # LLM errors in a row before a check gives up on the server
REAP_S = 2.0             # how long to wait for a killed child to go away
DEFAULT_CACHE = os.path.join(REPO, ".cache", "bringup")  # gitignored
HEBREW_WARN_SINGULAR = 2  # more than this many singular replies out of 10: WARN

# README pointers (section names, so they survive edits)
FIX_SETUP = 'README "Orin deploy" (deploy/setup_orin.sh) / "Mac quickstart" (make models)'
FIX_AUDIO = 'README "Orin deploy": "If the mic or speaker isn\'t found" (python -m sounddevice)'
FIX_SERIAL = 'README "Orin deploy": "If the lights don\'t react"; "Frame lights: WLED on the ESP32"'
FIX_DISPLAY = 'README "Bench test checklist" step 2; mirror-face.service locks 1920x1080 (xrandr --output <out> --mode 1920x1080)'
FIX_MEMORY = 'README "Memory on the Orin": make the cuts a-e in order, re-measure after each'
FIX_E2E = 'README "Bench test checklist" step 3; "Measured on a MacBook Air M2" for expected latencies'
FIX_HEBREW = "persona.py's Hebrew prompt (plural address); read the replies: tools/hebrew_address_check.py"

# Plausible minimum sizes (bytes) by file name; anything smaller is a broken download.
MIN_SIZES = {
    "silero_vad.onnx": 1_000_000,
    "en_US-lessac-medium.onnx": 30_000_000,
    "phonikud-tts-he.onnx": 30_000_000,
    "he_IL-oracle-medium.onnx": 30_000_000,
    "phonikud-1.0.int8.onnx": 100_000_000,
    "renikud-plus.int8.onnx": 100_000_000,
    "pocket-tts-english-ipa.onnx": 100_000_000,
    "ivrit-turbo-q5_0.bin": 300_000_000,
    "ggml-small.bin": 200_000_000,
    "gemma-3-4b-it-Q4_K_M.gguf": 1_500_000_000,
    "gemma-3-1b-it-Q4_K_M.gguf": 400_000_000,
}
DEFAULT_MIN_SIZE = 1_000_000   # other .onnx/.bin/.gguf files
JSON_MIN_SIZE = 200            # voice configs


def is_orin() -> bool:
    return os.path.exists("/etc/nv_tegra_release")


def _platform() -> str:
    return sys.platform


# Caveat for every child below: a process stuck in uninterruptible sleep (state D,
# e.g. inside a wedged ALSA/USB driver) ignores SIGKILL until the kernel call returns.
# So after a kill we wait at most REAP_S for it and then walk away (it stays a zombie
# or keeps running); the check reports the hang either way and never blocks on it.
def _run_child(cmd: list, timeout_s: float, env: Optional[dict] = None) -> dict:
    """Run cmd with a hard deadline. {"rc", "stdout", "stderr", "timed_out"}; on a
    timeout the child is killed and whatever it printed so far is returned."""
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             stdin=subprocess.DEVNULL, env=env, cwd=REPO)
    except OSError as e:
        return {"rc": None, "stdout": "", "stderr": str(e), "timed_out": False}
    timed_out = False
    try:
        out, err = p.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        p.kill()
        try:
            out, err = p.communicate(timeout=REAP_S)  # reap; collects the partial output
        except subprocess.TimeoutExpired as e:  # D state: give up on it
            out, err = e.stdout or b"", e.stderr or b""
            for f in (p.stdout, p.stderr):
                try:
                    f.close()
                except Exception:  # noqa: BLE001
                    pass
    return {"rc": p.returncode, "stdout": (out or b"").decode("utf-8", "replace"),
            "stderr": (err or b"").decode("utf-8", "replace"), "timed_out": timed_out}


def _cmd_output(cmd: list, timeout_s: float = 10.0) -> str:
    """stdout of a short command ('' if it fails or hangs)."""
    r = _run_child(cmd, timeout_s)
    return "" if r["timed_out"] else r["stdout"]


def call_with_budget(fn: Callable, budget_s: float):
    """(finished, value, exception) of fn() run on a daemon thread for at most budget_s.
    An unfinished call is abandoned (its thread runs on in the background)."""
    box: dict = {}

    def run():
        try:
            box["value"] = fn()
        except BaseException as e:  # noqa: BLE001 - handed back to the caller
            box["exc"] = e

    t = threading.Thread(target=run, name="budgeted", daemon=True)
    t.start()
    t.join(max(0.0, budget_s))
    if t.is_alive():
        return False, None, None
    return True, box.get("value"), box.get("exc")


# --------------------------------------------------------------------------
# results and the runner
# --------------------------------------------------------------------------
@dataclass
class Result:
    name: str
    status: str
    reason: str
    fix: str = ""
    details: dict = field(default_factory=dict)
    seconds: float = 0.0


@dataclass
class Check:
    name: str
    fn: Callable  # fn(ctx) -> Result | list[Result]
    skip: Optional[str] = None  # set: don't run, report SKIP with this reason


class Context:
    """What the checks share: the config, args and the results so far."""

    def __init__(self, cfg=None, config_path=None, args=None):
        self.cfg = cfg
        self.config_path = config_path
        self.args = args
        self.results: list[Result] = []
        # set when an e2e turn was abandoned at its budget: its daemon thread may still
        # be inside STT/LLM/TTS, so later LLM checks are skipped and main hard-exits
        self.abandoned = threading.Event()

    def status_of(self, name: str) -> Optional[str]:
        for r in self.results:
            if r.name == name:
                return r.status
        return None


def run_checks(checks: list[Check], ctx: Context,
               report: Callable[[Result], None] = lambda r: None) -> list[Result]:
    """Run the checks in order. A check that raises is a FAIL with the exception;
    a skipped check is a SKIP. Each result is reported as soon as it is known."""
    for chk in checks:
        t0 = time.monotonic()
        if chk.skip:
            out = [Result(chk.name, SKIP, chk.skip)]
        else:
            try:
                got = chk.fn(ctx)
                out = got if isinstance(got, list) else [got]
            except Exception as e:  # noqa: BLE001 - one broken check must not stop the rest
                log.debug("check %s crashed", chk.name, exc_info=True)
                out = [Result(chk.name, FAIL, f"check crashed: {type(e).__name__}: {e}",
                              details={"traceback": traceback.format_exc()})]
        dt = round(time.monotonic() - t0, 2)
        for r in out:
            r.seconds = dt
            ctx.results.append(r)
            report(r)
    return ctx.results


def overall(results: list[Result]) -> str:
    """The worst status (SKIP counts as nothing); PASS for no results."""
    worst = max((r.status for r in results), key=_RANK.get, default=PASS)
    return PASS if worst == SKIP else worst


def exit_code(results: list[Result]) -> int:
    return 1 if any(r.status == FAIL for r in results) else 0


def counts(results: list[Result]) -> dict:
    return {s: sum(r.status == s for r in results) for s in (PASS, WARN, FAIL, SKIP)}


def report_doc(results: list[Result], meta: dict) -> dict:
    return {"overall": overall(results), "exit_code": exit_code(results),
            "counts": counts(results), "meta": meta,
            "results": [asdict(r) for r in results]}


def write_json(path: str, results: list[Result], meta: dict) -> None:
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report_doc(results, meta), f, ensure_ascii=False, indent=1, default=str)


def summary_table(results: list[Result], width: int = 72) -> str:
    name_w = max([len(r.name) for r in results] + [5])
    lines = [f"{'check'.ljust(name_w)}  status  reason", "-" * min(width + name_w, 110)]
    for r in results:
        reason = r.reason if len(r.reason) <= width else r.reason[:width - 3] + "..."
        lines.append(f"{r.name.ljust(name_w)}  {r.status.ljust(6)}  {reason}")
    c = counts(results)
    lines.append("-" * min(width + name_w, 110))
    lines.append(f"overall {overall(results)}: {c[PASS]} pass, {c[WARN]} warn, "
                 f"{c[FAIL]} fail, {c[SKIP]} skipped")
    return "\n".join(lines)


def print_result(r: Result) -> None:
    print(f"[{r.status}] {r.name}: {r.reason}", flush=True)
    if r.fix and r.status in (WARN, FAIL):
        print(f"       fix: {r.fix}", flush=True)


# --------------------------------------------------------------------------
# 1. config and model files
# --------------------------------------------------------------------------
def _unit_env(unit_text: str, var: str) -> Optional[str]:
    """VAR's value after systemd reads these unit (and drop-in) lines in order:
    `Environment=` takes several space-separated, optionally quoted, assignments;
    later ones win; an empty `Environment=` resets everything set before it."""
    val = None
    for line in unit_text.splitlines():
        line = line.strip()
        if not line.startswith("Environment="):
            continue
        rest = line[len("Environment="):].strip()
        if not rest:
            val = None
            continue
        try:
            words = shlex.split(rest)
        except ValueError:  # unbalanced quotes: systemd would reject it too
            continue
        for w in words:
            k, sep, v = w.partition("=")
            if sep and k == var:
                val = v
    return val


SERVER_UNITS = (("stt", "mirror-stt", "STT_MODEL"), ("llm", "mirror-llm", "LLM_MODEL"))


def server_model_names(prefer_units: bool, env: Optional[dict] = None,
                       units_dir: Optional[str] = None,
                       repo_units_dir: Optional[str] = None) -> dict:
    """{"stt"|"llm": {"name": model file, "source": where it came from}}.

    The units: the installed user unit (else the repo's copy in deploy/systemd), then
    its drop-ins (*.conf, in name order), read like systemd does. prefer_units (the
    Orin): the units win over $STT_MODEL/$LLM_MODEL in this shell, which systemd
    never sees. Otherwise (the Mac's make servers-mac) the environment wins."""
    env = os.environ if env is None else env
    units_dir = units_dir or os.path.expanduser("~/.config/systemd/user")
    repo_units_dir = repo_units_dir or os.path.join(REPO, "deploy", "systemd")
    out = {}
    for key, unit, var in SERVER_UNITS:
        installed = os.path.join(units_dir, f"{unit}.service")
        base = installed if os.path.exists(installed) else os.path.join(
            repo_units_dir, f"{unit}.service")
        dropin_dir = os.path.join(units_dir, f"{unit}.service.d")
        dropins = sorted(os.path.join(dropin_dir, n) for n in _listdir(dropin_dir)
                         if n.endswith(".conf"))
        name, source, text = None, None, ""
        for path in [base] + dropins:
            try:
                with open(path, encoding="utf-8") as f:
                    text += f.read() + "\n"
            except OSError:
                continue
            got = _unit_env(text, var)
            if got != name:
                name, source = got, path
        from_env = env.get(var)
        if from_env and (not prefer_units or not name):
            name, source = from_env, f"${var}"
        out[key] = {"name": os.path.basename(name or ""), "source": source}
    return out


def model_files(cfg, names: Optional[dict] = None) -> list[tuple[str, str]]:
    """(label, path) of every model file the config and the server units use
    (names: server_model_names(); looked up with the env winning if not given)."""
    t = cfg.tts
    models_dir = cfg.models_dir
    files = [("vad", os.path.join(models_dir, "silero_vad.onnx"))]
    use_pocket = t.engine == "pocket"
    files += [("piper en voice", t.piper_voice_en),
              ("piper en voice config", t.piper_voice_en + ".json")]
    if t.he_engine == "phonikud":
        files += [("phonikud he voice", t.phonikud_voice_he),
                  ("phonikud he voice config", t.phonikud_voice_config_he
                   or t.phonikud_voice_he + ".json"),
                  ("phonikud model", t.phonikud_model)]
    if (t.he_engine == "phonikud" and t.he_g2p == "renikud") or use_pocket:
        files.append(("renikud g2p", t.renikud_model))
    if use_pocket:
        files.append(("pocket model", t.pocket_model))
        if t.pocket_voice_file:
            files.append(("pocket voice file", t.pocket_voice_file))
    if names is None:
        names = server_model_names(prefer_units=False)
    for key, label in (("stt", "whisper model"), ("llm", "llm model")):
        if names.get(key, {}).get("name"):
            files.append((label, os.path.join(models_dir, names[key]["name"])))
    return files


def min_size(path: str) -> int:
    base = os.path.basename(path)
    if base in MIN_SIZES:
        return MIN_SIZES[base]
    if base.endswith(".json"):
        return JSON_MIN_SIZE
    if base.endswith(".npy"):
        return 1000
    return DEFAULT_MIN_SIZE


def check_files(files: list[tuple[str, str]]) -> tuple[list[str], list[str], dict]:
    """(missing, too_small, sizes) for (label, path) pairs."""
    missing, small, sizes = [], [], {}
    for label, path in files:
        try:
            size = os.path.getsize(path)
        except OSError:
            missing.append(f"{label} ({path})")
            continue
        sizes[path] = size
        if size < min_size(path):
            small.append(f"{label} ({path}: {size} B < {min_size(path)} B)")
    return missing, small, sizes


def check_config(ctx: Context) -> Result:
    from mirror.config import load_config, resolve_repo_path

    try:
        ctx.cfg = load_config(ctx.config_path)
    except Exception as e:  # noqa: BLE001
        return Result("config", FAIL, f"{ctx.config_path} does not load: {e}",
                      "fix the TOML (compare config/orin.toml in git)")
    cfg = ctx.cfg
    prefer_units = is_orin() or os.path.basename(ctx.config_path).startswith("orin")
    names = server_model_names(prefer_units)
    files = model_files(cfg, names)
    servers = ", ".join(f"{k} {v['name']} (from {v['source']})" for k, v in names.items()
                        if v["name"])
    missing, small, sizes = check_files(files)
    extra = []
    world = resolve_repo_path(cfg.conversation.world_file, ctx.config_path)
    if not os.path.exists(world):
        extra.append(f"world file {world}")
    bank = resolve_repo_path(cfg.face.bank, ctx.config_path)
    if cfg.face.renderer == "sprite" and not os.path.isdir(bank):
        extra.append(f"face bank {bank}")
    hf = os.path.join(cfg.models_dir, "hf")
    details = {"config": ctx.config_path, "files": dict(files), "sizes": sizes,
               "server_models": names}
    if missing or small or extra:
        bad = missing + small + extra
        return Result("config", FAIL, f"{len(bad)} bad file(s): " + "; ".join(bad),
                      FIX_SETUP, details)
    if not os.path.isdir(hf):
        return Result("config", WARN, f"{len(files)} model files OK, but no HF cache at {hf}",
                      FIX_SETUP + " (fetch_models.sh fills hf/)", details)
    total = sum(sizes.values()) / 1e9
    return Result("config", PASS, f"loaded; {len(files)} model files present ({total:.1f} GB);"
                  f" servers: {servers or 'none found'}", details=details)


# --------------------------------------------------------------------------
# 2. audio (each direction in a subprocess with a hard timeout)
# --------------------------------------------------------------------------
def probe_audio(kind: str, config_path: str, emit: Callable[[dict], None]) -> None:
    """Runs in the child: match the device, open it for ~1 s, report the format.
    Each step emit()s what it learned, so a hang still tells the parent how far it got."""
    import numpy as np
    import sounddevice as sd

    from mirror.audio import Mic, Player, find_device
    from mirror.config import load_config

    cfg = load_config(config_path)
    want = cfg.audio.input_device if kind == "input" else cfg.audio.output_device
    key = f"max_{kind}_channels"
    devices = [{"index": i, "name": d["name"], "channels": d[key]}
               for i, d in enumerate(sd.query_devices()) if d[key] > 0]
    idx = find_device(want, kind, cfg.audio.prefer_alsa_plug)
    name = sd.query_devices(idx if idx is not None else None, kind)["name"]
    matched = idx is not None if want else True
    emit({"kind": kind, "wanted": want, "devices": devices, "index": idx, "name": name,
          "matched": matched})
    if not matched:
        return
    if kind == "input":
        with Mic(idx, cfg.audio.sample_rate) as mic:
            emit({"rate": mic.device_rate, "channels": mic.channels, "stage": "opened"})
            t0, peak = time.monotonic(), 0.0
            while time.monotonic() - t0 < AUDIO_RUN_S:
                peak = max(peak, float(np.abs(mic.read()).max()))
            emit({"reads": mic.reads, "peak": round(peak, 4)})
    else:
        sr = 22050  # Piper's rate
        player = Player(idx)
        try:
            player.open(sr)
            emit({"rate": player._out_rate, "channels": player._channels, "stage": "opened"})
            player.play(np.zeros(int(AUDIO_RUN_S * sr), np.float32), sr, lambda _l: None)
        finally:
            player.close()
    emit({"ok": True, "stage": "done"})


def parse_probe(stdout) -> dict:
    """Merge the child's PROBE lines (later keys win)."""
    if isinstance(stdout, bytes):
        stdout = stdout.decode("utf-8", "replace")
    out: dict = {}
    for line in (stdout or "").splitlines():
        if line.startswith("PROBE "):
            out.update(json.loads(line[6:]))
    return out


def _run_probe(kind: str, config_path: str, timeout_s: float = AUDIO_TIMEOUT_S,
               cmd: Optional[list] = None) -> dict:
    cmd = cmd or [sys.executable, os.path.abspath(__file__), "--config", config_path,
                  "--probe-audio", kind]
    r = _run_child(cmd, timeout_s)
    out = parse_probe(r["stdout"])
    if r["timed_out"]:
        out["error"] = (f"hung after stage {out.get('stage', 'device lookup')!r}: "
                        f"killed after {timeout_s:.0f} s")
    elif not out.get("ok") and "error" not in out and out.get("matched", True):
        tail = (r["stderr"].strip().splitlines() or ["no output"])[-1]
        out["error"] = f"probe exited {r['rc']}: {tail}"
    return out


def audio_result(kind: str, probe: dict) -> Result:
    name = f"audio {kind}"
    if "error" in probe:
        fmt = (f" (negotiated {probe['rate']} Hz x {probe['channels']} ch)"
               if probe.get("rate") else "")
        return Result(name, FAIL, f"{probe.get('name', kind)}: {probe['error']}{fmt}",
                      FIX_AUDIO, probe)
    if not probe.get("matched"):
        names = ", ".join(d["name"] for d in probe.get("devices", [])) or "none"
        return Result(name, FAIL, f"no {kind} device matches {probe.get('wanted')!r} "
                      f"(have: {names})", FIX_AUDIO, probe)
    fmt = f"{probe.get('rate')} Hz x {probe.get('channels')} ch"
    dev = f"[{probe.get('index')}] {probe.get('name')}" if probe.get("index") is not None \
        else f"default ({probe.get('name')})"
    if kind == "input":
        if not probe.get("reads"):
            return Result(name, FAIL, f"{dev} opened at {fmt} but gave no audio", FIX_AUDIO, probe)
        if probe.get("peak", 0) == 0:
            return Result(name, WARN, f"{dev} {fmt}: only digital silence in 1 s (muted?)",
                          FIX_AUDIO, probe)
        return Result(name, PASS, f"{dev} opened at {fmt}, {probe['reads']} chunks "
                      f"(peak {probe['peak']})", details=probe)
    return Result(name, PASS, f"{dev} opened at {fmt}, played 1 s of silence", details=probe)


def check_audio(ctx: Context) -> list[Result]:
    return [audio_result(k, _run_probe(k, ctx.config_path)) for k in ("input", "output")]


# --------------------------------------------------------------------------
# 3. serial (WLED)
# --------------------------------------------------------------------------
def check_serial(ctx: Context) -> Result:
    from mirror.lights import PRESETS
    from mirror.states import State

    lc = ctx.cfg.lights
    if lc.kind != "wled":
        return Result("serial", WARN, f'lights kind is "{lc.kind}": no WLED to test',
                      'set [lights] kind = "wled" + serial_port for the real frame')
    port = lc.serial_port
    if not port or not os.path.exists(port):
        ports = sorted("/dev/" + n for n in _listdir("/dev")
                       if n.startswith(("ttyUSB", "ttyACM")))
        return Result("serial", FAIL, f"{port} does not exist (found: {', '.join(ports) or 'none'})",
                      FIX_SERIAL, {"port": port, "found": ports})
    import serial

    data = b'{"ps":%d}\n' % PRESETS[State.IDLE]
    try:
        with serial.Serial(port, lc.baud, timeout=1, write_timeout=1) as s:
            s.write(data)
            s.flush()
    except Exception as e:  # noqa: BLE001
        hint = " (permission: are you in 'dialout'? reboot after setup_orin.sh)" \
            if "ermission" in str(e) else ""
        return Result("serial", FAIL, f"{port}: {e}{hint}", FIX_SERIAL, {"port": port})
    return Result("serial", PASS, f"{port} @ {lc.baud}: sent idle preset {data.strip().decode()}"
                  " (the frame should breathe pink)", details={"port": port})


def _listdir(d: str) -> list[str]:
    try:
        return os.listdir(d)
    except OSError:
        return []


# --------------------------------------------------------------------------
# 4. display
# --------------------------------------------------------------------------
def parse_xrandr(text: str) -> Optional[tuple[str, int, int]]:
    """(output, width, height) of the first connected output with a current mode."""
    for line in text.splitlines():
        m = re.match(r"(\S+) connected (?:primary )?(\d+)x(\d+)\+\d+\+\d+", line)
        if m:
            return m.group(1), int(m.group(2)), int(m.group(3))
    return None


def check_display(ctx: Context) -> Result:
    if not is_orin():
        return Result("display", SKIP, "not the Orin")
    env = dict(os.environ)
    if not env.get("DISPLAY"):
        xauth = os.path.join(env.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}"),
                             "gdm", "Xauthority")
        if not os.path.exists(xauth):
            return Result("display", SKIP, "DISPLAY not set and no GDM session found",
                          FIX_DISPLAY)
        env.update(DISPLAY=":0", XAUTHORITY=xauth)  # what the face units use
    if shutil.which("xrandr") is None:
        return Result("display", WARN, "xrandr not installed (apt install x11-xserver-utils)",
                      FIX_DISPLAY)
    p = _run_child(["xrandr"], 10.0, env=env)
    if p["timed_out"]:
        return Result("display", WARN, "xrandr hung", FIX_DISPLAY)
    mode = parse_xrandr(p["stdout"])
    if p["rc"] or mode is None:
        why = p["stderr"].strip().splitlines()[-1] if p["stderr"].strip() else "no connected output"
        return Result("display", WARN, f"xrandr on {env['DISPLAY']}: {why}", FIX_DISPLAY)
    out, w, h = mode
    det = {"output": out, "width": w, "height": h, "display": env["DISPLAY"]}
    if (w, h) != (ctx.cfg.face.screen_w, ctx.cfg.face.screen_h):
        return Result("display", WARN, f"{out} is {w}x{h}, want "
                      f"{ctx.cfg.face.screen_w}x{ctx.cfg.face.screen_h}", FIX_DISPLAY, det)
    return Result("display", PASS, f"{out} at {w}x{h}", details=det)


# --------------------------------------------------------------------------
# 5. servers
# --------------------------------------------------------------------------
def server_fix() -> str:
    if is_orin():
        return ("systemctl --user start mirror-stt mirror-llm (logs: journalctl --user "
                "-u mirror-stt -u mirror-llm); README \"Day-to-day commands on the Orin\"")
    return 'make servers-mac (README "Mac quickstart")'


def health(url: str, timeout_s: float = 3.0) -> tuple[bool, str]:
    import requests

    try:
        r = requests.get(url.rstrip("/") + "/health", timeout=timeout_s)
    except requests.exceptions.RequestException as e:
        return False, f"unreachable ({type(e).__name__})"
    body = r.text.strip()[:80]
    if r.status_code == 200:
        return True, f"healthy ({body})"
    return False, f"HTTP {r.status_code} {body} (still loading?)"


def check_servers(ctx: Context) -> list[Result]:
    out = []
    for name, label, url in (("stt server", "whisper-server", ctx.cfg.servers.stt_url),
                             ("llm server", "llama-server", ctx.cfg.servers.llm_url)):
        ok, why = health(url)
        out.append(Result(name, PASS if ok else FAIL, f"{label} {url}: {why}",
                          "" if ok else server_fix(), {"url": url}))
    return out


# --------------------------------------------------------------------------
# 6. memory
# --------------------------------------------------------------------------
def memory_status(available_mb: float) -> str:
    if available_mb < MEM_FAIL_MB:
        return FAIL
    if available_mb < MEM_WARN_MB:
        return WARN
    return PASS


def parse_free(text: str) -> Optional[dict]:
    """free -m: {"total", "used", "available"} in MB from the Mem: line."""
    lines = text.splitlines()
    if not lines:
        return None
    head = lines[0].split()
    for line in lines[1:]:
        if line.startswith("Mem:"):
            vals = line.split()[1:]
            row = dict(zip(head, vals))
            try:
                return {"total": int(row["total"]), "used": int(row["used"]),
                        "available": int(row.get("available", row.get("free")))}
            except (KeyError, ValueError, TypeError):
                return None
    return None


def parse_tegrastats(line: str) -> Optional[tuple[int, int]]:
    m = re.search(r"RAM (\d+)/(\d+)MB", line)
    return (int(m.group(1)), int(m.group(2))) if m else None


def parse_vm_stat(text: str) -> Optional[dict]:
    """vm_stat: {"page", "free", "inactive", "speculative", "purgeable"} (pages)."""
    m = re.search(r"page size of (\d+) bytes", text)
    if not m:
        return None
    out = {"page": int(m.group(1))}
    for key, label in (("free", "Pages free"), ("inactive", "Pages inactive"),
                       ("speculative", "Pages speculative"), ("purgeable", "Pages purgeable")):
        mm = re.search(rf"{label}:\s+(\d+)", text)
        out[key] = int(mm.group(1)) if mm else 0
    return out


def mac_available_mb(vm: dict) -> float:
    pages = vm["free"] + vm["inactive"] + vm["speculative"] + vm["purgeable"]
    return pages * vm["page"] / 1048576


def _tegrastats_line(timeout_s: float = 4.0, cmd: Optional[list] = None) -> Optional[str]:
    """tegrastats' first line, or None if it prints nothing within timeout_s (it can
    sit silent without root). The child is always killed and reaped (bounded, see
    the D-state caveat at _run_child)."""
    if cmd is None:
        exe = shutil.which("tegrastats")
        if exe is None:
            return None
        cmd = [exe, "--interval", "1000"]
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                             stdin=subprocess.DEVNULL, text=True)
    except OSError:
        return None
    lines: list = []
    reader = threading.Thread(target=lambda: lines.append(p.stdout.readline()),
                              name="tegrastats", daemon=True)
    reader.start()
    try:
        reader.join(timeout_s)  # the hard deadline: readline itself never times out
    finally:
        p.kill()
        try:
            p.wait(REAP_S)
        except subprocess.TimeoutExpired:
            log.warning("tegrastats did not exit after SIGKILL")
    line = lines[0].strip() if lines and lines[0] else ""
    return line or None


def memory_result(available_mb: float, source: str, servers_up: bool, details: dict) -> Result:
    st = memory_status(available_mb)
    note = "" if servers_up else " (servers down: re-measure with everything running)"
    reason = (f"{available_mb:.0f} MB available ({source}); target >= {MEM_WARN_MB} MB, "
              f"fail < {MEM_FAIL_MB}{note}")
    return Result("memory", st, reason, FIX_MEMORY if st != PASS or not servers_up else "",
                  dict(details, available_mb=round(available_mb)))


def check_memory(ctx: Context) -> Result:
    servers_up = (ctx.status_of("stt server") == PASS and ctx.status_of("llm server") == PASS)
    if _platform() == "darwin":
        vm = _cmd_output(["vm_stat"])
        parsed = parse_vm_stat(vm)
        if parsed is None:
            return Result("memory", WARN, "could not parse vm_stat")
        swap = _cmd_output(["sysctl", "-n", "vm.swapusage"]).strip()
        r = memory_result(mac_available_mb(parsed), "vm_stat free+inactive+spec+purgeable",
                          servers_up, {"vm_stat": parsed, "swap": swap})
        m = re.search(r"used = ([\d.]+)M", swap)
        if m:
            r.reason += f"; swap used {float(m.group(1)):.0f} MB"
        return r
    if shutil.which("free") is None:
        return Result("memory", WARN, "free(1) not found")
    free = parse_free(_cmd_output(["free", "-m"]))
    if free is None:
        return Result("memory", WARN, "could not parse free -m")
    details = {"free": free}
    source = "free -m"
    tl = _tegrastats_line() if is_orin() else None
    if tl:
        ram = parse_tegrastats(tl)
        details["tegrastats"] = tl
        if ram:
            source += f"; tegrastats RAM {ram[0]}/{ram[1]}MB"
    return memory_result(free["available"], source, servers_up, details)


# --------------------------------------------------------------------------
# 7. end to end (record_demo's pipeline pieces)
# --------------------------------------------------------------------------
E2E_QUESTIONS = ("en_wake", "he_wake")  # keys of record_demo.QUESTIONS


def wav_cache_key(text: str, tts_cfg) -> str:
    t = tts_cfg
    parts = [text, t.engine, t.piper_voice_en, t.he_engine, t.he_g2p, t.phonikud_voice_he,
             t.pocket_voice, str(t.pocket_voice_file)]
    return hashlib.sha1("\x00".join(parts).encode()).hexdigest()[:12]


def question_wavs(cfg, cache_dir: str, tts=None) -> dict:
    """{key: (lang, text, wav path)}, synthesized with the repo's voices once and
    cached by text + voice settings. tts: a loaded router (else loaded on demand)."""
    import numpy as np
    import record_demo as rd

    os.makedirs(cache_dir, exist_ok=True)
    out = {}
    for key in E2E_QUESTIONS:
        lang, text = rd.QUESTIONS[key]
        path = os.path.join(cache_dir, f"{key}-{wav_cache_key(text, cfg.tts)}.wav")
        if not os.path.exists(path):
            if tts is None:
                from mirror.voice_main import load_tts
                tts = load_tts(cfg.tts)[0]
            samples, sr, _ = tts.synth_visemes(text, lang)
            pad = np.zeros(int(0.2 * sr), np.float32)
            rd.write_wav(path + ".part", np.concatenate([pad, samples, pad]), sr)
            os.replace(path + ".part", path)
        out[key] = (lang, text, path)
    return out


def latency_status(wake: bool, first_s: Optional[float]) -> str:
    if not wake or first_s is None:
        return FAIL
    if first_s <= E2E_PASS_S:
        return PASS
    if first_s <= E2E_WARN_S:
        return WARN
    return FAIL


def turn_result(lang: str, turn: dict) -> Result:
    name = f"e2e {'English' if lang == 'en' else 'Hebrew'}"
    heard = turn.get("heard")
    wake = bool(turn.get("wake"))
    first = turn.get("first_sentence_s")
    st = latency_status(wake, first)
    if heard is None:
        reason = "no utterance reached STT (VAD found nothing, or STT failed)"
    elif not wake:
        reason = f"no wake match in {heard!r}"
    elif first is None:
        reason = f"heard {heard!r}; the LLM gave no sentence (fallback spoken)"
    else:
        filler = turn.get("filler_s")
        reason = (f"wake ok, lang {turn.get('lang')}; filler "
                  f"{'n/a' if filler is None else f'{filler:.2f} s'}, first sentence "
                  f"{first:.2f} s (pass <= {E2E_PASS_S:g}, warn <= {E2E_WARN_S:g})")
        if turn.get("lang") != lang:
            st = max(st, WARN, key=_RANK.get)
            reason += f"; detected {turn.get('lang')}, expected {lang}"
    return Result(name, st, reason, FIX_E2E if st != PASS else "", turn)


def run_turn(controller, segmenter, vad, tl, player, stt, llm, wav: str, sr16: int) -> dict:
    """One fresh conversation: the WAV through VAD then controller.on_segment, timed on
    record_demo's Timeline (the player never sleeps; real waits become timeline gaps)."""
    import record_demo as rd
    from mirror.wake import match_wake

    audio, sr = rd.read_wav(wav)
    _start, seg = rd.feed_question(tl, segmenter, vad, audio, sr, sr16)
    turn: dict = {"wav": wav}
    if seg is None:
        return turn
    speech_end = tl.cursor - segmenter.min_silence_s  # as voice_main measures on the mic
    n_stt, n_llm, n_clips = len(stt.calls), len(llm.calls), len(player.clips)
    tl.anchor()
    controller.on_segment(seg)
    tl.release()
    if len(stt.calls) > n_stt:
        c = stt.calls[-1]
        turn.update(heard=c["text"], lang=c["lang"], stt_s=c["seconds"],
                    wake=match_wake(c["text"]) is not None)
    if len(llm.calls) > n_llm:
        c = llm.calls[-1]
        turn.update(llm_first_sentence_s=c["first_sentence_s"], llm_total_s=c.get("total_s"))
    clips = player.clips[n_clips:]
    filler = next((c for c in clips if c["kind"] == "filler"), None)
    reply = [c for c in clips if c["kind"] == "reply"]
    turn["filler_s"] = round(filler["start"] - speech_end, 3) if filler else None
    got_sentence = turn.get("llm_first_sentence_s") is not None
    turn["llm_error"] = len(llm.calls) > n_llm and not got_sentence
    turn["first_sentence_s"] = (round(reply[0]["start"] - speech_end, 3)
                                if reply and got_sentence else None)
    turn["reply"] = " ".join(c["text"] for c in reply)
    return turn


class GuardedSTT:
    """STT wrapper that refuses new requests once `abandoned` is set."""

    def __init__(self, stt, abandoned: threading.Event):
        self._stt, self._abandoned = stt, abandoned

    def __getattr__(self, name):  # .calls etc. of the wrapped TimedSTT
        return getattr(self._stt, name)

    def transcribe(self, samples, sr: int = 16000, language=None):
        from mirror.stt import STTError

        if self._abandoned.is_set():
            raise STTError("bring-up turn abandoned")
        return self._stt.transcribe(samples, sr, language=language)


class GuardedLLM:
    """LLM wrapper that refuses new requests, and stops a stream, once `abandoned`
    is set (the stream's HTTP response is closed by the controller's worker)."""

    def __init__(self, llm, abandoned: threading.Event):
        self._llm, self._abandoned = llm, abandoned

    def __getattr__(self, name):
        return getattr(self._llm, name)

    def _check(self):
        from mirror.llm import LLMError

        if self._abandoned.is_set():
            raise LLMError("bring-up turn abandoned")

    def warm(self, lang):
        self._check()
        return self._llm.warm(lang)

    def reply_stream(self, history, user_text, lang, cancel=None, angle=None):
        self._check()
        for sentence in self._llm.reply_stream(history, user_text, lang, cancel=cancel,
                                               angle=angle):
            self._check()
            yield sentence


class GuardedTTS:
    """TTS wrapper that refuses to synthesize once `abandoned` is set."""

    def __init__(self, tts, abandoned: threading.Event):
        self._tts, self._abandoned = tts, abandoned

    def __getattr__(self, name):
        return getattr(self._tts, name)

    def _check(self):
        if self._abandoned.is_set():
            raise RuntimeError("bring-up turn abandoned")

    def synth_visemes(self, text, lang):
        self._check()
        return self._tts.synth_visemes(text, lang)

    def synth(self, text, lang):
        self._check()
        return self._tts.synth(text, lang)

    def warmup(self):
        self._check()
        self._tts.warmup()


def make_turn_fn(run: Callable, wav: str, abandoned: threading.Event, show=print) -> Callable:
    """A run_turns fn for run(wav): does nothing once abandoned, and never prints
    after it (an abandoned turn that finishes late stays silent)."""
    def fn():
        if abandoned.is_set():
            return {}
        turn = run(wav)
        if not abandoned.is_set():
            show(f"       heard: {turn.get('heard')!r}\n       reply: {turn.get('reply')!r}")
        return turn
    return fn


def warmup_failure(names, done: bool, unwarmed, exc) -> Optional[list]:
    """FAIL results for every e2e turn if the LLM warmup didn't finish, raised, or
    failed for MAX_LLM_ERRORS languages; None to go ahead."""
    if not done:
        why = "the llama-server warmup did not finish within the e2e budget"
    elif exc is not None:
        why = f"warmup crashed: {type(exc).__name__}: {exc}"
    elif len(unwarmed or ()) >= MAX_LLM_ERRORS:
        why = f"{len(unwarmed)} LLM warmup errors in a row (llama-server stalled?)"
    else:
        return None
    return [Result(n, FAIL, f"not run: {why}", server_fix()) for n in names]


def run_turns(specs: list, budget_s: float, max_llm_errors: int = MAX_LLM_ERRORS,
              clock: Callable[[], float] = time.monotonic,
              abandoned: Optional[threading.Event] = None) -> list[Result]:
    """specs: (name, lang, fn) with fn() -> a run_turn dict. Each turn runs on a
    budgeted thread inside one shared budget; a stalled turn is abandoned (and
    `abandoned` set, so its guarded components stop), and every later one FAILs as
    not run, and so do the rest after max_llm_errors LLM errors in a row."""
    abandoned = abandoned if abandoned is not None else threading.Event()
    deadline = clock() + budget_s
    results: list[Result] = []
    stop_reason = None
    llm_errors = 0
    for name, lang, fn in specs:
        if stop_reason is None and clock() >= deadline:
            stop_reason = f"the e2e budget of {budget_s:.0f} s is used up"
        if stop_reason is not None:
            results.append(Result(name, FAIL, f"not run: {stop_reason}", server_fix()))
            continue
        done, turn, exc = call_with_budget(fn, deadline - clock())
        if not done:
            abandoned.set()
            stop_reason = f"the e2e budget of {budget_s:.0f} s is used up"
            results.append(Result(name, FAIL, f"no answer within the e2e budget of "
                                  f"{budget_s:.0f} s (a stalled server?); abandoned",
                                  server_fix()))
            continue
        if exc is not None:
            results.append(Result(name, FAIL, f"turn crashed: {type(exc).__name__}: {exc}",
                                  FIX_E2E))
            continue
        r = turn_result(lang, turn)
        r.name = name
        results.append(r)
        llm_errors = llm_errors + 1 if turn.get("llm_error") else 0
        if llm_errors >= max_llm_errors:
            stop_reason = f"{llm_errors} LLM errors in a row (llama-server stalled?)"
    return results


def voice_result(wanted: str, got: str) -> Optional[Result]:
    """A WARN when the config asks for Pocket (the oracle's cloned voice) but
    load_tts fell back to Piper; None when the engine is the one asked for."""
    if wanted == got:
        return None
    return Result("voice", WARN,
                  f"tts.engine is {wanted!r} but the mirror speaks with {got!r}: the "
                  "cloned voice failed to load (journalctl: 'falling back to piper')",
                  "copy the voice file (README: Pocket TTS) or check its path in [tts]")


def check_e2e(ctx: Context) -> list[Result]:
    names = ("e2e English", "e2e Hebrew")
    if ctx.status_of("stt server") != PASS or ctx.status_of("llm server") != PASS:
        return [Result(n, FAIL, "servers are not up (see the servers check)", server_fix())
                for n in names]
    import record_demo as rd
    from mirror.audio import CHUNK
    from mirror.conversation import Controller
    from mirror.fillers import FillerBank
    from mirror.persona import FILLERS
    from mirror.stt import WhisperServerSTT
    from mirror.vad import Segmenter, SileroVad
    from mirror.voice_main import FILLER_BUDGET_S, load_tts, warm_llm
    from hebrew_address_check import build_llm

    cfg = ctx.cfg
    rd._setup_env(cfg)
    router, _, engine = load_tts(cfg.tts)
    fallback = voice_result(cfg.tts.engine, engine)
    wavs = question_wavs(cfg, ctx.args.cache, router)
    texts: dict = {}
    tl = rd.Timeline(cap_s=1e9)  # no capping: we want the real waits
    player = rd.TimelinePlayer(tl, texts)
    face = rd.TimelineFace(tl)
    vad = SileroVad(os.path.join(os.path.abspath(cfg.models_dir), "silero_vad.onnx"))
    segmenter = Segmenter(vad, frame_samples=CHUNK, sample_rate=cfg.audio.sample_rate)
    stop = ctx.abandoned
    whisper = WhisperServerSTT(cfg.servers.stt_url, cfg.servers.timeout_s,
                               cfg.servers.stt_language)
    stt = GuardedSTT(rd.TimedSTT(whisper), stop)
    llm = GuardedLLM(rd.TimedLLM(build_llm(cfg, ctx.config_path)), stop)
    text_tts = rd.TextTTS(router, texts, kind="filler")
    tts = GuardedTTS(text_tts, stop)
    fillers = FillerBank(tts, FILLERS,
                         max_clip_s=FILLER_BUDGET_S if engine == "pocket" else None)
    fillers.prepare()
    text_tts.kind = "reply"  # on the wrapped TextTTS, not on the guard
    t0 = time.monotonic()  # the budget covers the warmups and both turns
    done, unwarmed, exc = call_with_budget(lambda: warm_llm(llm), E2E_BUDGET_S)
    if not done:
        stop.set()  # the warmup thread may still be talking to llama-server
    failed = warmup_failure(names, done, unwarmed, exc)
    if failed:
        return failed

    def run(wav):
        controller = Controller(stt, llm, tts, player, rd._NullLights(), face,
                                cfg.conversation, clock=tl.now, fillers=fillers,
                                first_sentence_timeout_s=cfg.servers.timeout_s)
        for u in unwarmed:
            controller.mark_unwarmed(u)
        try:
            return run_turn(controller, segmenter, vad, tl, player, stt, llm, wav,
                            cfg.audio.sample_rate)
        finally:
            controller.close()

    def show(text):
        print(text, flush=True)

    specs = [(name, wavs[key][0], make_turn_fn(run, wavs[key][2], stop, show))
             for key, name in zip(E2E_QUESTIONS, names)]
    results = run_turns(specs, E2E_BUDGET_S - (time.monotonic() - t0), abandoned=stop)
    return results + ([fallback] if fallback else [])


# --------------------------------------------------------------------------
# 8. Hebrew address
# --------------------------------------------------------------------------
def hebrew_result(res: dict) -> Result:
    s, a, f = res["singular"], res["answered"], res["failed"]
    det = {k: res[k] for k in ("n", "singular", "answered", "failed", "replies") if k in res}
    aborted = res.get("aborted")
    if a == 0:
        why = f"; stopped: {aborted}" if aborted else ""
        return Result("hebrew address", FAIL, f"all {f} requests failed{why}",
                      server_fix(), det)
    st = WARN if s > HEBREW_WARN_SINGULAR else PASS
    reason = f"singular address in {s}/{a} replies (warn > {HEBREW_WARN_SINGULAR})"
    if f:
        reason += f"; {f} request(s) failed"
        st = max(st, WARN, key=_RANK.get)
    if aborted:
        reason += f"; stopped after {a + f}/{res['n']}: {aborted}"
        st = max(st, WARN, key=_RANK.get)
    return Result("hebrew address", st, reason, FIX_HEBREW if st != PASS else "", det)


def check_hebrew(ctx: Context) -> Result:
    if ctx.status_of("llm server") != PASS:
        return Result("hebrew address", FAIL, "llama-server is not up", server_fix())
    if ctx.abandoned.is_set():
        return Result("hebrew address", WARN,
                      "skipped: an end-to-end turn was abandoned (LLM busy)", server_fix())
    from hebrew_address_check import build_llm, run_check

    res = run_check(build_llm(ctx.cfg, ctx.config_path), HEBREW_N, show=lambda _s: None,
                    budget_s=HEBREW_BUDGET_S, max_consecutive_errors=MAX_LLM_ERRORS)
    return hebrew_result(res)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def finish(code: int, abandoned: threading.Event, _exit: Callable = os._exit) -> int:
    """The exit code; if a budgeted thread was abandoned (it may sit in native code:
    an HTTP read, onnxruntime), flush and leave with os._exit instead of a normal
    interpreter shutdown, which could hang on or abort in that thread."""
    if abandoned.is_set():
        sys.stdout.flush()
        sys.stderr.flush()
        _exit(code)
    return code


def build_checks(args) -> list[Check]:
    quick = "--quick" if args.quick else None
    return [
        Check("config", check_config),
        Check("audio", check_audio, "--skip-audio" if args.skip_audio else None),
        Check("serial", check_serial),
        Check("display", check_display),
        Check("servers", check_servers),
        Check("memory", check_memory),
        Check("e2e", check_e2e, quick),
        Check("hebrew address", check_hebrew, quick),
    ]


def main(argv=None) -> int:
    default_cfg = os.path.join(REPO, "config", "orin.toml" if is_orin() else "mac.toml")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=default_cfg)
    ap.add_argument("--quick", action="store_true", help="skip the end-to-end and Hebrew checks")
    ap.add_argument("--skip-audio", action="store_true", help="skip the audio device check")
    ap.add_argument("--json", metavar="PATH", help="write a JSON report here")
    ap.add_argument("--cache", default=DEFAULT_CACHE,
                    help="where the synthesized question WAVs are cached "
                         "(default: the repo's .cache/bringup, gitignored)")
    ap.add_argument("-v", "--verbose", action="store_true", help="log the pipeline")
    ap.add_argument("--probe-audio", choices=("input", "output"), help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    args.config = os.path.abspath(args.config)
    os.chdir(REPO)  # config paths are relative to the repo, as under systemd
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(name)s %(message)s")

    if args.probe_audio:  # the child of the audio check
        def emit(d):
            print("PROBE " + json.dumps(d, default=str), flush=True)

        try:
            probe_audio(args.probe_audio, args.config, emit)
        except Exception as e:  # noqa: BLE001
            emit({"error": f"{type(e).__name__}: {e}"})
        return 0

    t0 = time.monotonic()
    ctx = Context(config_path=args.config, args=args)
    checks = build_checks(args)
    print(f"Mirror, Mirror bring-up check: {args.config} on {platform.node()} "
          f"({'Orin' if is_orin() else sys.platform})\n", flush=True)
    # without a config nothing else can run
    results = run_checks(checks[:1], ctx, print_result)
    if ctx.cfg is not None:
        results = run_checks(checks[1:], ctx, print_result)
    print("\n" + summary_table(results))
    meta = {"config": args.config, "host": platform.node(), "orin": is_orin(),
            "platform": sys.platform, "python": platform.python_version(),
            "quick": args.quick, "skip_audio": args.skip_audio,
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "seconds": round(time.monotonic() - t0, 1)}
    if args.json:
        write_json(args.json, results, meta)
        print(f"report: {args.json}")
    return finish(exit_code(results), ctx.abandoned)


if __name__ == "__main__":
    sys.exit(main())
