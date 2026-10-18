# Final whole-branch review: feat/mirror-software (0894f81..eca395e)

Reviewer: final whole-branch seat. Read-only on the checkout. One reviewer, several passes: stat, then the runtime modules
(voice_main, conversation, llm, audio, tts, stt, bus, lights, vad outline, face/app, face/sprite, face/bank, fillers,
persona.system_prompt, config), then deploy/ (setup_orin.sh, fetch_models.sh, all systemd units, xorg conf), the Makefile,
both configs and pyproject. Tests, assets, tools and docs were only spot-checked.

Verification run during review:
- `SDL_VIDEODRIVER=dummy .venv/bin/pytest -q`: 192 passed (23.8 s).
- `tests/test_tts.py` run in a loop on its own: `test_phonikud_alignments_real` failed on run 3 of 3 (see M1). The flake is
  **not order-dependent**. It comes from random variation in the synthesized speech.
- Offline check: `Tokenizer.from_pretrained('dicta-il/dictabert-large-char-menaked')` loaded in 0.25 s with `HF_HOME=models/hf`,
  `HF_HUB_OFFLINE=1` and every HTTP(S) proxy pointed at a dead port. So Phonikud's tokenizer comes from the local cache.
- Voice sample rates: en_US-lessac-medium = 22050 Hz and phonikud-tts-he = 22050 Hz. They match, so the Player does not
  reopen the output when the language changes.
- Installed versions that were tested: piper-tts 1.8.0, phonikud 0.4.1, phonikud-onnx 1.0.6, onnxruntime 1.31.0,
  sounddevice 0.5.6.

## Strengths

- **The answer pipeline shuts down cleanly** (conversation.py:150-185). Both workers are cancelled and joined in a
  `finally`, so even a `_Shutdown` BaseException cannot leave a thread behind. Queues are bounded (`clips` maxsize 1), and
  `_put`/`_get` both check for cancellation. Errors travel to the controller thread as queue items, so no exception is
  lost on a worker thread.
- **Signal handling is sound.** `_Shutdown(BaseException)` passes through every `except Exception` (`_guarded`, FillerBank,
  on_segment), and `Player.play` turns it into `_abort` (cancel the writer, join it, and only then touch the stream). That
  respects PortAudio's lack of thread safety. The `finally` in `run()` puts the lights on idle, closes the face bus,
  joins re-warm with a bound, and closes the output.
- **The voice and face processes are loosely coupled.** Bus messages are fire-and-forget UDP on 127.0.0.1. The receiver
  drops malformed datagrams, viseme defaults to "X", and `FaceBank.mouth_patch` falls back for unknown letters, so either
  process can restart on its own. `_push_idle` runs before models load, so after a voice crash the face is frozen for at
  most about 2 s plus import time.
- **Deafening is solid.** The gate is set on THINKING/SPEAKING before lights I/O. The mic thread discards audio and
  resets both the Segmenter and the Silero state on un-deafen. The main loop drains stale segments, and the Player waits
  out `stream.latency` before going back to LISTENING. Together these close the self-hearing loop.
- **Prompt caching is designed with care.** The time-of-day line goes last (persona.py:129-137), languages are pinned to
  slots, re-warm handles one slot at a time with a bounded wait in `_answer`, and a failed re-warm is retried.
- **The runtime is offline.** The only runtime network calls go to 127.0.0.1 (stt/llm). `HF_HUB_OFFLINE` and `HF_HOME`
  are set before tts is imported. Model downloads exist only in fetch_models.sh, with pinned revisions (Phonikud, Silero
  v5.1.2).
- **Deploy configuration is consistent.** llama flags are identical in the Makefile and mirror-llm.service
  (`-c 8192 -np 2 --cache-reuse 256 -ngl 99`). Ports 8080/8081 match both tomls. `models_dir` is checked by
  setup_orin.sh. ExecStartPre waits for health on both servers, and TimeoutStartSec=200 is longer than the 180 s loop.
  Every unit has `StartLimitIntervalSec=0` + `Restart=always`, so nothing gives up after a burst of failures. Linger,
  autologin on Xorg, DPMS in three places, and journald in RAM are all present.
- Ledger hygiene is good. Most carried items were actually closed: lights to idle on shutdown, Silero pinned, Makefile
  `kill -0` per pid, the lang compare lowercased, partial reply kept, THINKING on stall, cross-sentence stage directions
  (`_unclosed`).

## Issues

### Critical (must fix)

None found.

### Important (should fix)

**I1. Output and mic stream format on a raw ALSA hw device is unverified and likely to fail on the XVF3800.**
audio.py:65-72 (Mic `channels=1`, 16 kHz), audio.py:106-107 (Player `channels=1` at the voice rate, 22050 Hz),
config/orin.toml:6-7 (`input_device`/`output_device = "XVF3800"`).
- **What happens:** `find_device` returns the first PortAudio device whose name contains "XVF3800". On Linux ALSA that is
  the `hw:N,0` device, which neither resamples nor remixes channels.
- **Why it fails on the Orin:** the ReSpeaker's USB interface runs at 16 kHz or 48 kHz and exposes 2 channels. That
  makes an output at 22050 Hz mono (and possibly a mono input) an "Invalid sample rate / number of channels" error.
- **Why it worked on the Mac:** CoreAudio converts both silently.
- **Second risk:** PulseAudio from the GDM autologin session may already hold the card.
- **Effect at the event:** if the output fails, the mirror is silent and its mouth never moves. Every answer logs an error
  and drops to idle. If the input fails, the voice service crash-loops and reloads the models every time.
- **Fix:** when the device is opened, query its supported rate and channels (or open at 48000/16000 stereo). Then
  resample in `Player.play` (np.interp or resample_poly is enough for speech) and take channel 0 from a stereo Mic.
  Alternatively, select the ALSA `plughw`/`default` device. Either way, verify on the device before the event. This does
  not block the merge, but it blocks the event.

**I2. Audio I/O has no timeouts, so the mirror can hang silently with no watchdog.** audio.py:83 and audio.py:175.
- **Mic side (audio.py:83):** `Mic.read()` is a bare `self._q.get()`. If the USB mic drops out or PortAudio stops
  calling back, the audio thread blocks forever. `failed` is never set, so the process stays up with a normal-looking
  idle face and is deaf until someone restarts it.
- **Player side (audio.py:175):** `writer.join()` has no timeout. A `stream.write` that wedges hangs the main thread in
  SPEAKING forever.
- **Why it matters:** these are the failures that hours of unattended running would hit.
- **Fix:**
  - `Mic.read`: `get(timeout=2.0)` and raise. The audio thread already sets `failed`, so run() returns 1 and systemd
    restarts the service.
  - `Player.play`: `writer.join(len(samples)/sr + 2.0)`, then `_abort` and raise if it is still alive.
  - Optionally, `WatchdogSec=` with sd_notify from the main loop.

**I3. The serial write to WLED can block the main thread.** lights.py:71-74; this is the Task 8 deferred minor, promoted.
- **What happens:** `serial.Serial(..., timeout=1)` sets a read timeout only. `write()` and `flush()` have no
  `write_timeout`.
- **When it bites:** if the ESP32 browns out or wedges, which is plausible on a 12 V branch that is shared with the amp,
  `flush()` blocks. It is called from `_set_state` on the controller thread, so the whole conversation freezes.
- **Fix:** add `write_timeout=0.5`; `SerialTimeoutException` then follows the existing close-and-retry path. One line.

**I4. Runtime dependencies are not pinned for the one-shot Orin install.** pyproject.toml:21-24.
- **What happens:** setup_orin.sh installs whatever is newest on install day. `piper-tts>=1.8,<2` is coupled to 1.8.0
  internals (`patch_voice_with_alignment`, `phoneme_ids_to_audio(include_alignments=True)`, `PiperConfig.from_dict`).
  `phonikud` and `phonikud-onnx` have no version at all.
- **Effect:** a release between now and November could break Hebrew speech or the alignments on the device only.
- **Fix:** pin exactly the tested set: piper-tts==1.8.0, phonikud==0.4.1, phonikud-onnx==1.0.6. Ideally also onnxruntime
  and sounddevice. This also closes the Task 13 deferred item.

### Minor (nice to have)

- **M1 (tests/test_tts.py:224):** `test_phonikud_alignments_real` is flaky, but not because of test order. Piper's
  stochastic durations sometimes yield no "D" segment longer than the min-hold, so `{"B","C","D"} <= shapes` fails. The
  observed shapes were `{A,B,C,G,H,X}`. Fix: assert `{"B","C"}` plus "at least 4 distinct shapes". Cheap; do it before
  merge so `make test` stays trustworthy.
- **M2 (voice_main.py:269):** the `segments` queue is unbounded and drained only after answers. Under steady crowd or
  music noise, 15 s segments arrive while the main thread runs STT on each one. If STT ever runs slower than real time, a
  backlog builds and old wake words get answered. A `maxsize` (e.g. 3) that drops the oldest is a cheap guard.
- **M3 (conversation.py:31, 113):** if llama-server restarts (Restart=always), both slots are cold, but the controller
  only re-warms after 600 s idle. Cheap improvement: call `mark_unwarmed(lang)` when `_answer` sees an LLMError, so the
  next idle tick re-warms.
- **M4 (face/app.py:101-105):** the face has no staleness timeout. If voice is SIGKILLed mid-reply, the face keeps
  SPEAKING with the last mouth level until voice restarts. In practice that is bounded at about 2 s, because
  `_push_idle` runs first. Optional fix: drop to IDLE after 3 s with no messages while SPEAKING.
- **M5 (config/mac.toml:2, 13-16 vs config.py:89):** `models_dir` and the tts model paths resolve against the cwd, while
  `world_file` and `bank` resolve against the repo root. Both work today (make runs from the repo root, the Orin uses
  absolute paths plus WorkingDirectory). Document it or make it consistent.
- **M6 (voice_main.py:228-231):** a SIGTERM before the `try` (FaceBus/WledSerial/find_device) prints a traceback. This is
  cosmetic.
- **M7 (audio.py:175-183):** the final `writer.join()` and the tail sleep sit outside the abort `try`. This is the Task 14
  deferred item. The shutdown window is tiny, and the stream would be closed under a live writer only at process exit.
- **M8 (face/app.py:11):** unused `import os`.

## Deferred-minor triage (progress.md)

**Must fix before merge (cheap, and each affects unattended operation or deploy reproducibility):**
1. Task 8: no `write_timeout` on Serial. This is I3.
2. Task 13: pin piper-tts, plus phonikud and phonikud-onnx. This is I4.
3. Task 4: "Mic queue unbounded, status ignored". The real problem is the blocking `Mic.read` (I2). Fix it together with
   the Player join timeout.
4. The final-review flake note: fix the assertion (M1). The cause is nondeterminism, not test order.

**Must do before the event, but not merge-blocking (on-device):**
- Task 12: NovaLCT menu names, `nvpmodel.service` After=, and the CUDA build of whisper v1.9.5 / llama v0.7.0 on JetPack 6.
- Task 13: Orin RAM for the 308 MB G2P together with Gemma at `-c 8192` plus whisper-small CUDA on 8 GB.
- Task 14: the idle OutputStream behaviour on the Orin.
- Task 3: SileroVad ONNX I/O on aarch64. It was verified on the Mac in Task 11.
- I1 (audio format and device selection).
- Task 19: e2e latency unmeasured.
- Hebrew singular/plural quality.

**Already resolved on the branch (can be struck):**
- Task 11: lights to idle on shutdown (`_lights_to_idle`), Silero pinned (v5.1.2), Makefile `kill -0` per pid.
- Task 5: lang compare case (now `.lower()`, accepts he/hebrew/iw).
- Task 18: no THINKING during stall (conversation.py:227-228), partial reply lost (spoken sentences are kept),
  cross-sentence stage directions (`_unclosed`).
- Task 9: push initial IDLE, drain, VAD reset.

**Can wait (cosmetic, test-strength, tools-only, or bounded):**
- Task 1: hardcoded `.venv/bin/pytest`, cwd-relative test_config, `recvfrom(1024)` (messages are about 50 B), and the
  "Sonnet 5.5" trailer on 2 commits. Do not rewrite history for that.
- Task 2: niqqud range and tail tests.
- Task 3: split-remainder test.
- Task 4: `on_level` try/finally. `Player.play` now catches BaseException around the level loop.
- Task 5: `{"text": null}`.
- Task 6: null content, prompt wording, weak assertions.
- Task 7: MMS empty text and dtypes. This only affects the fallback engine.
- Task 8: `_last_open_attempt`, untested paths.
- Task 9: tick lock during sink I/O. Only the main thread calls it, so this is benign.
- Task 10: test strength, 3 px width.
- Task 11: hardcoded wav constants, missing loop unit tests, audio thread not joined (daemon), skipped chunks, torch
  FutureWarning (already filtered).
- Task 13: inline the README license pointer. Do this before sharing the repo, not before merge.
- Task 14: D→A also gets B, sub-frame protected log, M7.
- Task 15: bank validation. It fails loudly at startup, which is acceptable.
- Task 16 and Task 17: tools-only or test-strength items, ECC, per-side eye patches.
- Task 18: joins wait up to 20 s on blocked reads. This is bounded by `servers.timeout_s` and well under systemd's 90 s
  stop timeout.
- Task 19: prompt_ms missing at max_sentences, id_slot honoured.

## Declined to judge

- Persona and prompt content (oracle tone, Hebrew phrasing quality, oracle_world.md facts): these are product decisions,
  and the user owns them.
- Licence of the Phonikud voice: the user ruled on it (non-commercial installation).
- Stub face bank appearance and the real LivePortrait bank: these are art assets, and R9/R13 make the stub an explicit
  placeholder.
- NovaStar/MSD300 mapping of the 252 px face at screen (0,0): hardware configuration outside the code. Task 12 on-device.
- Thermal behaviour of `jetson_clocks` in a desert enclosure: hardware and enclosure design.
- tools/ (stub_bank, build_bank, demo_face) internals: offline authoring tools that never run at the event.
- Logging visitor transcripts at INFO into a RAM-only journal: privacy policy is the user's call. Volatile storage limits
  exposure.

## Assessment

**Ready to merge? With fixes.**

**Reasoning:** The architecture is sound. Threading, shutdown, the bus protocol, config and unit consistency, and the
offline guarantee all hold up, and the 192 tests pass. Before merge, the branch needs four small fixes: the serial
write_timeout, the Mic read and Player join timeouts, exact dependency pins, and the flaky alignment test. Each is a line
or a few, and each blocks a silent hang or a deploy-day breakage. The audio device format (I1) is the biggest open risk
for the event. It must be checked first when the Orin and ReSpeaker arrive, and it will probably need a resample or
device-selection change.
