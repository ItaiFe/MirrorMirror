# SDD ledger — plan: docs/superpowers/plans/2026-10-17-mirror-software.md
Spec: MIRROR_CONTEXT.md. Branch: feat/mirror-software (from main 0894f81).

## Pre-flight scan
| Pair / task | Produces vs consumes | Finding |
|---|---|---|
| T1↔T9 | FaceBus.send(state, mouth); Config.conversation | T9 names `ConversationConfig`; T1 doesn't name section classes → ruling R2 |
| T4↔T9 | Player.play(samples, sr, on_level) | consistent; T9 must pass samples+sr from tts.synth |
| T5↔T9 | Transcript(text, lang), STTError | consistent |
| T6↔T9 | LlamaServerLLM.reply(history, user_text, lang), LLMError, FALLBACK | consistent |
| T7↔T9 | TTSRouter.synth(text, lang)->(np.ndarray, sr) | consistent |
| T8↔T9 | Lights.set_state(State) | consistent |
| T2↔T9 | match_wake -> WakeMatch(remainder) normalized text; is_hallucination | remainder is normalized (lowercase, no punctuation) — acceptable for LLM input |
| T3↔T11 | Segmenter(is_speech,...).feed; SileroVad | consistent |
| T5↔T11 | whisper-server language field pinned in T11 | T5 must use a fixture constant easy to change |
| T7↔T1 | `tts` extra (torch) | tests must not need torch → ruling R3 |
| T1 self | .gitignore list | missing `.superpowers/` → ruling R1 |
| T2 self | tests vs matching rule | consistent ("Mirrors" ratio .92; "מרא" in variant set) |
| T3 self | tests vs Segmenter params | consistent |
| T4 self | test patches sleep; impl uses monotonic clock | implementer must make the clock patchable → note in dispatch |
| T6 self | trim tests vs split regex | consistent |
| T9 self | busy flag vs drop-during-speaking test | consistent |
| T10 self | regions vs 252 layout | consistent |
| T11 self | make dev / servers-mac | consistent |
| T12 self | on-device steps | user ruling: implement files, skip on-device verification |

Ruling R1: T1 .gitignore also ignores `.superpowers/` — SDD workspace must stay untracked — cost if wrong: none.
Ruling R2: T1 names section dataclasses AudioConfig, ServersConfig, TtsConfig, FaceConfig, LightsConfig, ConversationConfig — T9 consumes ConversationConfig — cost if wrong: rename.
Ruling R3: heavy deps (piper, transformers, torch, onnxruntime, sounddevice, pyserial) imported lazily inside the classes that need them, so `make test` runs with only core deps — cost if wrong: slightly later import errors at runtime.
Ruling R4: Python 3.11 venv via uv on Mac; code stays 3.10-compatible (tomllib with tomli fallback) for JetPack 6 — cost if wrong: minor syntax fixes.
Ruling R5: work on branch feat/mirror-software in place rather than a separate worktree; main untouched until finish — cost if wrong: none.

## Progress
Task 1: minor (deferred): Makefile test hardcodes .venv/bin/pytest
Task 1: minor (deferred): test_config paths relative to cwd
Task 1: minor (deferred): bus recvfrom(1024) silent truncation
Task 1: minor (deferred): commit trailer says Sonnet 5.5 not Opus 5.5
Task 1: complete (commits 0894f81..042cc2a, review clean)
Task 2: minor (deferred): niqqud range strips maqaf/paseq/sof pasuq (joins words) — plan-mandated range
Task 2: minor (deferred): no test for "על הקיר" tail or fuzzy tail
Task 2: complete (commits 042cc2a..7e819b2, review clean)
Task 3: minor (deferred): max-length split remainder untested; chunk test compares counts only
Task 3: minor (deferred): Segmenter.reset() doesn't reset SileroVad state — Task 11 wiring must handle (carried into T11 dispatch)
Task 3: note: SileroVad ONNX I/O unverified (64-sample context, sr 0-d int64) — verify in Task 11 manual run
Task 3: complete (commits 7e819b2..a574f13, review clean)
Task 4: minor (deferred): Mic queue unbounded, status ignored — stale audio after playback; Task 11 should drain mic queue after answers (carried)
Task 4: minor (deferred): Player.play no try/finally if on_level raises
Task 4: complete (commits a574f13..ab93c2c, review clean)
Task 5: minor (deferred): lang compare case-sensitive ("he-IL"/"HE" → en); {"text": null} → "None"
Task 5: carried to T11: pin whisper-server language field name in a test fixture
Task 5: complete (commits ab93c2c..e88f17f, review clean)
Task 6: ⚠️ resolved: unknown lang KeyError in reply() — callers pass Transcript.lang which STT (T5) normalizes to he/en; FALLBACK lookup in T9 uses same lang
Task 6: minor (deferred): null content → "None"; SYSTEM_PROMPT[lang] not defensive; weak long-sentence assertion; HTTPError path untested; Hebrew prompt "את המראה" ambiguous (suggest "את היא המראה")
Task 6: complete (commits e88f17f..083d3a8, review clean)
Task 7: ⚠️ resolved: piper voice download/config is Task 11 fetch_models + config (already in plan)
Task 7: carried to T11/T12: set HF_HUB_OFFLINE=1 (and HF_HOME to models/hf) at runtime so MMS loads offline; verify real Piper/MMS APIs in T11 manual run
Task 7: minor (deferred): empty text to MMS; non-int16 int dtypes; warmup test only counts calls
Task 7: complete (commits 083d3a8..960ec0d, review clean)
Task 8: ⚠️ resolved: State is str Enum with lowercase values (T1)
Task 8: minor (deferred): no write_timeout on Serial (can block on wedged USB); _last_open_attempt not reset after successful open; ImportError/reopen paths untested
Task 8: complete (commits 960ec0d..3490890, review clean)
Task 9: fix round 1/5 (2 addressed, 0 open — unguarded face sink escaping on_segment/tick; tts-failure test strength; commits 196558d..8b96c97)
Task 9: minor (deferred): tick() holds lock during sink I/O; _busy reset outside lock (benign)
Task 9: carried to T11: push initial IDLE at startup; drain mic/segment queue after each answer; reset Segmenter+SileroVad after playback
Task 9: complete (commits 3490890..8b96c97, review clean after 1 fix round)
Task 10: fix round 1/5 (2 addressed, 0 open — outline notches; corner + scale tests; commits e66606c..ee15e5f)
Task 10: minor (deferred): test_eye_corners_lit passes pre-fix too (not discriminating); test_scales per-coordinate tolerance loose; 3px min width below 252
Task 10: complete (commits 8b96c97..ee15e5f, review clean after 1 fix round)
Ruling R6: Task 11 adds a `--wav PATH` option to mirror-voice (feed a WAV file instead of the mic, then exit) so the pipeline can be smoke-tested end-to-end without a human speaking — the plan's manual step needs a person at the mic; cost if wrong: one small extra CLI flag.
Ruling R7: Gemma GGUF fetched from an ungated mirror (ggml-org/gemma-3-4b-it-GGUF, fallback unsloth) since google/ repos are license-gated — cost if wrong: swap URL.
Task 11: implementer DONE_WITH_CONCERNS (1011301): whisper-server returns "language":"hebrew" → stt.py now accepts he/hebrew/iw; latency 6.5 s EN / 13–17 s HE on 8 GB Mac (MMS CPU + whole-reply synthesis); Hebrew wake misheard without niqqud; gemma-3-1b left in models/. Review dispatched.
Ruling R8: user requested Hebrew voice switch to Phonikud+Piper (Task 13, appended to plan); MMS kept as configurable fallback — Phonikud checkpoints are CC non-commercial, acceptable for a non-commercial art installation — cost if wrong: license swap needed for commercial use.
Task 11: minor (deferred): Makefile kill -0 multi-pid misses single-server death; signals during startup/teardown give tracebacks; shutdown doesn't push IDLE/off to lights (carry to T12); Silero URL unpinned (pin v5 tag); --wav tail & latency constants hard-coded; no unit tests for audio_loop/AnswerGate/_drain/WavSource; audio thread never joined; skip 1–2 chunks after un-deafen; torch FutureWarning in test output
Task 11: complete (commits ee15e5f..1011301, review clean)
Ruling R9: user chose sprite-bank face; Tasks 14–17 appended (visemes, bank format, compositor, tools+stub+guide); order 13→14→15→17→16→12; stub bank from the reference image keeps the app working until real LivePortrait assets exist — cost if wrong: stub look is crude until assets are made.
Ruling R10: Phonikud voice checkpoints are CC BY-NC 4.0 + academic/educational-only restriction; kept as default for development, MMS fallback one config line away; user must decide (ask creators or revert) before the event — cost if wrong: switch he_engine to mms.
Task 13: minor (deferred): pin piper-tts range (coupled to 1.8.0 internals); README license pointer goes to gitignored report — inline it; no fast stub tests for PhonikudHebrewTTS.phonemes/synth; Orin RAM for 308 MB G2P unmeasured
Task 13: note: Phonikud voice alignments available (sum exactly to audio length) — use in Task 14
Task 13: complete (commits 1011301..c5f6f75, review clean)
User decision (2026-10-17): keep the Phonikud Hebrew voice files — installation is non-commercial. Resolves R10.
Ruling R11: viseme min-hold — A, F, G, H are "protected" shapes (visually essential: lip closure/contact) and are never absorbed; a protected segment shorter than 1 frame (0.033 s) is lengthened to 1 frame by taking time from the following segment; only B, C, D, E, X segments shorter than 2 frames (0.067 s) are absorbed into the previous segment — plan's flat 2-frame rule removed all lip closures — cost if wrong: slightly jittery mouth on fast consonants.
Ruling R12: investigate the ~4.6 s sd.play() start delay on the Mac within Task 14's fix round; if real, switch Player to one persistent sd.OutputStream opened at startup (keep the play() interface and pacing) — latency is a spec-level goal (2–4 s) — cost if wrong: small audio.py change.
Ruling R13: user supplied the final front-facing source image; saved as assets/face/source/trickster.png (1254x1254); Task 17 stub bank uses it instead of the framed reference.
Task 14: review → Needs fixes (protected donor shrink breaks R11; stream torn down under live writer on shutdown). Fix round 1 dispatched (+ early write-error stop, non-fatal startup open — Ruling R14: startup output.open failure logged and retried on first play, to avoid systemd crash-loop).
Task 14: minor (deferred): D→A also gets B (beyond brief, untested); unlogged sub-frame protected when no donor; torch FutureWarning in test output; Orin idle OutputStream behaviour unverified
Ruling R15: user asked for filler lines; Task 18 adds pre-synthesized fillers + streamed sentence-by-sentence replies (LLM SSE + synth-ahead); runs after 14, before 15–17 — cost if wrong: fillers can be disabled by config.
User decision: persona = oracle (mysterious, playful, riddles/omens, never cruel); folded into Task 18 with oracle-voice fillers and fallback.
User request: oracle knows Midburn — config/oracle_world.md (editable background: Negev, Flamingo camp, Man/Temple, 10 principles, burner culture, taboos) + time-of-day line; folded into Task 18; cache_prompt for latency. Event dates/theme for 2026 not found online — left out, user can add.
Task 14: fix round 1/5 (3 addressed, 0 open — legal donors; writer cancel/join before close; early write-error stop; non-fatal open; commits bbd79cb..2424fce)
Task 14: minor (deferred): final writer.join()/tail sleep outside the abort try — BaseException there bypasses _abort
Task 14: complete (commits c5f6f75..2424fce, review clean after 1 fix round)
Task 18: minor (deferred → Task 19): cross-sentence stage directions leak in reply_stream; no THINKING during mid-reply stall; joins wait on blocked HTTP reads (≤20 s); partial reply lost from history on mid-reply TTS failure; synth-ahead overlap untested; per-line cancel path untested; torch FutureWarning
Task 18: complete (commits 2424fce..a88b232, review clean). Latency on 8 GB Mac: filler 1.5–1.7 s, first sentence 5–7 s warm.
Ruling R16: add Task 19 (latency + Hebrew tuning) after Task 15: per-language llama-server slot pinning (-np 2, id_slot en=0 he=1) + prompt-cache warmup at startup, stage-direction leak fix, THINKING on mid-reply stall, Hebrew few-shot examples for plural address — user was told these fixes are coming — cost if wrong: small llm/config changes.
Ruling R17: order changed to 15 → 17 → 16 → 19 → 12 so the user sees the talking sprite face sooner (user asked when) — cost if wrong: latency fixes land ~1–2 h later.
Ruling R18: Task 17 dispatched while Task 15's review runs (disjoint files) to get the sprite demo sooner; if 15's review changes the bank API, 17 gets the change in its own fix round — cost if wrong: one rework round on 17.
Task 15: minor (deferred): unused `what` param; malformed eyes/expressions sections raise bare KeyError; no box bounds check; untested wrong-size patch / eye-box mismatch / loop dtype
Task 15: carried to T16: use FaceBank.mouth_patch/eye_patch, never index dicts directly
Task 15: complete (commits a88b232..c7c8ddd, review clean)
Ruling R19: Task 12 (deploy files only; on-device steps skipped per user) dispatched in parallel with Task 17 — disjoint files (deploy/, README.md, Makefile kill-0 fix vs tools/, assets/, docs/FACE_ASSETS.md) — cost if wrong: a README merge touch-up.
Task 17: implementer DONE_WITH_CONCERNS (e10a0b7); controller viewed contact sheet — mouth/eyes read well; source image committed separately.
Ruling R20: one eye patch serves both eyes (bank format) — Task 16 flips the patch horizontally for the right eye (face is symmetric) instead of changing the bank format now — cost if wrong: per-side patches later for asymmetric real banks.
Ruling R21: Task 16 dispatched while Task 17's review runs (17 is a dependency but its output already verified visually) — cost if wrong: one fix round on 16.
Task 17: review → Needs fixes (untested expr/shape_E..H paths; silent ECC failure; late missing-file errors). ⚠️ renderer/bank config keys resolved: Task 16 brief adds exactly `face.renderer`/`face.bank`. Fix round 1 dispatched (+ square box check, stale-file cleanup).
Task 17: minor (deferred): ECC mask coords vs input; cavity threshold can blacken deep beard shadow; A_mild/A_wide identical files; emoji in guide; per-side eye patches (patches_right) for real banks
Task 17: fix round 1/5 (5 addressed, 0 open — expr/shape tests; ECC & shift warnings; upfront missing-frame check; square box; stale cleanup; commit d4cc863)
Task 17: minor (deferred): ECC-failure warning untested
Task 17: complete (commits c7c8ddd..d4cc863, review clean after 1 fix round)
Task 12: first review stalled (>1.5 h, no output) — stopped and re-dispatched; Task 16 implementer also stalled 2 h after rendering demo — stopped and resumed with finish instructions + wider mouth (Ruling R22: jaw ≈ mouth×0.85 if safe; speaking must read clearly at distance).
Task 12: review Approved with 3 Important (DPMS re-apply → Ruling R23: user timer every 10 min; XDG_RUNTIME_DIR guard; /etc/gdm3 check) → fix round 1 dispatched (+ torch guard, xrandr only when needed, README usermod note).
Task 12: minor (deferred): NovaLCT menu names unverified; nvpmodel.service After= existence unverified; CUDA build of pinned llama/whisper versions on JetPack 6 untested
Task 12: fix round 1/5 (6 addressed, 0 open — dpms timer; user-bus guard; gdm3 check; torch guard; xrandr conditional; usermod note; commit ed5146b)
Task 12: complete (commits d4cc863..ed5146b, review clean after 1 fix round; on-device verification pending hardware)
Task 16: implementer stalled twice; finished by fresh agent (c7bf9c1): 170/171 (1 flaky slow Phonikud alignment test), render 3.0 ms median; jaw = min(0.6, mouth×0.85) approved (R22). Review dispatched. Task 19 dispatched in parallel (disjoint files).
Task 16: review → Needs fixes (Important: repo_path duplicates voice_main.resolve_repo_path → move to mirror/config.py). Extra tools/demo_face.py — Ruling R24: controller-requested demo tool, kept in scope.
Ruling R25: Task 16 fix round waits until Task 19 commits (fix touches voice_main.py, which Task 19 is editing) — cost if wrong: short delay.
Task 16: minor (fold into fix round): double squint (expression + eye patch); smirk fires immediately on return to idle; expose public eye-name accessor for demo; note unsprung jaw + hysteresis. Deferred: jaw test conflates patch change.
Note: tests/test_tts.py::test_phonikud_alignments_real order-dependent flake — investigate in final review.
Task 19: implementer DONE_WITH_CONCERNS (4fc86e6): 186/186; pinned slots work (1–17 tokens re-eval); e2e smoke blocked by Mac swap (9.6 GB) + PortAudio hang; Hebrew still mixes singular/plural; cache_reuse unsupported for Gemma 3 (harmless).
Ruling R26: per-slot context 2048 is thin for Hebrew (~1174-token prompt + 6-message history + reply) — raise to `-c 8192 -np 2` (4096/slot) in Makefile and mirror-llm.service; Gemma 3 4B KV at 8k fits the Orin's budget — cost if wrong: ~0.3–0.5 GB more RAM; revert to 6144.
Task 16: fix round 1/5 (5 addressed, 0 open — shared resolve_repo_path; squint; smirk reschedule; eye_name; comments; commit bd228fb)
Task 16: minor (deferred): unused `import os` in mirror/face/app.py; jaw test conflates patch change
Task 16: complete (commits ed5146b..bd228fb, review clean after 1 fix round)
Task 19: review → Needs fixes (re-warm thread not stopped on exit; re-warm overlaps live request on same slot). Fix round 1 dispatched (+ Hebrew example 3 number, anti-parroting header, strip "תשובה:", parenthetical directions, exc_info, retry failed startup warm, R26 -c 8192).
Task 19: minor (deferred): prompt_ms missing when stream cut at max_sentences; id_slot honoured by installed llama.cpp (server log suggests yes)
Task 19: fix round 1/5 (all addressed — close() joins re-warm; one-slot re-warm + bounded wait in _answer; Hebrew examples; label/paren stripping; exc_info; warm retry; -c 8192; commit eca395e)
Task 19: complete (commits c7bf9c1..eca395e, review clean after 1 fix round; e2e latency unmeasured on Mac due to swap)
All tasks complete (1–19). Final whole-branch review next.
Final review: ready with fixes — I1 audio formats on Orin, I2 mic/writer timeouts, I3 serial write_timeout, I4 dependency pins, flaky phonikud test; one fix wave dispatched (+ cheap M-items).
Final review fix wave: all addressed (89ef275), 212 tests; final review clean.

## 2026-10-19
Task 26 (varied answers, dfb4322): review → Needs fixes (Important: angle note could be spoken). Fix round 1 (a136827): angle-echo filter in trim_reply + stream; explicit stage-direction word list; Hebrew wording. Variety (seed 1, 25 replies/lang): en distinct-1 0.340 → 0.409, he stock words 40% → 8%.
Follow-up (f10fce0): 40% of replies ran past max_chars and were spoken cut off → prompt asks for < 40 words; a later sentence that doesn't fit, or an unfinished one (max_tokens), is dropped instead of cut. Re-measure: 0/50 cut, avg 194 chars.
895f9de: crude-sentence filter (en + he; whole words, so Sussex/cocktail/כוס stay). Seen: "החרמנים" in a Hebrew reply.
Voice: user asked for Jafar's voice → declined (real actor's voice); offered an original villain-style voice. 35caa1f + 6b24dc5: tts.voice_speed / voice_room effect (default off). User designed a voice in ElevenLabs → models/voices/mirror.npy (Pocket clone, not in git); 4bcd2fd: Pocket pauses capped at 0.4 s; acf3d01: config/mac.toml uses it; Orin keeps Piper until measured. Mac (M2, 2 threads): RTF 0.22 en / 0.25–0.32 he, ~0.9 GB peak RSS. Under the live demo the 8 GB Mac swaps (Hebrew synth 8–13 s, same with Piper earlier).
2b292be: world file touches the theme "now and then" ("carry" 12 → 1 in 25 en replies). A "never open with You seek" rule made stock words worse (52%) → reverted.
d8be975: hebrew_address_check sends angles. Singular address ~5/20 both with and without angles (after reading out את false positives), mostly on feminine questions. Known 4B limit; above the bring-up 20% PASS line.
Open: merge choice (user); Orin bring-up + memory with Pocket (hardware Oct 24+); user to listen to the Hebrew in the cloned voice.
