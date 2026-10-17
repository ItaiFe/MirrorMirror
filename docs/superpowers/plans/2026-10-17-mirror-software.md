# Mirror, Mirror Software Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the software for the talking mirror: a face on the LED panels, a bilingual (Hebrew/English) voice conversation that starts with a wake phrase, and frame lights that follow the conversation. Develop and run it on a Mac, then deploy it to the Jetson Orin Nano.

**Architecture:** One Python package, `mirror`, runs as two processes. `mirror-voice` handles mic → speech segments → wake phrase → STT → LLM → TTS → playback, and sends state to the lights. `mirror-face` draws the face. The voice process sends `{state, mouth}` to the face over localhost UDP. Speech-to-text and the LLM run as separate HTTP servers (whisper.cpp `whisper-server`, llama.cpp `llama-server`), the same on the Mac (Metal) and the Orin (CUDA). Each machine has its own TOML config file.

**Tech Stack:** Python 3.10+, pygame 2, numpy, sounddevice, requests, pyserial, onnxruntime (Silero VAD), piper-tts, transformers + torch (MMS Hebrew TTS), tomli, pytest. External: whisper.cpp, llama.cpp.

**Spec:** `MIRROR_CONTEXT.md` (repo root). Full brief: https://claude.ai/artifact/UmGjaoohxeX6HtMgGn3iWn

## Global Constraints

- Fully offline at runtime. Models are downloaded at setup time only.
- Same code on Mac and Orin. Machine differences live only in `config/mac.toml` and `config/orin.toml`.
- Face area: 252 × 252 px drawn at the top-left of a 1920 × 1080 screen on the Orin. On the Mac it's shown in a window scaled ×3.
- Wake phrases: "mirror mirror" or "מראה מראה", fuzzy matched at the start of an utterance. "on the wall" / "שעל הקיר" are optional.
- Languages: `he` and `en`. Reply in the detected language; anything else is treated as `en`.
- Persona: a magic mirror. Playful, a little vain, never cruel. 1–3 sentences per reply.
- States: `idle`, `listening`, `thinking`, `speaking`. Lights use WLED presets 1–4 in that order.
- Audio: 16 kHz mono capture. Mouth level is the RMS of the playing TTS audio in 20 ms hops, normalized 0..1.
- No disk writes during operation except logs to stdout (journald runs in volatile mode on the Orin).
- All tests run on a Mac with no hardware and no model servers (`SDL_VIDEODRIVER=dummy`, fakes for audio/HTTP/serial).

## Review Focus

1. **Whisper hallucinations on noise.** Whisper turns wind or silence into "Thank you." / "תודה רבה" / empty text. These must never wake the mirror or count as a question (Task 2 test `test_hallucinations_ignored`).
2. **The mirror hears itself.** Audio segments captured while it's speaking must be dropped, even if AEC lets some through (Task 9 test `test_segments_during_speaking_dropped`).
3. **A model server is down or slow.** STT or LLM timeouts must not crash the voice process. The mirror says a short canned line in the right language and returns to idle (Task 9 test `test_llm_failure_speaks_fallback`).
4. **Wake phrase alone, then a pause.** "Mirror mirror" … pause … question must work as well as the phrase and question in one breath. If nothing follows within 8 s, it returns to idle (Task 9 tests `test_wake_then_question_next_segment`, `test_listening_timeout_returns_idle`).
5. **Runaway or off-language replies.** Replies are trimmed to 3 sentences and 300 characters. A reply in an unsupported language is sent to the TTS for the question's language anyway (Task 6 test `test_reply_trimmed`).

---

## File Structure

```
pyproject.toml              package + deps + entry points (mirror-voice, mirror-face)
Makefile                    dev, test, servers-mac targets
config/mac.toml             dev config
config/orin.toml            deploy config
mirror/__init__.py
mirror/config.py            Config dataclasses + load_config(path)
mirror/states.py            State enum
mirror/bus.py               FaceBus UDP sender/receiver
mirror/wake.py              normalize(), match_wake()
mirror/vad.py               Segmenter + SileroVad
mirror/audio.py             mouth_envelope(), Mic, Player
mirror/stt.py               Transcript, WhisperServerSTT
mirror/persona.py           system prompts, fallback lines
mirror/llm.py               LlamaServerLLM, trim_reply()
mirror/tts.py               PiperTTS, MmsHebrewTTS, TTSRouter
mirror/lights.py            WledSerial, LogLights
mirror/conversation.py      Controller (state machine)
mirror/voice_main.py        wires the voice process
mirror/face/draw.py         draw_face()
mirror/face/app.py          face process main loop
deploy/setup_orin.sh        build servers, fetch models, install units
deploy/fetch_models.sh      shared model download (Mac + Orin)
deploy/systemd/*.service    mirror-stt, mirror-llm, mirror-voice, mirror-face (user units)
deploy/xorg/10-mirror.conf  force 1920x1080, no blanking
tests/...                   one test file per module
```

---

### Task 1: Project skeleton, config, states, face bus

**Files:**
- Create: `pyproject.toml`, `Makefile`, `.gitignore` (`models/`, `.venv/`, `__pycache__/`), `config/mac.toml`, `config/orin.toml`, `mirror/__init__.py`, `mirror/config.py`, `mirror/states.py`, `mirror/bus.py`
- Test: `tests/test_config.py`, `tests/test_bus.py`

**Interfaces:**
- Produces:
  - `class State(str, Enum)`: `IDLE="idle"`, `LISTENING="listening"`, `THINKING="thinking"`, `SPEAKING="speaking"`
  - `@dataclass Config` with sections `audio` (`input_device: str|None`, `output_device: str|None`, `sample_rate=16000`), `servers` (`stt_url`, `llm_url`, `timeout_s=20`), `tts` (`piper_voice_en: str`, `mms_model_he="facebook/mms-tts-heb"`), `face` (`size=252`, `screen_w`, `screen_h`, `fullscreen: bool`, `window_scale: int`, `bus_port=7777`), `lights` (`kind: "wled"|"log"`, `serial_port: str|None`, `baud=115200`), `conversation` (`listen_timeout_s=8.0`, `followup_s=6.0`, `max_reply_sentences=3`, `max_reply_chars=300`)
  - `load_config(path: str) -> Config`
  - `FaceBus.sender(port) -> FaceBus` with `send(state: State, mouth: float) -> None`; `FaceBus.receiver(port)` with `poll() -> tuple[State, float] | None` (non-blocking, returns latest message, drains queue)

- [ ] **Step 1: Write failing tests**

```python
def test_load_mac_config():
    c = load_config("config/mac.toml")
    assert c.face.size == 252 and c.face.fullscreen is False and c.face.window_scale == 3
    assert c.lights.kind == "log"
    assert c.servers.llm_url.startswith("http://127.0.0.1")

def test_load_orin_config():
    c = load_config("config/orin.toml")
    assert (c.face.screen_w, c.face.screen_h) == (1920, 1080) and c.face.fullscreen is True
    assert c.lights.kind == "wled" and c.lights.serial_port == "/dev/ttyUSB0"

def test_bus_roundtrip():
    rx = FaceBus.receiver(17777); tx = FaceBus.sender(17777)
    tx.send(State.THINKING, 0.0); tx.send(State.SPEAKING, 0.42)
    time.sleep(0.05)
    assert rx.poll() == (State.SPEAKING, 0.42)   # latest wins
    assert rx.poll() is None
```

- [ ] **Step 2: Run `pytest tests/test_config.py tests/test_bus.py -v`.** Expected: FAIL (import errors)
- [ ] **Step 3: Implement.** Config files: Mac uses `input_device`/`output_device` unset (system default), `lights.kind="log"`, `stt_url="http://127.0.0.1:8081"`, `llm_url="http://127.0.0.1:8080"`. Orin uses `input_device="XVF3800"`, `output_device="XVF3800"` (substring match on the device name), `/dev/ttyUSB0`, fullscreen. Bus messages are JSON `{"s": "speaking", "m": 0.42}`. `pyproject.toml` declares the entry points `mirror-voice = mirror.voice_main:main` and `mirror-face = mirror.face.app:main`, and the extras `tts` (piper-tts, transformers, torch). `Makefile` targets: `test` (`SDL_VIDEODRIVER=dummy pytest -q`), `servers-mac`, `dev` (filled in Task 11).
- [ ] **Step 4: Run tests.** Expected: PASS
- [ ] **Step 5: Commit** `feat: project skeleton, config, state bus`

---

### Task 2: Wake phrase matching

**Files:**
- Create: `mirror/wake.py`
- Test: `tests/test_wake.py`

**Interfaces:**
- Produces: `normalize(text: str) -> str` (lowercase, Hebrew niqqud U+0591–U+05C7 removed, punctuation → space, whitespace collapsed); `@dataclass WakeMatch(remainder: str)`; `match_wake(text: str) -> WakeMatch | None`; `is_hallucination(text: str) -> bool`

Matching: normalize, then look at the first two tokens. Each must be within `difflib.SequenceMatcher` ratio ≥ 0.75 of `mirror` (English), or within the Hebrew variant set `{"מראה","מראא","מרא","מירא"}` or ratio ≥ 0.75 of `מראה`. Then drop an optional following `on the wall` / `שעל הקיר` / `על הקיר` (each token ratio ≥ 0.75). Whatever is left is `remainder`.

- [ ] **Step 1: Write failing tests**

```python
@pytest.mark.parametrize("text,rem", [
    ("Mirror, mirror on the wall, who is the fairest?", "who is the fairest"),
    ("mirror mirror", ""),
    ("Mirrors mirror, tell me a joke", "tell me a joke"),
    ("מראה מראה שעל הקיר, מי הכי יפה?", "מי הכי יפה"),
    ("מַרְאָה מַרְאָה", ""),
    ("מרא מראה ספרי לי בדיחה", "ספרי לי בדיחה"),
])
def test_matches(text, rem):
    assert match_wake(text).remainder == rem

@pytest.mark.parametrize("text", ["the mirror is nice", "hello there", "מה קורה", "", "mirror"])
def test_no_match(text):
    assert match_wake(text) is None

@pytest.mark.parametrize("text", ["", "  ", "Thank you.", "תודה רבה.", "[BLANK_AUDIO]", "(wind blowing)", "you"])
def test_hallucinations_ignored(text):
    assert is_hallucination(text)
    assert match_wake(text) is None
```

- [ ] **Step 2: Run `pytest tests/test_wake.py -v`.** Expected: FAIL
- [ ] **Step 3: Implement** per the matching rule above. Hallucinations are empty after normalization, bracketed or parenthesized tags, or one of: `thank you`, `thanks for watching`, `you`, `תודה`, `תודה רבה`, `תודה שצפיתם`.
- [ ] **Step 4: Run tests.** Expected: PASS
- [ ] **Step 5: Commit** `feat: bilingual wake phrase matching`

---

### Task 3: Speech segmenter (VAD)

**Files:**
- Create: `mirror/vad.py`
- Test: `tests/test_vad.py`

**Interfaces:**
- Produces: `class Segmenter(is_speech: Callable[[np.ndarray], bool], frame_samples=512, sample_rate=16000, min_silence_ms=700, min_speech_ms=250, max_segment_s=15.0, preroll_ms=300)` with `feed(chunk: np.ndarray) -> list[np.ndarray]` (float32 mono, any chunk length; returns completed segments) and `reset() -> None`; `class SileroVad(model_path: str, threshold=0.5)`, callable `(frame: np.ndarray) -> bool`, running Silero VAD v5 ONNX via onnxruntime on 512-sample frames at 16 kHz while keeping its own state.

- [ ] **Step 1: Write failing tests**, with a fake `is_speech = lambda f: np.abs(f).mean() > 0.1`:
  - `test_single_utterance`: 0.5 s silence + 1.0 s tone (amplitude 0.5) + 1.0 s silence → exactly 1 segment, length between 1.0 s and 1.0 + 0.3 + 0.7 s of samples
  - `test_short_blip_dropped`: 0.1 s tone surrounded by silence → 0 segments
  - `test_max_length_split`: 20 s continuous tone → first segment emitted at 15 s
  - `test_arbitrary_chunk_sizes`: the same audio fed in chunks of 100, 512 and 3000 samples gives the same segment count
  - `test_reset_discards_partial`: feed 0.5 s of tone, `reset()`, then 1 s silence → 0 segments
- [ ] **Step 2: Run `pytest tests/test_vad.py -v`.** Expected: FAIL
- [ ] **Step 3: Implement** `Segmenter` (a ring buffer for preroll; a frame counter for silence) and `SileroVad` (not unit-tested; it's exercised in Task 11's manual run).
- [ ] **Step 4: Run tests.** Expected: PASS
- [ ] **Step 5: Commit** `feat: VAD speech segmenter`

---

### Task 4: Audio I/O and mouth envelope

**Files:**
- Create: `mirror/audio.py`
- Test: `tests/test_audio.py`

**Interfaces:**
- Produces:
  - `mouth_envelope(samples: np.ndarray, sr: int, hop_s=0.02) -> np.ndarray` returns RMS per hop, divided by the 95th percentile and clipped to 0..1 (all zeros for silent input)
  - `find_device(name_substr: str | None, kind: "input"|"output") -> int | None` (sounddevice index; `None` means the default)
  - `class Mic(device, sample_rate)`: context manager; `read() -> np.ndarray` blocks for the next ~32 ms chunk (float32 mono)
  - `class Player(device)`: `play(samples: np.ndarray, sr: int, on_level: Callable[[float], None]) -> None` blocks until done, calling `on_level` every 20 ms with the envelope value for the hop currently playing, then once with `0.0` at the end

- [ ] **Step 1: Write failing tests**
  - `test_envelope_silence`: zeros → all 0, length = `ceil(len/hop)`
  - `test_envelope_tracks_loudness`: 0.2 s at amplitude 0.05 followed by 0.2 s at 0.5 → mean of the second half > 3× the mean of the first half; max ≤ 1.0
  - `test_player_reports_levels` (monkeypatch `sounddevice.play`/`wait` to no-ops and `time.sleep` to a recorder): `on_level` is called `ceil(dur/0.02)+1` times and the last value is 0.0
- [ ] **Step 2: Run `pytest tests/test_audio.py -v`.** Expected: FAIL
- [ ] **Step 3: Implement.** `Player.play` starts `sd.play`, then steps through the envelope on a 20 ms clock measured with `time.monotonic()` against the start time (no drift), then calls `sd.wait()`.
- [ ] **Step 4: Run tests.** Expected: PASS
- [ ] **Step 5: Commit** `feat: audio io and mouth envelope`

---

### Task 5: Speech-to-text client

**Files:**
- Create: `mirror/stt.py`
- Test: `tests/test_stt.py`

**Interfaces:**
- Produces: `@dataclass Transcript(text: str, lang: str)` (`lang` is `"he"` or `"en"`; anything else maps to `"en"`); `class WhisperServerSTT(url: str, timeout_s: float)` with `transcribe(samples: np.ndarray, sr=16000) -> Transcript`. It raises `STTError` on HTTP or timeout failure.

The request is a POST to `{url}/inference` as multipart, with `file` = a 16-bit PCM WAV built in memory, `response_format=verbose_json`, `language=auto` and `temperature=0`. Read `text` and the detected language field from the response. Before finishing this task, check the field name against a running `whisper-server` built in Task 11's `make servers-mac` and pin it in a test fixture.

- [ ] **Step 1: Write failing tests** (with `requests` mocked via `responses` or monkeypatch):
  - `test_parses_hebrew`: response `{"text":" מראה מראה ","language":"he"}` → `Transcript("מראה מראה","he")`
  - `test_unknown_lang_maps_to_en`: `"language":"ar"` → `lang == "en"`
  - `test_timeout_raises`: requests timeout → `STTError`
  - `test_wav_payload`: the posted file starts with `b"RIFF"` and declares 16000 Hz
- [ ] **Step 2: Run `pytest tests/test_stt.py -v`.** Expected: FAIL
- [ ] **Step 3: Implement.** Add `responses` to the dev dependencies.
- [ ] **Step 4: Run tests.** Expected: PASS
- [ ] **Step 5: Commit** `feat: whisper-server STT client`

---

### Task 6: Persona and LLM client

**Files:**
- Create: `mirror/persona.py`, `mirror/llm.py`
- Test: `tests/test_llm.py`

**Interfaces:**
- Produces:
  - `SYSTEM_PROMPT: dict[str, str]` keyed by `"he"`/`"en"`. Both say: you are the magic mirror on the wall at Midburn; playful, a little vain, flattering with a wink, never cruel or insulting about anyone's looks; answer in 1–3 short spoken sentences; no emojis, lists or stage directions; answer only in Hebrew / only in English. The Hebrew prompt is written in Hebrew and addresses the visitor in a gender-neutral way.
  - `FALLBACK: dict[str, str]`: `"en": "The mirror's mist is too thick right now. Ask me again in a moment."`, `"he": "הערפל על המראה סמיך מדי כרגע. שאלו אותי שוב עוד רגע."`
  - `trim_reply(text: str, max_sentences: int, max_chars: int) -> str`
  - `class LlamaServerLLM(url, timeout_s, max_sentences=3, max_chars=300)` with `reply(history: list[dict], user_text: str, lang: str) -> str`. It POSTs to `{url}/v1/chat/completions` with `[system] + history + [user]`, `max_tokens=120` and `temperature=0.8`, returns the trimmed text, and raises `LLMError` on failure or an empty reply.

- [ ] **Step 1: Write failing tests**
  - `test_reply_trimmed`: `trim_reply("One. Two! Three? Four. Five.", 3, 300) == "One. Two! Three?"`; a 500-character sentence is cut to ≤ 300 characters at the last space and ends with `…`; `*smiles*` and emojis are stripped
  - `test_hebrew_sentence_split`: `trim_reply("אחת. שתיים. שלוש. ארבע.", 3, 300) == "אחת. שתיים. שלוש."`
  - `test_request_shape` (mocked HTTP): the first message is `SYSTEM_PROMPT["he"]` when `lang="he"`, the history is preserved in order, and the last message is the user's
  - `test_llm_error_on_timeout_and_empty`: timeout → `LLMError`; `{"choices":[{"message":{"content":"  "}}]}` → `LLMError`
- [ ] **Step 2: Run `pytest tests/test_llm.py -v`.** Expected: FAIL
- [ ] **Step 3: Implement.** Split sentences on `(?<=[.!?…])\s+`.
- [ ] **Step 4: Run tests.** Expected: PASS
- [ ] **Step 5: Commit** `feat: persona and llama-server client`

---

### Task 7: Text-to-speech

**Files:**
- Create: `mirror/tts.py`
- Test: `tests/test_tts.py`

**Interfaces:**
- Produces: a `Synth` protocol `synth(text: str) -> tuple[np.ndarray, int]` (float32 mono, sample rate); `PiperTTS(voice_path: str)`; `MmsHebrewTTS(model_name: str)` (transformers `VitsModel` + `AutoTokenizer`, CUDA when available, otherwise CPU, otherwise MPS); `class TTSRouter(en: Synth, he: Synth)` with `synth(text, lang) -> tuple[np.ndarray, int]`, where an unknown `lang` goes to `en`; `warmup() -> None` synthesizes one short phrase in each language so the first real reply is fast.

- [ ] **Step 1: Write failing tests** with fake synths returning distinguishable arrays:
  - `test_router_picks_language`: `he` → the Hebrew fake, `en` → the English fake, `"fr"` → English
  - `test_router_output_float32_mono`: int16 output from a fake is converted to float32 in −1..1
  - `@pytest.mark.slow test_piper_real` and `test_mms_real`: skipped unless the model files exist; they synthesize "hello" / "שלום" and assert the result is longer than 0.2 s
- [ ] **Step 2: Run `pytest tests/test_tts.py -v -m "not slow"`.** Expected: FAIL
- [ ] **Step 3: Implement.** Piper is loaded through the `piper` Python API (`PiperVoice.load`); voice `en_US-lessac-medium`.
- [ ] **Step 4: Run tests.** Expected: PASS
- [ ] **Step 5: Commit** `feat: bilingual TTS router`

---

### Task 8: Frame lights

**Files:**
- Create: `mirror/lights.py`
- Test: `tests/test_lights.py`

**Interfaces:**
- Produces: a `Lights` protocol `set_state(state: State) -> None`; `PRESETS = {IDLE: 1, LISTENING: 2, THINKING: 3, SPEAKING: 4}`; `WledSerial(port, baud, write: Callable[[bytes], None] | None = None)` sends `b'{"ps":N}\n'` and only when the state changes. If the port can't be opened or a write fails, it logs once and keeps running (it retries opening every 10 s on the next `set_state`). `LogLights` logs each state change.

- [ ] **Step 1: Write failing tests**
  - `test_sends_preset_json`: SPEAKING → write receives `b'{"ps":4}\n'`
  - `test_dedupes_same_state`: IDLE twice → one write
  - `test_write_failure_does_not_raise`: write raises `OSError` → `set_state` returns normally
- [ ] **Step 2: Run `pytest tests/test_lights.py -v`.** Expected: FAIL
- [ ] **Step 3: Implement.** The four presets themselves are created by hand in the WLED web UI. Task 12's README documents them: idle = slow pink breathe, listening = steady teal, thinking = rotating chase, speaking = bright pulse.
- [ ] **Step 4: Run tests.** Expected: PASS
- [ ] **Step 5: Commit** `feat: WLED serial frame lights`

---

### Task 9: Conversation controller

**Files:**
- Create: `mirror/conversation.py`
- Test: `tests/test_conversation.py`

**Interfaces:**
- Consumes: `State`, `match_wake`, `is_hallucination`, `Transcript`, `STTError`, `LLMError`, `FALLBACK`, `TTSRouter.synth`, `Player.play`, `Lights.set_state`, `FaceBus.send`
- Produces: `class Controller(stt, llm, tts, player, lights, face, cfg: ConversationConfig, clock=time.monotonic)` with `on_segment(samples: np.ndarray) -> None` and `tick() -> None` (called about every 100 ms for timeouts) and a `state: State` property

Behavior:
- `IDLE` + segment → transcribe. Hallucination or no wake match → stay `IDLE`. Wake match with a non-empty remainder → answer the remainder. Wake match with an empty remainder → `LISTENING` (deadline `listen_timeout_s`).
- `LISTENING` + segment → transcribe → answer it unless it's a hallucination. `tick()` past the deadline → `IDLE`, history cleared.
- Answer: `THINKING` → `llm.reply(history, text, lang)` → `SPEAKING` → `tts.synth` → `player.play(on_level=lambda m: face.send(SPEAKING, m))` → append the user and assistant turns to the history (keep the last 6 messages) → `LISTENING` with deadline `followup_s` (a follow-up question needs no wake phrase).
- `STTError` → back to the previous resting state. `LLMError` → speak `FALLBACK[lang]`, then `IDLE`.
- Segments arriving while `THINKING` or `SPEAKING` are dropped. `on_segment` runs on the voice thread, so they show up when queued: the caller drains the queue after playback ends (Task 11).
- Every state change calls `lights.set_state` and `face.send(state, 0.0)`.

- [ ] **Step 1: Write failing tests** with fakes (scripted STT transcripts, LLM replies, TTS returning 0.1 s of audio, a recording player/lights/face, an injectable clock):
  - `test_wake_and_question_one_breath`: transcript "mirror mirror who is fairest" → LLM called with `"who is fairest"`, `"en"`; lights sequence `[THINKING, SPEAKING, LISTENING]`
  - `test_wake_then_question_next_segment`: "מראה מראה" → `LISTENING`; next "מי הכי יפה" (he) → LLM called with lang `he`
  - `test_listening_timeout_returns_idle`: wake only, advance the clock 8.1 s, `tick()` → `IDLE`
  - `test_followup_without_wake`: after a reply, the next segment (within 6 s) is answered without a wake phrase and the history passed to the LLM has 2 messages
  - `test_non_wake_ignored_in_idle`: "hello there" → no LLM call, state `IDLE`
  - `test_hallucination_in_listening_ignored`: wake → `LISTENING`; "Thank you." → still `LISTENING`, no LLM call
  - `test_segments_during_speaking_dropped`: the fake player calls `controller.on_segment(...)` mid-playback → the STT is not called for it
  - `test_llm_failure_speaks_fallback`: LLM raises `LLMError` with lang `he` → TTS is called with `FALLBACK["he"]` and `"he"`; the final state is `IDLE`
  - `test_stt_failure_stays_idle`: the STT raises → state `IDLE`, nothing played
  - `test_mouth_levels_forwarded`: the face receives `(SPEAKING, level)` messages from the player's `on_level`
- [ ] **Step 2: Run `pytest tests/test_conversation.py -v`.** Expected: FAIL
- [ ] **Step 3: Implement** `Controller` as above. While busy, it sets an internal `busy` flag that `on_segment` checks.
- [ ] **Step 4: Run tests.** Expected: PASS
- [ ] **Step 5: Commit** `feat: conversation state machine`

---

### Task 10: Face drawing

**Files:**
- Create: `mirror/face/__init__.py`, `mirror/face/draw.py`
- Test: `tests/test_face_draw.py`

**Interfaces:**
- Produces: `draw_face(surf: pygame.Surface, state: State, mouth: float, t: float, blink: float) -> None` draws on a square surface of any size (designed at 252). `blink` runs 0 (open) to 1 (closed). `t` is seconds, used for the idle drift.

Look (the builder's call within these rules): pure black background, since only lit pixels show through the acrylic. Two large almond eyes with glowing irises, arched brows, and a mouth whose opening height is proportional to `mouth`. Accent colour by state: idle `#ff7a8a` (flamingo pink), listening `#6fd5d0` (teal), thinking `#e8b457` (amber, eyes glance up-left and up-right), speaking `#ff7a8a`. Line widths are at least 3 px at 252 px so they survive the P2.976 pitch. Idle drifts the eyes by ±3 px on a slow sine.

- [ ] **Step 1: Write failing tests** (`SDL_VIDEODRIVER=dummy`, 252×252 surface):
  - `test_background_black`: corner pixel (0,0) is `(0,0,0)`
  - `test_mouth_opens`: count lit pixels in the mouth region (rows 170–230, cols 76–176) → `mouth=1.0` gives at least 1.5× the count of `mouth=0.0`
  - `test_blink_closes_eyes`: lit pixels in the eye band (rows 70–130) with `blink=1.0` are fewer than 40% of those with `blink=0.0`
  - `test_state_colour`: the dominant non-black hue with `State.LISTENING` is closer to teal than to pink
  - `test_scales`: drawing on a 756×756 surface doesn't raise and lights more than 1% of the pixels
- [ ] **Step 2: Run `SDL_VIDEODRIVER=dummy pytest tests/test_face_draw.py -v`.** Expected: FAIL
- [ ] **Step 3: Implement** with pygame primitives (ellipse, arc, polygon) in coordinates relative to `surf.get_width()`.
- [ ] **Step 4: Run tests.** Expected: PASS
- [ ] **Step 5: Commit** `feat: face drawing`

---

### Task 11: Process entry points and Mac dev loop

**Files:**
- Create: `mirror/face/app.py`, `mirror/voice_main.py`, `deploy/fetch_models.sh`
- Modify: `Makefile` (`servers-mac`, `dev`)
- Test: `tests/test_face_app.py`

**Interfaces:**
- Consumes: everything above
- Produces:
  - `mirror-face --config PATH`: opens the display (fullscreen at `screen_w×screen_h` with the face blitted at (0,0), or a window of `size*window_scale` with the face scaled), runs at 30 fps, polls the `FaceBus`, schedules a blink every 3–6 s at random (150 ms long), and smooths the mouth with a 60 ms attack and 120 ms release. Exits on Esc or SIGTERM.
  - `step_mouth(current: float, target: float, dt: float) -> float` in `app.py` (pure, tested)
  - `mirror-voice --config PATH`: builds the Mic, SileroVad and Segmenter, the clients, the TTSRouter (warmed up), the lights and the bus. The mic thread pushes segments into a `queue.Queue`. The main loop pulls segments, calls `controller.on_segment`, calls `tick()` every 100 ms, and drains the queue after each answer. SIGTERM shuts it down cleanly.
  - `deploy/fetch_models.sh MODELS_DIR`: downloads `ggml-small.bin` (whisper.cpp, multilingual), `gemma-3-4b-it-Q4_K_M.gguf`, the Silero VAD v5 onnx, and Piper `en_US-lessac-medium.onnx` + `.json`, and pre-caches `facebook/mms-tts-heb` into `MODELS_DIR/hf`. It skips files that already exist.
  - `make servers-mac`: `brew install whisper-cpp llama.cpp` if missing; runs `whisper-server -m models/ggml-small.bin --port 8081` and `llama-server -m models/gemma-3-4b-it-Q4_K_M.gguf --port 8080 -c 4096 -ngl 99`
  - `make dev`: `servers-mac` in the background, then `mirror-face --config config/mac.toml &` and `mirror-voice --config config/mac.toml`

- [ ] **Step 1: Write failing test** `test_step_mouth`: from 0 → 1 with `dt=0.06` reaches ≥ 0.6; from 1 → 0 with `dt=0.06` stays ≥ 0.3; the result is always within 0..1
- [ ] **Step 2: Run `pytest tests/test_face_app.py -v`.** Expected: FAIL
- [ ] **Step 3: Implement** the entry points, the script and the Makefile targets. Confirm the `whisper-server` language field name and pin it in Task 5's fixture.
- [ ] **Step 4: Run `make test`.** Expected: all PASS
- [ ] **Step 5: Manual Mac check.** With `make dev`: say "Mirror mirror on the wall, who is the fairest of them all?" → the face goes teal → amber → it speaks in English with the mouth moving. Say "מראה מראה שעל הקיר, מי הכי יפה?" → it answers in Hebrew. Stay silent for 10 s after a reply → back to idle pink. Write the measured end-of-speech → first-audio delay in the README.
- [ ] **Step 6: Commit** `feat: voice and face entry points, mac dev loop`

---

### Task 12: Orin deployment

**Files:**
- Create: `deploy/setup_orin.sh`, `deploy/systemd/mirror-stt.service`, `deploy/systemd/mirror-llm.service`, `deploy/systemd/mirror-voice.service`, `deploy/systemd/mirror-face.service`, `deploy/xorg/10-mirror.conf`, `README.md`

**Interfaces:**
- Consumes: the entry points and `fetch_models.sh` from Task 11
- Produces: an Orin that boots straight into a working mirror

`setup_orin.sh` (idempotent, run from the repo on the Orin):
1. apt: `build-essential cmake git python3-venv libsdl2-dev portaudio19-dev`
2. Clone and build whisper.cpp and llama.cpp with `-DGGML_CUDA=ON` into `~/mirror-bin`
3. Create a venv, `pip install -e .[tts]` with the NVIDIA JetPack torch wheel (pinned URL in the script with a comment on how to update it)
4. `deploy/fetch_models.sh ~/mirror-models`
5. Install the user units into `~/.config/systemd/user/`, `loginctl enable-linger`, enable all four
6. GDM autologin for the mirror user; copy `10-mirror.conf` into `/etc/X11/xorg.conf.d/`; turn off screen blanking (`xset s off -dpms` in the face unit's `ExecStartPre`)
7. `Storage=volatile` in `/etc/systemd/journald.conf`; disable swap on the SD card (`systemctl mask nvzramconfig` stays default; remove any swapfile)
8. `sudo nvpmodel -m 0` (max power mode) and `jetson_clocks` at boot

Units: all `Restart=always`, `RestartSec=2`. `mirror-voice` has `After=mirror-stt mirror-llm`. `mirror-face` has `Environment=DISPLAY=:0` and `After=graphical-session.target`.

- [ ] **Step 1: Run `bash -n` on every script and `systemd-analyze verify` on the units (run on the Orin).** Expected: no errors
- [ ] **Step 2: Write the README**: Mac quickstart (`make servers-mac`, `make dev`), Orin deploy (`git pull && deploy/setup_orin.sh`), the NovaLCT setup steps (Windows laptop over USB-B, 252×252 at (0,0), save to the receiving card), the WLED preset table from Task 8, and the bench test checklist from the brief's build order
- [ ] **Step 3: On the Orin (once the hardware is in hand): reboot.** Expected: the face appears on the panels within 90 s, and "Mirror mirror" gets an answer in both languages. `tegrastats` during an answer shows RAM under 7 GB.
- [ ] **Step 4: Commit** `feat: orin deployment`
