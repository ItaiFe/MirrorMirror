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

---

### Task 13: Hebrew voice via Phonikud + Piper (replaces MMS)

**Why:** MMS reads unvocalized Hebrew badly and is slow on CPU (7–9 s for three sentences on the Mac). Phonikud adds full Hebrew phonetics (vowels and stress, as IPA) to plain Hebrew text, and a Piper voice trained on those phonemes speaks it. Both are small ONNX models that run on CPU, offline.

**Files:**
- Modify: `mirror/tts.py` (add `PhonikudHebrewTTS`; keep `MmsHebrewTTS` as a fallback class), `mirror/config.py` + `config/*.toml` (`tts.he_engine = "phonikud"|"mms"`, Phonikud model paths), `mirror/voice_main.py` (build the configured Hebrew engine), `deploy/fetch_models.sh` (download the Phonikud and Hebrew Piper models), `pyproject.toml` (deps)
- Test: `tests/test_tts.py`

**Interfaces:**
- Produces: `PhonikudHebrewTTS(phonikud_model: str, voice_model: str, voice_config: str | None = None)` implementing `Synth.synth(text) -> (np.ndarray, int)`. It runs text → Phonikud (adds diacritics/phonetic marks) → IPA phonemes → Piper ONNX voice → audio.
- `TTSRouter` is unchanged; `voice_main` passes the configured Hebrew engine as `he=`.

- [ ] **Step 1: Find the real packages and models.** Expected: the Phonikud G2P (pip `phonikud` / `phonikud-onnx` and its ONNX model) and a Hebrew Piper checkpoint (HF `Phonikud/phonikud-tts-checkpoints` or the maintainers' Piper ONNX exports), plus how phonemes are fed to the voice (piper's phoneme-input API or `piper-onnx`). Record the URLs, versions and licenses in the report.
- [ ] **Step 2: Write failing tests.** Fast test (no models): `TTSRouter` with `he=` set to a fake still routes `he`. A `@pytest.mark.slow` test `test_phonikud_real` is skipped unless the models are present; it synthesizes "מראה מראה שעל הקיר, מי הכי יפה?" and asserts more than 0.5 s of float32 mono audio at the voice's sample rate.
- [ ] **Step 3: Implement** `PhonikudHebrewTTS`. Imports are lazy and models load once in `__init__`. Prefer onnxruntime on CPU, so the GPU stays free for the LLM.
- [ ] **Step 4: Run tests.** Expected: all pass, and the slow test passes on the Mac with models downloaded.
- [ ] **Step 5: Smoke + timing.** Run `mirror-voice --wav` with the Hebrew WAV (generate it with the new voice), measure synth time for a 3-sentence reply against MMS, and listen-check one output, saving it to the scratchpad.
- [ ] **Step 6: Commit** `feat: Phonikud Hebrew voice`

---

## Sprite-bank face (Tasks 14–17)

**Why:** the user chose the layered 2D sprite bank from the avatar research (`reports/Offline talking avatar options.md`, doc https://claude.ai/artifact/QsoJqX5tKrDjg3NMgvGZwz). The look is a sculpted bronze trickster face on pure black. The art is made offline from one AI image, and the Orin composites it on the CPU only. The GPU stays with the LLM.

**Shared decisions:**
- Mouth shapes: Rhubarb set `A B C D E F G H X` (A closed p/b/m · B clenched, most consonants incl. Hebrew ʁ χ · C open e/ɛ · D wide a · E rounded o · F puckered u/w · G f/v · H l · X rest), each with `mild` and `wide` variants.
- The face bus message gains a viseme: `{"s": state, "m": mouth, "v": "D"}`. `FaceBus.send(state, mouth, viseme="X")` and `poll() -> (State, float, str)`. Old senders without `v` mean `"X"`.
- Assets live in `assets/face/<bank-name>/`, which is git-tracked for small banks. `config.face.renderer = "sprite" | "procedural"` and `config.face.bank = "assets/face/<name>"`.
- Until the real LivePortrait assets exist, a **stub bank** generated from one image by warps alone keeps everything running end to end.

### Task 14: Viseme timeline from Piper phoneme alignments

**Files:** Create `mirror/visemes.py`. Modify `mirror/tts.py` (alignment-capable synth), `mirror/bus.py` (viseme field), `mirror/conversation.py` (forward visemes) and `mirror/face/app.py` (accept the 3-tuple from `poll`). Test `tests/test_visemes.py`, update `tests/test_bus.py`, `tests/test_conversation.py` and `tests/test_tts.py`.

**Interfaces:**
- `PHONEME_TO_SHAPE: dict[str, str]`, IPA phoneme → shape letter. Unknown phonemes map to `B`, pauses and silence to `X`.
- `@dataclass VisemeTrack(times: np.ndarray, shapes: list[str])`, with times = shape start seconds, sorted. `at(t: float) -> str` returns the shape in effect at `t`, or `"X"` outside the track.
- `track_from_alignments(alignments: list[tuple[str, int]], sr: int, total_samples: int) -> VisemeTrack`. Rescales the summed phoneme samples to `total_samples` (fixes Piper's ~25% short durations) and merges consecutive identical shapes. Enforces a minimum hold of 2 frames at 30 fps (≈0.067 s) by absorbing shorter segments into the previous one. Inserts `B` between `A` and `D` when they are adjacent.
- `TTSRouter.synth_visemes(text, lang) -> tuple[np.ndarray, int, VisemeTrack | None]`. Engines that can align (Piper and the Phonikud/Piper Hebrew voice, if Task 13's report says alignments work) return a track, MMS returns `None`. `synth()` stays as is.
- Controller: speak via `synth_visemes`. In `on_level` at hop k, t = k × 0.02 s + 0.05 s lead, and it sends `face.send(SPEAKING, level, track.at(t) if track else rms_shape(level))`, where `rms_shape(level)` gives `X` <0.08, `B` <0.35, `C` <0.65, else `D`.

**Tests (names + assertions):**
- `test_rescale_to_audio_length`: alignments summing to 750 samples with `total_samples=1000` give a final segment ending at 1000/sr
- `test_merge_and_min_hold`: `[("p",300),("a",4000),("a",1000),("t",100),("o",3000)]` at 22050 Hz gives shapes `["A","D","E"]` (the 100-sample `t` is absorbed)
- `test_insert_b_between_a_and_d`: `[("m",3000),("a",3000)]` gives `["A","B","D"]` with B lasting ≥ 1 frame
- `test_hebrew_phonemes`: `ʁ` and `χ` map to `B`
- `test_bus_viseme_roundtrip` and `test_bus_old_message_defaults_x`
- `test_controller_forwards_visemes`: a fake TTS returns a track `[0:"X", 0.1:"D"]`; at hop 5 (t=0.15 s incl. lead) the face receives `(SPEAKING, level, "D")`
- `test_controller_rms_fallback_when_no_track`
- `@slow test_piper_alignments_real`: the real English Piper voice with `include_alignments=True` returns a track covering ≥ 90% of the audio duration

Commit: `feat: viseme timeline from piper alignments`

### Task 15: Face asset bank format and loader

**Files:** Create `mirror/face/bank.py` and `assets/face/README.md` (the format spec). Test `tests/test_face_bank.py` with a tiny synthetic bank built in `tmp_path`.

**Format** (`<bank>/manifest.json` plus PNG/NPY files, all 252×252 coordinates):
```json
{"size": 252,
 "hub": "hub.png", "mask": "mask.png",
 "mouth": {"box": [x, y, w, h], "patches": {"A_mild": "mouth/A_mild.png", "...": "..."}},
 "eyes": {"box_left": [...], "box_right": [...], "patches": {"open": "...", "half": "...", "closed": "...", "look_up": "..."}},
 "expressions": {"smirk": "expr/smirk.png", "squint": "...", "brows_up": "..."},
 "warps": {"jaw": "warps/jaw.npy", "brows": "...", "breathe": "...", "gaze_x": "...", "gaze_y": "..."},
 "idle_loops": ["loops/idle1.npy"]}
```
Patches are RGBA at their box size, alpha feathered. Expressions are full-size RGBA. Warps are float32 `(252, 252, 2)` displacement fields in pixels at weight 1.0. Loops are uint8 `(N, 252, 252, 3)`.

**Interfaces:** `@dataclass FaceBank` (fields mirror the manifest, loaded as numpy arrays) and `load_bank(path: str) -> FaceBank`. Missing mouth patches fall back: `X_*` → hub (no patch), `E/F/G/H` → `C`, `wide` ↔ `mild`. Missing eye patches fall back to `open`. It raises `BankError` with the file name when a referenced file is missing or a size is wrong.

**Tests:** `test_load_minimal_bank` (hub + mask + one mouth patch loads), `test_mouth_fallbacks` (`E_wide` resolves to `C_mild` when only that exists), `test_bad_size_raises`, `test_missing_file_names_file`.

Commit: `feat: face asset bank format`

### Task 16: Sprite compositor renderer

**Files:** Create `mirror/face/sprite.py`. Modify `mirror/face/app.py` (choose the renderer by `config.face.renderer`, pass the viseme) and `mirror/config.py` + both TOMLs (`renderer = "sprite"`, `bank = "assets/face/stub"`; the existing tests are updated). Test `tests/test_face_sprite.py`.

**Interfaces:** `class SpriteFace(bank: FaceBank, rng: random.Random | None = None)` with `update(dt: float, state: State, mouth: float, viseme: str) -> None` (advances blink, breathing, gaze and expression springs) and `render() -> np.ndarray` (uint8 252×252×3). `app.py` blits `render()` through `pygame.surfarray`.

**Behaviour:**
- **Base:** the hub, or the current idle-loop frame if loops exist (ping-pong at 15 fps).
- **Warp:** one `cv2.remap` per frame. The grid is identity plus Σ weight·field. The mask is warped with the same grid and multiplied in, `borderValue=0`.
- **Weights:**
  - jaw = clamp(mouth, 0, 1) × 0.6 (rigid hinge, so ≤ 12 px at the field's scale)
  - breathe = 0.5 + 0.5·sin(2π t / 4.5 s)
  - gaze_x/gaze_y: idle wander every 2–5 s; listening centred; thinking (0.6, −0.7)
  - brows: listening 0.8, thinking 0.4, else 0
- **Patches:** alpha-blend the eyes (blink every 2–6 s at random over 150 ms: open → half → closed → half → open; thinking uses `squint` if present), the expression (idle: `smirk` fades in at random every 20–40 s for 2 s), and the mouth patch for `viseme` + (`wide` if mouth > 0.5 else `mild`).
- **Mouth crossfade:** 2 frames between shapes. An unchanged viseme keeps its patch with no crossfade.
- **State changes:** all weights move with a spring (time constant 120 ms).

**Tests** (stub bank from Task 17's generator, or a synthetic one):
- `test_render_shape_dtype`
- `test_background_stays_black` (corners 0)
- `test_mouth_patch_changes_mouth_region` (`D` vs `X`: mean abs diff in the mouth box > 10)
- `test_blink_closes_eyes` (force blink: the eye box gets darker)
- `test_jaw_warp_moves_chin` (mouth 1.0 shifts the lit-pixel centroid in the lower third down by ≥ 2 px)
- `test_crossfade_two_frames` (a frame between X and D differs from both)
- `test_render_under_5ms` (median of 100 renders < 5 ms on the dev Mac; skipped in CI if `MIRROR_SKIP_PERF` is set)

Commit: `feat: sprite compositor face`

### Task 17: Bank builder tools, stub bank and asset guide

**Files:** Create `tools/build_bank.py`, `tools/stub_bank.py`, `docs/FACE_ASSETS.md` and `assets/face/stub/` (generated, committed). Test `tests/test_build_bank.py`.

- `tools/stub_bank.py SOURCE.png OUT_DIR`: crops the face (centre square by default, `--box x,y,w,h` to override) and scales to 252. It makes the mask by luminance threshold plus feather, generates the warp fields with moving-least-squares from a default 20-point face layout scaled to the image (`--points points.json` to override), and synthesizes mouth patches procedurally: an A–D ladder by warping the jaw region open and painting the cavity black, with E/F by horizontal squeeze of C. Eye half/closed come from a lid warp plus darkening. It writes `manifest.json`. Purpose: a working bank today from any single image. Run it on the user's reference image `assets/face/source/trickster.png` (already front-facing on pure black; use the full frame, centred square).
- `tools/build_bank.py RAW_DIR OUT_DIR`: the real pipeline for LivePortrait exports. `RAW_DIR` holds `hub.png`, `lip_00..lip_04.png`, `eye_00..eye_03.png`, `shape_E.png` … `shape_H.png` and `expr_*.png`, all full frames. It aligns each frame to the hub (ECC or phase correlation, sub-pixel), downscales to 252, cuts the mouth/eye boxes with feathered masks, matches each patch's colour mean/std to the hub, paints the mouth interior (pixels darker than a threshold inside the mouth box) pure black, maps the lip ladder to `A,B,C,D` mild/wide, generates the warp fields as in the stub, and writes `manifest.json`.
- `docs/FACE_ASSETS.md`: the step-by-step guide for the user. Generate the front-facing source image (with prompt text), rent a GPU on RunPod, run ComfyUI-LivePortraitKJ with the exact parameter sweeps, record the webcam driving clips for E/F/G/H and the expressions, export into `RAW_DIR` naming, then run `build_bank.py` and switch `config.face.bank`.

**Tests:** `test_stub_bank_roundtrip` (generate from a synthetic face image into `tmp_path`, `load_bank` succeeds, every A–D mild/wide patch exists, warps have the right shape), `test_build_bank_aligns` (a shifted copy of the hub as `lip_00` comes out aligned within 1 px), `test_black_cavity` (the darkest region inside the mouth box is exactly 0 after the build).

Commit: `feat: face bank tools, stub bank and asset guide`

**Order:** 14 (after 13 lands), then 15 → 17 → 16 (16's tests use 17's stub generator), then the remaining Task 12 (Orin deploy).

---

### Task 18: Filler lines while thinking, and the reply streamed sentence by sentence

**Why:** the reply now starts ~4.3 s after the visitor stops talking. Two changes make the mirror feel immediate. (1) As soon as a question is understood, it speaks a short pre-recorded filler ("Mmmm… now that is an interesting thing to say…") while the LLM works. (2) The LLM reply is streamed, and the first sentence is synthesized and spoken while the rest is still being generated.

**Files:** Create `mirror/fillers.py`. Modify `mirror/llm.py` (streaming), `mirror/conversation.py` (filler + streamed playback), `mirror/persona.py` (filler copy) and `mirror/voice_main.py` (pre-synthesize fillers at startup). Tests: `tests/test_fillers.py`, `tests/test_llm.py`, `tests/test_conversation.py`.

**Filler copy** (`persona.FILLERS: dict[str, list[str]]`, in the oracle's voice, ≤ 2.5 s spoken each):
- en: "Mmmm… now that is an interesting thing to ask.", "Hmm. The mists are parting…", "Ahh. I have seen this question before, in a dream.", "Patience. The glass is turning.", "Mmm, the stars are whispering…", "Well, well… the threads are moving.", "Interesting. The answer is near.", "Hmmm… let me look beyond the glass."
- he: "מממ… זו באמת שאלה מעניינת.", "הממ. הערפל מתפזר…", "אהה. את השאלה הזו כבר ראיתי, בחלום.", "סבלנות. הזכוכית מסתובבת.", "מממ, הכוכבים לוחשים…", "טוב, טוב… החוטים זזים.", "מעניין. התשובה קרובה.", "הממ… תנו לי להביט מעבר לזכוכית."

**Persona change (user decision):** the mirror is an **oracle**. Rewrite `persona.SYSTEM_PROMPT` in both languages: you are the oracle that lives in the mirror at Midburn; you answer mysteriously and playfully, in riddles, omens, half-prophecies and metaphors (stars, mists, threads, reflections), always with a wink and warmth; you never give a flat, literal answer, but the visitor should feel there is real meaning in it; never cruel, never frightening, never predicts death, illness or harm; never reveal you are an AI or a computer; 1–3 short spoken sentences; no emojis, lists or stage directions; reply only in the visitor's language; Hebrew addresses the visitor in gender-neutral plural forms, and the oracle speaks of itself in masculine first person. Update `FALLBACK` to oracle style: en "The mists are too thick right now. Ask me again in a moment.", he "הערפל סמיך מדי כרגע. שאלו אותי שוב עוד רגע.". Update `tests/test_llm.py` assertions that pin the old fallback text. Also update the persona line in `MIRROR_CONTEXT.md`.

**Interfaces:**
- `class FillerBank(tts: TTSRouter, lines: dict[str, list[str]], rng: random.Random | None = None)`. `prepare()` synthesizes every line at startup via `tts.synth_visemes` and caches `(samples, sr, track)`. `pick(lang) -> tuple[np.ndarray, int, VisemeTrack | None] | None` picks at random, never the same line twice in a row per language, and returns None if nothing is cached for `lang`.
- `LlamaServerLLM.reply_stream(history, user_text, lang) -> Iterator[str]` POSTs with `"stream": true`, parses the SSE `data:` lines (`choices[0].delta.content`), and yields **complete sentences** (split on `(?<=[.!?…])\s+`, final remainder flushed at the end). It stops after `max_sentences`, cuts at `max_chars` total, and strips stage directions and emoji per sentence with the same rules as `trim_reply`. It raises `LLMError` on HTTP or timeout errors before the first sentence. After the first sentence, a mid-stream error ends the stream quietly (log WARNING). `reply()` stays as is.
- Controller answer flow (replaces the single `llm.reply` + one `synth` + one `play`):
  1. → THINKING. Start the LLM stream in a worker thread that pushes sentences into a `queue.Queue` (sentinel at the end, exception object on failure).
  2. If `fillers.pick(lang)` returns a clip and `cfg.filler_enabled`: → SPEAKING, play it (visemes forwarded as usual), then → THINKING if no sentence is ready yet.
  3. For each sentence from the queue: `synth_visemes` → SPEAKING → play. Synthesize sentence n+1 while sentence n plays (a one-deep synth-ahead in a second worker), so there is no gap between sentences. Append the full joined reply to history at the end.
  4. An `LLMError` before any sentence (raised from the queue) leads to the existing FALLBACK path. A timeout waiting for the first sentence (`servers.timeout_s`) is treated the same.
  5. Segments arriving during all of this are still dropped (busy flag unchanged), and no exception escapes `on_segment`.
- Config: `conversation.filler_enabled = true` (both TOMLs) and `conversation.filler_min_wait_s = 0.0`. The filler only plays if no sentence has arrived within `filler_min_wait_s` after THINKING starts, so keep 0.0 by default (always play).
- `voice_main`: build `FillerBank(tts, persona.FILLERS)` and call `prepare()` after TTS warmup. Log how long it took.

**Tests (names + assertions):**
- `test_pick_never_repeats`: 50 picks with 3 lines never repeat back to back
- `test_pick_unknown_lang_none`
- `test_reply_stream_yields_sentences`: fake SSE chunks `"Mirror"`, `", mirror. Who"`, `" is fair? You"`, `" are."` yield `["Mirror, mirror.", "Who is fair?", "You are."]`
- `test_reply_stream_limits`: stops after 3 sentences and respects `max_chars`
- `test_reply_stream_error_before_first_raises` and `test_reply_stream_error_after_first_ends_quietly`
- `test_filler_plays_before_reply`: the player's call order is filler clip, then sentence 1, then sentence 2. Lights go THINKING → SPEAKING (filler) → … → LISTENING
- `test_filler_disabled_skips`
- `test_first_sentence_synth_while_second_generates`: with a fake stream that blocks before sentence 2, sentence 1 is played before sentence 2 exists
- `test_stream_failure_before_first_sentence_uses_fallback`: the FALLBACK text is spoken, then IDLE
- `test_history_gets_joined_reply`
- Manual: a `mirror-voice --wav` smoke in English and Hebrew reports end-of-speech → first audio (the filler) and → first reply sentence. Target: filler < 1.2 s on the Mac.

Commit: `feat: filler lines and streamed replies`

**Order:** after Task 14; before 15–17.

---

### Task 19: Latency and Hebrew tuning

**Why:** the Task 18 smoke on the 8 GB Mac had the first reply sentence 5–7 s after end of speech. llama-server re-read the ~800–1,000-token oracle prompt (1–1.6 s) because requests landed on fresh slots and the cache went cold after idle. The 4B model also slips into singular Hebrew address.

**Files:** `mirror/llm.py`, `mirror/persona.py`, `mirror/conversation.py`, `mirror/voice_main.py`, `Makefile` + `deploy/systemd/mirror-llm.service` (server flags), tests in `tests/test_llm.py`, `tests/test_conversation.py`.

- **Slot pinning:** run llama-server with `-np 2 --cache-reuse 256`. Every request sends `"id_slot": 0` for `en` and `1` for `he`, plus `"cache_prompt": true`. Make the slot map a module constant.
- **Warmup:** at startup, after TTS warmup, `LlamaServerLLM.warm(lang)` for both languages: one request with the full system prompt, the user text "." and `max_tokens: 1`. Log the prompt-eval time from the response's `timings.prompt_ms`. Re-warm a language's slot if more than 10 minutes have passed since its last request, in a background thread from `tick()`, never while busy.
- **Stage directions across sentences:** in the streaming splitter, don't split while the buffer has an unclosed `*`, `[` or `(`. `_clean` also strips stray lone `*[]()`. Test: `"*leans in. whispers* The stars know. [a long pause. then] Yes."` gives `["The stars know.", "Yes."]`.
- **Mid-reply stall:** if no sentence clip is ready within 0.6 s after the previous one finished, go to THINKING (face and lights) until the next clip arrives. Test with a fake stream that stalls.
- **History on a mid-reply failure:** sentences already spoken are added to history even if TTS or playback fails later in the reply.
- **Hebrew address:** add 3 short Hebrew few-shot example pairs (user question → oracle answer) to the Hebrew system prompt, each using plural address ("אתם", "שאלו", "לכם") and correct gender agreement. Keep them inside the cached prefix (before the world text). Test that the Hebrew prompt contains them and that the English prompt does not.
- **Test hygiene:** filter the torch `jit.script` FutureWarning in pytest config so test output is pristine.
- **Smoke:** repeat Task 18's English and Hebrew `--wav` runs twice each (warm) and report end-of-speech → filler and → first sentence, plus `timings.prompt_ms` per request. Target: first sentence ≤ 3.5 s warm on the Mac. Report honestly if swap prevents it.

Commit: `perf: pinned prompt cache, warmup, stall handling, Hebrew examples`

---

### Task 20: RenikudPlus as the Hebrew pronunciation engine

**Why:** [RenikudPlus](https://github.com/maxmelichov/RenikudPlus) (Apache-2.0, ONNX, model `notmax123/RenikudPlus`) reports 86.2% overall word accuracy against Phonikud's 66.4%, including gender 99.3% against 37.5%. Hebrew text→IPA is exactly the first stage of our Hebrew voice. It also takes a speaker-gender hint (0 unknown, 1 male, 2 female). The oracle is masculine, so use 1.

**Files:** `mirror/tts.py` (G2P backend selectable inside the Hebrew voice), `mirror/config.py` + both TOMLs (`tts.he_g2p = "renikud" | "phonikud"`, `tts.renikud_model`), `deploy/fetch_models.sh` (pinned download), `pyproject.toml` (`renikud-plus[cpu]` pinned), `tests/test_tts.py`.

- [ ] **Step 1: Phoneme compatibility.** Compare RenikudPlus's output symbol inventory on ~200 varied Hebrew sentences (persona, fillers, oracle replies from logs, numbers) with the Hebrew Piper voice's `phoneme_id_map`. Report unmapped symbols. If any exist, add a small explicit symbol-mapping table (e.g. different stress or affricate notation), with tests. If too many symbols are unmappable, STOP and report — the voice would need retraining.
- [ ] **Step 2: Implement** `RenikudG2P(model_path, speaker=1)` with `phonemize(text) -> str`, used by the Hebrew voice when `he_g2p = "renikud"`. Keep Phonikud selectable. Load lazily, once, on CPU, using the int8 model by default if quality holds.
- [ ] **Step 3: A/B listening set.** Synthesize the same 15 Hebrew lines (all 8 Hebrew fillers, the fallback, and 6 typical oracle replies) with both G2Ps into WAVs, and write a side-by-side page listing each line's IPA from both. Measure G2P time per sentence and RSS.
- [ ] **Step 4: Tests:** a fast stub test for the backend switch, plus a slow real-model test (skipped without the model) asserting non-empty IPA made only of mapped symbols.
- [ ] **Step 5: Default.** Set `he_g2p = "renikud"` only if Step 3 shows no regressions in a fluent read of the IPA and listening. Otherwise leave Phonikud as the default and report.

Commit: `feat: RenikudPlus Hebrew G2P option`

---

### Task 21: Fix the English prompt-cache miss and harden Hebrew address

**Why (from the Hebrew LLM bake-off, 2026-10-18):** with Gemma 3, llama-server turns off `cache_reuse` because of sliding-window attention. The English slot then re-processes ~515 of its 764 prompt tokens every turn (1.5–3.3 s on the Mac). Separately, Gemma 3 4B slips into singular Hebrew address in 3–5 of 10 replies, mostly when the visitor speaks in the singular or feminine. The current few-shot examples are all gender-neutral. The bake-off kept Gemma 3 4B as the model (best oracle tone; Gemma 4 E2B is the runner-up, so make the model file a config value).

**Files:** `Makefile`, `deploy/systemd/mirror-llm.service`, `mirror/persona.py`, `mirror/config.py` + TOMLs (`servers.llm_model` already exists? if not, add it), `tests/test_llm.py`.

- [ ] **Step 1:** Add `--swa-full` to llama-server in both the Makefile and the service file. Measure on the Mac with a short script that sends two consecutive English requests with the full system prompt through `LlamaServerLLM`, and read `timings.prompt_n` on the second. Expected: the second English request re-processes ≤ 20 tokens. Record llama-server's reported KV memory with and without the flag. If the memory increase is above 600 MB at `-c 8192 -np 2`, use `-c 6144` instead and re-measure.
- [ ] **Step 2:** Add 2 Hebrew few-shot pairs where the **question** is in the singular (one masculine, one feminine, e.g. "מה מחכה לי השנה?" asked by a woman), and the oracle **answer stays in the gender-neutral plural** with correct agreement. Keep the anti-parroting header. Test: the Hebrew prompt contains the new examples, and every example answer passes a simple singular-marker check (no standalone את/אתה/לך/שלך).
- [ ] **Step 3:** A plural-address check on generated replies is out of scope (no live LLM in tests). Instead, add `tools/hebrew_address_check.py`, a small script that runs N Hebrew questions against the running llama-server and reports the singular-marker rate, so it can be re-run on the Orin.

Commit: `perf: full SWA cache for English slot; singular-question Hebrew examples`

---

### Task 22: Hebrew-tuned speech recognition (ivrit-ai Whisper turbo) with script-based language detection

**Why (STT experiment, 2026-10-18; `scratchpad/stt/stt.md`):** on 38 synthetic clips, ivrit-ai's Whisper large-v3-turbo fine-tune caught 15–16/16 Hebrew wake phrases (whisper small 8/16) and cut Hebrew WER from 58% to 18%. But it labels English as Hebrew, and sometimes transliterates it ("מיר מיר"). English WER was 27%. Picking the language from the transcript's script scored he 26/26 and en 10/12. With `-ac 768` (15.36 s audio context, matching the VAD's 15 s cap), latency is 1.84 s on the Mac against small's 0.76 s. RSS is about the same (649 vs 631 MB).

**Decision (controller ruling R27):** the Midburn crowd is overwhelmingly Hebrew-speaking, and a missed wake phrase is the worst failure, so ivrit q5_0 becomes the default. Small stays one config line away.

**Files:** `deploy/fetch_models.sh` (download ivrit-ai's official ggml f16 at a pinned revision and quantize to q5_0 with `whisper-quantize`; skip if present), `Makefile` + `deploy/systemd/mirror-stt.service` (`STT_MODEL` env/variable, `-ac 768`), `mirror/stt.py` (language from script), `mirror/wake.py` (matcher tweaks), `tests/test_stt.py`, `tests/test_wake.py`, README (models list, how to switch back).

- [ ] **Language from script:** `script_language(text) -> "he" | "en" | None` counts Hebrew letters (U+05D0–U+05EA) against Latin letters: more Hebrew means `he`, more Latin means `en`, and none means `None` (fall back to the server's language field, mapped as today). `WhisperServerSTT.transcribe` uses the script decision first. Tests: Hebrew text gives he, English gives en, mixed with a Hebrew majority gives he, digits only gives the fallback.
- [ ] **English replies when the transcript is transliterated:** if `script_language` says `he` but the server reported "english", trust the script, since that means it was transcribed into Hebrew letters. Document this. No other heuristics.
- [ ] **Wake matcher:** accept the transliterations "מיר"/"מירור"/"מירר" as English-mirror tokens. Lower the per-token similarity threshold for "mirror" to 0.7 (keep the "the mirror is nice" test negative). Allow a third repeated wake token ("mirror mirror mirror …", "מראה מראה מראה") and strip it from the remainder. Tests: "mirer mirror, what…" matches; "מיר מיר, מה…" matches; "mirror mirror mirror who is fairest" gives remainder "who is fairest"; all existing negatives stay negative.
- [ ] **Server config:** `-ac 768` plus `STT_MODEL` defaulting to `ivrit-turbo-q5_0.bin` in both launchers. Add a test that both launchers carry the same STT flags (like Task 21's LLM test).
- [ ] **Smoke:** re-run the experiment's harness (`scratchpad/stt/`) against the new defaults, through the app's `WhisperServerSTT` + `match_wake`. Report wake hit rate he/en, language accuracy and latency. Expected: he wakes ≥ 15/16, en language ≥ 10/12.

Commit: `feat: ivrit-ai Whisper turbo STT with script-based language detection`

---

### Task 23: Pocket TTS as an optional voice engine (off by default)

**Why (Pocket TTS experiment, 2026-10-18; `scratchpad/pocket/pocket.md`):** [Pocket TTS ONNX](https://huggingface.co/thewh1teagle/pocket-tts-onnx) is one CC-BY-4.0 model that speaks English and Hebrew (fed RenikudPlus IPA), with built-in voices (javert CC0, deepest; michael CC-BY) and offline voice cloning to a 0.77 MB `.npy`. Intelligibility matched the current stack, and peak RAM was about 800 MB against about 830 MB. It's slower: M2 RTF 0.16–0.18 at 2 threads, an estimated 0.33–0.45 on the Orin. It gives no phoneme timings, but a RenikudPlus/espeak IPA + energy-envelope estimate matched 55% of frames and 77% of lip closures. Known defect: short sentences sometimes stop after the first word, depending on the seed. Default stays Piper + Phonikud voice + RenikudPlus G2P until it's measured on the Orin and listened to.

**Files:** `mirror/tts.py` (new `PocketTTS` Synth with `synth_visemes`), `mirror/visemes.py` (`estimate_track(ipa, samples, sr)`), `mirror/config.py` + TOMLs (`tts.engine = "piper" | "pocket"`, `tts.pocket_model`, `tts.pocket_voice = "javert"`, `tts.pocket_voice_file` optional cloned `.npy`, `tts.pocket_threads = 2`), `mirror/voice_main.py` (build the router by engine, with fallback to piper on load failure and a WARNING), `deploy/fetch_models.sh` (pinned download, ~232 MB), `pyproject.toml` (pin the package; it claims Python ≥3.13 but runs on 3.10, so install it the same way as renikud-plus), `tools/clone_voice.py` (offline: reference WAV → `.npy`), tests.

- [ ] `PocketTTS(model_path, voice, voice_file=None, threads=2)` implements `synth(text)` and `synth_visemes(text, lang)`. Hebrew text goes through the existing `RenikudG2P` to IPA. English uses the model's own path, with espeak IPA only for the viseme estimate if available (else RMS fallback). Lazy imports, CPU only.
- [ ] **Truncation guard:** if the output duration is < 45% of an expected duration (phoneme count × 0.06 s, or characters × 0.055 s), re-roll with a new seed, at most 2 times, then accept the longest. Test with a fake model that truncates on the first call.
- [ ] `estimate_track`: distribute the IPA's phonemes over voiced regions from a 20 ms energy envelope, map them through `PHONEME_TO_SHAPE`, then apply the existing `track_from_alignments` hold rules. Tests: closures (m/b/p) appear as A; the track spans the audio; silence gives X.
- [ ] **Fillers:** when engine = pocket, check each filler clip's duration at prepare time and log a WARNING for any > 2.5 s.
- [ ] `tools/clone_voice.py REF.wav OUT.npy` runs offline (the encoder needs ~1.5 GB, so it's for the Mac, not the Orin). README section on choosing or cloning a voice, the consent note (Kyutai's terms forbid cloning without consent), and switching engines.
- [ ] Smoke on the Mac, with no servers needed: synth plus estimated visemes for 3 English + 3 Hebrew lines with engine = pocket, saving WAVs and printing the RTF.

Commit: `feat: optional Pocket TTS engine with estimated visemes`

---

### Task 24: systemd watchdog for the voice and face processes

**Why:** systemd restarts a crashed process, but not a hung one. If the voice loop or the face loop wedges (an audio or driver deadlock), the mirror goes silent or frozen for the rest of the night.

**Files:** `mirror/watchdog.py` (new), `mirror/voice_main.py`, `mirror/face/app.py`, `deploy/systemd/mirror-voice.service`, `deploy/systemd/mirror-face.service`, tests.

- `watchdog.notify(msg: str) -> bool` sends to the `$NOTIFY_SOCKET` datagram socket (abstract `@` names included) with stdlib `socket` only, no new dependency. It returns False, a no-op, when the variable is unset (the Mac). `ready()` sends `READY=1`; `ping()` sends `WATCHDOG=1`, rate-limited to at most once per `WATCHDOG_USEC/4` (read from the env; default 15 s).
- **voice:** `ready()` once after "ready", and `ping()` from the main loop on every tick. A main loop stuck in on_segment for more than the timeout gets restarted. That's acceptable, since a single answer is bounded far below 60 s by the existing budgets.
- **face:** `ready()` after the first frame, `ping()` each frame (rate-limited).
- **Units:** `Type=notify`, `WatchdogSec=60`, `NotifyAccess=main`; keep `Restart=always`. Make sure `TimeoutStartSec` covers model loading (voice already has 200 s).
- **Tests:** notify to a real temporary AF_UNIX datagram socket (path form) receives `READY=1`/`WATCHDOG=1`; unset env gives a no-op; the rate limit holds; voice_main calls ready/ping (with a fake watchdog); unit files contain the new keys.

Commit: `feat: systemd watchdog pings from voice and face`

### Task 25: Idle lures — the oracle beckons when no one is around

**Why:** a lone mirror on the playa needs to draw people in. When it has been idle for a while, it murmurs a short pre-recorded line, in either language at random, with the face speaking it.

**Files:** `mirror/persona.py` (`LURES`), `mirror/fillers.py` (reuse FillerBank for lures, or a sibling `LureBank`), `mirror/conversation.py` (a lure in `tick()` when idle), `mirror/config.py` + TOMLs (`conversation.lures_enabled = true`, `lure_min_idle_s = 120`, `lure_interval_s = [180, 420]`, `lure_quiet_hours = []` as a list of "HH:MM-HH:MM" local-time windows when lures are off; default empty), tests.

- **Copy (oracle voice, ≤ 3 s each):** en "Come closer, traveller… the glass has something to show you.", "I see you there. Ask, and the mirror will answer.", "Mirror, mirror… say the words, and I will wake.", "The dust remembers you. Come, ask me anything."; he "התקרבו, נוסעים… לזכוכית יש משהו להראות לכם.", "אני רואה אתכם. שאלו, והמראה תענה.", "מראה, מראה… אמרו את המילים, ואתעורר.", "האבק זוכר אתכם. בואו, שאלו אותי הכול."
- **Controller:** in `tick()`, when state is IDLE, not busy, lures are enabled, now is outside quiet hours, idle time ≥ `lure_min_idle_s`, and the randomized interval has elapsed since the last lure or interaction, play a lure. It goes through the same play path as a filler (SPEAKING with visemes, then back to IDLE), uses the busy flag so segments are dropped while it speaks, never repeats the same line twice in a row, and alternates languages randomly. Any interaction resets the idle timer. Lure playback must not change the conversation history.
- **Tests:** no lure before min idle; lure after min idle + interval; none during quiet hours (including windows crossing midnight); none while busy or not IDLE; a speech segment after a lure is handled normally; history unchanged; an injectable clock throughout.

Commit: `feat: idle lures — the oracle beckons passers-by`

**Order:** 24 then 25 (both touch voice_main.py / conversation.py).

---

### Task 26: Varied oracle answers (and keep words inside parentheses)

**Why (live demo, 2026-10-19):** the user found the answers repetitive. Most replies reuse the same imagery ("shimmer", "thread of silver", "ochre", "violet", "star", "dust remembers"), because the persona prompt lists those images and every request carries identical instructions. The 4B model on the Orin should be better than the Mac's 1B, but both need a variety signal. Separately, the stage-direction filter removed words the model put in parentheses mid-sentence ("it doesn't (hold) a pattern. It simply (is).") and left broken sentences.

**Files:** `mirror/persona.py`, `mirror/llm.py`, `mirror/conversation.py` (pass a per-request angle), `config/*.toml` + `mirror/config.py` (sampling), `tools/variety_check.py` (new), tests.

- **Angle per answer.** `persona.pick_angle(rng, recent: list[str]) -> Angle(mode, imagery)`.
  - `mode` is one of riddle, prophecy, playful dare, tiny story, practical advice in disguise, question back, cheeky compliment, or blunt truth with a wink.
  - `imagery` is one of ≥ 25 image worlds: fire, water, birds, flamingos, music and drums, bicycles, art cars, food and tea, colours, numbers, animals, weather and wind, the Temple, the Man, shoes, maps, keys, clocks, seeds, salt, honey, moths, lanterns, kites, sand dunes, the moon's phases, and more.
  - Avoid repeating the last 3 modes and the last 5 image worlds (kept in the Controller per conversation, and across conversations for a short while).
  - The angle is appended as a short hidden instruction at the END of the user message, in the visitor's language (Hebrew translation for he), e.g. "(Answer as a playful dare, using images of bicycles. Start by answering their actual question.)". The system prompt stays byte-identical, so the prompt cache stays warm.
- **Persona prompt:** remove the fixed imagery list (stars, mists, threads, reflections). Add: answer the visitor's actual question in the first sentence; never reuse an image or phrase from your previous replies; avoid stock words (shimmer, thread, whisper, ochre, violet, tapestry, echo, dust remembers) unless they truly fit. Same for Hebrew. Keep the oracle's mysterious, playful, never-cruel character and length limits. Re-check the few-shot examples so they don't model the overused images.
- **Sampling** in the config, sent with each request: `temperature 0.95`, `top_p 0.95`, `min_p 0.05`, `repeat_penalty 1.12`, `presence_penalty 0.4`, `frequency_penalty 0.3`. Check against llama-server v0.7.0's /v1/chat/completions accepted fields, and set them in config for both TOMLs.
- **Parentheses fix:** `_clean` removes a parenthetical only when it is a stage direction: it stands alone as the whole sentence, or sits at the start or end of a sentence and looks like a direction (≤ 4 words, lowercase or adverb-like such as "softly", "laughs", "leans in"). A mid-sentence parenthetical keeps its words without the brackets. Asterisk and bracket directions are unchanged. Tests: "it doesn't (hold) a pattern." → "it doesn't hold a pattern."; "(softly) Look up." → "Look up."; "Look up. (pause)" → "Look up."; Hebrew equivalents.
- **`tools/variety_check.py`:** run N=20 requests (10 different burner questions × 2, and 5 repeats of the same question) against the running llama-server with the full pipeline's prompt plus angle. Report distinct-1 and distinct-2 ratios across replies, the most frequent content words, and the share of replies containing any stock word. A `--no-angle` switch gives a before/after comparison. Unit-test the metrics with fixed strings.
- **Tests:** angle choice is deterministic with a seeded rng and respects the no-repeat windows; the user message carries the angle while the system prompt is unchanged between two requests; sampling fields are present in the request body; the parentheses cases above.

Commit: `feat: varied oracle answers (angles, sampling); keep mid-sentence parentheticals`
