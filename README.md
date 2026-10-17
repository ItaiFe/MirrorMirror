# Mirror, Mirror

Software for the talking LED mirror. Spec: `MIRROR_CONTEXT.md`.

## Mac dev loop

```sh
uv venv --python 3.11 .venv && uv pip install --python .venv -e '.[dev,hw,tts]'
make models        # deploy/fetch_models.sh models (whisper small, Gemma 3 4B, Silero, Piper, MMS cache)
make test
make servers-mac   # just whisper-server :8081 + llama-server :8080 (Ctrl-C stops both)
make dev           # servers in background (logs in /tmp/mirror-dev), wait for /health, then face + voice
```

`make dev LLM_MODEL=models/other.gguf` swaps the LLM. The face window is 756x756 (252 x 3).

`mirror-voice --config config/mac.toml --wav FILE.wav` feeds a WAV through VAD and the
conversation controller exactly like mic audio (then 1 s of silence), plays the reply and exits.

## Measured on a MacBook Air M2, 8 GB (2026-10-17)

End of speech to first reply audio, from the `--wav` smoke runs (includes ~0.7 s of VAD
hang-over silence; the whole reply is synthesized before playback starts):

| Input | LLM | STT | LLM reply | TTS | End-of-speech -> first audio |
|---|---|---|---|---|---|
| English (Piper WAV) | Gemma 3 4B Q4_K_M | 0.9-1.6 s | 2.9-3.9 s | Piper ~1.2-1.6 s | 6.5-6.7 s |
| Hebrew (MMS WAV, vocalized) | Gemma 3 4B Q4_K_M | 1.5-2.1 s | 4.2-5.0 s | MMS ~7-9 s | 13.3-16.6 s |
| English | Gemma 3 1B Q4_K_M | 0.85 s | 0.3 s | 0.16 s | 1.9 s |
| Hebrew (vocalized) | Gemma 3 1B Q4_K_M | 0.8 s | 1.9 s | MMS 3.7 s | 6.8 s |
| English, live via `make dev` (speaker -> mic) | Gemma 3 4B | 4.3 s | 6.1 s | 3.1 s | 14.0 s |

The 8 GB Mac was swapping with all three models loaded, so these numbers are pessimistic;
the Orin numbers are what count. Over the 2-4 s target: MMS Hebrew synthesis on CPU and
synthesizing the whole three-sentence reply before playback are the main costs.

Notes:
- whisper-server reports the language as a full name (`"language": "hebrew"`).
- MMS reads unvocalized Hebrew poorly: "מראה מראה שעל הקיר" synthesized without niqqud was
  heard by whisper as "מה אמר אש על הקיר" (no wake). With niqqud it woke.
