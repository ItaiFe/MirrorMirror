# Training the oracle's Hebrew voice

Goal: a Piper (VITS) Hebrew voice that sounds like the oracle's ElevenLabs voice,
fine-tuned from a Phonikud Hebrew checkpoint. Compared with today's Hebrew (the cloned
Pocket voice speaking Hebrew through an English model):

- native Hebrew pronunciation, learned from Hebrew audio;
- about 5x faster and ~0.6 GB less memory on the Orin;
- exact lip sync again (Piper reports when each sound is spoken).

Two parts: generate the audio with ElevenLabs (steps 1-6), then
prepare the training data, fine-tune on a rented GPU and wire the voice in (step 7).

## The sentences

`data/voice_he/` holds 1,280 Hebrew sentences, ~53,000 characters, about an hour of
speech:

| File | Lines | What |
|---|---|---|
| `sentences_oracle.txt` | 320 | the oracle's own lines: prophecies, riddles, greetings, fillers (plural address) |
| `sentences_everyday.txt` | 320 | conversational Hebrew: questions, exclamations, both genders, all tenses |
| `sentences_story.txt` | 320 | descriptive and narrative sentences, longer clauses |
| `sentences_coverage.txt` | 320 | numbers and dates in words, Midburn words, loanwords, names, 100 short replies |

No niqqud, digits, Latin letters or abbreviations, so what ElevenLabs reads is what the
mirror's own G2P (RenikudPlus) turns into phonemes for training. To add sentences, add
lines to any file (or a new `sentences_*.txt`); existing ones keep their ids.

## 1. ElevenLabs account

- A plan with at least **~55,000 credits** left this month (the full set, plus a few
  retries). On `eleven_v3`, one credit per character at the time of writing; check your plan.
- Use **Eleven v3** (`eleven_v3`): it supports Hebrew. Multilingual v2 does not and
  reportedly produces unintelligible Hebrew.

## 2. Voice ID and API key

- **Voice ID**: ElevenLabs -> *My Voices* -> your oracle voice -> the `...` menu ->
  *Copy voice ID*.
- **API key**: ElevenLabs -> *Developers* -> *API keys* -> *Create key* (Text to Speech
  access is enough). Keep it private: don't paste it into chat or commit it.

In a terminal in the repo:

```sh
export ELEVENLABS_API_KEY='sk_...'      # yours; only this terminal sees it
export VOICE=your_voice_id
```

## 3. Check the numbers (free, no API calls)

```sh
.venv/bin/python tools/elevenlabs_dataset.py data/voice_he/sentences_*.txt \
    --voice-id $VOICE --out dataset_he --dry-run
```

It prints the sentence count, characters, the credit estimate and the audio estimate.

## 4. A test batch: 20 clips

```sh
.venv/bin/python tools/elevenlabs_dataset.py data/voice_he/sentences_*.txt \
    --voice-id $VOICE --out dataset_he --limit 20
open dataset_he/wavs
```

Listen to several. They should sound like the oracle: same voice, steady character,
clear Hebrew. Consistency matters more than drama for training: every clip should be
the same "performance". If they vary too much, raise `--stability` (v3 takes 0.0
creative, 0.5 natural, 1.0 robust); try `--stability 1.0` on a fresh `--out` folder and
compare. If your voice was designed on a newer model (`eleven_v4`, when your account
has it), try `--model eleven_v4` the same way and keep whichever sounds best in Hebrew.
**Use one model and one set of settings for the whole dataset.**

## 5. The full run

Same command without `--limit` (and with the settings you chose):

```sh
.venv/bin/python tools/elevenlabs_dataset.py data/voice_he/sentences_*.txt \
    --voice-id $VOICE --out dataset_he
```

- It skips clips that already exist, so you can stop it (Ctrl-C) and run it again.
- It retries rate limits and server errors, and stops cleanly if the key is wrong or
  the credits run out (top up, then run it again).
- At the end it writes `dataset_he/metadata.csv` (`id|text`) and prints the total minutes.

## 6. Quality check

```sh
.venv/bin/python tools/elevenlabs_dataset.py data/voice_he/sentences_*.txt \
    --voice-id $VOICE --out dataset_he --check
```

It flags clips that are very short, too fast or slow for their text, have long silences
or clip. With the speech server running (`make servers-mac`, or only whisper-server),
add `--stt-url http://127.0.0.1:8081` to also transcribe every clip and flag the ones
whose words don't match the text (misreads). Listen to the flagged clips; delete the
bad wavs and run step 5 again to regenerate just those.

## 7. Train the voice

### What the training is built on (research, October 2026)

- **The checkpoints.** [Phonikud/phonikud-tts-checkpoints](https://huggingface.co/Phonikud/phonikud-tts-checkpoints)
  (non-commercial licence) has Piper Lightning checkpoints, 846 MB each, and their ONNX
  exports. `model.onnx` there is byte-identical to our `models/phonikud-tts-he.onnx`
  (63,511,038 bytes, sha256 `dfe0a8f3…d968`, from the Hugging Face file metadata), and
  `model.config.json` is identical to our `.onnx.json`: `"phoneme_type": "raw"`, 22050 Hz,
  157 symbols in the phoneme_id_map, `num_symbols` 256, single speaker. Its checkpoint is
  the unsuffixed **`epoch=4842-step=3124000.ckpt`** (the suffixed ones go with
  `michael.onnx` and `shaul.onnx`, which differ in size and hash, and with `ranlevi`).
  That is the one to start from: the voice we run today was exported from it, so the
  phoneme ids, symbol inventory and sample rate are already the ones the mirror feeds it.
  (The ckpt-to-onnx pairing is inferred from the names and the matching hash of the
  export, not from re-exporting the 846 MB file.)
- **How they were trained.** The checkpoint's own header (read with a 2 MB range
  request, not a full download) says: PyTorch Lightning 1.7.7, i.e. the legacy
  [rhasspy/piper](https://github.com/rhasspy/piper) `piper_train` (archived; the paper
  cites it); `quality: medium` (the ~20M-parameter model; 63 MB ONNX like every Piper
  medium voice); 22050 Hz; batch size 24; learning rate 2e-4 with decay 0.999875 per
  epoch (1.09e-4 by epoch 4842); validation split 0; resumed from
  `epoch=4672-step=3107340.ckpt`. The config's key layout and `"piper_version": "1.0.0"`
  are the legacy preprocess output; `"raw"` is not one of its phoneme types, so the
  config was edited or a fork was used. No Phonikud training script or Piper fork is
  public: [phonikud/phonikud](https://github.com/phonikud/phonikud) (G2P) and
  [phonikud/phonikud-tts](https://github.com/phonikud/phonikud-tts) (inference) have
  none. The [paper](https://arxiv.org/abs/2506.12311) (section 5.2) says they fine-tuned
  Piper from pre-trained English checkpoints on ~20 h of synthetic Hebrew audio (made with
  Gemini), transcribed to IPA with Phonikud, for ~10 h on one RTX 4090 with default
  hyperparameters (High size in the paper's table; the released checkpoints are medium).
- **The trainer we use.** [OHF-Voice/piper1-gpl](https://github.com/OHF-Voice/piper1-gpl),
  the maintained Piper (`python -m piper.train`, Lightning 2), pinned at **v1.8.0**
  (commit `639388b6`). Its
  [TRAINING.md](https://github.com/OHF-Voice/piper1-gpl/blob/v1.8.0/docs/TRAINING.md)
  takes a CSV of exact phoneme ids (`id|text|0 1 2 …`, "the model will be trained with the
  exact phoneme ids you specify") and a phoneme/id map file (`--data.phonemes_path`), and
  fine-tunes from existing (legacy) Piper checkpoints. The docs call the flag
  `--data.data_type`; in the v1.8.0 code
  ([dataset.py](https://github.com/OHF-Voice/piper1-gpl/blob/v1.8.0/src/piper/train/vits/dataset.py))
  it is `--data.dataset_type phoneme_ids`. Its model defaults are the medium architecture,
  the same as the checkpoint's hyperparameters. v1.8.0 also has
  `--model.warmstart_ckpt` (copy all matching weights, fresh optimizer,
  [lightning.py](https://github.com/OHF-Voice/piper1-gpl/blob/v1.8.0/src/piper/train/vits/lightning.py)),
  which we use by default instead of a full `--ckpt_path` resume of a Lightning 1.7
  checkpoint. Export is `python -m piper.train.export_onnx --checkpoint … --output-file …`.
  Feeding exact ids means no Hebrew G2P or espeak runs on the GPU box at all.

### 7a. Prepare (on the Mac)

When the dataset is complete and checked (step 6):

```sh
.venv/bin/python tools/prepare_piper_dataset.py      # dataset_he/ -> train_he/
```

It phonemizes every sentence with the mirror's own Hebrew path (the G2P of `tts.he_g2p`,
RenikudPlus by default, then the same filter the voice uses at runtime,
`mirror.tts.keep_voice_symbols`) and turns the phonemes into ids with the checkpoint's
phoneme_id_map exactly as piper does at runtime. It skips clips flagged in
`dataset_he/check.tsv` (`--keep-flagged` keeps them), missing or wrong-format wavs, empty
phonemes, symbols outside the map, and sentences where the G2P dropped a word; it lists
them in `train_he/dropped.tsv` and prints a summary. `train_he/` (git-ignored) has
`metadata_ids.csv`, `metadata_phonemes.csv` (readable), `phoneme_id_map.json`,
`config.json` (the voice config, which becomes the new `.onnx.json`), `samples_ids.csv`
(five held-out sentences for listening) and `wavs/`.

If you later switch `tts.he_g2p`, retrain: the voice learns the G2P's habits.

### 7b. Rent a GPU and copy the data

Any CUDA box with ≥16 GB VRAM: an RTX 4090 or A6000/A100 on RunPod, Lambda or Vast
(Ubuntu + a PyTorch image), ~50 GB disk. A 4090 is ~$0.4-0.8/h; budget 2-5 h.

```sh
BOX=root@1.2.3.4            # with -p PORT / -i KEY as your provider says
rsync -av --progress train_he deploy/train_piper_hebrew.sh $BOX:piper-he/
```

### 7c. Train (on the box)

```sh
ssh $BOX
tmux new -s train
cd piper-he && ./train_piper_hebrew.sh setup     # packages, piper1-gpl v1.8.0, the 846 MB checkpoint (sha256-checked)
./train_piper_hebrew.sh train                     # fine-tune; rerun to resume after an interruption
```

The settings are variables at the top of the script (batch 32, learning rate 1e-4,
validation every 10 epochs, at most 1000 epochs; `START_MODE=resume` for a full
Lightning resume instead of the weights-only warm start). Roughly 38 steps per epoch;
expect ~100 epochs per hour or less on a 4090 (read the real rate off the progress bar).

**Listening while it trains**, from a second tmux window (`Ctrl-b c`):

```sh
./train_piper_hebrew.sh sample        # exports the newest checkpoint, synthesizes the 5 held-out lines
# on the Mac:
rsync -av $BOX:piper-he/out/samples/ samples/ && open samples
```

or TensorBoard (losses, plus audio of 5 test clips at every validation):
`piper1-gpl/.venv/bin/python -m tensorboard.main --logdir run --bind_all` in `~/piper-he`
on the box and `ssh -L 6006:localhost:6006 $BOX`, then http://localhost:6006.

**When to stop:** sample every ~100 epochs. It should sound like the ElevenLabs voice
after a few hundred epochs; stop when two rounds in a row sound the same, or go back to
an earlier checkpoint if the held-out lines start to sound mumbled, breathy or garbled
(overfitting 90 minutes). Checkpoints are in `run/lightning_logs/version_*/checkpoints/`
(`last.ckpt` and the 5 best by validation mel loss); choose by ear.

### 7d. Export and bring it home

```sh
./train_piper_hebrew.sh export run/lightning_logs/version_0/checkpoints/<the one>.ckpt
# on the Mac:
rsync -av $BOX:piper-he/out/he_IL-oracle-medium.onnx* models/
```

That gives `models/he_IL-oracle-medium.onnx` and `.onnx.json` (a copy of
`train_he/config.json`: the checkpoint's config, same phoneme_id_map). Then stop the GPU
instance. Copy the `.ckpt` you chose too if you might fine-tune further.

### 7e. Test it locally

```sh
.venv/bin/python - <<'EOF'
import wave, numpy as np
from mirror import tts
voice = tts.PhonikudHebrewTTS("models/phonikud-1.0.int8.onnx", "models/he_IL-oracle-medium.onnx",
                              g2p=tts.RenikudG2P("models/renikud-plus.int8.onnx"))
for n, text in enumerate(["שלום, אני המראה. מה תרצה לדעת?",
                          "בשנת אלפיים עשרים ושש המראה התחילה לדבר.",
                          "העתיד שלך כתוב בכוכבים."], 1):
    audio, sr = voice.synth(text)
    with wave.open(f"/tmp/oracle_he_{n}.wav", "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
    print(f"/tmp/oracle_he_{n}.wav  {len(audio) / sr:.1f} s  {voice.phonemes(text)}")
EOF
afplay /tmp/oracle_he_1.wav
```

Compare with the same lines through today's voice (`models/phonikud-tts-he.onnx`).

### 7f. Wire it in

In `config/mac.toml` and `config/orin.toml` (and copy the two files to the Orin's
models directory):

```toml
[tts]
phonikud_voice_he = "models/he_IL-oracle-medium.onnx"   # config: same path + ".json"
he_engine = "phonikud"
he_g2p = "renikud"      # the G2P the voice was trained with
```

This alone only takes effect with `engine = "piper"`, which also puts English on Piper
(`piper_voice_en`, the lessac voice), not on the oracle's cloned Pocket voice.

**TODO (code change, not done yet):** today `tts.engine = "pocket"` serves *both*
languages from Pocket (`build_tts` in `mirror/voice_main.py` builds
`TTSRouter(pocket.for_lang("en"), pocket.for_lang("he"))`). For Pocket English + the new
Piper Hebrew, add a mode (e.g. `engine = "pocket"` with a new `he_on_piper = true`, or
`engine = "mixed"`) that builds the router from `pocket.for_lang("en")` and
`build_hebrew_tts(tts_cfg)`. The two engines' sample rates differ (Pocket 24 kHz, Piper
22.05 kHz), so check that `build_tts`'s single returned rate and the audio output resample
per utterance; keep the fallback to the piper stack if Pocket fails to load; add a test
in `tests/test_voice_main.py`. Hebrew then also gets Piper's phoneme alignments, so the
mouth follows the phonemes again.

The Phonikud checkpoints are licensed non-commercial (CC-NC); this installation is
non-commercial, as with the current Hebrew voice files.
