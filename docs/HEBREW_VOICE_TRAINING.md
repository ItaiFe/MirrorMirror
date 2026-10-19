# Training the oracle's Hebrew voice

Goal: a Piper (VITS) Hebrew voice that sounds like the oracle's ElevenLabs voice,
fine-tuned from a Phonikud Hebrew checkpoint. Compared with today's Hebrew (the cloned
Pocket voice speaking Hebrew through an English model):

- native Hebrew pronunciation, learned from Hebrew audio;
- about 5x faster and ~0.6 GB less memory on the Orin;
- exact lip sync again (Piper reports when each sound is spoken).

Two parts: **you** generate the audio with ElevenLabs (steps 1-6), then **Claude**
prepares and runs the training and wires the voice in (step 7).

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

## 7. Hand it over

Tell Claude the dataset is ready (`dataset_he/` stays out of git: it's large). Claude
then:

1. turns each sentence into phonemes with the mirror's own G2P (RenikudPlus), exactly as
   at runtime;
2. fine-tunes a Phonikud Piper checkpoint
   ([Phonikud/phonikud-tts-checkpoints](https://huggingface.co/Phonikud/phonikud-tts-checkpoints))
   on a rented cloud GPU, a few hours (~$5-20), checking samples along the way;
3. exports the ONNX voice, compares it with today's Hebrew, and wires it in: Hebrew on the
   new Piper voice, English on the cloned Pocket voice (or a matching English Piper voice
   trained the same way).

The Phonikud checkpoints are licensed non-commercial (CC-NC); this installation is
non-commercial, as with the current Hebrew voice files.
