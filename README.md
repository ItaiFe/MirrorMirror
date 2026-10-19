# Mirror, Mirror

Software for the talking LED mirror (Midburn, Flamingo camp). The spec is
`MIRROR_CONTEXT.md`; how the face pictures are made is `docs/FACE_ASSETS.md`.

Four programs make the mirror. On the Orin each one is a systemd service that
starts at boot and restarts itself if it crashes:

| Service | What it does |
|---|---|
| `mirror-stt` | whisper.cpp server with ivrit-ai's Hebrew Whisper turbo: turns speech into text (Hebrew or English), port 8081 |
| `mirror-llm` | llama.cpp server with Gemma 3 4B: the oracle's brain, port 8080 |
| `mirror-voice` | listens to the mic, wakes on "Mirror, mirror" / "מראה מראה", asks the brain, speaks the answer, drives the frame lights and tells the face what to do |
| `mirror-face` | draws the face full-screen; the LED panels show its top-left 252 x 252 pixels |

Contents: [Mac quickstart](#mac-quickstart) · [Orin deploy](#orin-deploy) ·
[LED panels (NovaLCT)](#led-panels-one-time-novalct-setup) ·
[Frame lights (WLED)](#frame-lights-wled-on-the-esp32) ·
[Bench test checklist](#bench-test-checklist) ·
[Bring-up check](#one-command-bring-up-check) · [Measurements](#measured-on-a-macbook-air-m2-8-gb-2026-10-17)

---

## Mac quickstart

Everything runs on a Mac too: the face in a window, your Mac's mic and speakers,
and the frame lights printed to the log.

```sh
uv venv --python 3.11 .venv && uv pip install --python .venv -e '.[dev,hw,tts]'
# the RenikudPlus Hebrew G2P declares Python >= 3.12 but runs on 3.10/3.11:
uv pip install --python .venv --no-deps --python-version 3.12 renikud-plus==0.5.0
make models        # deploy/fetch_models.sh models (ivrit-ai Whisper turbo q5_0, whisper small, Gemma 3 4B, Silero, Piper, Phonikud, RenikudPlus, MMS cache)
make test
make servers-mac   # just whisper-server :8081 + llama-server :8080 (Ctrl-C stops both)
make dev           # servers in background (logs in /tmp/mirror-dev), wait for /health, then face + voice
```

Then say "Mirror, mirror on the wall, ..." or "מראה, מראה שעל הקיר, ..." and ask a question.

- `make servers-mac` installs `whisper-cpp` and `llama.cpp` with Homebrew if they are missing.
- `make dev LLM_MODEL=models/other.gguf` swaps the LLM. The face window is 756x756 (252 x 3).
- Speech recognition is ivrit-ai's Hebrew fine-tune of Whisper large-v3-turbo
  (`models/ivrit-turbo-q5_0.bin`: `make models` downloads ivrit-ai's official f16 ggml
  at a pinned revision and quantizes it with `whisper-quantize`), run with `-ac 768`
  (a 15.36 s audio context: the VAD caps segments at 15 s). It hears the Hebrew wake
  phrase far better than stock small (15-16/16 vs 8/16 on synthetic clips), but it
  labels every clip "hebrew", so the reply language comes from the transcript's script
  (more Hebrew letters than Latin -> Hebrew). English it transliterates into Hebrew
  letters ("מיר מיר") gets a Hebrew reply. Back to stock small:
  `make dev STT_MODEL=models/ggml-small.bin`; on the Orin
  `systemctl --user edit mirror-stt` -> `[Service]` `Environment=STT_MODEL=ggml-small.bin`.
- `mirror-voice --config config/mac.toml --wav FILE.wav` feeds a WAV through VAD and the
  conversation controller exactly like mic audio (then 1 s of silence), plays the reply and exits.
- The Hebrew voice is a G2P (RenikudPlus by default, `tts.he_g2p = "renikud"`; Phonikud
  with `"phonikud"`) feeding the Phonikud Piper voice. For the old MMS fallback voice
  (`tts.he_engine = "mms"`) add the `mms` extra: `-e '.[dev,hw,tts,mms]'` (pulls in torch).
- The oracle's background story is `config/oracle_world.md`; edit it freely, it is read at start.

### The oracle's voice: Pocket TTS

`tts.engine = "pocket"` swaps both voices for [Pocket TTS ONNX](https://huggingface.co/thewh1teagle/pocket-tts-onnx)
(CC-BY-4.0: Kyutai's Pocket TTS, ONNX export and Hebrew IPA adapter by thewh1teagle).
One model speaks English (as text) and Hebrew (as RenikudPlus IPA), in one voice: the
oracle's cloned voice (`pocket_voice_file`, below), on both the Mac and the Orin
(~5x slower than Piper: RTF ~0.2-0.3 on an M2 at 2 threads; not yet measured on the
Orin). If Pocket or the voice file fails to load, the mirror logs a WARNING and uses the piper stack.

```sh
# not on PyPI and declares Python >= 3.13, but runs on 3.10/3.11 (uv can't override that):
uv pip install --python .venv -e '.[pocket]'
.venv/bin/python -m ensurepip
.venv/bin/python -m pip install --no-deps --ignore-requires-python \
    "pocket-tts-onnx @ git+https://github.com/thewh1teagle/pocket-tts-onnx@9860fc0fe0ee5ffb13a4bae2fb99e9bd913acbe5"
```

`make models` fetches `pocket-tts-english-ipa.onnx` (232 MB, pinned, sha256-checked);
`setup_orin.sh` installs the package. Then in the config's `[tts]`:

- `engine = "pocket"`; `pocket_threads = 2` (the upstream design point).
- `pocket_voice`: a built-in voice. `javert` (CC0, the deepest) is the default; `michael`
  (CC-BY) is steadier on short lines; `marius` is CC0. `cosette` is non-commercial, and
  `omer`/`liat` have undocumented provenance.
- `pocket_voice_file`: a cloned voice instead. Clone on the Mac, never on the Orin (the
  encoder needs ~1.5 GB): `python tools/clone_voice.py REF.wav OUT.npy` (10-20 s of one
  clear speaker), copy `OUT.npy` over and point `pocket_voice_file` at it. **Kyutai's
  terms forbid cloning or imitating a voice without its owner's explicit consent.**
  The oracle's own voice is `models/voices/mirror.npy`, cloned from a synthetic voice
  designed in ElevenLabs (models/ isn't in git): copy it with
  `scp models/voices/mirror.npy mirror@ORIN:mirror-models/mirror-voice.npy`. Measured on an M2
  (2 threads): RTF 0.22 (en) / 0.25-0.32 (he), ~0.9 GB peak RSS with RenikudPlus loaded.

Differences from Piper: Pocket gives no phoneme timings, so the mouth track is estimated
from the IPA (espeak for English) and the audio's loudness (about 55% of frames and 77%
of lip closures match Piper's exact track). Short sentences sometimes stop after the
first word; a clip shorter than 45% of its expected length is re-rolled with a new seed
(at most twice, keeping the longest). Leading silence over 0.1 s (often ~1 s on English)
is trimmed, and pauses inside a line are cut to 0.4 s (a cloned voice keeps its
reference's slow pacing otherwise). Its voices speak slower, so most filler lines run past the 2.5 s filler
budget: the startup log warns about each one.

---

## Orin deploy

You need: the Orin Nano dev kit on its own power adapter (plugged into a normal
220 V outlet), a high-endurance microSD card, internet access (Ethernet or Wi-Fi)
for the one-time setup, and a screen + keyboard for the first boot (any HDMI/DP
monitor, or the LED panels once NovaLCT is done).

1. **Flash JetPack 6** onto the microSD card with NVIDIA's SD card image
   (or SDK Manager). Boot it and, in the first-boot wizard, create the user
   **`mirror`**. The config (`config/orin.toml`) expects `/home/mirror`.
2. **Get the code** (in a terminal on the Orin, or over `ssh mirror@<orin-ip>`):

   ```sh
   sudo apt install -y git
   git clone <this repo's URL> ~/MirrorMirror
   cd ~/MirrorMirror
   deploy/setup_orin.sh
   ```

   Run it logged in as `mirror` directly (in a terminal on the Orin's desktop, or
   `ssh mirror@<orin-ip>`), not through `sudo -u mirror` or `su mirror`: it needs
   that user's own systemd session, and it stops with a message if it can't find it.
   It also needs the desktop (GDM) that JetPack's SD card image comes with.
   It asks for your password (sudo) and takes a while (building whisper.cpp and
   llama.cpp with CUDA, plus about 4 GB of model downloads: count on about an hour).
3. **Reboot**: `sudo reboot`. The Orin logs itself in, and about a minute later the
   face is on screen and the mirror listens.

The script also sets the system timezone to `Asia/Jerusalem` (override with
`--timezone TZ`), because the lures' quiet hours and the oracle's time-of-day talk use local time.

**Updating** after code changes: `cd ~/MirrorMirror && git pull && deploy/setup_orin.sh`.
The script is safe to re-run: finished steps are skipped and the running services are
restarted with the new code.

What `deploy/setup_orin.sh` does, step by step:

1. Installs build tools and libraries with apt, and adds you to the `dialout`
   (serial port for the lights) and `audio` groups. The new groups take effect after
   the reboot at the end.
2. Builds `whisper-server` and `llama-server` with CUDA into `~/mirror-bin`
   (versions pinned in the script: `WHISPER_REF`, `LLAMA_REF`).
3. Creates the Python venv `~/MirrorMirror/.venv` and installs the app
   (`pip install -e '.[hw,tts]'`). TTS and VAD run on the CPU, so plain onnxruntime
   is fine. `deploy/setup_orin.sh --with-mms` also installs NVIDIA's JetPack torch
   wheel + transformers for the MMS fallback voice (off by default).
4. Downloads the models into `~/mirror-models` (`deploy/fetch_models.sh`).
5. Installs the four services as systemd *user* units (`~/.config/systemd/user/`)
   and turns on "linger", so they start at boot without anyone logging in.
6. Sets up the desktop for the face: automatic login on Xorg (not Wayland), never
   blank the screen, no lock screen, no pop-ups (`deploy/xorg/10-mirror.conf` + GNOME settings,
   and `mirror-dpms.timer` re-applies "no blanking" every 10 minutes).
7. Spares the SD card: logs are kept in RAM only, and any swap *file* is removed
   (the zram swap in RAM stays).
8. Sets the power mode (`nvpmodel -m 0`) and runs `jetson_clocks` at every boot.
   On JetPack 6.2 the fastest Orin Nano mode is "MAXN SUPER": find its number with
   `sudo nvpmodel -q --verbose` and re-run with `NVP_MODE=<number> deploy/setup_orin.sh`.

### Day-to-day commands on the Orin

```sh
systemctl --user status mirror-stt mirror-llm mirror-voice mirror-face   # is everything up?
journalctl --user -u mirror-voice -f          # watch the conversation log live (Ctrl-C to stop)
systemctl --user restart mirror-voice         # restart one service (e.g. after editing config/orin.toml)
systemctl --user stop mirror-voice mirror-face   # stop the mirror (the frame lights return to idle)
tegrastats                                    # memory, CPU/GPU load and temperatures, once a second
```

Logs live in RAM, so they are gone after a reboot. That is deliberate.

**If the face doesn't appear:** `journalctl --user -u mirror-face`. The face talks to the
X display `:0`. If the desktop runs on another display (in a terminal on the Orin's
desktop, `echo $DISPLAY`), change `Environment=DISPLAY=` in
`~/.config/systemd/user/mirror-face.service`, then `systemctl --user daemon-reload`
and `systemctl --user restart mirror-face`.
**If the mic or speaker isn't found, or the mirror is silent:** at startup
`journalctl --user -u mirror-voice` logs the chosen devices (`input device: [N] ...`,
`output device: [N] ...`) and the formats opened (`mic open: 48000 Hz x 2 ch ...`,
`audio output open: ...`). To see every device PortAudio offers, with its index and
channel counts, run on the Orin:

```sh
.venv/bin/python -m sounddevice    # in the repo checkout; one line per device: index, name, (in, out)
```

The config picks the device whose name contains `input_device` / `output_device`
(`[audio]` in `config/orin.toml`, default `XVF3800`). The ReSpeaker's raw ALSA device
(`... (hw:2,0)`) only takes 16 or 48 kHz and 2 channels; the mirror negotiates a format it
accepts and resamples itself, so that is fine. With `prefer_alsa_plug = true` and several
matching names, a converting device (`plughw`, `pulse`, `pipewire`, `default`) is preferred.
If the card is held by PulseAudio ("Device unavailable" / "busy"), set
`input_device = "pulse"` and `output_device = "pulse"` (or `"default"`) to go through it,
then `systemctl --user restart mirror-voice`. A mic that delivers no audio for 2 s is
reopened; after 3 failures in a row the service exits and systemd restarts it.
**If the lights don't react:** check that the ESP32 shows up as `/dev/ttyUSB0`
(`ls /dev/ttyUSB* /dev/ttyACM*`); otherwise set `serial_port` in `config/orin.toml`.

---

## LED panels: one-time NovaLCT setup

The NovaStar cards have to be told once what panels they drive. That is done with
NovaStar's **NovaLCT** program, which only runs on Windows, so this uses a
**borrowed Windows laptop** (a normal Intel/AMD one; NovaLCT doesn't run on ARM
Windows or on a Mac). The settings are then saved inside the receiving card, and
the laptop is never needed again.

### What to bring

- The Windows laptop, with **NovaLCT already downloaded and installed** from
  novastar.tech (Support → Software). Install it while you have internet; it also
  installs the USB driver. Restart the laptop once after installing.
- **MSD300** sending card, its **power adapter**, and the **USB-A to USB-B cable**.
- **MRV412** receiving card and an **Ethernet (Cat5e) cable**.
- **One panel**, its HUB75 ribbon cable, and the **5 V supply** for the panel and
  the MRV412 (with its mains cable).
- If the panel seller gave you a configuration file for the panel (`.rcfgx` or
  `.rcfg`), bring it on a USB stick: it saves most of step 4.

### Steps

1. **Wire it up.** Panel ribbon into one of the MRV412's HUB75 ports (J1). MRV412
   powered from the 5 V supply (mind + and -). Ethernet cable from the MSD300's
   output port 1 to the MRV412's input port. MSD300 powered from its adapter.
   USB-B cable from the MSD300 to the laptop. Then switch on the 5 V supply.
2. **Start NovaLCT.** It should find the MSD300 on a COM port by itself (shown at the
   bottom of the window).
3. **Log in to the advanced settings.** Menu **User → Advanced Synchronous System
   User Login**, password **`admin`** (that is NovaStar's documented default password;
   if someone changed it, ask them). Then click **Screen Configuration** and pick
   the MSD300's COM port.
4. **Tell the receiving card about the panel** (the **Receiving Card** tab):
   - With a config file from the seller: **Load from File**, pick the `.rcfgx`.
   - Without one: click **Smart Settings** (the smart setup wizard) and follow it.
     It asks for the panel's chip and size and then shows test patterns on the panel;
     you answer what you see (which dots light up, which colour). The panels are
     P2.976; a 250 x 250 mm P2.976 module is 84 x 84 pixels (check the label on the back).
5. **Set the cabinet size to 252 x 252** (width 252, height 252: the whole face,
   e.g. 3 x 3 modules of 84 x 84). Set the module arrangement to match how the ribbons
   are chained.
   With only one panel on the bench, only that panel lights up; that's expected.
6. **Send to Receiving Card.** The panel should now show the picture correctly.
7. **Place the image at 0,0.** In the **Screen Connection** tab: 1 x 1 cabinet,
   the receiving card at **X = 0, Y = 0**, width 252, height 252. Send it. In the
   **Sending Card** tab, set the input to **DVI, 1920 x 1080 @ 60 Hz**.
8. **SAVE to the receiving card.** Click **Save** (in the Receiving Card tab:
   **Save to Receiving Card**; in the Screen Connection tab: **Save**). Without this,
   everything is forgotten at the next power cut.
9. **Check it stuck:** switch the 5 V supply off and on again. The panel should come
   back with the right picture. Use NovaLCT's **Test** tools (colour bars, grid) to
   look for wrong colours or shifted rows.

After this, plug the Orin's DisplayPort-to-DVI cable into the MSD300 instead of
the laptop. The top-left 252 x 252 pixels of the Orin's screen are the face.

---

## Frame lights: WLED on the ESP32

The frame strip (WS2815, 12 V) is driven by an ESP32 running **WLED**. The Orin
plugs into the ESP32 by USB and sends it one of four presets as the mirror changes
state (the line `{"ps":N}` over serial at 115200 baud).

### Flash WLED (from the Mac, in Chrome)

1. Plug the ESP32 into the Mac with a USB **data** cable (some cables only charge).
2. Open **https://install.wled.me** in **Chrome** (Safari can't talk to USB serial).
3. Click **Install**, choose the ESP32's port in the pop-up (usually
   `cu.usbserial-...` or `cu.SLAB_USBtoUART`), and confirm. It takes about a minute.
   If no port appears, install the USB-serial driver for the chip printed next to the
   ESP32's USB socket (CP2102: Silicon Labs; CH340: WCH) and try again.
4. When it asks, enter your Wi-Fi so you can reach the WLED web page from the Mac
   (only needed for setup; at the event the lights run without any network).
   Open the address it shows (or the "WLED-AP" Wi-Fi it creates, password `wled1234`,
   then http://4.3.2.1).

### Set up WLED

- **Config → LED Preferences:** LED type WS281x, **colour order GRB** (if colours come
  out wrong, try the others), the **GPIO** pin your data wire is on, and the number of
  LEDs (60 per metre: 6 m = 360). Turn off the Automatic Brightness Limiter (it assumes
  5 V LEDs; the 12 V supply and its 10 A fuse protect the strip). Save.
- **Config → Sync Interfaces → Serial:** baud rate **115200**. Save. (It is usually
  the default already; the mirror sends at 115200, see `[lights]` in `config/orin.toml`.)
- **Presets:** make these four, with exactly these **ID numbers** (pick the effect and
  colour in the main screen, then **+ Preset**, set the ID, Save):

| ID | Mirror state | Look |
|---|---|---|
| 1 | idle | slow pink breathe (effect "Breathe", pink, low speed, medium brightness) |
| 2 | listening | steady teal (effect "Solid", teal) |
| 3 | thinking | rotating chase (effect "Chase", a light colour on a dim background) |
| 4 | speaking | bright pulse (effect "Breathe", full brightness, high speed) |

To try a preset from the Orin by hand (with `mirror-voice` stopped):

```sh
stty -F /dev/ttyUSB0 115200 raw && echo '{"ps":2}' > /dev/ttyUSB0
```

When the mirror is stopped (`systemctl --user stop mirror-voice`), it switches the
lights back to preset 1 (idle) on the way out.

---

## Bench test checklist

Test each step on the bench before moving to the next (this follows the brief's
build order). Tick them off.

### One-command bring-up check

Before ticking boxes by hand, let one command say what works and what to fix:

```sh
cd ~/MirrorMirror
.venv/bin/python tools/bringup_check.py --json /tmp/bringup.json   # on the Orin (config/orin.toml)
.venv/bin/python tools/bringup_check.py --config config/mac.toml --quick --skip-audio   # on the Mac
```

It does not start anything: start the servers first (`systemctl --user start mirror-stt
mirror-llm` on the Orin, `make servers-mac` on the Mac), and stop `mirror-voice` so the
mic, speaker and serial port are free. In order, each check prints PASS, WARN, FAIL or
SKIP, a one-line reason and, if something is wrong, where in this README to look:

1. **config**: the config loads, and every model file it and the server units use exists
   with a plausible size (a truncated download fails). The server models are read from
   the installed units and their drop-ins (on the Mac `$STT_MODEL`/`$LLM_MODEL` win), and
   the line says which file each came from.
2. **audio input / output**: the `input_device` / `output_device` substring matches a
   device, and the mic and speaker open for about 1 s. It reports the format they
   negotiated. Each runs in a subprocess that is killed after 15 s, so a hung
   ALSA/CoreAudio device can't block the check.
3. **serial**: the WLED port exists and accepts the idle preset (the frame breathes pink).
   With `kind = "log"` this is a WARN.
4. **display** (Orin): xrandr's current mode is 1920x1080 (on `:0`, as the face units use).
5. **stt server / llm server**: whisper-server and llama-server answer `/health`.
6. **memory**: `free -m` plus one `tegrastats` line (`vm_stat` on the Mac). It FAILs
   under 300 MB available and WARNs under 500 MB: see "Memory on the Orin" below.
7. **e2e English / Hebrew**: an English and a Hebrew "Mirror, mirror ..." question,
   synthesized once with the repo's voices and cached in `.cache/bringup` (gitignored), go
   through the real VAD -> STT -> wake -> LLM (streaming) -> TTS path, with a fake
   speaker (`tools/record_demo.py`'s timeline). It prints the transcript and the reply,
   and reports whether the wake phrase matched, the detected language, and the time from
   the end of speech to the filler and to the first reply sentence. PASS when the wake
   matched and the first sentence starts within 6 s, WARN up to 10 s.
   The whole check gets 120 s (warmups included). It stops after 2 LLM errors in a row.
8. **hebrew address**: `tools/hebrew_address_check.py` with 10 questions. WARN if more
   than 2 of the 10 replies address the visitor in the singular. It stops after
   180 s or after 2 failed requests in a row.

`--quick` skips 7 and 8, `--skip-audio` skips 2, and `--json PATH` also writes the
full report (every reason, transcript and measurement) as JSON. A summary table comes
last. The exit code is 0 when nothing FAILed.

**1. Panels light up**
- [ ] NovaLCT setup done and **saved to the receiving card** (section above).
- [ ] After switching the 5 V supply off and on, the panel shows the test image correctly.

**2. The Orin drives the panels**
- [ ] `deploy/setup_orin.sh` finished without errors, then `sudo reboot`.
- [ ] Orin → DP-to-DVI → MSD300: within 90 s of power-on the face appears in the
      panel area (top-left 252 x 252), with no login needed.
- [ ] Unplug the Orin's power, plug it back in: the face comes back by itself.
- [ ] Left alone for 15 minutes: the screen does not blank.

**3. The voice loop works at a desk**
- [ ] **First**, with everything running: `free -m` shows at least 0.5 GB available
      (`tegrastats`: `RAM x/7620MB` under about 7100). If not, see "Memory on the Orin"
      below for the order of cuts.
- [ ] ReSpeaker plugged in; `arecord -l` and `aplay -l` both list it.
- [ ] "Mirror, mirror on the wall, what do you see?" gets a spoken English answer.
- [ ] "מראה, מראה שעל הקיר, מה צופן לי העתיד?" gets a spoken Hebrew answer.
- [ ] Delay from the end of your question to the first sound is acceptable (target 2-4 s;
      `journalctl --user -u mirror-voice | grep latency`).

**4. The face comes alive**
- [ ] Idle: it blinks. Listening, thinking and speaking each look different.
- [ ] The mouth moves with the voice.

**5. Sound and frame lights**
- [ ] Speakers checked: 4 or 8 ohm, **never** 70/100 V line mode. Amp volume starts low.
- [ ] Amp and strip on 12 V, each branch through its own 10 A fuse.
- [ ] WLED presets 1-4 set up; the frame follows the mirror: idle → listening →
      thinking → speaking → idle.
- [ ] `systemctl --user stop mirror-voice` leaves the frame on idle (pink breathe).

**6. Unattended endurance**
- [ ] 8 hours running with regular questions, inside the closed enclosure, in heat.
- [ ] Afterwards: `systemctl --user show -p NRestarts mirror-voice mirror-face mirror-llm mirror-stt`
      shows few or no restarts, and `tegrastats` temperatures stayed reasonable.
- [ ] Pull the power mid-answer, plug it back in: everything comes back on its own.

### Memory on the Orin

The 8 GB Orin is tight: with everything running, the estimate is about 7.1-7.4 GB of the
7.62 GB `tegrastats` shows. That counts llama-server at about 3.8 GB (with the 8-bit KV
cache and no host-RAM prompt cache, `-cram 0`), whisper-server at about 1 GB,
mirror-voice at about 1 GB (about 1.6 GB with the oracle's cloned Pocket voice, the
default), mirror-face at about 0.2 GB, and JetPack plus GNOME at
about 1.3 GB. There is no swap file, only zram. If memory runs out, the kernel kills a
server mid-event, which costs a 10-20 s restart and fallback replies.
**Measure memory first, before the rest of step 3.** Start everything, ask one English
and one Hebrew question, then run:

```sh
free -m          # the "available" column, in MB
tegrastats       # RAM x/7620MB, once a second (Ctrl-C to stop)
```

**Target: at least 0.5 GB free**, meaning `available` at 500 MB or more, or RAM under
about 7100/7620MB. If you are short, make these cuts in order. After each one, restart
that service and measure again. Stop as soon as you reach the target.

The llama-server cuts (a-c) each replace the unit's command line with a drop-in. Write
the whole line, with your changes, into
`~/.config/systemd/user/mirror-llm.service.d/override.conf`.
`systemctl --user edit mirror-llm` opens that file in an editor. The empty `ExecStart=`
line clears the original command first. Here is an example with cuts a and b both made:

```sh
mkdir -p ~/.config/systemd/user/mirror-llm.service.d
cat > ~/.config/systemd/user/mirror-llm.service.d/override.conf <<'CONF'
[Service]
ExecStart=
ExecStart=%h/mirror-bin/llama-server -m %h/mirror-models/${LLM_MODEL} -ngl 99 -c 8192 -np 2 --cache-reuse 256 -fa on -ctk q8_0 -ctv q8_0 -cram 0 -ub 256 --host 127.0.0.1 --port 8080
CONF
systemctl --user daemon-reload && systemctl --user restart mirror-llm
```

a. **Drop `--swa-full`** (estimated saving about 0.3 GB with the 8-bit KV cache). Remove it from the
   line. The cost: the English slot re-processes about 500 prompt tokens per turn again,
   which adds to the English reply latency.
b. **`-ub 256`** (estimated saving about 0.25 GB of compute buffer). Add it to the line. Prompt
   processing gets a little slower.
c. **`-c 6144`** (a smaller KV cache). Change `-c 8192` to `-c 6144`. Each of the two
   slots then holds 3072 tokens, which is still enough for the system prompt and a short
   history.
d. **Go back to the stock Whisper small model** (estimated saving about 0.4 GB). Hebrew recognition
   gets worse: in the Task 22 tests, 8/16 Hebrew wakes instead of 16/16. Check that
   `~/mirror-models/ggml-small.bin` exists, then run:

   ```sh
   mkdir -p ~/.config/systemd/user/mirror-stt.service.d
   printf '[Service]\nEnvironment=STT_MODEL=ggml-small.bin\n' \
       > ~/.config/systemd/user/mirror-stt.service.d/override.conf
   systemctl --user daemon-reload && systemctl --user restart mirror-stt
   ```

e. **Advanced: a bare X session instead of GNOME** (estimated saving about 0.5-0.8 GB: gnome-shell,
   evolution, tracker and gnome-software no longer run). This has not been tried on the
   Orin yet. GDM keeps autologging you in on Xorg, so `DISPLAY=:0` and the
   `%t/gdm/Xauthority` path that the face units use stay valid. Only the session changes,
   from GNOME to Openbox with no panels, so mirror-face is the only window. The apt
   install needs the network once:

   ```sh
   sudo apt install -y openbox            # installs /usr/share/xsessions/openbox.desktop
   sudo mkdir -p /var/lib/AccountsService/users
   sudo tee /var/lib/AccountsService/users/$USER >/dev/null <<'CONF'
   [User]
   Session=openbox
   XSession=openbox
   CONF
   sudo reboot
   ```

   Before you change anything, note the stock session's name. Run
   `ls /usr/share/xsessions` (usually `ubuntu.desktop`, which means `ubuntu`), and check
   `cat /var/lib/AccountsService/users/$USER` for the current `Session=` / `XSession=`
   values. If that file already exists, edit its two lines instead of replacing it.

   After the reboot, the face should come up on its own. The user units start at boot
   (setup_orin.sh enables linger), not with the desktop, and mirror-face retries every 2 s
   until X is up. Check it with `ps -e | grep -c gnome-shell`, which should print 0. Then
   check `free -m` again. Without GNOME, its blanking settings no longer apply. The units'
   `xset s off -dpms` covers that. Run the step-2 "left alone for 15 minutes" check again.

   **To undo this cut, or if Openbox doesn't start** (a black screen, or GDM loops back to
   its login screen), SSH in or switch to a text console with Ctrl+Alt+F3 and log in.
   Then set both lines back to the stock session name you noted and reboot:

   ```sh
   ls /usr/share/xsessions                # e.g. ubuntu.desktop -> "ubuntu"
   sudo sed -i 's/^Session=.*/Session=ubuntu/; s/^XSession=.*/XSession=ubuntu/' \
       /var/lib/AccountsService/users/$USER
   sudo reboot
   ```

f. **Last resort: the Piper voices instead of the cloned voice** (estimated saving about
   0.6 GB; the oracle then sounds like the stock English and Hebrew voices). In
   `config/orin.toml` `[tts]` set `engine = "piper"`, then
   `systemctl --user restart mirror-voice`.

**7. Into the frame**
- [ ] Mic at face height behind foam; the wake phrase works from where people will stand.
- [ ] The mirror illusion holds outdoors at midday and at night.

---

## Measured on a MacBook Air M2, 8 GB (2026-10-17)

End of speech to first reply audio, from the `--wav` smoke runs (includes ~0.7 s of VAD
hang-over silence; the whole reply is synthesized before playback starts):

| Input | LLM | STT | LLM reply | TTS | End-of-speech -> first audio |
|---|---|---|---|---|---|
| English (Piper WAV) | Gemma 3 4B Q4_K_M | 0.9-1.6 s | 2.9-3.9 s | Piper ~1.2-1.6 s | 6.5-6.7 s |
| Hebrew (MMS WAV, vocalized) | Gemma 3 4B Q4_K_M | 1.5-2.1 s | 4.2-5.0 s | MMS ~7-9 s | 13.3-16.6 s |
| Hebrew (Phonikud WAV, unvocalized) | Gemma 3 4B Q4_K_M | 0.9 s | 2.5-3.4 s | Phonikud ~0.5 s | 4.5-5.5 s |
| English | Gemma 3 1B Q4_K_M | 0.85 s | 0.3 s | 0.16 s | 1.9 s |
| Hebrew (vocalized) | Gemma 3 1B Q4_K_M | 0.8 s | 1.9 s | MMS 3.7 s | 6.8 s |
| English, live via `make dev` (speaker -> mic) | Gemma 3 4B | 4.3 s | 6.1 s | 3.1 s | 14.0 s |

STT alone with the default model (Task 22, 2026-10-18): ivrit-ai Whisper turbo q5_0 with
`-ac 768`, 38 synthetic TTS clips (Phonikud Hebrew, Piper English, some with 10 dB noise)
on the same swapping 8 GB Mac, through the app's STT client and wake matcher:

| STT model | Median / max per clip | Hebrew wakes | English wakes | Language he / en |
|---|---|---|---|---|
| ivrit-ai turbo q5_0, `-ac 768` (default) | 1.85 / 2.46 s | 16/16 | 7/7 | 26/26 / 10/12 |
| stock `ggml-small.bin` (old default, spike run) | 0.76 / 1.01 s | 8/16 | 7/7 | 20/26 / 12/12 |

The STT column in the table above was measured with the old small model. Real voices
through the mic will score lower than these clean synthetic clips.

The 8 GB Mac was swapping with all three models loaded, so these numbers are pessimistic;
the Orin numbers are what count. Over the 2-4 s target: MMS Hebrew synthesis on CPU and
synthesizing the whole three-sentence reply before playback are the main costs.

Notes:
- whisper-server reports the language as a full name (`"language": "hebrew"`). The app now decides
  the language from the transcript's script and uses this field only when there are no letters.
- MMS reads unvocalized Hebrew poorly: "מראה מראה שעל הקיר" synthesized without niqqud was
  heard by whisper as "מה אמר אש על הקיר" (no wake). With niqqud it woke.
- The Hebrew voice is now Phonikud + Piper (`tts.he_engine = "phonikud"`; `"mms"` is the
  fallback). It synthesizes a three-sentence reply in ~0.5 s vs MMS ~2.5-4 s. Whisper hears
  "מראה, מראה שעל הקיר, מי הכי יפה?" correctly (wake); without the comma it hears "מהה, מהה".
  The Phonikud voice checkpoints are non-commercial (see the Task 13 report for the license).
