# Mirror, Mirror — Midburn installation

Full brief: https://claude.ai/artifact/UmGjaoohxeX6HtMgGn3iWn

## What it is
A 2 m standing mirror for Midburn (Flamingo camp). Wooden frame, two-way
mirror acrylic instead of glass. Behind the acrylic sit LED panels
(75x75 cm) that render an animated face. A visitor says "Mirror, mirror (on the
wall)" or "מראה מראה (שעל הקיר)", speaks in Hebrew or English, and the mirror answers out loud in the same
language. Fully offline. Placed away from loud areas.

## Hardware
- Compute: NVIDIA Jetson Orin Nano 8 GB dev kit.
- Display path: Orin DisplayPort -> DP-to-DVI cable -> NovaStar MSD300
  sending card -> Cat5e -> NovaStar MRV412-N receiving card -> 5 V HUB75
  panels (P2.976 -> 252x252 px). The Orin sees a normal monitor. Output
  fixed at 1920x1080; the face is drawn in a 252x252 rectangle at the
  top-left, mapped to the panels in NovaLCT. Panel config is saved to the
  receiving card.
- Mic + sound card: Seeed ReSpeaker XVF3800 USB (no ESP32S3), USB-A to
  USB-C. Onboard AEC/beamforming. Playback goes out through its 3.5 mm
  jack so AEC has the reference.
- Audio out: ReSpeaker 3.5 mm -> TPA3116D2 (XH-M567, 12 V) -> 2x Herdio
  3.5" passive outdoor speakers.
- Frame lighting: WS2815 12 V strip, 6 m, on an ESP32 running WLED.
  Orin sends state changes over USB serial (WLED JSON API).
- Trigger: wake phrase in either language, no button. Silero VAD cuts
  speech segments from the ReSpeaker's processed audio; whisper.cpp
  transcribes each one; wake if the normalized text starts with
  "mirror mirror" or "מראה מראה" (fuzzy match, spelling variants
  allowed). "on the wall" / "שעל הקיר" optional. Speech after the
  phrase in the same segment is the question.
- Power at the event: 220 V AC standard outlet feeding all supplies. 12 V 600 W (amp, strip, fan; each branch fused at 10 A);
  separate 5 V (panels, cards); Orin on its own supply.
- Storage: high-endurance microSD (no SSD). Keep writes minimal: logs
  in RAM (volatile journald), no swap on the card.
- Frame strip data: ESP32 3.3 V straight to WS2815, no level shifter.
  Keep the data wire under 30 cm.
- NovaLCT: one-off setup from a Windows laptop over USB-B to the MSD300.

## Software to build
1. Face renderer: full-screen window (pygame or SDL2) drawing the face
   into the panel rectangle. States: idle (blink, small drifts),
   listening, thinking, speaking. Mouth opening follows TTS audio RMS.
2. Voice loop: wake phrase -> record until silence (VAD) ->
   whisper.cpp (multilingual small, CUDA, language auto-detect he/en) ->
   llama.cpp server with Gemma 3 4B instruct Q4_K_M (CUDA), told to
   answer in the detected language -> TTS: Piper for English,
   MMS-TTS (facebook/mms-tts-heb) for Hebrew -> play through ReSpeaker
   while feeding amplitude to the face.
3. Frame lights: map the four states to WLED presets over serial.
4. Supervisor: systemd units for each process, restart on failure,
   start on boot, no desktop login required.

## Constraints
- Fully offline. No internet at the event.
- 8 GB shared between STT, LLM, TTS and the renderer. Test them together.
- 2-4 s response delay is acceptable; cover it with the thinking state.
- Must run for hours unattended in heat and dust.
- Hebrew and English, auto-detected. Reply in the visitor's language.
  Fallback if Hebrew output is too weak: understand Hebrew, reply in
  English.
- Persona: a magic mirror. Playful, a little vain, never cruel to the
  person in front of it. Replies of one to three sentences.

## Still to check (hardware labels, no decisions)
- Speaker impedance (4 or 8 ohm; never 70/100 V line mode)
