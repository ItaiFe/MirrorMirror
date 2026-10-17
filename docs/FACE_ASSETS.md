# Making the mirror's face

The mirror shows a face 252 × 252 pixels in size. Only lit pixels show through the
two-way acrylic, so black is invisible. The face software doesn't animate a
3D model. It works from a **face bank**: one still "hub" image of the face, plus
small replacement pictures (*patches*) for the mouth shapes and the eyes, plus
a few warp fields that gently move the jaw, brows, eyes and head. The format
is described in `assets/face/README.md`.

You can make a bank two ways:

| | Stub bank (`tools/stub_bank.py`) | Real bank (`tools/build_bank.py`) |
|---|---|---|
| Input | one image | ~12 to 20 images rendered with LivePortrait |
| Needs | your laptop, about a second | a rented GPU for about 1 to 2 hours (about US$1 to 2) |
| Looks | mouth tears open along the lip line, painted black inside | real lips, teeth and eyelids |

The repo already contains a stub bank made from `assets/face/source/trickster.png`
in `assets/face/stub/`. Everything works with it today. This guide covers
both ways.

All commands run from the repo folder with the project's Python:

```sh
cd ~/MirrorMirror
source .venv/bin/activate
```

---

## 1. The source image

The source image matters more than anything else. It needs:

- **Front-facing and symmetric.** Looking straight at the camera, head not
  tilted.
- **Pure black background** (RGB 0,0,0), with nothing else in the frame.
- **Square, at least 1024 × 1024**, with the face filling the frame: the top of
  the head near the top edge and the chin or beard near the bottom edge.
- **Mouth closed** (a closed smile is fine), **eyes open**, irises centred.
- **Nothing covering the lips.** A moustache *above* the lip is fine, but
  hair or a beard crossing the lip line is not.
- **Even, warm lighting** with no deep shadows across the eyes or the mouth.
  Dark areas disappear on the LEDs.

### Prompt (for Midjourney, DALL·E, Flux, SDXL…)

```
A front-facing symmetrical portrait of a sculpted bronze trickster face, mask-like,
looking straight at the viewer, centred, the face fills the square frame from forehead
to beard tip, pointed elf ears, ornate filigree on the forehead, curled moustache,
closed mischievous smiling mouth with clearly visible lips, pointed beard with an
ornament, eyes open with golden irises, warm even studio lighting, highly detailed
polished bronze, isolated on a pure black background, no background, no shadow
```

Negative prompt, if your tool takes one:

```
tilted head, profile, three-quarter view, open mouth, teeth, hands, hair over mouth,
background, vignette, gradient background, text, watermark, frame, cropped chin
```

Settings: aspect ratio **1:1**, at least 1024 px. With Midjourney, add
`--ar 1:1 --style raw`. Generate several and pick the most symmetric one with
the clearest lips.

Save the image as `assets/face/source/<name>.png`. If the background isn't
quite black, darken it to pure black in any image editor (levels or curves),
or the mask will pick up the noise.

## 2. A stub bank right away

```sh
python tools/stub_bank.py assets/face/source/trickster.png assets/face/stub \
    --points assets/face/source/trickster_points.json
```

- `--points` names the file of **control points**: where the eyes, brows,
  moustache, mouth corners, lips, chin and beard are. Without it, a generic
  face layout is used.
- `--box x,y,w,h` crops a region of the source, in source pixels. The default
  is the centred square of the full image.

### Fitting the control points to a new face

1. Run once without `--points`, then open `OUT_DIR/hub.png` (252 × 252) in an
   image viewer that shows the pixel position under the cursor. Preview's
   Inspector does this, and so do GIMP and Photoshop.
2. Copy `assets/face/source/trickster_points.json` to `<name>_points.json` and
   edit each `[x, y]` to the matching spot on your hub:

   | point | where |
   |---|---|
   | `eye_l`, `eye_r` | centre of each iris. `eye_half_size` is half the eye opening's width and height |
   | `brow_l`, `brow_r` | middle of each brow ridge (the only things the brow warp lifts) |
   | `moustache_l/_c/_r`, `moustache_tip_l/_r` | lower edge of the moustache, and its curled tips (kept still) |
   | `mouth_l`, `mouth_r` | the corners of the mouth (kept still, so the grin stays a grin) |
   | `lip_upper` | the lip line (where the lips meet) at the centre |
   | `lip_lower`, `chin`, `jaw_l/_r`, `beard_tip`, `ornament_tip` | what drops when the jaw opens |
   | `nose_tip`, `cheek_*`, `ear_*`, `earlobe_*`, `forehead_*` | anchors that never move |

3. Re-run with `--points assets/face/source/<name>_points.json`.

The mouth box is worked out from the points: from 6 px above the mouth
corners to 10 px below the chin, and 10 px past each corner.

## 3. A real bank with LivePortrait (rented GPU)

LivePortrait re-renders your source image with a different expression: mouth
open by a set amount, eyes closed and so on. You make about 12 still images
this way, then `build_bank.py` turns them into a bank.

### 3.1 Rent a GPU on RunPod

1. Create an account at <https://www.runpod.io> and add **US$10** of credit
   (Billing → Add credit).
2. **Pods → Deploy.** Choose **Community Cloud**, GPU **RTX 4090** (24 GB),
   or an RTX 3090 or A40 if no 4090 is free.
3. **Template:** search for **ComfyUI** and pick a template that includes
   ComfyUI-Manager, such as *"ComfyUI"* by RunPod or *"ComfyUI – AI-Dock"*.
4. **Container disk 30 GB, volume disk 50 GB.** Deploy. It costs about
   US$0.40 to 0.70 per hour while running.
5. When the pod shows *Running*, click **Connect → HTTP 8188** (or the
   template's "ComfyUI" button). ComfyUI opens in your browser.

> **Stop the pod when you're done (Pods → ⏹ Stop, then 🗑 Terminate).** A
> stopped pod still bills for its disk, and a running one bills by the hour.

### 3.2 Install the LivePortrait nodes

In ComfyUI: **Manager → Custom Nodes Manager**, then search for and **Install**:

- **ComfyUI-LivePortraitKJ** (by kijai). This is the video-driven pipeline,
  used for the webcam clips.
- **ComfyUI-AdvancedLivePortrait** (by PowerHouseMan). Its **Expression
  Editor (PHM)** node has numeric sliders, which makes exact parameter sweeps
  easy.
- **ComfyUI-VideoHelperSuite** (loads your webcam videos).

Click **Restart**, then refresh the browser. The LivePortrait models download
by themselves the first time a workflow runs, which takes a few minutes.

Upload your source image: **drag it onto the ComfyUI canvas**, or use the
*Load Image* node's "choose file to upload".

### 3.3 Sweep the lips and eyes (Expression Editor)

Build this small workflow:
`Load Image` (your source) → `Expression Editor (PHM)` → `Save Image`.

Fixed settings for **every** render:

| slider | value | why |
|---|---|---|
| `rotate_pitch`, `rotate_yaw`, `rotate_roll` | 0 | the head must not move |
| `wink`, `eee`, `woo`, `smile` | 0 unless listed below | |
| `src_ratio` | 1 | |
| `sample_ratio` | 1 | |
| `crop_factor` | 1.7 | |

The node pastes the edited face back into the **full** source frame. That's
what `build_bank.py` expects, so keep the output at the source's size.

Render each row and save it under the file name shown. Click *Queue Prompt*,
then right-click the result and choose *Save Image*:

| file | changed sliders |
|---|---|
| `hub.png` | everything 0 (the neutral face as LivePortrait renders it) |
| `lip_00.png` | `aaa` = 0 |
| `lip_01.png` | `aaa` = 15 |
| `lip_02.png` | `aaa` = 30 |
| `lip_03.png` | `aaa` = 50 |
| `lip_04.png` | `aaa` = 75 |
| `eye_00.png` | `blink` = 0 (open) |
| `eye_01.png` | `blink` = −8 (half closed) |
| `eye_02.png` | `blink` = −20 (closed) |
| `eye_03.png` | `pupil_y` = 12 (eyes look up; if they look down, use −12) |
| `shape_E.png` | `woo` = 8, `aaa` = 20 (rounded "oh") |
| `shape_F.png` | `woo` = 15, `aaa` = 5 (puckered "oo") |
| `expr_smirk.png` | `smile` = 0.6, `wink` = 3 |
| `expr_squint.png` | `blink` = −6, `eyebrow` = −5 |
| `expr_brows_up.png` | `eyebrow` = 12 |

Check each one as you go:

- **The lip ladder should open steadily.** If `aaa` = 75 tears the face or
  shows mush, use 60 and spread the others evenly (0, 12, 25, 40, 60).
- **The mouth interior should be dark.** `build_bank.py` paints everything
  darker than 24/255 around the lips pure black. Teeth stay.
- **Nothing else should move.** If the moustache or beard swims, lower the
  value.

The bank maps the ladder like this:

| mouth patch | from |
|---|---|
| A (closed) | `lip_00` |
| B | mild `lip_01`, wide `lip_02` |
| C | mild `lip_02`, wide `lip_03` |
| D (widest) | mild `lip_03`, wide `lip_04` |

### 3.4 Webcam shapes G and H, and anything the sliders can't do

Shapes **G** (teeth on lower lip: "f", "v") and **H** (tongue up: "l") have no
slider, so you copy them from your own face:

1. On your laptop, take **webcam photos** (or 3-second clips) of yourself
   facing the camera with your head still, holding each mouth shape: "fff"
   (G), "lll" with your tongue visible behind your teeth (H). E ("oh") and F
   ("oo") also work this way if the slider versions look poor.
2. Upload a photo to ComfyUI and connect it to the Expression Editor's
   **`sample_image`** input. Set **`sample_parts` = OnlyMouth** and
   `sample_ratio` = 1, with every slider at 0. Save as `shape_G.png` or
   `shape_H.png`.
3. For clips, use the LivePortraitKJ example workflow instead (in the node's
   `examples/` folder; drag the `.json` onto the canvas). Load your source as
   the image and your clip in *Load Video (VHS)*, use **relative motion**,
   keep **stitching on** and **lip_zero off**, and save the output frames.
   Then pick the frame that shows the shape best.

Missing E to H patches are fine. The software falls back to the C patch.

**If LivePortrait says no face was found** (common with sculpted faces), try
the *InsightFace* cropper instead of MediaPipe in the LivePortraitKJ
workflow, raise `crop_factor` to 2.0 to 2.3, or generate a source image with
more human proportions.

### 3.5 Download and build

1. In the RunPod web terminal (Connect → *Start Web Terminal*) run
   `cd /workspace/ComfyUI/output && zip -r /workspace/raw.zip .`, then
   download `raw.zip` from the Jupyter file browser (Connect → *Jupyter Lab*).
   Some templates keep outputs in `/ComfyUI/output` or `/workspace/output`.
   Alternatively, save each image from the browser as you go.
2. **Stop and terminate the pod.**
3. On your laptop, put the files in one folder, say `~/face_raw/`, named
   exactly as in the table above: `hub.png`, `lip_00.png` to `lip_04.png`,
   `eye_00.png` to `eye_03.png`, optional `shape_E.png` to `shape_H.png` and
   optional `expr_*.png`.
4. Build:

   ```sh
   python tools/build_bank.py ~/face_raw assets/face/trickster \
       --points assets/face/source/trickster_points.json
   ```

   - Every frame is aligned to `hub.png` automatically, to a fraction of a
     pixel. Small shifts from LivePortrait are fine.
   - Use the same `--points` file as for the stub: it places the mouth and
     eye boxes. If you generated a new source image, fit new points first
     (section 2).
   - `--box x,y,w,h` works as in the stub, if your frames need cropping.

### 3.6 Switch the mirror to the new bank

In **both** `config/mac.toml` and `config/orin.toml`, under `[face]`, set:

```toml
renderer = "sprite"
bank = "assets/face/trickster"
```

Start the face app on the Mac (`mirror-face --config config/mac.toml`), talk
to it, and watch the mouth. If one shape looks wrong, re-render just that
image, rebuild, and restart. Commit the new `assets/face/trickster/` folder
so the Orin gets it with the next deploy.

## Troubleshooting

| symptom | fix |
|---|---|
| grey fringe or specks around the face | the background isn't pure black: darken it in the source and rebuild |
| black holes in the closed mouth | the source's lip shadows are too dark; lower `CAVITY_LUM` in `tools/build_bank.py` |
| the moustache moves when the jaw opens | move the `moustache_*` points onto the moustache's lower edge |
| the chin doesn't move, or the cheeks move too | check `lip_lower`, `chin`, `jaw_l/_r` (they move) against `cheek_*` and `mouth_l/_r` (they don't) |
| patches look offset | `hub.png` and the other frames came from different sources or crops; render them all from one source image |
