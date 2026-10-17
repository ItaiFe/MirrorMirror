# Face asset bank format

A bank is a directory with `manifest.json` plus PNG/NPY files. All coordinates
are in a 252x252 square. `mirror.face.bank.load_bank(path)` loads it and raises
`BankError` (naming the file) when a referenced file is missing or has the
wrong size.

```json
{"size": 252,
 "hub": "hub.png", "mask": "mask.png",
 "mouth": {"box": [x, y, w, h], "patches": {"A_mild": "mouth/A_mild.png"}},
 "eyes": {"box_left": [x, y, w, h], "box_right": [x, y, w, h],
          "patches": {"open": "...", "half": "...", "closed": "...", "look_up": "..."}},
 "expressions": {"smirk": "expr/smirk.png", "squint": "...", "brows_up": "..."},
 "warps": {"jaw": "warps/jaw.npy", "brows": "...", "breathe": "...", "gaze_x": "...", "gaze_y": "..."},
 "idle_loops": ["loops/idle1.npy"]}
```

`eyes`, `expressions`, `warps` and `idle_loops` are optional. Both eye boxes
must have the same size (one patch per name serves both eyes).

## In-memory representation (numpy)

| item | dtype / shape | notes |
|---|---|---|
| hub | uint8 (252,252,3) | RGB base face |
| mask | float32 (252,252) | 0..1 from an 8-bit grayscale PNG |
| mouth/eye patches | float32 (h,w,4) | RGBA 0..1, straight alpha, feathered; (h,w) = box (h,w) |
| expressions | float32 (252,252,4) | full-size RGBA 0..1 |
| warps | float32 (252,252,2) | displacement in pixels at weight 1.0 (.npy) |
| idle loops | uint8 (N,252,252,3) | RGB frames (.npy) |

## Fallbacks (`FaceBank.mouth_patch(shape, wide)`, `FaceBank.eye_patch(name)`)

Mouth patch keys are `<shape>_<mild|wide>` (shapes A-H, X). Lookup order:
exact width, then the other width (`wide` <-> `mild`); for `E/F/G/H`, then the
same two for `C`. `X`, or nothing found, returns `None` meaning draw no patch
(the hub's own mouth shows). Missing eye names fall back to `open`.
