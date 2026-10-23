#!/usr/bin/env bash
# Fine-tune the oracle's Hebrew Piper (VITS, medium) voice on a rented CUDA GPU box.
#
#   deploy/train_piper_hebrew.sh setup     # system packages, piper1-gpl, the checkpoint
#   deploy/train_piper_hebrew.sh train     # fine-tune (resumes its own run if restarted)
#   deploy/train_piper_hebrew.sh sample    # export the latest checkpoint, synthesize samples/
#   deploy/train_piper_hebrew.sh export [CKPT]   # final .onnx + .onnx.json into $OUT_DIR
#   deploy/train_piper_hebrew.sh all       # setup + train + export
#
# Before: on the Mac, `tools/prepare_piper_dataset.py` writes train_he/ and you rsync it
# to $DATA_DIR on this box (docs/HEBREW_VOICE_TRAINING.md, section 7).
#
# What it does: OHF-Voice/piper1-gpl (the maintained Piper, `python -m piper.train`),
# pinned at $PIPER_REF, trains from our exact phoneme ids (`--data.dataset_type
# phoneme_ids`, the checkpoint's phoneme_id_map via `--data.phonemes_path`), starting
# from the Phonikud checkpoint the mirror's current Hebrew voice was exported from.
# The ids were made on the Mac by the mirror's own G2P, so nothing Hebrew-specific
# (no G2P, no espeak phonemization) runs here.
#
# Written for Ubuntu 22.04/24.04 + an NVIDIA GPU with a recent driver (RunPod/Lambda
# PyTorch images are fine). It has NOT been run end to end; see the UNVERIFIED notes.
# Run it inside tmux (`tmux new -s train`) so a dropped ssh doesn't kill training.
set -euo pipefail

# ---- settings (override any of them from the environment) ---------------------------
WORK="${WORK:-$HOME/piper-he}"                 # everything lives under here
DATA_DIR="${DATA_DIR:-$WORK/train_he}"         # the rsynced output of prepare_piper_dataset.py
PIPER_REF="${PIPER_REF:-v1.8.0}"               # piper1-gpl tag (commit 639388b6, 2026-09-04)
CKPT_REPO="${CKPT_REPO:-https://huggingface.co/Phonikud/phonikud-tts-checkpoints/resolve/main}"
# The base checkpoint: its export is byte-identical to models/phonikud-tts-he.onnx
# (model.onnx, sha256 dfe0a8f3...). 846 MB.
CKPT_FILE="${CKPT_FILE:-epoch=4842-step=3124000.ckpt}"
CKPT_SHA256="${CKPT_SHA256:-36b9658dfcd8609d926bc75c6b765a8b68c0501cd1ef9dde97353c42169294fd}"
VOICE_NAME="${VOICE_NAME:-he_IL-oracle-medium}"
RUN_DIR="${RUN_DIR:-$WORK/run}"                # lightning_logs/, checkpoints, tensorboard
OUT_DIR="${OUT_DIR:-$WORK/out}"                # exported voice + samples
# START_MODE=warmstart: copy the checkpoint's weights, fresh optimizer, epochs count
#   from 0 (piper1-gpl --model.warmstart_ckpt). Robust to the checkpoint being from the
#   legacy rhasspy trainer (PyTorch Lightning 1.7.7).
# START_MODE=resume: Lightning --ckpt_path, restoring optimizer + epoch 4842 too (what
#   piper's docs recommend); MAX_EPOCHS then counts from 4843.
START_MODE="${START_MODE:-warmstart}"
BATCH_SIZE="${BATCH_SIZE:-32}"                 # 24 GB GPU: 32 is fine; 16 on a 12 GB card
LEARNING_RATE="${LEARNING_RATE:-1e-4}"         # where the base run's decayed LR was (1.09e-4)
MAX_EPOCHS="${MAX_EPOCHS:-1000}"               # an upper bound: stop earlier by ear
VAL_EVERY="${VAL_EVERY:-10}"                   # validation (+ tensorboard audio) every N epochs
VALIDATION_SPLIT="${VALIDATION_SPLIT:-0.05}"   # ~60 of ~1,200 clips
NUM_WORKERS="${NUM_WORKERS:-4}"
TORCH_VERSION="${TORCH_VERSION:-2.11.*}"       # torch + torchaudio pair (see setup)
TORCH_CUDA="${TORCH_CUDA:-cu130}"              # wheel index: cu130 needs a CUDA 13 driver; cu128 otherwise
PY="$WORK/piper1-gpl/.venv/bin/python"
# --------------------------------------------------------------------------------------

log() { printf '\n== %s\n' "$*"; }

latest_ckpt() {  # newest checkpoint of our own run (empty if none)
    { find "$RUN_DIR" -name '*.ckpt' -printf '%T@ %p\n' 2>/dev/null || true; } \
        | sort -n | tail -1 | cut -d' ' -f2-
}

setup() {
    log "system packages"
    if command -v apt-get >/dev/null; then
        SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO=sudo
        $SUDO apt-get update -qq
        $SUDO apt-get install -y -qq build-essential cmake ninja-build git curl python3-venv \
            python3-dev tmux >/dev/null
    fi
    nvidia-smi || { echo "no NVIDIA GPU visible (nvidia-smi failed)" >&2; exit 1; }
    mkdir -p "$WORK" "$RUN_DIR" "$OUT_DIR"

    log "piper1-gpl $PIPER_REF"
    if [ ! -d "$WORK/piper1-gpl/.git" ]; then
        git clone https://github.com/OHF-Voice/piper1-gpl.git "$WORK/piper1-gpl"
    fi
    git -C "$WORK/piper1-gpl" fetch --tags -q
    git -C "$WORK/piper1-gpl" checkout -q "$PIPER_REF"
    if [ ! -x "$PY" ]; then
        python3 -m venv "$WORK/piper1-gpl/.venv"
    fi
    (
        cd "$WORK/piper1-gpl"
        # pip picks the default CUDA build of torch (2.x). If torch.cuda.is_available()
        # is False below, install the wheel matching the driver first, e.g.
        #   .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu124
        .venv/bin/pip install -q --upgrade pip wheel setuptools
        .venv/bin/pip install -q -e '.[train]'
        # The UTMOS predictor (val_mos) needs torchaudio, and v1.8.0's val_mos
        # ModelCheckpoint makes Lightning 2.6 raise at the first validation without it
        # (piper's comment says it only warns). torchaudio stopped at 2.11, so pin
        # torch to the matching pair (seen 2026-10-23: pip had pulled torch 2.14).
        .venv/bin/pip install -q "torch==$TORCH_VERSION" "torchaudio==$TORCH_VERSION" \
            --index-url "https://download.pytorch.org/whl/$TORCH_CUDA"
        # The monotonic alignment Cython extension (docs/TRAINING.md), plus the in-repo
        # dev build of the package's C extension.
        .venv/bin/pip install -q scikit-build cmake ninja
        ./build_monotonic_align.sh
        .venv/bin/python setup.py build_ext --inplace
        # torch >= 2.9 exports with dynamo (needs onnxscript, and gives a different
        # graph); the TorchScript exporter makes the same graph as the shipped voice.
        grep -q 'dynamo=False' src/piper/train/export_onnx.py ||
            sed -i 's/^        verbose=False,$/        verbose=False,\n        dynamo=False,/' \
                src/piper/train/export_onnx.py
    )
    "$PY" -c "import torch; assert torch.cuda.is_available(), 'torch has no CUDA'; \
print('torch', torch.__version__, torch.cuda.get_device_name(0))"

    log "checkpoint $CKPT_FILE"
    local ckpt="$WORK/ckpt/$CKPT_FILE"
    mkdir -p "$WORK/ckpt"
    if [ ! -s "$ckpt" ]; then
        # '=' must be %-encoded in the URL
        curl -fL --retry 5 -C - -o "$ckpt.part" "$CKPT_REPO/${CKPT_FILE//=/%3D}"
        mv "$ckpt.part" "$ckpt"
    fi
    echo "$CKPT_SHA256  $ckpt" | sha256sum -c -
}

check_data() {
    for f in metadata_ids.csv phoneme_id_map.json config.json wavs; do
        [ -e "$DATA_DIR/$f" ] || { echo "missing $DATA_DIR/$f: run tools/prepare_piper_dataset.py and rsync train_he/ here" >&2; exit 1; }
    done
    echo "$(wc -l < "$DATA_DIR/metadata_ids.csv") clips in $DATA_DIR"
}

train() {
    check_data
    [ -x "$PY" ] || { echo "run setup first" >&2; exit 1; }
    local start=() resume epochs="$MAX_EPOCHS"
    # A full resume restores the base run's epoch counter (4842), so count on from it.
    [ "$START_MODE" = resume ] && epochs="$((4843 + MAX_EPOCHS))"
    resume="$(latest_ckpt)"
    if [ -n "$resume" ]; then
        # Our own run was interrupted: carry on from its newest checkpoint. (Never pass
        # warmstart_ckpt here: it is applied in on_fit_start, after the restore, and
        # would overwrite the resumed weights.)
        log "resuming our run from $resume"
        start=(--ckpt_path "$resume")
    elif [ "$START_MODE" = resume ]; then
        log "starting from $CKPT_FILE (full Lightning resume)"
        start=(--ckpt_path "$WORK/ckpt/$CKPT_FILE")
    else
        log "starting from $CKPT_FILE (weights only)"
        start=(--model.warmstart_ckpt "$WORK/ckpt/$CKPT_FILE")
    fi
    start+=(--trainer.max_epochs "$epochs")

    # Notes on the flags:
    # - dataset_type phoneme_ids: CSV is id|text|ids and the ids are used as-is (piper's
    #   docs/TRAINING.md calls it --data.data_type, but the v1.8.0 code's argument is
    #   dataset_type). phonemes_path copies the checkpoint's map into the written config.
    # - espeak_voice is required by the CLI but unused with phoneme ids.
    # - the model defaults of piper1-gpl are the medium architecture, matching the
    #   checkpoint's hyperparameters (22050 Hz, resblock 2, upsample 8,8,4, 192 channels).
    # - trim_silence (Silero VAD, keeps 0.25 s) is piper1-gpl's default; ElevenLabs clips
    #   have little silence, so it changes little.
    # - the config it writes ($RUN_DIR/piper_config.json) is NOT used for the voice;
    #   export uses $DATA_DIR/config.json, the checkpoint's own config.
    cd "$WORK/piper1-gpl"
    "$PY" -m piper.train fit \
        --data.voice_name "$VOICE_NAME" \
        --data.csv_path "$DATA_DIR/metadata_ids.csv" \
        --data.audio_dir "$DATA_DIR/wavs" \
        --data.dataset_type phoneme_ids \
        --data.phonemes_path "$DATA_DIR/phoneme_id_map.json" \
        --data.num_symbols 256 \
        --data.espeak_voice he \
        --data.cache_dir "$WORK/cache" \
        --data.config_path "$RUN_DIR/piper_config.json" \
        --data.batch_size "$BATCH_SIZE" \
        --data.validation_split "$VALIDATION_SPLIT" \
        --data.num_test_examples 5 \
        --data.num_workers "$NUM_WORKERS" \
        --model.sample_rate 22050 \
        --model.learning_rate "$LEARNING_RATE" \
        --trainer.default_root_dir "$RUN_DIR" \
        --trainer.check_val_every_n_epoch "$VAL_EVERY" \
        "${start[@]}" 2>&1 | tee -a "$RUN_DIR/train.log"
}

export_onnx() {  # export_onnx CKPT ONNX
    (cd "$WORK/piper1-gpl" && "$PY" -m piper.train.export_onnx --checkpoint "$1" --output-file "$2")
    cp "$DATA_DIR/config.json" "$2.json"
}

sample() {
    local ckpt="${1:-$(latest_ckpt)}"
    [ -n "$ckpt" ] || { echo "no checkpoint in $RUN_DIR yet" >&2; exit 1; }
    local tag; tag="$(basename "$ckpt" .ckpt | tr -c 'A-Za-z0-9=._-' '_')"
    local dir="$OUT_DIR/samples/$tag"
    mkdir -p "$dir"
    log "samples from $ckpt -> $dir"
    export_onnx "$ckpt" "$dir/voice.onnx"
    # samples_ids.csv: held-out sentences, phoneme ids made on the Mac by the mirror's G2P
    "$PY" - "$dir/voice.onnx" "$DATA_DIR/samples_ids.csv" "$dir" <<'EOF'
import csv, sys, wave
import numpy as np, onnxruntime as ort
model, ids_csv, out = sys.argv[1:]
sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
for sid, text, ids in csv.reader(open(ids_csv, encoding="utf-8"), delimiter="|"):
    x = np.array([[int(i) for i in ids.split()]], dtype=np.int64)
    audio = sess.run(None, {"input": x, "input_lengths": np.array([x.shape[1]], np.int64),
                            "scales": np.array([0.667, 1.0, 0.8], np.float32)})[0].squeeze()
    pcm = (np.clip(audio / max(1e-6, float(np.abs(audio).max())), -1, 1) * 32767 * 0.9).astype("<i2")
    with wave.open(f"{out}/{sid}.wav", "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(22050); w.writeframes(pcm.tobytes())
    print(f"{out}/{sid}.wav  {text}")
EOF
    echo "copy them home:  rsync -av BOX:$dir/ ./samples/$tag/"
}

export_final() {
    local ckpt="${1:-$(latest_ckpt)}"
    [ -n "$ckpt" ] || { echo "no checkpoint in $RUN_DIR yet" >&2; exit 1; }
    mkdir -p "$OUT_DIR"
    log "exporting $ckpt"
    export_onnx "$ckpt" "$OUT_DIR/$VOICE_NAME.onnx"
    ls -la "$OUT_DIR/$VOICE_NAME.onnx" "$OUT_DIR/$VOICE_NAME.onnx.json"
    echo "copy home:  rsync -av BOX:$OUT_DIR/$VOICE_NAME.onnx* models/"
}

case "${1:-}" in
    setup) setup ;;
    train) train ;;
    sample) sample "${2:-}" ;;
    export) export_final "${2:-}" ;;
    all) setup; train; export_final ;;
    *) sed -n '2,12p' "$0"; exit 2 ;;
esac

# Time and when to stop (estimates, UNVERIFIED on this exact setup):
# - ~1,200 train clips / batch 32 = ~38 steps per epoch. A medium VITS step at batch 32
#   on an RTX 4090 / A100 is very roughly 0.5-1 s, so ~20-40 s per epoch, ~100 epochs per
#   40-60 min. Read the real rate off the progress bar in the first minutes.
# - A fine-tune from a same-language checkpoint usually sounds like the new speaker after
#   a few hundred epochs (~10k steps) and stops improving somewhere around 500-1000
#   (20-40k steps). Run `sample` every ~100 epochs (in a second tmux window, while
#   training continues) and compare; stop when two rounds in a row sound the same, or
#   if it starts to sound worse (overfitting on 90 min: breathy, mumbled, or garbled on
#   the held-out lines). Budget: 2-5 h of GPU time.
# - TensorBoard has the losses and, every VAL_EVERY epochs, audio for the 5 test clips:
#     $PY -m tensorboard.main --logdir $RUN_DIR --bind_all --port 6006
#   and on the Mac `ssh -L 6006:localhost:6006 BOX`, then http://localhost:6006.
# - Checkpoints: lightning_logs/version_*/checkpoints/ keeps last.ckpt and the 5 best by
#   val_mel (and by val_mos if the UTMOS predictor loaded). Pick by ear, not by number:
#   `deploy/train_piper_hebrew.sh export PATH/TO/epoch=....ckpt`.
