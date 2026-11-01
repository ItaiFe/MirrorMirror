#!/usr/bin/env bash
# Fine-tune the oracle's English Piper (VITS, medium) voice on a rented CUDA GPU box.
#
#   deploy/train_piper_english.sh setup     # system packages, piper1-gpl, the checkpoint
#   deploy/train_piper_english.sh train     # fine-tune (resumes its own run if restarted)
#   deploy/train_piper_english.sh sample    # export the latest checkpoint, synthesize samples/
#   deploy/train_piper_english.sh export [CKPT]   # final .onnx + .onnx.json into $OUT_DIR
#   deploy/train_piper_english.sh all       # setup + train + export
#
# Before: on the Mac, `tools/prepare_piper_dataset.py --g2p espeak-en` writes train_en/
# (phoneme ids from piper's own espeak-ng en-us, exactly what PiperTTS feeds the lessac
# voice) and you rsync it to $DATA_DIR on this box.
#
# The same steps, flags and fixes as deploy/train_piper_hebrew.sh (piper1-gpl v1.8.0,
# --data.dataset_type phoneme_ids, the torch/torchaudio 2.11 pin, dynamo=False export);
# only the settings differ: it warm-starts from rhasspy's en_US lessac medium
# checkpoint, the one models/en_US-lessac-medium.onnx was exported from, so the new
# voice keeps lessac's phoneme_id_map and takes the very same input.
# Every setting below (and in the Hebrew script) can still be overridden from the
# environment. UNVERIFIED end to end, like the Hebrew script.
set -euo pipefail

export WORK="${WORK:-$HOME/piper-en}"
export DATA_DIR="${DATA_DIR:-$WORK/train_en}"
export PREP_OUT="${PREP_OUT:-train_en}"
export CKPT_REPO="${CKPT_REPO:-https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/en/en_US/lessac/medium}"
# 845,898,328 bytes; sha256 = the Hugging Face x-linked-etag of the file.
export CKPT_FILE="${CKPT_FILE:-epoch=2164-step=1355540.ckpt}"
export CKPT_SHA256="${CKPT_SHA256:-ab7e5b8dab40f834b7cc58ae4ad7b7009c954b901e4ddbb784bbe11ce379a1cd}"
export CKPT_EPOCH="${CKPT_EPOCH:-2164}"
export VOICE_NAME="${VOICE_NAME:-en_US-oracle-medium}"
export ESPEAK_VOICE="${ESPEAK_VOICE:-en-us}"
export RUN_DIR="${RUN_DIR:-$WORK/run}"
export OUT_DIR="${OUT_DIR:-$WORK/out}"

case "${1:-}" in
    setup|train|sample|export|all)
        exec bash "$(dirname "${BASH_SOURCE[0]}")/train_piper_hebrew.sh" "$@" ;;
    *) sed -n '2,12p' "$0"; exit 2 ;;
esac
