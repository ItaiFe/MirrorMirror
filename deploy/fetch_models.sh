#!/usr/bin/env bash
# Download every model the mirror needs into MODELS_DIR (run once, online).
#   deploy/fetch_models.sh MODELS_DIR
# Files that already exist are skipped; interrupted downloads resume.
# Set PYTHON to the interpreter that has huggingface_hub (default: python3).
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "usage: $0 MODELS_DIR" >&2
    exit 2
fi
DIR="$1"
PYTHON="${PYTHON:-python3}"
mkdir -p "$DIR"

HF=https://huggingface.co
fetch() {  # fetch URL FILENAME
    local url="$1" out="$DIR/$2"
    if [ -s "$out" ]; then
        echo "skip  $2 (exists)"
        return
    fi
    echo "fetch $2"
    curl -fL --retry 3 -C - -o "$out.part" "$url"
    mv "$out.part" "$out"
}

fetch "$HF/ggerganov/whisper.cpp/resolve/main/ggml-small.bin" ggml-small.bin
fetch "$HF/ggml-org/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it-Q4_K_M.gguf" \
    gemma-3-4b-it-Q4_K_M.gguf
fetch "https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.onnx" \
    silero_vad.onnx
PIPER="$HF/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium"
fetch "$PIPER/en_US-lessac-medium.onnx" en_US-lessac-medium.onnx
fetch "$PIPER/en_US-lessac-medium.onnx.json" en_US-lessac-medium.onnx.json

echo "cache facebook/mms-tts-heb -> $DIR/hf"
HF_HOME="$DIR/hf" "$PYTHON" -c \
    "from huggingface_hub import snapshot_download; snapshot_download('facebook/mms-tts-heb', ignore_patterns=['*.bin'])"
echo "done"
