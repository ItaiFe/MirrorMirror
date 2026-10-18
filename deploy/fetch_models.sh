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
# Silero VAD pinned to the v5.1.2 tag (master moves).
fetch "https://raw.githubusercontent.com/snakers4/silero-vad/v5.1.2/src/silero_vad/data/silero_vad.onnx" \
    silero_vad.onnx
PIPER="$HF/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium"
fetch "$PIPER/en_US-lessac-medium.onnx" en_US-lessac-medium.onnx
fetch "$PIPER/en_US-lessac-medium.onnx.json" en_US-lessac-medium.onnx.json

# Hebrew voice: Phonikud G2P (MIT) + Phonikud Piper voice (non-commercial, see
# https://huggingface.co/Phonikud/phonikud-tts-checkpoints), pinned revisions.
PHONIKUD_G2P="$HF/Phonikud/phonikud-onnx/resolve/b806189fe1fc0085b1012b7560ffb5e8ecfd72a2"
PHONIKUD_TTS="$HF/Phonikud/phonikud-tts-checkpoints/resolve/ff64ba389a3833c49a85c58fc51a7060a6984b87"
fetch "$PHONIKUD_G2P/phonikud-1.0.int8.onnx" phonikud-1.0.int8.onnx
fetch "$PHONIKUD_TTS/model.onnx" phonikud-tts-he.onnx
fetch "$PHONIKUD_TTS/model.config.json" phonikud-tts-he.onnx.json
# RenikudPlus G2P (Apache-2.0), the int8 model, pinned revision. It is loaded from
# this path with no rescoring datastore, so it never touches the network at runtime.
RENIKUD="$HF/notmax123/RenikudPlus/resolve/679c56ca449d41873fb8ff7711ddf7563d28198f"
fetch "$RENIKUD/model_int8.onnx" renikud-plus.int8.onnx

# Hugging Face cache: Phonikud's tokenizer (loaded by name, so on the main ref)
# and the MMS Hebrew fallback voice.
echo "cache dicta-il/dictabert-large-char-menaked tokenizer, facebook/mms-tts-heb -> $DIR/hf"
HF_HOME="$DIR/hf" "$PYTHON" -c "
from huggingface_hub import snapshot_download
snapshot_download('dicta-il/dictabert-large-char-menaked', allow_patterns=['tokenizer.json'])
snapshot_download('facebook/mms-tts-heb', ignore_patterns=['*.bin'])"
echo "done"
