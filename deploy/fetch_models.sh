#!/usr/bin/env bash
# Download every model the mirror needs into MODELS_DIR (run once, online).
#   deploy/fetch_models.sh MODELS_DIR
# Files that already exist are skipped; interrupted downloads resume.
# Set PYTHON to the interpreter that has huggingface_hub (default: python3), and
# WHISPER_QUANTIZE to whisper.cpp's quantize tool (default: whisper-quantize on PATH).
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "usage: $0 MODELS_DIR" >&2
    exit 2
fi
DIR="$1"
PYTHON="${PYTHON:-python3}"
WHISPER_QUANTIZE="${WHISPER_QUANTIZE:-whisper-quantize}"
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

# Stock multilingual small: the fallback STT model (STT_MODEL=ggml-small.bin).
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

# The default STT model: ivrit-ai's Hebrew Whisper large-v3-turbo (Apache-2.0), the
# official ggml export (f16 only) at a pinned revision, quantized here to q5_0.
# The f16 is deleted once the q5_0 exists. Last, so a missing whisper-quantize
# does not stop the other downloads.
IVRIT="$HF/ivrit-ai/whisper-large-v3-turbo-ggml/resolve/2130c78e4a9cb4914cc4df91a1c3031407789705"
if [ -s "$DIR/ivrit-turbo-q5_0.bin" ]; then
    echo "skip  ivrit-turbo-q5_0.bin (exists)"
elif ! command -v "$WHISPER_QUANTIZE" >/dev/null; then
    echo "ERROR: $WHISPER_QUANTIZE not found, so ivrit-turbo-q5_0.bin (the default STT model)" >&2
    echo "       was not made; everything else is in $DIR. Install it (brew install whisper-cpp)" >&2
    echo "       or set WHISPER_QUANTIZE, then re-run this script." >&2
    exit 1
else
    fetch "$IVRIT/ggml-model.bin" ivrit-turbo-f16.bin
    echo "quantize ivrit-turbo-f16.bin -> ivrit-turbo-q5_0.bin"
    "$WHISPER_QUANTIZE" "$DIR/ivrit-turbo-f16.bin" "$DIR/ivrit-turbo-q5_0.bin.part" q5_0
    mv "$DIR/ivrit-turbo-q5_0.bin.part" "$DIR/ivrit-turbo-q5_0.bin"
    rm -f "$DIR/ivrit-turbo-f16.bin"
fi
echo "done"
