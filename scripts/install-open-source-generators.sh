#!/usr/bin/env bash
set -euo pipefail

# Installs the Apache-2.0 local image-to-image engines. Generation still
# requires an NVIDIA CUDA machine; this script does not enable a hosted API.
ROOT="${OPEN_SOURCE_MODELS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.local}"
QWEN_ROOT="${QWEN_ROOT:-$ROOT/qwen}"
FLUX_ROOT="${FLUX_SCHNELL_ROOT:-$ROOT/flux-schnell}"
QWEN_MODEL_ID="${QWEN_MODEL_ID:-Qwen/Qwen-Image-Edit-2511}"
FLUX_MODEL_ID="${FLUX_SCHNELL_MODEL_ID:-black-forest-labs/FLUX.1-schnell}"

mkdir -p "$QWEN_ROOT" "$FLUX_ROOT"

for target in qwen flux; do
  if [[ "$target" == "qwen" ]]; then
    venv="$QWEN_ROOT/.venv"
  else
    venv="$FLUX_ROOT/.venv"
  fi
  if [[ ! -x "$venv/bin/python" ]]; then
    uv venv --python 3.11 "$venv"
  fi
  uv pip install --python "$venv/bin/python" --index-url https://pypi.org/simple \
    "torch>=2.10" torchvision "diffusers>=0.35.0" transformers accelerate \
    safetensors huggingface_hub sentencepiece protobuf numpy pillow
done

echo "Downloading Qwen Image Edit 2511..."
hf download "$QWEN_MODEL_ID" --local-dir "$QWEN_ROOT/Qwen-Image-Edit-2511"
echo "Downloading FLUX.1 Schnell..."
hf download "$FLUX_MODEL_ID" --local-dir "$FLUX_ROOT/FLUX.1-schnell"

echo
echo "Open-source engines installed."
echo "Qwen:        $QWEN_ROOT/Qwen-Image-Edit-2511"
echo "FLUX Schnell: $FLUX_ROOT/FLUX.1-schnell"
echo "Both require a CUDA-capable NVIDIA GPU at generation time."