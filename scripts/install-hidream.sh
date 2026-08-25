#!/usr/bin/env bash
set -euo pipefail

# Low-memory HiDream installer.
#
# The official Hugging Face snapshot downloader starts all 8 multi-gigabyte
# shards together and can be killed on small machines. This installer downloads
# one file at a time, so it is safe to resume and does not require the whole
# model in RAM.

ROOT="${HIDREAM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.local/hidream}"
REPO_DIR="${HIDREAM_REPO:-$ROOT/HiDream-O1-Image}"
MODEL_DIR="${HIDREAM_MODEL_PATH:-$ROOT/HiDream-O1-Image-Dev}"
PYTHON_BIN="${HIDREAM_PYTHON:-$ROOT/.venv/bin/python}"
MODEL_ID="${HIDREAM_MODEL_ID:-HiDream-ai/HiDream-O1-Image-Dev}"

mkdir -p "$ROOT"

if [[ ! -f "$REPO_DIR/inference.py" ]]; then
  echo "Cloning official HiDream repository..."
  git clone --depth 1 https://github.com/HiDream-ai/HiDream-O1-Image.git "$REPO_DIR"
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Creating isolated Python environment..."
  uv venv --python 3.11 "$ROOT/.venv"
fi

echo "Installing HiDream Python dependencies..."
uv pip install --python "$PYTHON_BIN" --index-url https://pypi.org/simple \
  "torch>=2.10" torchvision "transformers==4.57.1" diffusers accelerate \
  einops scipy flask openai huggingface_hub safetensors numpy pillow tqdm \
  python-dotenv

mkdir -p "$MODEL_DIR"
echo "Downloading model one file at a time..."

# Keep this list explicit: it matches the public Dev model snapshot and avoids
# the concurrent multi-file behavior of `hf download`.
files=(
  config.json
  configuration.json
  generation_config.json
  merges.txt
  preprocessor_config.json
  tokenizer.json
  tokenizer_config.json
  video_preprocessor_config.json
  vocab.json
  model.safetensors.index.json
  model-00001-of-00008.safetensors
  model-00002-of-00008.safetensors
  model-00003-of-00008.safetensors
  model-00004-of-00008.safetensors
  model-00005-of-00008.safetensors
  model-00006-of-00008.safetensors
  model-00007-of-00008.safetensors
  model-00008-of-00008.safetensors
)

for file in "${files[@]}"; do
  echo "  -> $file"
  hf download "$MODEL_ID" "$file" --local-dir "$MODEL_DIR"
done

echo
echo "HiDream installation complete."
echo "Repository: $REPO_DIR"
echo "Model:      $MODEL_DIR"
echo "Python:     $PYTHON_BIN"
echo
echo "Generation still requires an NVIDIA CUDA GPU."