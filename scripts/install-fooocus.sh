#!/usr/bin/env bash
set -euo pipefail

# Installs the official Fooocus runtime in an isolated Python 3.10
# environment. Model checkpoints are intentionally not downloaded here:
# Fooocus is an SDXL CUDA backend and CPU generation is not a viable
# production path for Atelier.
ROOT="${FOOOCUS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.local/fooocus}"
REPO_URL="${FOOOCUS_REPO_URL:-https://github.com/lllyasviel/Fooocus.git}"
PYTHON_BIN="${FOOOCUS_PYTHON_BIN:-$(command -v python3.10 || printf '%s' python3.10)}"
VENV="$ROOT/.venv"

if [[ ! -d "$ROOT/.git" ]]; then
  mkdir -p "$(dirname "$ROOT")"
  git clone --depth=1 "$REPO_URL" "$ROOT"
else
  git -C "$ROOT" fetch --depth=1 origin main
  git -C "$ROOT" reset --hard origin/main
fi

if [[ ! -x "$VENV/bin/python" ]]; then
  uv venv --python "$PYTHON_BIN" "$VENV"
fi

CUDA_AVAILABLE=false
if [[ -e /dev/nvidiactl ]] || compgen -G "/dev/nvidia[0-9]*" >/dev/null; then
  CUDA_AVAILABLE=true
fi

if [[ "${FOOOCUS_INSTALL_DEPS:-auto}" == "true" ]] || [[ "$CUDA_AVAILABLE" == "true" && "${FOOOCUS_INSTALL_DEPS:-auto}" != "false" ]]; then
  # Fooocus's launcher normally installs a CUDA PyTorch build automatically.
  uv pip install --python "$VENV/bin/python" \
    --no-cache \
    --index-url https://download.pytorch.org/whl/cu121 \
    "torch==2.1.0" "torchvision==0.16.0"
  uv pip install --python "$VENV/bin/python" \
    --no-cache \
    --index-url https://pypi.org/simple \
    -r "$ROOT/requirements_versions.txt"
else
  echo "No NVIDIA GPU detected; skipping Fooocus's heavyweight runtime dependencies."
  echo "Set FOOOCUS_INSTALL_DEPS=true on a CUDA machine to finish installation."
fi

cat <<EOF
Fooocus runtime installed:
  repository: $ROOT
  python:     $VENV/bin/python
  models:     skipped (CUDA worker required)

On a CUDA machine, install the NVIDIA PyTorch build in this venv and let
Fooocus download its SDXL checkpoints before enabling the provider.
EOF