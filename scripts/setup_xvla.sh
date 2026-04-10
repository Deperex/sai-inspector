#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "==> Checking uv"
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed."
  echo "Install it with:"
  echo '  curl -LsSf https://astral.sh/uv/install.sh | sh'
  exit 1
fi

XVLA_DIR="$ROOT_DIR/third_party/policies/xvla"
if [ ! -d "$XVLA_DIR" ]; then
  echo "XVLA submodule not found at $XVLA_DIR"
  exit 1
fi

echo "==> Creating XVLA venv"
if [ ! -d ".venv-xvla" ]; then
  uv venv .venv-xvla --python 3.10 --seed
fi

# shellcheck disable=SC1091
source .venv-xvla/bin/activate

echo "==> Upgrading pip tooling"
python -m pip install --upgrade pip setuptools wheel

echo "==> Installing XVLA requirements"
python -m pip install -r "$XVLA_DIR/requirements.txt"

echo "==> Installing CUDA-compatible PyTorch for this VM"
python -m pip uninstall -y torch torchvision torchaudio || true
python -m pip install --no-cache-dir --force-reinstall \
  torch==2.6.0 \
  torchvision==0.21.0 \
  torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124

echo "==> XVLA setup complete"
echo "Activate XVLA env with: source $ROOT_DIR/.venv-xvla/bin/activate"
