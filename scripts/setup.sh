#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "==> Initializing git submodules"
git submodule update --init --recursive

echo "==> Checking uv"
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed."
  echo "Install it with:"
  echo '  curl -LsSf https://astral.sh/uv/install.sh | sh'
  exit 1
fi

echo "==> Checking node"
if ! command -v node >/dev/null 2>&1; then
  echo "node is not installed."
  exit 1
fi

echo "==> Checking pnpm"
if ! command -v pnpm >/dev/null 2>&1; then
  echo "pnpm is not installed."
  echo "Install it with:"
  echo "  npm install -g pnpm"
  exit 1
fi

echo "==> Creating Python venv"
if [ ! -d ".venv" ]; then
  uv venv --python 3.10 --seed
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Upgrading pip tooling"
python -m pip install --upgrade pip setuptools wheel

echo "==> Installing Python project"
uv pip install -e .

echo "==> Installing LIBERO requirements"
python -m pip install -r ./third_party/envs/libero/requirements.txt

echo "==> Installing torch build"
python -m pip install \
  torch==1.11.0+cu113 \
  torchvision==0.12.0+cu113 \
  torchaudio==0.11.0 \
  --extra-index-url https://download.pytorch.org/whl/cu113

echo "==> Installing LIBERO editable"
uv pip install -e ./third_party/envs/libero

if [ -d "./third_party/policies/xvla" ]; then
  echo "==> XVLA submodule detected at third_party/policies/xvla"
  echo "==> Installing XVLA requirements"
  python -m pip install -r ./third_party/policies/xvla/requirements.txt
fi

echo "==> Installing frontend dependencies"
cd "$ROOT_DIR/frontend"
pnpm install

echo "==> Ensuring runtime export directory exists"
mkdir -p "$ROOT_DIR/exports"

echo "==> Setup complete"
echo "Activate python env with: source $ROOT_DIR/.venv/bin/activate"
echo "Run inspector with: bash $ROOT_DIR/scripts/run_inspector.sh libero-spatial-v0 0"
