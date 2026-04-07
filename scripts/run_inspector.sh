#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ENV_NAME="${1:-}"
TASK_ID="${2:-}"

if [ -z "$ENV_NAME" ] || [ -z "$TASK_ID" ]; then
  echo "Usage: bash scripts/run_inspector.sh <env-name> <task-id>"
  exit 1
fi

if [ ! -d ".venv" ]; then
  echo "Missing .venv. Run scripts/setup.sh first."
  exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Exporting env=${ENV_NAME} task=${TASK_ID}"
python scripts/export_libero_env.py --env "$ENV_NAME" --task-id "$TASK_ID"

echo "==> Open:"
echo "http://localhost:5173/?env=${ENV_NAME}&task=${TASK_ID}"

echo "==> Starting frontend"
cd "$ROOT_DIR/frontend"
pnpm dev --host 0.0.0.0