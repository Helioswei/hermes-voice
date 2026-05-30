#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON=""
for candidate in python3.13 python3.12 python3; do
    if command -v "$candidate" &>/dev/null; then
        PYTHON="$candidate"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "Error: Python 3.12+ is required. Install it via 'brew install python@3.13'"
    exit 1
fi

echo "==> Installing system dependencies …"
if ! command -v brew &>/dev/null; then
    echo "Error: Homebrew is required. Install it from https://brew.sh"
    exit 1
fi

brew list portaudio &>/dev/null || brew install portaudio

echo "==> Creating virtual environment …"
"$PYTHON" -m venv .venv
source .venv/bin/activate

echo "==> Installing Python dependencies …"
pip install --upgrade pip
pip install -r requirements.txt

echo "==> Done!"
echo ""
echo "Before running, make sure you have:"
echo "  1. Hermes Gateway running  (hermes gateway)"
echo "  2. HERMES_API_KEY set in your environment"
echo ""
echo "Usage:  source .venv/bin/activate && python -m voice.main"
echo "Quick:  bash scripts/start.sh"
