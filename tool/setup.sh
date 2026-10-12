#!/usr/bin/env bash
# One-time (or repeatable) environment setup for FazlaChart on Linux/macOS.
# Creates .venv (if missing) and installs the project in editable mode with dev extras.
#
# Usage:
#   ./tool/setup.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"

if [ ! -x ".venv/bin/python" ]; then
    echo "Creating virtual environment in .venv ..."
    "$PYTHON_BIN" -m venv .venv
else
    echo "Virtual environment already exists, reusing .venv"
fi

echo "Installing project (editable, with dev extras) ..."
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[dev]"

if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    cp .env.example .env
    echo "Created .env from .env.example (fill in API keys only when you reach the LLM-agent phase)."
fi

echo ""
echo "Setup complete. Try:"
echo "  ./tool/run_scanner.sh --once"
echo "  ./tool/run_calibration.sh"
