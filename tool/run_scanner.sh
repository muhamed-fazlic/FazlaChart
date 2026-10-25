#!/usr/bin/env bash
# Runs the Market Scanner using the project's virtual environment.
# Forwards all arguments to `python -m scanner`. See `python -m scanner --help`.
#
# Usage:
#   ./tool/run_scanner.sh --once
#   ./tool/run_scanner.sh --symbols BTCUSDT,ETHUSDT --interval 60
#   ./tool/run_scanner.sh --top-n 60            # default: loops forever, every 5 min

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ ! -x ".venv/bin/python" ]; then
    echo "No .venv found. Run ./tool/setup.sh first." >&2
    exit 1
fi

export PYTHONIOENCODING=utf-8
exec .venv/bin/python -m scanner "$@"
