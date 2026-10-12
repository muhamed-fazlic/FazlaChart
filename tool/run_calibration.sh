#!/usr/bin/env bash
# Runs the label calibration backtest using the project's virtual environment.
# Forwards all arguments to `python -m scanner.calibrate`.
#
# Usage:
#   ./tool/run_calibration.sh --symbols BTCUSDT,ETHUSDT,INJUSDT --timeframe 1h
#   ./tool/run_calibration.sh --symbols SOLUSDT --timeframe 15m --horizons 4,12,48

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ ! -x ".venv/bin/python" ]; then
    echo "No .venv found. Run ./tool/setup.sh first." >&2
    exit 1
fi

if [ "$#" -eq 0 ]; then
    set -- --symbols BTCUSDT,ETHUSDT,INJUSDT --timeframe 1h --horizons 4,12,24
fi

export PYTHONIOENCODING=utf-8
exec .venv/bin/python -m scanner.calibrate "$@"
