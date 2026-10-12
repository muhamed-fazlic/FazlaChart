#!/usr/bin/env bash
# Runs the Derivatives/Risk agent locally against a single symbol, using Ollama by default.
# Requires Ollama running locally (https://ollama.com) with the target model pulled:
#   ollama pull llama3.1:8b
#
# Usage:
#   ./tool/run_derivatives_agent.sh --symbol INJUSDT --timeframe 1h
#   ./tool/run_derivatives_agent.sh --symbol BTCUSDT --timeframe 4h --model llama3.1:8b

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ ! -x ".venv/bin/python" ]; then
    echo "No .venv found. Run ./tool/setup.sh first." >&2
    exit 1
fi

if [ "$#" -eq 0 ]; then
    echo "Usage: $0 --symbol INJUSDT [--timeframe 1h] [--horizon-candles 12] [--provider ollama] [--model llama3.1:8b]" >&2
    exit 1
fi

export PYTHONIOENCODING=utf-8
exec .venv/bin/python -m agents_crypto "$@"
