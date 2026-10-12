#!/usr/bin/env bash
# Starts the local web dashboard (reads scanner_data/scanner.db).
# The scanner itself (./tool/run_scanner.sh) should be running separately;
# this process only reads SQLite, it never calls Binance itself.
#
# Usage:
#   ./tool/run_dashboard.sh            # http://127.0.0.1:8000
#   PORT=9000 ./tool/run_dashboard.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ ! -x ".venv/bin/python" ]; then
    echo "No .venv found. Run ./tool/setup.sh first." >&2
    exit 1
fi

PORT="${PORT:-8000}"
echo "Dashboard starting at http://0.0.0.0:${PORT} (Ctrl+C to stop)"
exec .venv/bin/python -m uvicorn webapp.main:app --host 0.0.0.0 --port "$PORT"
