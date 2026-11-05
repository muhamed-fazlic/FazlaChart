#!/usr/bin/env bash
# Phase 8 "canonical" portfolio backtest -- the ONE run that actually combines
# every bias-fix/execution-realism feature built in Korak 2-5 instead of each
# being reachable only via its own separately-remembered CLI flag:
#   - point-in-time universe filter (Korak 4, survivorship-bias fix)
#   - real funding-rate settlement cost (Korak 5)
#   - per-symbol cost tiers (Korak 5, on by default already)
#   - SL/TP tie-break, OOS validation (Korak 5 / audit follow-up, on by default)
#   - recorded in the experiment registry so the Deflated Sharpe Ratio shown
#     on the dashboard is corrected for every prior attempt of this exact,
#     fully-corrected configuration (not pooled with older, uncorrected runs
#     under the same family name -- that would compare apples to oranges)
#
# This exists because, without it, every "full" run was a hand-typed SSH
# command that could (and did) silently drop --include-funding-cost or
# --point-in-time-universe-cache -- meaning the two most important bias
# corrections built in this project were never actually exercised against
# the live dashboard's report, even though the code fully supported them.
#
# Usage (no args = canonical 20-symbol/4-timeframe production run):
#   ./tool/run_full_portfolio_backtest.sh
# Usage (override anything, e.g. a quick 2-symbol smoke test):
#   ./tool/run_full_portfolio_backtest.sh --symbols BTCUSDT ETHUSDT --timeframes 1h

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ ! -x ".venv/bin/python" ]; then
    echo "No .venv found. Run ./tool/setup.sh first." >&2
    exit 1
fi

PIT_CACHE="scanner_data/point_in_time_universe.json"
if [ "$#" -eq 0 ]; then
    if [ ! -f "$PIT_CACHE" ]; then
        echo "WARNING: $PIT_CACHE not found -- running WITHOUT the point-in-time" >&2
        echo "universe filter, so this run still carries survivorship bias. Build it" >&2
        echo "first with: python -m scanner.backtesting.historical_universe --help" >&2
        PIT_ARGS=()
    else
        PIT_ARGS=(--point-in-time-universe-cache "$PIT_CACHE")
    fi
    set -- \
        --symbols AAVEUSDT ADAUSDT AVAXUSDT BCHUSDT BNBUSDT BTCUSDT DOGEUSDT ETHUSDT INJUSDT LINKUSDT \
                  NEARUSDT ONDOUSDT OPUSDT SOLUSDT SUIUSDT TAOUSDT TRXUSDT UNIUSDT WLDUSDT XRPUSDT \
        --timeframes 5m 15m 1h 4h \
        --kline-limit 1500 --warmup-candles 1500 \
        --include-funding-cost \
        "${PIT_ARGS[@]}" \
        --report-json scanner_data/portfolio_backtest_report.json \
        --trades-json scanner_data/portfolio_backtest_trades.json \
        --experiment-family portfolio_backtest_20symbol_warmup1500_full
fi

export PYTHONIOENCODING=utf-8
exec .venv/bin/python -m scanner.backtesting.portfolio_backtest "$@"
