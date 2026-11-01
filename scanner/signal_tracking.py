"""Live forward-outcome tracking: closed-loop validation for the Top Signal
Now panel.

A calibration win rate is a *historical* backtest statistic. It says
nothing on its own about whether the specific, live signal shown to a user
right now will actually play out the same way — regimes shift, and a
backtest can't observe data it hasn't been re-run against yet. This module
closes that loop: every time a symbol+direction first qualifies for the
panel, it is logged here with its exact entry/SL/TP and the win rate it was
promised; a resolver watches it every scan cycle and marks it resolved once
price actually hits the stop, the target, or the horizon runs out — so the
promised win rate can be checked against what genuinely happened live, not
just against its own backtest.

Resolution is intentionally simple and honest about its own precision: it
compares each scan cycle's observed close (sampled every
``poll_interval_seconds``, typically 5 minutes) against the stored SL/TP,
not true tick-by-tick order-book data. A brief wick between two scan cycles
that touches a level and reverts before the next sample could be missed —
a known, acceptable limitation (and still far finer-grained than waiting
for the full reference-timeframe candle to close), not a claim of
millisecond-perfect fills.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from agents_crypto.best_signal import BEST_SIGNAL_HORIZON_CANDLES, compute_candidates
from scanner.calibration import DEFAULT_ROUND_TRIP_COST_PCT
from scanner.storage import (
    fetch_open_signal_outcome,
    fetch_open_signal_outcomes,
    log_signal_outcome,
    resolve_signal_outcome,
)

logger = logging.getLogger(__name__)

# How many minutes one candle spans per timeframe — used to convert
# "horizon_candles" into a wall-clock timeout for resolving a signal that
# never hit its stop or target.
TIMEFRAME_MINUTES: dict[str, int] = {"5m": 5, "15m": 15, "1h": 60, "4h": 240}

# How many of the Top Signal Now panel's ranked candidates to actually track
# to an outcome. Deliberately a bit wider than the panel's own default
# display limit (5) so a symbol that qualifies but narrowly misses the
# visible shortlist is still tracked, without trying to track every
# marginally-significant match across the whole universe.
DEFAULT_TRACKING_LIMIT = 10


def log_new_signals(conn: Any, rows: list[dict[str, Any]], limit: int = DEFAULT_TRACKING_LIMIT) -> int:
    """Log any newly-qualifying Top Signal Now candidate that isn't already
    being tracked. Returns how many new rows were logged this cycle."""
    result = compute_candidates(rows, limit=limit)
    logged = 0
    for candidate in result["signals"]:
        entry = candidate.get("entry")
        if entry is None:
            continue  # no risk levels could be computed (missing ATR) -- nothing to track
        symbol, timeframe, direction = candidate["symbol"], candidate["timeframe"], candidate["direction"]
        if fetch_open_signal_outcome(conn, symbol, timeframe, direction) is not None:
            continue  # already tracking this exact opportunity
        log_signal_outcome(
            conn,
            symbol=symbol,
            timeframe=timeframe,
            direction=direction,
            label=candidate["label"],
            entry_price=entry["price"],
            stop_loss=entry["stop_loss"],
            take_profit=entry["take_profit"],
            take_profit_conservative=entry.get("take_profit_conservative"),
            promised_win_rate_pct=candidate.get("win_rate_pct"),
            promised_win_rate_ci_low_pct=candidate.get("win_rate_ci_low_pct"),
            promised_n=candidate.get("n"),
            horizon_candles=BEST_SIGNAL_HORIZON_CANDLES,
        )
        logged += 1
    return logged


def _net_return_pct(direction: str, entry_price: float, exit_price: float) -> float:
    raw_pct = (exit_price - entry_price) / entry_price * 100.0
    directed = raw_pct if direction == "long" else -raw_pct
    return directed - DEFAULT_ROUND_TRIP_COST_PCT


def resolve_open_signals(conn: Any, rows: list[dict[str, Any]]) -> int:
    """Check every open tracked signal against this cycle's latest scan
    data and resolve any that have hit their stop, their target, or run out
    of horizon time. Returns how many were resolved this cycle."""
    rows_by_symbol = {row["symbol"]: row for row in rows}
    resolved = 0
    for outcome in fetch_open_signal_outcomes(conn):
        current = rows_by_symbol.get(outcome["symbol"])
        if current is None or current["reference_timeframe"] != outcome["timeframe"]:
            continue  # symbol dropped out of rotation this cycle, or timeframe mismatch -- wait for it to come back
        close = current["close_price"]
        direction, stop_loss, take_profit = outcome["direction"], outcome["stop_loss"], outcome["take_profit"]

        if direction == "long":
            sl_hit, tp_hit = close <= stop_loss, close >= take_profit
        else:
            sl_hit, tp_hit = close >= stop_loss, close <= take_profit

        status: str | None = None
        if sl_hit:  # conservative: a stop touch always wins a same-cycle tie against a target touch
            status = "stop_loss"
        elif tp_hit:
            status = "take_profit"
        else:
            minutes_per_candle = TIMEFRAME_MINUTES.get(outcome["timeframe"], 60)
            horizon_minutes = outcome["horizon_candles"] * minutes_per_candle
            logged_at = datetime.fromisoformat(outcome["logged_at"])
            elapsed_minutes = (datetime.now(UTC) - logged_at).total_seconds() / 60.0
            if elapsed_minutes >= horizon_minutes:
                status = "timeout"

        if status is None:
            continue

        net_return = _net_return_pct(direction, outcome["entry_price"], close)
        resolve_signal_outcome(conn, outcome["id"], status, close, net_return)
        resolved += 1
    return resolved
