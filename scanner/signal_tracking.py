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
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from agents_crypto.best_signal import BEST_SIGNAL_HORIZON_CANDLES, compute_candidates
from agents_crypto.risk_calc import TP1_WEIGHT, TP2_WEIGHT, TP3_WEIGHT
from scanner.calibration import DEFAULT_ROUND_TRIP_COST_PCT
from scanner.storage import (
    fetch_open_signal_outcome,
    fetch_open_signal_outcomes,
    log_signal_outcome,
    resolve_signal_outcome,
    update_signal_outcome_progress,
)

logger = logging.getLogger(__name__)

# How many minutes one candle spans per timeframe — used to convert
# "horizon_candles" into a wall-clock timeout for resolving a signal that
# never hit its stop or target.
TIMEFRAME_MINUTES: dict[str, int] = {"5m": 5, "15m": 15, "1h": 60, "4h": 240}

# Timeframes managed through three partial targets (TP1/TP2/TP3, see
# agents_crypto.risk_calc) instead of one single exit -- currently just 4h,
# the only timeframe above 1h this scanner tracks; a longer-running signal's
# moves tend to run further, so locking in profit progressively (rather than
# one all-or-nothing target) captures more of that without giving back a
# banked gain. Extend this set if/when a 1d or 1w timeframe is added.
MULTI_TARGET_TIMEFRAMES: frozenset[str] = frozenset({"4h"})

_TP_WEIGHTS = (TP1_WEIGHT, TP2_WEIGHT, TP3_WEIGHT)

# How many of the Top Signal Now panel's ranked candidates to actually track
# to an outcome, **per timeframe** (see ``agents_crypto.best_signal.TRACKED_TIMEFRAMES``
# and ``compute_candidates``'s ``limit`` parameter) -- matches the dashboard's
# own per-timeframe display limit exactly, so this tracks precisely what a
# user would have seen on the dashboard at that moment, on every timeframe,
# not a subtly different (and much larger, now that there are 4 independent
# timeframes instead of 1) recomputation.
DEFAULT_TRACKING_LIMIT = 5


def log_new_signals(
    conn: Any,
    rows: list[dict[str, Any]],
    limit: int = DEFAULT_TRACKING_LIMIT,
    on_new_signal: Callable[[dict[str, Any]], None] | None = None,
    muted_buckets: set[tuple[str, str]] | None = None,
) -> int:
    """Log any newly-qualifying Top Signal Now candidate that isn't already
    being tracked. Returns how many new rows were logged this cycle.

    ``on_new_signal``, if given, is called once per newly-logged candidate
    (e.g. to push a Telegram alert — see :mod:`scanner.telegram_alerts`) —
    never for a candidate that was already being tracked, so an alert fires
    exactly once per genuinely new opportunity. A failing callback is
    logged and swallowed, never allowed to stop the rest of this cycle's
    logging.

    ``muted_buckets``, if given, is forwarded to
    :func:`agents_crypto.best_signal.compute_candidates` so a bucket the
    live calibration-drift detector has muted (see
    :mod:`scanner.drift_detector`) is never newly tracked either — keeping
    this tracker's own input in sync with what the dashboard shows.
    """
    result = compute_candidates(rows, limit=limit, muted_buckets=muted_buckets)
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
            regime=candidate.get("regime"),
            pattern=candidate.get("pattern"),
            quality_score=(candidate.get("quality_score") or {}).get("total"),
            regime_fallback=candidate.get("regime_fallback"),
            take_profit_extended=entry.get("take_profit_extended"),
            multi_target=timeframe in MULTI_TARGET_TIMEFRAMES,
        )
        logged += 1
        if on_new_signal is not None:
            try:
                on_new_signal(candidate)
            except Exception:
                logger.exception("on_new_signal callback failed for %s %s %s; continuing", symbol, timeframe, direction)
    return logged


def _net_return_pct(direction: str, entry_price: float, exit_price: float) -> float:
    raw_pct = (exit_price - entry_price) / entry_price * 100.0
    directed = raw_pct if direction == "long" else -raw_pct
    return directed - DEFAULT_ROUND_TRIP_COST_PCT


def _weighted_net_return_pct(direction: str, entry_price: float, legs: list[tuple[float, float]]) -> float:
    """Net return across several partial closes of the same position, each
    its own ``(weight, exit_price)`` -- weights should sum to ~1.0. Applies
    the round-trip cost per leg (proportional to that leg's weight), so the
    total cost drag across all legs still equals one full round trip's
    worth, just realistically distributed across each partial exit."""
    total = 0.0
    for weight, exit_price in legs:
        raw_pct = (exit_price - entry_price) / entry_price * 100.0
        directed = raw_pct if direction == "long" else -raw_pct
        total += weight * (directed - DEFAULT_ROUND_TRIP_COST_PCT)
    return total


def _timeout_status(outcome: dict[str, Any]) -> str | None:
    minutes_per_candle = TIMEFRAME_MINUTES.get(outcome["timeframe"], 60)
    horizon_minutes = outcome["horizon_candles"] * minutes_per_candle
    logged_at = datetime.fromisoformat(outcome["logged_at"])
    elapsed_minutes = (datetime.now(UTC) - logged_at).total_seconds() / 60.0
    return "timeout" if elapsed_minutes >= horizon_minutes else None


def _resolve_single_target_signal(conn: Any, outcome: dict[str, Any], close: float) -> int:
    """Original one-target resolution: hit the stop, hit the target, or
    time out -- unchanged behaviour for timeframes not in
    MULTI_TARGET_TIMEFRAMES."""
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
        status = _timeout_status(outcome)

    if status is None:
        return 0
    net_return = _net_return_pct(direction, outcome["entry_price"], close)
    resolve_signal_outcome(conn, outcome["id"], status, close, net_return)
    return 1


def _effective_stop_for(targets_hit: int, entry_price: float, stop_loss: float, tp1_price: float) -> float:
    """Where the stop sits after N targets have been banked: untouched
    until TP1, breakeven (entry) after TP1, trailed up to the TP1 price
    itself after TP2 -- so a pullback after TP2 can never give back more
    than the TP1 leg already banked."""
    if targets_hit <= 0:
        return stop_loss
    if targets_hit == 1:
        return entry_price
    return tp1_price


def _partial_close_net_return_pct(
    direction: str, entry_price: float, targets: list[float], targets_hit: int, remainder_exit_price: float
) -> float:
    """Weighted return for a multi-target trade ending (via its moved stop
    or a horizon timeout) after ``targets_hit`` of its three targets were
    already banked -- the banked legs count at their own target price, and
    whatever size is still open counts at ``remainder_exit_price``."""
    legs = [(_TP_WEIGHTS[i], targets[i]) for i in range(targets_hit)]
    remaining_weight = 1.0 - sum(weight for weight, _ in legs)
    legs.append((remaining_weight, remainder_exit_price))
    return _weighted_net_return_pct(direction, entry_price, legs)


def _resolve_multi_target_signal(conn: Any, outcome: dict[str, Any], close: float) -> int:
    """TP1/TP2/TP3 partial-close resolution (see
    agents_crypto.risk_calc.TP1_WEIGHT/TP2_WEIGHT/TP3_WEIGHT and
    MULTI_TARGET_TIMEFRAMES above). A target banked this cycle only moves
    the stop and keeps the trade open (returns 0, not resolved) -- the
    trade only actually resolves once its (moved) stop is hit, all three
    targets are banked, or the horizon times out."""
    direction = outcome["direction"]
    entry_price = outcome["entry_price"]
    stop_loss = outcome["stop_loss"]
    targets_hit = outcome.get("targets_hit") or 0
    effective_stop = outcome.get("effective_stop_loss")
    if effective_stop is None:
        effective_stop = stop_loss
    targets = [outcome["take_profit_conservative"], outcome["take_profit"], outcome["take_profit_extended"]]

    stop_hit = close <= effective_stop if direction == "long" else close >= effective_stop
    if stop_hit:
        status = {0: "stop_loss", 1: "tp1_be", 2: "tp2_be"}[targets_hit]
        net_return = _partial_close_net_return_pct(direction, entry_price, targets, targets_hit, close)
        resolve_signal_outcome(conn, outcome["id"], status, close, net_return)
        return 1

    # Bank every target reached this cycle (possibly more than one, if a
    # fast move jumped past an intermediate level between scan cycles).
    new_targets_hit = targets_hit
    while new_targets_hit < 3:
        next_target = targets[new_targets_hit]
        reached = close >= next_target if direction == "long" else close <= next_target
        if not reached:
            break
        new_targets_hit += 1

    if new_targets_hit == 3:
        legs = list(zip(_TP_WEIGHTS, targets, strict=False))
        net_return = _weighted_net_return_pct(direction, entry_price, legs)
        resolve_signal_outcome(conn, outcome["id"], "full_run", targets[-1], net_return)
        return 1

    if new_targets_hit != targets_hit:
        new_effective_stop = _effective_stop_for(new_targets_hit, entry_price, stop_loss, targets[0])
        update_signal_outcome_progress(
            conn, outcome["id"], targets_hit=new_targets_hit, effective_stop_loss=new_effective_stop
        )
        return 0  # stop just moved -- trade stays open, not a resolution

    status = _timeout_status(outcome)
    if status is None:
        return 0
    net_return = _partial_close_net_return_pct(direction, entry_price, targets, targets_hit, close)
    resolve_signal_outcome(conn, outcome["id"], status, close, net_return)
    return 1


def resolve_open_signals(conn: Any, rows: list[dict[str, Any]]) -> int:
    """Check every open tracked signal against this cycle's latest scan
    data and resolve any that have hit their stop, their target, or run out
    of horizon time. Returns how many were resolved this cycle.

    Multi-target signals (see MULTI_TARGET_TIMEFRAMES) are handled
    separately: banking a target without hitting the (moved) stop doesn't
    resolve the trade, it just updates its progress and keeps it open."""
    rows_by_symbol = {row["symbol"]: row for row in rows}
    resolved = 0
    for outcome in fetch_open_signal_outcomes(conn):
        current = rows_by_symbol.get(outcome["symbol"])
        if current is None:
            continue  # symbol dropped out of rotation this cycle -- wait for it to come back
        # Each tracked signal carries its own timeframe (5m/15m/1h/4h, not
        # just the scan row's flattened reference timeframe) -- resolve it
        # against *that* timeframe's own close price from the full
        # snapshot, never the flattened one, so a 4h signal is never
        # (mis)resolved against a 1h close.
        tf_snapshot = current.get("snapshot", {}).get("timeframes", {}).get(outcome["timeframe"])
        if tf_snapshot is None or tf_snapshot.get("close") is None:
            continue  # this cycle's snapshot doesn't carry the outcome's own timeframe yet -- wait for it
        close = tf_snapshot["close"]

        if outcome.get("multi_target"):
            resolved += _resolve_multi_target_signal(conn, outcome, close)
        else:
            resolved += _resolve_single_target_signal(conn, outcome, close)
    return resolved

