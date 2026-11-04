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

import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from agents_crypto.best_signal import BEST_SIGNAL_HORIZON_CANDLES, compute_candidates
from agents_crypto.risk_calc import TP1_WEIGHT, TP2_WEIGHT, TP3_WEIGHT
from scanner.btc_context import BTC_SYMBOL
from scanner.calibration import DEFAULT_ROUND_TRIP_COST_PCT
from scanner.storage import (
    fetch_open_signal_outcome,
    fetch_open_signal_outcomes,
    log_signal_outcome,
    resolve_signal_outcome,
    touch_signal_outcome_excursion,
    update_signal_outcome_progress,
)

logger = logging.getLogger(__name__)

# How many minutes one candle spans per timeframe — used to convert
# "horizon_candles" into a wall-clock timeout for resolving a signal that
# never hit its stop or target.
TIMEFRAME_MINUTES: dict[str, int] = {"5m": 5, "15m": 15, "1h": 60, "4h": 240}

# Timeframes managed through three partial targets (TP1/TP2/TP3, see
# agents_crypto.risk_calc) instead of one single exit -- any timeframe of
# 1h or longer (i.e. > 59 minutes), since a longer-running signal's moves
# tend to run further, so locking in profit progressively (rather than one
# all-or-nothing target) captures more of that without giving back a
# banked gain. Derived from TIMEFRAME_MINUTES so adding a 1d/1w timeframe
# there automatically makes it multi-target too, with no change needed here.
MULTI_TARGET_TIMEFRAMES: frozenset[str] = frozenset(
    tf for tf, minutes in TIMEFRAME_MINUTES.items() if minutes > 59
)

# How many candles the *live tracker* waits before declaring an unresolved
# trade a timeout -- deliberately separate from BEST_SIGNAL_HORIZON_CANDLES
# (which stays the backtest/calibration labeling horizon and must not
# change just because this does, or every promised win rate would silently
# mean something different). A multi-target (1h/4h) trade already has its
# stop moved to breakeven after TP1 (see _effective_stop_for below), so
# letting it run longer for TP2/TP3 adds no extra downside risk; a
# single-target (5m/15m) trade has no such protection yet, so it keeps a
# shorter leash -- still longer than the 12-candle backtest horizon, but far
# short of 72, partly because funding-rate costs for a held futures
# position are not modeled anywhere in this project yet (see
# PHASE8_PROFITABILITY_RESEARCH_TODO.md Korak 5).
TIMEOUT_CANDLES_MULTI_TARGET = 72
TIMEOUT_CANDLES_SINGLE_TARGET = 24


def _timeout_candles_for(timeframe: str) -> int:
    return TIMEOUT_CANDLES_MULTI_TARGET if timeframe in MULTI_TARGET_TIMEFRAMES else TIMEOUT_CANDLES_SINGLE_TARGET


_TP_WEIGHTS = (TP1_WEIGHT, TP2_WEIGHT, TP3_WEIGHT)

# How many of the Top Signal Now panel's ranked candidates to actually track
# to an outcome, **per timeframe** (see ``agents_crypto.best_signal.TRACKED_TIMEFRAMES``
# and ``compute_candidates``'s ``limit`` parameter) -- matches the dashboard's
# own per-timeframe display limit exactly, so this tracks precisely what a
# user would have seen on the dashboard at that moment, on every timeframe,
# not a subtly different (and much larger, now that there are 4 independent
# timeframes instead of 1) recomputation.
DEFAULT_TRACKING_LIMIT = 5


def _build_entry_snapshot_json(
    row: dict[str, Any] | None, tf_snapshot: dict[str, Any] | None, candidate: dict[str, Any], btc_row: dict[str, Any] | None
) -> str:
    """Freeze the indicator/derivatives/quality/correlation picture at the
    moment a signal is logged, as a flat JSON blob -- the live ``scans``
    rows this is computed from are pruned after 60 days
    (``storage.prune_old_scans``), so without this, a resolved trade
    older than that has no way to be analyzed against what the market
    actually looked like at entry (e.g. "do wins conditioned on high RSI
    look different from wins conditioned on low RSI"). Deliberately a
    loose, additive dict (not its own schema) so new fields can be added
    over time without a migration -- anything here is for later read-only
    analysis, never fed back into a live trading decision.

    ``row``/``tf_snapshot`` can be ``None`` (e.g. if the symbol momentarily
    dropped out of this cycle's rows between ranking and logging) --
    returns whatever partial context is still available rather than
    failing the whole log attempt."""
    derivatives = (row or {}).get("snapshot", {}).get("derivatives", {}) if row else {}
    snapshot: dict[str, Any] = {
        "rsi": (tf_snapshot or {}).get("rsi"),
        "ema9": (tf_snapshot or {}).get("ema9"),
        "ema20": (tf_snapshot or {}).get("ema20"),
        "ema50": (tf_snapshot or {}).get("ema50"),
        "ema200": (tf_snapshot or {}).get("ema200"),
        "macd": (tf_snapshot or {}).get("macd"),
        "macd_signal": (tf_snapshot or {}).get("macd_signal"),
        "macd_hist": (tf_snapshot or {}).get("macd_hist"),
        "atr": (tf_snapshot or {}).get("atr"),
        "bollinger_upper": (tf_snapshot or {}).get("bollinger_upper"),
        "bollinger_lower": (tf_snapshot or {}).get("bollinger_lower"),
        "volatility_regime": (tf_snapshot or {}).get("volatility_regime"),
        "relative_volume": (tf_snapshot or {}).get("relative_volume"),
        "volume_change_pct": (tf_snapshot or {}).get("volume_change_pct"),
        "support": (tf_snapshot or {}).get("support"),
        "resistance": (tf_snapshot or {}).get("resistance"),
        "drawdown_pct": (tf_snapshot or {}).get("drawdown_pct"),
        "momentum_fading": (tf_snapshot or {}).get("momentum_fading"),
        "anomalies": (tf_snapshot or {}).get("anomalies"),
        "funding_rate": derivatives.get("funding_rate"),
        "open_interest": derivatives.get("open_interest"),
        "open_interest_change_pct_5m": derivatives.get("open_interest_change_pct_5m"),
        "open_interest_change_pct_15m": derivatives.get("open_interest_change_pct_15m"),
        "open_interest_change_pct_60m": derivatives.get("open_interest_change_pct_60m"),
        "quality_score_breakdown": candidate.get("quality_score"),
        "diverges_from_btc": candidate.get("diverges_from_btc"),
        "correlation": candidate.get("correlation"),
        "current_market_regime": candidate.get("current_market_regime"),
        "btc_trend": (btc_row or {}).get("trend"),
        "btc_close": (btc_row or {}).get("close_price"),
    }
    return json.dumps(snapshot)


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
    rows_by_symbol = {row["symbol"]: row for row in rows}
    btc_row = rows_by_symbol.get(BTC_SYMBOL)
    logged = 0
    for candidate in result["signals"]:
        entry = candidate.get("entry")
        if entry is None:
            continue  # no risk levels could be computed (missing ATR) -- nothing to track
        symbol, timeframe, direction = candidate["symbol"], candidate["timeframe"], candidate["direction"]
        if fetch_open_signal_outcome(conn, symbol, timeframe, direction) is not None:
            continue  # already tracking this exact opportunity
        row = rows_by_symbol.get(symbol)
        tf_snapshot = (row or {}).get("snapshot", {}).get("timeframes", {}).get(timeframe)
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
            entry_snapshot_json=_build_entry_snapshot_json(row, tf_snapshot, candidate, btc_row),
            timeout_candles=_timeout_candles_for(timeframe),
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
    # timeout_candles is the live tracker's own patience (see
    # _timeout_candles_for); outcomes logged before that column existed
    # fall back to recomputing it from the timeframe, never to the
    # unrelated horizon_candles/backtest-labeling value.
    timeout_candles = outcome.get("timeout_candles") or _timeout_candles_for(outcome["timeframe"])
    minutes_per_candle = TIMEFRAME_MINUTES.get(outcome["timeframe"], 60)
    horizon_minutes = timeout_candles * minutes_per_candle
    logged_at = datetime.fromisoformat(outcome["logged_at"])
    elapsed_minutes = (datetime.now(UTC) - logged_at).total_seconds() / 60.0
    return "timeout" if elapsed_minutes >= horizon_minutes else None


def _directional_excursion_pct(direction: str, entry_price: float, close: float) -> float:
    """How far ``close`` has moved from ``entry_price``, in the trade's own
    favor (positive) or against it (negative) -- the sign-normalized,
    direction-agnostic building block for MFE/MAE tracking, so a short's
    favorable drop and a long's favorable rise both read as positive."""
    raw_pct = (close - entry_price) / entry_price * 100.0
    return raw_pct if direction == "long" else -raw_pct


def _updated_excursion(
    outcome: dict[str, Any], direction: str, entry_price: float, close: float
) -> tuple[float, float, float, float]:
    """Roll this cycle's close into the running MFE/MAE extrema already
    stored on ``outcome``, returning the (possibly unchanged)
    ``(mfe_pct, mfe_price, mae_pct, mae_price)`` to persist this cycle.
    Falls back to a zero-excursion baseline at the entry price for any
    trade that predates these columns (see storage._migrate_schema's
    backfill, which should make that fallback a no-op in practice)."""
    current = _directional_excursion_pct(direction, entry_price, close)
    mfe_pct = outcome.get("mfe_pct")
    mfe_price = outcome.get("mfe_price")
    if mfe_pct is None or current > mfe_pct:
        mfe_pct, mfe_price = current, close
    mae_pct = outcome.get("mae_pct")
    mae_price = outcome.get("mae_price")
    if mae_pct is None or current < mae_pct:
        mae_pct, mae_price = current, close
    return mfe_pct, mfe_price, mae_pct, mae_price


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

    mfe_pct, mfe_price, mae_pct, mae_price = _updated_excursion(outcome, direction, outcome["entry_price"], close)
    if status is None:
        touch_signal_outcome_excursion(
            conn, outcome["id"], mfe_pct=mfe_pct, mfe_price=mfe_price, mae_pct=mae_pct, mae_price=mae_price
        )
        return 0
    net_return = _net_return_pct(direction, outcome["entry_price"], close)
    resolve_signal_outcome(
        conn,
        outcome["id"],
        status,
        close,
        net_return,
        mfe_pct=mfe_pct,
        mfe_price=mfe_price,
        mae_pct=mae_pct,
        mae_price=mae_price,
    )
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
    mfe_pct, mfe_price, mae_pct, mae_price = _updated_excursion(outcome, direction, entry_price, close)
    if stop_hit:
        status = {0: "stop_loss", 1: "tp1_be", 2: "tp2_be"}[targets_hit]
        net_return = _partial_close_net_return_pct(direction, entry_price, targets, targets_hit, close)
        resolve_signal_outcome(
            conn,
            outcome["id"],
            status,
            close,
            net_return,
            mfe_pct=mfe_pct,
            mfe_price=mfe_price,
            mae_pct=mae_pct,
            mae_price=mae_price,
        )
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
        resolve_signal_outcome(
            conn,
            outcome["id"],
            "full_run",
            targets[-1],
            net_return,
            mfe_pct=mfe_pct,
            mfe_price=mfe_price,
            mae_pct=mae_pct,
            mae_price=mae_price,
        )
        return 1

    if new_targets_hit != targets_hit:
        new_effective_stop = _effective_stop_for(new_targets_hit, entry_price, stop_loss, targets[0])
        update_signal_outcome_progress(
            conn,
            outcome["id"],
            targets_hit=new_targets_hit,
            effective_stop_loss=new_effective_stop,
            mfe_pct=mfe_pct,
            mfe_price=mfe_price,
            mae_pct=mae_pct,
            mae_price=mae_price,
        )
        return 0  # stop just moved -- trade stays open, not a resolution

    status = _timeout_status(outcome)
    if status is None:
        touch_signal_outcome_excursion(
            conn, outcome["id"], mfe_pct=mfe_pct, mfe_price=mfe_price, mae_pct=mae_pct, mae_price=mae_price
        )
        return 0
    net_return = _partial_close_net_return_pct(direction, entry_price, targets, targets_hit, close)
    resolve_signal_outcome(
        conn,
        outcome["id"],
        status,
        close,
        net_return,
        mfe_pct=mfe_pct,
        mfe_price=mfe_price,
        mae_pct=mae_pct,
        mae_price=mae_price,
    )
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

