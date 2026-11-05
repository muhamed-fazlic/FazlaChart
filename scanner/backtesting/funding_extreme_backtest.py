"""Funding-extreme reversion backtest -- trial/family #2 (standalone).

Implements the hypothesis pre-registered in
``PREREGISTRATION_FUNDING_EXTREME_REVERSION.md`` (read that file FIRST for
the full, frozen specification -- this module's docstrings summarize it,
they do not replace it): when a USDT-perp's funding rate sits in the
extreme upper/lower 5% of its own rolling, strictly-past-only distribution,
the crowded side is expected to mean-revert over the next 8h (one funding
cycle).

This is a deliberately SEPARATE module from
``scanner.backtesting.portfolio_backtest`` -- that module's whole pipeline
is built around indicator/label-driven candidates with an optional SL/TP
exit; this hypothesis's entry (funding-rate percentile) and exit (fixed 8h
hold, no SL/TP) are both fundamentally different, so bolting it onto the
existing pipeline would mean threading a parallel, mostly-unused code path
through it instead of a clean, independently auditable module.

Frozen parameters (do NOT tune after the one pre-registered run -- see the
pre-registration doc's "look once" rule):

* Rolling percentile window: up to 270 past funding settlements (90 days),
  minimum 90 past observations required before a symbol may emit ANY
  signal.
* Entry: percentile > 95 -> short; percentile < 5 -> long.
* Exit: fixed 8h holding time (one funding cycle), no SL/TP.
* Cooldown: a new entry on the same (symbol, direction) is allowed no
  earlier than 16h after the previous one on that same (symbol, direction).
* Cross-symbol dedup: surviving (post-cooldown) entries are grouped by
  direction only and transitively merged whenever their
  ``[entry_time, entry_time + 8h]`` windows overlap, keeping one
  representative per cluster -- mirrors the design pattern (not the exact
  grouping key) of ``portfolio_backtest._dedupe_correlated_events``.

Several implementation details are NOT pinned by the pre-registration doc
(it says so explicitly and asks for them to be documented, not silently
decided). Each one is called out in its own function's docstring below and
also summarized in an appendix this module's author appended to the end of
``PREREGISTRATION_FUNDING_EXTREME_REVERSION.md``.

Usage (one-and-only pre-registered run)::

    python -m scanner.backtesting.funding_extreme_backtest \\
        --end-timestamp 1790253820000 \\
        --output scanner_data/funding_extreme_backtest_report.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import deque
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd

from scanner.backtesting.data_snapshot import (
    DEFAULT_FUNDING_CACHE_DIR,
    DEFAULT_KLINE_CACHE_DIR,
    cached_fetch_full_funding_history,
    cached_load_extended_history,
    combined_dataset_fingerprint,
    dataset_fingerprint,
    funding_dataset_fingerprint,
    parse_end_timestamp,
)
from scanner.backtesting.funding_simulator import FundingEvent, funding_cost_pct
from scanner.backtesting.historical_universe import is_symbol_in_universe_at, load_universe_cache
from scanner.calibration import DEFAULT_ROUND_TRIP_COST_PCT, round_trip_cost_pct_for
from scanner.validation.statistical_validation import (
    BLOCK_BOOTSTRAP_CONFIDENCE_PCT,
    BLOCK_BOOTSTRAP_N_RESAMPLES,
    block_bootstrap_confidence_interval,
)
from tradingagents.dataflows.errors import VendorError

# --- Frozen spec constants (PREREGISTRATION_FUNDING_EXTREME_REVERSION.md section 2) ---
FUNDING_WINDOW_MAX_OBSERVATIONS = 270
FUNDING_WINDOW_MIN_OBSERVATIONS = 90
HOLD_HOURS = 8
COOLDOWN_HOURS = 16
SHORT_PERCENTILE_THRESHOLD = 95.0
LONG_PERCENTILE_THRESHOLD = 5.0

# Frozen target edge for the pre-registered "undecidable" power calculation
# (section 5): n_required = (z_alpha + z_beta)^2 * sigma^2 / d^2.
TARGET_EDGE_PCT = 0.3
Z_ALPHA_95 = 1.96
Z_BETA_80 = 0.84

# Same 20-symbol universe as the already-closed trend/anomaly family -- see
# tool/run_full_portfolio_backtest.sh's --symbols list, reused verbatim, not
# re-derived or re-picked for this new hypothesis.
DEFAULT_SYMBOLS: tuple[str, ...] = (
    "AAVEUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "BCHUSDT",
    "BNBUSDT",
    "BTCUSDT",
    "DOGEUSDT",
    "ETHUSDT",
    "INJUSDT",
    "LINKUSDT",
    "NEARUSDT",
    "ONDOUSDT",
    "OPUSDT",
    "SOLUSDT",
    "SUIUSDT",
    "TAOUSDT",
    "TRXUSDT",
    "UNIUSDT",
    "WLDUSDT",
    "XRPUSDT",
)

# Documented design choice (not pinned by the pre-registration doc -- see the
# appendix appended to it): 1h candles give enough resolution to locate a
# price close to any funding-settlement entry/exit time. 8000 candles (~333
# days) comfortably exceeds the ~8-month real ceiling already measured for
# this universe (see the doc's section 2a) -- cached_load_extended_history
# stops early once a symbol's real history is exhausted, so this is a safe
# upper bound, not an assumption that 8000 candles actually exist.
DEFAULT_PRICE_TIMEFRAME = "1h"
DEFAULT_PRICE_CANDLES = 8000


@dataclass(frozen=True)
class FundingSignal:
    """One point in time where a symbol's rolling funding-rate percentile
    crossed a frozen entry threshold."""

    symbol: str
    direction: str  # "long" or "short"
    entry_time: datetime
    percentile: float
    funding_rate: float

    @property
    def exit_time(self) -> datetime:
        return self.entry_time + timedelta(hours=HOLD_HOURS)


@dataclass
class Trade:
    """One realized (priced, cost-adjusted) funding-extreme-reversion
    trade -- one surviving cluster representative after cooldown + dedup."""

    symbol: str
    direction: str
    entry_time: datetime
    exit_time: datetime
    percentile: float
    entry_price: float
    exit_price: float
    gross_return_pct: float
    cost_pct: float
    funding_cost_pct: float
    net_return_pct: float


def rolling_percentile(window: list[float], value: float) -> float:
    """Percentage (0-100) of ``window`` (strictly-prior funding rates)
    strictly less than ``value``.

    Documented design choice (not pinned by the pre-registration doc):
    strict ``<`` rather than ``<=`` -- a simple, dependency-free
    rank-percentile definition where ties within the window never inflate
    the computed percentile of the current observation. An empty window
    (should never actually reach here given the 90-observation minimum
    gate in :func:`generate_raw_signals`, checked anyway for safety)
    returns the neutral midpoint, 50.0.
    """
    if not window:
        return 50.0
    count_less = sum(1 for w in window if w < value)
    return (count_less / len(window)) * 100.0


def generate_raw_signals(symbol: str, funding_events: list[FundingEvent]) -> list[FundingSignal]:
    """One signal per funding settlement where the rolling percentile of
    that settlement's rate (computed from ONLY strictly-prior settlements)
    crosses the frozen 95/5 thresholds. ``funding_events`` must already be
    sorted oldest-first (``fetch_full_funding_history``/
    ``cached_fetch_full_funding_history`` already guarantee this).

    A ``deque(maxlen=270)`` naturally implements both frozen rules at
    once: its length never exceeds the 270-observation rolling cap, and
    before that cap is reached its length still correctly reports the true
    count of strictly-prior observations seen so far -- exactly what the
    90-observation minimum warm-up gate needs to check. The current
    event's own rate is appended to the window only AFTER being evaluated
    against it, so it can never see itself (no look-ahead).
    """
    window: deque[float] = deque(maxlen=FUNDING_WINDOW_MAX_OBSERVATIONS)
    signals: list[FundingSignal] = []
    for event in funding_events:
        if len(window) >= FUNDING_WINDOW_MIN_OBSERVATIONS:
            pct = rolling_percentile(list(window), event.rate)
            if pct > SHORT_PERCENTILE_THRESHOLD:
                signals.append(FundingSignal(symbol, "short", event.timestamp, pct, event.rate))
            elif pct < LONG_PERCENTILE_THRESHOLD:
                signals.append(FundingSignal(symbol, "long", event.timestamp, pct, event.rate))
        window.append(event.rate)
    return signals


def filter_signals_to_tradable_range(
    signals: list[FundingSignal], candle_min_time: datetime, candle_max_time: datetime
) -> list[FundingSignal]:
    """Drops signals whose entry time or 8h-later exit time falls outside
    the symbol's available candle history -- those cannot be realized as
    an actual trade with a known entry/exit price (funding history
    routinely goes back much further than candle history, see the
    pre-registration doc's section 2a).

    Documented design choice: applied BEFORE cooldown/dedup, since an
    untradeable signal should not consume a cooldown slot or "win" a
    cross-symbol dedup cluster that a genuinely tradable signal could
    otherwise have represented.
    """
    return [s for s in signals if candle_min_time <= s.entry_time and s.exit_time <= candle_max_time]


def apply_cooldown(signals: list[FundingSignal]) -> list[FundingSignal]:
    """Frozen rule (pre-registration doc section 2): a new entry on the
    same (symbol, direction) is allowed no earlier than 16h after the
    previous entry on that same (symbol, direction) -- applied
    sequentially in chronological order per (symbol, direction) group,
    BEFORE cross-symbol dedup."""
    by_key: dict[tuple[str, str], list[FundingSignal]] = {}
    for s in signals:
        by_key.setdefault((s.symbol, s.direction), []).append(s)

    kept: list[FundingSignal] = []
    for group in by_key.values():
        group.sort(key=lambda s: s.entry_time)
        last_kept_time: datetime | None = None
        for s in group:
            if last_kept_time is None or s.entry_time >= last_kept_time + timedelta(hours=COOLDOWN_HOURS):
                kept.append(s)
                last_kept_time = s.entry_time
    return kept


def dedupe_correlated_signals(signals: list[FundingSignal]) -> tuple[list[FundingSignal], int]:
    """Cross-symbol correlated-event dedup (pre-registration doc section
    2), mirroring the DESIGN PATTERN (not the exact grouping key) of
    ``portfolio_backtest._dedupe_correlated_events``: groups surviving
    (post-cooldown) ``signals`` by ``direction`` ONLY -- not symbol, since
    a single market-wide funding-extreme episode can fire the same
    direction on many correlated altcoins at once -- sorts each group
    chronologically, and transitively merges any whose
    ``[entry_time, entry_time + 8h]`` windows overlap (if A overlaps B and
    B overlaps C, all three land in one cluster even if A and C don't
    directly overlap) into one cluster.

    Documented design choice (not pinned by the pre-registration doc, which
    explicitly asks for one to be picked and recorded): keeps the cluster
    member with the MOST EXTREME percentile (farthest from the neutral
    midpoint, 50) as representative, not simply the first chronologically
    -- this keeps the clearest signal of each cluster, the same spirit as
    ``_dedupe_correlated_events`` keeping the highest-confidence bucket
    member rather than merely the earliest. Ties are broken to the
    earliest chronologically, since ``max()`` returns the first maximum it
    encounters and each group is iterated in chronological order.

    Returns ``(kept_signals, n_removed)``.
    """
    by_direction: dict[str, list[FundingSignal]] = {}
    for s in signals:
        by_direction.setdefault(s.direction, []).append(s)

    kept: list[FundingSignal] = []
    for group in by_direction.values():
        group.sort(key=lambda s: s.entry_time)
        cluster: list[FundingSignal] = []
        cluster_end: datetime | None = None
        for s in group:
            if cluster_end is None or s.entry_time > cluster_end:
                if cluster:
                    kept.append(max(cluster, key=lambda c: abs(c.percentile - 50.0)))
                cluster = [s]
                cluster_end = s.exit_time
            else:
                cluster.append(s)
                cluster_end = max(cluster_end, s.exit_time)
        if cluster:
            kept.append(max(cluster, key=lambda c: abs(c.percentile - 50.0)))
    return kept, len(signals) - len(kept)


def _price_at(history: pd.DataFrame, target_time: datetime) -> float | None:
    """Close price of the most recent candle at or before ``target_time``
    -- NEVER a candle that opens after it, preserving the no-look-ahead
    discipline the rest of this module follows. Returns ``None`` if no
    such candle exists (target_time before the symbol's earliest fetched
    candle)."""
    eligible = history[history["open_time"] <= pd.Timestamp(target_time)]
    if eligible.empty:
        return None
    return float(eligible.iloc[-1]["close"])


def build_trade(
    signal: FundingSignal,
    history: pd.DataFrame,
    funding_events: list[FundingEvent],
    base_cost_pct: float,
) -> Trade | None:
    """Realizes one ``FundingSignal`` into a priced, cost-adjusted
    ``Trade``: fixed 8h time-based exit (frozen, no SL/TP), the project's
    standard fee/spread cost-tier model, PLUS the actual funding
    paid/received during the holding period.

    Returns ``None`` if either price is unavailable (should not normally
    happen given :func:`filter_signals_to_tradable_range` was already
    applied upstream, but checked defensively here too)."""
    entry_price = _price_at(history, signal.entry_time)
    exit_price = _price_at(history, signal.exit_time)
    if entry_price is None or exit_price is None or entry_price == 0:
        return None

    raw_delta_pct = (exit_price - entry_price) / entry_price * 100.0
    gross_return_pct = raw_delta_pct if signal.direction == "long" else -raw_delta_pct

    cost_pct = round_trip_cost_pct_for(signal.symbol, base_cost_pct)
    funding_cost = funding_cost_pct(signal.direction, signal.entry_time, signal.exit_time, funding_events)
    net_return_pct = gross_return_pct - cost_pct - funding_cost

    return Trade(
        symbol=signal.symbol,
        direction=signal.direction,
        entry_time=signal.entry_time,
        exit_time=signal.exit_time,
        percentile=signal.percentile,
        entry_price=entry_price,
        exit_price=exit_price,
        gross_return_pct=round(gross_return_pct, 6),
        cost_pct=round(cost_pct, 6),
        funding_cost_pct=round(funding_cost, 6),
        net_return_pct=round(net_return_pct, 6),
    )


def _safe_block_bootstrap(
    returns: list[float], n_resamples: int, confidence_pct: float
) -> dict[str, Any] | None:
    """Wraps ``block_bootstrap_confidence_interval``, returning ``None``
    (rather than raising) when there are too few trades (< 4) for it to
    run -- this run's realistic expectation, per the pre-registration
    doc's section 5, is "a few dozen events at most", so this is not a
    hypothetical edge case."""
    try:
        result = block_bootstrap_confidence_interval(
            returns, n_resamples=n_resamples, confidence_pct=confidence_pct
        )
    except ValueError:
        return None
    return {
        "observed_pct": round(result["observed"], 4),
        "ci_low_pct": round(result["ci_low"], 4),
        "ci_high_pct": round(result["ci_high"], 4),
        "confidence_pct": result["confidence_pct"],
        "block_size": result["block_size"],
        "n_resamples": result["n_resamples"],
        "edge_survives_lower_bound": bool(result["ci_low"] > 0.0),
    }


def run_backtest(
    symbols: tuple[str, ...] = DEFAULT_SYMBOLS,
    *,
    end_timestamp_ms: int,
    price_timeframe: str = DEFAULT_PRICE_TIMEFRAME,
    price_candles: int = DEFAULT_PRICE_CANDLES,
    kline_cache_dir: str = DEFAULT_KLINE_CACHE_DIR,
    funding_cache_dir: str = DEFAULT_FUNDING_CACHE_DIR,
    base_cost_pct: float = DEFAULT_ROUND_TRIP_COST_PCT,
    bootstrap_n_resamples: int = BLOCK_BOOTSTRAP_N_RESAMPLES,
    bootstrap_confidence_pct: float = BLOCK_BOOTSTRAP_CONFIDENCE_PCT,
    point_in_time_universe_cache: str | None = None,
) -> dict[str, Any]:
    """Runs the full funding-extreme-reversion pipeline for ``symbols``
    over data anchored at the frozen ``end_timestamp_ms`` and returns the
    JSON-serializable report dict (see this module's CLI for how it is
    written to disk).

    ``end_timestamp_ms`` is a required keyword argument on purpose -- there
    is no silent "now" default here, unlike the live-scanner-facing helpers
    this module calls. Every call into ``cached_load_extended_history``/
    ``cached_fetch_full_funding_history`` below passes it explicitly, so
    the reserved holdout after it can never leak into this run.

    A symbol with no usable candle or funding data is skipped (recorded in
    the report's ``skipped_symbols``, not silently dropped) rather than
    aborting the whole run -- per the task's explicit instruction, this is
    a per-symbol data problem, not a reason to touch any of the frozen
    universe-wide parameters.
    """
    point_in_time_windows = None
    if point_in_time_universe_cache is not None:
        try:
            point_in_time_windows = load_universe_cache(point_in_time_universe_cache)
        except (FileNotFoundError, OSError, ValueError) as exc:
            print(f"WARNING: could not load point-in-time universe cache ({exc}); filter disabled.")

    all_signals: list[FundingSignal] = []
    histories: dict[str, pd.DataFrame] = {}
    funding_by_symbol: dict[str, list[FundingEvent]] = {}
    candle_fingerprints: list[str] = []
    funding_fingerprints: list[str] = []
    skipped_symbols: list[dict[str, str]] = []

    for symbol in symbols:
        try:
            history = cached_load_extended_history(
                symbol, price_timeframe, price_candles, end_timestamp_ms, cache_dir=kline_cache_dir
            )
        except VendorError as exc:
            skipped_symbols.append({"symbol": symbol, "reason": f"candle fetch failed: {exc}"})
            print(f"Skipping {symbol}: candle fetch failed ({exc})")
            continue
        if history.empty:
            skipped_symbols.append({"symbol": symbol, "reason": "no candle data available"})
            print(f"Skipping {symbol}: no candle data available")
            continue

        funding_events = cached_fetch_full_funding_history(symbol, end_timestamp_ms, cache_dir=funding_cache_dir)
        if not funding_events:
            skipped_symbols.append({"symbol": symbol, "reason": "no funding data available"})
            print(f"Skipping {symbol}: no funding data available")
            continue

        histories[symbol] = history
        funding_by_symbol[symbol] = funding_events
        candle_fingerprints.append(dataset_fingerprint(history))
        funding_fingerprints.append(funding_dataset_fingerprint(funding_events))

        raw_signals = generate_raw_signals(symbol, funding_events)
        candle_min_time = history["open_time"].min().to_pydatetime()
        candle_max_time = history["open_time"].max().to_pydatetime()
        all_signals.extend(filter_signals_to_tradable_range(raw_signals, candle_min_time, candle_max_time))
        print(f"{symbol}: {len(funding_events)} funding settlements, {len(raw_signals)} raw signals")

    all_signals.sort(key=lambda s: s.entry_time)
    n_raw_signals = len(all_signals)

    after_cooldown = apply_cooldown(all_signals)
    after_cooldown.sort(key=lambda s: s.entry_time)
    n_after_cooldown = len(after_cooldown)

    deduped, n_removed_dedup = dedupe_correlated_signals(after_cooldown)
    deduped.sort(key=lambda s: s.entry_time)

    point_in_time_filtered_count = 0
    if point_in_time_windows is not None:
        kept = [s for s in deduped if is_symbol_in_universe_at(point_in_time_windows, s.symbol, s.entry_time)]
        point_in_time_filtered_count = len(deduped) - len(kept)
        deduped = kept

    trades: list[Trade] = []
    for signal in deduped:
        trade = build_trade(signal, histories[signal.symbol], funding_by_symbol[signal.symbol], base_cost_pct)
        if trade is not None:
            trades.append(trade)
    trades.sort(key=lambda t: t.entry_time)

    n_independent = len(trades)
    net_returns = [t.net_return_pct for t in trades]
    gross_returns = [t.gross_return_pct for t in trades]

    win_rate_pct = round(sum(1 for r in net_returns if r > 0) / len(net_returns) * 100.0, 2) if net_returns else None
    # Documented design choice: sample standard deviation (ddof=1, i.e.
    # statistics.stdev not statistics.pstdev) -- we are estimating the
    # population sigma that feeds the power calculation below from a
    # single finite observed sample, the standard inferential choice.
    measured_sigma_pct = round(statistics.stdev(net_returns), 6) if len(net_returns) >= 2 else None

    net_bootstrap = _safe_block_bootstrap(net_returns, bootstrap_n_resamples, bootstrap_confidence_pct)
    gross_bootstrap = _safe_block_bootstrap(gross_returns, bootstrap_n_resamples, bootstrap_confidence_pct)

    n_required: int | None = None
    undecidable: bool | None = None
    if measured_sigma_pct is not None and measured_sigma_pct > 0:
        n_required = math.ceil(((Z_ALPHA_95 + Z_BETA_80) ** 2) * (measured_sigma_pct**2) / (TARGET_EDGE_PCT**2))
        undecidable = n_independent < n_required

    dataset_hash = combined_dataset_fingerprint(candle_fingerprints + funding_fingerprints)

    return {
        "hypothesis": "funding_extreme_reversion",
        "generated_at": datetime.now(UTC).isoformat(),
        "end_timestamp_ms": end_timestamp_ms,
        "end_timestamp_iso": datetime.fromtimestamp(end_timestamp_ms / 1000.0, tz=UTC).isoformat(),
        "spec": {
            "funding_window_max_observations": FUNDING_WINDOW_MAX_OBSERVATIONS,
            "funding_window_min_observations": FUNDING_WINDOW_MIN_OBSERVATIONS,
            "hold_hours": HOLD_HOURS,
            "cooldown_hours": COOLDOWN_HOURS,
            "short_percentile_threshold": SHORT_PERCENTILE_THRESHOLD,
            "long_percentile_threshold": LONG_PERCENTILE_THRESHOLD,
            "base_cost_pct": base_cost_pct,
            "price_timeframe": price_timeframe,
        },
        "symbols_requested": list(symbols),
        "symbols_used": sorted(histories.keys()),
        "skipped_symbols": skipped_symbols,
        "point_in_time_universe": (
            {
                "cache_path": point_in_time_universe_cache,
                "trades_excluded": point_in_time_filtered_count,
            }
            if point_in_time_windows is not None
            else None
        ),
        "n_raw_signals": n_raw_signals,
        "n_after_cooldown": n_after_cooldown,
        "n_removed_cross_symbol_dedup": n_removed_dedup,
        "n_independent": n_independent,
        "win_rate_pct": win_rate_pct,
        "measured_sigma_pct": measured_sigma_pct,
        "sigma_definition": "sample standard deviation (ddof=1) of net per-trade returns",
        "net_block_bootstrap": net_bootstrap,
        "gross_block_bootstrap": gross_bootstrap,
        "undecidable_threshold": {
            "target_edge_pct": TARGET_EDGE_PCT,
            "z_alpha_95": Z_ALPHA_95,
            "z_beta_80": Z_BETA_80,
            "n_required": n_required,
            "undecidable": undecidable,
        },
        "dataset_hash": dataset_hash,
        "trades": [
            {**asdict(t), "entry_time": t.entry_time.isoformat(), "exit_time": t.exit_time.isoformat()}
            for t in trades
        ],
    }


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Funding-extreme reversion backtest (pre-registered, look-once trial family #2)."
    )
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS), help="Symbols to include")
    parser.add_argument(
        "--end-timestamp",
        required=True,
        help="Frozen end-of-train/validation boundary (ISO 8601 or raw epoch-ms); holdout after it is never touched",
    )
    parser.add_argument("--output", default="scanner_data/funding_extreme_backtest_report.json")
    parser.add_argument("--price-timeframe", default=DEFAULT_PRICE_TIMEFRAME)
    parser.add_argument("--price-candles", type=int, default=DEFAULT_PRICE_CANDLES)
    parser.add_argument("--kline-cache-dir", default=DEFAULT_KLINE_CACHE_DIR)
    parser.add_argument("--funding-cache-dir", default=DEFAULT_FUNDING_CACHE_DIR)
    parser.add_argument("--cost-pct", type=float, default=DEFAULT_ROUND_TRIP_COST_PCT)
    parser.add_argument("--bootstrap-resamples", type=int, default=BLOCK_BOOTSTRAP_N_RESAMPLES)
    parser.add_argument(
        "--point-in-time-universe-cache",
        default=None,
        help="Optional path to a historical_universe.py cache (same mechanism as portfolio_backtest.py)",
    )
    args = parser.parse_args()

    end_timestamp_ms = parse_end_timestamp(args.end_timestamp)
    report = run_backtest(
        tuple(args.symbols),
        end_timestamp_ms=end_timestamp_ms,
        price_timeframe=args.price_timeframe,
        price_candles=args.price_candles,
        kline_cache_dir=args.kline_cache_dir,
        funding_cache_dir=args.funding_cache_dir,
        base_cost_pct=args.cost_pct,
        bootstrap_n_resamples=args.bootstrap_resamples,
        point_in_time_universe_cache=args.point_in_time_universe_cache,
    )

    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=_json_default)

    print(f"n_independent={report['n_independent']}")
    print(f"net_block_bootstrap={report['net_block_bootstrap']}")
    print(f"gross_block_bootstrap={report['gross_block_bootstrap']}")
    print(f"undecidable_threshold={report['undecidable_threshold']}")
    print(f"dataset_hash={report['dataset_hash']}")
    print(f"Report written to {args.output}")
    print("DONE")


if __name__ == "__main__":
    _main()
