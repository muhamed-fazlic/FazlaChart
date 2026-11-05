"""Cross-sectional momentum backtest -- trial/family #3 (standalone).

Implements the hypothesis pre-registered in
``PREREGISTRATION_CROSS_SECTIONAL_MOMENTUM.md`` (read that file FIRST for
the full, frozen specification -- this module's docstrings summarize it,
they do not replace it): at every daily rebalance, long the 5 symbols with
the highest trailing-24h % price change and short the 5 symbols with the
lowest trailing-24h % price change, among the point-in-time universe of 20
USDT-perps, equal-weighted within each leg, dollar-neutral (50% of capital
per leg). Unlike the first two (closed) hypotheses in this project -- both
per-symbol, direction-exposed bets -- this one is cross-sectional and
market-neutral by design; the beta check in this module's report is the
empirical test of whether that design goal actually holds.

This is a deliberately SEPARATE module from
``scanner.backtesting.portfolio_backtest`` and
``scanner.backtesting.funding_extreme_backtest`` -- both of those are
per-symbol/per-trade pipelines; this hypothesis's unit of observation is
one basket-wide return PER REBALANCE (not per symbol, not per trade), with
a fundamentally different cost model (turnover-based, not per-trade
round-trip), so it gets its own clean, independently auditable module
rather than being threaded through either existing pipeline.

Frozen parameters (do NOT tune after the one pre-registered run -- see the
pre-registration doc's "look once" rule):

* Ranking: 24h-trailing % price change (close[t] / close[t-24h] - 1),
  computed from only data up to and including t (no look-ahead).
* Universe: same 20 USDT-perp symbols as the prior two (closed)
  hypotheses, point-in-time filtered the same way.
* Minimum eligible symbols per rebalance: 2*N=10; fewer -> skip that
  rebalance entirely (never shrink the basket size).
* Baskets: long top-5 / short bottom-5 by 24h return among eligible
  symbols. N=5 is frozen.
* Weighting: equal-weight within each leg; both legs equal size
  (dollar-neutral).
* Rebalance: daily (24h), time-based exit (whole basket closes and a new
  one forms at the next rebalance). No SL/TP.
* Costs: turnover-based (see :func:`changed_symbols_and_turnover`) -- a
  symbol staying in the SAME leg between consecutive rebalances pays no
  cost; a symbol entering, exiting, or flipping legs pays the full
  per-symbol round-trip cost (``round_trip_cost_pct_for``), scaled by its
  capital weight.
* Funding: real funding paid/received over the ~24h hold, via
  ``funding_simulator.funding_cost_pct``, reused as-is (no new model).

Several implementation details are NOT pinned by the pre-registration doc.
Each one is called out in its own function's docstring below and also
summarized in the "## 9. Implementation notes" section appended to the end
of ``PREREGISTRATION_CROSS_SECTIONAL_MOMENTUM.md``.

Usage (one-and-only pre-registered run)::

    python -m scanner.backtesting.cross_sectional_momentum_backtest \\
        --end-timestamp 1790253820000 \\
        --output scanner_data/cross_sectional_momentum_backtest_report.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
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
    safe_block_bootstrap,
)
from tradingagents.dataflows.errors import VendorError

# --- Frozen spec constants (PREREGISTRATION_CROSS_SECTIONAL_MOMENTUM.md section 2) ---
LOOKBACK_HOURS = 24
REBALANCE_HOURS = 24
BASKET_SIZE = 5  # N
MIN_ELIGIBLE_SYMBOLS = 2 * BASKET_SIZE  # 10

# Frozen target edge for the pre-registered "undecidable" power calculation
# (section 5): n_required = (z_alpha + z_beta)^2 * sigma^2 / d^2.
TARGET_EDGE_PCT = 0.3
Z_ALPHA_95 = 1.96
Z_BETA_80 = 0.84

# Same 20-symbol universe as the two closed hypotheses -- reused verbatim,
# not re-derived or re-picked for this new hypothesis.
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

# Documented design choice (not pinned by the pre-registration doc -- see
# the appendix appended to it): 1h candles, consistent with the funding
# hypothesis's own implementation choice, give enough resolution to locate
# a price close to any 24h-grid rebalance timestamp. 8000 candles (~333
# days) comfortably exceeds the ~8-month real ceiling already measured for
# this universe -- cached_load_extended_history stops early once a
# symbol's real history is exhausted, so this is a safe upper bound.
DEFAULT_PRICE_TIMEFRAME = "1h"
DEFAULT_PRICE_CANDLES = 8000

# Beta-check threshold (pre-registration doc section 3): |r| > 0.3 means
# "market-neutral" does not hold empirically, regardless of design intent.
BETA_FLAG_THRESHOLD = 0.3
BTC_SYMBOL = "BTCUSDT"


def _price_at(history: pd.DataFrame, target_time: datetime) -> float | None:
    """Close price of the most recent candle at or before ``target_time``
    -- NEVER a candle opening after it, to preserve the no-look-ahead
    discipline the rest of this module follows (same convention as
    ``funding_extreme_backtest._price_at``). Returns ``None`` if no such
    candle exists."""
    eligible = history[history["open_time"] <= pd.Timestamp(target_time)]
    if eligible.empty:
        return None
    return float(eligible.iloc[-1]["close"])


def build_rebalance_grid(histories: dict[str, pd.DataFrame]) -> list[datetime]:
    """Builds the daily rebalance-timestamp grid.

    Documented design choice (not pinned by the pre-registration doc):
    anchored to the EARLIEST candle timestamp across all requested
    symbols' histories (not necessarily all 20 are listed that early --
    point-in-time eligibility is checked per-rebalance separately), then
    stepped forward by exactly 24h until the latest candle timestamp
    across all histories is reached. A simple, deterministic, data-driven
    choice: it does not hand-pick a particular hour-of-day, it simply uses
    whatever hour the earliest available candle happens to fall on and
    keeps every subsequent grid point exactly 24h apart from it, so no
    rebalance timestamp is ever invented ahead of any real data.
    """
    if not histories:
        return []
    earliest = min(h["open_time"].min().to_pydatetime() for h in histories.values())
    latest = max(h["open_time"].max().to_pydatetime() for h in histories.values())
    grid: list[datetime] = []
    t = earliest
    step = timedelta(hours=REBALANCE_HOURS)
    while t <= latest:
        grid.append(t)
        t += step
    return grid


def eligible_symbols_at(
    t: datetime,
    histories: dict[str, pd.DataFrame],
    point_in_time_windows: list | None,
) -> dict[str, float]:
    """24h-trailing % return (``close[t] / close[t-24h] - 1``), in
    percentage points, for every symbol that is (a) part of the
    point-in-time universe at ``t`` (if a cache was supplied) and (b) has a
    priced candle at both ``t-24h`` and ``t`` (no look-ahead: both prices
    are at or before their respective target times). A symbol missing
    either price point (not yet listed, or history gap) is simply not
    included -- NOT an error, the minimum-eligible-count rule in the
    caller handles that.
    """
    t_prev = t - timedelta(hours=LOOKBACK_HOURS)
    result: dict[str, float] = {}
    for symbol, history in histories.items():
        if point_in_time_windows is not None and not is_symbol_in_universe_at(point_in_time_windows, symbol, t):
            continue
        price_now = _price_at(history, t)
        price_prev = _price_at(history, t_prev)
        if price_now is None or price_prev is None or price_prev == 0:
            continue
        result[symbol] = (price_now / price_prev - 1.0) * 100.0
    return result


def form_basket(momentum_by_symbol: dict[str, float]) -> dict[str, str] | None:
    """Returns ``{symbol: "long"/"short"}`` for the top-5/bottom-5 basket,
    or ``None`` if fewer than ``MIN_ELIGIBLE_SYMBOLS`` symbols are
    eligible (frozen skip rule, pre-registration doc section 2)."""
    if len(momentum_by_symbol) < MIN_ELIGIBLE_SYMBOLS:
        return None
    ranked = sorted(momentum_by_symbol.items(), key=lambda kv: kv[1], reverse=True)
    longs = ranked[:BASKET_SIZE]
    shorts = ranked[-BASKET_SIZE:]
    basket: dict[str, str] = {}
    for symbol, _ in longs:
        basket[symbol] = "long"
    for symbol, _ in shorts:
        basket[symbol] = "short"
    return basket


def changed_symbols_and_turnover(
    prev_basket: dict[str, str] | None, curr_basket: dict[str, str]
) -> tuple[set[str], float]:
    """Implements the frozen turnover-cost rule (pre-registration doc
    section 2):

    * A symbol in the SAME leg at both ``prev_basket`` and ``curr_basket``
      pays no cost (position never closed/reopened).
    * A symbol newly entering a leg, leaving a leg entirely (not present
      in ``curr_basket`` at all), or flipping directly from one leg to the
      other pays the full round-trip cost once (not twice for a flip --
      the doc explicitly says "puni round-trip trosak", singular).

    Returns ``(changed_symbols, turnover)`` where ``changed_symbols`` is
    the set of ALL symbols needing a cost charge this rebalance (both
    fresh/flipped entrants AND pure exits), and ``turnover`` is the
    documented diagnostic number (0-1): the fraction of TODAY's basket
    slots (``2*N``) that are not a same-leg continuation from the prior
    rebalance. The first rebalance (``prev_basket is None``) has no prior
    basket to compare against, so every current-basket symbol counts as a
    fresh entrant (cost charged, turnover 1.0) -- a documented design
    choice, not otherwise pinned.

    Documented design choice: the turnover denominator is TODAY's basket
    size (``2*N``), not the union of yesterday's and today's symbols --
    simplest, most direct reading of "udio pozicija koje su se promijenile
    po rebalansu" (today's basket IS "pozicije" for that rebalance).
    """
    prev = prev_basket or {}
    changed_in_current = {s for s, leg in curr_basket.items() if prev.get(s) != leg}
    pure_exits = {s for s in prev if s not in curr_basket}
    changed_symbols = changed_in_current | pure_exits
    turnover = len(changed_in_current) / len(curr_basket) if curr_basket else 0.0
    return changed_symbols, turnover


@dataclass
class RebalancePeriod:
    """One realized, priced, cost-adjusted basket-wide rebalance period --
    the unit of observation for this hypothesis (NOT a per-symbol trade,
    see the pre-registration doc's section 3 on ``n_independent``)."""

    rebalance_time: datetime
    exit_time: datetime
    n_eligible: int
    basket: dict[str, str]
    gross_return_pct: float
    cost_pct: float
    funding_cost_pct: float
    net_return_pct: float
    turnover: float
    btc_return_pct: float | None


def _leg_weight(n_legs_symbols: int) -> float:
    """Per-symbol capital weight: each leg represents half of total
    capital (dollar-neutral 50%/50%, pre-registration doc section 2),
    equal-weighted across ``n_legs_symbols`` (== ``BASKET_SIZE`` by
    construction once a basket is formed). Documented design choice (the
    doc notes the capital-unit choice -- 50/50 vs leveraged 100/100 --
    does not change the % return "while symmetric"; 50/50, the simplest/
    least-levered reading, is used here so a single basket-wide % return
    number is directly comparable in magnitude to a single-leg
    directional bet, rather than doubled by leverage)."""
    return 0.5 / n_legs_symbols


def build_rebalance_period(
    t: datetime,
    t_exit: datetime,
    momentum_by_symbol: dict[str, float],
    basket: dict[str, str],
    prev_basket: dict[str, str] | None,
    histories: dict[str, pd.DataFrame],
    funding_by_symbol: dict[str, list[FundingEvent]],
    base_cost_pct: float,
) -> RebalancePeriod | None:
    """Prices one formed basket end-to-end. Returns ``None`` if any
    basket-member symbol's exit price is unavailable (cannot be realized
    as an actual priced period) -- a data problem, not a reason to shrink
    the basket or otherwise deviate from the frozen spec."""
    weight = _leg_weight(BASKET_SIZE)
    gross_return_pct = 0.0
    for symbol, leg in basket.items():
        entry_price = _price_at(histories[symbol], t)
        exit_price = _price_at(histories[symbol], t_exit)
        if entry_price is None or exit_price is None or entry_price == 0:
            return None
        raw_delta_pct = (exit_price - entry_price) / entry_price * 100.0
        symbol_return_pct = raw_delta_pct if leg == "long" else -raw_delta_pct
        gross_return_pct += weight * symbol_return_pct

    changed_symbols, turnover = changed_symbols_and_turnover(prev_basket, basket)
    cost_pct = sum(weight * round_trip_cost_pct_for(s, base_cost_pct) for s in changed_symbols)

    funding_total_pct = 0.0
    for symbol, leg in basket.items():
        events = funding_by_symbol.get(symbol, [])
        funding_total_pct += weight * funding_cost_pct(leg, t, t_exit, events)

    net_return_pct = gross_return_pct - cost_pct - funding_total_pct

    btc_return_pct: float | None = None
    if BTC_SYMBOL in histories:
        btc_entry = _price_at(histories[BTC_SYMBOL], t)
        btc_exit = _price_at(histories[BTC_SYMBOL], t_exit)
        if btc_entry is not None and btc_exit is not None and btc_entry != 0:
            btc_return_pct = (btc_exit - btc_entry) / btc_entry * 100.0

    return RebalancePeriod(
        rebalance_time=t,
        exit_time=t_exit,
        n_eligible=len(momentum_by_symbol),
        basket=dict(basket),
        gross_return_pct=round(gross_return_pct, 6),
        cost_pct=round(cost_pct, 6),
        funding_cost_pct=round(funding_total_pct, 6),
        net_return_pct=round(net_return_pct, 6),
        turnover=round(turnover, 6),
        btc_return_pct=round(btc_return_pct, 6) if btc_return_pct is not None else None,
    )


def pearson_correlation(xs: list[float], ys: list[float]) -> float | None:
    """Pearson correlation coefficient between two equal-length series.
    Returns ``None`` if fewer than 2 paired points or either series has
    zero variance (correlation undefined)."""
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0 or var_y == 0:
        return None
    return cov / math.sqrt(var_x * var_y)


# ``_safe_block_bootstrap`` used to be defined here as a private copy;
# it now lives once as ``scanner.validation.statistical_validation.safe_block_bootstrap``.
_safe_block_bootstrap = safe_block_bootstrap


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
    """Runs the full cross-sectional-momentum pipeline for ``symbols`` over
    data anchored at the frozen ``end_timestamp_ms`` and returns the
    JSON-serializable report dict.

    ``end_timestamp_ms`` is a required keyword argument on purpose -- every
    call into ``cached_load_extended_history``/
    ``cached_fetch_full_funding_history`` below passes it explicitly, so
    the reserved holdout after it can never leak into this run.

    A symbol with no usable candle data is skipped (recorded in the
    report's ``skipped_symbols``) rather than aborting the whole run; a
    symbol with candle data but no funding data is still used for ranking
    and price returns, simply contributing 0 funding cost for any period
    it is held (funding is additive on top of price return/cost, not a
    gating requirement for participation -- a documented design choice,
    since the pre-registration doc does not say whether missing funding
    data should exclude a symbol from the universe).
    """
    point_in_time_windows = None
    if point_in_time_universe_cache is not None:
        try:
            point_in_time_windows = load_universe_cache(point_in_time_universe_cache)
        except (FileNotFoundError, OSError, ValueError) as exc:
            print(f"WARNING: could not load point-in-time universe cache ({exc}); filter disabled.")

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

        histories[symbol] = history
        candle_fingerprints.append(dataset_fingerprint(history))

        funding_events = cached_fetch_full_funding_history(symbol, end_timestamp_ms, cache_dir=funding_cache_dir)
        funding_by_symbol[symbol] = funding_events
        if funding_events:
            funding_fingerprints.append(funding_dataset_fingerprint(funding_events))
        else:
            print(f"{symbol}: no funding data available (funding cost will be 0 for this symbol)")
        print(f"{symbol}: {len(history)} candles, {len(funding_events)} funding settlements")

    grid = build_rebalance_grid(histories)

    periods: list[RebalancePeriod] = []
    n_skipped_min_eligible = 0
    prev_basket: dict[str, str] | None = None
    for t in grid:
        t_exit = t + timedelta(hours=REBALANCE_HOURS)
        if not grid or t_exit > grid[-1]:
            # No further grid point to price the exit against -- cannot
            # form a basket for this (necessarily the last) grid entry
            # without peeking past available data.
            continue
        momentum_by_symbol = eligible_symbols_at(t, histories, point_in_time_windows)
        basket = form_basket(momentum_by_symbol)
        if basket is None:
            n_skipped_min_eligible += 1
            continue
        period = build_rebalance_period(
            t, t_exit, momentum_by_symbol, basket, prev_basket, histories, funding_by_symbol, base_cost_pct
        )
        if period is None:
            n_skipped_min_eligible += 1
            continue
        periods.append(period)
        prev_basket = basket

    n_independent = len(periods)
    net_returns = [p.net_return_pct for p in periods]
    gross_returns = [p.gross_return_pct for p in periods]
    turnovers = [p.turnover for p in periods]
    btc_returns = [p.btc_return_pct for p in periods if p.btc_return_pct is not None]
    net_returns_for_beta = [p.net_return_pct for p in periods if p.btc_return_pct is not None]

    win_rate_pct = round(sum(1 for r in net_returns if r > 0) / len(net_returns) * 100.0, 2) if net_returns else None
    avg_turnover = round(sum(turnovers) / len(turnovers), 6) if turnovers else None
    # Documented design choice: sample standard deviation (ddof=1), the
    # standard inferential choice when estimating a population sigma from
    # a single finite observed sample -- same convention as the funding
    # hypothesis's implementation, for consistency.
    measured_sigma_pct = round(statistics.stdev(net_returns), 6) if len(net_returns) >= 2 else None

    beta_correlation = pearson_correlation(net_returns_for_beta, btc_returns)
    beta_flag = bool(beta_correlation is not None and abs(beta_correlation) > BETA_FLAG_THRESHOLD)

    net_bootstrap = _safe_block_bootstrap(net_returns, bootstrap_n_resamples, bootstrap_confidence_pct)
    gross_bootstrap = _safe_block_bootstrap(gross_returns, bootstrap_n_resamples, bootstrap_confidence_pct)

    n_required: int | None = None
    undecidable: bool | None = None
    if measured_sigma_pct is not None and measured_sigma_pct > 0:
        n_required = math.ceil(((Z_ALPHA_95 + Z_BETA_80) ** 2) * (measured_sigma_pct**2) / (TARGET_EDGE_PCT**2))
        undecidable = n_independent < n_required

    dataset_hash = combined_dataset_fingerprint(candle_fingerprints + funding_fingerprints)

    return {
        "hypothesis": "cross_sectional_momentum",
        "generated_at": datetime.now(UTC).isoformat(),
        "end_timestamp_ms": end_timestamp_ms,
        "end_timestamp_iso": datetime.fromtimestamp(end_timestamp_ms / 1000.0, tz=UTC).isoformat(),
        "spec": {
            "lookback_hours": LOOKBACK_HOURS,
            "rebalance_hours": REBALANCE_HOURS,
            "basket_size_n": BASKET_SIZE,
            "min_eligible_symbols": MIN_ELIGIBLE_SYMBOLS,
            "base_cost_pct": base_cost_pct,
            "price_timeframe": price_timeframe,
        },
        "symbols_requested": list(symbols),
        "symbols_used": sorted(histories.keys()),
        "skipped_symbols": skipped_symbols,
        "point_in_time_universe": (
            {"cache_path": point_in_time_universe_cache}
            if point_in_time_windows is not None
            else None
        ),
        "n_rebalance_grid_points": len(grid),
        "n_skipped_min_eligible_or_pricing": n_skipped_min_eligible,
        "n_independent": n_independent,
        "win_rate_pct": win_rate_pct,
        "avg_daily_turnover": avg_turnover,
        "beta_check": {
            "correlation_with_btc": round(beta_correlation, 6) if beta_correlation is not None else None,
            "n_paired_observations": len(btc_returns),
            "threshold": BETA_FLAG_THRESHOLD,
            "market_neutral_violated": beta_flag,
        },
        "measured_sigma_pct": measured_sigma_pct,
        "sigma_definition": "sample standard deviation (ddof=1) of net per-rebalance basket returns",
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
        "rebalance_periods": [
            {
                "rebalance_time": p.rebalance_time.isoformat(),
                "exit_time": p.exit_time.isoformat(),
                "n_eligible": p.n_eligible,
                "basket": p.basket,
                "gross_return_pct": p.gross_return_pct,
                "cost_pct": p.cost_pct,
                "funding_cost_pct": p.funding_cost_pct,
                "net_return_pct": p.net_return_pct,
                "turnover": p.turnover,
                "btc_return_pct": p.btc_return_pct,
            }
            for p in periods
        ],
    }


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-sectional momentum backtest (pre-registered, look-once trial family #3)."
    )
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS), help="Symbols to include")
    parser.add_argument(
        "--end-timestamp",
        required=True,
        help="Frozen end-of-train/validation boundary (ISO 8601 or raw epoch-ms); holdout after it is never touched",
    )
    parser.add_argument("--output", default="scanner_data/cross_sectional_momentum_backtest_report.json")
    parser.add_argument("--price-timeframe", default=DEFAULT_PRICE_TIMEFRAME)
    parser.add_argument("--price-candles", type=int, default=DEFAULT_PRICE_CANDLES)
    parser.add_argument("--kline-cache-dir", default=DEFAULT_KLINE_CACHE_DIR)
    parser.add_argument("--funding-cache-dir", default=DEFAULT_FUNDING_CACHE_DIR)
    parser.add_argument("--cost-pct", type=float, default=DEFAULT_ROUND_TRIP_COST_PCT)
    parser.add_argument("--bootstrap-resamples", type=int, default=BLOCK_BOOTSTRAP_N_RESAMPLES)
    parser.add_argument(
        "--point-in-time-universe-cache",
        default=None,
        help="Optional path to a historical_universe.py cache (same mechanism as the prior two hypotheses)",
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
    print(f"avg_daily_turnover={report['avg_daily_turnover']}")
    print(f"beta_check={report['beta_check']}")
    print(f"net_block_bootstrap={report['net_block_bootstrap']}")
    print(f"gross_block_bootstrap={report['gross_block_bootstrap']}")
    print(f"undecidable_threshold={report['undecidable_threshold']}")
    print(f"dataset_hash={report['dataset_hash']}")
    print(f"Report written to {args.output}")
    print("DONE")


if __name__ == "__main__":
    _main()
