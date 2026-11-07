"""Generalized cross-sectional long/short basket backtest engine -- Strategy
Lab Layer 1 infrastructure (``STRATEGY_LAB_DESIGN.md`` / Korak 10,
``PHASE8_PROFITABILITY_RESEARCH_TODO.md``).

This is a DELIBERATE EXTRACTION/GENERALIZATION of the mechanics already
proven and frozen in ``scanner.backtesting.cross_sectional_momentum_backtest``
(trial/family #3, closed in Korak 9) -- basket construction, turnover-based
costs, funding cost, BTC-beta check, block-bootstrap, dataset hashing are
all the SAME, battle-tested logic, just parametrized instead of hard-coded
to one frozen ranking signal (24h momentum).

It exists so new Layer 1 templates that are "the same cross-sectional
long-top/short-bottom mechanics, different ranking metric" (momentum_30d,
low_vol, funding_carry_xs -- see Korak 10) do not each require a brand-new,
independently-written 500-line backtest module copy-pasted from the frozen
one. The frozen ``cross_sectional_momentum_backtest.py`` module itself is
NEVER imported from or modified by this file -- its one pre-registered run
stays exactly as it was (see that module's own "do NOT tune" rule).

Importantly, this module is Layer 1 (Strategy Lab) infrastructure, not a
new pre-registered hypothesis: each ranking signal plugged in here is run
through the Strategy Lab miner's small, explicitly-declared, bounded grid
(``strategy_miner.ALL_TEMPLATE_VARIANTS``), same anti-overfitting
discipline as the rest of Strategy Lab, rather than this engine being
tuned freely.

A ``RankingFn`` computes, for one rebalance timestamp ``t``, a score per
eligible symbol where a HIGHER score means "more attractive for the LONG
leg" (and therefore a lower score means "more attractive for the SHORT
leg") -- so inverting a factor (e.g. "I want to go long low-volatility
symbols") is simply a matter of writing the ranking function to return the
negative of the raw metric, not a separate code path through basket
formation.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

import pandas as pd

from scanner.backtesting.cross_sectional_momentum_backtest import (
    BTC_SYMBOL,
    RebalancePeriod,
    _leg_weight,
    _price_at,
    _safe_block_bootstrap,
    changed_symbols_and_turnover,
    pearson_correlation,
)
from scanner.backtesting.data_snapshot import (
    DEFAULT_FUNDING_CACHE_DIR,
    DEFAULT_KLINE_CACHE_DIR,
    cached_fetch_full_funding_history,
    cached_load_extended_history,
    combined_dataset_fingerprint,
    dataset_fingerprint,
    funding_dataset_fingerprint,
)
from scanner.backtesting.funding_simulator import FundingEvent
from scanner.backtesting.historical_universe import is_symbol_in_universe_at, load_universe_cache
from scanner.calibration import DEFAULT_ROUND_TRIP_COST_PCT, round_trip_cost_pct_for
from scanner.validation.statistical_validation import (
    BLOCK_BOOTSTRAP_CONFIDENCE_PCT,
    BLOCK_BOOTSTRAP_N_RESAMPLES,
)
from tradingagents.dataflows.errors import VendorError

# Same 20-symbol universe as the frozen cross-sectional-momentum hypothesis
# -- reused verbatim as the default (callers may override with a bigger,
# point-in-time-ranked universe, e.g. the 170-symbol Korak 10 build).
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

DEFAULT_PRICE_TIMEFRAME = "1h"
DEFAULT_PRICE_CANDLES = 8000

TARGET_EDGE_PCT = 0.3
Z_ALPHA_95 = 1.96
Z_BETA_80 = 0.84
BETA_FLAG_THRESHOLD = 0.3


class RankingFn(Protocol):
    def __call__(
        self,
        t: datetime,
        histories: dict[str, pd.DataFrame],
        funding_by_symbol: dict[str, list[FundingEvent]],
        lookback_hours: int,
    ) -> dict[str, float]: ...


@dataclass(frozen=True)
class CrossSectionalConfig:
    """Bounded, explicitly-declared parameters for one Layer 1 template
    variant -- mirrors the frozen pre-registration doc's "section 2" style
    of spec, just reusable across templates instead of one-off constants."""

    hypothesis: str
    ranking_fn: RankingFn
    lookback_hours: int
    rebalance_hours: int
    basket_size: int = 5
    min_eligible_multiplier: int = 2

    @property
    def min_eligible_symbols(self) -> int:
        return self.min_eligible_multiplier * self.basket_size


def build_rebalance_grid(histories: dict[str, pd.DataFrame], rebalance_hours: int) -> list[datetime]:
    """Same anchoring rule as the frozen momentum module's
    ``build_rebalance_grid``, parametrized by rebalance cadence."""
    if not histories:
        return []
    earliest = min(h["open_time"].min().to_pydatetime() for h in histories.values())
    latest = max(h["open_time"].max().to_pydatetime() for h in histories.values())
    grid: list[datetime] = []
    t = earliest
    step = timedelta(hours=rebalance_hours)
    while t <= latest:
        grid.append(t)
        t += step
    return grid


def momentum_ranking_fn(
    t: datetime,
    histories: dict[str, pd.DataFrame],
    funding_by_symbol: dict[str, list[FundingEvent]],
    lookback_hours: int,
) -> dict[str, float]:
    """Trailing % price change over ``lookback_hours`` -- higher return =
    more attractive for the long leg (plain momentum, same ranking
    direction as the frozen 24h hypothesis, just a configurable lookback
    for the Korak 10 ``momentum_30d`` template)."""
    t_prev = t - timedelta(hours=lookback_hours)
    result: dict[str, float] = {}
    for symbol, history in histories.items():
        price_now = _price_at(history, t)
        price_prev = _price_at(history, t_prev)
        if price_now is None or price_prev is None or price_prev == 0:
            continue
        result[symbol] = (price_now / price_prev - 1.0) * 100.0
    return result


def low_vol_ranking_fn(
    t: datetime,
    histories: dict[str, pd.DataFrame],
    funding_by_symbol: dict[str, list[FundingEvent]],
    lookback_hours: int,
) -> dict[str, float]:
    """Negative trailing realized volatility over ``lookback_hours``
    (standard deviation of hourly log-ish returns, sample stdev, ddof=1)
    -- NEGATED so the symbol with the LOWEST volatility gets the HIGHEST
    score (long leg), matching the well-known "low-volatility factor"
    direction (long calm symbols, short turbulent ones), not plain
    momentum. Requires at least 3 priced candles in the lookback window to
    compute a stdev; fewer -> symbol excluded from this rebalance."""
    t_start = t - timedelta(hours=lookback_hours)
    result: dict[str, float] = {}
    for symbol, history in histories.items():
        window = history[(history["open_time"] > pd.Timestamp(t_start)) & (history["open_time"] <= pd.Timestamp(t))]
        closes = window["close"].astype(float).tolist()
        if len(closes) < 3:
            continue
        returns = [(closes[i] / closes[i - 1] - 1.0) * 100.0 for i in range(1, len(closes)) if closes[i - 1] != 0]
        if len(returns) < 2:
            continue
        vol = statistics.stdev(returns)
        result[symbol] = -vol
    return result


def funding_carry_ranking_fn(
    t: datetime,
    histories: dict[str, pd.DataFrame],
    funding_by_symbol: dict[str, list[FundingEvent]],
    lookback_hours: int,
) -> dict[str, float]:
    """Negative trailing average funding rate over ``lookback_hours`` --
    NEGATED so the symbol with the most NEGATIVE average funding rate
    (longs are PAID funding) gets the highest score (long leg), and the
    symbol with the most POSITIVE average funding (shorts are paid) is
    most attractive short -- the standard "collect funding" carry-trade
    direction, dollar-neutral so it is a funding-income bet, not a
    directional price bet. Only settlements strictly within
    ``[t - lookback_hours, t]`` are used (no look-ahead); a symbol with no
    settlements in that window is excluded from this rebalance."""
    t_start = t - timedelta(hours=lookback_hours)
    result: dict[str, float] = {}
    for symbol, events in funding_by_symbol.items():
        if symbol not in histories:
            continue
        rates = [e.rate for e in events if t_start < e.timestamp <= t]
        if not rates:
            continue
        avg_rate = sum(rates) / len(rates)
        result[symbol] = -avg_rate * 100.0
    return result


def form_basket(scores_by_symbol: dict[str, float], config: CrossSectionalConfig) -> dict[str, str] | None:
    """Top-N/bottom-N basket by ``scores_by_symbol`` (higher = long),
    ``None`` if fewer than ``config.min_eligible_symbols`` are scored this
    rebalance (same frozen skip rule as the momentum module, parametrized
    by basket size)."""
    if len(scores_by_symbol) < config.min_eligible_symbols:
        return None
    ranked = sorted(scores_by_symbol.items(), key=lambda kv: kv[1], reverse=True)
    n = config.basket_size
    longs = ranked[:n]
    shorts = ranked[-n:]
    basket: dict[str, str] = {}
    for symbol, _ in longs:
        basket[symbol] = "long"
    for symbol, _ in shorts:
        basket[symbol] = "short"
    return basket


def build_rebalance_period(
    t: datetime,
    t_exit: datetime,
    n_eligible: int,
    basket: dict[str, str],
    prev_basket: dict[str, str] | None,
    histories: dict[str, pd.DataFrame],
    funding_by_symbol: dict[str, list[FundingEvent]],
    base_cost_pct: float,
    basket_size: int,
) -> RebalancePeriod | None:
    """Identical pricing/cost/funding/beta mechanics to the frozen
    momentum module's ``build_rebalance_period``, parametrized by
    ``basket_size`` (for the leg-weight denominator)."""
    from scanner.backtesting.funding_simulator import funding_cost_pct

    weight = _leg_weight(basket_size)
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
        n_eligible=n_eligible,
        basket=dict(basket),
        gross_return_pct=round(gross_return_pct, 6),
        cost_pct=round(cost_pct, 6),
        funding_cost_pct=round(funding_total_pct, 6),
        net_return_pct=round(net_return_pct, 6),
        turnover=round(turnover, 6),
        btc_return_pct=round(btc_return_pct, 6) if btc_return_pct is not None else None,
    )


def run_backtest(
    config: CrossSectionalConfig,
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
    """Runs ``config``'s cross-sectional long/short basket pipeline over
    data anchored at the frozen ``end_timestamp_ms`` -- same no-look-ahead
    guarantee and overall shape as the frozen momentum module's
    ``run_backtest``, generalized over ranking signal/lookback/rebalance
    cadence/basket size."""
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
            continue
        if history.empty:
            skipped_symbols.append({"symbol": symbol, "reason": "no candle data available"})
            continue

        histories[symbol] = history
        candle_fingerprints.append(dataset_fingerprint(history))

        funding_events = cached_fetch_full_funding_history(symbol, end_timestamp_ms, cache_dir=funding_cache_dir)
        funding_by_symbol[symbol] = funding_events
        if funding_events:
            funding_fingerprints.append(funding_dataset_fingerprint(funding_events))

    grid = build_rebalance_grid(histories, config.rebalance_hours)

    periods: list[RebalancePeriod] = []
    n_skipped_min_eligible = 0
    prev_basket: dict[str, str] | None = None
    for t in grid:
        t_exit = t + timedelta(hours=config.rebalance_hours)
        if not grid or t_exit > grid[-1]:
            continue
        if point_in_time_windows is not None:
            eligible_now = {
                s for s in histories if is_symbol_in_universe_at(point_in_time_windows, s, t)
            }
            scoped_histories = {s: histories[s] for s in eligible_now}
        else:
            scoped_histories = histories
        scores = config.ranking_fn(t, scoped_histories, funding_by_symbol, config.lookback_hours)
        basket = form_basket(scores, config)
        if basket is None:
            n_skipped_min_eligible += 1
            continue
        period = build_rebalance_period(
            t, t_exit, len(scores), basket, prev_basket, histories, funding_by_symbol, base_cost_pct,
            config.basket_size,
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
        "hypothesis": config.hypothesis,
        "generated_at": datetime.now(UTC).isoformat(),
        "end_timestamp_ms": end_timestamp_ms,
        "end_timestamp_iso": datetime.fromtimestamp(end_timestamp_ms / 1000.0, tz=UTC).isoformat(),
        "spec": {
            "lookback_hours": config.lookback_hours,
            "rebalance_hours": config.rebalance_hours,
            "basket_size_n": config.basket_size,
            "min_eligible_symbols": config.min_eligible_symbols,
            "base_cost_pct": base_cost_pct,
            "price_timeframe": price_timeframe,
        },
        "symbols_requested": list(symbols),
        "symbols_used": sorted(histories.keys()),
        "skipped_symbols": skipped_symbols,
        "point_in_time_universe": (
            {"cache_path": point_in_time_universe_cache} if point_in_time_windows is not None else None
        ),
        "n_rebalance_grid_points": len(grid),
        "n_skipped_min_eligible_or_pricing": n_skipped_min_eligible,
        "n_independent": n_independent,
        "win_rate_pct": win_rate_pct,
        "avg_turnover": avg_turnover,
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
