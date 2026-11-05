"""Phase 8 Korak 2: portfolio-level counterfactual backtest.

Answers the project's central open question — does ``agents_crypto.best_signal``'s
ranking logic have a genuine, provable edge, net of realistic portfolio
constraints and trading costs — by replaying history chronologically with
**no look-ahead** and simulating how a single account, with a limited number
of concurrent positions, would actually have traded the signals the live
system would have surfaced at each point in time.

This is deliberately NOT wired into the always-running scanner service: it
fetches ~1500 klines per (symbol, timeframe) pair over the network and does
nontrivial CPU work per candle, which the project's 2 vCPU/4GB VPS has no
spare budget for during normal operation (see PHASE8_PROFITABILITY_RESEARCH_TODO.md).
Run it on demand, e.g.::

    python -m scanner.backtesting.portfolio_backtest --symbols BTCUSDT ETHUSDT SOLUSDT

Design notes (see PHASE8_PROFITABILITY_RESEARCH_TODO.md "Korak 2" for the
full discussion):

* Calibration stats are **per (symbol, timeframe)**, exactly like production
  (``agents_crypto.calibration_lookup.get_cached_calibration_stats`` keys its
  cache by symbol+timeframe, never pooling across symbols) — this module
  mirrors that by building an independent, point-in-time-gated stats
  accumulator per (symbol, timeframe), fed only by that same symbol's own
  historical cases.
* A historical case's *existence* is knowable at ``entry_time``, but its
  *outcome* only becomes knowable at ``resolution_time`` (``entry_time`` plus
  the horizon). The accumulator is therefore gated by resolution time, not
  entry time: before evaluating whether a candidate at time T is
  "significant", every case whose outcome resolved at or before T — and only
  those — is folded into that (symbol, timeframe, label, regime) bucket's
  stats. This avoids a subtler look-ahead bug than sorting by entry_time
  alone would miss.
* The regime-vs-blended fallback mirrors
  ``agents_crypto.calibration_lookup.find_cached_regime_matched_stats``
  exactly: if the current regime bucket has ANY resolved cases at all, it is
  used (even below the minimum sample size — that gate is applied
  afterwards, uniformly, to whichever bucket was chosen); otherwise the
  all-regime blend is used.
* Net returns are direction-corrected (short cases negate the raw price
  delta) and cost-adjusted (round-trip fee/spread estimate, optionally plus
  extra slippage), exactly mirroring ``scanner.calibration.summarize``'s
  correction — ``LabeledCase.forward_return_pct`` itself is raw and
  uncorrected, so any direct consumer must replicate this.
* Correlation control is a deliberately simplified v1 proxy:
  ``max_concurrent_per_direction`` caps how many same-direction positions can
  be open at once, standing in for the live system's BTC-divergence
  correlation tag (``agents_crypto.best_signal._tag_correlated_exposure``),
  which would require a parallel point-in-time BTC trend series per
  timestamp to replicate here. This is a known, documented approximation,
  not a claim of equivalent rigor.
* Sharpe/Sortino are annualized using trades-per-year (``n_trades /
  years_span``) rather than resampling the equity curve to fixed calendar
  periods, since trades are irregularly spaced in time. This is a common,
  simple retail-backtest approximation — it is reported as such, not as a
  rigorous time-series Sharpe ratio.
* Turnover/cost accounting is approximated as
  ``n_trades * (round_trip_cost_pct + extra_slippage_pct)``, non-compounded
  (i.e. expressed in percentage points subtracted from gross return, not
  compounded per trade — the per-trade cost IS compounded through the equity
  curve already, since it is baked into each trade's net_return_pct before
  the R-multiple/equity impact is computed; this aggregate figure is only a
  convenience summary of total cost drag).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from agents_crypto.best_signal import (
    _DEFAULT_MIN_SAMPLE_SIZE,
    BEST_SIGNAL_HORIZON_CANDLES,
    BEST_SIGNAL_MIN_SAMPLE_SIZE,
    TRACKED_TIMEFRAMES,
)
from scanner.backtesting.funding_simulator import FundingEvent, fetch_funding_history
from scanner.backtesting.historical_universe import (
    UniverseWindow,
    is_symbol_in_universe_at,
    load_universe_cache,
)
from scanner.calibration import (
    DEFAULT_ROUND_TRIP_COST_PCT,
    LABEL_DIRECTION,
    OOS_MIN_TEST_N,
    OOS_TEST_FRACTION,
    LabeledCase,
    OutOfSampleResult,
    backtest_symbol,
    load_extended_history,
    round_trip_cost_pct_for,
    validate_out_of_sample,
    wilson_score_interval,
)
from scanner.signal_tracking import TIMEFRAME_MINUTES
from scanner.validation.experiment_registry import (
    count_prior_trials,
    fetch_experiments,
    init_db as init_experiment_registry,
    record_experiment,
)
from scanner.validation.independence import (
    MIN_INDEPENDENT_EVENTS_N,
    IndependenceClusterStats,
    summarize_with_independence,
)
from scanner.validation.statistical_validation import deflated_sharpe_ratio
from tradingagents.dataflows.errors import VendorError


@dataclass
class CandidateEvent:
    """One point in time where the live ranking logic would have surfaced a
    statistically-significant, tradable signal for a single (symbol,
    timeframe) — the best-ranked label among whatever fired on that candle,
    exactly mirroring how ``agents_crypto.best_signal.compute_candidates``
    picks one winner per (symbol, timeframe, direction) bucket."""

    symbol: str
    timeframe: str
    direction: str
    label: str
    entry_time: datetime
    resolution_time: datetime
    net_return_pct: float
    r_multiple: float | None
    n: int
    win_rate_pct: float
    win_rate_ci_low_pct: float
    regime: str | None
    regime_fallback: bool


@dataclass
class Trade:
    """A closed, simulated portfolio position."""

    symbol: str
    timeframe: str
    direction: str
    label: str
    entry_time: datetime
    resolution_time: datetime
    net_return_pct: float
    r_multiple: float | None
    equity_before: float
    pnl: float
    equity_after: float
    win_rate_ci_low_pct: float
    n: int


@dataclass
class PortfolioBacktestConfig:
    symbols: tuple[str, ...]
    timeframes: tuple[str, ...] = TRACKED_TIMEFRAMES
    horizon_candles: int = BEST_SIGNAL_HORIZON_CANDLES
    kline_limit: int = 1500
    # Extra candles fetched *before* kline_limit's tradable window purely to
    # seed the stats accumulator, so the first tradable candle already has
    # realistic calibration history behind it (mirrors live production,
    # which always already has ~1500 candles of prior calibration before a
    # decision is ever shown to a user). Set to 0 to reproduce the old
    # (cold-start-biased) v1 behaviour.
    warmup_candles: int = 1500
    initial_capital: float = 10_000.0
    # Fixed-fractional position sizing: each trade risks this % of equity
    # *at the time it was opened* (not current equity at close), same
    # convention as standard fixed-fractional backtesting.
    risk_per_trade_pct: float = 1.0
    max_concurrent_positions: int = 5
    max_concurrent_per_direction: int = 3
    round_trip_cost_pct: float = DEFAULT_ROUND_TRIP_COST_PCT
    extra_slippage_pct: float = 0.0
    # Phase 8 Korak 5 (execution realism): when True, round_trip_cost_pct is
    # no longer applied flat to every symbol -- each symbol's own liquidity
    # tier surcharge (see scanner.calibration.round_trip_cost_pct_for) is
    # added on top instead. Majors (BTC/ETH/BNB) are unaffected (+0.0%); the
    # existing test suite only ever uses majors with a base cost of 0.0, so
    # this is on by default with zero risk of changing any existing
    # numeric assertion.
    use_symbol_cost_tiers: bool = True
    # Phase 8 Korak 5 (execution realism): when True, fetch each symbol's
    # real funding-rate settlement history once (cached across timeframes)
    # and charge every directional case the funding it would actually have
    # accrued between entry and exit. Off by default because it requires a
    # network call per symbol -- existing callers/tests that monkeypatch
    # backtest_symbol without a funding_events kwarg are unaffected unless
    # they opt in.
    include_funding_cost: bool = False
    # Phase 8 Korak 5 (execution realism): how to resolve a single candle
    # whose high/low range touches BOTH the stop and the target -- OHLC data
    # alone can't say which was hit first. "conservative" (default,
    # unchanged behaviour) assumes the stop hit first, never overstating a
    # label's edge. "optimistic" assumes the target hit first -- run the
    # same backtest with both and compare: if the win rate swings wildly,
    # that label's apparent edge is mostly an artifact of this ambiguity,
    # not a real signal. See scanner.calibration._walk_forward_exit's
    # docstring for the full explanation (incl. why a true higher-
    # resolution fix is NOT implemented).
    sl_tp_tie_break: str = "conservative"
    # Phase 8 Korak 6 (audit follow-up): in-sample/out-of-sample split check
    # (scanner.calibration.validate_out_of_sample) -- flags labels whose
    # apparent edge only held up on the data used to measure it, not on the
    # held-out most-recent `oos_test_fraction` of cases. Pure re-computation
    # on cases already fetched for the backtest (no extra network calls), so
    # unlike include_funding_cost this is safe to default to True.
    include_oos_validation: bool = True
    oos_test_fraction: float = OOS_TEST_FRACTION
    oos_min_test_n: int = OOS_MIN_TEST_N
    # Phase 8 Korak 6 (this one is the actual "Korak 6" per
    # PHASE8_PROFITABILITY_RESEARCH_TODO.md -- the "Korak 6" label on
    # include_oos_validation above is a pre-existing, harmless naming quirk
    # that predates this field and is left alone as out of scope here):
    # sample-independence correction. Every Wilson interval above (and the
    # OOS check above it) silently assumes each case in a (label, horizon)
    # group is an independent trial, but several symbols can fire the same
    # label from one correlated market-wide move (e.g. a BTC pump) within
    # the same window -- one event counted as several. This re-runs the
    # Wilson interval over the number of time-overlapping-event CLUSTERS
    # instead of raw case count. Pure re-computation on cases already
    # fetched (no extra network calls), so safe to default to True, same as
    # include_oos_validation. See scanner.validation.independence's module
    # docstring for the full method and its deliberately bounded scope.
    include_independence_validation: bool = True
    independence_min_n: int = MIN_INDEPENDENT_EVENTS_N
    # Phase 8 Korak 4 (survivorship-bias fix): when set, any candidate event
    # whose symbol was NOT in the point-in-time top-N universe at its own
    # entry_time is dropped before the portfolio simulation runs -- i.e. a
    # trade is only counted if that symbol would actually have been
    # selectable back then, not just because it is in today's --symbols
    # list. See scanner.backtesting.historical_universe.
    point_in_time_windows: tuple[UniverseWindow, ...] | None = None
    # Display-only metadata (which cache file produced point_in_time_windows,
    # if any) so the dashboard report can show it without re-deriving it.
    point_in_time_universe_cache_path: str | None = None


@dataclass
class PortfolioBacktestResult:
    trades: list[Trade]
    equity_curve: list[tuple[datetime, float]]
    metrics: dict[str, Any]
    skipped_symbols: list[str] = field(default_factory=list)
    # How many otherwise-significant candidate events were dropped because
    # their symbol was not in the point-in-time universe at entry_time (0 if
    # config.point_in_time_windows was None, i.e. the filter was off).
    point_in_time_filtered_count: int = 0
    # Phase 8 Korak 6 (audit follow-up): in/out-of-sample split check, one
    # entry per (label, horizon) seen across every symbol/timeframe in this
    # run -- pooled across the whole universe since OutOfSampleResult groups
    # by label, not by symbol. Empty if config.include_oos_validation is
    # False or there were no cases at all.
    oos_validation: list[OutOfSampleResult] = field(default_factory=list)
    # Phase 8 Korak 6: same grouping as oos_validation above, but each
    # entry compares the raw (today's) win-rate CI against one recomputed
    # over independent-event clusters -- see
    # scanner.validation.independence.summarize_with_independence. Empty if
    # config.include_independence_validation is False or there were no
    # cases at all.
    independence_validation: list[IndependenceClusterStats] = field(default_factory=list)


def _parse_time(iso_timestamp: str) -> datetime:
    ts = datetime.fromisoformat(iso_timestamp)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return ts


def _net_return_pct(case: LabeledCase, cost_pct: float, extra_slippage_pct: float) -> float:
    """Direction-corrected, cost-adjusted net return for one case — mirrors
    ``scanner.calibration.summarize``'s correction, which is NOT baked into
    ``LabeledCase.forward_return_pct`` itself.

    Also subtracts ``case.funding_cost_pct`` (Phase 8 Korak 5) -- 0.0 for
    every case unless ``backtest_symbol`` was given real ``funding_events``,
    so this is a no-op for existing callers/tests that never populate it."""
    direction = LABEL_DIRECTION[case.label]
    signed = case.forward_return_pct if direction == "long" else -case.forward_return_pct
    return signed - cost_pct - extra_slippage_pct - case.funding_cost_pct


def _build_candidate_events(
    symbol: str,
    timeframe: str,
    cases: list[LabeledCase],
    min_n: int,
    cost_pct: float,
    extra_slippage_pct: float,
    horizon_candles: int,
    decision_start_time: datetime | None = None,
) -> list[CandidateEvent]:
    """Replay one (symbol, timeframe)'s own case history, in isolation,
    through a resolution-time-gated stats accumulator, and emit one
    CandidateEvent per candle where a significant directional label fired —
    exactly the "would this have been surfaced live" decision.

    ``decision_start_time``, if given, marks the end of a "warm-up" prefix:
    candles at or before this time still feed the stats accumulator (so the
    first tradable candle already has realistic calibration history behind
    it, mirroring live production's ever-present ~1500-candle calibration
    cache) but are never themselves emitted as CandidateEvents."""
    minutes_per_candle = TIMEFRAME_MINUTES.get(timeframe, 60)
    horizon_delta = timedelta(minutes=minutes_per_candle * horizon_candles)

    directional = [c for c in cases if LABEL_DIRECTION.get(c.label) is not None]
    if not directional:
        return []

    by_entry_time: dict[datetime, list[LabeledCase]] = defaultdict(list)
    for case in directional:
        by_entry_time[_parse_time(case.entry_time)].append(case)

    resolution_order = sorted(directional, key=lambda c: _parse_time(c.entry_time) + horizon_delta)

    all_stats: dict[str, list[float]] = defaultdict(list)
    regime_stats: dict[tuple[str, str], list[float]] = defaultdict(list)

    events: list[CandidateEvent] = []
    pointer = 0
    for entry_dt in sorted(by_entry_time):
        # Flush every case whose outcome is already knowable as of entry_dt
        # into the accumulator BEFORE evaluating this candle's candidates.
        while pointer < len(resolution_order):
            resolved_case = resolution_order[pointer]
            resolution_time = _parse_time(resolved_case.entry_time) + horizon_delta
            if resolution_time > entry_dt:
                break
            net_return = _net_return_pct(resolved_case, cost_pct, extra_slippage_pct)
            all_stats[resolved_case.label].append(net_return)
            if resolved_case.regime is not None:
                regime_stats[(resolved_case.label, resolved_case.regime)].append(net_return)
            pointer += 1

        if decision_start_time is not None and entry_dt <= decision_start_time:
            continue  # warm-up candle: fed the accumulator above, but not tradable

        significant: list[CandidateEvent] = []
        for case in by_entry_time[entry_dt]:
            regime_bucket = regime_stats.get((case.label, case.regime)) if case.regime is not None else None
            if regime_bucket:
                returns, used_fallback = regime_bucket, False
            else:
                returns, used_fallback = all_stats.get(case.label, []), True
            n = len(returns)
            if n < min_n:
                continue
            wins = sum(1 for r in returns if r > 0)
            win_rate_pct = wins / n * 100.0
            ci_low, _ci_high = wilson_score_interval(win_rate_pct, n)
            if ci_low is None or ci_low <= 50.0:
                continue
            net_return_this_case = _net_return_pct(case, cost_pct, extra_slippage_pct)
            r_multiple = (
                net_return_this_case / case.sl_distance_pct
                if case.sl_distance_pct and case.sl_distance_pct > 0
                else None
            )
            significant.append(
                CandidateEvent(
                    symbol=symbol,
                    timeframe=timeframe,
                    direction=LABEL_DIRECTION[case.label],
                    label=case.label,
                    entry_time=entry_dt,
                    resolution_time=entry_dt + horizon_delta,
                    net_return_pct=net_return_this_case,
                    r_multiple=r_multiple,
                    n=n,
                    win_rate_pct=round(win_rate_pct, 1),
                    win_rate_ci_low_pct=ci_low,
                    regime=case.regime,
                    regime_fallback=used_fallback,
                )
            )
        if significant:
            # Same rule as compute_candidates(): one winner per bucket, the
            # highest confidence-lower-bound label wins.
            events.append(max(significant, key=lambda e: e.win_rate_ci_low_pct))
    return events


def _count_independent_episodes(trades: list[Trade]) -> int:
    """Clusters trades sharing a direction and overlapping time windows into
    "episodes", so correlated simultaneous trades aren't counted as
    independent samples when judging statistical confidence. A deliberate
    simplification: true correlation also crosses directions/symbols via
    shared BTC exposure, which this per-direction clustering doesn't capture."""
    episodes = 0
    open_until: dict[str, datetime] = {}
    for trade in sorted(trades, key=lambda t: t.entry_time):
        current_end = open_until.get(trade.direction)
        if current_end is not None and trade.entry_time <= current_end:
            open_until[trade.direction] = max(current_end, trade.resolution_time)
        else:
            episodes += 1
            open_until[trade.direction] = trade.resolution_time
    return episodes


def _compute_metrics(
    trades: list[Trade],
    equity_curve: list[tuple[datetime, float]],
    config: PortfolioBacktestConfig,
    concurrency_samples: list[int],
) -> dict[str, Any]:
    if not trades:
        return {
            "n_trades": 0,
            "n_independent_episodes": 0,
            "net_profit": 0.0,
            "net_profit_pct": 0.0,
            "final_equity": config.initial_capital,
            "note": "No trades were significant enough to open across the requested universe/date range.",
        }

    returns_pct_of_equity = [t.pnl / t.equity_before * 100.0 for t in trades]
    pnl_values = [t.pnl for t in trades]
    n = len(trades)

    gross_profit = sum(p for p in pnl_values if p > 0)
    gross_loss = sum(p for p in pnl_values if p < 0)
    if gross_loss != 0:
        profit_factor: float | None = round(gross_profit / abs(gross_loss), 2)
    else:
        profit_factor = None if gross_profit == 0 else float("inf")

    equity_values = [e for _, e in equity_curve]
    final_equity = equity_values[-1]
    net_profit = final_equity - config.initial_capital
    net_profit_pct = net_profit / config.initial_capital * 100.0

    start_time = trades[0].entry_time
    end_time = trades[-1].resolution_time
    years_span = max((end_time - start_time).total_seconds() / 86400.0 / 365.25, 1e-9)
    cagr_pct = (
        ((final_equity / config.initial_capital) ** (1.0 / years_span) - 1.0) * 100.0
        if final_equity > 0
        else None
    )

    mean_return = statistics.mean(returns_pct_of_equity)
    stdev_return = statistics.pstdev(returns_pct_of_equity) if n > 1 else 0.0
    trades_per_year = n / years_span
    sharpe_ratio = (mean_return / stdev_return * math.sqrt(trades_per_year)) if stdev_return > 0 else None

    downside_returns = [r for r in returns_pct_of_equity if r < 0]
    downside_dev = statistics.pstdev(downside_returns) if len(downside_returns) > 1 else (
        abs(downside_returns[0]) if downside_returns else 0.0
    )
    sortino_ratio = (mean_return / downside_dev * math.sqrt(trades_per_year)) if downside_dev > 0 else None

    peak = equity_values[0]
    max_drawdown_pct = 0.0
    for value in equity_values:
        peak = max(peak, value)
        if peak > 0:
            max_drawdown_pct = max(max_drawdown_pct, (peak - value) / peak * 100.0)
    calmar_ratio = (cagr_pct / max_drawdown_pct) if (cagr_pct is not None and max_drawdown_pct > 0) else None

    r_multiples = [t.r_multiple for t in trades if t.r_multiple is not None]
    expectancy_r = statistics.mean(r_multiples) if r_multiples else None

    avg_concurrency = statistics.mean(concurrency_samples) if concurrency_samples else 0.0
    exposure_utilization = (
        avg_concurrency / config.max_concurrent_positions if config.max_concurrent_positions > 0 else None
    )
    exposure_adjusted_return_pct = (
        net_profit_pct / exposure_utilization if exposure_utilization else None
    )

    total_cost_pct_points = n * (config.round_trip_cost_pct + config.extra_slippage_pct)

    return {
        "n_trades": n,
        "n_independent_episodes": _count_independent_episodes(trades),
        "win_rate_pct": round(sum(1 for r in returns_pct_of_equity if r > 0) / n * 100.0, 1),
        "net_profit": round(net_profit, 2),
        "net_profit_pct": round(net_profit_pct, 2),
        "final_equity": round(final_equity, 2),
        "years_span": round(years_span, 3),
        "cagr_pct": round(cagr_pct, 2) if cagr_pct is not None else None,
        "sharpe_ratio": round(sharpe_ratio, 2) if sharpe_ratio is not None else None,
        "sortino_ratio": round(sortino_ratio, 2) if sortino_ratio is not None else None,
        "profit_factor": profit_factor,
        "max_drawdown_pct": round(max_drawdown_pct, 2),
        "calmar_ratio": round(calmar_ratio, 2) if calmar_ratio is not None else None,
        "expectancy_pct_of_equity": round(mean_return, 3),
        "expectancy_r_multiple": round(expectancy_r, 3) if expectancy_r is not None else None,
        "avg_concurrent_positions": round(avg_concurrency, 2),
        "exposure_adjusted_return_pct": (
            round(exposure_adjusted_return_pct, 2) if exposure_adjusted_return_pct is not None else None
        ),
        "trades_per_year": round(trades_per_year, 1),
        "total_cost_pct_points": round(total_cost_pct_points, 2),
    }


def _breakdown_by(trades: list[Trade], key: Any) -> list[dict[str, Any]]:
    """Groups ``trades`` by ``key(trade)`` and reports n / win rate / average
    R-multiple per group, sorted by group key -- shared helper behind the
    dashboard report's "by timeframe" / "by direction" / "by label" tables."""
    groups: dict[str, list[Trade]] = defaultdict(list)
    for trade in trades:
        groups[key(trade)].append(trade)

    rows = []
    for group_key, group_trades in sorted(groups.items()):
        r_multiples = [t.r_multiple for t in group_trades if t.r_multiple is not None]
        wins = sum(1 for t in group_trades if t.net_return_pct > 0)
        rows.append(
            {
                "key": group_key,
                "n": len(group_trades),
                "win_rate_pct": round(wins / len(group_trades) * 100.0, 1),
                "avg_r_multiple": round(sum(r_multiples) / len(r_multiples), 3) if r_multiples else None,
                "net_profit": round(sum(t.pnl for t in group_trades), 2),
            }
        )
    return rows


def _build_statistical_validation(
    result: PortfolioBacktestResult,
    n_trials: int,
    trial_sharpe_ratios: list[float] | None = None,
) -> dict[str, Any] | None:
    """Phase 8 Korak 3: Deflated Sharpe Ratio for this report's trade
    returns, honestly correcting the plain Sharpe ratio for ``n_trials``
    (how many backtest attempts preceded this one) and, when
    ``trial_sharpe_ratios`` is supplied (the per-trade Sharpe ratio of
    every prior recorded trial in the same family, from
    ``scanner.validation.experiment_registry`` -- see the CLI entry point
    below), for how spread out those prior attempts' Sharpe ratios actually
    were. Without ``trial_sharpe_ratios`` the deflation benchmark is 0 (no
    multiple-testing inflation applied, since there is nothing to estimate
    variance from) even if ``n_trials`` is large. Returns ``None`` if there
    are too few trades for a Sharpe ratio to be meaningful at all (fewer
    than 2, or all trades had the exact same return), rather than reporting
    a misleading or undefined statistic."""
    returns = [t.net_return_pct for t in result.trades]
    if len(returns) < 2:
        return None
    try:
        dsr = deflated_sharpe_ratio(returns, n_trials=n_trials, trial_sharpe_ratios=trial_sharpe_ratios)
    except ValueError:
        return None
    return {
        "observed_sharpe": round(dsr.observed_sharpe, 4),
        "deflated_sharpe_benchmark": round(dsr.deflated_sharpe_benchmark, 4),
        "probabilistic_sharpe_ratio": round(dsr.probabilistic_sharpe_ratio, 4),
        "likely_skill": dsr.likely_skill,
        "n_trials": dsr.n_trials,
        "n_observations": dsr.n_observations,
        "skewness": round(dsr.skewness, 3),
        "kurtosis": round(dsr.kurtosis, 3),
    }


def build_dashboard_report(
    config: PortfolioBacktestConfig,
    result: PortfolioBacktestResult,
    *,
    n_trials: int = 1,
    trial_sharpe_ratios: list[float] | None = None,
) -> dict[str, Any]:
    """Assembles the full JSON payload the dashboard's "🧪 Backtest" tab
    reads (see ``webapp/main.py``'s ``/api/backtest/report`` endpoint) --
    metrics plus breakdowns, a downsampled equity curve, and a Deflated
    Sharpe Ratio statistical-validation block, with enough metadata (when
    it ran, with what config) that a stale/cached report is never mistaken
    for a live result.

    ``n_trials`` should be the honest count of how many backtest attempts
    (across this and prior runs of the same strategy family) preceded this
    one, and ``trial_sharpe_ratios`` the per-trade Sharpe ratio of every one
    of those prior trials (this one included) -- pass both from
    ``scanner.validation.experiment_registry.count_prior_trials()``/
    ``fetch_experiments()`` when calling this from a context that also
    records the experiment; the CLI entry point below does this
    automatically via ``--experiment-family``. Both default to values that
    apply no multiple-testing deflation at all, for callers with no
    registry to consult.
    """
    # Downsample the equity curve for the frontend chart -- a multi-hundred
    # trade run doesn't need a point per trade to show the shape of the curve.
    max_points = 200
    curve = result.equity_curve
    step = max(1, len(curve) // max_points)
    equity_curve = [{"time": t.isoformat(), "equity": round(e, 2)} for t, e in curve[::step]]
    if curve and equity_curve and equity_curve[-1]["time"] != curve[-1][0].isoformat():
        equity_curve.append({"time": curve[-1][0].isoformat(), "equity": round(curve[-1][1], 2)})

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "config": {
            "symbols": list(config.symbols),
            "timeframes": list(config.timeframes),
            "horizon_candles": config.horizon_candles,
            "kline_limit": config.kline_limit,
            "warmup_candles": config.warmup_candles,
            "initial_capital": config.initial_capital,
            "risk_per_trade_pct": config.risk_per_trade_pct,
            "max_concurrent_positions": config.max_concurrent_positions,
            "max_concurrent_per_direction": config.max_concurrent_per_direction,
            "round_trip_cost_pct": config.round_trip_cost_pct,
            "extra_slippage_pct": config.extra_slippage_pct,
            "use_symbol_cost_tiers": config.use_symbol_cost_tiers,
            "include_funding_cost": config.include_funding_cost,
            "sl_tp_tie_break": config.sl_tp_tie_break,
            "include_oos_validation": config.include_oos_validation,
            "oos_test_fraction": config.oos_test_fraction,
            "oos_min_test_n": config.oos_min_test_n,
            "include_independence_validation": config.include_independence_validation,
            "independence_min_n": config.independence_min_n,
            "point_in_time_universe_cache_path": config.point_in_time_universe_cache_path,
        },
        "metrics": result.metrics,
        "skipped_symbols": result.skipped_symbols,
        "breakdown": {
            "by_timeframe": _breakdown_by(result.trades, lambda t: t.timeframe),
            "by_direction": _breakdown_by(result.trades, lambda t: t.direction),
            "by_label": _breakdown_by(result.trades, lambda t: t.label),
        },
        "equity_curve": equity_curve,
        "statistical_validation": _build_statistical_validation(result, n_trials, trial_sharpe_ratios),
        "oos_validation": [
            {
                "label": r.label,
                "direction": r.direction,
                "horizon_candles": r.horizon_candles,
                "train_n": r.train_n,
                "train_win_rate_pct": r.train_win_rate_pct,
                "test_n": r.test_n,
                "test_win_rate_pct": r.test_win_rate_pct,
                "test_win_rate_ci_low_pct": r.test_win_rate_ci_low_pct,
                "holds_up": r.holds_up,
                "warning": r.warning,
            }
            for r in result.oos_validation
        ],
        "independence_validation": [
            {
                "label": r.label,
                "direction": r.direction,
                "horizon_candles": r.horizon_candles,
                "n_raw": r.n_raw,
                "win_rate_raw_pct": r.win_rate_raw_pct,
                "win_rate_raw_ci_low_pct": r.win_rate_raw_ci_low_pct,
                "n_independent_events": r.n_independent_events,
                "win_rate_independent_pct": r.win_rate_independent_pct,
                "win_rate_independent_ci_low_pct": r.win_rate_independent_ci_low_pct,
                "holds_up_independent": r.holds_up_independent,
                "warning": r.warning,
            }
            for r in result.independence_validation
        ],
        "point_in_time_universe": (
            {
                "enabled": True,
                "cache_path": config.point_in_time_universe_cache_path,
                "trades_excluded": result.point_in_time_filtered_count,
            }
            if config.point_in_time_windows is not None
            else {"enabled": False}
        ),
    }


def run_portfolio_backtest(config: PortfolioBacktestConfig) -> PortfolioBacktestResult:
    """Fetches history for every (symbol, timeframe) pair in ``config``,
    builds each pair's own no-look-ahead candidate-event stream, merges them
    into one global chronological timeline, and simulates a single portfolio
    trading them subject to concurrency/direction caps and fixed-fractional
    position sizing."""
    all_events: list[CandidateEvent] = []
    all_cases: list[LabeledCase] = []
    skipped: list[str] = []

    for symbol in config.symbols:
        funding_events: list[FundingEvent] | None = None
        if config.include_funding_cost:
            funding_events = fetch_funding_history(symbol)
        cost_pct = (
            round_trip_cost_pct_for(symbol, config.round_trip_cost_pct)
            if config.use_symbol_cost_tiers
            else config.round_trip_cost_pct
        )
        for timeframe in config.timeframes:
            try:
                if config.warmup_candles > 0:
                    history = load_extended_history(symbol, timeframe, config.warmup_candles + config.kline_limit)
                    # The last config.kline_limit candles are "tradable"; everything
                    # before that is warm-up-only (feeds the stats accumulator, see
                    # _build_candidate_events, but never emits a CandidateEvent).
                    boundary_idx = max(0, len(history) - config.kline_limit)
                    decision_start_time = history["open_time"].iloc[boundary_idx] if boundary_idx < len(history) else None
                    kwargs: dict[str, Any] = {"history": history}
                    if config.include_funding_cost:
                        kwargs["funding_events"] = funding_events
                    if config.sl_tp_tie_break != "conservative":
                        kwargs["sl_tp_tie_break"] = config.sl_tp_tie_break
                    cases = backtest_symbol(symbol, timeframe, horizons=(config.horizon_candles,), **kwargs)
                else:
                    decision_start_time = None
                    kwargs = {"limit": config.kline_limit}
                    if config.include_funding_cost:
                        kwargs["funding_events"] = funding_events
                    if config.sl_tp_tie_break != "conservative":
                        kwargs["sl_tp_tie_break"] = config.sl_tp_tie_break
                    cases = backtest_symbol(symbol, timeframe, horizons=(config.horizon_candles,), **kwargs)
            except VendorError:
                skipped.append(f"{symbol}:{timeframe}")
                continue
            all_cases.extend(cases)
            min_n = BEST_SIGNAL_MIN_SAMPLE_SIZE.get(timeframe, _DEFAULT_MIN_SAMPLE_SIZE)
            all_events.extend(
                _build_candidate_events(
                    symbol,
                    timeframe,
                    cases,
                    min_n,
                    cost_pct,
                    config.extra_slippage_pct,
                    config.horizon_candles,
                    decision_start_time,
                )
            )

    all_events.sort(key=lambda e: e.entry_time)

    point_in_time_filtered_count = 0
    if config.point_in_time_windows is not None:
        kept_events = [
            e for e in all_events if is_symbol_in_universe_at(config.point_in_time_windows, e.symbol, e.entry_time)
        ]
        point_in_time_filtered_count = len(all_events) - len(kept_events)
        all_events = kept_events

    equity = config.initial_capital
    open_positions: list[dict[str, Any]] = []
    trades: list[Trade] = []
    equity_curve: list[tuple[datetime, float]] = []
    concurrency_samples: list[int] = []

    def _close(position: dict[str, Any]) -> None:
        nonlocal equity
        candidate: CandidateEvent = position["candidate"]
        equity_at_entry = position["equity_at_entry"]
        risk_amount = equity_at_entry * (config.risk_per_trade_pct / 100.0)
        if candidate.r_multiple is not None:
            pnl = risk_amount * candidate.r_multiple
        else:
            # No SL distance recorded for this case (shouldn't normally
            # happen for a directional label) -- fall back to applying the
            # raw net return directly against the entry-time equity.
            pnl = equity_at_entry * (candidate.net_return_pct / 100.0)
        equity_before = equity
        equity += pnl
        trades.append(
            Trade(
                symbol=candidate.symbol,
                timeframe=candidate.timeframe,
                direction=candidate.direction,
                label=candidate.label,
                entry_time=candidate.entry_time,
                resolution_time=candidate.resolution_time,
                net_return_pct=candidate.net_return_pct,
                r_multiple=candidate.r_multiple,
                equity_before=equity_before,
                pnl=pnl,
                equity_after=equity,
                win_rate_ci_low_pct=candidate.win_rate_ci_low_pct,
                n=candidate.n,
            )
        )
        equity_curve.append((candidate.resolution_time, equity))

    i = 0
    n_events = len(all_events)
    while i < n_events:
        current_time = all_events[i].entry_time
        batch: list[CandidateEvent] = []
        while i < n_events and all_events[i].entry_time == current_time:
            batch.append(all_events[i])
            i += 1
        batch.sort(key=lambda e: e.win_rate_ci_low_pct, reverse=True)

        # Close out anything whose outcome has resolved by now, oldest first.
        still_open = []
        for position in sorted(open_positions, key=lambda p: p["candidate"].resolution_time):
            if position["candidate"].resolution_time <= current_time:
                _close(position)
            else:
                still_open.append(position)
        open_positions = still_open

        for candidate in batch:
            if len(open_positions) >= config.max_concurrent_positions:
                break
            same_direction_open = sum(1 for p in open_positions if p["candidate"].direction == candidate.direction)
            if same_direction_open >= config.max_concurrent_per_direction:
                continue
            open_positions.append({"candidate": candidate, "equity_at_entry": equity})
            concurrency_samples.append(len(open_positions))

    for position in sorted(open_positions, key=lambda p: p["candidate"].resolution_time):
        _close(position)

    metrics = _compute_metrics(trades, equity_curve, config, concurrency_samples)
    oos_validation: list[OutOfSampleResult] = []
    if config.include_oos_validation and all_cases:
        oos_validation = validate_out_of_sample(
            all_cases,
            cost_pct=config.round_trip_cost_pct,
            test_fraction=config.oos_test_fraction,
            min_test_n=config.oos_min_test_n,
        )
        # Pooled summary rides along in `metrics` so it also lands "for free"
        # in the experiment registry's result_json (see record_experiment
        # below), not just in this run's own report.
        metrics["oos_labels_holding_up"] = sum(1 for r in oos_validation if r.holds_up)
        metrics["oos_labels_total"] = len(oos_validation)

    independence_validation: list[IndependenceClusterStats] = []
    if config.include_independence_validation and all_cases:
        independence_validation = summarize_with_independence(
            all_cases,
            cost_pct=config.round_trip_cost_pct,
            min_independent_n=config.independence_min_n,
        )
        # Same "rides along in metrics" pattern as oos_labels_holding_up
        # above, so a stale/old registry entry and a fresh one are both
        # comparable without re-opening the full report.
        metrics["independent_events_total"] = sum(r.n_independent_events for r in independence_validation)
        metrics["raw_cases_total_for_independence_check"] = sum(r.n_raw for r in independence_validation)
        metrics["labels_holding_up_independent"] = sum(
            1 for r in independence_validation if r.holds_up_independent
        )
        metrics["labels_total_independent_check"] = len(independence_validation)
    return PortfolioBacktestResult(
        trades=trades,
        equity_curve=equity_curve,
        metrics=metrics,
        skipped_symbols=skipped,
        point_in_time_filtered_count=point_in_time_filtered_count,
        oos_validation=oos_validation,
        independence_validation=independence_validation,
    )


def _main() -> None:
    parser = argparse.ArgumentParser(description="Phase 8 Korak 2 portfolio-level counterfactual backtest.")
    parser.add_argument("--symbols", nargs="+", required=True, help="Symbols to include, e.g. BTCUSDT ETHUSDT")
    parser.add_argument(
        "--timeframes", nargs="+", default=list(TRACKED_TIMEFRAMES), help="Timeframes to include (default: all tracked)"
    )
    parser.add_argument("--initial-capital", type=float, default=10_000.0)
    parser.add_argument("--risk-per-trade-pct", type=float, default=1.0)
    parser.add_argument("--max-concurrent-positions", type=int, default=5)
    parser.add_argument("--max-concurrent-per-direction", type=int, default=3)
    parser.add_argument("--kline-limit", type=int, default=1500)
    parser.add_argument(
        "--warmup-candles",
        type=int,
        default=1500,
        help="Extra candles fetched before the tradable window, to seed calibration stats and avoid cold-start "
        "bias (0 reproduces the old v1 behaviour)",
    )
    parser.add_argument("--extra-slippage-pct", type=float, default=0.0)
    parser.add_argument(
        "--no-symbol-cost-tiers",
        action="store_true",
        help="Phase 8 Korak 5: disable per-symbol liquidity cost tiers (see "
        "scanner.calibration.round_trip_cost_pct_for) and use --round-trip-cost-pct flat for every symbol, "
        "reproducing pre-Korak-5 behaviour.",
    )
    parser.add_argument(
        "--round-trip-cost-pct",
        type=float,
        default=DEFAULT_ROUND_TRIP_COST_PCT,
        help="Base round-trip cost (fees + assumed spread) in percentage points, applied flat unless "
        "per-symbol cost tiers are enabled (default), in which case it is this run's Tier-1 (majors) cost "
        "and other tiers add a surcharge on top.",
    )
    parser.add_argument(
        "--include-funding-cost",
        action="store_true",
        help="Phase 8 Korak 5: fetch each symbol's real Binance funding-rate settlement history and charge "
        "every trade the funding it would actually have accrued between entry and exit. Off by default "
        "because it requires one network call per symbol.",
    )
    parser.add_argument(
        "--sl-tp-tie-break",
        choices=["conservative", "optimistic"],
        default="conservative",
        help="Phase 8 Korak 5: how to resolve a candle whose range touches both SL and TP in the same bar. "
        "'conservative' (default) assumes the stop hit first, never overstating a label's edge. 'optimistic' "
        "assumes the target hit first -- run both and compare win rates to see how much of a label's apparent "
        "edge depends on this ambiguity.",
    )
    parser.add_argument(
        "--no-oos-validation",
        action="store_true",
        help="Phase 8 Korak 6: skip the in-sample/out-of-sample split check (scanner.calibration."
        "validate_out_of_sample). On by default -- it's pure re-computation on cases already fetched, no extra "
        "network calls.",
    )
    parser.add_argument(
        "--oos-test-fraction",
        type=float,
        default=OOS_TEST_FRACTION,
        help="Most-recent fraction of each (label, horizon)'s cases held out as the out-of-sample test window "
        f"(default {OOS_TEST_FRACTION}).",
    )
    parser.add_argument(
        "--oos-min-test-n",
        type=int,
        default=OOS_MIN_TEST_N,
        help=f"Minimum held-out test cases required before an OOS result is trusted (default {OOS_MIN_TEST_N}).",
    )
    parser.add_argument(
        "--no-independence-validation",
        action="store_true",
        help="Phase 8 Korak 6: skip the sample-independence correction (scanner.validation.independence) that "
        "re-runs the Wilson interval over clusters of time-overlapping cases (any symbol) instead of raw case "
        "count. On by default -- it's pure re-computation on cases already fetched, no extra network calls.",
    )
    parser.add_argument(
        "--independence-min-n",
        type=int,
        default=MIN_INDEPENDENT_EVENTS_N,
        help="Minimum independent-event clusters required before an independence-corrected result is trusted "
        f"(default {MIN_INDEPENDENT_EVENTS_N}).",
    )
    parser.add_argument(
        "--point-in-time-universe-cache",
        help="Phase 8 Korak 4: path to a cache produced by "
        "'python -m scanner.backtesting.historical_universe'. When given, any trade whose symbol was not in "
        "the point-in-time top-N universe at its own entry time is excluded, fixing survivorship bias from "
        "using today's top-N symbol list across the whole backtest period.",
    )
    parser.add_argument("--trades-json", help="Optional path to dump the full trade log as JSON")
    parser.add_argument(
        "--report-json",
        help="Optional path to write the dashboard-ready report (metrics + breakdowns + equity curve) as JSON, "
        "e.g. scanner_data/portfolio_backtest_report.json for the webapp's 'Backtest' tab to pick up",
    )
    parser.add_argument(
        "--experiment-family",
        help="Phase 8 Korak 3: if given, this run is counted as one more trial of this strategy family in "
        "the experiment registry (scanner_data/experiment_registry.db by default) BEFORE its results are "
        "used, so the Deflated Sharpe Ratio in --report-json is honestly corrected for every prior attempt, "
        "not just this one",
    )
    parser.add_argument(
        "--experiment-registry-path",
        default="scanner_data/experiment_registry.db",
        help="Path to the experiment registry SQLite database (default: scanner_data/experiment_registry.db)",
    )
    args = parser.parse_args()

    point_in_time_windows: tuple[UniverseWindow, ...] | None = None
    if args.point_in_time_universe_cache:
        point_in_time_windows = tuple(load_universe_cache(args.point_in_time_universe_cache))

    config = PortfolioBacktestConfig(
        symbols=tuple(args.symbols),
        timeframes=tuple(args.timeframes),
        initial_capital=args.initial_capital,
        risk_per_trade_pct=args.risk_per_trade_pct,
        max_concurrent_positions=args.max_concurrent_positions,
        max_concurrent_per_direction=args.max_concurrent_per_direction,
        kline_limit=args.kline_limit,
        warmup_candles=args.warmup_candles,
        round_trip_cost_pct=args.round_trip_cost_pct,
        extra_slippage_pct=args.extra_slippage_pct,
        use_symbol_cost_tiers=not args.no_symbol_cost_tiers,
        include_funding_cost=args.include_funding_cost,
        sl_tp_tie_break=args.sl_tp_tie_break,
        include_oos_validation=not args.no_oos_validation,
        oos_test_fraction=args.oos_test_fraction,
        oos_min_test_n=args.oos_min_test_n,
        include_independence_validation=not args.no_independence_validation,
        independence_min_n=args.independence_min_n,
        point_in_time_windows=point_in_time_windows,
        point_in_time_universe_cache_path=args.point_in_time_universe_cache,
    )
    result = run_portfolio_backtest(config)

    print(json.dumps(result.metrics, indent=2))
    if result.skipped_symbols:
        print(f"\nSkipped (vendor error): {', '.join(result.skipped_symbols)}")
    if config.point_in_time_windows is not None:
        print(f"Point-in-time universe filter excluded {result.point_in_time_filtered_count} candidate trade(s)")

    if args.trades_json:
        with open(args.trades_json, "w", encoding="utf-8") as fh:
            json.dump(
                [
                    {
                        "symbol": t.symbol,
                        "timeframe": t.timeframe,
                        "direction": t.direction,
                        "label": t.label,
                        "entry_time": t.entry_time.isoformat(),
                        "resolution_time": t.resolution_time.isoformat(),
                        "net_return_pct": t.net_return_pct,
                        "r_multiple": t.r_multiple,
                        "equity_before": t.equity_before,
                        "pnl": t.pnl,
                        "equity_after": t.equity_after,
                        "win_rate_ci_low_pct": t.win_rate_ci_low_pct,
                        "n": t.n,
                    }
                    for t in result.trades
                ],
                fh,
                indent=2,
            )
        print(f"Wrote {len(result.trades)} trades to {args.trades_json}")

    if args.report_json:
        n_trials = 1
        trial_sharpe_ratios: list[float] | None = None
        if args.experiment_family:
            conn = init_experiment_registry(args.experiment_registry_path)
            n_trials = count_prior_trials(conn, args.experiment_family) + 1
            notes = f"CLI run: {len(config.symbols)} symbols, --report-json {args.report_json}"
            if config.point_in_time_windows is not None:
                notes += (
                    f" (point-in-time universe filter active, excluded "
                    f"{result.point_in_time_filtered_count} candidate trade(s))"
                )
            record_experiment(
                conn,
                strategy_family=args.experiment_family,
                timeframe=",".join(config.timeframes),
                horizon_candles=config.horizon_candles,
                sl_tp_config={
                    "round_trip_cost_pct": config.round_trip_cost_pct,
                    "extra_slippage_pct": config.extra_slippage_pct,
                    "warmup_candles": config.warmup_candles,
                    "use_symbol_cost_tiers": config.use_symbol_cost_tiers,
                    "include_funding_cost": config.include_funding_cost,
                    "sl_tp_tie_break": config.sl_tp_tie_break,
                    "include_oos_validation": config.include_oos_validation,
                    "oos_test_fraction": config.oos_test_fraction,
                    "oos_min_test_n": config.oos_min_test_n,
                    "include_independence_validation": config.include_independence_validation,
                    "independence_min_n": config.independence_min_n,
                },
                n_trades=result.metrics.get("n_trades"),
                sharpe_ratio=result.metrics.get("sharpe_ratio"),
                net_profit_pct=result.metrics.get("net_profit_pct"),
                result=result.metrics,
                notes=notes,
            )
            print(f"Recorded trial #{n_trials} of experiment family '{args.experiment_family}'")

            # Pull every prior (and this) trial's Sharpe ratio so the DSR
            # benchmark reflects how spread out this family's real attempts
            # have actually been, instead of applying no deflation at all.
            prior_experiments = fetch_experiments(conn, args.experiment_family, limit=10_000)
            sharpes = [e["sharpe_ratio"] for e in prior_experiments if e["sharpe_ratio"] is not None]
            if len(sharpes) >= 2:
                trial_sharpe_ratios = sharpes

        report = build_dashboard_report(config, result, n_trials=n_trials, trial_sharpe_ratios=trial_sharpe_ratios)
        with open(args.report_json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"Wrote dashboard report to {args.report_json}")


if __name__ == "__main__":
    _main()
