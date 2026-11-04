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
from scanner.calibration import (
    DEFAULT_ROUND_TRIP_COST_PCT,
    LABEL_DIRECTION,
    LabeledCase,
    backtest_symbol,
    load_extended_history,
    wilson_score_interval,
)
from scanner.signal_tracking import TIMEFRAME_MINUTES
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


@dataclass
class PortfolioBacktestResult:
    trades: list[Trade]
    equity_curve: list[tuple[datetime, float]]
    metrics: dict[str, Any]
    skipped_symbols: list[str] = field(default_factory=list)


def _parse_time(iso_timestamp: str) -> datetime:
    ts = datetime.fromisoformat(iso_timestamp)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return ts


def _net_return_pct(case: LabeledCase, cost_pct: float, extra_slippage_pct: float) -> float:
    """Direction-corrected, cost-adjusted net return for one case — mirrors
    ``scanner.calibration.summarize``'s correction, which is NOT baked into
    ``LabeledCase.forward_return_pct`` itself."""
    direction = LABEL_DIRECTION[case.label]
    signed = case.forward_return_pct if direction == "long" else -case.forward_return_pct
    return signed - cost_pct - extra_slippage_pct


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


def build_dashboard_report(config: PortfolioBacktestConfig, result: PortfolioBacktestResult) -> dict[str, Any]:
    """Assembles the full JSON payload the dashboard's "🧪 Backtest" tab
    reads (see ``webapp/main.py``'s ``/api/backtest/report`` endpoint) --
    metrics plus breakdowns and a downsampled equity curve, with enough
    metadata (when it ran, with what config) that a stale/cached report is
    never mistaken for a live result."""
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
        },
        "metrics": result.metrics,
        "skipped_symbols": result.skipped_symbols,
        "breakdown": {
            "by_timeframe": _breakdown_by(result.trades, lambda t: t.timeframe),
            "by_direction": _breakdown_by(result.trades, lambda t: t.direction),
            "by_label": _breakdown_by(result.trades, lambda t: t.label),
        },
        "equity_curve": equity_curve,
    }


def run_portfolio_backtest(config: PortfolioBacktestConfig) -> PortfolioBacktestResult:
    """Fetches history for every (symbol, timeframe) pair in ``config``,
    builds each pair's own no-look-ahead candidate-event stream, merges them
    into one global chronological timeline, and simulates a single portfolio
    trading them subject to concurrency/direction caps and fixed-fractional
    position sizing."""
    all_events: list[CandidateEvent] = []
    skipped: list[str] = []

    for symbol in config.symbols:
        for timeframe in config.timeframes:
            try:
                if config.warmup_candles > 0:
                    history = load_extended_history(symbol, timeframe, config.warmup_candles + config.kline_limit)
                    # The last config.kline_limit candles are "tradable"; everything
                    # before that is warm-up-only (feeds the stats accumulator, see
                    # _build_candidate_events, but never emits a CandidateEvent).
                    boundary_idx = max(0, len(history) - config.kline_limit)
                    decision_start_time = history["open_time"].iloc[boundary_idx] if boundary_idx < len(history) else None
                    cases = backtest_symbol(symbol, timeframe, horizons=(config.horizon_candles,), history=history)
                else:
                    decision_start_time = None
                    cases = backtest_symbol(
                        symbol, timeframe, horizons=(config.horizon_candles,), limit=config.kline_limit
                    )
            except VendorError:
                skipped.append(f"{symbol}:{timeframe}")
                continue
            min_n = BEST_SIGNAL_MIN_SAMPLE_SIZE.get(timeframe, _DEFAULT_MIN_SAMPLE_SIZE)
            all_events.extend(
                _build_candidate_events(
                    symbol,
                    timeframe,
                    cases,
                    min_n,
                    config.round_trip_cost_pct,
                    config.extra_slippage_pct,
                    config.horizon_candles,
                    decision_start_time,
                )
            )

    all_events.sort(key=lambda e: e.entry_time)

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
    return PortfolioBacktestResult(
        trades=trades, equity_curve=equity_curve, metrics=metrics, skipped_symbols=skipped
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
    parser.add_argument("--trades-json", help="Optional path to dump the full trade log as JSON")
    parser.add_argument(
        "--report-json",
        help="Optional path to write the dashboard-ready report (metrics + breakdowns + equity curve) as JSON, "
        "e.g. scanner_data/portfolio_backtest_report.json for the webapp's 'Backtest' tab to pick up",
    )
    args = parser.parse_args()

    config = PortfolioBacktestConfig(
        symbols=tuple(args.symbols),
        timeframes=tuple(args.timeframes),
        initial_capital=args.initial_capital,
        risk_per_trade_pct=args.risk_per_trade_pct,
        max_concurrent_positions=args.max_concurrent_positions,
        max_concurrent_per_direction=args.max_concurrent_per_direction,
        kline_limit=args.kline_limit,
        warmup_candles=args.warmup_candles,
        extra_slippage_pct=args.extra_slippage_pct,
    )
    result = run_portfolio_backtest(config)

    print(json.dumps(result.metrics, indent=2))
    if result.skipped_symbols:
        print(f"\nSkipped (vendor error): {', '.join(result.skipped_symbols)}")

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
        report = build_dashboard_report(config, result)
        with open(args.report_json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"Wrote dashboard report to {args.report_json}")


if __name__ == "__main__":
    _main()
