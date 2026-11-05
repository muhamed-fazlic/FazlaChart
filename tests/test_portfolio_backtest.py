"""Unit tests for Phase 8 Korak 2's portfolio-level counterfactual backtest.

All tests build synthetic ``LabeledCase`` lists directly -- no network calls,
no reliance on real market history -- so they can precisely control what a
"no look-ahead" violation or a portfolio admission edge case would look
like, and assert the module gets it right.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from scanner.backtesting.historical_universe import UniverseWindow
from scanner.backtesting.portfolio_backtest import (
    PortfolioBacktestConfig,
    _build_candidate_events,
    _compute_metrics,
    _count_independent_episodes,
    _net_return_pct,
    run_portfolio_backtest,
)
from scanner.calibration import DEFAULT_ROUND_TRIP_COST_PCT, LabeledCase
from tradingagents.dataflows.errors import VendorError

pytestmark = pytest.mark.unit

_T0 = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)


def _case(
    label: str,
    hours_after_t0: int,
    forward_return_pct: float,
    regime: str | None = "normal",
    sl_distance_pct: float | None = 2.0,
    symbol: str = "BTCUSDT",
    timeframe: str = "1h",
) -> LabeledCase:
    entry_time = (_T0 + timedelta(hours=hours_after_t0)).isoformat()
    return LabeledCase(
        symbol=symbol,
        timeframe=timeframe,
        label=label,
        entry_time=entry_time,
        entry_close=100.0,
        forward_return_pct=forward_return_pct,
        horizon_candles=1,
        regime=regime,
        exit_reason="timeout",
        sl_distance_pct=sl_distance_pct,
    )


class TestNetReturnCorrection:
    def test_long_label_keeps_sign_minus_cost(self):
        case = _case("trend:bullish", 0, forward_return_pct=5.0)
        assert _net_return_pct(case, DEFAULT_ROUND_TRIP_COST_PCT, 0.0) == pytest.approx(5.0 - 0.10)

    def test_short_label_flips_sign_minus_cost(self):
        # Price dropped 5% -- a short should show this as a +5% gain, net of cost.
        case = _case("trend:bearish", 0, forward_return_pct=-5.0)
        assert _net_return_pct(case, DEFAULT_ROUND_TRIP_COST_PCT, 0.0) == pytest.approx(5.0 - 0.10)

    def test_extra_slippage_subtracted_too(self):
        case = _case("trend:bullish", 0, forward_return_pct=5.0)
        assert _net_return_pct(case, DEFAULT_ROUND_TRIP_COST_PCT, 0.25) == pytest.approx(5.0 - 0.10 - 0.25)


class TestNoLookAhead:
    def test_candidate_only_sees_already_resolved_cases(self):
        # 1h timeframe, horizon_candles=1 -> each case resolves exactly 1h
        # after its own entry. Five consecutive 100%-win "trend:bullish"
        # cases, one per hour. With min_n=4, a 100% win rate only clears the
        # Wilson 95% CI-low>50% gate once n>=4 (n=3's CI-low is ~44%, n=4's
        # is ~51%), so the candle at hour 4 is the first with any chance of
        # firing -- and it must do so using exactly the 4 PRIOR already-
        # resolved cases (hours 0-3), never its own not-yet-resolved outcome.
        cases = [_case("trend:bullish", h, forward_return_pct=5.0) for h in range(5)]
        events = _build_candidate_events(
            "BTCUSDT", "1h", cases, min_n=4, cost_pct=0.0, extra_slippage_pct=0.0, horizon_candles=1
        )
        assert len(events) == 1
        event = events[0]
        assert event.entry_time == _T0 + timedelta(hours=4)
        assert event.n == 4

    def test_insufficient_history_never_fires(self):
        cases = [_case("trend:bullish", h, forward_return_pct=5.0) for h in range(3)]
        events = _build_candidate_events(
            "BTCUSDT", "1h", cases, min_n=10, cost_pct=0.0, extra_slippage_pct=0.0, horizon_candles=1
        )
        assert events == []

    def test_low_win_rate_history_never_fires(self):
        # Every prior case lost money -- even once min_n is reached, the
        # Wilson CI-low must stay at/below 50% and suppress the candidate.
        cases = [_case("trend:bullish", h, forward_return_pct=-5.0) for h in range(10)]
        events = _build_candidate_events(
            "BTCUSDT", "1h", cases, min_n=2, cost_pct=0.0, extra_slippage_pct=0.0, horizon_candles=1
        )
        assert events == []

    def test_undirected_labels_never_produce_candidates(self):
        cases = [_case("trend:neutral", h, forward_return_pct=1.0) for h in range(10)]
        events = _build_candidate_events(
            "BTCUSDT", "1h", cases, min_n=2, cost_pct=0.0, extra_slippage_pct=0.0, horizon_candles=1
        )
        assert events == []


class TestWarmup:
    def test_decision_start_time_suppresses_pre_cutoff_events_but_still_feeds_stats(self):
        # Same 5-candle, 100%-win setup as test_candidate_only_sees_already_resolved_cases,
        # which (with min_n=4) would otherwise fire exactly once, at hour 4.
        cases = [_case("trend:bullish", h, forward_return_pct=5.0) for h in range(5)]
        events = _build_candidate_events(
            "BTCUSDT",
            "1h",
            cases,
            min_n=4,
            cost_pct=0.0,
            extra_slippage_pct=0.0,
            horizon_candles=1,
            decision_start_time=_T0 + timedelta(hours=4),
        )
        # The only candle that would have fired is AT the cutoff (<=), so it
        # must now be suppressed as "warm-up only", not emitted as a trade.
        assert events == []

    def test_decision_start_time_allows_events_after_cutoff(self):
        cases = [_case("trend:bullish", h, forward_return_pct=5.0) for h in range(6)]
        events = _build_candidate_events(
            "BTCUSDT",
            "1h",
            cases,
            min_n=4,
            cost_pct=0.0,
            extra_slippage_pct=0.0,
            horizon_candles=1,
            decision_start_time=_T0 + timedelta(hours=3),
        )
        # Hour 4 is after the cutoff and has 4 prior resolved cases (hours
        # 0-3) feeding the accumulator -- it should fire normally.
        assert any(e.entry_time == _T0 + timedelta(hours=4) for e in events)

    def test_none_decision_start_time_behaves_like_before(self):
        cases = [_case("trend:bullish", h, forward_return_pct=5.0) for h in range(5)]
        events = _build_candidate_events(
            "BTCUSDT", "1h", cases, min_n=4, cost_pct=0.0, extra_slippage_pct=0.0, horizon_candles=1
        )
        assert len(events) == 1
        assert events[0].entry_time == _T0 + timedelta(hours=4)


class TestRegimeFallback:
    def test_regime_bucket_used_even_below_min_n_no_fallback_to_blend(self):
        # One prior "high" regime case (a loss, so if it were used alone it
        # wouldn't even matter -- the real point is testing bucket SELECTION):
        # the blended bucket has plenty of winning "normal" regime history,
        # but since the "high" regime bucket has >=1 resolved case, it must
        # be preferred and (since its n=1 < min_n=2) suppress the candidate,
        # rather than silently falling back to the much larger blended n.
        cases = [
            _case("trend:bullish", 0, forward_return_pct=5.0, regime="high"),
        ] + [_case("trend:bullish", h, forward_return_pct=5.0, regime="normal") for h in range(1, 6)]
        # Add a later "high" regime candle to evaluate against the (still
        # too-small) regime-specific bucket.
        cases.append(_case("trend:bullish", 10, forward_return_pct=5.0, regime="high"))
        events = _build_candidate_events(
            "BTCUSDT", "1h", cases, min_n=2, cost_pct=0.0, extra_slippage_pct=0.0, horizon_candles=1
        )
        # The candle at hour 10 should NOT fire: its own regime ("high") only
        # has 1 prior resolved case, below min_n, and must not silently
        # borrow the well-populated "normal" blend instead.
        assert all(e.entry_time != _T0 + timedelta(hours=10) for e in events)

    def test_falls_back_to_blend_when_regime_bucket_empty(self):
        # No prior "high" regime cases exist at all -- the regime-specific
        # bucket is empty, so the blended bucket (all regimes) must be used.
        cases = [_case("trend:bullish", h, forward_return_pct=5.0, regime="normal") for h in range(5)]
        cases.append(_case("trend:bullish", 10, forward_return_pct=5.0, regime="high"))
        events = _build_candidate_events(
            "BTCUSDT", "1h", cases, min_n=2, cost_pct=0.0, extra_slippage_pct=0.0, horizon_candles=1
        )
        assert any(e.entry_time == _T0 + timedelta(hours=10) for e in events)


class TestPortfolioAdmission:
    def _config(self, **overrides) -> PortfolioBacktestConfig:
        base = {
            "symbols": ("BTCUSDT",),
            "timeframes": ("1h",),
            "initial_capital": 10_000.0,
            "risk_per_trade_pct": 1.0,
            "max_concurrent_positions": 2,
            "max_concurrent_per_direction": 1,
            "round_trip_cost_pct": 0.0,
            "extra_slippage_pct": 0.0,
            "warmup_candles": 0,
        }
        base.update(overrides)
        return PortfolioBacktestConfig(**base)

    def test_respects_max_concurrent_positions(self, monkeypatch):
        # Enough history (> BEST_SIGNAL_MIN_SAMPLE_SIZE["1h"] == 30) of a
        # 100%-win "trend:bullish" run that every symbol shares identically,
        # so all three symbols fire a significant long candidate at exactly
        # the same hours -- a genuine simultaneous-admission scenario -- but
        # cap the portfolio at 2 concurrent positions overall.
        warm_up = [_case("trend:bullish", h, forward_return_pct=5.0) for h in range(45)]
        from scanner.backtesting import portfolio_backtest as module

        def fake_backtest_symbol(symbol, timeframe, horizons, limit):
            return warm_up

        monkeypatch.setattr(module, "backtest_symbol", fake_backtest_symbol)
        config = self._config(
            symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT"), max_concurrent_positions=2, horizon_candles=1
        )
        result = run_portfolio_backtest(config)
        assert result.trades, "expected at least one trade to be admitted"
        # At no point should more positions have been admitted concurrently
        # than the cap -- approximate this by checking concurrency samples
        # recorded in the metrics never exceeded the cap.
        assert result.metrics["avg_concurrent_positions"] <= config.max_concurrent_positions

    def test_respects_max_concurrent_per_direction(self, monkeypatch):
        warm_up = [_case("trend:bullish", h, forward_return_pct=5.0) for h in range(45)]
        from scanner.backtesting import portfolio_backtest as module

        monkeypatch.setattr(module, "backtest_symbol", lambda symbol, timeframe, horizons, limit: warm_up)
        config = self._config(
            symbols=("BTCUSDT", "ETHUSDT"),
            max_concurrent_positions=5,
            max_concurrent_per_direction=1,
            horizon_candles=1,
        )
        result = run_portfolio_backtest(config)
        assert result.trades, "expected at least one trade to be admitted"
        # Every trade here is a "long" -- with a per-direction cap of 1, no
        # two should ever have been open on top of each other.
        opens_by_time = sorted(result.trades, key=lambda t: t.entry_time)
        for trade in opens_by_time:
            overlapping = [
                t
                for t in opens_by_time
                if t is not trade and t.entry_time <= trade.entry_time < t.resolution_time
            ]
            assert len(overlapping) == 0

    def test_warmup_candles_fetches_extended_history_and_passes_decision_start_time(self, monkeypatch):
        import pandas as pd

        from scanner.backtesting import portfolio_backtest as module

        # 10 warm-up candles + 5 "tradable" candles = 15 total; the boundary
        # between them is what decision_start_time should be computed from.
        open_times = pd.date_range(_T0, periods=15, freq="1h", tz="UTC")
        fake_history = pd.DataFrame({"open_time": open_times})

        captured: dict[str, object] = {}

        def fake_load_extended_history(symbol, timeframe, total_candles):
            captured["total_candles"] = total_candles
            return fake_history

        def fake_backtest_symbol(symbol, timeframe, horizons, history):
            captured["history_passed"] = history is fake_history
            return []

        monkeypatch.setattr(module, "load_extended_history", fake_load_extended_history)
        monkeypatch.setattr(module, "backtest_symbol", fake_backtest_symbol)
        config = self._config(kline_limit=5, warmup_candles=10)
        run_portfolio_backtest(config)

        assert captured["total_candles"] == 15
        assert captured["history_passed"] is True

    def test_vendor_error_is_skipped_not_raised(self, monkeypatch):
        from scanner.backtesting import portfolio_backtest as module

        def raise_vendor_error(symbol, timeframe, horizons, limit):
            raise VendorError("boom")

        monkeypatch.setattr(module, "backtest_symbol", raise_vendor_error)
        config = self._config(symbols=("BADUSDT",))
        result = run_portfolio_backtest(config)
        assert result.skipped_symbols == ["BADUSDT:1h"]
        assert result.trades == []


class TestPointInTimeUniverseFilter:
    """Phase 8 Korak 4: a trade must only count if its symbol was actually
    in the point-in-time universe at its own entry_time."""

    def _config(self, **overrides) -> PortfolioBacktestConfig:
        base = {
            "symbols": ("BTCUSDT", "ETHUSDT"),
            "timeframes": ("1h",),
            "initial_capital": 10_000.0,
            "risk_per_trade_pct": 1.0,
            "max_concurrent_positions": 5,
            "max_concurrent_per_direction": 5,
            "round_trip_cost_pct": 0.0,
            "extra_slippage_pct": 0.0,
            "warmup_candles": 0,
            "horizon_candles": 1,
        }
        base.update(overrides)
        return PortfolioBacktestConfig(**base)

    def test_no_windows_configured_keeps_every_symbol(self, monkeypatch):
        warm_up = [_case("trend:bullish", h, forward_return_pct=5.0) for h in range(45)]
        from scanner.backtesting import portfolio_backtest as module

        monkeypatch.setattr(module, "backtest_symbol", lambda symbol, timeframe, horizons, limit: warm_up)
        result = run_portfolio_backtest(self._config())

        assert result.point_in_time_filtered_count == 0
        assert {t.symbol for t in result.trades} == {"BTCUSDT", "ETHUSDT"}

    def test_excludes_symbol_not_in_universe_at_entry_time(self, monkeypatch):
        warm_up = [_case("trend:bullish", h, forward_return_pct=5.0) for h in range(45)]
        from scanner.backtesting import portfolio_backtest as module

        monkeypatch.setattr(module, "backtest_symbol", lambda symbol, timeframe, horizons, limit: warm_up)
        window = UniverseWindow(
            start=_T0 - timedelta(days=1), end=_T0 + timedelta(days=10), symbols=frozenset({"BTCUSDT"})
        )
        result = run_portfolio_backtest(self._config(point_in_time_windows=(window,)))

        assert result.trades, "expected at least one trade to be admitted"
        assert {t.symbol for t in result.trades} == {"BTCUSDT"}
        assert result.point_in_time_filtered_count > 0

    def test_timestamp_outside_every_window_excludes_everything(self, monkeypatch):
        warm_up = [_case("trend:bullish", h, forward_return_pct=5.0) for h in range(45)]
        from scanner.backtesting import portfolio_backtest as module

        monkeypatch.setattr(module, "backtest_symbol", lambda symbol, timeframe, horizons, limit: warm_up)
        # A window far in the future -- none of the trades (all near _T0)
        # fall inside it, so every candidate should be dropped.
        window = UniverseWindow(
            start=_T0 + timedelta(days=365), end=_T0 + timedelta(days=400), symbols=frozenset({"BTCUSDT", "ETHUSDT"})
        )
        result = run_portfolio_backtest(self._config(point_in_time_windows=(window,)))

        assert result.trades == []
        assert result.point_in_time_filtered_count > 0


class TestCostTierAndFundingWiring:
    """Phase 8 Korak 5: per-symbol liquidity cost tiers and (opt-in) funding
    cost, both wired into `run_portfolio_backtest`'s main loop."""

    def _config(self, **overrides) -> PortfolioBacktestConfig:
        base = {
            "symbols": ("BTCUSDT",),
            "timeframes": ("1h",),
            "initial_capital": 10_000.0,
            "risk_per_trade_pct": 1.0,
            "max_concurrent_positions": 5,
            "max_concurrent_per_direction": 5,
            "round_trip_cost_pct": 0.0,
            "extra_slippage_pct": 0.0,
            "warmup_candles": 0,
            "horizon_candles": 1,
        }
        base.update(overrides)
        return PortfolioBacktestConfig(**base)

    def test_tier1_symbol_pays_no_surcharge_by_default(self, monkeypatch):
        warm_up = [_case("trend:bullish", h, forward_return_pct=5.0, symbol="BTCUSDT") for h in range(45)]
        from scanner.backtesting import portfolio_backtest as module

        monkeypatch.setattr(module, "backtest_symbol", lambda symbol, timeframe, horizons, limit: warm_up)
        result = run_portfolio_backtest(self._config(symbols=("BTCUSDT",)))
        assert result.trades
        assert all(t.net_return_pct == pytest.approx(5.0) for t in result.trades)

    def test_tier3_symbol_pays_surcharge_by_default(self, monkeypatch):
        warm_up = [_case("trend:bullish", h, forward_return_pct=5.0, symbol="SOMEALTUSDT") for h in range(45)]
        from scanner.backtesting import portfolio_backtest as module

        monkeypatch.setattr(module, "backtest_symbol", lambda symbol, timeframe, horizons, limit: warm_up)
        result = run_portfolio_backtest(self._config(symbols=("SOMEALTUSDT",)))
        assert result.trades
        # Tier-3 surcharge is 0.05 pp on top of the 0.0 base configured here.
        assert all(t.net_return_pct == pytest.approx(5.0 - 0.05) for t in result.trades)

    def test_use_symbol_cost_tiers_false_reverts_to_flat_cost(self, monkeypatch):
        warm_up = [_case("trend:bullish", h, forward_return_pct=5.0, symbol="SOMEALTUSDT") for h in range(45)]
        from scanner.backtesting import portfolio_backtest as module

        monkeypatch.setattr(module, "backtest_symbol", lambda symbol, timeframe, horizons, limit: warm_up)
        result = run_portfolio_backtest(
            self._config(symbols=("SOMEALTUSDT",), use_symbol_cost_tiers=False)
        )
        assert result.trades
        assert all(t.net_return_pct == pytest.approx(5.0) for t in result.trades)

    def test_include_funding_cost_false_never_calls_fetch_or_passes_kwarg(self, monkeypatch):
        """Default (off) behaviour must stay compatible with old-style
        monkeypatched `backtest_symbol` lambdas that don't accept a
        `funding_events` kwarg at all -- this would raise a TypeError if
        the wiring always passed it unconditionally."""
        warm_up = [_case("trend:bullish", h, forward_return_pct=5.0) for h in range(45)]
        from scanner.backtesting import portfolio_backtest as module

        fetch_called = []
        monkeypatch.setattr(
            module, "fetch_funding_history", lambda symbol, limit=1000: fetch_called.append(symbol) or []
        )
        monkeypatch.setattr(module, "backtest_symbol", lambda symbol, timeframe, horizons, limit: warm_up)
        result = run_portfolio_backtest(self._config())

        assert result.trades
        assert fetch_called == []

    def test_include_funding_cost_true_fetches_once_per_symbol_and_forwards_it(self, monkeypatch):
        warm_up = [_case("trend:bullish", h, forward_return_pct=5.0) for h in range(45)]
        from scanner.backtesting import portfolio_backtest as module
        from scanner.backtesting.funding_simulator import FundingEvent

        fetch_calls = []
        sentinel_events = [FundingEvent(timestamp=_T0, rate=0.0001)]

        def fake_fetch(symbol, limit=1000):
            fetch_calls.append(symbol)
            return sentinel_events

        received_funding_events = []

        def fake_backtest_symbol(symbol, timeframe, horizons, limit, funding_events=None):
            received_funding_events.append(funding_events)
            return warm_up

        monkeypatch.setattr(module, "fetch_funding_history", fake_fetch)
        monkeypatch.setattr(module, "backtest_symbol", fake_backtest_symbol)
        config = self._config(timeframes=("1h", "4h"), include_funding_cost=True)
        run_portfolio_backtest(config)

        # One fetch per symbol, cached/reused across every timeframe -- not
        # one fetch per (symbol, timeframe) pair.
        assert fetch_calls == ["BTCUSDT"]
        assert received_funding_events == [sentinel_events, sentinel_events]



    def test_known_trades_produce_expected_net_profit_and_win_rate(self):
        from scanner.backtesting.portfolio_backtest import Trade

        t1_entry = _T0
        t1_res = _T0 + timedelta(hours=1)
        t2_entry = _T0 + timedelta(hours=1)
        t2_res = _T0 + timedelta(hours=2)
        trades = [
            Trade(
                symbol="BTCUSDT",
                timeframe="1h",
                direction="long",
                label="trend:bullish",
                entry_time=t1_entry,
                resolution_time=t1_res,
                net_return_pct=4.0,
                r_multiple=2.0,
                equity_before=10_000.0,
                pnl=200.0,
                equity_after=10_200.0,
                win_rate_ci_low_pct=55.0,
                n=50,
            ),
            Trade(
                symbol="BTCUSDT",
                timeframe="1h",
                direction="short",
                label="trend:bearish",
                entry_time=t2_entry,
                resolution_time=t2_res,
                net_return_pct=-2.0,
                r_multiple=-1.0,
                equity_before=10_200.0,
                pnl=-102.0,
                equity_after=10_098.0,
                win_rate_ci_low_pct=55.0,
                n=50,
            ),
        ]
        equity_curve = [(t1_res, 10_200.0), (t2_res, 10_098.0)]
        config = PortfolioBacktestConfig(symbols=("BTCUSDT",), initial_capital=10_000.0)
        metrics = _compute_metrics(trades, equity_curve, config, concurrency_samples=[1, 1])

        assert metrics["n_trades"] == 2
        assert metrics["final_equity"] == pytest.approx(10_098.0)
        assert metrics["net_profit"] == pytest.approx(98.0)
        assert metrics["net_profit_pct"] == pytest.approx(0.98)
        assert metrics["win_rate_pct"] == pytest.approx(50.0)
        assert metrics["expectancy_r_multiple"] == pytest.approx(0.5)

    def test_no_trades_returns_zeroed_summary(self):
        config = PortfolioBacktestConfig(symbols=("BTCUSDT",))
        metrics = _compute_metrics([], [], config, concurrency_samples=[])
        assert metrics["n_trades"] == 0
        assert metrics["final_equity"] == config.initial_capital


class TestDashboardReport:
    def test_report_shape_and_breakdowns(self):
        from scanner.backtesting.portfolio_backtest import (
            PortfolioBacktestResult,
            Trade,
            build_dashboard_report,
        )

        t1_entry = _T0
        t1_res = _T0 + timedelta(hours=1)
        t2_entry = _T0 + timedelta(hours=1)
        t2_res = _T0 + timedelta(hours=2)
        trades = [
            Trade(
                symbol="BTCUSDT",
                timeframe="1h",
                direction="long",
                label="trend:bullish",
                entry_time=t1_entry,
                resolution_time=t1_res,
                net_return_pct=4.0,
                r_multiple=2.0,
                equity_before=10_000.0,
                pnl=200.0,
                equity_after=10_200.0,
                win_rate_ci_low_pct=55.0,
                n=50,
            ),
            Trade(
                symbol="ETHUSDT",
                timeframe="4h",
                direction="short",
                label="trend:bearish",
                entry_time=t2_entry,
                resolution_time=t2_res,
                net_return_pct=-2.0,
                r_multiple=-1.0,
                equity_before=10_200.0,
                pnl=-102.0,
                equity_after=10_098.0,
                win_rate_ci_low_pct=55.0,
                n=50,
            ),
        ]
        equity_curve = [(t1_res, 10_200.0), (t2_res, 10_098.0)]
        config = PortfolioBacktestConfig(symbols=("BTCUSDT", "ETHUSDT"), initial_capital=10_000.0)
        metrics = _compute_metrics(trades, equity_curve, config, concurrency_samples=[1, 1])
        result = PortfolioBacktestResult(
            trades=trades, equity_curve=equity_curve, metrics=metrics, skipped_symbols=["BADUSDT"]
        )

        report = build_dashboard_report(config, result)

        assert "generated_at" in report
        assert report["config"]["symbols"] == ["BTCUSDT", "ETHUSDT"]
        assert report["metrics"]["n_trades"] == 2
        assert report["skipped_symbols"] == ["BADUSDT"]

        by_tf = {row["key"]: row for row in report["breakdown"]["by_timeframe"]}
        assert by_tf["1h"]["n"] == 1
        assert by_tf["4h"]["n"] == 1
        by_direction = {row["key"]: row for row in report["breakdown"]["by_direction"]}
        assert by_direction["long"]["avg_r_multiple"] == pytest.approx(2.0)
        assert by_direction["short"]["avg_r_multiple"] == pytest.approx(-1.0)

        assert len(report["equity_curve"]) == 2
        assert report["equity_curve"][-1]["equity"] == pytest.approx(10_098.0)

        assert report["statistical_validation"] is not None
        assert report["statistical_validation"]["n_observations"] == 2
        assert report["statistical_validation"]["n_trials"] == 1
        assert report["statistical_validation"]["deflated_sharpe_benchmark"] == 0.0

        assert report["point_in_time_universe"] == {"enabled": False}

    def test_report_shows_point_in_time_universe_when_enabled(self):
        from scanner.backtesting.historical_universe import UniverseWindow
        from scanner.backtesting.portfolio_backtest import (
            PortfolioBacktestResult,
            build_dashboard_report,
        )

        window = UniverseWindow(start=_T0, end=_T0 + timedelta(days=1), symbols=frozenset({"BTCUSDT"}))
        config = PortfolioBacktestConfig(
            symbols=("BTCUSDT",),
            point_in_time_windows=(window,),
            point_in_time_universe_cache_path="scanner_data/point_in_time_universe.json",
        )
        metrics = _compute_metrics([], [], config, concurrency_samples=[])
        result = PortfolioBacktestResult(trades=[], equity_curve=[], metrics=metrics, point_in_time_filtered_count=3)

        report = build_dashboard_report(config, result)

        assert report["point_in_time_universe"] == {
            "enabled": True,
            "cache_path": "scanner_data/point_in_time_universe.json",
            "trades_excluded": 3,
        }

    def test_empty_trades_produce_empty_breakdowns_and_curve(self):
        from scanner.backtesting.portfolio_backtest import (
            PortfolioBacktestResult,
            build_dashboard_report,
        )

        config = PortfolioBacktestConfig(symbols=("BTCUSDT",))
        metrics = _compute_metrics([], [], config, concurrency_samples=[])
        result = PortfolioBacktestResult(trades=[], equity_curve=[], metrics=metrics)

        report = build_dashboard_report(config, result)

        assert report["breakdown"]["by_timeframe"] == []
        assert report["equity_curve"] == []
        assert report["statistical_validation"] is None

    def test_n_trials_is_forwarded_into_statistical_validation(self):
        from scanner.backtesting.portfolio_backtest import (
            PortfolioBacktestResult,
            Trade,
            build_dashboard_report,
        )

        trades = [
            Trade(
                symbol="BTCUSDT",
                timeframe="1h",
                direction="long",
                label="trend:bullish",
                entry_time=_T0,
                resolution_time=_T0 + timedelta(hours=i + 1),
                net_return_pct=1.0 if i % 2 == 0 else -0.5,
                r_multiple=1.0 if i % 2 == 0 else -0.5,
                equity_before=10_000.0,
                pnl=100.0 if i % 2 == 0 else -50.0,
                equity_after=10_000.0,
                win_rate_ci_low_pct=55.0,
                n=50,
            )
            for i in range(10)
        ]
        config = PortfolioBacktestConfig(symbols=("BTCUSDT",))
        metrics = _compute_metrics(trades, [(_T0, 10_000.0)], config, concurrency_samples=[1])
        result = PortfolioBacktestResult(trades=trades, equity_curve=[(_T0, 10_000.0)], metrics=metrics)

        report = build_dashboard_report(config, result, n_trials=25)

        assert report["statistical_validation"]["n_trials"] == 25
        # No trial_sharpe_ratios/variance is available at this call site, so
        # the deflation benchmark stays 0 regardless of n_trials -- n_trials
        # is still recorded faithfully for transparency, even though it
        # currently has no numeric effect without a variance estimate (see
        # deflated_sharpe_ratio's docstring).
        assert report["statistical_validation"]["deflated_sharpe_benchmark"] == 0.0


class TestIndependentEpisodes:
    def test_overlapping_same_direction_trades_count_as_one_episode(self):
        from scanner.backtesting.portfolio_backtest import Trade

        def trade(entry_h, res_h, direction="long"):
            return Trade(
                symbol="BTCUSDT",
                timeframe="1h",
                direction=direction,
                label="trend:bullish",
                entry_time=_T0 + timedelta(hours=entry_h),
                resolution_time=_T0 + timedelta(hours=res_h),
                net_return_pct=1.0,
                r_multiple=0.5,
                equity_before=10_000.0,
                pnl=50.0,
                equity_after=10_050.0,
                win_rate_ci_low_pct=55.0,
                n=50,
            )

        overlapping = [trade(0, 3), trade(1, 4), trade(2, 5)]
        assert _count_independent_episodes(overlapping) == 1

        non_overlapping = [trade(0, 1), trade(5, 6), trade(10, 11)]
        assert _count_independent_episodes(non_overlapping) == 3

        different_directions_same_window = [trade(0, 3, "long"), trade(1, 4, "short")]
        assert _count_independent_episodes(different_directions_same_window) == 2
