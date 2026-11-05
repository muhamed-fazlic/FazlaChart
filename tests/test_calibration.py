"""Unit tests for the calibration backtest's label/direction bookkeeping.
Synthetic data only; the network-calling parts (`backtest_symbol`'s
`client.get_klines`) are exercised separately via a manual smoke run.
"""

from __future__ import annotations

import inspect

import pandas as pd
import pytest

from scanner.calibration import (
    LABEL_DIRECTION,
    OOS_MIN_TEST_N,
    LabeledCase,
    _walk_forward_exit,
    load_extended_history,
    summarize,
    train_test_split_cases,
    validate_out_of_sample,
)

pytestmark = pytest.mark.unit


def _case(label: str, forward_return_pct: float, horizon: int = 4) -> LabeledCase:
    return LabeledCase(
        symbol="TESTUSDT",
        timeframe="1h",
        label=label,
        entry_time="2026-01-01T00:00:00+00:00",
        entry_close=100.0,
        forward_return_pct=forward_return_pct,
        horizon_candles=horizon,
    )


def test_every_label_the_anomaly_module_can_emit_has_a_declared_direction():
    """The calibration module must not silently skip a label's direction —
    every anomaly the scanner can flag (bar the ones needing live funding/OI
    history, which this historical backtest cannot replay) is accounted for."""
    from tradingagents.dataflows.vendors.binance.anomaly import detect_anomalies

    source = inspect.getsource(detect_anomalies)
    for flag in (
        "price_up_volume_not_confirming",
        "oi_rising_price_flat_funding_spiking",
        "failed_breakout",
        "volatility_squeeze_expansion",
        "squeeze_breakout_long",
        "squeeze_breakout_short",
        "ema9_21_bullish_cross",
        "ema9_21_bearish_cross",
        "bullish_pullback_confluence",
        "bearish_pullback_confluence",
    ):
        assert flag in source  # sanity: these are still the anomaly names in use
        assert f"anomaly:{flag}" in LABEL_DIRECTION


def test_long_label_profits_when_price_rises_above_cost():
    cases = [_case("trend:bullish", forward_return_pct=1.0)] * 10
    stats = summarize(cases, cost_pct=0.1)
    assert len(stats) == 1
    assert stats[0].win_rate_pct == 100.0
    assert stats[0].avg_net_return_pct == pytest.approx(0.9)


def test_long_label_loses_when_price_falls():
    cases = [_case("trend:bullish", forward_return_pct=-1.0)] * 10
    stats = summarize(cases, cost_pct=0.1)
    assert stats[0].win_rate_pct == 0.0
    assert stats[0].avg_loss_return_pct == pytest.approx(-1.1)


def test_short_label_profits_when_price_falls():
    cases = [_case("trend:bearish", forward_return_pct=-2.0)] * 10
    stats = summarize(cases, cost_pct=0.1)
    assert stats[0].win_rate_pct == 100.0
    assert stats[0].avg_net_return_pct == pytest.approx(1.9)


def test_undirected_label_reports_magnitude_not_win_rate():
    cases = [_case("trend:neutral", forward_return_pct=-3.0), _case("trend:neutral", forward_return_pct=3.0)]
    stats = summarize(cases, cost_pct=0.1)
    assert stats[0].win_rate_pct is None
    assert stats[0].avg_win_return_pct is None
    assert stats[0].avg_net_return_pct == pytest.approx(3.0)  # mean of |-3| and |3|


def test_summarize_empty_input_returns_empty_list():
    assert summarize([]) == []


def test_stats_sorted_by_sample_size_descending():
    cases = [_case("trend:bullish", 1.0, horizon=4)] * 3 + [_case("trend:bearish", -1.0, horizon=4)] * 7
    stats = summarize(cases)
    assert [s.n for s in stats] == [7, 3]


def test_confidence_interval_widens_with_small_sample_size():
    """Same 70% point estimate, far fewer trials — the interval must be wider,
    so a flashy-looking win rate on thin evidence doesn't read as equally
    trustworthy as one backed by many cases."""
    from scanner.calibration import wilson_score_interval

    low_n, high_n = wilson_score_interval(70.0, n=10)
    low_big, high_big = wilson_score_interval(70.0, n=500)
    assert (high_n - low_n) > (high_big - low_big)


def test_confidence_interval_stays_within_bounds():
    from scanner.calibration import wilson_score_interval

    low, high = wilson_score_interval(100.0, n=5)
    assert 0.0 <= low <= high <= 100.0


def test_confidence_interval_none_when_win_rate_is_none():
    from scanner.calibration import wilson_score_interval

    assert wilson_score_interval(None, n=50) == (None, None)


def test_undirected_label_has_no_confidence_interval():
    cases = [_case("trend:neutral", forward_return_pct=-3.0), _case("trend:neutral", forward_return_pct=3.0)]
    stats = summarize(cases, cost_pct=0.1)
    assert stats[0].win_rate_ci_low_pct is None
    assert stats[0].win_rate_ci_high_pct is None


def test_directed_label_includes_confidence_interval():
    cases = [_case("trend:bullish", forward_return_pct=1.0)] * 10
    stats = summarize(cases, cost_pct=0.1)
    assert stats[0].win_rate_ci_low_pct is not None
    assert stats[0].win_rate_ci_low_pct <= stats[0].win_rate_pct <= stats[0].win_rate_ci_high_pct


def _ohlc_df(bars: list[tuple[float, float]]) -> pd.DataFrame:
    """Build a minimal high/low dataframe indexed exactly like the real
    history: row 0 is the entry candle itself (never consulted by
    ``_walk_forward_exit``, which only looks at ``i + h`` for ``h >= 1``),
    and ``bars`` supplies (low, high) for the candles that follow it."""
    rows = [{"low": 0.0, "high": 0.0}] + [{"low": lo, "high": hi} for lo, hi in bars]
    return pd.DataFrame(rows)


class TestWalkForwardExit:
    """``_walk_forward_exit`` powers the path-dependent backtest: it must
    replicate exactly the SL/TP a live user is shown (same ATR multipliers
    as ``agents_crypto.risk_calc.compute_risk_levels``) and resolve ties
    conservatively when a single candle's range covers both levels."""

    def test_long_take_profit_hit_first(self):
        # entry 100, atr 2 -> SL 97.0, TP 105.0 (1.5x/2.5x ATR)
        df = _ohlc_df([(99.0, 101.0), (104.5, 105.5)])
        hit_candle, exit_price, reason = _walk_forward_exit(df, 0, "long", 100.0, 2.0, max_horizon=5)
        assert hit_candle == 2
        assert exit_price == pytest.approx(105.0)
        assert reason == "take_profit"

    def test_long_stop_loss_hit_first(self):
        df = _ohlc_df([(96.5, 101.0)])
        hit_candle, exit_price, reason = _walk_forward_exit(df, 0, "long", 100.0, 2.0, max_horizon=5)
        assert hit_candle == 1
        assert exit_price == pytest.approx(97.0)
        assert reason == "stop_loss"

    def test_long_both_levels_touched_same_candle_assumes_stop_first(self):
        # Pathological wide-range candle spanning both SL (97) and TP (105) —
        # can't know which happened first from OHLC alone, so assume the
        # worse (stop) outcome rather than overstate the edge.
        df = _ohlc_df([(96.0, 106.0)])
        hit_candle, exit_price, reason = _walk_forward_exit(df, 0, "long", 100.0, 2.0, max_horizon=5)
        assert reason == "stop_loss"
        assert exit_price == pytest.approx(97.0)

    def test_short_take_profit_hit_first(self):
        # entry 100, atr 2 -> SL 103.0, TP 95.0 (mirrored for shorts)
        df = _ohlc_df([(96.0, 102.0), (94.5, 99.0)])
        hit_candle, exit_price, reason = _walk_forward_exit(df, 0, "short", 100.0, 2.0, max_horizon=5)
        assert hit_candle == 2
        assert exit_price == pytest.approx(95.0)
        assert reason == "take_profit"

    def test_short_stop_loss_hit_first(self):
        df = _ohlc_df([(99.0, 103.5)])
        hit_candle, exit_price, reason = _walk_forward_exit(df, 0, "short", 100.0, 2.0, max_horizon=5)
        assert hit_candle == 1
        assert exit_price == pytest.approx(103.0)
        assert reason == "stop_loss"

    def test_neither_level_touched_within_horizon_times_out(self):
        df = _ohlc_df([(99.0, 101.0), (99.5, 100.5), (99.0, 101.0)])
        hit_candle, exit_price, reason = _walk_forward_exit(df, 0, "long", 100.0, 2.0, max_horizon=3)
        assert hit_candle is None
        assert exit_price is None
        assert reason == "timeout"

    def test_zero_or_missing_atr_yields_no_decision(self):
        df = _ohlc_df([(90.0, 110.0)])
        assert _walk_forward_exit(df, 0, "long", 100.0, 0.0, max_horizon=5) == (None, None, None)
        assert _walk_forward_exit(df, 0, "long", 100.0, float("nan"), max_horizon=5) == (None, None, None)

    def test_exhausting_dataframe_before_horizon_times_out(self):
        # Only one future candle available even though max_horizon asks for 5.
        df = _ohlc_df([(99.0, 101.0)])
        hit_candle, exit_price, reason = _walk_forward_exit(df, 0, "long", 100.0, 2.0, max_horizon=5)
        assert hit_candle is None
        assert reason == "timeout"

    def test_swapped_multipliers_change_which_level_is_touched(self):
        # Same candle, same (long) direction, but with SL/TP multipliers
        # swapped -- entry 100, atr 2 -> with defaults (1.5x/2.5x) SL=97,
        # TP=105; swapped (2.5x/1.5x) SL=95, TP=103. A candle that only
        # reaches 103 hits the swapped TP but not the default one.
        df = _ohlc_df([(99.0, 103.5)])
        default_hit, _, default_reason = _walk_forward_exit(df, 0, "long", 100.0, 2.0, max_horizon=5)
        assert default_hit is None and default_reason == "timeout"
        swapped_hit, swapped_price, swapped_reason = _walk_forward_exit(
            df, 0, "long", 100.0, 2.0, max_horizon=5, sl_atr_multiplier=2.5, tp_atr_multiplier=1.5
        )
        assert swapped_hit == 1
        assert swapped_reason == "take_profit"
        assert swapped_price == pytest.approx(103.0)


def _timed_case(label: str, entry_time: str, forward_return_pct: float, horizon: int = 4) -> LabeledCase:
    return LabeledCase(
        symbol="TESTUSDT",
        timeframe="1h",
        label=label,
        entry_time=entry_time,
        entry_close=100.0,
        forward_return_pct=forward_return_pct,
        horizon_candles=horizon,
    )


class TestOutOfSampleValidation:
    def test_split_keeps_oldest_cases_in_train_and_newest_in_test(self):
        cases = [_timed_case("trend:bullish", f"2026-01-{day:02d}T00:00:00+00:00", 1.0) for day in range(1, 11)]
        train, test = train_test_split_cases(cases, test_fraction=0.3)
        assert len(train) == 7
        assert len(test) == 3
        assert max(c.entry_time for c in train) < min(c.entry_time for c in test)

    def test_edge_that_holds_up_on_holdout_is_flagged_true(self):
        # Consistently profitable long in both train and test windows.
        cases = [
            _timed_case("trend:bullish", f"2026-01-{day:02d}T00:00:00+00:00", 2.0) for day in range(1, 51)
        ]
        results = validate_out_of_sample(cases, min_test_n=5)
        assert len(results) == 1
        assert results[0].holds_up is True
        assert results[0].warning is None

    def test_edge_that_only_worked_in_sample_is_flagged_false(self):
        # Profitable in the (older) train window, but a clear loser in the
        # (newer) held-out test window — the edge did not survive.
        train_cases = [
            _timed_case("trend:bullish", f"2026-01-{day:02d}T00:00:00+00:00", 2.0) for day in range(1, 71)
        ]
        test_cases = [
            _timed_case("trend:bullish", f"2026-03-{day:02d}T00:00:00+00:00", -2.0) for day in range(1, 31)
        ]
        results = validate_out_of_sample(train_cases + test_cases, test_fraction=0.3, min_test_n=5)
        assert len(results) == 1
        assert results[0].holds_up is False
        assert results[0].warning is not None
        assert "NE potvrđuje" in results[0].warning

    def test_undersized_test_window_is_flagged_not_held_up(self):
        cases = [_timed_case("trend:bullish", f"2026-01-{day:02d}T00:00:00+00:00", 2.0) for day in range(1, 11)]
        results = validate_out_of_sample(cases, test_fraction=0.3, min_test_n=OOS_MIN_TEST_N)
        assert len(results) == 1
        assert results[0].holds_up is False
        assert "premalo" in results[0].warning

    def test_undirected_label_is_excluded_from_results(self):
        cases = [
            _timed_case("trend:neutral", f"2026-01-{day:02d}T00:00:00+00:00", 1.0) for day in range(1, 51)
        ]
        assert validate_out_of_sample(cases, min_test_n=5) == []

    def test_empty_cases_returns_empty_list(self):
        assert validate_out_of_sample([]) == []


def _kline(open_time_ms: int, close: float = 100.0) -> dict:
    """Minimal synthetic kline dict matching the shape client.get_klines returns."""
    return {
        "open_time": open_time_ms,
        "open": str(close),
        "high": str(close),
        "low": str(close),
        "close": str(close),
        "volume": "1.0",
        "close_time": open_time_ms + 59_999,
        "quote_volume": "100.0",
        "trades": 1,
        "taker_buy_base_volume": "0.5",
        "taker_buy_quote_volume": "50.0",
        "ignore": "0",
    }


class TestLoadExtendedHistory:
    """`load_extended_history` pages backwards past Binance's single-request
    1500-candle cap via `end_time`; these tests mock `client.get_klines`
    directly so no network call is ever made."""

    _HOUR_MS = 3_600_000

    def test_single_batch_when_total_candles_fits_in_one_request(self, monkeypatch):
        import scanner.calibration as calibration_module

        klines = [_kline(i * self._HOUR_MS) for i in range(10)]

        def fake_get_klines(symbol, timeframe, limit, end_time=None):
            assert end_time is None  # first (and only) batch never passes end_time
            assert limit == 10
            return klines

        monkeypatch.setattr(calibration_module.client, "get_klines", fake_get_klines)
        df = load_extended_history("BTCUSDT", "1h", total_candles=10, batch_limit=1500)
        assert len(df) == 10
        assert list(df["open_time"]) == sorted(df["open_time"])

    def test_pages_backwards_across_multiple_batches(self, monkeypatch):
        import scanner.calibration as calibration_module

        # 25 total candles with a batch_limit of 10 -> 3 requests (10, 10, 5),
        # each one older than the last, paginated via end_time.
        all_klines = [_kline(i * self._HOUR_MS) for i in range(25)]
        calls: list[dict] = []

        def fake_get_klines(symbol, timeframe, limit, end_time=None):
            calls.append({"limit": limit, "end_time": end_time})
            if end_time is None:
                batch = all_klines[-limit:]
            else:
                # Return the `limit` candles whose open_time is <= end_time.
                eligible = [k for k in all_klines if k["open_time"] <= end_time]
                batch = eligible[-limit:]
            return batch

        monkeypatch.setattr(calibration_module.client, "get_klines", fake_get_klines)
        df = load_extended_history("BTCUSDT", "1h", total_candles=25, batch_limit=10)

        assert len(calls) == 3
        assert len(df) == 25
        # Must be sorted ascending (oldest first) with no gaps/duplicates.
        open_times = list(df["open_time"])
        assert open_times == sorted(open_times)
        assert len(set(open_times)) == 25

    def test_stops_early_when_vendor_runs_out_of_history(self, monkeypatch):
        import scanner.calibration as calibration_module

        # Only 15 candles actually exist, but total_candles asks for 50.
        all_klines = [_kline(i * self._HOUR_MS) for i in range(15)]

        def fake_get_klines(symbol, timeframe, limit, end_time=None):
            if end_time is None:
                return all_klines[-min(limit, len(all_klines)) :]
            eligible = [k for k in all_klines if k["open_time"] <= end_time]
            return eligible[-min(limit, len(eligible)) :]

        monkeypatch.setattr(calibration_module.client, "get_klines", fake_get_klines)
        df = load_extended_history("BTCUSDT", "1h", total_candles=50, batch_limit=10)
        assert len(df) == 15  # didn't loop forever waiting for candles that don't exist

    def test_indicators_computed_once_over_full_assembled_series(self, monkeypatch):
        import scanner.calibration as calibration_module

        klines = [_kline(i * self._HOUR_MS, close=100.0 + i) for i in range(60)]

        def fake_get_klines(symbol, timeframe, limit, end_time=None):
            return klines[-limit:] if end_time is None else [k for k in klines if k["open_time"] <= end_time][-limit:]

        monkeypatch.setattr(calibration_module.client, "get_klines", fake_get_klines)
        df = load_extended_history("BTCUSDT", "1h", total_candles=60, batch_limit=30)
        # An EMA column should exist and be fully populated (no NaNs from a
        # broken/discontinuous series caused by per-batch indicator computation).
        assert "close_20_ema" in df.columns
        assert df["close_20_ema"].notna().sum() > 0


class TestRoundTripCostPctFor:
    """Phase 8 Korak 5: per-symbol liquidity-tier cost surcharge."""

    def test_tier1_major_gets_no_surcharge(self):
        from scanner.calibration import round_trip_cost_pct_for

        assert round_trip_cost_pct_for("BTCUSDT", base_cost_pct=0.1) == pytest.approx(0.1)
        assert round_trip_cost_pct_for("ETHUSDT", base_cost_pct=0.1) == pytest.approx(0.1)
        assert round_trip_cost_pct_for("BNBUSDT", base_cost_pct=0.1) == pytest.approx(0.1)

    def test_tier2_large_cap_gets_small_surcharge(self):
        from scanner.calibration import round_trip_cost_pct_for

        assert round_trip_cost_pct_for("SOLUSDT", base_cost_pct=0.1) == pytest.approx(0.12)
        assert round_trip_cost_pct_for("XRPUSDT", base_cost_pct=0.1) == pytest.approx(0.12)

    def test_tier3_everything_else_gets_largest_surcharge(self):
        from scanner.calibration import round_trip_cost_pct_for

        assert round_trip_cost_pct_for("SOMERANDOMCOINUSDT", base_cost_pct=0.1) == pytest.approx(0.15)

    def test_default_base_cost_is_the_module_default(self):
        from scanner.calibration import DEFAULT_ROUND_TRIP_COST_PCT, round_trip_cost_pct_for

        assert round_trip_cost_pct_for("BTCUSDT") == pytest.approx(DEFAULT_ROUND_TRIP_COST_PCT)


class TestWalkForwardExitTieBreak:
    """Phase 8 Korak 5: the default ("conservative") tie-break is unchanged
    from before; "optimistic" is a new, explicitly opt-in alternative."""

    def test_optimistic_tie_break_assumes_target_hit_first(self):
        df = _ohlc_df([(96.0, 106.0)])  # spans both SL (97) and TP (105)
        hit_candle, exit_price, reason = _walk_forward_exit(
            df, 0, "long", 100.0, 2.0, max_horizon=5, tie_break="optimistic"
        )
        assert reason == "take_profit"
        assert exit_price == pytest.approx(105.0)

    def test_conservative_is_still_the_default(self):
        df = _ohlc_df([(96.0, 106.0)])
        hit_candle, exit_price, reason = _walk_forward_exit(df, 0, "long", 100.0, 2.0, max_horizon=5)
        assert reason == "stop_loss"
        assert exit_price == pytest.approx(97.0)


def _trend_kline(open_time_ms: int, base: float) -> dict:
    """A steadily-climbing OHLCV candle with enough intra-candle range to
    yield a non-zero ATR, for feeding a real `backtest_symbol` walk-forward
    (as opposed to `_kline`'s flat candles used elsewhere in this file,
    which would never trigger either SL or TP)."""
    return {
        "open_time": open_time_ms,
        "open": str(base - 0.2),
        "high": str(base + 0.5),
        "low": str(base - 0.5),
        "close": str(base),
        "volume": "100.0",
        "close_time": open_time_ms + 3_599_999,
        "quote_volume": "10000.0",
        "trades": 10,
        "taker_buy_base_volume": "50.0",
        "taker_buy_quote_volume": "5000.0",
        "ignore": "0",
    }


class TestBacktestSymbolFundingCost:
    """Phase 8 Korak 5: `backtest_symbol`'s optional `funding_events` wiring.
    Uses a real (synthetic) OHLCV series run through the actual indicator /
    trend-classification pipeline, rather than mocking those internals, so
    the test fails if the funding wiring ever stops lining up with the real
    entry/exit timestamps `_walk_forward_exit` resolves."""

    _HOUR_MS = 3_600_000
    _N_CANDLES = 90

    def _history(self) -> pd.DataFrame:
        from tradingagents.dataflows.vendors.binance.indicators import (
            add_indicators,
            klines_to_dataframe,
        )

        klines = [
            _trend_kline(i * self._HOUR_MS, base=100.0 + i * 0.5) for i in range(self._N_CANDLES)
        ]
        return add_indicators(klines_to_dataframe(klines))

    def test_funding_events_none_means_zero_cost_on_every_case(self):
        from scanner.calibration import backtest_symbol

        df = self._history()
        cases = backtest_symbol("TESTUSDT", "1h", horizons=(4,), history=df, min_history=60)
        assert cases, "expected at least one case from a steady 90-candle uptrend"
        assert all(c.funding_cost_pct == 0.0 for c in cases)

    def test_funding_events_given_charges_directional_cases(self):
        from scanner.backtesting.funding_simulator import FundingEvent
        from scanner.calibration import backtest_symbol

        df = self._history()
        # A settlement exactly on every candle boundary in the series --
        # every multi-candle-horizon long case must therefore pick up at
        # least one (positive-rate) settlement and so pay a nonzero cost.
        funding_events = [
            FundingEvent(timestamp=df["open_time"].iloc[i].to_pydatetime(), rate=0.0001)
            for i in range(len(df))
        ]
        cases = backtest_symbol(
            "TESTUSDT", "1h", horizons=(4,), history=df, min_history=60, funding_events=funding_events
        )
        long_cases = [c for c in cases if c.label == "trend:bullish"]
        assert long_cases, "expected the steady uptrend to be classified bullish at least once"
        assert all(c.funding_cost_pct > 0.0 for c in long_cases)

    def test_sl_tp_tie_break_is_forwarded_from_backtest_symbol(self, monkeypatch):
        """`backtest_symbol`'s `sl_tp_tie_break` kwarg must actually reach
        `_walk_forward_exit` -- assert via the call args it's invoked with,
        rather than re-deriving an end-to-end price scenario."""
        import scanner.calibration as calibration_module

        df = self._history()
        seen_tie_breaks: list[str] = []
        real_walk_forward_exit = calibration_module._walk_forward_exit

        def spy(df_, i, direction, entry_close, entry_atr, max_horizon, tie_break="conservative", *args):
            seen_tie_breaks.append(tie_break)
            return real_walk_forward_exit(df_, i, direction, entry_close, entry_atr, max_horizon, tie_break, *args)

        monkeypatch.setattr(calibration_module, "_walk_forward_exit", spy)
        calibration_module.backtest_symbol(
            "TESTUSDT", "1h", horizons=(4,), history=df, min_history=60, sl_tp_tie_break="optimistic"
        )
        assert seen_tie_breaks, "expected _walk_forward_exit to have been called at least once"
        assert all(tb == "optimistic" for tb in seen_tie_breaks)


class TestInvertSignals:
    """Phase 8 "strategy variants" framework: `backtest_symbol(invert_signals=True)`
    must trade every directional label the OPPOSITE way, with SL/TP ATR
    multipliers swapped -- not just negate the original return -- and must
    store `forward_return_pct` pre-negated so the rest of the pipeline
    (which still re-signs using the label's static, non-inverted
    `LABEL_DIRECTION`) scores it correctly."""

    _HOUR_MS = 3_600_000
    _N_CANDLES = 90

    def _history(self) -> pd.DataFrame:
        from tradingagents.dataflows.vendors.binance.indicators import (
            add_indicators,
            klines_to_dataframe,
        )

        klines = [_trend_kline(i * self._HOUR_MS, base=100.0 + i * 0.5) for i in range(self._N_CANDLES)]
        return add_indicators(klines_to_dataframe(klines))

    def test_inverted_case_has_same_label_and_horizon_as_original(self):
        from scanner.calibration import backtest_symbol

        df = self._history()
        original = backtest_symbol("TESTUSDT", "1h", horizons=(4,), history=df, min_history=60)
        inverted = backtest_symbol(
            "TESTUSDT", "1h", horizons=(4,), history=df, min_history=60, invert_signals=True
        )
        original_keys = {(c.label, c.entry_time, c.horizon_candles) for c in original}
        inverted_keys = {(c.label, c.entry_time, c.horizon_candles) for c in inverted}
        assert original_keys == inverted_keys, "inverting must not change which (label, candle) cases exist"

    def test_inverted_sl_distance_pct_uses_the_swapped_multiplier(self):
        from agents_crypto.risk_calc import DEFAULT_SL_ATR_MULTIPLIER, DEFAULT_TP_ATR_MULTIPLIER
        from scanner.calibration import backtest_symbol

        df = self._history()
        original = {
            (c.label, c.entry_time): c
            for c in backtest_symbol("TESTUSDT", "1h", horizons=(4,), history=df, min_history=60)
        }
        inverted = {
            (c.label, c.entry_time): c
            for c in backtest_symbol(
                "TESTUSDT", "1h", horizons=(4,), history=df, min_history=60, invert_signals=True
            )
        }
        checked = 0
        for key, orig_case in original.items():
            if orig_case.sl_distance_pct is None:
                continue  # undirected label
            inv_case = inverted[key]
            # Same underlying ATR -> ratio of the two stored distances must
            # equal the ratio of the two (swapped) multipliers, regardless
            # of the exact ATR value at that candle.
            ratio = inv_case.sl_distance_pct / orig_case.sl_distance_pct
            assert ratio == pytest.approx(DEFAULT_TP_ATR_MULTIPLIER / DEFAULT_SL_ATR_MULTIPLIER, rel=1e-6)
            checked += 1
        assert checked, "expected at least one directional label to compare"

    def test_inverted_walk_direction_and_multipliers_are_swapped(self, monkeypatch):
        """Asserts directly against the call args `_walk_forward_exit` is
        invoked with -- the opposite direction and swapped multipliers --
        rather than re-deriving the outcome end-to-end."""
        import scanner.calibration as calibration_module

        df = self._history()
        seen: list[tuple[str, float, float]] = []
        real_walk_forward_exit = calibration_module._walk_forward_exit

        def spy(df_, i, direction, entry_close, entry_atr, max_horizon, tie_break="conservative", *mults):
            seen.append((direction, *mults))
            return real_walk_forward_exit(df_, i, direction, entry_close, entry_atr, max_horizon, tie_break, *mults)

        monkeypatch.setattr(calibration_module, "_walk_forward_exit", spy)
        calibration_module.backtest_symbol(
            "TESTUSDT", "1h", horizons=(4,), history=df, min_history=60, invert_signals=True
        )
        assert seen, "expected _walk_forward_exit to have been called at least once"
        from agents_crypto.risk_calc import DEFAULT_SL_ATR_MULTIPLIER, DEFAULT_TP_ATR_MULTIPLIER

        for direction, sl_mult, tp_mult in seen:
            assert direction in ("long", "short")
            assert sl_mult == pytest.approx(DEFAULT_TP_ATR_MULTIPLIER)
            assert tp_mult == pytest.approx(DEFAULT_SL_ATR_MULTIPLIER)

    def test_inverted_return_is_pre_negated_relative_to_a_symmetric_walk(self, monkeypatch):
        """With SYMMETRIC SL/TP (swap is a no-op on the levels themselves),
        the only remaining difference between the original and inverted
        exit price is the direction walked -- so the inverted return must
        come out as the exact negation of what a plain direction flip (with
        no multiplier swap at all) would have produced."""
        import scanner.calibration as calibration_module

        monkeypatch.setattr(calibration_module, "DEFAULT_SL_ATR_MULTIPLIER", 2.0)
        monkeypatch.setattr(calibration_module, "DEFAULT_TP_ATR_MULTIPLIER", 2.0)
        df = self._history()
        original = {
            (c.label, c.entry_time): c
            for c in calibration_module.backtest_symbol("TESTUSDT", "1h", horizons=(4,), history=df, min_history=60)
        }
        inverted = {
            (c.label, c.entry_time): c
            for c in calibration_module.backtest_symbol(
                "TESTUSDT", "1h", horizons=(4,), history=df, min_history=60, invert_signals=True
            )
        }
        checked = 0
        for key, orig_case in original.items():
            if orig_case.exit_reason not in ("take_profit", "stop_loss"):
                continue  # only directional, SL/TP-resolved cases are comparable here
            inv_case = inverted[key]
            assert inv_case.forward_return_pct == pytest.approx(-orig_case.forward_return_pct, rel=1e-6)
            checked += 1
        assert checked, "expected at least one SL/TP-resolved directional case to compare"


class TestSlTpOverrides:
    """Korak 2: `backtest_symbol(sl_tp_overrides=...)` -- a per-label
    ATR-multiplier override, calibrated from live MFE/MAE data (see
    `scanner.analysis.sl_tp_calibration`), that lets a re-run test whether
    a label's own empirical excursion distribution beats the flat
    default."""

    _HOUR_MS = 3_600_000
    _N_CANDLES = 90

    def _history(self) -> pd.DataFrame:
        from tradingagents.dataflows.vendors.binance.indicators import (
            add_indicators,
            klines_to_dataframe,
        )

        klines = [_trend_kline(i * self._HOUR_MS, base=100.0 + i * 0.5) for i in range(self._N_CANDLES)]
        return add_indicators(klines_to_dataframe(klines))

    def test_none_overrides_matches_omitting_the_argument(self):
        from scanner.calibration import backtest_symbol

        df = self._history()
        without_arg = backtest_symbol("TESTUSDT", "1h", horizons=(4,), history=df, min_history=60)
        with_none = backtest_symbol(
            "TESTUSDT", "1h", horizons=(4,), history=df, min_history=60, sl_tp_overrides=None
        )
        assert [(c.label, c.entry_time, c.forward_return_pct) for c in without_arg] == [
            (c.label, c.entry_time, c.forward_return_pct) for c in with_none
        ]

    def test_override_changes_sl_distance_only_for_the_overridden_label(self):
        from scanner.calibration import backtest_symbol

        df = self._history()
        baseline = {(c.label, c.entry_time): c for c in backtest_symbol("TESTUSDT", "1h", horizons=(4,), history=df, min_history=60)}
        overridden = {
            (c.label, c.entry_time): c
            for c in backtest_symbol(
                "TESTUSDT",
                "1h",
                horizons=(4,),
                history=df,
                min_history=60,
                sl_tp_overrides={"trend:bullish": (1.0, 1.0)},
            )
        }
        changed_any = False
        for key, base_case in baseline.items():
            over_case = overridden[key]
            if key[0] == "trend:bullish":
                if base_case.sl_distance_pct is not None:
                    assert over_case.sl_distance_pct != pytest.approx(base_case.sl_distance_pct)
                    changed_any = True
            else:
                assert over_case.sl_distance_pct == base_case.sl_distance_pct
        assert changed_any, "expected at least one trend:bullish case with a changed sl_distance_pct"

    def test_overrides_and_invert_signals_cannot_be_combined(self):
        from scanner.calibration import backtest_symbol

        df = self._history()
        with pytest.raises(ValueError):
            backtest_symbol(
                "TESTUSDT",
                "1h",
                horizons=(4,),
                history=df,
                min_history=60,
                invert_signals=True,
                sl_tp_overrides={"trend:bullish": (1.0, 1.0)},
            )

    def test_walk_forward_exit_receives_the_overridden_multipliers(self, monkeypatch):
        import scanner.calibration as calibration_module

        df = self._history()
        seen_mults: list[tuple[float, float]] = []
        real_walk_forward_exit = calibration_module._walk_forward_exit

        def spy(df_, i, direction, entry_close, entry_atr, max_horizon, tie_break="conservative", *mults):
            seen_mults.append(mults)
            return real_walk_forward_exit(df_, i, direction, entry_close, entry_atr, max_horizon, tie_break, *mults)

        monkeypatch.setattr(calibration_module, "_walk_forward_exit", spy)
        calibration_module.backtest_symbol(
            "TESTUSDT",
            "1h",
            horizons=(4,),
            history=df,
            min_history=60,
            sl_tp_overrides={"trend:bullish": (0.7, 3.3)},
        )
        assert (0.7, 3.3) in seen_mults, "expected at least one walk to use the overridden (sl, tp) multiplier pair"


