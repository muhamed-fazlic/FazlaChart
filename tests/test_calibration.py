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
    LabeledCase,
    OOS_MIN_TEST_N,
    _walk_forward_exit,
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
