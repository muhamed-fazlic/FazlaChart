"""Unit tests for the calibration backtest's label/direction bookkeeping.
Synthetic data only; the network-calling parts (`backtest_symbol`'s
`client.get_klines`) are exercised separately via a manual smoke run.
"""

from __future__ import annotations

import inspect

import pytest

from scanner.calibration import LABEL_DIRECTION, LabeledCase, summarize

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
