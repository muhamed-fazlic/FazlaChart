"""Unit tests for deterministic ATR-based risk sizing."""

from __future__ import annotations

import pytest

from agents_crypto.risk_calc import KELLY_MAX_FRACTION, compute_kelly_fraction, compute_risk_levels

pytestmark = pytest.mark.unit


def test_long_levels_below_above_entry():
    levels = compute_risk_levels(close=100.0, atr=2.0, bias="long")
    assert levels.stop_loss < levels.entry < levels.take_profit


def test_short_levels_above_below_entry():
    levels = compute_risk_levels(close=100.0, atr=2.0, bias="short")
    assert levels.take_profit < levels.entry < levels.stop_loss


def test_risk_reward_ratio_matches_multiplier_ratio():
    levels = compute_risk_levels(close=100.0, atr=1.0, bias="long", sl_atr_multiplier=1.5, tp_atr_multiplier=3.0)
    assert levels.risk_reward_ratio == pytest.approx(2.0)


@pytest.mark.parametrize("close,atr", [(0, 1.0), (-5, 1.0), (100, 0), (100, -1)])
def test_rejects_non_positive_price_or_atr(close, atr):
    with pytest.raises(ValueError):
        compute_risk_levels(close=close, atr=atr, bias="long")


def test_kelly_fraction_positive_edge_recommends_nonzero_size():
    # 60% win rate, avg win 2%, avg loss -1% -> clear positive edge.
    result = compute_kelly_fraction(60.0, avg_win_return_pct=2.0, avg_loss_return_pct=-1.0)
    assert result.full_kelly_fraction > 0
    assert 0 < result.recommended_fraction <= KELLY_MAX_FRACTION
    assert result.warning is None


def test_kelly_fraction_negative_edge_recommends_zero():
    # 40% win rate with a 1:1 win/loss size has no edge at all.
    result = compute_kelly_fraction(40.0, avg_win_return_pct=1.0, avg_loss_return_pct=-1.0)
    assert result.full_kelly_fraction <= 0
    assert result.recommended_fraction == 0.0
    assert result.warning is not None


def test_kelly_fraction_missing_avg_returns_is_safe():
    result = compute_kelly_fraction(65.0, avg_win_return_pct=None, avg_loss_return_pct=-1.0)
    assert result.recommended_fraction == 0.0
    assert result.warning is not None


def test_kelly_fraction_is_capped_at_max_fraction():
    # An extreme, unrealistic edge should still be capped, not scaled unboundedly.
    result = compute_kelly_fraction(95.0, avg_win_return_pct=10.0, avg_loss_return_pct=-0.1)
    assert result.recommended_fraction == pytest.approx(KELLY_MAX_FRACTION)
    assert result.warning is not None


def test_kelly_fraction_half_kelly_is_half_of_full():
    result = compute_kelly_fraction(60.0, avg_win_return_pct=2.0, avg_loss_return_pct=-1.0, fraction_multiplier=0.5)
    assert result.recommended_fraction == pytest.approx(result.full_kelly_fraction * 0.5, rel=1e-6)


def test_rejects_invalid_bias():
    with pytest.raises(ValueError):
        compute_risk_levels(close=100.0, atr=1.0, bias="sideways")  # type: ignore[arg-type]


def test_take_profit_conservative_sits_between_entry_and_take_profit():
    levels = compute_risk_levels(close=100.0, atr=2.0, bias="long")
    assert levels.entry < levels.take_profit_conservative < levels.take_profit

    levels_short = compute_risk_levels(close=100.0, atr=2.0, bias="short")
    assert levels_short.take_profit < levels_short.take_profit_conservative < levels_short.entry


def test_no_extension_warning_without_rsi_or_ema20():
    levels = compute_risk_levels(close=100.0, atr=2.0, bias="short")
    assert levels.extension_warning is None
    assert levels.patient_entry is None


def test_no_extension_warning_when_not_extended():
    # RSI is oversold-ish but price hasn't actually travelled far from EMA20.
    levels = compute_risk_levels(close=100.0, atr=2.0, bias="short", rsi=30.0, ema20=100.5)
    assert levels.extension_warning is None
    assert levels.patient_entry is None


def test_extension_warning_for_short_chasing_oversold_move():
    # Price is 5 ATR below EMA20 and RSI is deeply oversold: textbook
    # "entering at the bottom of the cycle, then it bounces" setup.
    levels = compute_risk_levels(close=90.0, atr=2.0, bias="short", rsi=25.0, ema20=100.0)
    assert levels.extension_warning is not None
    assert levels.patient_entry is not None
    # Patient entry retraces back toward EMA20, i.e. sits above the current close.
    assert levels.entry < levels.patient_entry < 100.0


def test_extension_warning_for_long_chasing_overbought_move():
    levels = compute_risk_levels(close=110.0, atr=2.0, bias="long", rsi=75.0, ema20=100.0)
    assert levels.extension_warning is not None
    assert levels.patient_entry is not None
    assert 100.0 < levels.patient_entry < levels.entry


def test_no_extension_warning_when_rsi_extreme_but_wrong_direction():
    # RSI overbought but bias is short (not the chasing-an-oversold-dip case) → no warning.
    levels = compute_risk_levels(close=110.0, atr=2.0, bias="short", rsi=75.0, ema20=100.0)
    assert levels.extension_warning is None
