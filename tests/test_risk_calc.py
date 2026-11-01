"""Unit tests for deterministic ATR-based risk sizing."""

from __future__ import annotations

import pytest

from agents_crypto.risk_calc import compute_risk_levels

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
