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
