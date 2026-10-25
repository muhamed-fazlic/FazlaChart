"""Unit tests for classic floor-trader daily pivot points. Pure math, no
network calls.
"""

from __future__ import annotations

import pytest

from tradingagents.dataflows.vendors.binance.pivots import pivot_points_from_ohlc

pytestmark = pytest.mark.unit


def test_pivot_points_match_the_worked_example():
    # Vrh = 0.06239, Dno = 0.05524, Zatvaranje = 0.05645
    levels = pivot_points_from_ohlc(prev_high=0.06239, prev_low=0.05524, prev_close=0.05645)
    assert levels["pivot"] == pytest.approx(0.05803, abs=1e-5)
    assert levels["r1"] == pytest.approx(0.06081, abs=1e-5)
    assert levels["s1"] == pytest.approx(0.05366, abs=1e-5)


def test_pivot_points_r2_s2_bracket_the_daily_range_around_pivot():
    levels = pivot_points_from_ohlc(prev_high=110.0, prev_low=90.0, prev_close=100.0)
    day_range = 110.0 - 90.0
    assert levels["r2"] == pytest.approx(levels["pivot"] + day_range)
    assert levels["s2"] == pytest.approx(levels["pivot"] - day_range)


def test_pivot_points_are_ordered_support_below_pivot_below_resistance():
    levels = pivot_points_from_ohlc(prev_high=110.0, prev_low=90.0, prev_close=102.0)
    assert levels["s3"] < levels["s2"] < levels["s1"] < levels["pivot"]
    assert levels["pivot"] < levels["r1"] < levels["r2"] < levels["r3"]
