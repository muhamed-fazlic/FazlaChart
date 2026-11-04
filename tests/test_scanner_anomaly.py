"""Unit tests for single-symbol anomaly detection. Synthetic data only."""

from __future__ import annotations

import pandas as pd
import pytest

from tradingagents.dataflows.vendors.binance.anomaly import (
    bearish_pullback_confluence,
    bullish_pullback_confluence,
    detect_anomalies,
    ema9_21_bearish_cross,
    ema9_21_bullish_cross,
    failed_breakout,
    oi_rising_price_flat_funding_spiking,
    price_up_volume_not_confirming,
    squeeze_breakout_direction,
    volatility_squeeze_expansion,
)
from tradingagents.dataflows.vendors.binance.indicators import add_indicators, klines_to_dataframe

pytestmark = pytest.mark.unit


def _make_klines(closes: list[float], volumes: list[float] | None = None, highs: list[float] | None = None) -> list[dict]:
    volumes = volumes or [1000.0] * len(closes)
    klines = []
    for i, close in enumerate(closes):
        open_price = closes[i - 1] if i > 0 else close
        high = highs[i] if highs else max(open_price, close) * 1.001
        klines.append(
            {
                "open_time": i * 3_600_000,
                "open": open_price,
                "high": high,
                "low": min(open_price, close) * 0.999,
                "close": close,
                "volume": volumes[i],
                "quote_volume": volumes[i] * close,
            }
        )
    return klines


def test_price_up_volume_not_confirming_fires_on_quiet_rally():
    closes = [100.0] * 25 + [101.0, 102.5, 104.0]
    volumes = [1000.0] * 25 + [900.0, 950.0, 1000.0]  # no volume expansion
    df = klines_to_dataframe(_make_klines(closes, volumes))
    assert price_up_volume_not_confirming(df) is True


def test_price_up_volume_not_confirming_is_false_when_volume_confirms():
    closes = [100.0] * 25 + [101.0, 102.5, 104.0]
    volumes = [1000.0] * 25 + [3000.0, 3200.0, 3500.0]  # clear expansion
    df = klines_to_dataframe(_make_klines(closes, volumes))
    assert price_up_volume_not_confirming(df) is False


def test_oi_rising_price_flat_funding_spiking():
    assert oi_rising_price_flat_funding_spiking(0.1, 8.0, 0.001) is True
    assert oi_rising_price_flat_funding_spiking(2.0, 8.0, 0.001) is False  # price moved too much
    assert oi_rising_price_flat_funding_spiking(0.1, None, 0.001) is False  # no OI history


def test_failed_breakout_detects_pierce_and_close_back_below():
    closes = [100.0] * 10 + [101.0]
    highs = [100.5] * 10 + [103.0]  # pierced resistance intrabar
    df = klines_to_dataframe(_make_klines(closes, highs=highs))
    assert failed_breakout(df, resistance=102.0) is True


def test_failed_breakout_false_on_clean_breakout_that_holds():
    closes = [100.0] * 10 + [105.0]
    df = klines_to_dataframe(_make_klines(closes))
    assert failed_breakout(df, resistance=102.0) is False


def test_squeeze_breakout_direction_is_long_when_close_clears_the_squeeze_range_high():
    quiet = [100.0 + (0.05 if i % 2 == 0 else -0.05) for i in range(40)]
    closes = quiet + [108.0]
    highs = [c + 0.1 for c in quiet] + [108.2]
    lows = [c - 0.1 for c in quiet] + [103.0]
    df = add_indicators(klines_to_dataframe(_make_klines(closes, highs=highs)))
    df["low"] = lows  # _make_klines derives low from close/open, not an independent list -- override directly
    assert volatility_squeeze_expansion(df) is True
    assert squeeze_breakout_direction(df) == "long"


def test_squeeze_breakout_direction_is_short_when_close_breaks_the_squeeze_range_low():
    quiet = [100.0 + (0.05 if i % 2 == 0 else -0.05) for i in range(40)]
    closes = quiet + [92.0]
    highs = [c + 0.1 for c in quiet] + [97.0]
    lows = [c - 0.1 for c in quiet] + [91.8]
    df = add_indicators(klines_to_dataframe(_make_klines(closes, highs=highs)))
    df["low"] = lows
    assert volatility_squeeze_expansion(df) is True
    assert squeeze_breakout_direction(df) == "short"


def test_squeeze_breakout_direction_is_none_without_a_squeeze_at_all():
    quiet = [100.0 + (0.05 if i % 2 == 0 else -0.05) for i in range(40)]
    df = add_indicators(klines_to_dataframe(_make_klines(quiet + [100.06])))
    assert volatility_squeeze_expansion(df) is False
    assert squeeze_breakout_direction(df) is None


def test_squeeze_breakout_direction_is_none_when_squeeze_expands_but_close_stays_inside_prior_range():
    """A squeeze can start expanding without yet committing to a side -- the
    breakout candle's close must actually clear the preceding range's own
    high/low, not just register as a bigger-than-usual candle."""
    quiet = [100.0 + (0.05 if i % 2 == 0 else -0.05) for i in range(40)]
    closes = quiet + [103.0]
    highs = [110.0] * 40 + [103.2]
    lows = [90.0] * 40 + [102.8]
    df = add_indicators(klines_to_dataframe(_make_klines(closes, highs=highs)))
    df["low"] = lows
    assert volatility_squeeze_expansion(df) is True
    assert squeeze_breakout_direction(df) is None


def test_detect_anomalies_includes_directional_squeeze_breakout_flag():
    quiet = [100.0 + (0.05 if i % 2 == 0 else -0.05) for i in range(40)]
    closes = quiet + [108.0]
    highs = [c + 0.1 for c in quiet] + [108.2]
    lows = [c - 0.1 for c in quiet] + [103.0]
    df = add_indicators(klines_to_dataframe(_make_klines(closes, highs=highs)))
    df["low"] = lows
    anomalies = detect_anomalies(df, resistance=999.0, funding_rate=0.0, oi_change_pct_60m=None)
    assert "volatility_squeeze_expansion" in anomalies
    assert "squeeze_breakout_long" in anomalies
    assert "squeeze_breakout_short" not in anomalies


def test_detect_anomalies_aggregates_all_checks():
    closes = [100.0] * 25 + [101.0, 102.5, 104.0]
    volumes = [1000.0] * 25 + [900.0, 950.0, 1000.0]
    df = add_indicators(klines_to_dataframe(_make_klines(closes, volumes)))
    anomalies = detect_anomalies(df, resistance=999.0, funding_rate=0.0, oi_change_pct_60m=None)
    assert "price_up_volume_not_confirming" in anomalies


def _build_confluence_df(
    *,
    broader_trend: str,
    swing_direction: str,
    ema_cross: str | None,
    pocket_close: float | None,
) -> pd.DataFrame:
    """Build a 201-row synthetic frame with hand-placed swing high/low and
    EMA columns, so the confluence conditions (broader trend, Fibonacci
    pocket, fresh EMA9/21 cross) can be dialed in directly instead of
    reverse-engineered from a price sequence. Only the columns the
    confluence/fib/ema-cross functions actually read are populated.
    """
    n = 201
    high = [100.0] * n
    low = [95.0] * n
    # fibonacci_levels' 50-candle window is rows [150:200) (it excludes the
    # last, still-forming row). Placing the swing low at the start of that
    # window and the swing high at its end makes it an "up" swing (the most
    # recent extreme is the high); swapping the positions makes it "down".
    if swing_direction == "up":
        low[150], high[199] = 90.0, 130.0
    else:
        high[150], low[199] = 130.0, 90.0

    close_9_ema = [95.0] * n
    close_21_ema = [95.0] * n
    if ema_cross == "bullish_cross":
        close_9_ema[-2], close_21_ema[-2] = 94.0, 95.0  # bearish the candle before
        close_9_ema[-1], close_21_ema[-1] = 96.0, 95.0  # flips bullish on the last candle
    elif ema_cross == "bearish_cross":
        close_9_ema[-2], close_21_ema[-2] = 96.0, 95.0
        close_9_ema[-1], close_21_ema[-1] = 94.0, 95.0

    close = [100.0] * n
    if pocket_close is not None:
        close[-1] = pocket_close

    return pd.DataFrame(
        {
            "high": high,
            "low": low,
            "close": close,
            "close_9_ema": close_9_ema,
            "close_21_ema": close_21_ema,
            "close_50_ema": [110.0 if broader_trend == "bullish" else 90.0] * n,
            "close_200_ema": [100.0] * n,
        }
    )


def test_bullish_pullback_confluence_fires_when_all_three_conditions_align():
    df = _build_confluence_df(
        broader_trend="bullish",
        swing_direction="up",
        ema_cross="bullish_cross",
        pocket_close=107.0,  # inside the 0.5-0.618 retracement of the 90-130 up-swing
    )
    assert ema9_21_bullish_cross(df) is True
    assert bullish_pullback_confluence(df) is True


def test_bullish_pullback_confluence_false_without_a_fresh_cross():
    df = _build_confluence_df(
        broader_trend="bullish", swing_direction="up", ema_cross=None, pocket_close=107.0
    )
    assert bullish_pullback_confluence(df) is False


def test_bullish_pullback_confluence_false_outside_the_golden_pocket():
    df = _build_confluence_df(
        broader_trend="bullish",
        swing_direction="up",
        ema_cross="bullish_cross",
        pocket_close=129.0,  # near the swing high, not the retracement pocket
    )
    assert bullish_pullback_confluence(df) is False


def test_bullish_pullback_confluence_false_when_broader_trend_is_not_bullish():
    df = _build_confluence_df(
        broader_trend="bearish",
        swing_direction="up",
        ema_cross="bullish_cross",
        pocket_close=107.0,
    )
    assert bullish_pullback_confluence(df) is False


def test_bearish_pullback_confluence_fires_when_all_three_conditions_align():
    df = _build_confluence_df(
        broader_trend="bearish",
        swing_direction="down",
        ema_cross="bearish_cross",
        pocket_close=112.0,  # inside the mirrored 0.5-0.618 pocket (110-114.72) of the 90-130 down-swing
    )
    assert ema9_21_bearish_cross(df) is True
    assert bearish_pullback_confluence(df) is True
