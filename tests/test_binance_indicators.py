"""Unit tests for Binance kline -> indicator conversion. No network calls:
all data is synthetic, built to exercise one behavior per test.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tradingagents.dataflows.vendors.binance.indicators import (
    add_indicators,
    classify_trend,
    drawdown_pct,
    ema_cross,
    ema_cross_event,
    fibonacci_levels,
    klines_to_dataframe,
    relative_volume,
    support_resistance,
    volume_change_pct,
)

pytestmark = pytest.mark.unit


def _make_klines(closes: list[float], volumes: list[float] | None = None, start_ms: int = 0) -> list[dict]:
    """Build minimal Binance-shaped kline dicts from a list of closes."""
    volumes = volumes or [1000.0] * len(closes)
    klines = []
    for i, close in enumerate(closes):
        open_price = closes[i - 1] if i > 0 else close
        klines.append(
            {
                "open_time": start_ms + i * 3_600_000,
                "open": open_price,
                "high": max(open_price, close) * 1.001,
                "low": min(open_price, close) * 0.999,
                "close": close,
                "volume": volumes[i],
                "quote_volume": volumes[i] * close,
            }
        )
    return klines


def test_klines_to_dataframe_is_numeric_and_sorted():
    klines = _make_klines([10, 11, 12])
    df = klines_to_dataframe(klines)
    assert list(df["close"]) == [10, 11, 12]
    assert pd.api.types.is_datetime64_any_dtype(df["open_time"])


def test_classify_trend_bullish_on_steady_uptrend():
    closes = [100 + i * 0.6 for i in range(220)]
    df = add_indicators(klines_to_dataframe(_make_klines(closes)))
    assert classify_trend(df) == "bullish"


def test_classify_trend_bearish_on_steady_downtrend():
    closes = [200 - i * 0.6 for i in range(220)]
    df = add_indicators(klines_to_dataframe(_make_klines(closes)))
    assert classify_trend(df) == "bearish"


def test_classify_trend_neutral_on_flat_chop():
    closes = [100, 101, 99, 100, 101, 99, 100] * 10
    df = add_indicators(klines_to_dataframe(_make_klines(closes)))
    assert classify_trend(df) == "neutral"


def test_relative_volume_flags_a_volume_spike():
    volumes = [1000.0] * 25 + [5000.0]
    df = klines_to_dataframe(_make_klines([100.0] * 26, volumes))
    assert relative_volume(df) > 4.0


def test_volume_change_pct_matches_relative_volume():
    volumes = [1000.0] * 25 + [2000.0]
    df = klines_to_dataframe(_make_klines([100.0] * 26, volumes))
    assert volume_change_pct(df) == pytest.approx((relative_volume(df) - 1.0) * 100.0)


def test_support_resistance_excludes_the_forming_candle():
    closes = [100, 90, 110, 95, 105, 999]  # last candle is an outlier still forming
    df = klines_to_dataframe(_make_klines(closes))
    support, resistance = support_resistance(df, lookback=5)
    assert support == pytest.approx(min(closes[:-1]) * 0.999, rel=1e-3)
    assert resistance == pytest.approx(max(closes[:-1]) * 1.001, rel=1e-3)


def test_ema_cross_bullish_when_ema9_above_ema21():
    closes = [100 + i * 0.6 for i in range(60)]  # steady uptrend: EMA9 > EMA21
    df = add_indicators(klines_to_dataframe(_make_klines(closes)))
    assert ema_cross(df) == "bullish"


def test_ema_cross_bearish_when_ema9_below_ema21():
    closes = [200 - i * 0.6 for i in range(60)]  # steady downtrend: EMA9 < EMA21
    df = add_indicators(klines_to_dataframe(_make_klines(closes)))
    assert ema_cross(df) == "bearish"


def test_ema_cross_event_detects_a_fresh_bullish_cross():
    # Down then sharply up: EMA9 flips above EMA21 right on the last candle.
    closes = [200 - i * 0.6 for i in range(40)] + [180 + i * 8 for i in range(3)]
    df = add_indicators(klines_to_dataframe(_make_klines(closes)))
    assert ema_cross_event(df) == "bullish_cross"


def test_ema_cross_event_is_none_mid_trend():
    # Steady uptrend throughout: EMA9 stays above EMA21 the whole time, no fresh cross.
    closes = [100 + i * 0.6 for i in range(60)]
    df = add_indicators(klines_to_dataframe(_make_klines(closes)))
    assert ema_cross_event(df) is None


def test_fibonacci_levels_up_swing_matches_formula():
    # Clean up-swing: low at the start of the window, high at the end.
    closes = [100 + i for i in range(51)] + [150]  # last candle still forming
    df = klines_to_dataframe(_make_klines(closes))
    fib = fibonacci_levels(df, lookback=50)
    assert fib["swing_direction"] == "up"
    swing_high, swing_low, rng = fib["swing_high"], fib["swing_low"], fib["range"]
    assert rng == pytest.approx(swing_high - swing_low)
    assert fib["retracement"]["0.618"] == pytest.approx(swing_high - rng * 0.618, abs=0.01)
    assert fib["extension"]["0.618"] == pytest.approx(swing_high + rng * 0.618, abs=0.01)


def test_fibonacci_levels_down_swing_is_mirrored():
    # Clean down-swing: high at the start of the window, low at the end.
    closes = [150 - i for i in range(51)] + [100]  # last candle still forming
    df = klines_to_dataframe(_make_klines(closes))
    fib = fibonacci_levels(df, lookback=50)
    assert fib["swing_direction"] == "down"
    swing_low, rng = fib["swing_low"], fib["range"]
    assert fib["retracement"]["0.618"] == pytest.approx(swing_low + rng * 0.618, abs=0.01)
    assert fib["extension"]["0.618"] == pytest.approx(swing_low - rng * 0.618, abs=0.01)


def _make_klines_ohlc(ohlc: list[tuple[float, float, float, float]], start_ms: int = 0) -> list[dict]:
    """Build kline dicts from explicit (open, high, low, close) tuples, for
    tests where the exact high/low matters (unlike :func:`_make_klines`,
    which derives high/low from the close)."""
    return [
        {
            "open_time": start_ms + i * 3_600_000,
            "open": o,
            "high": h,
            "low": lo,
            "close": c,
            "volume": 1000.0,
            "quote_volume": 1000.0 * c,
        }
        for i, (o, h, lo, c) in enumerate(ohlc)
    ]


def test_drawdown_pct_matches_the_worked_example():
    # Vrh = 0.06239, Dno = 0.05524 -> Pad (%) = 11.46%
    ohlc = [
        (0.05000, 0.05000, 0.04800, 0.05000),
        (0.05800, 0.06239, 0.05800, 0.06100),  # local high (Vrh) in the window
        (0.05700, 0.05700, 0.05524, 0.05600),  # local low (Dno) after the peak
        (0.05645, 0.05700, 0.05600, 0.05645),  # still forming, excluded
    ]
    df = klines_to_dataframe(_make_klines_ohlc(ohlc))
    assert drawdown_pct(df, lookback=5) == pytest.approx(11.46, abs=0.01)
