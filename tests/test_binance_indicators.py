"""Unit tests for Binance kline -> indicator conversion. No network calls:
all data is synthetic, built to exercise one behavior per test.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tradingagents.dataflows.vendors.binance.indicators import (
    add_indicators,
    classify_trend,
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
