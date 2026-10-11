"""Unit tests for single-symbol anomaly detection. Synthetic data only."""

from __future__ import annotations

import pytest

from tradingagents.dataflows.vendors.binance.anomaly import (
    detect_anomalies,
    failed_breakout,
    oi_rising_price_flat_funding_spiking,
    price_up_volume_not_confirming,
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


def test_detect_anomalies_aggregates_all_checks():
    closes = [100.0] * 25 + [101.0, 102.5, 104.0]
    volumes = [1000.0] * 25 + [900.0, 950.0, 1000.0]
    df = add_indicators(klines_to_dataframe(_make_klines(closes, volumes)))
    anomalies = detect_anomalies(df, resistance=999.0, funding_rate=0.0, oi_change_pct_60m=None)
    assert "price_up_volume_not_confirming" in anomalies
