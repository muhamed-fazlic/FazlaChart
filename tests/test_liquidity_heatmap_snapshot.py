"""Unit tests for the liquidity-heatmap snapshot builder's real order-book
augmentation — klines/depth are mocked, nothing touches the network."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from tradingagents.dataflows.errors import VendorUnavailableError
from tradingagents.dataflows.vendors.binance.snapshot import build_liquidity_heatmap_snapshot

pytestmark = pytest.mark.unit

_KLINE_COLUMNS = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
)

_KLINES = [
    dict(
        zip(
            _KLINE_COLUMNS,
            [1700000000000 + i * 3600000, "100", "101", "99", str(100 + i * 0.1), "1000", 0, "0", 0, "0", "0", "0"],
            strict=False,
        )
    )
    for i in range(50)
]

_DEPTH = {"bids": [["99.0", "500"], ["98.0", "1"]], "asks": [["101.0", "600"], ["102.0", "1"]]}


def test_heatmap_snapshot_includes_orderbook_walls_when_depth_available():
    with (
        patch(
            "tradingagents.dataflows.vendors.binance.snapshot.client.get_klines",
            return_value=_KLINES,
        ),
        patch(
            "tradingagents.dataflows.vendors.binance.snapshot.client.get_order_book_depth",
            return_value=_DEPTH,
        ),
    ):
        result = build_liquidity_heatmap_snapshot("BTCUSDT", timeframe="1h")

    assert result["orderbook_walls"] is not None
    assert "orderbook_walls_error" not in result


def test_heatmap_snapshot_degrades_gracefully_when_depth_unavailable():
    with (
        patch(
            "tradingagents.dataflows.vendors.binance.snapshot.client.get_klines",
            return_value=_KLINES,
        ),
        patch(
            "tradingagents.dataflows.vendors.binance.snapshot.client.get_order_book_depth",
            side_effect=VendorUnavailableError("depth endpoint down"),
        ),
    ):
        result = build_liquidity_heatmap_snapshot("BTCUSDT", timeframe="1h")

    assert result["orderbook_walls"] is None
    assert "orderbook_walls_error" in result
    # The OHLCV proxy itself must still be present and usable.
    assert "poc_price" in result
