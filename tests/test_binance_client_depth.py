"""Unit tests for the Binance futures client's order-book depth call —
``requests`` is mocked so nothing touches the real network."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from tradingagents.dataflows.errors import NoMarketDataError
from tradingagents.dataflows.vendors.binance import client

pytestmark = pytest.mark.unit


def test_get_order_book_depth_returns_raw_bids_and_asks():
    fake_response = {"bids": [["100.0", "1.5"]], "asks": [["101.0", "2.0"]], "lastUpdateId": 123}
    with patch.object(client, "_get", return_value=fake_response) as mock_get:
        result = client.get_order_book_depth("BTCUSDT", limit=500)
    mock_get.assert_called_once_with("/fapi/v1/depth", {"symbol": "BTCUSDT", "limit": 500})
    assert result == fake_response


def test_get_order_book_depth_rejects_invalid_limit():
    with pytest.raises(ValueError):
        client.get_order_book_depth("BTCUSDT", limit=42)


def test_get_order_book_depth_raises_no_market_data_on_empty_book():
    with patch.object(client, "_get", return_value={"bids": [], "asks": []}):
        with pytest.raises(NoMarketDataError):
            client.get_order_book_depth("BTCUSDT")
