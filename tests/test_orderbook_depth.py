"""Unit tests for real order-book liquidity-wall detection. Pure math on a
hand-built depth snapshot -- no network calls.
"""

from __future__ import annotations

import pytest

from tradingagents.dataflows.vendors.binance.orderbook_depth import compute_orderbook_walls

pytestmark = pytest.mark.unit


def _depth(bids, asks):
    return {"bids": [[str(p), str(q)] for p, q in bids], "asks": [[str(p), str(q)] for p, q in asks]}


def test_flags_a_bid_level_with_much_larger_size_than_its_neighbors():
    bids = [(99.0, 1.0), (98.5, 1.1), (98.0, 50.0), (97.5, 0.9), (97.0, 1.0)]
    asks = [(100.0, 1.0), (100.5, 1.0)]
    result = compute_orderbook_walls(_depth(bids, asks), current_price=99.5)
    assert len(result["bid_walls"]) == 1
    assert result["bid_walls"][0]["price"] == 98.0
    assert result["bid_walls"][0]["side"] == "bid"
    assert result["bid_walls"][0]["notional_usdt"] == pytest.approx(98.0 * 50.0, rel=1e-6)


def test_quiet_evenly_sized_book_reports_no_walls():
    bids = [(99.0, 1.0), (98.5, 1.0), (98.0, 1.0)]
    asks = [(100.0, 1.0), (100.5, 1.0), (101.0, 1.0)]
    result = compute_orderbook_walls(_depth(bids, asks), current_price=99.5)
    assert result["bid_walls"] == []
    assert result["ask_walls"] == []


def test_distance_pct_is_signed_relative_to_current_price():
    bids = [(90.0, 10.0), (89.5, 0.1), (89.0, 0.1)]
    asks = [(110.0, 10.0), (110.5, 0.1), (111.0, 0.1)]
    result = compute_orderbook_walls(_depth(bids, asks), current_price=100.0)
    bid_wall = result["bid_walls"][0]
    ask_wall = result["ask_walls"][0]
    assert bid_wall["distance_pct"] == pytest.approx(-10.0)
    assert ask_wall["distance_pct"] == pytest.approx(10.0)


def test_best_bid_and_ask_reported_from_top_of_book():
    bids = [(99.0, 1.0), (98.0, 1.0)]
    asks = [(101.0, 1.0), (102.0, 1.0)]
    result = compute_orderbook_walls(_depth(bids, asks), current_price=100.0)
    assert result["best_bid"] == 99.0
    assert result["best_ask"] == 101.0


def test_empty_side_does_not_crash():
    result = compute_orderbook_walls(_depth([], []), current_price=100.0)
    assert result["bid_walls"] == []
    assert result["ask_walls"] == []
    assert result["best_bid"] is None
    assert result["best_ask"] is None


def test_top_n_limits_number_of_walls_returned():
    small_bids = [(100.0 - i, 1.0) for i in range(7)]  # evenly small -> sets a low median
    big_bids = [(90.0, 50.0), (89.0, 55.0), (88.0, 60.0), (87.0, 65.0)]  # clear outlier walls
    result = compute_orderbook_walls(_depth(small_bids + big_bids, []), current_price=100.0, top_n=3)
    assert len(result["bid_walls"]) == 3
