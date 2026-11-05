"""Unit tests for Phase 8 Korak 5's funding-cost simulator. No network
calls — ``fetch_funding_history`` is exercised against a monkeypatched
client, and ``funding_cost_pct`` against hand-built ``FundingEvent`` lists.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from scanner.backtesting.funding_simulator import (
    FundingEvent,
    fetch_funding_history,
    funding_cost_pct,
)
from tradingagents.dataflows.errors import VendorError

pytestmark = pytest.mark.unit


def _dt(hour: int) -> datetime:
    return datetime(2026, 1, 1, tzinfo=UTC) + timedelta(hours=hour)


class TestFetchFundingHistory:
    def test_parses_and_sorts_rows_oldest_first(self, monkeypatch):
        from scanner.backtesting import funding_simulator as module

        raw = [
            {"fundingTime": 1767225600000, "fundingRate": "0.0001"},  # 2026-01-01 00:00 UTC
            {"fundingTime": 1767196800000, "fundingRate": "-0.0002"},  # 2025-12-31 16:00 UTC
        ]
        monkeypatch.setattr(module.client, "get_funding_rate_history", lambda symbol, limit=1000: raw)
        events = fetch_funding_history("BTCUSDT")
        assert [e.rate for e in events] == [-0.0002, 0.0001]
        assert events[0].timestamp < events[1].timestamp

    def test_returns_empty_list_on_vendor_error(self, monkeypatch):
        from scanner.backtesting import funding_simulator as module

        def raise_vendor_error(symbol, limit=1000):
            raise VendorError("boom")

        monkeypatch.setattr(module.client, "get_funding_rate_history", raise_vendor_error)
        assert fetch_funding_history("BTCUSDT") == []

    def test_skips_malformed_rows_without_raising(self, monkeypatch):
        from scanner.backtesting import funding_simulator as module

        raw = [
            {"fundingTime": 1767225600000, "fundingRate": "0.0001"},
            {"fundingTime": "not-a-number", "fundingRate": "0.0002"},
            {"missing": "fields"},
        ]
        monkeypatch.setattr(module.client, "get_funding_rate_history", lambda symbol, limit=1000: raw)
        events = fetch_funding_history("BTCUSDT")
        assert len(events) == 1
        assert events[0].rate == 0.0001


class TestFundingCostPct:
    def test_no_events_is_zero_cost(self):
        assert funding_cost_pct("long", _dt(0), _dt(8), []) == 0.0

    def test_entry_after_or_equal_exit_is_zero_cost(self):
        events = [FundingEvent(timestamp=_dt(4), rate=0.0001)]
        assert funding_cost_pct("long", _dt(8), _dt(8), events) == 0.0
        assert funding_cost_pct("long", _dt(9), _dt(8), events) == 0.0

    def test_settlement_exactly_at_entry_is_not_charged(self):
        # Position wasn't open yet at that instant.
        events = [FundingEvent(timestamp=_dt(0), rate=0.0001)]
        assert funding_cost_pct("long", _dt(0), _dt(8), events) == 0.0

    def test_settlement_exactly_at_exit_is_charged(self):
        # Still open and holding exposure at that instant.
        events = [FundingEvent(timestamp=_dt(8), rate=0.0001)]
        assert funding_cost_pct("long", _dt(0), _dt(8), events) == pytest.approx(0.01)

    def test_long_pays_positive_funding_rate(self):
        events = [FundingEvent(timestamp=_dt(4), rate=0.0003)]
        # 0.0003 * 100 = 0.03 percentage points of cost
        assert funding_cost_pct("long", _dt(0), _dt(8), events) == pytest.approx(0.03)

    def test_short_is_credited_by_positive_funding_rate(self):
        events = [FundingEvent(timestamp=_dt(4), rate=0.0003)]
        assert funding_cost_pct("short", _dt(0), _dt(8), events) == pytest.approx(-0.03)

    def test_short_pays_negative_funding_rate(self):
        events = [FundingEvent(timestamp=_dt(4), rate=-0.0003)]
        assert funding_cost_pct("short", _dt(0), _dt(8), events) == pytest.approx(0.03)

    def test_multiple_settlements_within_window_sum(self):
        events = [
            FundingEvent(timestamp=_dt(4), rate=0.0001),
            FundingEvent(timestamp=_dt(12), rate=0.0002),
            FundingEvent(timestamp=_dt(20), rate=0.0001),
        ]
        # Trade spans 3 settlements (hours 4, 12, 20); a fourth at hour 28 is
        # outside the window and must not be counted.
        events.append(FundingEvent(timestamp=_dt(28), rate=10.0))
        result = funding_cost_pct("long", _dt(0), _dt(24), events)
        assert result == pytest.approx((0.0001 + 0.0002 + 0.0001) * 100.0)

    def test_settlement_before_entry_not_counted(self):
        events = [FundingEvent(timestamp=datetime(2025, 12, 31, 23, 0, tzinfo=UTC), rate=5.0)]
        result = funding_cost_pct("long", _dt(0), _dt(8), events)
        assert result == 0.0
