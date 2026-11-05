"""Unit tests for the `scanner.calibrate` standalone CLI's funding-cost and
SL/TP tie-break flags (audit follow-up: these features were already
implemented/tested in `scanner.calibration` but were only reachable from
`scanner.backtesting.portfolio_backtest`, not from this simpler standalone
tool -- see PHASE8_PROFITABILITY_RESEARCH_TODO.md)."""

from __future__ import annotations

import pytest

from scanner import calibrate

pytestmark = pytest.mark.unit


def test_default_never_passes_extra_kwargs_to_backtest_symbol(monkeypatch):
    received = []

    def fake_backtest_symbol(symbol, timeframe, horizons, limit, **kwargs):
        received.append(kwargs)
        return []

    monkeypatch.setattr(calibrate, "backtest_symbol", fake_backtest_symbol)
    calibrate.main(["--symbols", "BTCUSDT", "--timeframe", "1h"])

    assert received == [{}]


def test_include_funding_cost_fetches_and_forwards_funding_events(monkeypatch):
    sentinel = ["fake-funding-events"]
    fetch_calls = []

    def fake_fetch(symbol, limit=1000):
        fetch_calls.append(symbol)
        return sentinel

    received = []

    def fake_backtest_symbol(symbol, timeframe, horizons, limit, **kwargs):
        received.append(kwargs)
        return []

    monkeypatch.setattr(calibrate, "fetch_funding_history", fake_fetch)
    monkeypatch.setattr(calibrate, "backtest_symbol", fake_backtest_symbol)
    calibrate.main(["--symbols", "BTCUSDT", "--include-funding-cost"])

    assert fetch_calls == ["BTCUSDT"]
    assert received == [{"funding_events": sentinel}]


def test_optimistic_tie_break_is_forwarded(monkeypatch):
    received = []

    def fake_backtest_symbol(symbol, timeframe, horizons, limit, **kwargs):
        received.append(kwargs)
        return []

    monkeypatch.setattr(calibrate, "backtest_symbol", fake_backtest_symbol)
    calibrate.main(["--symbols", "BTCUSDT", "--sl-tp-tie-break", "optimistic"])

    assert received == [{"sl_tp_tie_break": "optimistic"}]
