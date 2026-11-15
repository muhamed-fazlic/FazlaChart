"""Tests for the live main strategy (``scanner.live_strategy``). No network:
the Binance client and funding fetch are replaced with in-memory fakes."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from scanner import live_strategy as ls
from scanner.backtesting.funding_simulator import FundingEvent
from scanner.calibration import round_trip_cost_pct_for

pytestmark = pytest.mark.unit

SPEC = ls.build_spec("top")
HOUR_MS = 3_600_000
DAY_MS = 24 * HOUR_MS
SYMBOLS = [f"S{i:02d}USDT" for i in range(12)] + ["BTCUSDT"]


class FakeMarket:
    """Each symbol's 1h price path is linear with its own slope, so the
    14d momentum ranking is fully determined by ``slopes``."""

    def __init__(self, slopes: dict[str, float]):
        self.slopes = slopes
        self.now: datetime = datetime(2026, 11, 2, 12, 0, tzinfo=UTC)
        self.funding_rate = 0.0001

    def price(self, symbol: str, at_ms: int) -> float:
        anchor_ms = int(datetime(2026, 9, 1, tzinfo=UTC).timestamp() * 1000)
        hours = (at_ms - anchor_ms) / HOUR_MS
        return 100.0 * (1.0 + self.slopes[symbol] * hours / 100_000.0)

    def get_klines(self, symbol, interval, limit=300, end_time=None):
        now_ms = int(self.now.timestamp() * 1000)
        step = HOUR_MS if interval == "1h" else DAY_MS
        last_open = now_ms - (now_ms % step)  # current, still-forming candle
        rows = []
        for i in range(limit - 1, -1, -1):
            open_ms = last_open - i * step
            rows.append(
                {
                    "open_time": open_ms,
                    "close_time": open_ms + step - 1,
                    "close": str(self.price(symbol, open_ms + step)),
                    "quote_volume": str(1e9 - SYMBOLS.index(symbol)),
                }
            )
        return rows

    def get_usdt_perpetual_symbols(self):
        return list(SYMBOLS)

    def get_24hr_ticker(self, symbol=None):
        now_ms = int(self.now.timestamp() * 1000)
        return [
            {"symbol": s, "lastPrice": str(self.price(s, now_ms)), "quoteVolume": str(1e9 - SYMBOLS.index(s))}
            for s in SYMBOLS
        ]

    def funding(self, symbol, limit=500):
        start = datetime(2026, 10, 1, tzinfo=UTC)
        return [FundingEvent(timestamp=start + timedelta(hours=8 * k), rate=self.funding_rate) for k in range(200)]


@pytest.fixture
def market(monkeypatch):
    slopes = {s: float(i - 6) for i, s in enumerate(SYMBOLS[:-1])}
    slopes["BTCUSDT"] = 0.5
    fake = FakeMarket(slopes)
    monkeypatch.setattr(ls.client, "get_klines", fake.get_klines)
    monkeypatch.setattr(ls.client, "get_usdt_perpetual_symbols", fake.get_usdt_perpetual_symbols)
    monkeypatch.setattr(ls.client, "get_24hr_ticker", fake.get_24hr_ticker)
    monkeypatch.setattr(ls, "fetch_funding_history", fake.funding)
    return fake


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    yield c
    c.close()


def test_spec_is_read_from_strategy_lab_declarations():
    assert SPEC.template == "momentum_30d"
    assert SPEC.tercile == "top"
    assert SPEC.lookback_hours == 336
    assert SPEC.rebalance_hours == 168
    assert SPEC.basket_size == 5
    assert SPEC.tercile_rank_range == (1, 57)


def test_first_step_opens_long_top5_short_bottom5(market, conn):
    runner = ls.LiveStrategyRunner(conn, SPEC)
    result = runner.step(market.now)
    assert result["rebalanced"] is True

    positions = ls.fetch_open_positions(conn, SPEC.key)
    longs = {p["symbol"] for p in positions if p["direction"] == "long"}
    shorts = {p["symbol"] for p in positions if p["direction"] == "short"}
    # slopes are i-6 for S00..S11 -> strongest are S07..S11, weakest S00..S04
    assert longs == {f"S{i:02d}USDT" for i in range(7, 12)}
    assert shorts == {f"S{i:02d}USDT" for i in range(0, 5)}
    assert all(p["cost_pct"] == pytest.approx(round_trip_cost_pct_for(p["symbol"])) for p in positions)

    reb = ls.fetch_open_rebalance(conn, SPEC.key)
    assert reb["turnover"] == 1.0
    assert len(json.loads(reb["basket_json"])) == 10


def test_not_due_again_before_rebalance_interval(market, conn):
    runner = ls.LiveStrategyRunner(conn, SPEC)
    runner.step(market.now)
    market.now += timedelta(hours=100)
    result = runner.step(market.now)
    assert result["rebalanced"] is False
    assert result["marked"] == 10
    assert len(ls.fetch_rebalances(conn, SPEC.key)) == 1


def test_mark_to_market_tracks_excursion_and_funding(market, conn):
    runner = ls.LiveStrategyRunner(conn, SPEC)
    runner.step(market.now)
    market.now += timedelta(hours=24)
    runner.step(market.now)
    pos = {p["symbol"]: p for p in ls.fetch_open_positions(conn, SPEC.key)}
    top_long = pos["S11USDT"]
    assert top_long["mfe_pct"] > 0  # rising price, long leg
    assert top_long["funding_pct"] > 0  # positive funding: longs pay
    assert pos["S00USDT"]["funding_pct"] < 0  # shorts receive
    view = ls.position_view(top_long)
    assert view["unrealized_net_pct"] == pytest.approx(
        view["unrealized_gross_pct"] - top_long["cost_pct"] - top_long["funding_pct"], abs=1e-3
    )


def test_second_rebalance_keeps_same_side_closes_rest(market, conn):
    runner = ls.LiveStrategyRunner(conn, SPEC)
    runner.step(market.now)
    first = ls.fetch_open_rebalance(conn, SPEC.key)

    # Flip S07 (was a long) into the weakest, S00 (was a short) into the strongest.
    market.slopes["S07USDT"] = -20.0
    market.slopes["S00USDT"] = 20.0
    market.now += timedelta(hours=SPEC.rebalance_hours + 400)  # well past the lookback too
    result = runner.step(market.now)
    assert result["rebalanced"] is True
    summary = result["rebalance"]
    assert "S08USDT" in summary["kept"]
    closed_syms = {c["symbol"] for c in summary["closed"]}
    assert {"S07USDT", "S00USDT"} <= closed_syms

    closed_first = next(r for r in ls.fetch_rebalances(conn, SPEC.key) if r["id"] == first["id"])
    assert closed_first["status"] == "closed"
    assert closed_first["net_return_pct"] == pytest.approx(
        closed_first["gross_return_pct"] - first["cost_pct"] - closed_first["funding_pct"], abs=1e-6
    )

    # A kept leg remains ONE open trade with its original entry.
    kept = [p for p in ls.fetch_open_positions(conn, SPEC.key) if p["symbol"] == "S08USDT"]
    assert len(kept) == 1 and kept[0]["opened_rebalance_id"] == first["id"]

    # Flipped symbols: old leg closed, new leg opened on the other side.
    s07 = ls.fetch_positions(conn, SPEC.key)
    s07 = [p for p in s07 if p["symbol"] == "S07USDT"]
    assert {(p["direction"], p["status"]) for p in s07} == {("long", "closed"), ("short", "open")}
    closed_leg = next(p for p in s07 if p["status"] == "closed")
    assert closed_leg["net_return_pct"] == pytest.approx(
        closed_leg["gross_return_pct"] - closed_leg["cost_pct"] - closed_leg["funding_pct"], abs=1e-6
    )

    new_reb = ls.fetch_open_rebalance(conn, SPEC.key)
    assert 0 < new_reb["turnover"] < 1


def test_summary_aggregates(market, conn):
    runner = ls.LiveStrategyRunner(conn, SPEC)
    runner.step(market.now)
    market.slopes["S07USDT"] = -20.0
    market.now += timedelta(hours=SPEC.rebalance_hours + 1)
    runner.step(market.now)
    s = ls.build_summary(conn, SPEC)
    assert s["open_count"] == 10
    assert s["periods"]["n"] == 1
    assert s["compounded_net_pct"] == pytest.approx(s["periods"]["avg_pct"], abs=1e-3)
    assert s["trades"]["all"]["n"] == s["closed_count"] >= 1
    assert s["next_rebalance_at"] is not None


def test_skips_when_too_few_eligible(market, conn, monkeypatch):
    monkeypatch.setattr(ls.client, "get_usdt_perpetual_symbols", lambda: SYMBOLS[:4])
    runner = ls.LiveStrategyRunner(conn, SPEC)
    result = runner.step(market.now)
    assert result["rebalanced"] is False
    assert ls.fetch_latest_rebalance(conn, SPEC.key)["status"] == "skipped"
    assert ls.fetch_open_positions(conn, SPEC.key) == []


def test_failure_rolls_back_and_backs_off(market, conn, monkeypatch):
    runner = ls.LiveStrategyRunner(conn, SPEC)

    calls = {"n": 0}

    def boom(symbol, base):
        # First 10 calls are basket_cost_pct (before any write); the 11th is
        # inside the transaction, after the rebalance row was inserted.
        calls["n"] += 1
        if calls["n"] > 10:
            raise RuntimeError("fails mid-transaction")
        return round_trip_cost_pct_for(symbol, base)

    monkeypatch.setattr(ls, "round_trip_cost_pct_for", boom)
    runner.step(market.now)
    assert ls.fetch_rebalances(conn, SPEC.key) == []
    assert ls.fetch_positions(conn, SPEC.key) == []
    # within the back-off window it doesn't hammer the API again
    assert runner.is_rebalance_due(market.now + timedelta(minutes=5)) is False
    assert runner.is_rebalance_due(market.now + ls.RETRY_AFTER_FAILURE) is True


def test_ranking_ignores_still_forming_candle(market):
    now = market.now + timedelta(minutes=30)
    market.now = now
    h = ls._closed_history("S11USDT", "1h", 50, now)
    assert h is not None
    assert h["open_time"].iloc[-1] + timedelta(hours=1) <= now


def test_rebalance_message_mentions_legs(market, conn):
    runner = ls.LiveStrategyRunner(conn, SPEC)
    summary = runner.step(market.now)["rebalance"]
    msg = ls.format_rebalance_message(summary)
    assert "S11USDT" in msg and "S00USDT" in msg and "Eksploratorno" in msg


def test_main_strategy_ranks_whole_scanner_universe(market, conn, monkeypatch):
    assert ls.MAIN_STRATEGY.tercile == "all" and ls.MAIN_STRATEGY.tercile_rank_range is None
    with pytest.raises(ValueError):
        ls.LiveStrategyRunner(conn)  # whole-universe spec needs a provider

    # No liquidity ranking at all in this mode.
    monkeypatch.setattr(ls, "_avg_daily_quote_volume", lambda *a: pytest.fail("should not rank by volume"))
    scanned = [s for s in SYMBOLS if s != "S05USDT"]
    runner = ls.LiveStrategyRunner(conn, universe_provider=lambda: scanned)
    summary = runner.step(market.now)["rebalance"]
    assert summary["universe_size"] == len(scanned)
    assert "S05USDT" not in summary["basket"]
    assert ls.fetch_latest_rebalance(conn, ls.MAIN_STRATEGY.key) is not None
