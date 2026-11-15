"""Live (forward) execution of the Strategy Lab's #1-ranked candidate as the
Market Scanner's MAIN strategy, with its own trade tracking.

Currently: ``momentum_30d`` / "14d lookback, N=5" -- cross-sectional
momentum, dollar-neutral: every ``rebalance_hours`` (168h, weekly) rank
EVERY coin the scanner tracks (universe ``"all"``, owner's choice; the Lab's
#1 was measured on the ``top`` liquidity tercile only -- that variant is
still available via ``build_spec("top")``) by its trailing ``lookback_hours``
(336h, 14d) % price change, go LONG the top ``basket_size`` (5) and SHORT
the bottom 5, hold until the next rebalance.

Same mechanics as the Layer 1 backtest it was mined from -- the ranking
(``cross_sectional_engine.momentum_ranking_fn``), basket formation
(``form_basket``), leg weight (``_leg_weight``), turnover cost rule
(``changed_symbols_and_turnover``), per-symbol cost model
(``round_trip_cost_pct_for``) and funding sign convention
(``funding_cost_pct``) are all imported, never reimplemented, so the live
track record measures the same thing the leaderboard ranked.

IMPORTANT, and shown on the dashboard: this candidate is a Layer 1
*exploratory* rank (no CI, no Layer 2 confirmation). Running it live is a
forward test, not a deployment of a proven edge.

Deliberate live-vs-backtest differences (documented, not hidden):

* Ranking uses only fully CLOSED 1h candles (the backtest reads the close of
  the candle opening at ``t``; live, that candle hasn't closed yet).
* Entry/exit fill = Binance 24h-ticker ``lastPrice`` at rebalance time, not
  a candle close.
* (``"top"`` variant only) Liquidity tercile = ranks 1-57 by trailing 30d average daily quote volume
  (same definition as ``historical_universe``), but only the top
  ``LIQUIDITY_PREFILTER_N`` symbols by 24h quote volume are ranked, to avoid
  ~500 kline requests per rebalance. A symbol outside the 24h top-150 being
  inside the 30d top-57 is possible but rare.
* The first rebalance happens as soon as the scanner starts; later ones
  every 168h after the previous one (the backtest grid is anchored to its
  earliest candle, which has no live equivalent).

Two levels of record, both in ``scanner.db``:

* ``live_strategy_positions`` -- one row per LEG held from entry to exit
  (a symbol kept in the same leg across rebalances stays ONE trade). This
  is what "Analitika trejdova" shows per trade.
* ``live_strategy_rebalances`` -- one row per holding period, with the
  basket-level return computed exactly like a backtest ``RebalancePeriod``
  (weights, turnover cost, funding), so live periods are directly
  comparable to the leaderboard's per-period numbers.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd

from scanner.backtesting import cross_sectional_engine as cse
from scanner.backtesting.cross_sectional_momentum_backtest import (
    BTC_SYMBOL,
    _leg_weight,
    changed_symbols_and_turnover,
)
from scanner.backtesting.funding_simulator import (
    FundingEvent,
    fetch_funding_history,
    funding_cost_pct,
)
from scanner.calibration import DEFAULT_ROUND_TRIP_COST_PCT, round_trip_cost_pct_for
from tradingagents.dataflows.errors import VendorError
from tradingagents.dataflows.vendors.binance import client

logger = logging.getLogger(__name__)

LIQUIDITY_PREFILTER_N = 150
VOLUME_LOOKBACK_DAYS = 30
VOLUME_MIN_HISTORY_DAYS = 7
# Extra 1h candles fetched beyond the lookback so the "price lookback_hours
# ago" lookup always has a candle at or before it.
KLINE_BUFFER_CANDLES = 48
# A symbol whose newest CLOSED 1h candle is older than this is treated as
# stale (halted/delisting) and excluded from ranking.
MAX_CANDLE_STALENESS = timedelta(hours=3)
# After a failed rebalance attempt (vendor outage etc.) wait this long
# before retrying, instead of re-running the expensive fetch every cycle.
RETRY_AFTER_FAILURE = timedelta(minutes=15)
# Open legs' accrued funding is refreshed at most this often.
FUNDING_REFRESH_INTERVAL = timedelta(hours=1)
FUNDING_HISTORY_LIMIT = 500


@dataclass(frozen=True)
class LiveStrategySpec:
    key: str
    template: str
    tercile: str
    variant_label: str
    lookback_hours: int
    rebalance_hours: int
    basket_size: int
    # None = rank the whole universe the scanner currently tracks (passed in
    # via ``LiveStrategyRunner(universe_provider=...)``) instead of a
    # liquidity tercile.
    tercile_rank_range: tuple[int, int] | None
    price_timeframe: str = "1h"
    base_cost_pct: float = DEFAULT_ROUND_TRIP_COST_PCT

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "template": self.template,
            "tercile": self.tercile,
            "variant_label": self.variant_label,
            "lookback_hours": self.lookback_hours,
            "rebalance_hours": self.rebalance_hours,
            "basket_size": self.basket_size,
            "tercile_rank_range": list(self.tercile_rank_range) if self.tercile_rank_range else None,
            "price_timeframe": self.price_timeframe,
            "base_cost_pct": self.base_cost_pct,
        }


def build_spec(tercile: str = "all") -> LiveStrategySpec:
    """Parameters are read from the Strategy Lab's own declarations
    (``MOMENTUM_30D_VARIANTS``, ``DEFAULT_TERCILE_RANK_RANGES``), not
    re-typed, so the live strategy can't silently drift from what was mined.

    ``tercile="all"`` (the main strategy, owner's choice 2026-11-15) ranks
    every coin the scanner tracks instead of only the top liquidity tercile
    the Lab ranked #1. NOTE: that is a wider universe than the one the Lab
    measured -- the Lab's 1.33%/period applies to ``"top"`` only."""
    from scanner.backtesting.strategy_miner import (
        DEFAULT_TERCILE_RANK_RANGES,
        MOMENTUM_30D_VARIANTS,
    )

    template, label = "momentum_30d", "14d lookback, N=5"
    variant = next(v for v in MOMENTUM_30D_VARIANTS if v.variant_label == label)
    return LiveStrategySpec(
        key=f"{template}|{tercile}|{label}",
        template=template,
        tercile=tercile,
        variant_label=label,
        lookback_hours=variant.params["lookback_hours"],
        rebalance_hours=variant.params["rebalance_hours"],
        basket_size=variant.params["basket_size"],
        tercile_rank_range=None if tercile == "all" else DEFAULT_TERCILE_RANK_RANGES[tercile],
    )


MAIN_STRATEGY: LiveStrategySpec = build_spec("all")


# --- Storage --------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS live_strategy_rebalances (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_key TEXT NOT NULL,
    rebalanced_at TEXT NOT NULL,
    status TEXT NOT NULL,              -- open | closed | skipped
    note TEXT,
    universe_size INTEGER,
    n_eligible INTEGER,
    basket_json TEXT,                  -- {symbol: {side, price, score, rank}}
    turnover REAL,
    cost_pct REAL,
    btc_entry_price REAL,
    closed_at TEXT,
    gross_return_pct REAL,
    funding_pct REAL,
    net_return_pct REAL,
    btc_return_pct REAL
);
CREATE INDEX IF NOT EXISTS idx_lsr_strategy_time ON live_strategy_rebalances (strategy_key, rebalanced_at);

CREATE TABLE IF NOT EXISTS live_strategy_positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_key TEXT NOT NULL,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL,           -- long | short
    status TEXT NOT NULL,              -- open | closed
    opened_at TEXT NOT NULL,
    opened_rebalance_id INTEGER NOT NULL,
    entry_price REAL NOT NULL,
    entry_score REAL,                  -- trailing momentum % at entry
    entry_rank INTEGER,                -- 1 = strongest; N = weakest in the ranking
    cost_pct REAL NOT NULL,            -- full round-trip cost estimate for this leg
    last_price REAL,
    last_price_at TEXT,
    funding_pct REAL NOT NULL DEFAULT 0,  -- accrued (sign-adjusted cost, + = paid)
    funding_updated_at TEXT,
    mfe_pct REAL NOT NULL DEFAULT 0,
    mae_pct REAL NOT NULL DEFAULT 0,
    closed_at TEXT,
    closed_rebalance_id INTEGER,
    exit_price REAL,
    gross_return_pct REAL,
    net_return_pct REAL
);
CREATE INDEX IF NOT EXISTS idx_lsp_strategy_status ON live_strategy_positions (strategy_key, status);
"""


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)
    conn.commit()


def tables_exist(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name IN"
        " ('live_strategy_rebalances', 'live_strategy_positions')"
    ).fetchone()
    return bool(row and row[0] == 2)


def _rows(cursor: sqlite3.Cursor) -> list[dict[str, Any]]:
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, r, strict=False)) for r in cursor.fetchall()]


def fetch_latest_rebalance(conn: sqlite3.Connection, strategy_key: str) -> dict[str, Any] | None:
    rows = _rows(
        conn.execute(
            "SELECT * FROM live_strategy_rebalances WHERE strategy_key = ? ORDER BY rebalanced_at DESC, id DESC LIMIT 1",
            (strategy_key,),
        )
    )
    return rows[0] if rows else None


def fetch_open_rebalance(conn: sqlite3.Connection, strategy_key: str) -> dict[str, Any] | None:
    rows = _rows(
        conn.execute(
            "SELECT * FROM live_strategy_rebalances WHERE strategy_key = ? AND status = 'open'"
            " ORDER BY rebalanced_at DESC, id DESC LIMIT 1",
            (strategy_key,),
        )
    )
    return rows[0] if rows else None


def fetch_rebalances(conn: sqlite3.Connection, strategy_key: str, limit: int = 200) -> list[dict[str, Any]]:
    return _rows(
        conn.execute(
            "SELECT * FROM live_strategy_rebalances WHERE strategy_key = ? ORDER BY rebalanced_at DESC, id DESC LIMIT ?",
            (strategy_key, limit),
        )
    )


def fetch_open_positions(conn: sqlite3.Connection, strategy_key: str) -> list[dict[str, Any]]:
    return _rows(
        conn.execute(
            "SELECT * FROM live_strategy_positions WHERE strategy_key = ? AND status = 'open' ORDER BY direction, entry_rank",
            (strategy_key,),
        )
    )


def fetch_positions(
    conn: sqlite3.Connection,
    strategy_key: str,
    *,
    status: str | None = None,
    since: str | None = None,
    until: str | None = None,
    limit: int = 2000,
) -> list[dict[str, Any]]:
    clauses = ["strategy_key = ?"]
    params: list[Any] = [strategy_key]
    if status in ("open", "closed"):
        clauses.append("status = ?")
        params.append(status)
    if since is not None:
        clauses.append("opened_at >= ?")
        params.append(since)
    if until is not None:
        clauses.append("opened_at <= ?")
        params.append(until)
    params.append(limit)
    return _rows(
        conn.execute(
            f"SELECT * FROM live_strategy_positions WHERE {' AND '.join(clauses)} ORDER BY opened_at DESC, id DESC LIMIT ?",
            params,
        )
    )


# --- Pure computation (unit-tested without network) -----------------------


def directional_return_pct(direction: str, entry_price: float, price: float) -> float:
    raw = (price - entry_price) / entry_price * 100.0
    return raw if direction == "long" else -raw


def rank_symbols(scores: dict[str, float]) -> dict[str, int]:
    """1 = highest score (strongest momentum)."""
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return {symbol: i + 1 for i, (symbol, _) in enumerate(ordered)}


def compute_basket(
    spec: LiveStrategySpec, histories: dict[str, pd.DataFrame], now: datetime
) -> tuple[dict[str, str] | None, dict[str, float]]:
    """Scores every symbol with the backtest's own ranking function and
    forms the basket with the backtest's own ``form_basket``. ``histories``
    must contain CLOSED candles only."""
    config = cse.CrossSectionalConfig(
        hypothesis=spec.template,
        ranking_fn=cse.momentum_ranking_fn,
        lookback_hours=spec.lookback_hours,
        rebalance_hours=spec.rebalance_hours,
        basket_size=spec.basket_size,
    )
    scores = cse.momentum_ranking_fn(now, histories, {}, spec.lookback_hours)
    return cse.form_basket(scores, config), scores


def compute_period_return(
    spec: LiveStrategySpec,
    basket: dict[str, dict[str, Any]],
    exit_prices: dict[str, float],
    funding_by_symbol: dict[str, list[FundingEvent]],
    entry_time: datetime,
    exit_time: datetime,
    cost_pct: float,
) -> dict[str, float] | None:
    """Basket-level return for one holding period, same formula as
    ``cross_sectional_engine.build_rebalance_period``. ``None`` if any leg
    lacks an exit price (can't be valued honestly)."""
    weight = _leg_weight(spec.basket_size)
    gross = 0.0
    funding = 0.0
    for symbol, leg in basket.items():
        exit_price = exit_prices.get(symbol)
        if exit_price is None or not leg.get("price"):
            return None
        gross += weight * directional_return_pct(leg["side"], leg["price"], exit_price)
        funding += weight * funding_cost_pct(leg["side"], entry_time, exit_time, funding_by_symbol.get(symbol, []))
    return {
        "gross_return_pct": round(gross, 6),
        "funding_pct": round(funding, 6),
        "net_return_pct": round(gross - cost_pct - funding, 6),
    }


def basket_cost_pct(spec: LiveStrategySpec, prev_sides: dict[str, str] | None, new_sides: dict[str, str]) -> tuple[float, float]:
    changed, turnover = changed_symbols_and_turnover(prev_sides, new_sides)
    weight = _leg_weight(spec.basket_size)
    cost = sum(weight * round_trip_cost_pct_for(s, spec.base_cost_pct) for s in changed)
    return round(cost, 6), round(turnover, 6)


# --- Live runner ----------------------------------------------------------


def _iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat()


def _parse(iso: str) -> datetime:
    dt = datetime.fromisoformat(iso)
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def _closed_history(symbol: str, timeframe: str, n_candles: int, now: datetime) -> pd.DataFrame | None:
    raw = client.get_klines(symbol, timeframe, limit=min(1500, n_candles))
    df = pd.DataFrame(raw)
    if df.empty:
        return None
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df["close"] = df["close"].astype(float)
    df = df[df["close_time"] < pd.Timestamp(now)]
    if df.empty or pd.Timestamp(now) - df["close_time"].iloc[-1] > MAX_CANDLE_STALENESS:
        return None
    return df[["open_time", "close"]].reset_index(drop=True)


def _avg_daily_quote_volume(symbol: str, now: datetime) -> float | None:
    """Same definition as ``historical_universe._avg_daily_quote_volume``:
    trailing daily candles that CLOSED before ``now``."""
    try:
        klines = client.get_klines(symbol, "1d", limit=VOLUME_LOOKBACK_DAYS + 1)
    except VendorError:
        return None
    now_ms = int(now.timestamp() * 1000)
    closed = [k for k in klines if int(k["close_time"]) < now_ms][-VOLUME_LOOKBACK_DAYS:]
    if len(closed) < VOLUME_MIN_HISTORY_DAYS:
        return None
    return sum(float(k["quote_volume"]) for k in closed) / len(closed)


class LiveStrategyRunner:
    """Called once per scanner cycle via :meth:`step`. Never raises."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        spec: LiveStrategySpec = MAIN_STRATEGY,
        on_rebalance: Callable[[dict[str, Any]], None] | None = None,
        universe_provider: Callable[[], list[str]] | None = None,
    ):
        if spec.tercile_rank_range is None and universe_provider is None:
            raise ValueError("a whole-universe spec needs a universe_provider")
        self.conn = conn
        self.spec = spec
        self.on_rebalance = on_rebalance
        self.universe_provider = universe_provider
        self._last_failed_attempt: datetime | None = None
        ensure_schema(conn)

    # -- scheduling --

    def next_rebalance_at(self) -> datetime | None:
        latest = fetch_latest_rebalance(self.conn, self.spec.key)
        if latest is None:
            return None
        return _parse(latest["rebalanced_at"]) + timedelta(hours=self.spec.rebalance_hours)

    def is_rebalance_due(self, now: datetime) -> bool:
        if self._last_failed_attempt is not None and now - self._last_failed_attempt < RETRY_AFTER_FAILURE:
            return False
        due_at = self.next_rebalance_at()
        return due_at is None or now >= due_at

    def step(self, now: datetime | None = None) -> dict[str, Any]:
        now = now or datetime.now(UTC)
        result: dict[str, Any] = {"rebalanced": False, "marked": 0}
        try:
            prices = self._fetch_last_prices()
        except Exception:
            logger.exception("Live strategy: could not fetch prices this cycle")
            return result
        try:
            if self.is_rebalance_due(now):
                summary = self.rebalance(now, prices)
                result["rebalanced"] = summary is not None
                result["rebalance"] = summary
            result["marked"] = self.mark_to_market(now, prices)
        except Exception:
            self._last_failed_attempt = now
            logger.exception("Live strategy step failed; will retry")
        return result

    # -- data --

    def _fetch_last_prices(self) -> dict[str, float]:
        tickers = client.get_24hr_ticker()
        return {t["symbol"]: float(t["lastPrice"]) for t in tickers if t.get("lastPrice")}

    def _tercile_universe(self, now: datetime) -> list[str]:
        perpetuals = set(client.get_usdt_perpetual_symbols())
        tickers = client.get_24hr_ticker()
        prefilter = sorted(
            (t for t in tickers if t.get("symbol") in perpetuals),
            key=lambda t: float(t.get("quoteVolume", 0.0)),
            reverse=True,
        )[:LIQUIDITY_PREFILTER_N]
        ranked: list[tuple[str, float]] = []
        for t in prefilter:
            vol = _avg_daily_quote_volume(t["symbol"], now)
            if vol is not None:
                ranked.append((t["symbol"], vol))
        ranked.sort(key=lambda kv: kv[1], reverse=True)
        lo, hi = self.spec.tercile_rank_range
        return [s for s, _ in ranked[lo - 1 : hi]]

    def _histories(self, symbols: list[str], now: datetime) -> dict[str, pd.DataFrame]:
        n = self.spec.lookback_hours + KLINE_BUFFER_CANDLES
        out: dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            try:
                h = _closed_history(symbol, self.spec.price_timeframe, n, now)
            except VendorError as exc:
                logger.warning("Live strategy: skipping %s (%s)", symbol, exc)
                continue
            if h is not None:
                out[symbol] = h
        return out

    # -- rebalance --

    def rebalance(self, now: datetime, prices: dict[str, float]) -> dict[str, Any] | None:
        if self.spec.tercile_rank_range is None:
            universe = sorted(set(self.universe_provider()))
        else:
            universe = self._tercile_universe(now)
        histories = self._histories(universe, now)
        basket, scores = compute_basket(self.spec, histories, now)
        if basket is None:
            self.conn.execute(
                "INSERT INTO live_strategy_rebalances (strategy_key, rebalanced_at, status, note, universe_size, n_eligible)"
                " VALUES (?, ?, 'skipped', ?, ?, ?)",
                (self.spec.key, _iso(now), "premalo simbola sa dovoljno istorije", len(universe), len(scores)),
            )
            self.conn.commit()
            logger.warning("Live strategy: rebalance skipped (%d eligible)", len(scores))
            return None
        missing = [s for s in basket if s not in prices]
        if missing:
            raise RuntimeError(f"no live price for basket symbols {missing}")

        ranks = rank_symbols(scores)
        prev = fetch_open_rebalance(self.conn, self.spec.key)
        prev_basket: dict[str, dict[str, Any]] = json.loads(prev["basket_json"]) if prev else {}
        prev_sides = {s: leg["side"] for s, leg in prev_basket.items()} if prev else None
        open_positions = fetch_open_positions(self.conn, self.spec.key)

        # Funding for everything we're about to value (previous basket + legs we close).
        need_funding = set(prev_basket) | {p["symbol"] for p in open_positions}
        funding = {s: fetch_funding_history(s, FUNDING_HISTORY_LIMIT) for s in need_funding}

        cost_pct, turnover = basket_cost_pct(self.spec, prev_sides, basket)
        new_basket = {
            s: {"side": side, "price": prices[s], "score": round(scores[s], 4), "rank": ranks[s]}
            for s, side in basket.items()
        }
        now_iso = _iso(now)
        cur = self.conn.cursor()
        try:
            # 1) close the previous period
            if prev is not None:
                period = compute_period_return(
                    self.spec, prev_basket, prices, funding, _parse(prev["rebalanced_at"]), now, prev["cost_pct"] or 0.0
                )
                btc_ret = None
                if prev["btc_entry_price"] and prices.get(BTC_SYMBOL):
                    btc_ret = round((prices[BTC_SYMBOL] / prev["btc_entry_price"] - 1.0) * 100.0, 6)
                cur.execute(
                    "UPDATE live_strategy_rebalances SET status='closed', closed_at=?, gross_return_pct=?,"
                    " funding_pct=?, net_return_pct=?, btc_return_pct=? WHERE id=?",
                    (
                        now_iso,
                        period["gross_return_pct"] if period else None,
                        period["funding_pct"] if period else None,
                        period["net_return_pct"] if period else None,
                        btc_ret,
                        prev["id"],
                    ),
                )

            # 2) open the new period
            cur.execute(
                "INSERT INTO live_strategy_rebalances (strategy_key, rebalanced_at, status, universe_size, n_eligible,"
                " basket_json, turnover, cost_pct, btc_entry_price) VALUES (?, ?, 'open', ?, ?, ?, ?, ?, ?)",
                (
                    self.spec.key,
                    now_iso,
                    len(universe),
                    len(scores),
                    json.dumps(new_basket),
                    turnover,
                    cost_pct,
                    prices.get(BTC_SYMBOL),
                ),
            )
            rebalance_id = cur.lastrowid

            # 3) legs: keep same-side holdings, close the rest, open new ones
            kept: set[str] = set()
            closed: list[dict[str, Any]] = []
            for pos in open_positions:
                if basket.get(pos["symbol"]) == pos["direction"]:
                    kept.add(pos["symbol"])
                    continue
                exit_price = prices.get(pos["symbol"], pos["last_price"] or pos["entry_price"])
                gross = directional_return_pct(pos["direction"], pos["entry_price"], exit_price)
                fund = funding_cost_pct(pos["direction"], _parse(pos["opened_at"]), now, funding.get(pos["symbol"], []))
                net = gross - pos["cost_pct"] - fund
                cur.execute(
                    "UPDATE live_strategy_positions SET status='closed', closed_at=?, closed_rebalance_id=?,"
                    " exit_price=?, last_price=?, last_price_at=?, gross_return_pct=?, funding_pct=?,"
                    " funding_updated_at=?, net_return_pct=? WHERE id=?",
                    (now_iso, rebalance_id, exit_price, exit_price, now_iso, round(gross, 6), round(fund, 6),
                     now_iso, round(net, 6), pos["id"]),
                )
                closed.append({"symbol": pos["symbol"], "direction": pos["direction"], "net_return_pct": round(net, 4)})

            opened: list[dict[str, Any]] = []
            for symbol, leg in new_basket.items():
                if symbol in kept:
                    continue
                cur.execute(
                    "INSERT INTO live_strategy_positions (strategy_key, symbol, direction, status, opened_at,"
                    " opened_rebalance_id, entry_price, entry_score, entry_rank, cost_pct, last_price, last_price_at,"
                    " funding_updated_at) VALUES (?, ?, ?, 'open', ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        self.spec.key, symbol, leg["side"], now_iso, rebalance_id, leg["price"], leg["score"],
                        leg["rank"], round(round_trip_cost_pct_for(symbol, self.spec.base_cost_pct), 6),
                        leg["price"], now_iso, now_iso,
                    ),
                )
                opened.append({"symbol": symbol, "direction": leg["side"], "price": leg["price"], "score": leg["score"]})
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

        self._last_failed_attempt = None
        summary = {
            "rebalance_id": rebalance_id,
            "rebalanced_at": now_iso,
            "universe_size": len(universe),
            "n_eligible": len(scores),
            "basket": new_basket,
            "kept": sorted(kept),
            "opened": opened,
            "closed": closed,
            "turnover": turnover,
            "cost_pct": cost_pct,
            "next_rebalance_at": _iso(now + timedelta(hours=self.spec.rebalance_hours)),
        }
        logger.info(
            "Live strategy rebalanced: %d opened, %d kept, %d closed (universe %d, eligible %d)",
            len(opened), len(kept), len(closed), len(universe), len(scores),
        )
        if self.on_rebalance is not None:
            try:
                self.on_rebalance(summary)
            except Exception:
                logger.exception("Live strategy on_rebalance callback failed")
        return summary

    # -- per-cycle mark-to-market --

    def mark_to_market(self, now: datetime, prices: dict[str, float]) -> int:
        positions = fetch_open_positions(self.conn, self.spec.key)
        now_iso = _iso(now)
        marked = 0
        for pos in positions:
            price = prices.get(pos["symbol"])
            if price is None:
                continue
            excursion = directional_return_pct(pos["direction"], pos["entry_price"], price)
            mfe = max(pos["mfe_pct"] or 0.0, excursion)
            mae = min(pos["mae_pct"] or 0.0, excursion)
            fund = pos["funding_pct"]
            fund_at = pos["funding_updated_at"]
            if fund_at is None or now - _parse(fund_at) >= FUNDING_REFRESH_INTERVAL:
                events = fetch_funding_history(pos["symbol"], FUNDING_HISTORY_LIMIT)
                fund = funding_cost_pct(pos["direction"], _parse(pos["opened_at"]), now, events)
                fund_at = now_iso
            self.conn.execute(
                "UPDATE live_strategy_positions SET last_price=?, last_price_at=?, mfe_pct=?, mae_pct=?,"
                " funding_pct=?, funding_updated_at=? WHERE id=?",
                (price, now_iso, round(mfe, 6), round(mae, 6), round(fund, 6), fund_at, pos["id"]),
            )
            marked += 1
        if marked:
            self.conn.commit()
        return marked


# --- Read-side analytics (used by the webapp) -----------------------------


def position_view(pos: dict[str, Any]) -> dict[str, Any]:
    """Adds live unrealized numbers to an open leg (closed legs pass through)."""
    out = dict(pos)
    if pos["status"] == "open" and pos.get("last_price"):
        gross = directional_return_pct(pos["direction"], pos["entry_price"], pos["last_price"])
        out["unrealized_gross_pct"] = round(gross, 4)
        out["unrealized_net_pct"] = round(gross - pos["cost_pct"] - (pos["funding_pct"] or 0.0), 4)
    end = _parse(pos["closed_at"]) if pos.get("closed_at") else datetime.now(UTC)
    out["held_hours"] = round((end - _parse(pos["opened_at"])).total_seconds() / 3600.0, 1)
    return out


def _stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "win_rate_pct": None, "avg_pct": None, "sum_pct": None, "best_pct": None, "worst_pct": None}
    wins = sum(1 for v in values if v > 0)
    return {
        "n": len(values),
        "win_rate_pct": round(wins / len(values) * 100.0, 1),
        "avg_pct": round(sum(values) / len(values), 4),
        "sum_pct": round(sum(values), 4),
        "best_pct": round(max(values), 4),
        "worst_pct": round(min(values), 4),
    }


def build_summary(
    conn: sqlite3.Connection,
    spec: LiveStrategySpec = MAIN_STRATEGY,
    *,
    since: str | None = None,
    until: str | None = None,
) -> dict[str, Any]:
    positions = [position_view(p) for p in fetch_positions(conn, spec.key, since=since, until=until)]
    open_pos = [p for p in positions if p["status"] == "open"]
    closed_pos = [p for p in positions if p["status"] == "closed" and p["net_return_pct"] is not None]

    rebalances = fetch_rebalances(conn, spec.key)
    if since is not None:
        rebalances = [r for r in rebalances if r["rebalanced_at"] >= since]
    if until is not None:
        rebalances = [r for r in rebalances if r["rebalanced_at"] <= until]
    closed_periods = sorted(
        (r for r in rebalances if r["status"] == "closed" and r["net_return_pct"] is not None),
        key=lambda r: r["rebalanced_at"],
    )
    equity = 1.0
    curve = []
    for r in closed_periods:
        equity *= 1.0 + r["net_return_pct"] / 100.0
        curve.append({"at": r["closed_at"], "equity_pct": round((equity - 1.0) * 100.0, 4)})

    current = next((r for r in rebalances if r["status"] == "open"), None)
    current_period_unrealized = None
    if current is not None and open_pos:
        weight = _leg_weight(spec.basket_size)
        basket = json.loads(current["basket_json"])
        prices = {p["symbol"]: p["last_price"] for p in open_pos if p.get("last_price")}
        if all(s in prices for s in basket):
            gross = sum(
                weight * directional_return_pct(leg["side"], leg["price"], prices[s]) for s, leg in basket.items()
            )
            current_period_unrealized = round(gross - (current["cost_pct"] or 0.0), 4)

    latest = fetch_latest_rebalance(conn, spec.key)
    next_at = (
        _iso(_parse(latest["rebalanced_at"]) + timedelta(hours=spec.rebalance_hours)) if latest is not None else None
    )
    return {
        "strategy": spec.as_dict(),
        "disclaimer": (
            "Layer 1 eksploratorni kandidat (#1 po kompozitnom skoru), bez CI i bez Layer 2 potvrde. "
            "Ovo je forward test uživo, ne dokazan edge."
        ),
        "last_rebalance_at": latest["rebalanced_at"] if latest else None,
        "next_rebalance_at": next_at,
        "open_count": len(open_pos),
        "closed_count": len(closed_pos),
        "trades": {
            "all": _stats([p["net_return_pct"] for p in closed_pos]),
            "long": _stats([p["net_return_pct"] for p in closed_pos if p["direction"] == "long"]),
            "short": _stats([p["net_return_pct"] for p in closed_pos if p["direction"] == "short"]),
        },
        "periods": _stats([r["net_return_pct"] for r in closed_periods]),
        "compounded_net_pct": round((equity - 1.0) * 100.0, 4) if closed_periods else None,
        "equity_curve": curve,
        "current_period": (
            {
                "rebalanced_at": current["rebalanced_at"],
                "unrealized_net_pct": current_period_unrealized,
                "cost_pct": current["cost_pct"],
                "turnover": current["turnover"],
                "universe_size": current["universe_size"],
                "n_eligible": current["n_eligible"],
            }
            if current
            else None
        ),
        "open_positions": open_pos,
    }


def format_rebalance_message(summary: dict[str, Any], spec: LiveStrategySpec = MAIN_STRATEGY) -> str:
    def legs(side: str) -> str:
        items = sorted(
            ((s, leg) for s, leg in summary["basket"].items() if leg["side"] == side),
            key=lambda kv: kv[1]["rank"],
        )
        return "\n".join(f"  {s} @ {leg['price']:.6g} ({leg['score']:+.1f}%)" for s, leg in items)

    closed = ", ".join(f"{c['symbol']} {c['direction']} {c['net_return_pct']:+.2f}%" for c in summary["closed"]) or "—"
    return (
        f"🎯 Glavna strategija — rebalans\n{spec.template} / {spec.tercile} / {spec.variant_label}\n\n"
        f"LONG:\n{legs('long')}\n\nSHORT:\n{legs('short')}\n\n"
        f"Zadržano: {', '.join(summary['kept']) or '—'}\nZatvoreno: {closed}\n"
        f"Turnover {summary['turnover']:.0%}, trošak {summary['cost_pct']:.3f}%\n"
        f"Sljedeći rebalans: {summary['next_rebalance_at'][:16].replace('T', ' ')} UTC\n\n"
        "⚠️ Eksploratorno, nije potvrđeno (Layer 1)."
    )
