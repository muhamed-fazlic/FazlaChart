"""Phase 8 Korak 5: funding-cost simulator.

The portfolio backtest (and the underlying per-candle calibration in
``scanner.calibration``) replays klines only — it has never modeled the
funding payments a real USDT-M perpetual position actually accrues every 8
hours it stays open. A long position pays funding when the funding rate is
positive (and is paid when negative); a short position is the mirror image.
Ignoring this systematically overstates edge, especially for trades held
across one or more funding settlements — i.e. precisely the trades that
would otherwise look best on longer timeframes (1h/4h), where a position
routinely spans multiple 8h settlements, while a typical 5m/15m trade
usually closes before its first settlement and is barely affected.

This module fetches Binance's historical funding-rate settlements for a
symbol and lets a caller compute exactly how much of a trade's return was
actually funding cost, given its direction, entry time and exit time.

Known limitation (documented, not silently papered over): Binance's
``/fapi/v1/fundingRate`` endpoint returns the *most recent* ``limit``
settlements when no explicit time range is requested (max 1000, i.e. ~333
days of 8h settlements). For a backtest window older than that, the oldest
part of the window would simply have no funding data and ``funding_cost_pct``
falls back to 0.0 for trades entirely before the earliest fetched
settlement — treated as "unknown, assume no funding cost" rather than
raising, consistent with this project's general stance of excluding what
it cannot honestly measure instead of guessing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from tradingagents.dataflows.errors import VendorError
from tradingagents.dataflows.vendors.binance import client

# Binance settles funding every 8 hours for USDT-M perpetuals.
FUNDING_INTERVAL_HOURS = 8
MAX_FUNDING_SETTLEMENTS = 1000  # Binance's own API cap per request


@dataclass(frozen=True)
class FundingEvent:
    """One settled funding payment. ``rate`` is a raw fraction (e.g. 0.0001
    == 0.01%), the same units Binance returns — not already multiplied by
    100."""

    timestamp: datetime
    rate: float


def fetch_funding_history(symbol: str, limit: int = MAX_FUNDING_SETTLEMENTS) -> list[FundingEvent]:
    """Settled funding-rate history for ``symbol``, oldest first.

    Returns an empty list (not an exception) on any vendor error or
    malformed response — a caller should treat "no funding data available
    for this symbol" as "skip funding cost for it", not abort the whole
    backtest over one bad symbol.
    """
    try:
        raw = client.get_funding_rate_history(symbol, limit=limit)
    except VendorError:
        return []
    events: list[FundingEvent] = []
    for row in raw:
        try:
            ts = datetime.fromtimestamp(int(row["fundingTime"]) / 1000.0, tz=UTC)
            rate = float(row["fundingRate"])
        except (KeyError, TypeError, ValueError):
            continue
        events.append(FundingEvent(timestamp=ts, rate=rate))
    events.sort(key=lambda e: e.timestamp)
    return events


def funding_cost_pct(
    direction: str,
    entry_time: datetime,
    exit_time: datetime,
    events: list[FundingEvent],
) -> float:
    """Total funding cost for one trade, as a percentage-point reduction
    meant to be subtracted directly from that trade's net return — already
    sign-adjusted for ``direction`` so a caller never needs to re-flip it.

    A funding settlement at time ``t`` is charged only to a position that
    was genuinely open and holding exposure at that instant: ``entry_time <
    t <= exit_time``. A trade opened exactly at a settlement does not pay
    it (it wasn't open yet); a trade closed exactly at a settlement does
    (matching how Binance actually settles open positions at that instant).

    A positive funding rate means longs pay shorts. For ``direction ==
    "long"`` each applicable settlement's rate is added as-is (a positive
    rate becomes a positive, i.e. cost, number). For ``direction ==
    "short"`` the sign is flipped (a positive rate becomes a credit —
    subtracting a negative number increases the short's net return,
    correctly rewarding it for holding through positive funding).
    """
    if entry_time >= exit_time or not events:
        return 0.0
    total_rate = sum(e.rate for e in events if entry_time < e.timestamp <= exit_time)
    signed = total_rate if direction == "long" else -total_rate
    return round(signed * 100.0, 6)
