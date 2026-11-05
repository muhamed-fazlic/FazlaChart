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
``/fapi/v1/fundingRate`` endpoint is documented as "max 1000" but was
empirically measured (2026-11-05, direct API call) to actually cap every
response at **500** rows regardless of the requested ``limit`` -- i.e. a
single request only ever covers ~166.7 days (500 * 8h) of settlements, not
~333 days. ``fetch_funding_history`` (single request, anchored at "now")
retains this limitation and is fine for live/ad-hoc use. For a backtest
that needs a symbol's *entire* available funding history (e.g. the
funding-extreme-reversion hypothesis's rolling-percentile signal, which
needs a long warm-up), use :func:`fetch_full_funding_history` instead,
which pages backwards past the 500-row cap the same way
``scanner.calibration.load_extended_history`` pages backwards past
klines' per-request cap. For trades entirely before the earliest fetched
settlement, ``funding_cost_pct`` falls back to 0.0 -- treated as "unknown,
assume no funding cost" rather than raising, consistent with this
project's general stance of excluding what it cannot honestly measure
instead of guessing.
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

    Only the single most recent request's worth of history (at most 500
    settlements, ~166.7 days — see ``get_funding_rate_history``'s
    docstring), anchored at "now". For the funding-extreme-reversion
    hypothesis's full backtest window use :func:`fetch_full_funding_history`
    instead, which pages all the way back to each symbol's listing (or to
    an explicit frozen ``end_time_ms``).

    Returns an empty list (not an exception) on any vendor error or
    malformed response — a caller should treat "no funding data available
    for this symbol" as "skip funding cost for it", not abort the whole
    backtest over one bad symbol.
    """
    try:
        raw = client.get_funding_rate_history(symbol, limit=limit)
    except VendorError:
        return []
    return _parse_funding_rows(raw)


def _parse_funding_rows(raw: list[dict]) -> list[FundingEvent]:
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


def fetch_full_funding_history(
    symbol: str,
    end_time_ms: int | None = None,
    batch_limit: int = 500,
    max_batches: int = 50,
) -> list[FundingEvent]:
    """Page backwards (via ``endTime``) past the single-request ~500-row
    cap to assemble ``symbol``'s *entire* available funding-rate history,
    oldest first — mirrors ``scanner.calibration.load_extended_history``'s
    kline-pagination pattern.

    ``end_time_ms``, if given, anchors the whole page-back to end there
    instead of "now" — this is what makes a frozen/reproducible run (and a
    locked holdout boundary) possible: passing the pre-registered
    ``end_timestamp_ms`` here guarantees not a single funding settlement
    *after* that boundary is ever fetched, so the holdout cannot leak in
    through this path even though it is a separate data source from
    candles.

    ``max_batches`` is a hard safety cap (50 * 500 = 25 000 settlements =
    ~547 years) against a pathological infinite loop; real symbols stop
    far earlier once Binance returns an empty or short-of-``batch_limit``
    batch (meaning the symbol's listing date has been reached).
    """
    batches: list[list[FundingEvent]] = []
    cursor = end_time_ms
    for _ in range(max_batches):
        try:
            raw = client.get_funding_rate_history(symbol, limit=batch_limit, end_time=cursor)
        except VendorError:
            break
        batch = _parse_funding_rows(raw)
        if not batch:
            break
        batches.append(batch)
        oldest_ts_ms = int(batch[0].timestamp.timestamp() * 1000)
        if cursor is not None and oldest_ts_ms >= cursor:
            break  # vendor returned no new (older) settlements -- avoid an infinite loop
        cursor = oldest_ts_ms - 1
        if len(batch) < batch_limit:
            break  # reached the start of this symbol's available funding history
    all_events: dict[datetime, FundingEvent] = {}
    for batch in batches:
        for event in batch:
            all_events[event.timestamp] = event  # de-dup any overlap at batch boundaries
    events = sorted(all_events.values(), key=lambda e: e.timestamp)
    if end_time_ms is not None:
        boundary = datetime.fromtimestamp(end_time_ms / 1000.0, tz=UTC)
        events = [e for e in events if e.timestamp <= boundary]
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
