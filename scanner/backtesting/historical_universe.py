"""Phase 8 Korak 4: point-in-time universe (survivorship-bias fix).

The portfolio backtest (Korak 2) takes a fixed ``--symbols`` list and applies
it across the *entire* backtest window. In practice that list is normally
picked by looking at *today's* top-N by 24h quote volume (see
``scanner.market_scanner``'s ``get_top_symbols_by_quote_volume``). Using a
symbol's *current* popularity to decide whether it belongs in a test of
*past* periods is survivorship bias: coins that only became liquid/relevant
recently get credit for trades simulated before they would ever have been
selectable in real time, while coins that used to be liquid but have since
fallen off the radar never enter the test at all.

This module builds a point-in-time universe instead: for each rebalance
window across the backtest period, it ranks candidate symbols by their
*trailing* average daily quote volume as of that window's start date — using
only klines that would already have closed by then, never a future candle —
and records which symbols were actually in the top-N *at that time*.
``portfolio_backtest.py`` can then drop any simulated trade whose symbol
would not actually have been in the universe at its entry time, instead of
accepting every trade for every symbol in ``--symbols`` unconditionally
across the whole period.

Known limitation (documented, not silently papered over): Binance's
``exchangeInfo`` endpoint only lists symbols that are *currently* trading,
so fully delisted contracts cannot be auto-discovered this way. Use
``PointInTimeUniverseConfig.include_delisted`` to manually add any
historically-relevant delisted symbols you already know about — this is a
best-effort manual override, not an automatic fix for survivorship bias on
the "symbols that disappeared" side. The "symbols that only recently became
relevant" side (the more common case for this project's altcoin-heavy
universe) is handled properly.

Typical workflow::

    # Offline, run once (or whenever the cache should be refreshed) --
    # this makes many historical kline requests, one per candidate symbol
    # per rebalance window, so it is deliberately NOT run automatically.
    python -m scanner.backtesting.historical_universe \\
        --start 2024-06-01 --end 2025-11-01 --top-n 20 \\
        --cache-path scanner_data/point_in_time_universe.json

    # Then feed the cache into the portfolio backtest:
    python -m scanner.backtesting.portfolio_backtest --symbols ... \\
        --point-in-time-universe-cache scanner_data/point_in_time_universe.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from tradingagents.dataflows.errors import VendorError
from tradingagents.dataflows.vendors.binance import client

DEFAULT_TOP_N = 20
DEFAULT_REBALANCE_DAYS = 30
DEFAULT_LOOKBACK_DAYS = 30
DEFAULT_MIN_HISTORY_DAYS = 7


@dataclass(frozen=True)
class UniverseWindow:
    """One rebalance period's point-in-time top-N membership."""

    start: datetime  # inclusive
    end: datetime  # exclusive
    symbols: frozenset[str]


@dataclass
class PointInTimeUniverseConfig:
    start: datetime
    end: datetime
    top_n: int = DEFAULT_TOP_N
    rebalance_days: int = DEFAULT_REBALANCE_DAYS
    lookback_days: int = DEFAULT_LOOKBACK_DAYS
    # A symbol needs at least this many trailing daily candles before it is
    # even eligible to be ranked for a window -- guards against a symbol
    # that was listed 1-2 days ago getting ranked off a near-meaningless
    # average, while still letting genuinely-new-but-already-liquid symbols
    # in once they clear the bar.
    min_history_days: int = DEFAULT_MIN_HISTORY_DAYS
    # None -> auto-discover every currently-trading USDT perpetual via
    # exchangeInfo. Pass an explicit superset (e.g. your --symbols list
    # plus a few known always-top-liquidity majors) to cut request volume
    # dramatically when you only care whether specific symbols make the
    # cut, not the full market's top-N.
    candidate_symbols: tuple[str, ...] | None = None
    # Best-effort manual addition for contracts that are no longer trading
    # today and therefore invisible to exchangeInfo -- see module docstring.
    include_delisted: tuple[str, ...] = ()


def _avg_daily_quote_volume(
    symbol: str, as_of: datetime, lookback_days: int, min_history_days: int
) -> float | None:
    """Trailing average daily quote volume for ``symbol``, using only daily
    candles that closed strictly before ``as_of`` (no look-ahead). Returns
    ``None`` if the symbol has fewer than ``min_history_days`` of history by
    ``as_of`` -- this is the normal, expected case for a symbol that was not
    yet listed (or not yet liquid) at that point in time, not an error."""
    end_time_ms = int(as_of.timestamp() * 1000) - 1
    try:
        klines = client.get_klines(symbol, "1d", limit=lookback_days, end_time=end_time_ms)
    except VendorError:
        return None
    if len(klines) < min_history_days:
        return None
    volumes = [float(k["quote_volume"]) for k in klines]
    return sum(volumes) / len(volumes)


def build_point_in_time_universe(config: PointInTimeUniverseConfig) -> list[UniverseWindow]:
    """Builds the full sequence of rebalance windows spanning
    ``[config.start, config.end)``, ranking candidates by trailing quote
    volume as of each window's start date. Makes
    ``O(n_candidates * n_windows)`` kline requests -- expensive, meant to be
    run offline and cached, not inline in a hot path."""
    candidates = (
        list(config.candidate_symbols) if config.candidate_symbols else client.get_usdt_perpetual_symbols()
    )
    candidates = sorted(set(candidates) | set(config.include_delisted))

    windows: list[UniverseWindow] = []
    window_start = config.start
    step = timedelta(days=config.rebalance_days)
    while window_start < config.end:
        window_end = min(window_start + step, config.end)
        ranked: list[tuple[str, float]] = []
        for symbol in candidates:
            avg_volume = _avg_daily_quote_volume(
                symbol, window_start, config.lookback_days, config.min_history_days
            )
            if avg_volume is not None:
                ranked.append((symbol, avg_volume))
        ranked.sort(key=lambda pair: pair[1], reverse=True)
        top_symbols = frozenset(symbol for symbol, _volume in ranked[: config.top_n])
        windows.append(UniverseWindow(start=window_start, end=window_end, symbols=top_symbols))
        window_start = window_end
    return windows


def symbols_at(windows: list[UniverseWindow], timestamp: datetime) -> frozenset[str] | None:
    """Which symbols were in the point-in-time top-N at ``timestamp``, or
    ``None`` if ``timestamp`` falls outside every window (before the cache's
    coverage starts, or after its last rebalance)."""
    for window in windows:
        if window.start <= timestamp < window.end:
            return window.symbols
    return None


def is_symbol_in_universe_at(windows: list[UniverseWindow], symbol: str, timestamp: datetime) -> bool:
    """True if ``symbol`` was part of the point-in-time top-N at
    ``timestamp``. Conservative by design: a ``timestamp`` outside every
    known window returns ``False`` (the trade is excluded) rather than
    silently being let through -- widen the cache's date range if it does
    not cover your backtest's full period."""
    active = symbols_at(windows, timestamp)
    return active is not None and symbol in active


def save_universe_cache(windows: list[UniverseWindow], path: str | Path) -> None:
    payload = [
        {"start": w.start.isoformat(), "end": w.end.isoformat(), "symbols": sorted(w.symbols)} for w in windows
    ]
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_universe_cache(path: str | Path) -> list[UniverseWindow]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        UniverseWindow(
            start=_parse_iso(entry["start"]),
            end=_parse_iso(entry["end"]),
            symbols=frozenset(entry["symbols"]),
        )
        for entry in raw
    ]


def _parse_iso(value: str) -> datetime:
    ts = datetime.fromisoformat(value)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return ts


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 8 Korak 4: build a point-in-time (survivorship-bias-free) symbol universe cache."
    )
    parser.add_argument("--start", required=True, help="ISO date/time, e.g. 2024-06-01")
    parser.add_argument("--end", required=True, help="ISO date/time, e.g. 2025-11-01")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--rebalance-days", type=int, default=DEFAULT_REBALANCE_DAYS)
    parser.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS)
    parser.add_argument("--min-history-days", type=int, default=DEFAULT_MIN_HISTORY_DAYS)
    parser.add_argument(
        "--candidate-symbols",
        nargs="+",
        help="Restrict ranking to these symbols instead of auto-discovering every currently-trading USDT "
        "perpetual (much faster -- use this when you already know your --symbols superset)",
    )
    parser.add_argument(
        "--include-delisted",
        nargs="+",
        default=[],
        help="Manually add historically-relevant delisted symbols (best-effort, not auto-discovered)",
    )
    parser.add_argument("--cache-path", default="scanner_data/point_in_time_universe.json")
    args = parser.parse_args()

    config = PointInTimeUniverseConfig(
        start=_parse_iso(args.start),
        end=_parse_iso(args.end),
        top_n=args.top_n,
        rebalance_days=args.rebalance_days,
        lookback_days=args.lookback_days,
        min_history_days=args.min_history_days,
        candidate_symbols=tuple(args.candidate_symbols) if args.candidate_symbols else None,
        include_delisted=tuple(args.include_delisted),
    )
    windows = build_point_in_time_universe(config)
    save_universe_cache(windows, args.cache_path)
    print(f"Built {len(windows)} rebalance window(s), wrote cache to {args.cache_path}")
    for window in windows:
        print(f"  {window.start.date()} .. {window.end.date()}: {len(window.symbols)} symbols -> {sorted(window.symbols)}")


if __name__ == "__main__":
    _main()
