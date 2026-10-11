"""Scanner configuration: what to scan, how often, and with what thresholds."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ScannerConfig:
    # Symbol universe. If `symbols` is empty, the scanner ranks all USDT
    # perpetuals by 24h quote volume and scans the top `top_n_by_volume`.
    symbols: tuple[str, ...] = ()
    top_n_by_volume: int = 40

    # Timeframes analyzed for every symbol, shortest first.
    timeframes: tuple[str, ...] = ("5m", "15m", "1h", "4h")

    # Timeframes used to build the BTC trend-filter context.
    btc_timeframes: tuple[str, ...] = ("15m", "1h", "4h")

    # Seconds between full scan cycles.
    poll_interval_seconds: int = 300

    # Where structured snapshots are persisted for later review/backtesting.
    storage_path: str = "scanner_data/scanner.db"

    # A symbol is treated as "diverging from BTC" when its 1h trend differs
    # from BTC's 1h trend by at least this many trend-strength steps
    # (bearish=-1, neutral=0, bullish=+1), i.e. opposite trends only.
    btc_divergence_min_steps: int = 2

    # Only fetch derivatives (funding/OI) for symbols worth the extra calls;
    # set True to always fetch (safe at <= ~60 symbols given Binance's public
    # rate limits).
    always_fetch_derivatives: bool = True

    symbols_blocklist: tuple[str, ...] = field(default_factory=tuple)
