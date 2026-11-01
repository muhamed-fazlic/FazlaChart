"""Scanner configuration: what to scan, how often, and with what thresholds."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ScannerConfig:
    # Symbol universe. If `symbols` is empty, the scanner ranks all USDT
    # perpetuals by 24h quote volume and scans the top `top_n_by_volume`.
    # (Binance's public rate limit has comfortable headroom for this: even
    # at ~12 request-weight per symbol per cycle, 60 symbols every 300s is
    # a fraction of the ~2400/min weight budget.)
    symbols: tuple[str, ...] = ()
    top_n_by_volume: int = 120

    # Once a symbol has been scanned, keep refreshing it every cycle even if
    # it later drops out of the current top-N by volume — otherwise its last
    # snapshot just sits in the dashboard getting staler and staler forever.
    # `max_universe_size` is the hard ceiling on how large that "current +
    # everything ever seen" set is allowed to grow, so a cycle never scans an
    # unbounded number of symbols; least-recently-scanned known symbols are
    # dropped first once the ceiling is hit. Binance's public rate limit has
    # ample headroom for this (see the comment above `top_n_by_volume`).
    keep_known_symbols: bool = True
    max_universe_size: int = 250

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

    # Telegram push alerts for high-quality Top Signal Now candidates (see
    # scanner.telegram_alerts). Left unset by default -- alerting stays off
    # until both bot_token and chat_id are provided, typically from
    # TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID env vars (never as CLI flags, to
    # avoid a bot token showing up in `ps`/process-list output).
    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None
    telegram_min_win_rate_ci_low_pct: float = 60.0
    telegram_min_n: int = 50
    telegram_require_btc_alignment: bool = False

    # How long raw `scans` rows are kept before being pruned. Calibration
    # always re-derives its stats fresh from live Binance klines (never from
    # this table), and every dashboard read only looks at recent rows, so
    # old scan rows are pure disk growth with no analytical value once
    # they're this stale. `signal_outcomes` (the live forward-tracking
    # record) is never pruned by this setting -- that table is the actual
    # track record and stays tiny regardless.
    scan_retention_days: int = 60
    retention_check_interval_seconds: int = 86400
