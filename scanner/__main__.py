"""Entry point: ``python -m scanner`` runs the Market Scanner forever.

Examples:
    python -m scanner
    python -m scanner --symbols BTCUSDT,ETHUSDT,INJUSDT --interval 120
    python -m scanner --top-n 120 --once
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from scanner.config import ScannerConfig
from scanner.market_scanner import MarketScanner


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Binance futures Market Scanner")
    parser.add_argument("--symbols", type=str, default="", help="Comma-separated symbols, e.g. BTCUSDT,ETHUSDT")
    parser.add_argument("--top-n", type=int, default=120, help="Scan the top-N USDT perpetuals by 24h volume")
    parser.add_argument(
        "--max-universe-size",
        type=int,
        default=250,
        help="Upper bound on symbols per cycle once previously-known symbols are kept (see --no-keep-known-symbols)",
    )
    parser.add_argument(
        "--no-keep-known-symbols",
        action="store_true",
        help="Only scan the current top-N by volume; let symbols that drop out of rotation go stale instead of"
        " continuing to refresh them",
    )
    parser.add_argument("--interval", type=int, default=300, help="Seconds between scan cycles")
    parser.add_argument("--storage", type=str, default="scanner_data/scanner.db", help="SQLite file path")
    parser.add_argument("--once", action="store_true", help="Run a single cycle and exit (for testing)")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    symbols = tuple(s.strip().upper() for s in args.symbols.split(",") if s.strip())
    config = ScannerConfig(
        symbols=symbols,
        top_n_by_volume=args.top_n,
        poll_interval_seconds=args.interval,
        storage_path=args.storage,
        keep_known_symbols=not args.no_keep_known_symbols,
        max_universe_size=args.max_universe_size,
        # Telegram credentials come from the environment, never CLI flags
        # (which would otherwise leak a bot token into `ps`/process-list
        # output) -- see scanner/telegram_alerts.py.
        telegram_bot_token=os.environ.get("TELEGRAM_BOT_TOKEN") or None,
        telegram_chat_id=os.environ.get("TELEGRAM_CHAT_ID") or None,
        telegram_min_win_rate_ci_low_pct=float(os.environ.get("TELEGRAM_MIN_WIN_RATE_CI_LOW_PCT", "60.0")),
        telegram_min_n=int(os.environ.get("TELEGRAM_MIN_N", "50")),
        telegram_require_btc_alignment=os.environ.get("TELEGRAM_REQUIRE_BTC_ALIGNMENT", "").lower() in ("1", "true", "yes"),
    )
    scanner = MarketScanner(config)

    if args.once:
        result = scanner.run_once()
        print(f"Scanned {len(result['symbols'])} symbols; BTC reference trend: {result['btc']['reference_trend']}")
        return
    scanner.run_forever()


if __name__ == "__main__":
    main()
