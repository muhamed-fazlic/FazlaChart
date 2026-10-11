"""Entry point: ``python -m scanner`` runs the Market Scanner forever.

Examples:
    python -m scanner
    python -m scanner --symbols BTCUSDT,ETHUSDT,INJUSDT --interval 120
    python -m scanner --top-n 25 --once
"""

from __future__ import annotations

import argparse
import logging
import sys

from scanner.config import ScannerConfig
from scanner.market_scanner import MarketScanner


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Binance futures Market Scanner")
    parser.add_argument("--symbols", type=str, default="", help="Comma-separated symbols, e.g. BTCUSDT,ETHUSDT")
    parser.add_argument("--top-n", type=int, default=40, help="Scan the top-N USDT perpetuals by 24h volume")
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
    )
    scanner = MarketScanner(config)

    if args.once:
        result = scanner.run_once()
        print(f"Scanned {len(result['symbols'])} symbols; BTC reference trend: {result['btc']['reference_trend']}")
        return
    scanner.run_forever()


if __name__ == "__main__":
    main()
