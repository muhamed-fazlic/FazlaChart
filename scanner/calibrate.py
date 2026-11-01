"""CLI: run the calibration backtest and print a win-rate / avg-return table.

Examples:
    python -m scanner.calibrate --symbols BTCUSDT,ETHUSDT,INJUSDT --timeframe 1h
    python -m scanner.calibrate --symbols SOLUSDT --timeframe 15m --horizons 4,12,48 --limit 1500
"""

from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.table import Table

from scanner.calibration import CalibrationStats, backtest_symbol, summarize

console = Console(width=150)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scanner label calibration backtest")
    parser.add_argument("--symbols", type=str, required=True, help="Comma-separated symbols, e.g. BTCUSDT,ETHUSDT")
    parser.add_argument("--timeframe", type=str, default="1h", help="Kline interval, e.g. 15m, 1h, 4h")
    parser.add_argument("--horizons", type=str, default="4,12,24", help="Forward horizons in candles, comma-separated")
    parser.add_argument("--limit", type=int, default=1500, help="Historical candles to fetch (Binance max 1500)")
    parser.add_argument("--cost-pct", type=float, default=0.10, help="Round-trip trading cost, in percent")
    return parser.parse_args(argv)


def _render(symbol: str, timeframe: str, stats: list[CalibrationStats]) -> None:
    table = Table(title=f"{symbol} {timeframe} — label calibration", show_lines=False)
    table.add_column("Label", no_wrap=True)
    table.add_column("Dir.", justify="center", no_wrap=True)
    table.add_column("Horizon", justify="right", no_wrap=True)
    table.add_column("N", justify="right")
    table.add_column("Win rate % (95% CI)", justify="right", no_wrap=True)
    table.add_column("Avg net return %", justify="right", no_wrap=True)
    table.add_column("Avg win %", justify="right")
    table.add_column("Avg loss %", justify="right")

    for s in stats:
        if s.win_rate_pct is not None:
            win_rate_cell = f"{s.win_rate_pct} ({s.win_rate_ci_low_pct}–{s.win_rate_ci_high_pct})"
        else:
            win_rate_cell = "n/a (undirected)"
        table.add_row(
            s.label,
            s.direction or "—",
            str(s.horizon_candles),
            str(s.n),
            win_rate_cell,
            f"{s.avg_net_return_pct}",
            f"{s.avg_win_return_pct}" if s.avg_win_return_pct is not None else "—",
            f"{s.avg_loss_return_pct}" if s.avg_loss_return_pct is not None else "—",
        )
    console.print(table)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    horizons = tuple(int(h) for h in args.horizons.split(","))

    for symbol in symbols:
        console.print(f"[bold]Backtesting {symbol} ({args.timeframe}, last {args.limit} candles)...[/bold]")
        cases = backtest_symbol(symbol, args.timeframe, horizons, limit=args.limit)
        stats = summarize(cases, cost_pct=args.cost_pct)
        if not stats:
            console.print(f"[yellow]No cases produced for {symbol} — not enough history for the chosen horizons.[/yellow]")
            continue
        _render(symbol, args.timeframe, stats)


if __name__ == "__main__":
    main()
