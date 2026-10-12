"""CLI: python -m agents_crypto --symbol INJUSDT --timeframe 1h

Defaults to a local Ollama model (no API key, no cost). Override with
--provider/--model/--base-url for OpenAI/Anthropic/etc.
"""

from __future__ import annotations

import argparse
import sys

from rich.console import Console

from agents_crypto.derivatives_risk_agent import DEFAULT_MODEL, DEFAULT_PROVIDER, assess_symbol

console = Console(width=150)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Derivatives/Risk agent — interprets scanner + calibration data")
    parser.add_argument("--symbol", required=True, help="e.g. INJUSDT")
    parser.add_argument("--timeframe", default="1h", help="e.g. 15m, 1h, 4h")
    parser.add_argument("--horizon-candles", type=int, default=12, help="Calibration horizon to match (candles)")
    parser.add_argument("--provider", default=DEFAULT_PROVIDER, help="LLM provider (default: ollama)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LLM model name")
    parser.add_argument("--base-url", default=None, help="Override the provider's default endpoint")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    console.print(
        f"[bold]Assessing {args.symbol} ({args.timeframe}) with {args.provider}/{args.model}...[/bold]"
    )
    assessment = assess_symbol(
        args.symbol.upper(),
        timeframe=args.timeframe,
        horizon_candles=args.horizon_candles,
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
    )
    console.print_json(assessment.model_dump_json())


if __name__ == "__main__":
    main()
