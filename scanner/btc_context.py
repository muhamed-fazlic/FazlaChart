"""BTC trend/structure context, used as a filter before trusting any altcoin
signal — per the project brief, an altcoin setup is only as good as what BTC
is doing at the same time.
"""

from __future__ import annotations

from typing import Any

from tradingagents.dataflows.vendors.binance.snapshot import build_timeframe_snapshot

BTC_SYMBOL = "BTCUSDT"

_TREND_SCORE = {"bearish": -1, "neutral": 0, "bullish": 1}


def build_btc_context(timeframes: tuple[str, ...] = ("15m", "1h", "4h")) -> dict[str, Any]:
    """BTC trend + structure across several timeframes, for use as a filter."""
    per_timeframe = {tf: build_timeframe_snapshot(BTC_SYMBOL, tf) for tf in timeframes}
    return {
        "symbol": BTC_SYMBOL,
        "timeframes": per_timeframe,
        # 1h is the default "what's BTC doing right now" reference the
        # altcoin-divergence check uses.
        "reference_trend": per_timeframe.get("1h", next(iter(per_timeframe.values())))["trend"],
    }


def trend_score(trend: str) -> int:
    return _TREND_SCORE.get(trend, 0)


def diverges_from_btc(altcoin_trend: str, btc_trend: str, min_steps: int = 2) -> bool:
    """True when the altcoin trend and BTC trend point in opposite directions
    strongly enough to be noteworthy (e.g. altcoin bullish while BTC bearish)."""
    return abs(trend_score(altcoin_trend) - trend_score(btc_trend)) >= min_steps
