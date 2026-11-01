"""Deterministic crypto social-sentiment snapshot for the scanner/agent
pipeline.

Unlike the full TradingAgents sentiment analyst (:mod:`tradingagents.agents
.analysts.sentiment_analyst`), which hands raw text to an LLM inside the
multi-agent debate graph, this stays in the scanner's "measure first,
interpret second" philosophy: it counts StockTwits' own user-labeled
Bullish/Bearish tags and derives a label from the counts with a plain
threshold rule -- no model call, so it's cheap enough to compute on every
"Analyze with AI" click and never invents a confidence number.
"""

from __future__ import annotations

from typing import Any

from tradingagents.dataflows.vendors.stocktwits import fetch_stocktwits_sentiment_counts

# Below this many labeled (bullish+bearish) messages, the ratio is too noisy
# to call a label -- a handful of random posts swinging 100% one way means
# nothing for a low-chatter altcoin.
_MIN_LABELED_MESSAGES = 10

# A label requires the bullish/bearish split to clear this gap; a 55/45 split
# is not meaningfully different from a coin flip.
_LABEL_GAP_PCT = 15.0


def get_crypto_sentiment_snapshot(symbol: str, limit: int = 30) -> dict[str, Any]:
    """Fetch StockTwits' most recent messages for ``symbol`` and return a
    structured sentiment snapshot: raw counts plus a deterministic
    bullish/bearish/neutral label (or ``None`` when there isn't enough
    labeled chatter to say anything)."""
    counts = fetch_stocktwits_sentiment_counts(symbol, limit=limit)
    labeled = counts["bullish"] + counts["bearish"]

    label: str | None = None
    insufficient_reason: str | None = None
    if counts["error"]:
        insufficient_reason = f"StockTwits unavailable: {counts['error']}"
    elif labeled < _MIN_LABELED_MESSAGES:
        insufficient_reason = f"Only {labeled} labeled (bullish/bearish) messages — below the {_MIN_LABELED_MESSAGES} needed for a reliable read."
    else:
        bull_pct = counts["bullish_pct"] or 0.0
        bear_pct = counts["bearish_pct"] or 0.0
        if bull_pct - bear_pct >= _LABEL_GAP_PCT:
            label = "bullish"
        elif bear_pct - bull_pct >= _LABEL_GAP_PCT:
            label = "bearish"
        else:
            label = "neutral"

    return {
        "symbol": symbol,
        "label": label,
        "insufficient_reason": insufficient_reason,
        "bullish_count": counts["bullish"],
        "bearish_count": counts["bearish"],
        "unlabeled_count": counts["unlabeled"],
        "total_messages": counts["total"],
        "bullish_pct": counts["bullish_pct"],
        "bearish_pct": counts["bearish_pct"],
        "note": "Raw StockTwits user-labeled sentiment tags, not news analysis — a chatter-volume proxy, not a prediction.",
    }
