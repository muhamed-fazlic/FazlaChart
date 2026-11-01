"""Real order-book liquidity walls — augments (not replaces) the OHLCV
volume-profile proxy in :mod:`liquidity_heatmap` with an actual snapshot of
resting bid/ask size from Binance's public depth endpoint.

A "wall" here is a price level holding meaningfully more resting size than
a typical level nearby — these are genuine limit orders sitting in the
book right now, not a historical approximation, so they're a stronger
(though still imperfect — the book can be pulled or added to in seconds)
signal for where price is likely to react than the volume-profile proxy
alone. A large wall can act either as real support/resistance (it has to
be absorbed before price moves through) or as a magnet for a stop-hunt
(price often wicks just past a visible wall to trigger stops sitting
behind it, then reverses) — this module flags the wall, not which of the
two is more likely; that judgment belongs to the consuming agent/UI.
"""

from __future__ import annotations

import statistics
from typing import Any


def compute_orderbook_walls(
    depth: dict[str, Any],
    current_price: float,
    top_n: int = 5,
    wall_multiplier: float = 3.0,
) -> dict[str, Any]:
    """Flags unusually large resting-size levels in ``depth`` (Binance's
    ``GET /fapi/v1/depth`` response: ``{"bids": [[price, qty], ...], "asks": [...]}``).

    A level counts as a "wall" when its notional (price * qty) is at least
    ``wall_multiplier`` times the median notional of all levels on its own
    side — so a quiet, thin order book never falsely reports a "wall" just
    because every level on that side happens to be roughly the same small
    size.
    """
    bids = [(float(p), float(q)) for p, q in depth.get("bids", [])]
    asks = [(float(p), float(q)) for p, q in depth.get("asks", [])]

    def _walls(levels: list[tuple[float, float]], side: str) -> list[dict[str, Any]]:
        if not levels:
            return []
        notionals = [p * q for p, q in levels]
        median_notional = statistics.median(notionals) or 1.0
        threshold = median_notional * wall_multiplier
        walls = [
            {
                "side": side,
                "price": round(p, 8),
                "quantity": round(q, 8),
                "notional_usdt": round(p * q, 2),
                "distance_pct": round((p - current_price) / current_price * 100.0, 3) if current_price else None,
            }
            for p, q in levels
            if p * q >= threshold
        ]
        walls.sort(key=lambda w: w["notional_usdt"], reverse=True)
        return walls[:top_n]

    bid_walls = _walls(bids, "bid")
    ask_walls = _walls(asks, "ask")

    return {
        "bid_walls": bid_walls,
        "ask_walls": ask_walls,
        "best_bid": bids[0][0] if bids else None,
        "best_ask": asks[0][0] if asks else None,
        "levels_analyzed": {"bids": len(bids), "asks": len(asks)},
        "note": "Real order-book snapshot (point-in-time, not streamed) -- walls can be pulled/added within seconds.",
    }
