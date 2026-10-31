"""Approximate liquidity / stop-order heatmap from public OHLCV data.

Binance does not expose where other traders' SL/TP orders actually sit (no
exchange does, for anyone outside the matching engine), so this is **not**
real order-book data. It is a well-known proxy used by traders:

1. **Volume profile** — bin the recent price range and sum traded volume per
   bin. High-volume nodes (HVN) are prices the market spent a lot of time
   trading at; they tend to attract resting limit orders (TP/entries) and,
   just beyond them, cluster stop-losses. The point of control (POC) is the
   single highest-volume bin.
2. **Stop-hunt zones** — clusters of stop-losses are conventionally expected
   just beyond recent swing highs/lows (longs' stops below the last swing
   low, shorts' stops above the last swing high) and beyond round numbers,
   because that is where most retail traders place them.

This gives the AI/risk agent *some* structural context about where price is
likely to react, instead of none — but it is a heuristic approximation, not
ground truth, and the caveat belongs in every consumer of this module.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd


def _round_number_levels(price: float, count: int = 3) -> list[float]:
    """Nearby psychological round numbers, scaled to the coin's own price
    magnitude (round $1 steps make no sense for a $0.0003 coin)."""
    if price <= 0:
        return []
    # Step = a "nice" fraction of the price's own order of magnitude (e.g.
    # ~$1000 steps near $30000, ~$0.001 steps near $0.03).
    step = 10 ** round(math.log10(price)) / 10
    if step <= 0:
        return []
    levels: set[float] = set()
    base = round(price / step) * step
    for i in range(-count, count + 1):
        levels.add(round(base + i * step, 10))
    return sorted(levels)


def compute_liquidity_heatmap(
    df: pd.DataFrame,
    bins: int = 24,
    swing_lookback: int = 30,
) -> dict[str, Any]:
    """Volume-profile heatmap + likely stop-hunt zones for the given OHLCV
    frame (oldest-first, as returned by ``klines_to_dataframe``)."""
    if df.empty:
        return {"price_bins": [], "poc_price": None, "value_area_low": None, "value_area_high": None, "likely_stop_zones": [], "current_price": None}

    price_low = float(df["low"].min())
    price_high = float(df["high"].max())
    current_price = float(df["close"].iloc[-1])
    if price_high <= price_low:
        price_high = price_low * 1.0001

    bin_width = (price_high - price_low) / bins
    bucket_volume = [0.0] * bins

    # Simplified volume profile: assign each candle's volume to the bucket
    # containing its typical price (H+L+C)/3. Cheap (O(n)) and standard
    # practice for this kind of approximate profile.
    typical_prices = (df["high"] + df["low"] + df["close"]) / 3.0
    for typical, volume in zip(typical_prices, df["volume"]):
        idx = int((float(typical) - price_low) / bin_width) if bin_width > 0 else 0
        idx = min(max(idx, 0), bins - 1)
        bucket_volume[idx] += float(volume)

    total_volume = sum(bucket_volume) or 1.0
    price_bins = []
    for i, vol in enumerate(bucket_volume):
        low = price_low + i * bin_width
        high = low + bin_width
        price_bins.append(
            {
                "price_low": round(low, 8),
                "price_high": round(high, 8),
                "volume": round(vol, 3),
                "pct_of_total": round(vol / total_volume * 100.0, 2),
            }
        )

    poc_idx = max(range(bins), key=lambda i: bucket_volume[i])
    poc_price = round((price_bins[poc_idx]["price_low"] + price_bins[poc_idx]["price_high"]) / 2, 8)

    # Value area: expand out from POC until ~70% of total volume is covered —
    # the classic "where most trading happened" zone.
    covered = bucket_volume[poc_idx]
    lo_idx = hi_idx = poc_idx
    target = total_volume * 0.70
    while covered < target and (lo_idx > 0 or hi_idx < bins - 1):
        expand_low = bucket_volume[lo_idx - 1] if lo_idx > 0 else -1
        expand_high = bucket_volume[hi_idx + 1] if hi_idx < bins - 1 else -1
        if expand_high >= expand_low:
            hi_idx += 1
            covered += bucket_volume[hi_idx]
        else:
            lo_idx -= 1
            covered += bucket_volume[lo_idx]
    value_area_low = price_bins[lo_idx]["price_low"]
    value_area_high = price_bins[hi_idx]["price_high"]

    recent = df.tail(swing_lookback)
    swing_high = float(recent["high"].max())
    swing_low = float(recent["low"].min())

    likely_stop_zones = [
        {
            "price": round(swing_low * 0.995, 8),
            "reason": "long stop-loss cluster, just below recent swing low",
        },
        {
            "price": round(swing_high * 1.005, 8),
            "reason": "short stop-loss cluster, just above recent swing high",
        },
    ]
    for level in _round_number_levels(current_price):
        if price_low <= level <= price_high:
            likely_stop_zones.append({"price": level, "reason": "round-number psychological level"})

    return {
        "price_bins": price_bins,
        "poc_price": poc_price,
        "value_area_low": round(value_area_low, 8),
        "value_area_high": round(value_area_high, 8),
        "likely_stop_zones": likely_stop_zones,
        "current_price": round(current_price, 8),
        "note": "Approximate volume-profile proxy from public OHLCV — not real order-book SL/TP data.",
    }
