"""Classic floor-trader daily pivot points: P, R1-R3, S1-S3, computed from
the previous *completed* day's OHLC.

Pure math, no network calls — so it is unit-testable in isolation;
:mod:`tradingagents.dataflows.vendors.binance.snapshot` wires in the actual
previous-day candle fetched from Binance.
"""

from __future__ import annotations


def pivot_points_from_ohlc(prev_high: float, prev_low: float, prev_close: float) -> dict[str, float]:
    """Standard floor-trader pivots for the day following the given OHLC.

    P = (Vrh + Dno + Zatvaranje) / 3
    R1 = (2 x P) - Dno          S1 = (2 x P) - Vrh
    R2 = P + (Vrh - Dno)        S2 = P - (Vrh - Dno)
    R3 = Vrh + 2 x (P - Dno)    S3 = Dno - 2 x (Vrh - P)
    """
    pivot = (prev_high + prev_low + prev_close) / 3.0
    day_range = prev_high - prev_low
    return {
        "pivot": round(pivot, 8),
        "r1": round(2 * pivot - prev_low, 8),
        "s1": round(2 * pivot - prev_high, 8),
        "r2": round(pivot + day_range, 8),
        "s2": round(pivot - day_range, 8),
        "r3": round(prev_high + 2 * (pivot - prev_low), 8),
        "s3": round(prev_low - 2 * (prev_high - pivot), 8),
    }
