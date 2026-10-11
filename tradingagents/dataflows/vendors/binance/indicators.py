"""Technical-analysis layer on top of raw Binance klines.

Converts the vendor's list-of-dicts candles into a pandas DataFrame, adds
indicators via ``stockstats`` (the same library the stock Market Analyst
already uses, for consistency across asset classes), and derives the
higher-level facts the scanner snapshot needs: trend label, relative volume,
and simple support/resistance levels.

This module is pure computation — no network calls — so it is unit-testable
with a synthetic DataFrame.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from stockstats import wrap


def klines_to_dataframe(klines: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert Binance kline dicts into a numeric OHLCV DataFrame, oldest-first."""
    df = pd.DataFrame(klines)
    numeric_cols = ["open", "high", "low", "close", "volume", "quote_volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("open_time", drop=False)
    return df[["open_time", "open", "high", "low", "close", "volume", "quote_volume"]]


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` wrapped with stockstats, with the indicators
    the scanner needs pre-computed (so they are cached on the frame)."""
    wrapped = wrap(df.copy())
    for indicator in ("close_20_ema", "close_50_ema", "close_200_ema", "rsi", "macd", "macds", "macdh", "atr", "boll", "boll_ub", "boll_lb"):
        wrapped[indicator]  # triggers stockstats' lazy calculation
    return wrapped


def classify_trend(df: pd.DataFrame) -> str:
    """Label the trend from EMA ordering + slope on the last row.

    bullish: price > ema20 > ema50 and ema20 rising
    bearish: price < ema20 < ema50 and ema20 falling
    neutral: anything else (chop / transition)
    """
    if len(df) < 3:
        return "neutral"
    last = df.iloc[-1]
    prev = df.iloc[-3]
    price = last["close"]
    ema20, ema50 = last["close_20_ema"], last["close_50_ema"]
    ema20_prev = prev["close_20_ema"]
    rising = ema20 > ema20_prev
    falling = ema20 < ema20_prev

    if price > ema20 > ema50 and rising:
        return "bullish"
    if price < ema20 < ema50 and falling:
        return "bearish"
    return "neutral"


def relative_volume(df: pd.DataFrame, lookback: int = 20) -> float:
    """Last candle's volume divided by the average of the previous ``lookback``.

    > 1 means above-average participation; the scanner uses this to tell a
    real breakout from a move with no volume behind it.
    """
    if len(df) <= lookback:
        return 1.0
    recent = df["volume"].iloc[-(lookback + 1):-1]
    avg = recent.mean()
    if not avg or avg == 0:
        return 1.0
    return float(df["volume"].iloc[-1] / avg)


def volume_change_pct(df: pd.DataFrame, lookback: int = 20) -> float:
    """Percent change of the last candle's volume vs the trailing average."""
    return (relative_volume(df, lookback) - 1.0) * 100.0


def support_resistance(df: pd.DataFrame, lookback: int = 50) -> tuple[float, float]:
    """Simple support/resistance: lowest low / highest high over ``lookback``
    candles, excluding the current (still-forming) candle."""
    window = df.iloc[-(lookback + 1):-1] if len(df) > lookback else df.iloc[:-1]
    if window.empty:
        window = df
    return float(window["low"].min()), float(window["high"].max())


def momentum_fading(df: pd.DataFrame, lookback: int = 5) -> bool:
    """True when price made a new high over ``lookback`` candles but the MACD
    histogram did not — a classic bearish-divergence / fading-momentum flag."""
    if len(df) < lookback + 1:
        return False
    recent = df.iloc[-lookback:]
    new_high = recent["high"].iloc[-1] >= recent["high"].max()
    macdh_falling = recent["macdh"].iloc[-1] < recent["macdh"].iloc[0]
    return bool(new_high and macdh_falling)
