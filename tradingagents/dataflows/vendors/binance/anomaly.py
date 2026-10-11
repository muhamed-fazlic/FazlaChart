"""Single-symbol anomaly detection: situations where price/volume/momentum
behave differently from what the recent history would predict.

These are *flags for closer inspection*, not trade signals. Each function
takes the indicator-enriched DataFrame built by
:mod:`tradingagents.dataflows.vendors.binance.indicators` and returns True/False
for one specific pattern; :func:`detect_anomalies` collects the ones that fired.
"""

from __future__ import annotations

import pandas as pd


def price_up_volume_not_confirming(df: pd.DataFrame, price_move_pct: float = 1.5, lookback: int = 3) -> bool:
    """Price rose sharply over ``lookback`` candles while relative volume stayed flat/low."""
    if len(df) < lookback + 1:
        return False
    window = df.iloc[-lookback:]
    price_change = (window["close"].iloc[-1] - window["close"].iloc[0]) / window["close"].iloc[0] * 100.0
    avg_volume = df["volume"].iloc[-(lookback + 20):-lookback].mean() if len(df) > lookback + 20 else df["volume"].mean()
    recent_volume = window["volume"].mean()
    volume_confirming = avg_volume > 0 and recent_volume / avg_volume >= 1.2
    return bool(price_change >= price_move_pct and not volume_confirming)


def oi_rising_price_flat_funding_spiking(
    price_change_pct: float, oi_change_pct_60m: float | None, funding_rate: float,
    price_flat_threshold: float = 0.3, oi_rise_threshold: float = 5.0, funding_threshold: float = 0.0005,
) -> bool:
    """Open interest climbing while price goes nowhere and funding turns strongly
    positive: new leveraged positioning piling in without price confirmation."""
    if oi_change_pct_60m is None:
        return False
    return bool(
        abs(price_change_pct) <= price_flat_threshold
        and oi_change_pct_60m >= oi_rise_threshold
        and funding_rate >= funding_threshold
    )


def failed_breakout(df: pd.DataFrame, resistance: float, lookback: int = 6) -> bool:
    """Price pierced resistance within the last ``lookback`` candles (including
    the current one, so a single rejection-wick candle also counts) but the
    latest close is back below it."""
    if len(df) < lookback:
        return False
    window = df.iloc[-lookback:]
    pierced = bool((window["high"] > resistance).any())
    closed_back_below = bool(df["close"].iloc[-1] < resistance)
    return pierced and closed_back_below


def volatility_squeeze_expansion(df: pd.DataFrame, lookback: int = 20, expansion_ratio: float = 1.8) -> bool:
    """A long, tight Bollinger-band squeeze suddenly widening — volatility
    expansion after compression, often preceding a directional move."""
    if len(df) < lookback + 2:
        return False
    band_width = (df["boll_ub"] - df["boll_lb"]) / df["boll"]
    recent_avg = band_width.iloc[-(lookback + 1):-1].mean()
    current = band_width.iloc[-1]
    return bool(recent_avg > 0 and current / recent_avg >= expansion_ratio)


def detect_anomalies(
    df: pd.DataFrame,
    resistance: float,
    funding_rate: float = 0.0,
    oi_change_pct_60m: float | None = None,
) -> list[str]:
    """Run every single-symbol anomaly check and return the ones that fired."""
    flags: list[str] = []
    if len(df) < 2:
        return flags

    price_change_pct = (df["close"].iloc[-1] - df["close"].iloc[-2]) / df["close"].iloc[-2] * 100.0

    if price_up_volume_not_confirming(df):
        flags.append("price_up_volume_not_confirming")
    if oi_rising_price_flat_funding_spiking(price_change_pct, oi_change_pct_60m, funding_rate):
        flags.append("oi_rising_price_flat_funding_spiking")
    if failed_breakout(df, resistance):
        flags.append("failed_breakout")
    if volatility_squeeze_expansion(df):
        flags.append("volatility_squeeze_expansion")
    return flags
