"""Single-symbol anomaly detection: situations where price/volume/momentum
behave differently from what the recent history would predict.

These are *flags for closer inspection*, not trade signals. Each function
takes the indicator-enriched DataFrame built by
:mod:`tradingagents.dataflows.vendors.binance.indicators` and returns True/False
for one specific pattern; :func:`detect_anomalies` collects the ones that fired.
"""

from __future__ import annotations

import pandas as pd

from tradingagents.dataflows.vendors.binance.indicators import ema_cross_event, fibonacci_levels


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


def squeeze_breakout_direction(df: pd.DataFrame, lookback: int = 20, expansion_ratio: float = 1.8) -> str | None:
    """Which way price already broke, the moment a squeeze (see
    :func:`volatility_squeeze_expansion`) starts expanding. By itself, a
    squeeze expanding says nothing about direction -- it can break either
    way -- which is exactly why this is a separate, deliberately rarer check
    rather than a single non-directional flag: it only fires once the
    breakout candle's own close has already cleared the high or low of the
    compressed range that preceded it (the range of ``lookback`` candles
    *before* this expansion candle, so the candle currently breaking out
    doesn't get to vote on its own range). Still inside that range, or the
    squeeze isn't currently expanding at all -> None (no directional claim,
    on purpose -- see ``LABEL_DIRECTION`` in ``scanner.calibration``)."""
    if not volatility_squeeze_expansion(df, lookback, expansion_ratio):
        return None
    pre_window = df.iloc[-(lookback + 1):-1]
    recent_high, recent_low = pre_window["high"].max(), pre_window["low"].min()
    close = df["close"].iloc[-1]
    if close > recent_high:
        return "long"
    if close < recent_low:
        return "short"
    return None


def ema9_21_bullish_cross(df: pd.DataFrame) -> bool:
    """EMA9 just crossed above EMA21 on the latest closed candle — the
    actionable short-term-momentum-turning-up moment (see
    :func:`tradingagents.dataflows.vendors.binance.indicators.ema_cross_event`)."""
    return ema_cross_event(df) == "bullish_cross"


def ema9_21_bearish_cross(df: pd.DataFrame) -> bool:
    """EMA9 just crossed below EMA21 on the latest closed candle."""
    return ema_cross_event(df) == "bearish_cross"


def bullish_pullback_confluence(df: pd.DataFrame, lookback: int = 50) -> bool:
    """Three independent confirmations lining up for a long, not just one:

    1. The broader trend is bullish (EMA50 above EMA200).
    2. Price pulled back into the Fibonacci "golden pocket" (0.5-0.618
       retracement from the recent swing high) of an up-swing.
    3. EMA9 has just crossed back above EMA21, confirming the bounce.

    This is deliberately rare and highly selective — combining several real,
    independent signals is how a setup's win rate can realistically be
    pushed well above any single indicator alone, at the cost of fewer
    occurrences (see the calibration backtest's sample size for this label).
    """
    if len(df) < 200:
        return False
    last = df.iloc[-1]
    if not (last["close_50_ema"] > last["close_200_ema"]):
        return False
    fib = fibonacci_levels(df, lookback)
    if fib.get("swing_direction") != "up":
        return False
    retracement = fib.get("retracement", {})
    pocket_high, pocket_low = retracement.get("0.500"), retracement.get("0.618")
    if pocket_high is None or pocket_low is None:
        return False
    in_pocket = pocket_low <= last["close"] <= pocket_high
    return bool(in_pocket and ema9_21_bullish_cross(df))


def bearish_pullback_confluence(df: pd.DataFrame, lookback: int = 50) -> bool:
    """Mirror of :func:`bullish_pullback_confluence` for shorts: broader trend
    bearish (EMA50 below EMA200), price bounced into the golden pocket of a
    down-swing, and EMA9 has just crossed back below EMA21."""
    if len(df) < 200:
        return False
    last = df.iloc[-1]
    if not (last["close_50_ema"] < last["close_200_ema"]):
        return False
    fib = fibonacci_levels(df, lookback)
    if fib.get("swing_direction") != "down":
        return False
    retracement = fib.get("retracement", {})
    pocket_low, pocket_high = retracement.get("0.500"), retracement.get("0.618")
    if pocket_high is None or pocket_low is None:
        return False
    in_pocket = pocket_low <= last["close"] <= pocket_high
    return bool(in_pocket and ema9_21_bearish_cross(df))


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
    squeeze_direction = squeeze_breakout_direction(df)
    if squeeze_direction == "long":
        flags.append("squeeze_breakout_long")
    elif squeeze_direction == "short":
        flags.append("squeeze_breakout_short")
    if ema9_21_bullish_cross(df):
        flags.append("ema9_21_bullish_cross")
    if ema9_21_bearish_cross(df):
        flags.append("ema9_21_bearish_cross")
    if bullish_pullback_confluence(df):
        flags.append("bullish_pullback_confluence")
    if bearish_pullback_confluence(df):
        flags.append("bearish_pullback_confluence")
    return flags
