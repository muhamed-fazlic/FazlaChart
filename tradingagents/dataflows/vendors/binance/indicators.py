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
    for indicator in (
        "close_9_ema",
        "close_20_ema",
        "close_21_ema",
        "close_50_ema",
        "close_200_ema",
        "rsi",
        "macd",
        "macds",
        "macdh",
        "atr",
        "boll",
        "boll_ub",
        "boll_lb",
    ):
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


def ema_cross(df: pd.DataFrame) -> str:
    """Short-term momentum confirmation via EMA9 vs EMA21 (classic 9/21 cross).

    bullish: EMA9 > EMA21 (short-term momentum is up)
    bearish: EMA9 < EMA21 (short-term momentum is down)
    neutral: equal (rare, effectively never with float data)
    """
    if df.empty:
        return "neutral"
    last = df.iloc[-1]
    ema9, ema21 = last["close_9_ema"], last["close_21_ema"]
    if ema9 > ema21:
        return "bullish"
    if ema9 < ema21:
        return "bearish"
    return "neutral"


def ema_cross_event(df: pd.DataFrame) -> str | None:
    """A *fresh* EMA9/EMA21 cross on the latest closed candle versus the one
    before it — the actionable moment, unlike :func:`ema_cross`'s ongoing
    state (which stays "bullish" for the whole stretch after a cross)."""
    if len(df) < 2:
        return None
    last, prev = df.iloc[-1], df.iloc[-2]
    now_bullish = last["close_9_ema"] > last["close_21_ema"]
    was_bullish = prev["close_9_ema"] > prev["close_21_ema"]
    if now_bullish and not was_bullish:
        return "bullish_cross"
    if not now_bullish and was_bullish:
        return "bearish_cross"
    return None


# Standard retracement/extension ratios; retracement levels sit between the
# swing low and swing high (pullback/entry zones), extension levels sit
# beyond the swing high (continuation targets).
FIB_RETRACEMENT_RATIOS: tuple[float, ...] = (0.382, 0.5, 0.618)
FIB_EXTENSION_RATIOS: tuple[float, ...] = (0.236, 0.382, 0.618)


def fibonacci_levels(df: pd.DataFrame, lookback: int = 50) -> dict[str, Any]:
    """Fibonacci retracement/extension levels anchored to the most recent
    swing high/low (the same lookback window :func:`support_resistance` uses).

    Raspon (range) = Vrh (swing high) - Dno (swing low).

    If the swing high happened more recently than the swing low (price is
    currently in an up-move), retracement levels are measured down from the
    high (``Vrh - Raspon * ratio``) and extension levels up from the high
    (``Vrh + Raspon * ratio``) — exactly as in a standard uptrend Fibonacci
    tool. If the swing low is more recent (a down-move), both are mirrored
    around the low instead, which is how the same tool is used for shorts.
    """
    swing_low, swing_high = support_resistance(df, lookback)
    rng = swing_high - swing_low
    empty = {
        "swing_high": round(swing_high, 8),
        "swing_low": round(swing_low, 8),
        "range": round(rng, 8),
        "swing_direction": None,
        "retracement": {},
        "extension": {},
    }
    if rng <= 0:
        return empty

    window = df.iloc[-(lookback + 1):-1] if len(df) > lookback else df.iloc[:-1]
    if window.empty:
        return empty

    # Whichever extreme occurred later in time defines the active swing
    # direction (its index position in `window`, not calendar time, since
    # `window` is already in chronological order).
    high_pos = window["high"].to_numpy().argmax()
    low_pos = window["low"].to_numpy().argmin()
    swing_is_up = high_pos >= low_pos

    if swing_is_up:
        retracement = {f"{ratio:.3f}": round(swing_high - rng * ratio, 8) for ratio in FIB_RETRACEMENT_RATIOS}
        extension = {f"{ratio:.3f}": round(swing_high + rng * ratio, 8) for ratio in FIB_EXTENSION_RATIOS}
    else:
        retracement = {f"{ratio:.3f}": round(swing_low + rng * ratio, 8) for ratio in FIB_RETRACEMENT_RATIOS}
        extension = {f"{ratio:.3f}": round(swing_low - rng * ratio, 8) for ratio in FIB_EXTENSION_RATIOS}

    return {
        "swing_high": round(swing_high, 8),
        "swing_low": round(swing_low, 8),
        "range": round(rng, 8),
        "swing_direction": "up" if swing_is_up else "down",
        "retracement": retracement,
        "extension": extension,
    }


# Volatility-regime bucket thresholds: a label's win rate earned during a
# calm market can look very different from the same label during a wild one
# (wider stops get hit more, moves overshoot targets faster, etc). Bucketing
# by where current ATR% sits relative to its own trailing history lets the
# calibration answer "how has this label performed in conditions LIKE RIGHT
# NOW", not just an all-time blended average. Lives here (not in
# scanner.calibration, which imports it) so the live snapshot builder below
# can classify the regime for free off the same already-indicator-enriched
# dataframe, with no risk of a tradingagents -> scanner -> tradingagents
# import cycle.
_REGIME_LOW_PCTL = 1.0 / 3.0
_REGIME_HIGH_PCTL = 2.0 / 3.0


def regime_for_window(atr_pct_series: pd.Series, lookback: int = 100) -> str | None:
    """Classify the most recent value of ``atr_pct_series`` as "low"/"normal"/
    "high" relative to its own trailing ``lookback`` window — point-in-time,
    using only values up to and including the last one (no lookahead)."""
    if len(atr_pct_series) < max(20, lookback // 2):
        return None  # not enough trailing history yet to judge a regime
    window = atr_pct_series.iloc[-lookback:]
    current = window.iloc[-1]
    if pd.isna(current) or window.isna().all():
        return None
    percentile_rank = (window < current).mean()
    if percentile_rank <= _REGIME_LOW_PCTL:
        return "low"
    if percentile_rank >= _REGIME_HIGH_PCTL:
        return "high"
    return "normal"


def current_volatility_regime(df: pd.DataFrame, lookback: int = 100) -> str | None:
    """The live volatility regime ("low"/"normal"/"high"/None) for the most
    recent candle in ``df`` (already indicator-enriched via
    :func:`add_indicators`), so a live snapshot can be matched against
    regime-scoped calibration stats computed the same way by
    ``scanner.calibration.backtest_symbol``."""
    if "atr" not in df.columns or len(df) < 2:
        return None
    atr_pct = (df["atr"] / df["close"] * 100.0).replace([float("inf"), float("-inf")], pd.NA)
    return regime_for_window(atr_pct.dropna(), lookback)


def drawdown_pct(df: pd.DataFrame, lookback: int = 50) -> float:
    """Percent pullback from the most recent local high (Lokalni Vrh) to the
    lowest point reached since (Trenutno Dno), within ``lookback`` candles
    (excludes the still-forming candle, same convention as
    :func:`support_resistance`).

    Pad (%) = ((Lokalni Vrh - Trenutno Dno) / Lokalni Vrh) * 100
    """
    window = df.iloc[-(lookback + 1):-1] if len(df) > lookback else df.iloc[:-1]
    if window.empty:
        return 0.0
    peak_idx = window["high"].idxmax()
    peak = float(window.loc[peak_idx, "high"])
    if peak <= 0:
        return 0.0
    trough = float(window.loc[peak_idx:, "low"].min())
    return round((peak - trough) / peak * 100.0, 2)

