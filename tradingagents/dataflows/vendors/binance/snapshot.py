"""Build the structured per-symbol/per-timeframe market snapshot.

This is the JSON contract between the objective data layer (Binance klines +
derivatives) and anything that interprets it — an AI agent, a rule engine, or
the backtester. Keeping this a plain dict (no LLM calls, no prose) is the
point: the scanner measures the market; interpretation happens downstream.
"""

from __future__ import annotations

from typing import Annotated, Any

from tradingagents.dataflows.vendors.binance import client
from tradingagents.dataflows.vendors.binance.anomaly import detect_anomalies
from tradingagents.dataflows.vendors.binance.indicators import (
    add_indicators,
    classify_trend,
    drawdown_pct,
    ema_cross,
    fibonacci_levels,
    klines_to_dataframe,
    momentum_fading,
    relative_volume,
    support_resistance,
    volume_change_pct,
)
from tradingagents.dataflows.errors import NoMarketDataError, VendorUnavailableError
from tradingagents.dataflows.vendors.binance.liquidity_heatmap import compute_liquidity_heatmap
from tradingagents.dataflows.vendors.binance.orderbook_depth import compute_orderbook_walls
from tradingagents.dataflows.vendors.binance.pivots import pivot_points_from_ohlc


def build_timeframe_snapshot(
    symbol: Annotated[str, "futures symbol, e.g. INJUSDT"],
    timeframe: Annotated[str, "kline interval, e.g. 15m, 1h, 4h"],
    limit: int = 250,
    funding_rate: float = 0.0,
    open_interest_change_pct_60m: float | None = None,
) -> dict[str, Any]:
    """Objective OHLCV + indicator facts for one symbol on one timeframe.

    ``funding_rate``/``open_interest_change_pct_60m`` are symbol-level (not
    timeframe-specific) derivatives facts, passed in so this timeframe's
    anomaly check can use them without a second network round-trip; see
    :func:`build_symbol_snapshot`.
    """
    klines = client.get_klines(symbol, timeframe, limit=limit)
    df = klines_to_dataframe(klines)
    df = add_indicators(df)
    last = df.iloc[-1]
    support, resistance = support_resistance(df)
    fib = fibonacci_levels(df)

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "trend": classify_trend(df),
        "close": round(float(last["close"]), 8),
        "rsi": round(float(last["rsi"]), 2),
        "ema9": round(float(last["close_9_ema"]), 8),
        "ema20": round(float(last["close_20_ema"]), 8),
        "ema21": round(float(last["close_21_ema"]), 8),
        "ema50": round(float(last["close_50_ema"]), 8),
        "ema200": round(float(last["close_200_ema"]), 8) if len(df) >= 200 else None,
        "ema_cross": ema_cross(df),
        "macd": round(float(last["macd"]), 6),
        "macd_signal": round(float(last["macds"]), 6),
        "macd_hist": round(float(last["macdh"]), 6),
        "atr": round(float(last["atr"]), 8),
        "bollinger_upper": round(float(last["boll_ub"]), 8),
        "bollinger_lower": round(float(last["boll_lb"]), 8),
        "volume": float(last["volume"]),
        "relative_volume": round(relative_volume(df), 3),
        "volume_change_pct": round(volume_change_pct(df), 2),
        "support": round(support, 8),
        "resistance": round(resistance, 8),
        "drawdown_pct": drawdown_pct(df),
        "fibonacci": fib,
        "momentum_fading": momentum_fading(df),
        "anomalies": detect_anomalies(df, resistance, funding_rate, open_interest_change_pct_60m),
        "candle_open_time": last["open_time"].isoformat(),
    }


def build_derivatives_snapshot(symbol: Annotated[str, "futures symbol, e.g. INJUSDT"]) -> dict[str, Any]:
    """Funding rate + open-interest facts, independent of timeframe."""
    mark = client.get_mark_price_and_funding(symbol)
    oi_now = client.get_open_interest(symbol)
    oi_hist = client.get_open_interest_history(symbol, period="5m", limit=13)  # ~60 min of 5m buckets

    oi_value_now = float(oi_now["openInterest"])
    oi_changes = _oi_changes_over_windows(oi_hist, oi_value_now)

    return {
        "funding_rate": float(mark.get("lastFundingRate", 0.0)),
        "mark_price": float(mark.get("markPrice", 0.0)),
        "index_price": float(mark.get("indexPrice", 0.0)),
        "open_interest": oi_value_now,
        "open_interest_change_pct_5m": oi_changes.get(5),
        "open_interest_change_pct_15m": oi_changes.get(15),
        "open_interest_change_pct_60m": oi_changes.get(60),
    }


def build_daily_pivot_snapshot(symbol: Annotated[str, "futures symbol, e.g. INJUSDT"]) -> dict[str, float]:
    """Classic floor-trader pivot points (P, R1-3, S1-3) from the previous
    *completed* daily candle. Binance's most recent 1d kline is still forming
    intraday, so this uses the one before it; ``{}`` if there isn't one yet
    (brand-new listing)."""
    klines = client.get_klines(symbol, "1d", limit=3)
    if len(klines) < 2:
        return {}
    prev_day = klines[-2]
    return pivot_points_from_ohlc(float(prev_day["high"]), float(prev_day["low"]), float(prev_day["close"]))


def _oi_changes_over_windows(
    oi_hist: list[dict[str, Any]], oi_now: float, windows_minutes: tuple[int, ...] = (5, 15, 60)
) -> dict[int, float | None]:
    """Percent change of OI "now" vs each window, using 5-minute OI history
    buckets (``bucket_count = window_minutes // 5``)."""
    changes: dict[int, float | None] = {}
    for window in windows_minutes:
        bucket_count = max(1, window // 5)
        if len(oi_hist) <= bucket_count:
            changes[window] = None
            continue
        past_value = float(oi_hist[-1 - bucket_count]["sumOpenInterest"])
        changes[window] = round(((oi_now - past_value) / past_value) * 100.0, 3) if past_value else None
    return changes


def build_liquidity_heatmap_snapshot(
    symbol: Annotated[str, "futures symbol, e.g. INJUSDT"],
    timeframe: Annotated[str, "kline interval, e.g. 15m, 1h, 4h"] = "1h",
    limit: int = 200,
    bins: int = 24,
) -> dict[str, Any]:
    """Volume-profile proxy for where TP/SL orders likely cluster (see
    :mod:`tradingagents.dataflows.vendors.binance.liquidity_heatmap` for the
    approximation this is built on — it is not real order-book data)."""
    klines = client.get_klines(symbol, timeframe, limit=limit)
    df = klines_to_dataframe(klines)
    heatmap = compute_liquidity_heatmap(df, bins=bins)
    heatmap["symbol"] = symbol
    heatmap["timeframe"] = timeframe

    current_price = float(df["close"].iloc[-1])
    try:
        depth = client.get_order_book_depth(symbol)
        heatmap["orderbook_walls"] = compute_orderbook_walls(depth, current_price)
    except (NoMarketDataError, VendorUnavailableError) as exc:
        # Real depth is a nice-to-have on top of the OHLCV proxy, not a hard
        # requirement -- degrade gracefully rather than breaking the whole
        # heatmap response if Binance's depth endpoint hiccups.
        heatmap["orderbook_walls"] = None
        heatmap["orderbook_walls_error"] = str(exc)
    return heatmap


def build_symbol_snapshot(
    symbol: Annotated[str, "futures symbol, e.g. INJUSDT"],
    timeframes: Annotated[tuple[str, ...], "kline intervals to analyze"] = ("5m", "15m", "1h", "4h"),
) -> dict[str, Any]:
    """Full snapshot: one entry per timeframe plus the shared derivatives block.

    This is the structured, LLM-independent fact sheet described in the
    project brief — it is what gets fed to an interpreting agent or stored
    for later calibration, never a prediction by itself.
    """
    derivatives = build_derivatives_snapshot(symbol)
    pivot_points = build_daily_pivot_snapshot(symbol)
    timeframe_snapshots = {
        tf: build_timeframe_snapshot(
            symbol,
            tf,
            funding_rate=derivatives["funding_rate"],
            open_interest_change_pct_60m=derivatives["open_interest_change_pct_60m"],
        )
        for tf in timeframes
    }
    return {
        "symbol": symbol,
        "timeframes": timeframe_snapshots,
        "derivatives": derivatives,
        "pivot_points": pivot_points,
    }
