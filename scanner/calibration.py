"""Calibration backtest: does a scanner label actually predict a profitable
forward move, measured against history — not a model's self-reported
confidence.

For each historical candle, this reconstructs exactly what the live scanner
would have seen *at that moment* (trend label + anomaly flags, using only
data up to and including that candle — no lookahead) and then checks what
price actually did ``horizon`` candles later, net of a round-trip trading
cost. Aggregating many of these across history answers: "in N historical
situations labeled X, what fraction were actually profitable?"

This deliberately reuses the exact same `classify_trend` / `support_resistance`
/ `detect_anomalies` functions the live scanner calls, just on a historical
window ending earlier in time, so the backtest can never silently drift from
what the scanner actually flags in production.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from agents_crypto.risk_calc import DEFAULT_SL_ATR_MULTIPLIER, DEFAULT_TP_ATR_MULTIPLIER
from tradingagents.dataflows.errors import VendorError
from tradingagents.dataflows.vendors.binance import client
from tradingagents.dataflows.vendors.binance.anomaly import detect_anomalies
from tradingagents.dataflows.vendors.binance.indicators import (
    add_indicators,
    classify_trend,
    klines_to_dataframe,
    support_resistance,
)

# ~2x Binance USDT-M futures taker fee (0.04% each way) plus a slippage margin.
# A "win" must clear this, not just point the right direction.
DEFAULT_ROUND_TRIP_COST_PCT = 0.10

# Explicit, pre-committed direction for every label the scanner can emit — the
# project brief's core rule: never let a label's "success" be invented after
# the fact. "long"/"short" means the label implies that bias; None means the
# label's direction is ambiguous (e.g. a volatility expansion can break either
# way), so only magnitude/frequency is reported, never a win rate.
LABEL_DIRECTION: dict[str, str | None] = {
    "trend:bullish": "long",
    "trend:bearish": "short",
    "trend:neutral": None,
    "anomaly:price_up_volume_not_confirming": "short",
    "anomaly:oi_rising_price_flat_funding_spiking": "short",
    "anomaly:failed_breakout": "short",
    "anomaly:volatility_squeeze_expansion": None,
    "anomaly:ema9_21_bullish_cross": "long",
    "anomaly:ema9_21_bearish_cross": "short",
    "anomaly:bullish_pullback_confluence": "long",
    "anomaly:bearish_pullback_confluence": "short",
}


@dataclass
class LabeledCase:
    symbol: str
    timeframe: str
    label: str
    entry_time: str
    entry_close: float
    forward_return_pct: float
    horizon_candles: int
    # Point-in-time volatility regime ("low"/"normal"/"high"), computed from
    # only the data available as of this candle — never leaks future info.
    # None for cases built by older code paths/tests that don't set it.
    regime: str | None = None
    # How the position was actually closed in the path-dependent walk-forward
    # (see _walk_forward_exit): "take_profit", "stop_loss", "timeout" (neither
    # level touched within the horizon), or None for undirected labels/cases
    # built directly by tests that don't set it.
    exit_reason: str | None = None


@dataclass
class CalibrationStats:
    label: str
    direction: str | None
    horizon_candles: int
    n: int
    win_rate_pct: float | None
    avg_net_return_pct: float
    avg_win_return_pct: float | None
    avg_loss_return_pct: float | None
    # 95% Wilson score confidence interval on win_rate_pct — None whenever
    # win_rate_pct itself is None (undirected label or n=0). A point estimate
    # like "65%" means something very different at n=20 (±21 points) than at
    # n=500 (±4 points); downstream consumers (webapp, risk agent) should
    # treat a wide interval as "not enough evidence yet", not a reliable edge.
    win_rate_ci_low_pct: float | None = None
    win_rate_ci_high_pct: float | None = None
    # Which volatility regime these stats are scoped to ("low"/"normal"/
    # "high"), or None when aggregated across all regimes (the historical
    # default, still produced unless a regime filter is requested).
    regime: str | None = None


# z-score for a 95% confidence level, used by the Wilson score interval below.
_WILSON_Z_95 = 1.959963985


def wilson_score_interval(win_rate_pct: float | None, n: int, z: float = _WILSON_Z_95) -> tuple[float | None, float | None]:
    """95%-confidence Wilson score interval for a win rate, given as a
    percentage, over ``n`` independent trials.

    Preferred over the naive normal-approximation interval because it stays
    well-behaved (doesn't exceed [0, 100] or collapse to a point) even at
    small ``n`` or at win rates near 0%/100% — exactly the regime where a
    freshly-observed anomaly label is most likely to sit before enough
    history accumulates.
    """
    if win_rate_pct is None or n <= 0:
        return None, None
    p = win_rate_pct / 100.0
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5)) / denom
    low = max(0.0, center - margin) * 100.0
    high = min(1.0, center + margin) * 100.0
    return round(low, 1), round(high, 1)


def _load_history(symbol: str, timeframe: str, limit: int = 1500) -> pd.DataFrame:
    klines = client.get_klines(symbol, timeframe, limit=limit)
    return add_indicators(klines_to_dataframe(klines))


def _forward_return_pct(df: pd.DataFrame, i: int, horizon: int) -> float | None:
    if i + horizon >= len(df):
        return None
    entry, exit_ = df["close"].iloc[i], df["close"].iloc[i + horizon]
    return (exit_ - entry) / entry * 100.0


def _walk_forward_exit(
    df: pd.DataFrame, i: int, direction: str, entry_close: float, atr: float, max_horizon: int
) -> tuple[int | None, float | None, str | None]:
    """Simulate the same SL/TP a live user would actually be shown (same ATR
    multipliers as :func:`agents_crypto.risk_calc.compute_risk_levels`) and
    scan forward candle-by-candle to find which is touched first.

    Returns ``(candles_to_exit, exit_price, exit_reason)``. ``candles_to_exit``
    is ``None`` and ``exit_reason`` is ``"timeout"`` when neither level is
    touched within ``max_horizon`` candles (the trade is still open at the
    point every caller-requested horizon must eventually resolve against the
    close price instead). If both the stop and the target fall within the
    same candle's high/low range, the stop is assumed to hit first — the
    conservative assumption, since intra-candle order can't be known from
    OHLC data alone.
    """
    if pd.isna(atr) or atr <= 0:
        return None, None, None

    sl_distance = atr * DEFAULT_SL_ATR_MULTIPLIER
    tp_distance = atr * DEFAULT_TP_ATR_MULTIPLIER
    if direction == "long":
        stop_loss = entry_close - sl_distance
        take_profit = entry_close + tp_distance
    else:
        stop_loss = entry_close + sl_distance
        take_profit = entry_close - tp_distance

    for h in range(1, max_horizon + 1):
        if i + h >= len(df):
            break
        low = float(df["low"].iloc[i + h])
        high = float(df["high"].iloc[i + h])
        if direction == "long":
            sl_touched, tp_touched = low <= stop_loss, high >= take_profit
        else:
            sl_touched, tp_touched = high >= stop_loss, low <= take_profit
        if sl_touched:
            return h, stop_loss, "stop_loss"
        if tp_touched:
            return h, take_profit, "take_profit"
    return None, None, "timeout"


# Volatility-regime bucket thresholds: a label's win rate earned during a
# calm market can look very different from the same label during a wild one
# (wider stops get hit more, moves overshoot targets faster, etc). Bucketing
# by where current ATR% sits relative to its own trailing history lets the
# calibration answer "how has this label performed in conditions LIKE RIGHT
# NOW", not just an all-time blended average.
_REGIME_LOW_PCTL = 1.0 / 3.0
_REGIME_HIGH_PCTL = 2.0 / 3.0


def _regime_for_window(atr_pct_series: pd.Series, lookback: int = 100) -> str | None:
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
    """Public entry point: the live volatility regime for the most recent
    candle in ``df`` (already indicator-enriched via :func:`add_indicators`),
    so a live snapshot can be matched against regime-scoped calibration
    stats computed the same way by :func:`backtest_symbol`."""
    if "atr" not in df.columns or len(df) < 2:
        return None
    atr_pct = (df["atr"] / df["close"] * 100.0).replace([float("inf"), float("-inf")], pd.NA)
    return _regime_for_window(atr_pct.dropna(), lookback)


def fetch_current_regime(symbol: str, timeframe: str, limit: int = 150) -> str | None:
    """Convenience wrapper: fetch just enough recent history to classify
    ``symbol``'s *current* volatility regime (cheap compared to the full
    1500-candle backtest fetch), for callers (e.g. the webapp) that only
    need "what regime are we in right now", not the full backtest."""
    try:
        df = _load_history(symbol, timeframe, limit)
    except VendorError:
        return None
    return current_volatility_regime(df)


def backtest_symbol(
    symbol: str,
    timeframe: str = "1h",
    horizons: tuple[int, ...] = (4, 12, 24),
    min_history: int = 60,
    limit: int = 1500,
) -> list[LabeledCase]:
    """Walk every candle in ``symbol``'s history, label it the way the live
    scanner would (trend + anomalies, point-in-time), and record the outcome
    at each horizon.

    For directional labels ("long"/"short"), the outcome is **path-dependent**:
    it reconstructs the same ATR-based SL/TP a live user would actually be
    shown (see :func:`_walk_forward_exit`) and checks which level is touched
    first, scanning forward candle-by-candle. If neither is touched within a
    given horizon, the trade is marked closed at that horizon's candle
    instead (same as the old point-to-point behaviour). This answers "would
    the stop or the target have been hit first", not just "was price higher
    N candles later" — a label can show a positive forward return while the
    stop was actually hit and then price recovered, which the old point-to-
    point measure couldn't distinguish.

    Undirected labels (no declared long/short bias) keep the original
    point-to-point magnitude, since there's no SL/TP side to simulate.
    One candle yields one case per label it carries, per horizon.
    """
    df = _load_history(symbol, timeframe, limit)
    atr_pct = (df["atr"] / df["close"] * 100.0).replace([float("inf"), float("-inf")], pd.NA)
    max_horizon = max(horizons)
    cases: list[LabeledCase] = []

    for i in range(min_history, len(df) - max_horizon):
        window = df.iloc[: i + 1]  # only data available "as of" candle i
        trend = classify_trend(window)
        _support, resistance = support_resistance(window)
        anomalies = detect_anomalies(window, resistance, funding_rate=0.0, oi_change_pct_60m=None)
        labels = [f"trend:{trend}"] + [f"anomaly:{a}" for a in anomalies]
        entry_time = str(window["open_time"].iloc[-1])
        entry_close = float(window["close"].iloc[-1])
        regime = _regime_for_window(atr_pct.iloc[: i + 1].dropna())
        entry_atr = float(df["atr"].iloc[i])

        # Path-dependent exit depends only on direction (long vs short), not
        # on which label implied it, so at most two walk-forwards per candle
        # are needed no matter how many directional labels fire together.
        exit_by_direction: dict[str, tuple[int | None, float | None, str | None]] = {}
        for direction in ("long", "short"):
            needed = any(LABEL_DIRECTION.get(label) == direction for label in labels)
            if needed:
                exit_by_direction[direction] = _walk_forward_exit(
                    df, i, direction, entry_close, entry_atr, max_horizon
                )

        for label in labels:
            direction = LABEL_DIRECTION.get(label)
            if direction is None:
                # Undirected label: no SL/TP side is defined, fall back to the
                # original point-to-point magnitude per horizon.
                for horizon in horizons:
                    forward_return = _forward_return_pct(df, i, horizon)
                    if forward_return is None:
                        continue
                    cases.append(
                        LabeledCase(
                            symbol, timeframe, label, entry_time, entry_close, forward_return, horizon, regime
                        )
                    )
                continue

            hit_candle, exit_price, reason = exit_by_direction.get(direction, (None, None, None))
            if reason is None:
                continue  # no usable ATR at this candle (NaN/zero) — skip
            for horizon in horizons:
                if hit_candle is not None and hit_candle <= horizon:
                    # SL/TP resolved the trade before the horizon's end.
                    forward_return = (exit_price - entry_close) / entry_close * 100.0
                    horizon_exit_reason = reason
                else:
                    # Neither level touched within this horizon: close it out
                    # at the horizon's candle, same as the old point-to-point
                    # behaviour, so win rate still reflects "did this trade
                    # eventually make money", not just "did it hit TP".
                    forward_return = _forward_return_pct(df, i, horizon)
                    if forward_return is None:
                        continue
                    horizon_exit_reason = "timeout"
                cases.append(
                    LabeledCase(
                        symbol,
                        timeframe,
                        label,
                        entry_time,
                        entry_close,
                        forward_return,
                        horizon,
                        regime,
                        horizon_exit_reason,
                    )
                )
    return cases


def summarize(
    cases: list[LabeledCase],
    cost_pct: float = DEFAULT_ROUND_TRIP_COST_PCT,
    regime_filter: str | None = None,
) -> list[CalibrationStats]:
    """Aggregate win-rate / average net return per (label, horizon).

    Funding-rate anomalies are excluded from this historical backtest: it
    replays klines only, not funding history, so ``oi_rising_price_flat_funding_spiking``
    will never fire here and is reported with n=0 if present — a genuine gap,
    noted so it is not mistaken for "never happens".

    ``regime_filter``, when given ("low"/"normal"/"high"), restricts the
    aggregation to cases recorded in that volatility regime only — i.e. "how
    has this label performed specifically when the market looked like it
    does right now", not blended across every market condition in history.
    Cases with an unknown regime (``None``, e.g. too little trailing history
    at backtest-start) are excluded whenever a filter is requested.
    """
    if not cases:
        return []
    if regime_filter is not None:
        cases = [c for c in cases if c.regime == regime_filter]
        if not cases:
            return []
    df = pd.DataFrame([c.__dict__ for c in cases])
    stats: list[CalibrationStats] = []
    for (label, horizon), group in df.groupby(["label", "horizon_candles"]):
        direction = LABEL_DIRECTION.get(label)
        if direction == "short":
            net_return = -group["forward_return_pct"] - cost_pct
        elif direction == "long":
            net_return = group["forward_return_pct"] - cost_pct
        else:
            net_return = None  # undirected: report magnitude only, no win rate

        if net_return is not None:
            wins = net_return > 0
            win_rate = round(wins.mean() * 100, 1)
            avg_win = round(net_return[wins].mean(), 3) if wins.any() else None
            avg_loss = round(net_return[~wins].mean(), 3) if (~wins).any() else None
            avg_net = round(net_return.mean(), 3)
        else:
            win_rate = None
            avg_win = avg_loss = None
            avg_net = round(group["forward_return_pct"].abs().mean(), 3)  # avg magnitude of the move either way

        ci_low, ci_high = wilson_score_interval(win_rate, len(group))
        stats.append(
            CalibrationStats(
                label=label,
                direction=direction,
                horizon_candles=int(horizon),
                n=len(group),
                win_rate_pct=win_rate,
                avg_net_return_pct=avg_net,
                avg_win_return_pct=avg_win,
                avg_loss_return_pct=avg_loss,
                win_rate_ci_low_pct=ci_low,
                win_rate_ci_high_pct=ci_high,
                regime=regime_filter,
            )
        )
    return sorted(stats, key=lambda s: (-s.n, s.label, s.horizon_candles))
