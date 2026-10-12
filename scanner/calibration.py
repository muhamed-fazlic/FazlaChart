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


def _load_history(symbol: str, timeframe: str, limit: int = 1500) -> pd.DataFrame:
    klines = client.get_klines(symbol, timeframe, limit=limit)
    return add_indicators(klines_to_dataframe(klines))


def _forward_return_pct(df: pd.DataFrame, i: int, horizon: int) -> float | None:
    if i + horizon >= len(df):
        return None
    entry, exit_ = df["close"].iloc[i], df["close"].iloc[i + horizon]
    return (exit_ - entry) / entry * 100.0


def backtest_symbol(
    symbol: str,
    timeframe: str = "1h",
    horizons: tuple[int, ...] = (4, 12, 24),
    min_history: int = 60,
    limit: int = 1500,
) -> list[LabeledCase]:
    """Walk every candle in ``symbol``'s history, label it the way the live
    scanner would (trend + anomalies, point-in-time), and record the forward
    return at each horizon. One candle yields one case per label it carries.
    """
    df = _load_history(symbol, timeframe, limit)
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

        for horizon in horizons:
            forward_return = _forward_return_pct(df, i, horizon)
            if forward_return is None:
                continue
            for label in labels:
                cases.append(LabeledCase(symbol, timeframe, label, entry_time, entry_close, forward_return, horizon))
    return cases


def summarize(cases: list[LabeledCase], cost_pct: float = DEFAULT_ROUND_TRIP_COST_PCT) -> list[CalibrationStats]:
    """Aggregate win-rate / average net return per (label, horizon).

    Funding-rate anomalies are excluded from this historical backtest: it
    replays klines only, not funding history, so ``oi_rising_price_flat_funding_spiking``
    will never fire here and is reported with n=0 if present — a genuine gap,
    noted so it is not mistaken for "never happens".
    """
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
            )
        )
    return sorted(stats, key=lambda s: (-s.n, s.label, s.horizon_candles))
