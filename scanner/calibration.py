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
    current_volatility_regime,
    klines_to_dataframe,
    regime_for_window,
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
    "anomaly:squeeze_breakout_long": "long",
    "anomaly:squeeze_breakout_short": "short",
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
    # Stop-loss distance at entry, as a percentage of entry_close (same ATR
    # multiplier the live system would have shown — see _walk_forward_exit).
    # None for undirected labels (no SL/TP side exists) or cases built by
    # older code paths/tests that don't set it. Lets a consumer (e.g.
    # scanner.backtesting.portfolio_backtest) convert a raw percentage
    # outcome into a risk-normalized "R-multiple" without needing to
    # re-derive ATR itself.
    sl_distance_pct: float | None = None


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


@dataclass
class OutOfSampleResult:
    """Did a label's edge survive on data the "model" never got to fit to?

    Splits backtest cases by *time*, not a random shuffle: the oldest
    ``1 - test_fraction`` of cases become the train window, the most recent
    ``test_fraction`` become a held-out test window. This answers "would
    this edge still have looked real in the period right before now", which
    is what actually matters for trading it going forward — a random IID
    split would let information from the test period's neighbours leak into
    training and overstate how well the label generalizes.
    """

    label: str
    direction: str | None
    horizon_candles: int
    train_n: int
    train_win_rate_pct: float | None
    test_n: int
    test_win_rate_pct: float | None
    test_win_rate_ci_low_pct: float | None
    # True only when the held-out test window's conservative (CI lower
    # bound) win rate itself clears 50% — i.e. the edge isn't just an
    # in-sample artifact of whatever data it was originally measured on.
    holds_up: bool
    warning: str | None = None


# Below this many held-out cases, even a great-looking test win rate is too
# noisy to call "confirmed" — same floor used elsewhere (best-signal ranking)
# for consistency.
OOS_MIN_TEST_N = 20
# Most-recent fraction of cases held out as the "test" window.
OOS_TEST_FRACTION = 0.3


def train_test_split_cases(
    cases: list[LabeledCase], test_fraction: float = OOS_TEST_FRACTION
) -> tuple[list[LabeledCase], list[LabeledCase]]:
    """Chronological train/test split — see :class:`OutOfSampleResult`."""
    ordered = sorted(cases, key=lambda c: c.entry_time)
    split_idx = int(len(ordered) * (1 - test_fraction))
    return ordered[:split_idx], ordered[split_idx:]


def validate_out_of_sample(
    cases: list[LabeledCase],
    cost_pct: float = DEFAULT_ROUND_TRIP_COST_PCT,
    test_fraction: float = OOS_TEST_FRACTION,
    min_test_n: int = OOS_MIN_TEST_N,
) -> list[OutOfSampleResult]:
    """Per (label, horizon) comparison of in-sample (train) vs. held-out
    (test) win rate — flags labels that only "worked" on the data used to
    measure them, not on more recent history."""
    if not cases:
        return []
    train_cases, test_cases = train_test_split_cases(cases, test_fraction)
    train_by_key = {(s.label, s.horizon_candles): s for s in summarize(train_cases, cost_pct)}
    test_by_key = {(s.label, s.horizon_candles): s for s in summarize(test_cases, cost_pct)}

    results: list[OutOfSampleResult] = []
    for key, test_stat in test_by_key.items():
        if test_stat.direction is None:
            continue  # undirected label: no win rate to validate either side
        train_stat = train_by_key.get(key)
        warning = None
        holds_up = False
        if test_stat.n < min_test_n:
            warning = (
                f"Test prozor ima samo {test_stat.n} slučajeva — premalo za pouzdanu OOS provjeru "
                f"(potrebno \u2265{min_test_n})."
            )
        elif test_stat.win_rate_ci_low_pct is not None and test_stat.win_rate_ci_low_pct > 50.0:
            holds_up = True
        else:
            ci_low_txt = f"{test_stat.win_rate_ci_low_pct}%" if test_stat.win_rate_ci_low_pct is not None else "N/A"
            warning = (
                f"Edge se NE potvrđuje na najnovijem (held-out) periodu: donja granica 95% CI win rate-a je "
                f"{ci_low_txt} — label je možda bio dobar samo na starijim podacima (in-sample), ne i sada."
            )
        results.append(
            OutOfSampleResult(
                label=key[0],
                direction=test_stat.direction,
                horizon_candles=key[1],
                train_n=train_stat.n if train_stat else 0,
                train_win_rate_pct=train_stat.win_rate_pct if train_stat else None,
                test_n=test_stat.n,
                test_win_rate_pct=test_stat.win_rate_pct,
                test_win_rate_ci_low_pct=test_stat.win_rate_ci_low_pct,
                holds_up=holds_up,
                warning=warning,
            )
        )
    return sorted(results, key=lambda r: (-r.test_n, r.label, r.horizon_candles))


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
        regime = regime_for_window(atr_pct.iloc[: i + 1].dropna())
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
            sl_distance_pct = entry_atr * DEFAULT_SL_ATR_MULTIPLIER / entry_close * 100.0
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
                        sl_distance_pct,
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
