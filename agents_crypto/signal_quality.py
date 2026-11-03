"""Signal Quality Score (0-100): ranks already statistically-qualified
candidates (see ``agents_crypto.best_signal.compute_candidates``) by how
strongly LIVE conditions confirm their label right now, not just by how
good that label has looked on average across all of history.

Why this is needed: a calibration label such as "trend:bullish" blends
every historical moment that trend classifier ever fired for a symbol on a
timeframe into one win rate. Two live moments carrying that same label can
look very different in practice -- one with a decisive EMA stack, strong
volume, and every other tracked timeframe agreeing; another barely past the
trend threshold, thin volume, and timeframes pulling in different
directions. The calibration stats alone can't tell these apart; this score
is the live-conditions layer that does.

This NEVER replaces the statistical gate already in ``compute_candidates``
(minimum sample size, CI-low win rate > 50%, not muted by drift detection)
-- it only re-ranks candidates that already cleared it, so a label with no
proven historical edge can never outrank one that does just by looking
good right now. Every sub-score is defensive about missing snapshot fields
(older/thinner history, a brand-new listing, or a test fixture that only
sets a few fields): missing data always falls back to a neutral 50, never
a crash or a silent zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agents_crypto.risk_calc import (
    DEFAULT_SL_ATR_MULTIPLIER,
    ELEVATED_VOLATILITY_RISK_PCT,
    EXTREME_VOLATILITY_RISK_PCT,
)
from scanner.btc_context import trend_score

# Relative weight of each factor in the final 0-100 score -- the starting
# proposal from the project brief. Must sum to 1.0; intended to be tuned
# against historical/out-of-sample backtests later, not treated as final.
_WEIGHTS: dict[str, float] = {
    "trend_strength": 0.20,
    "momentum": 0.15,
    "volume_confirmation": 0.15,
    "multi_timeframe_alignment": 0.20,
    "volatility_conditions": 0.10,
    "historical_performance": 0.20,
}
assert abs(sum(_WEIGHTS.values()) - 1.0) < 1e-9, "quality score weights must sum to 1.0"


@dataclass(frozen=True)
class QualityScoreBreakdown:
    """Each sub-score is already 0-100 *before* weighting -- ``total`` is
    the weighted blend. Kept as a full breakdown (not just the total) so
    the dashboard can show *why* a candidate scored the way it did, not
    just one opaque number."""

    total: float
    trend_strength: float
    momentum: float
    volume_confirmation: float
    multi_timeframe_alignment: float
    volatility_conditions: float
    historical_performance: float


def _clip(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def _trend_strength_score(tf_snapshot: dict[str, Any], direction: str) -> float:
    """How decisively the EMA stack itself confirms the direction -- a
    bigger EMA20/EMA50 separation (relative to price) means a cleaner,
    more established trend, not one that just barely crossed the
    classifier's threshold. EMA200, when available (needs 200+ candles of
    history), adds a longer-horizon confirmation/contradiction swing."""
    close, ema20, ema50 = tf_snapshot.get("close"), tf_snapshot.get("ema20"), tf_snapshot.get("ema50")
    if not close or ema20 is None or ema50 is None:
        return 50.0  # not enough data to judge -- neutral, not penalized

    separation_pct = abs(ema20 - ema50) / close * 100.0
    # 0% separation -> 0 points; 1.5%+ separation -> fully maxed out.
    score = _clip(separation_pct / 1.5 * 100.0)

    ema200 = tf_snapshot.get("ema200")
    if ema200 is not None:
        confirms = (direction == "long" and close > ema200) or (direction == "short" and close < ema200)
        score = _clip(score + (15.0 if confirms else -15.0))
    return score


def _momentum_score(tf_snapshot: dict[str, Any], direction: str) -> float:
    """MACD histogram magnitude (relative to ATR, so it's comparable across
    symbols/timeframes of very different price/volatility scale) plus the
    fast EMA9/21 cross, both checked for agreement with ``direction``. A
    live ``momentum_fading`` flag (price making a new high/low without the
    MACD histogram confirming it -- the project's own existing bearish/
    bullish-divergence detector) is a direct penalty."""
    macd_hist, atr = tf_snapshot.get("macd_hist"), tf_snapshot.get("atr")
    if macd_hist is None or not atr:
        base = 50.0
    else:
        hist_ratio = macd_hist / atr
        aligned = (direction == "long" and hist_ratio > 0) or (direction == "short" and hist_ratio < 0)
        # |hist| >= 1x ATR is an unusually strong histogram reading -- the magnitude term caps there.
        magnitude = _clip(abs(hist_ratio) * 100.0)
        base = magnitude if aligned else _clip(50.0 - magnitude)

    ema_cross = tf_snapshot.get("ema_cross")
    if ema_cross in ("bullish", "bearish"):
        cross_aligned = (direction == "long" and ema_cross == "bullish") or (
            direction == "short" and ema_cross == "bearish"
        )
        base = _clip(base + (10.0 if cross_aligned else -10.0))

    if tf_snapshot.get("momentum_fading"):
        base = _clip(base - 25.0)
    return base


# The only anomaly label that directly measures "this move's own volume
# doesn't back it" -- see tradingagents.dataflows.vendors.binance.anomaly.
# There's no symmetric bearish-volume-contradiction label today, so a short
# candidate's volume score relies on relative_volume alone.
_VOLUME_CONTRADICTING_ANOMALY = {"long": "price_up_volume_not_confirming"}


def _volume_confirmation_score(tf_snapshot: dict[str, Any], direction: str) -> float:
    """Relative volume (vs. this symbol/timeframe's own trailing average) is
    the primary signal: a move on thin volume is much more likely to be
    noise than one with real participation behind it. The project's own
    ``price_up_volume_not_confirming`` anomaly is an additional direct
    penalty for a long candidate when active."""
    relative_volume = tf_snapshot.get("relative_volume")
    # <=0.5x average -> 0 points; >=2.0x average -> fully maxed out; unknown -> neutral.
    base = 50.0 if relative_volume is None else _clip((relative_volume - 0.5) / 1.5 * 100.0)

    contradicting_label = _VOLUME_CONTRADICTING_ANOMALY.get(direction)
    if contradicting_label and contradicting_label in (tf_snapshot.get("anomalies") or []):
        base = _clip(base - 30.0)
    return base


def _multi_timeframe_alignment_score(row: dict[str, Any], timeframe: str, direction: str) -> float:
    """Checks every *other* tracked timeframe's own trend against this
    candidate's direction -- a 1h long backed by 15m AND 4h also being
    bullish is a very different situation than one where 4h is outright
    bearish, even though both carry the exact same 1h calibration label."""
    from agents_crypto.best_signal import (
        TRACKED_TIMEFRAMES,  # local: avoids a module-load-order cycle, see module docstring
    )

    timeframes_snapshot = row.get("snapshot", {}).get("timeframes", {})
    candidate_sign = 1 if direction == "long" else -1

    other_scores = []
    for tf in TRACKED_TIMEFRAMES:
        if tf == timeframe:
            continue
        other_snapshot = timeframes_snapshot.get(tf)
        if not other_snapshot:
            continue
        other_scores.append(trend_score(other_snapshot.get("trend", "neutral")) * candidate_sign)

    if not other_scores:
        return 50.0  # no other timeframe data available yet -- neutral
    average = sum(other_scores) / len(other_scores)  # in [-1, 1]
    return _clip((average + 1.0) / 2.0 * 100.0)


def _volatility_conditions_score(tf_snapshot: dict[str, Any]) -> float:
    """Favors calmer volatility over extreme -- mirrors
    ``agents_crypto.risk_calc``'s own volatility-warning thresholds (the
    same ATR-based stop distance a live user would actually be shown), so a
    symbol already flagged there as having an unreliable ATR-sized stop
    scores lower here too, instead of the two disagreeing."""
    atr, close = tf_snapshot.get("atr"), tf_snapshot.get("close")
    if not atr or not close:
        return 50.0
    risk_pct = atr * DEFAULT_SL_ATR_MULTIPLIER / close * 100.0
    if risk_pct >= EXTREME_VOLATILITY_RISK_PCT:
        return 20.0
    if risk_pct >= ELEVATED_VOLATILITY_RISK_PCT:
        return 60.0
    return 100.0


def _historical_performance_score(timeframe: str, win_rate_ci_low_pct: float, n: int) -> float:
    """The CI-low win rate itself (already the conservative, never-the-raw-
    point-estimate number everything else in this project ranks by) plus a
    modest bonus for having well more than this timeframe's own required
    minimum sample size -- more evidence behind the same win rate is still
    worth something, even past the point the hard gate already clears."""
    from agents_crypto.best_signal import (  # local: avoids a module-load-order cycle, see module docstring
        _DEFAULT_MIN_SAMPLE_SIZE,
        BEST_SIGNAL_MIN_SAMPLE_SIZE,
    )

    base = _clip((win_rate_ci_low_pct - 50.0) / 25.0 * 100.0)
    min_n = BEST_SIGNAL_MIN_SAMPLE_SIZE.get(timeframe, _DEFAULT_MIN_SAMPLE_SIZE)
    sample_bonus = _clip(n / (min_n * 3.0) * 100.0)
    return _clip(base * 0.7 + sample_bonus * 0.3)


def compute_quality_score(
    row: dict[str, Any],
    tf_snapshot: dict[str, Any],
    timeframe: str,
    direction: str,
    win_rate_ci_low_pct: float,
    n: int,
) -> QualityScoreBreakdown:
    """The full weighted 0-100 Signal Quality Score for one candidate that
    has *already* passed ``compute_candidates``'s statistical gate.

    ``row`` is the full per-symbol scan row (needed for multi-timeframe
    alignment, which looks at every *other* tracked timeframe's own
    snapshot); ``tf_snapshot`` is this candidate's own timeframe snapshot.
    """
    trend_strength = _trend_strength_score(tf_snapshot, direction)
    momentum = _momentum_score(tf_snapshot, direction)
    volume_confirmation = _volume_confirmation_score(tf_snapshot, direction)
    multi_timeframe_alignment = _multi_timeframe_alignment_score(row, timeframe, direction)
    volatility_conditions = _volatility_conditions_score(tf_snapshot)
    historical_performance = _historical_performance_score(timeframe, win_rate_ci_low_pct, n)

    total = (
        trend_strength * _WEIGHTS["trend_strength"]
        + momentum * _WEIGHTS["momentum"]
        + volume_confirmation * _WEIGHTS["volume_confirmation"]
        + multi_timeframe_alignment * _WEIGHTS["multi_timeframe_alignment"]
        + volatility_conditions * _WEIGHTS["volatility_conditions"]
        + historical_performance * _WEIGHTS["historical_performance"]
    )

    return QualityScoreBreakdown(
        total=round(total, 1),
        trend_strength=round(trend_strength, 1),
        momentum=round(momentum, 1),
        volume_confirmation=round(volume_confirmation, 1),
        multi_timeframe_alignment=round(multi_timeframe_alignment, 1),
        volatility_conditions=round(volatility_conditions, 1),
        historical_performance=round(historical_performance, 1),
    )
