"""Tests for agents_crypto.signal_quality -- the live-conditions Signal
Quality Score computed on top of already statistically-qualified candidates.
"""

from agents_crypto.signal_quality import (
    QualityScoreBreakdown,
    _historical_performance_score,
    _momentum_score,
    _multi_timeframe_alignment_score,
    _trend_strength_score,
    _volatility_conditions_score,
    _volume_confirmation_score,
    compute_quality_score,
)


def _row(snapshot_by_tf: dict) -> dict:
    return {"snapshot": {"timeframes": snapshot_by_tf}}


# --- _trend_strength_score ---------------------------------------------


def test_trend_strength_neutral_when_emas_missing():
    assert _trend_strength_score({"close": 100.0}, "long") == 50.0


def test_trend_strength_rewards_wide_ema_separation_and_ema200_confirmation():
    snap = {"close": 100.0, "ema20": 102.0, "ema50": 100.5, "ema200": 95.0}
    score = _trend_strength_score(snap, "long")
    assert score > 50.0


def test_trend_strength_penalizes_ema200_contradiction():
    snap = {"close": 100.0, "ema20": 102.0, "ema50": 100.5, "ema200": 110.0}
    aligned = _trend_strength_score({"close": 100.0, "ema20": 102.0, "ema50": 100.5, "ema200": 95.0}, "long")
    contradicted = _trend_strength_score(snap, "long")
    assert contradicted < aligned


# --- _momentum_score ------------------------------------------------------


def test_momentum_neutral_when_macd_or_atr_missing():
    assert _momentum_score({"close": 100.0}, "long") == 50.0


def test_momentum_rewards_aligned_macd_histogram():
    snap = {"macd_hist": 0.5, "atr": 1.0}
    long_score = _momentum_score(snap, "long")
    short_score = _momentum_score(snap, "short")
    assert long_score > short_score


def test_momentum_fading_flag_penalizes_score():
    base_snap = {"macd_hist": 0.5, "atr": 1.0}
    fading_snap = {"macd_hist": 0.5, "atr": 1.0, "momentum_fading": True}
    assert _momentum_score(fading_snap, "long") < _momentum_score(base_snap, "long")


def test_momentum_ema_cross_alignment_bonus():
    snap_aligned = {"macd_hist": 0.0, "atr": 1.0, "ema_cross": "bullish"}
    snap_contradicted = {"macd_hist": 0.0, "atr": 1.0, "ema_cross": "bearish"}
    assert _momentum_score(snap_aligned, "long") > _momentum_score(snap_contradicted, "long")


# --- _volume_confirmation_score --------------------------------------------


def test_volume_confirmation_neutral_when_missing():
    assert _volume_confirmation_score({}, "long") == 50.0


def test_volume_confirmation_scales_with_relative_volume():
    low = _volume_confirmation_score({"relative_volume": 0.5}, "long")
    high = _volume_confirmation_score({"relative_volume": 2.0}, "long")
    assert low < high


def test_volume_confirmation_penalizes_contradicting_anomaly_for_long_only():
    snap = {"relative_volume": 1.0, "anomalies": ["price_up_volume_not_confirming"]}
    penalized = _volume_confirmation_score(snap, "long")
    unpenalized = _volume_confirmation_score({"relative_volume": 1.0}, "long")
    assert penalized < unpenalized
    # No symmetric short-side anomaly exists today -- short direction unaffected.
    assert _volume_confirmation_score(snap, "short") == unpenalized


# --- _multi_timeframe_alignment_score --------------------------------------


def test_multi_timeframe_alignment_neutral_when_no_other_timeframes():
    row = _row({"1h": {"trend": "bullish"}})
    assert _multi_timeframe_alignment_score(row, "1h", "long") == 50.0


def test_multi_timeframe_alignment_rewards_agreement():
    row = _row({"1h": {"trend": "bullish"}, "15m": {"trend": "bullish"}, "4h": {"trend": "bullish"}})
    score = _multi_timeframe_alignment_score(row, "1h", "long")
    assert score > 50.0


def test_multi_timeframe_alignment_penalizes_disagreement():
    row = _row({"1h": {"trend": "bullish"}, "15m": {"trend": "bearish"}, "4h": {"trend": "bearish"}})
    score = _multi_timeframe_alignment_score(row, "1h", "long")
    assert score < 50.0


# --- _volatility_conditions_score ------------------------------------------


def test_volatility_conditions_neutral_when_missing():
    assert _volatility_conditions_score({}) == 50.0


def test_volatility_conditions_favors_calm_over_extreme():
    calm = _volatility_conditions_score({"close": 100.0, "atr": 0.5})
    extreme = _volatility_conditions_score({"close": 100.0, "atr": 10.0})
    assert calm > extreme


# --- _historical_performance_score ------------------------------------------


def test_historical_performance_scales_with_ci_low_win_rate_and_sample_size():
    low_n = _historical_performance_score("1h", 55.0, 30)
    high_n = _historical_performance_score("1h", 55.0, 300)
    assert high_n > low_n

    low_win_rate = _historical_performance_score("1h", 51.0, 30)
    high_win_rate = _historical_performance_score("1h", 75.0, 30)
    assert high_win_rate > low_win_rate


# --- compute_quality_score (integration) ------------------------------------


def test_compute_quality_score_defensive_with_minimal_snapshot():
    """Matches the minimal {"trend", "close"} fixtures used throughout
    test_best_signal.py -- must never crash, and should land near neutral
    on every missing-data sub-score."""
    row = _row({"1h": {"trend": "bullish", "close": 100.0}})
    breakdown = compute_quality_score(row, {"trend": "bullish", "close": 100.0}, "1h", "long", 55.0, 40)
    assert isinstance(breakdown, QualityScoreBreakdown)
    assert 0.0 <= breakdown.total <= 100.0


def test_compute_quality_score_strong_conditions_outscore_weak_conditions():
    strong_snapshot = {
        "trend": "bullish",
        "close": 100.0,
        "ema20": 102.0,
        "ema50": 100.5,
        "ema200": 95.0,
        "macd_hist": 0.8,
        "atr": 1.0,
        "ema_cross": "bullish",
        "relative_volume": 2.5,
    }
    weak_snapshot = {
        "trend": "bullish",
        "close": 100.0,
        "ema20": 100.1,
        "ema50": 100.0,
        "macd_hist": -0.1,
        "atr": 1.0,
        "relative_volume": 0.4,
    }
    row = _row(
        {
            "1h": strong_snapshot,
            "15m": {"trend": "bullish"},
            "4h": {"trend": "bullish"},
        }
    )
    strong = compute_quality_score(row, strong_snapshot, "1h", "long", 65.0, 100)
    weak = compute_quality_score(_row({"1h": weak_snapshot}), weak_snapshot, "1h", "long", 51.0, 30)
    assert strong.total > weak.total
