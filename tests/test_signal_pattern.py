"""Tests for agents_crypto.signal_pattern -- the TRENDING/BREAKOUT/REVERSAL
classification of a qualified candidate's calibration label."""

from __future__ import annotations

from agents_crypto.signal_pattern import BREAKOUT, REVERSAL, TRENDING, classify_pattern
from scanner.calibration import LABEL_DIRECTION


def test_every_directional_label_has_an_explicit_pattern_classification():
    """Every label the backtest can assign a win rate to (i.e. every label
    with a declared long/short direction) must be explicitly classified --
    an unclassified label silently defaulting to TRENDING would be a
    correctness gap, not just a cosmetic one, so this is enforced here."""
    from agents_crypto.signal_pattern import _PATTERN_BY_LABEL

    directional_labels = [label for label, direction in LABEL_DIRECTION.items() if direction is not None]
    for label in directional_labels:
        assert label in _PATTERN_BY_LABEL, f"{label} has a declared direction but no pattern classification"


def test_trend_labels_classify_as_trending():
    assert classify_pattern("trend:bullish") == TRENDING
    assert classify_pattern("trend:bearish") == TRENDING


def test_pullback_confluence_classifies_as_trending_continuation():
    assert classify_pattern("anomaly:bullish_pullback_confluence") == TRENDING
    assert classify_pattern("anomaly:bearish_pullback_confluence") == TRENDING


def test_squeeze_breakout_and_ema_cross_classify_as_breakout():
    assert classify_pattern("anomaly:squeeze_breakout_long") == BREAKOUT
    assert classify_pattern("anomaly:squeeze_breakout_short") == BREAKOUT
    assert classify_pattern("anomaly:ema9_21_bullish_cross") == BREAKOUT
    assert classify_pattern("anomaly:ema9_21_bearish_cross") == BREAKOUT


def test_exhaustion_signals_classify_as_reversal():
    assert classify_pattern("anomaly:failed_breakout") == REVERSAL
    assert classify_pattern("anomaly:price_up_volume_not_confirming") == REVERSAL
    assert classify_pattern("anomaly:oi_rising_price_flat_funding_spiking") == REVERSAL


def test_unknown_label_defaults_to_trending_rather_than_crashing():
    assert classify_pattern("anomaly:some_future_label_nobody_registered_yet") == TRENDING
