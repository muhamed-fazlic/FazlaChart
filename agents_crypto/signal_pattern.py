"""Classifies a qualified candidate's calibration label into one of three
market-structure categories -- TRENDING, BREAKOUT, or REVERSAL -- purely as
extra context for a human reading the dashboard, not a separate filter or
ranking input. The statistical gate and the Signal Quality Score (see
``agents_crypto.signal_quality``) don't care which category a label falls
into; this only answers "what *kind* of situation is this", since the same
win rate can come from very different market behaviour:

- TRENDING: the broad trend classifier itself, or a pullback that
  continues an already-established trend. The position is "the existing
  move keeps going".
- BREAKOUT: a new move actually starting right now -- a squeeze finally
  resolving in a direction, or short-term momentum (EMA9/21) just
  flipping. The position is "something new is beginning".
- REVERSAL: signs the *current* move is exhausted or unconfirmed -- a
  breakout that failed and is reverting, a rally with no volume behind
  it, or open interest ballooning while price goes nowhere. The position
  is "what just happened is probably about to turn".

Every label the live scanner can emit is listed explicitly below (mirrored
by ``tests/test_signal_pattern.py``'s coverage check against
``scanner.calibration.LABEL_DIRECTION``) so a newly added label can never
silently fall through to a guessed default.
"""

from __future__ import annotations

TRENDING = "trending"
BREAKOUT = "breakout"
REVERSAL = "reversal"

_PATTERN_BY_LABEL: dict[str, str] = {
    "trend:bullish": TRENDING,
    "trend:bearish": TRENDING,
    "anomaly:bullish_pullback_confluence": TRENDING,
    "anomaly:bearish_pullback_confluence": TRENDING,
    "anomaly:squeeze_breakout_long": BREAKOUT,
    "anomaly:squeeze_breakout_short": BREAKOUT,
    "anomaly:ema9_21_bullish_cross": BREAKOUT,
    "anomaly:ema9_21_bearish_cross": BREAKOUT,
    "anomaly:failed_breakout": REVERSAL,
    "anomaly:price_up_volume_not_confirming": REVERSAL,
    "anomaly:oi_rising_price_flat_funding_spiking": REVERSAL,
}


def classify_pattern(label: str) -> str:
    """Never raises or returns None -- an unrecognised label (e.g. a test
    fixture, or a future label someone forgets to add here) falls back to
    TRENDING, the least specific/most conservative category, rather than
    breaking the candidate pipeline."""
    return _PATTERN_BY_LABEL.get(label, TRENDING)
