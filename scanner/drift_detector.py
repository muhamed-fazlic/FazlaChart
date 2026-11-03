"""Calibration drift guardrail: auto-mute a ``(timeframe, direction)``
bucket when its *live, realized* track record stops matching what it
promised, auto-restore it once the live evidence recovers.

This is deliberately the narrowest possible self-correction mechanism, not
a learning system:

* It never invents a new strategy, indicator, or ranking rule.
* It never touches the SL/TP logic, the Kelly formula, or how trend/regime
  is computed.
* It only ever flips one bit per bucket -- "currently recommended" vs
  "currently muted" -- based on a transparent, auditable comparison of two
  numbers any user can recompute by hand from ``signal_outcomes``: the
  promised win rate (the backtest's own 95%-CI lower bound, averaged over
  the resolved signals) against the realized win rate (the fraction of
  those same signals that actually closed with a positive net return).

Bucket granularity is ``(timeframe, direction)`` only -- not per-symbol and
not per-regime. Calibration itself is computed per-(symbol, timeframe,
label, regime), which is far too fine-grained for live drift detection: a
single altcoin in one regime might take months to accumulate the
:data:`MIN_RESOLVED_FOR_DRIFT_CHECK` resolved signals this needs. Grouping
across every symbol sharing a timeframe+direction trades some precision
(different coins/regimes get blended together) for actually being able to
reach a statistically usable sample size in a practical amount of time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from agents_crypto.best_signal import _DEFAULT_MIN_SAMPLE_SIZE, BEST_SIGNAL_MIN_SAMPLE_SIZE
from scanner.storage import (
    fetch_active_muted_bucket,
    fetch_all_active_muted_buckets,
    fetch_distinct_resolved_buckets,
    fetch_resolved_outcomes_by_bucket,
    mute_bucket,
    unmute_bucket,
)

logger = logging.getLogger(__name__)

# Reuses agents_crypto.best_signal.BEST_SIGNAL_MIN_SAMPLE_SIZE -- the same
# per-timeframe "don't trust a win rate below this many samples" bar the
# rest of the project already uses, applied here to *resolved live*
# signals instead of backtest cases. It matters just as much here: a
# resolved signal takes ``BEST_SIGNAL_HORIZON_CANDLES`` candles to close,
# so a 4h bucket accumulates live evidence ~48x slower than a 5m one --
# requiring the same flat count on both would mean a 4h bucket virtually
# never reaches enough history to ever be checked for drift at all, while
# a 5m bucket (noisier per-case, but resolving fast) still deserves the
# higher bar it gets in the backtest-case table.


def _min_resolved_for_check(timeframe: str) -> int:
    return BEST_SIGNAL_MIN_SAMPLE_SIZE.get(timeframe, _DEFAULT_MIN_SAMPLE_SIZE)


# Backward-compatible alias for the 1h bar specifically (unchanged from the
# original flat threshold) -- kept for any caller/test that only ever
# exercised the 1h bucket and doesn't need the full per-timeframe table.
MIN_RESOLVED_FOR_DRIFT_CHECK = BEST_SIGNAL_MIN_SAMPLE_SIZE["1h"]

# How many of the most recently resolved signals per bucket to judge by --
# a rolling window, not all-time history, so a bucket that was bad months
# ago but has since recovered reads as recovered.
DRIFT_WINDOW_SIZE = 60


# Realized win rate must fall at least this many percentage points short of
# the average promised (CI-low) win rate to count as "drifted" and trigger
# a mute. Deliberately generous (not a hair-trigger on ordinary variance)
# since the CI-low bound promised is already a conservative number.
DRIFT_THRESHOLD_PP = 12.0

# To restore a muted bucket, its rolling-window underperformance must shrink
# to less than half the mute threshold -- hysteresis so a bucket hovering
# right at the line doesn't flip muted/unmuted every cycle.
RECOVERY_THRESHOLD_PP = DRIFT_THRESHOLD_PP / 2.0


@dataclass(frozen=True)
class BucketDriftStats:
    timeframe: str
    direction: str
    n_resolved: int
    realized_win_rate_pct: float
    promised_win_rate_ci_low_pct_avg: float

    @property
    def drift_pp(self) -> float:
        """Positive means underperforming its promise; negative means
        beating it."""
        return self.promised_win_rate_ci_low_pct_avg - self.realized_win_rate_pct


def compute_bucket_drift(conn: Any, timeframe: str, direction: str) -> BucketDriftStats | None:
    """Realized vs. promised win rate for one bucket's rolling recent
    window of resolved signals. Returns ``None`` if fewer than this
    timeframe's minimum (see :func:`_min_resolved_for_check`) are available
    yet -- too little live evidence to judge either way."""
    resolved = fetch_resolved_outcomes_by_bucket(conn, timeframe, direction, limit=DRIFT_WINDOW_SIZE)
    if len(resolved) < _min_resolved_for_check(timeframe):
        return None

    wins = sum(1 for r in resolved if (r["realized_return_pct"] or 0.0) > 0.0)
    realized_win_rate_pct = wins / len(resolved) * 100.0

    promised_values = [r["promised_win_rate_ci_low_pct"] for r in resolved if r["promised_win_rate_ci_low_pct"] is not None]
    if not promised_values:
        return None
    promised_avg = sum(promised_values) / len(promised_values)

    return BucketDriftStats(
        timeframe=timeframe,
        direction=direction,
        n_resolved=len(resolved),
        realized_win_rate_pct=round(realized_win_rate_pct, 2),
        promised_win_rate_ci_low_pct_avg=round(promised_avg, 2),
    )


def evaluate_and_apply_drift_mutes(
    conn: Any,
    on_mute: Any = None,
    on_unmute: Any = None,
) -> dict[str, Any]:
    """Check every bucket with enough resolved live history and mute/unmute
    it as warranted. Meant to be called periodically from the scanner's
    main loop (similar cadence to :func:`scanner.storage.prune_old_scans`)
    -- cheap (a handful of indexed SQL queries), so safe to run every cycle,
    but doesn't need to.

    ``on_mute``/``on_unmute``, if given, are called with one
    :class:`BucketDriftStats` each time a bucket newly transitions state
    (e.g. to push a Telegram notice -- see
    :mod:`scanner.telegram_alerts`). A failing callback is logged and
    swallowed, never allowed to break the rest of the evaluation.

    Returns a summary dict (``muted``, ``unmuted`` symbol/direction lists)
    for logging/testing.
    """
    muted_now: list[str] = []
    unmuted_now: list[str] = []

    for timeframe, direction in fetch_distinct_resolved_buckets(conn):
        stats = compute_bucket_drift(conn, timeframe, direction)
        active_mute = fetch_active_muted_bucket(conn, timeframe, direction)

        if stats is None:
            continue  # not enough resolved history yet either way

        if active_mute is None:
            if stats.drift_pp >= DRIFT_THRESHOLD_PP:
                reason = (
                    f"realizovani win rate {stats.realized_win_rate_pct:.1f}% je "
                    f"{stats.drift_pp:.1f}pp ispod obecanih {stats.promised_win_rate_ci_low_pct_avg:.1f}% "
                    f"(n={stats.n_resolved} zatvorenih signala)"
                )
                mute_bucket(
                    conn,
                    timeframe=timeframe,
                    direction=direction,
                    reason=reason,
                    realized_win_rate_pct=stats.realized_win_rate_pct,
                    promised_win_rate_ci_low_pct_avg=stats.promised_win_rate_ci_low_pct_avg,
                    n_resolved=stats.n_resolved,
                )
                muted_now.append(f"{timeframe}/{direction}")
                _safe_callback(on_mute, stats)
        else:
            if stats.drift_pp < RECOVERY_THRESHOLD_PP:
                unmute_bucket(conn, active_mute["id"])
                unmuted_now.append(f"{timeframe}/{direction}")
                _safe_callback(on_unmute, stats)

    return {"muted": muted_now, "unmuted": unmuted_now}


def _safe_callback(callback: Any, stats: BucketDriftStats) -> None:
    if callback is None:
        return
    try:
        callback(stats)
    except Exception:
        logger.exception("drift mute/unmute callback failed for %s/%s; continuing", stats.timeframe, stats.direction)


def active_muted_bucket_set(conn: Any) -> set[tuple[str, str]]:
    """``(timeframe, direction)`` pairs currently muted, for
    :func:`agents_crypto.best_signal.compute_candidates`'s ``muted_buckets``
    argument."""
    return {(row["timeframe"], row["direction"]) for row in fetch_all_active_muted_buckets(conn)}
