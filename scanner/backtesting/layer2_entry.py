"""Strategy Lab Korak 10 item 5/6: the Layer 2 ENTRY threshold (deciding
whether a Layer 1 config is even allowed to enter the confirmation queue)
and the Holm/Bonferroni multiple-testing correction applied to its
eventual confirmatory CI.

Both are deliberately simple, explicitly documented approximations --
consistent with the rest of Strategy Lab's "honest, crude, clearly labeled"
statistical style (see ``strategy_miner.compute_stability_score``'s own
docstring for the same philosophy) rather than a from-scratch rigorous
multiple-hypothesis-testing framework.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# How many independent backtest events (trades/rebalance periods) a
# config's LATEST evaluation must have before it's even considered for
# Layer 2, on top of the consecutive-positive-windows check below. Not a
# formal power calculation (that's ``power.required_sample_size``'s job,
# applied separately once a candidate is already in the confirmation
# queue) -- this is a cheap pre-filter against "positive by luck on a
# handful of events" before paying for a full power-gate/confirmation
# cycle. 30 is the common rule-of-thumb floor for a sample mean's
# distribution to start looking approximately normal (central limit
# theorem heuristic), and matches the smallest sample size already
# treated as meaningful elsewhere in this codebase's block-bootstrap
# commentary.
MIN_N = 30

# "Pozitivan neto u >= 3 uzastopna as_of_ts prozora" (Korak 10 item 5):
# the config's most recent ENTRY_WINDOW evaluations (one per distinct
# as_of_ts) must ALL be net-positive.
ENTRY_WINDOW = 3

# "|BTC beta| < 0.3" (Korak 10 item 5) -- same threshold the cross-
# sectional engine itself already flags via ``beta_check``
# (cross_sectional_engine.BETA_FLAG_THRESHOLD), reused here rather than
# re-invented.
BETA_THRESHOLD = 0.3

# Base per-test false-positive rate before correction. 5% is the
# conventional default used elsewhere in this codebase's statistical
# validation (``scanner.validation.statistical_validation.benjamini_hochberg``'s
# own default).
BASE_ALPHA = 0.05


@dataclass(frozen=True)
class EntryThresholdResult:
    """Whether one config currently qualifies to be frozen into the Layer
    2 confirmation queue, and why/why not -- ``reason`` is always a short,
    human-readable string suitable for direct display, never raised as an
    exception (failing the threshold is the normal, expected outcome)."""

    qualifies: bool
    windows_checked: int
    windows_positive: int
    reason: str


def check_entry_threshold(
    recent_results: list[dict[str, Any]],
    *,
    latest_btc_beta: float | None,
    min_n: int = MIN_N,
    entry_window: int = ENTRY_WINDOW,
    beta_threshold: float = BETA_THRESHOLD,
) -> EntryThresholdResult:
    """``recent_results`` must be newest-first (as returned by
    ``strategy_lab_storage.fetch_recent_results_for_config``), each a dict
    with ``net_observed_pct``, ``gross_observed_pct``, ``n_independent``.

    All of the following must hold for ``qualifies`` to be True:
    * at least ``entry_window`` distinct evaluations exist yet
    * EVERY one of the most recent ``entry_window`` has
      ``net_observed_pct > 0`` (a positive streak, not just "mostly
      positive")
    * the LATEST evaluation has ``n_independent >= min_n``
    * the LATEST evaluation has ``gross_observed_pct > 0``
    * ``abs(latest_btc_beta) < beta_threshold`` (if beta is unavailable,
      the config fails this check rather than passing by default --
      trend_anomaly's report never carries a beta, which is fine since
      it's ``closed_prior`` and never meant to reach Layer 2 anyway)
    """
    windows_checked = min(len(recent_results), entry_window)
    window = recent_results[:entry_window]
    windows_positive = sum(1 for r in window if (r.get("net_observed_pct") or 0.0) > 0)

    if len(recent_results) < entry_window:
        return EntryThresholdResult(
            qualifies=False,
            windows_checked=windows_checked,
            windows_positive=windows_positive,
            reason=f"only {len(recent_results)} of {entry_window} required as_of_ts windows observed so far",
        )
    if windows_positive < entry_window:
        return EntryThresholdResult(
            qualifies=False,
            windows_checked=windows_checked,
            windows_positive=windows_positive,
            reason=f"only {windows_positive} of the last {entry_window} windows were net-positive",
        )

    latest = recent_results[0]
    n_independent = latest.get("n_independent") or 0
    if n_independent < min_n:
        return EntryThresholdResult(
            qualifies=False,
            windows_checked=windows_checked,
            windows_positive=windows_positive,
            reason=f"latest window has only n={n_independent}, below MIN_N={min_n}",
        )
    gross = latest.get("gross_observed_pct")
    if gross is None or gross <= 0:
        return EntryThresholdResult(
            qualifies=False,
            windows_checked=windows_checked,
            windows_positive=windows_positive,
            reason=f"latest gross_observed_pct is {gross!r}, not positive",
        )
    if latest_btc_beta is None or abs(latest_btc_beta) >= beta_threshold:
        return EntryThresholdResult(
            qualifies=False,
            windows_checked=windows_checked,
            windows_positive=windows_positive,
            reason=f"latest |BTC beta| is {latest_btc_beta!r}, not below {beta_threshold}",
        )

    return EntryThresholdResult(
        qualifies=True,
        windows_checked=windows_checked,
        windows_positive=windows_positive,
        reason=f"{windows_positive}/{entry_window} windows positive, n={n_independent}, "
        f"gross>0, |beta|={abs(latest_btc_beta):.3f} < {beta_threshold}",
    )


def holm_corrected_confidence_pct(candidates_confirmed_on_same_window: int, *, base_alpha: float = BASE_ALPHA) -> float:
    """Bonferroni-style correction of the block-bootstrap CI's confidence
    level, scaled by ``m`` = how many candidates are being confirmed
    against the SAME fresh Layer 2 holdout window in this run -- NOT
    Layer 1's all-time unique-configuration counter (that counter answers
    a different question, "how many configs are in the mined universe
    toward the 300 cap", and is scoped by ``strategy_miner`` /
    ``strategy_lab_scheduler`` separately; conflating the two made Layer 2
    needlessly stricter as Layer 1 mined more configs it never even
    offered to Layer 2 -- corrected per the actual Korak 10 agreement,
    overriding this module's own originally-written item 4 wording).

    Layer 2 only ever runs ONE confirmation per fresh window today (the
    oldest unconsumed frozen candidate -- see
    ``strategy_lab_storage.fetch_next_unconsumed_layer2_candidate``,
    consumed exactly once before the next one gets its own turn), so under
    the CURRENT architecture ``m`` is always 1 (base_alpha, unadjusted) --
    a full sequential Holm step-down table (which needs every candidate's
    p-value ranked together) doesn't apply cleanly to a single test
    anyway. Using the single-test Bonferroni bound (``alpha / m``) is the
    conservative, well-understood special case of Holm's correction for
    "one test examined out of m possible ones", and is deliberately named
    "Holm/Bonferroni" throughout this feature's docs for that reason --
    not a claim of implementing the full step-down procedure. If a future
    change ever confirms more than one candidate against the exact same
    window in one pass, ``m`` should be that count, not Layer 1's mining
    universe size.

    ``m = max(candidates_confirmed_on_same_window, 1)`` so an empty/
    near-empty count never divides by zero or produces an alpha above the
    base rate."""
    m = max(candidates_confirmed_on_same_window, 1)
    alpha_corrected = base_alpha / m
    alpha_corrected = min(max(alpha_corrected, 1e-6), base_alpha)
    return 100.0 * (1.0 - alpha_corrected)
