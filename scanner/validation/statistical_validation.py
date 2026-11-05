"""Phase 8 Korak 3: statistical validation of backtest results against
overfitting, multiple testing, and plain luck.

This module does NOT touch live trading or scanner code -- it is a pure
post-hoc analysis layer that takes already-computed backtest statistics
(a strategy's trade returns, a matrix of per-period performance across
multiple trial variants, or a list of p-values from separate hypothesis
tests) and answers: "how much of this looks skillful once we honestly
account for how many things we tried before finding it?"

Implements three independent checks, deliberately kept separate because
they answer different questions and have different inputs:

- :func:`deflated_sharpe_ratio` -- Bailey & Lopez de Prado (2014): the
  probability that the TRUE Sharpe ratio of a strategy is positive, after
  correcting the observed Sharpe for (a) how many strategy variants/trials
  were tried before this one was picked, and (b) non-Normality (skew/
  kurtosis) of the trade-return distribution, which a naive Sharpe ratio
  silently assumes away.
- :func:`probability_of_backtest_overfitting` -- Combinatorially Symmetric
  Cross-Validation (CSCV), Bailey, Borwein, Lopez de Prado & Zhu (2014):
  splits a matrix of per-trial returns into combinatorial train/test
  halves and measures how often the trial that looks best in-sample
  actually ranks below the out-of-sample median -- a direct, assumption-
  light empirical overfitting-probability estimate.
- :func:`benjamini_hochberg` -- false-discovery-rate correction for testing
  many hypotheses simultaneously (e.g. one per (symbol, timeframe, label)
  combination): standard correction for "if you test 50 things at 5%
  significance, roughly 2-3 will look significant by chance alone".
- :func:`block_bootstrap_confidence_interval` -- Phase 8 Korak 7 Test 3:
  moving-block bootstrap confidence interval for any summary statistic
  (mean return, Sharpe, etc.) computed over a chronologically-ordered
  series of trade returns. A naive bootstrap (resampling individual trades
  independently) silently assumes trades are i.i.d.; nearby trades often
  share the same market regime (a cluster of correlated moves, same as
  :mod:`scanner.validation.independence` addresses for raw win-rate
  counts), so resampling contiguous *blocks* of trades instead preserves
  whatever short-range serial correlation the real data has, giving an
  honestly wider (more conservative) confidence interval than treating
  every trade as independent evidence would.

Deliberately dependency-free: uses stdlib ``statistics.NormalDist`` for the
normal CDF/inverse-CDF instead of scipy, since this project has no numpy/
scipy dependency elsewhere -- keeps this importable on a small VPS without
adding packages for maths that is already in the standard library.
"""

from __future__ import annotations

import itertools
import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any

_NORMAL = NormalDist()
_EULER_MASCHERONI = 0.5772156649015329

# Phase 8 Korak 7 Test 3 defaults for block_bootstrap_confidence_interval,
# exported so callers (the portfolio backtest config) can reference the same
# constants instead of re-typing magic numbers -- same convention as
# scanner.validation.walk_forward's WALK_FORWARD_* constants.
BLOCK_BOOTSTRAP_N_RESAMPLES = 2000
BLOCK_BOOTSTRAP_CONFIDENCE_PCT = 95.0


@dataclass(frozen=True)
class DeflatedSharpeResult:
    """Result of :func:`deflated_sharpe_ratio`.

    ``probabilistic_sharpe_ratio`` (the DSR itself) is a probability in
    ``[0, 1]`` that the strategy's TRUE Sharpe ratio exceeds the deflated
    benchmark -- NOT a p-value and NOT the Sharpe ratio itself. Values
    above ~0.95 are the conventional "statistically convincing" threshold
    (see :attr:`likely_skill`).
    """

    observed_sharpe: float
    deflated_sharpe_benchmark: float
    probabilistic_sharpe_ratio: float
    n_trials: int
    n_observations: int
    skewness: float
    kurtosis: float

    @property
    def likely_skill(self) -> bool:
        """True if DSR > 0.95 -- the conventional threshold at which the
        observed edge is considered unlikely to be explained by having
        tried ``n_trials`` variants and gotten lucky with this one.

        Explicitly cast to native ``bool`` -- ``probabilistic_sharpe_ratio``
        can be a numpy scalar if it was derived from pandas/numpy arithmetic
        upstream, and a bare ``>`` comparison on one yields ``numpy.bool_``,
        which ``json.dump`` cannot serialize (and which, confusingly,
        reports its type name as plain "bool" in newer numpy versions)."""
        return bool(self.probabilistic_sharpe_ratio > 0.95)


def _skewness(values: Sequence[float]) -> float:
    n = len(values)
    if n < 3:
        return 0.0
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    if variance <= 0:
        return 0.0
    std = math.sqrt(variance)
    m3 = sum((v - mean) ** 3 for v in values) / n
    return m3 / (std**3)


def _kurtosis(values: Sequence[float]) -> float:
    """Raw (not excess) kurtosis -- a Normal distribution has kurtosis 3.0,
    matching the convention the DSR formula below expects."""
    n = len(values)
    if n < 4:
        return 3.0
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    if variance <= 0:
        return 3.0
    m4 = sum((v - mean) ** 4 for v in values) / n
    return m4 / (variance**2)


def expected_max_sharpe(n_trials: int, sharpe_variance_across_trials: float) -> float:
    """Expected value of the MAXIMUM Sharpe ratio across ``n_trials``
    independent attempts, assuming each trial's (non-annualized) Sharpe
    estimate is i.i.d. Normal with variance ``sharpe_variance_across_trials``
    (Bailey & Lopez de Prado 2014, eq. 7). This is the deflated benchmark
    SR* that a "best of N trials" result must clear -- NOT zero -- because
    the best of many noisy attempts is expected to look good even with no
    real edge at all.
    """
    if n_trials <= 1 or sharpe_variance_across_trials <= 0:
        return 0.0
    sigma = math.sqrt(sharpe_variance_across_trials)
    # Guard against N so large that 1 - 1/N or 1 - 1/(N*e) rounds to exactly
    # 1.0, at which point inv_cdf is undefined (asymptotes to +inf).
    p1 = min(1 - 1.0 / n_trials, 1 - 1e-12)
    p2 = min(1 - 1.0 / (n_trials * math.e), 1 - 1e-12)
    return sigma * ((1 - _EULER_MASCHERONI) * _NORMAL.inv_cdf(p1) + _EULER_MASCHERONI * _NORMAL.inv_cdf(p2))


def deflated_sharpe_ratio(
    trial_returns: Sequence[float],
    *,
    n_trials: int,
    sharpe_variance_across_trials: float | None = None,
    trial_sharpe_ratios: Sequence[float] | None = None,
) -> DeflatedSharpeResult:
    """Compute the Deflated Sharpe Ratio for ONE strategy's per-trade
    returns, correcting for having tried ``n_trials`` variants to find it.

    ``trial_returns`` should be the per-trade (or per-period) returns of
    the strategy actually selected -- its (non-annualized) Sharpe ratio is
    computed internally as ``mean(trial_returns) / stdev(trial_returns)``;
    callers that report an annualized Sharpe elsewhere should NOT pass an
    already-annualized series in, since skew/kurtosis correction assumes
    the raw per-observation distribution.

    ``sharpe_variance_across_trials`` is the empirical variance of the
    (non-annualized) Sharpe ratios across all ``n_trials`` attempted
    variants. Pass ``trial_sharpe_ratios`` (the full list of per-trial
    Sharpe ratios, this one included) to have the variance computed
    automatically; pass the variance directly if only a summary statistic
    is available. If neither is given, a variance of 0 is used, which
    collapses :func:`expected_max_sharpe` to 0 -- i.e. no benchmark
    inflation is applied even though ``n_trials`` was supplied. Prefer
    providing one of these two whenever ``n_trials > 1``.

    Raises ``ValueError`` if ``trial_returns`` is empty or has zero
    variance (Sharpe ratio undefined).
    """
    if not trial_returns:
        raise ValueError("trial_returns must be non-empty")
    n = len(trial_returns)
    mean = sum(trial_returns) / n
    variance = sum((r - mean) ** 2 for r in trial_returns) / n
    if variance <= 0:
        raise ValueError("trial_returns has zero variance; Sharpe ratio is undefined")
    std = math.sqrt(variance)
    observed_sharpe = mean / std
    skew = _skewness(trial_returns)
    kurt = _kurtosis(trial_returns)

    if sharpe_variance_across_trials is None:
        if trial_sharpe_ratios and len(trial_sharpe_ratios) > 1:
            sr_mean = sum(trial_sharpe_ratios) / len(trial_sharpe_ratios)
            sharpe_variance_across_trials = sum((s - sr_mean) ** 2 for s in trial_sharpe_ratios) / len(
                trial_sharpe_ratios
            )
        else:
            sharpe_variance_across_trials = 0.0

    benchmark = expected_max_sharpe(n_trials, sharpe_variance_across_trials) if n_trials > 1 else 0.0

    # Bailey & Lopez de Prado (2014), eq. 5: PSR against an arbitrary
    # benchmark SR*, correcting for skew/kurtosis of the return distribution.
    denom = math.sqrt(max(1 - skew * observed_sharpe + ((kurt - 1) / 4) * observed_sharpe**2, 1e-12))
    z = (observed_sharpe - benchmark) * math.sqrt(n - 1) / denom
    psr = _NORMAL.cdf(z)

    return DeflatedSharpeResult(
        observed_sharpe=observed_sharpe,
        deflated_sharpe_benchmark=benchmark,
        probabilistic_sharpe_ratio=psr,
        n_trials=n_trials,
        n_observations=n,
        skewness=skew,
        kurtosis=kurt,
    )


def probability_of_backtest_overfitting(
    performance_matrix: Sequence[Sequence[float]],
    n_splits: int = 10,
) -> dict[str, Any]:
    """Combinatorially Symmetric Cross-Validation (CSCV) estimate of the
    Probability of Backtest Overfitting (PBO), per Bailey, Borwein, Lopez
    de Prado & Zhu (2014).

    ``performance_matrix`` is a (periods x trials) grid: row ``i``, column
    ``j`` is trial ``j``'s return during period ``i`` (e.g. one row per
    week/month, one column per strategy variant or parameter combination
    that was backtested). ``n_splits`` (S, must be even) divides the
    periods into S equal contiguous blocks; every one of the C(S, S/2) ways
    to choose half the blocks as a "training" half (the complementary half
    becomes the "test" half) is evaluated. For each split, whichever trial
    looks best summed over the training half is checked against its rank
    in the TEST half's performance distribution -- a genuinely skillful
    strategy should usually still rank near the top out-of-sample too; an
    overfit one's out-of-sample rank should be no better than a coin flip.

    Returns a dict with:

    - ``pbo``: fraction of splits where the in-sample winner's logit-
      transformed out-of-sample rank is <= 0 (ranks at or below the test
      set's median) -- the headline overfitting probability. Values above
      ~0.5 mean the backtest's best-looking variant is no more likely than
      a coin flip to actually be good out-of-sample; values near 0 mean the
      in-sample winner reliably stays a winner.
    - ``n_splits``, ``n_combinations``: bookkeeping.
    - ``logits``: the raw per-split logit values, for callers that want the
      full distribution rather than just the binary PBO fraction.

    Raises ``ValueError`` for odd/degenerate ``n_splits``, fewer than 2
    trial columns, or too few periods to form ``n_splits`` equal blocks.
    """
    if n_splits % 2 != 0 or n_splits < 2:
        raise ValueError("n_splits must be an even number >= 2")
    n_periods = len(performance_matrix)
    n_trials = len(performance_matrix[0]) if n_periods else 0
    if n_trials < 2:
        raise ValueError("performance_matrix needs at least 2 trial columns")
    block_size = n_periods // n_splits
    if block_size < 1:
        raise ValueError("not enough periods to form n_splits equal blocks")

    # Any rows left over from the integer-division truncation are silently
    # dropped (kept out of every block) -- fine, since CSCV only needs
    # EQUAL-sized blocks, not full coverage of every period.
    blocks = [performance_matrix[i * block_size : (i + 1) * block_size] for i in range(n_splits)]

    half = n_splits // 2
    logits: list[float] = []
    for train_idx in itertools.combinations(range(n_splits), half):
        test_idx = [i for i in range(n_splits) if i not in train_idx]
        train_rows = [row for i in train_idx for row in blocks[i]]
        test_rows = [row for i in test_idx for row in blocks[i]]

        train_perf = [sum(row[j] for row in train_rows) for j in range(n_trials)]
        test_perf = [sum(row[j] for row in test_rows) for j in range(n_trials)]

        best_trial = max(range(n_trials), key=lambda j: train_perf[j])
        # Relative rank of the in-sample winner within the OOS performance
        # distribution (Bailey et al.'s w_c): fraction of trials it ties or
        # beats, continuity-corrected and clamped strictly inside (0, 1) so
        # the logit below is always defined.
        rank = sum(1 for v in test_perf if v <= test_perf[best_trial])
        relative_rank = (rank - 0.5) / n_trials
        relative_rank = min(max(relative_rank, 1e-6), 1 - 1e-6)
        logits.append(math.log(relative_rank / (1 - relative_rank)))

    pbo = sum(1 for lam in logits if lam <= 0) / len(logits)
    return {
        "pbo": pbo,
        "n_splits": n_splits,
        "n_combinations": len(logits),
        "logits": logits,
    }


def benjamini_hochberg(p_values: Sequence[float], alpha: float = 0.05) -> list[bool]:
    """Benjamini-Hochberg false-discovery-rate (FDR) correction.

    Returns a list (same order/length as ``p_values``) of booleans: ``True``
    means that hypothesis's p-value survives FDR correction at ``alpha``
    (still counts as a discovery after accounting for how many hypotheses
    were tested simultaneously -- e.g. one per (symbol, timeframe, label)
    combination), ``False`` means it does not.

    Standard step-up procedure: sort p-values ascending, find the largest
    rank ``k`` (1-indexed) such that ``p_(k) <= (k / m) * alpha``, and
    declare every hypothesis at or before that rank significant. An empty
    input returns an empty list.
    """
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    sorted_pvals = [p_values[i] for i in order]
    thresholds = [(k + 1) / m * alpha for k in range(m)]

    largest_k = -1
    for k in range(m):
        if sorted_pvals[k] <= thresholds[k]:
            largest_k = k

    significant = [False] * m
    for k in range(largest_k + 1):
        significant[order[k]] = True
    return significant


def _default_block_size(n: int) -> int:
    """Politis-White style heuristic (simplified, dependency-free): block
    length grows roughly as the cube root of the sample size, with a floor
    of 2 (a block of 1 degenerates to the naive i.i.d. bootstrap) and a
    cap so a single resample never collapses to re-running the whole
    series verbatim."""
    return max(2, min(round(n ** (1 / 3)), n // 2))


def block_bootstrap_confidence_interval(
    values: Sequence[float],
    *,
    statistic: Callable[[Sequence[float]], float] = lambda xs: sum(xs) / len(xs),
    block_size: int | None = None,
    n_resamples: int = BLOCK_BOOTSTRAP_N_RESAMPLES,
    confidence_pct: float = BLOCK_BOOTSTRAP_CONFIDENCE_PCT,
    seed: int | None = None,
) -> dict[str, Any]:
    """Moving-block bootstrap confidence interval for ``statistic`` applied
    to the chronologically-ordered series ``values`` (e.g. per-trade net
    return percentages, in execution order).

    Unlike a plain percentile bootstrap (which resamples individual
    observations with replacement, implicitly assuming they are
    independent), this resamples contiguous *blocks* of length
    ``block_size`` -- preserving whatever short-range serial correlation
    consecutive trades actually have (shared market regime, overlapping
    holding periods, etc.) instead of washing it out. ``block_size``
    defaults to a cube-root-of-n heuristic (see :func:`_default_block_size`)
    when not given explicitly.

    Procedure, repeated ``n_resamples`` times: draw
    ``ceil(len(values) / block_size)`` blocks (each a contiguous run of
    ``block_size`` consecutive values, start position chosen uniformly at
    random with replacement from every valid start position), concatenate
    them and truncate to ``len(values)``, then evaluate ``statistic`` on
    that resampled series. The confidence interval is the corresponding
    lower/upper percentile of the resulting distribution of statistics.

    Returns a dict with ``observed`` (the statistic on the real,
    unshuffled series), ``ci_low``/``ci_high``, ``confidence_pct``,
    ``block_size``, ``n_resamples``, and ``n_observations``.

    Raises ``ValueError`` if there are fewer than 4 values, ``block_size``
    is not a positive integer smaller than ``len(values)``, or
    ``confidence_pct`` is not strictly between 0 and 100.
    """
    n = len(values)
    if n < 4:
        raise ValueError("need at least 4 values for a block bootstrap")
    if not 0.0 < confidence_pct < 100.0:
        raise ValueError("confidence_pct must be strictly between 0 and 100")
    resolved_block_size = block_size if block_size is not None else _default_block_size(n)
    if resolved_block_size < 1 or resolved_block_size >= n:
        raise ValueError("block_size must be a positive integer smaller than len(values)")

    rng = random.Random(seed)
    n_blocks_needed = math.ceil(n / resolved_block_size)
    max_start = n - resolved_block_size  # inclusive

    bootstrap_stats: list[float] = []
    for _ in range(n_resamples):
        resampled: list[float] = []
        for _ in range(n_blocks_needed):
            start = rng.randint(0, max_start)
            resampled.extend(values[start : start + resolved_block_size])
        bootstrap_stats.append(statistic(resampled[:n]))

    bootstrap_stats.sort()
    tail = (100.0 - confidence_pct) / 2.0 / 100.0
    low_idx = int(tail * n_resamples)
    high_idx = min(n_resamples - 1, int((1.0 - tail) * n_resamples))

    return {
        "observed": statistic(values),
        "ci_low": bootstrap_stats[low_idx],
        "ci_high": bootstrap_stats[high_idx],
        "confidence_pct": confidence_pct,
        "block_size": resolved_block_size,
        "n_resamples": n_resamples,
        "n_observations": n,
    }


def safe_block_bootstrap(
    returns: Sequence[float], n_resamples: int, confidence_pct: float
) -> dict[str, Any] | None:
    """Wraps :func:`block_bootstrap_confidence_interval`, returning ``None``
    (rather than raising) when there are too few observations (< 4) for it
    to run, and renaming/rounding the raw result into the dashboard/report
    shape every backtest engine wants (``*_pct`` keys, rounded to 4 decimal
    places, plus ``edge_survives_lower_bound``).

    Extracted here because ``funding_extreme_backtest.py`` and
    ``cross_sectional_momentum_backtest.py`` each defined a byte-for-byte
    identical private copy of this function -- having it live once next to
    the primitive it wraps means any future change to the report shape (or
    to how "too few observations" is handled) only needs to happen once.
    """
    try:
        result = block_bootstrap_confidence_interval(
            returns, n_resamples=n_resamples, confidence_pct=confidence_pct
        )
    except ValueError:
        return None
    return {
        "observed_pct": round(result["observed"], 4),
        "ci_low_pct": round(result["ci_low"], 4),
        "ci_high_pct": round(result["ci_high"], 4),
        "confidence_pct": result["confidence_pct"],
        "block_size": result["block_size"],
        "n_resamples": result["n_resamples"],
        "edge_survives_lower_bound": bool(result["ci_low"] > 0.0),
    }
