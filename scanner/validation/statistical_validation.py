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

Deliberately dependency-free: uses stdlib ``statistics.NormalDist`` for the
normal CDF/inverse-CDF instead of scipy, since this project has no numpy/
scipy dependency elsewhere -- keeps this importable on a small VPS without
adding packages for maths that is already in the standard library.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any

_NORMAL = NormalDist()
_EULER_MASCHERONI = 0.5772156649015329


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
        tried ``n_trials`` variants and gotten lucky with this one."""
        return self.probabilistic_sharpe_ratio > 0.95


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
