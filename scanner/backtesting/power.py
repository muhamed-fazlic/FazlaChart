"""Reusable sample-size (power) calculation, extracted out of the
frozen ``funding_extreme_backtest.py`` / ``cross_sectional_momentum_backtest.py``
modules, which each independently inline the same formula next to their
"undecidable" verdict (see their ``n_required`` computation). Those two
modules are left untouched (frozen, do-not-edit, per the project's "look
once" discipline) — this module exists so NEW code (Strategy Lab's Layer 2
confirmation gate) has one shared, tested place to compute "how many
independent events would we need to see to trust a detector of size
``target_edge_pct``", instead of a third copy-pasted inline formula.

Formula (standard two-sided z-test sample-size calculation for a mean):

    n_required = ceil( (z_alpha/2 + z_beta)^2 * sigma^2 / d^2 )

where ``sigma`` is the measured (sample) standard deviation of the
per-event return, ``d`` is the minimum effect size worth detecting
(``target_edge_pct``), ``z_alpha/2`` is the two-sided critical value for
significance level ``alpha`` and ``z_beta`` is the critical value for the
desired statistical power (``1 - beta``).

Defaults (``alpha=0.05``, ``power=0.80``, ``target_edge_pct=0.3``) match
the values already frozen into the two closed hypothesis families
(``Z_ALPHA_95 = 1.96``, ``Z_BETA_80 = 0.84``, ``TARGET_EDGE_PCT = 0.3``) so
a candidate's required-n is directly comparable to those families' numbers
in the research registry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Two-sided 95% significance (alpha=0.05) and 80% power (beta=0.20) --
# same constants already frozen into funding_extreme_backtest.py /
# cross_sectional_momentum_backtest.py's inline n_required calculation.
DEFAULT_Z_ALPHA = 1.96
DEFAULT_Z_BETA = 0.84
DEFAULT_TARGET_EDGE_PCT = 0.3


@dataclass(frozen=True)
class PowerResult:
    n_required: int
    sigma_pct: float
    target_edge_pct: float
    z_alpha: float
    z_beta: float

    def is_sufficient(self, n_independent: int) -> bool:
        """Whether ``n_independent`` observed events already meet or
        exceed ``n_required`` -- i.e. the sample is large enough that a
        "no edge" verdict would be decisive, not merely undecidable for
        lack of data."""
        return n_independent >= self.n_required


def required_sample_size(
    sigma_pct: float,
    *,
    target_edge_pct: float = DEFAULT_TARGET_EDGE_PCT,
    z_alpha: float = DEFAULT_Z_ALPHA,
    z_beta: float = DEFAULT_Z_BETA,
) -> PowerResult:
    """How many independent events are required to detect a mean effect
    of at least ``target_edge_pct`` with power ``1-beta`` at significance
    ``alpha``, given an already-measured ``sigma_pct`` (sample standard
    deviation of the per-event return, same units as ``target_edge_pct``).

    Raises ``ValueError`` for a non-positive ``sigma_pct``/``target_edge_pct``
    -- both must be strictly positive for the formula to mean anything
    (a zero-variance or zero-effect-size input is a caller bug, not a
    valid "n_required is undefined" case worth silently tolerating).
    """
    if sigma_pct <= 0:
        raise ValueError(f"sigma_pct must be positive, got {sigma_pct!r}")
    if target_edge_pct <= 0:
        raise ValueError(f"target_edge_pct must be positive, got {target_edge_pct!r}")
    n_required = math.ceil(((z_alpha + z_beta) ** 2) * (sigma_pct**2) / (target_edge_pct**2))
    return PowerResult(
        n_required=n_required,
        sigma_pct=sigma_pct,
        target_edge_pct=target_edge_pct,
        z_alpha=z_alpha,
        z_beta=z_beta,
    )


def estimate_days_needed(
    *,
    n_required: int,
    n_independent_so_far: int,
    events_per_day: float,
) -> float:
    """Rough, descriptive-only estimate of how many MORE calendar days of
    fresh data are needed before ``n_required`` independent events have
    likely accumulated, extrapolating linearly from the candidate's own
    historically-measured event rate (``events_per_day``, e.g. from its
    Layer 1 training-period mining run). Returns ``0.0`` once
    ``n_independent_so_far`` already meets ``n_required``.

    This is intentionally a simple linear projection, not a new
    statistical claim -- it only answers "how long do we expect to keep
    waiting", never "will it pass". A non-positive ``events_per_day``
    (candidate never fired in training) returns ``math.inf`` since no
    finite amount of waiting would help.
    """
    remaining = n_required - n_independent_so_far
    if remaining <= 0:
        return 0.0
    if events_per_day <= 0:
        return math.inf
    return remaining / events_per_day
