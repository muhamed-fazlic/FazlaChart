"""Phase 8 Korak 7, Test 2: walk-forward (multi-period) validation.

``scanner.calibration.validate_out_of_sample`` (Korak 6's "Audit" sibling)
already splits history into one train window and one held-out test window —
but that is only ONE random look at "does the edge survive on unseen data".
A label can pass that single split purely by luck of where the 70/30
boundary happened to fall (e.g. the held-out 30% happens to land in a
market regime that favoured the label for unrelated reasons). Walk-forward
validation answers the stronger question the Phase 8 TODO's Korak 7 Test 2
asks for: does the edge hold up *consistently*, across several different
calendar periods, not just one.

Method: the full, time-ordered case list is split into ``n_folds + 1``
equal-count chronological chunks. Fold *k* (1-indexed) trains on every
chunk up to and including chunk *k-1* and tests on chunk *k* — an
**expanding (anchored) training window**, not a fixed-size rolling one, so
no extra "window size" parameter is needed and every fold's test chunk is
still data that fold's training chunk could never have seen. This directly
generalizes ``train_test_split_cases``'s single 70/30 split into N folds;
with ``n_folds=1`` the two splits are equivalent chunk boundaries (though
this module still answers a different, stricter question — see below).

A label only "holds up" (``holds_up_walk_forward=True``) if **every single
evaluated fold** confirms the edge (test window's 95% Wilson CI lower bound
> 50%), not merely a majority — deliberately the strictest, most
conservative criterion available, same bias as every other check in this
project (Korak 6's independence correction, the OOS single-split check):
never overstate confidence. A fold is "evaluated" only once it has at least
``min_test_n`` test cases; folds below that are reported but excluded from
the verdict, same floor and same reasoning as ``OOS_MIN_TEST_N`` /
``MIN_INDEPENDENT_EVENTS_N`` elsewhere. If fewer than
``min_folds_evaluated`` folds ever had enough data, no verdict is given at
all (too little evidence either way), rather than silently defaulting to
False/True.

Scope, deliberately bounded (documented, not hidden — same pattern as every
other Phase 8 deferral):

* This operates on **raw case counts per fold**, exactly like
  ``validate_out_of_sample`` does — it does NOT apply Korak 6's
  sample-independence correction inside each fold. The two checks are
  complementary, not substitutes: this one asks "does the (possibly
  overstated) raw win rate stay consistent across time", Korak 6 asks "is
  the apparent win rate overstated by correlated, double-counted cases in
  the first place". A label can pass this check for the wrong reason (the
  same handful of correlated market moves repeating in every fold) and
  still fail Korak 6's independence check — that is not a bug in either
  module, it is two different, deliberately narrow questions. Running
  independence-correction *inside* each already-smaller fold would shrink
  the usable sample further still (each fold already sees roughly ``1/
  (n_folds+1)`` of the data); that combined check is an explicit, larger
  stretch goal, not attempted here.
* Folds are built from the **pooled, cross-symbol, cross-label** case
  list's chronological order (mirrors ``validate_out_of_sample``'s own
  global 70/30 split) — i.e. every label's fold *k* test window is the same
  calendar period, not each label's own independent most-recent slice. This
  keeps "fold 3" meaning one consistent thing across every row of the
  resulting table.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from scanner.calibration import LabeledCase, summarize

# Same floor as calibration.OOS_MIN_TEST_N / independence.MIN_INDEPENDENT_EVENTS_N
# — below this many held-out cases, even a great-looking fold is too noisy.
WALK_FORWARD_MIN_TEST_N = 20
# Chunks the pooled history into n_folds + 1 equal-count pieces (one for the
# very first fold's training window, one test window per fold).
WALK_FORWARD_N_FOLDS = 4
# Require at least this many *scorable* folds (>= WALK_FORWARD_MIN_TEST_N
# test cases each) before a walk-forward verdict is trusted at all — one or
# two scorable folds out of four tells you almost nothing either way.
WALK_FORWARD_MIN_FOLDS_EVALUATED = 2


@dataclass
class WalkForwardFold:
    """One fold's held-out test-window result for a single (label, horizon)."""

    fold_index: int  # 1-based, chronological order
    train_n: int
    test_n: int
    test_win_rate_pct: float | None
    test_win_rate_ci_low_pct: float | None
    # False when test_n < the configured min_test_n — too sparse to count
    # toward (or against) the overall verdict, but still reported for
    # transparency rather than silently dropped.
    evaluated: bool
    holds_up: bool


@dataclass
class WalkForwardResult:
    """Walk-forward verdict for one (label, horizon_candles) group, pooled
    across every symbol/timeframe that fed the backtest."""

    label: str
    direction: str | None
    horizon_candles: int
    n_folds_total: int
    n_folds_evaluated: int
    n_folds_holding_up: int
    # True only when n_folds_evaluated >= min_folds_evaluated AND every
    # single one of those evaluated folds held up — see module docstring.
    holds_up_walk_forward: bool
    folds: list[WalkForwardFold] = field(default_factory=list)
    warning: str | None = None


def walk_forward_folds(
    cases: list[LabeledCase], n_folds: int = WALK_FORWARD_N_FOLDS
) -> list[tuple[list[LabeledCase], list[LabeledCase]]]:
    """Splits ``cases`` (sorted by entry_time) into ``n_folds`` expanding-
    window (train, test) pairs. Returns ``[]`` if there isn't even enough
    history for one case per chunk (``n_folds + 1`` chunks total)."""
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")
    ordered = sorted(cases, key=lambda c: c.entry_time)
    total = len(ordered)
    n_chunks = n_folds + 1
    if total < n_chunks:
        return []
    boundaries = [total * i // n_chunks for i in range(n_chunks + 1)]
    chunks = [ordered[boundaries[i] : boundaries[i + 1]] for i in range(n_chunks)]

    folds: list[tuple[list[LabeledCase], list[LabeledCase]]] = []
    cumulative = list(chunks[0])
    for k in range(n_folds):
        test_chunk = chunks[k + 1]
        folds.append((list(cumulative), test_chunk))
        cumulative = cumulative + test_chunk
    return folds


def validate_walk_forward(
    cases: list[LabeledCase],
    cost_pct: float,
    n_folds: int = WALK_FORWARD_N_FOLDS,
    min_test_n: int = WALK_FORWARD_MIN_TEST_N,
    min_folds_evaluated: int = WALK_FORWARD_MIN_FOLDS_EVALUATED,
) -> list[WalkForwardResult]:
    """Per (label, horizon) walk-forward verdict — see module docstring for
    the full method and its deliberately bounded scope.

    ``cost_pct`` must match whatever was used to grade these cases
    elsewhere in the same report — mirrors ``validate_out_of_sample``'s own
    simplification of a single flat cost, not per-symbol tiers.
    """
    if not cases:
        return []
    folds = walk_forward_folds(cases, n_folds)
    if not folds:
        return []

    by_key: dict[tuple[str, int], WalkForwardResult] = {}
    for fold_index, (train_cases, test_cases) in enumerate(folds, start=1):
        train_by_key = {(s.label, s.horizon_candles): s for s in summarize(train_cases, cost_pct)}
        test_by_key = {(s.label, s.horizon_candles): s for s in summarize(test_cases, cost_pct)}
        for key, test_stat in test_by_key.items():
            if test_stat.direction is None:
                continue  # undirected label: no win rate to validate either side
            train_stat = train_by_key.get(key)
            evaluated = test_stat.n >= min_test_n
            holds_up = bool(
                evaluated and test_stat.win_rate_ci_low_pct is not None and test_stat.win_rate_ci_low_pct > 50.0
            )
            result = by_key.setdefault(
                key,
                WalkForwardResult(
                    label=key[0],
                    direction=test_stat.direction,
                    horizon_candles=key[1],
                    n_folds_total=n_folds,
                    n_folds_evaluated=0,
                    n_folds_holding_up=0,
                    holds_up_walk_forward=False,
                ),
            )
            result.folds.append(
                WalkForwardFold(
                    fold_index=fold_index,
                    train_n=train_stat.n if train_stat else 0,
                    test_n=test_stat.n,
                    test_win_rate_pct=test_stat.win_rate_pct,
                    test_win_rate_ci_low_pct=test_stat.win_rate_ci_low_pct,
                    evaluated=evaluated,
                    holds_up=holds_up,
                )
            )

    results: list[WalkForwardResult] = []
    for result in by_key.values():
        result.folds.sort(key=lambda f: f.fold_index)
        evaluated_folds = [f for f in result.folds if f.evaluated]
        result.n_folds_evaluated = len(evaluated_folds)
        result.n_folds_holding_up = sum(1 for f in evaluated_folds if f.holds_up)
        if result.n_folds_evaluated < min_folds_evaluated:
            result.warning = (
                f"Samo {result.n_folds_evaluated}/{result.n_folds_total} foldova je imalo dovoljno slučajeva u "
                f"test prozoru (min. {min_test_n} po foldu) -- premalo za pouzdanu walk-forward provjeru "
                f"(potrebno \u2265{min_folds_evaluated} ocijenjenih foldova)."
            )
        elif result.n_folds_holding_up == result.n_folds_evaluated:
            result.holds_up_walk_forward = True
        else:
            result.warning = (
                f"Edge se NE drži konzistentno kroz vrijeme: samo {result.n_folds_holding_up}/"
                f"{result.n_folds_evaluated} ocijenjenih foldova je potvrdilo edge (potrebno SVI, ne samo "
                "jedan slučajan test prozor)."
            )
        results.append(result)
    return sorted(results, key=lambda r: (-r.n_folds_evaluated, r.label, r.horizon_candles))
