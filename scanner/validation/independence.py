"""Phase 8 Korak 6: sample-independence correction for the Wilson interval.

``scanner.calibration.wilson_score_interval`` is documented as valid "over n
independent trials" -- but every caller that feeds it a (label, horizon)
group's raw case count (``scanner.calibration.summarize``,
``validate_out_of_sample``, the portfolio backtest's candidate-significance
filter, the live drift detector, telegram alert gating) is silently
violating that assumption whenever two or more of those cases were open in
the market *at the same time*. A single BTC pump can fire the same label on
five correlated altcoins within the same hour; that is one market event,
not five independent confirmations of the label's edge, yet today's code
counts it as five trials and reports a confidence interval that is far
tighter than the evidence actually supports.

This module adds a second, more conservative Wilson interval computed over
the number of *independent events* instead of raw case count, so a label
that "looks good" only because a handful of correlated market moves got
counted many times over can be told apart from one with a genuinely broad
base of unrelated confirmations.

Scope, deliberately bounded (documented, not hidden -- same pattern as
Korak 5's two deferred items):

* Clustering is done *within* each ``(label, horizon_candles)`` group -- the
  same grouping ``calibration.summarize()``/``validate_out_of_sample()``
  already use -- not globally across every label/timeframe/symbol at once.
  A full cross-label event-clustering model (e.g. "this BTC crash caused
  one bearish-label cluster across 6 labels and 3 timeframes simultaneously")
  would be a materially larger, riskier refactor; it is an explicit stretch
  goal, not something silently skipped.
* This is the **backtest/offline side only**. The live ``signal_outcomes``
  tracking table (``scanner/storage.py``) is not touched here -- online
  clustering is harder (you cannot know a signal will overlap with a future
  one at the moment it opens) and live data accumulates far slower than the
  thousands of historical cases a backtest already has. The urgent question
  this module answers is whether the *already-computed* out-of-sample
  result is itself reliable; that only requires the offline side.
* A cluster's single representative outcome is the **mean net return across
  its member cases** -- symmetric and unbiased (neither "any win counts as a
  cluster win" nor "only the first case counts", both of which would bias
  the corrected win rate in a chosen direction). This is a deliberate
  design choice, not the only defensible one; revisit it if a different
  aggregation rule turns out to matter in practice.
* Exact exit time is not persisted on ``LabeledCase``, so each case's "live"
  window is approximated as ``[entry_time, entry_time + horizon_candles *
  timeframe_minutes]`` -- the *longest possible* duration, since an early
  SL/TP exit can only shorten the true window. This deliberately errs
  towards finding MORE overlap than may have truly existed, never less --
  the safe direction for a correction meant to stop overstating confidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from scanner.calibration import (
    LABEL_DIRECTION,
    LabeledCase,
    net_return_pct_series,
    wilson_score_interval,
)
from scanner.signal_tracking import TIMEFRAME_MINUTES

# Below this many independent events, even a great-looking corrected win
# rate is too noisy to call "confirmed" -- same floor used by
# calibration.OOS_MIN_TEST_N, for the same reason.
MIN_INDEPENDENT_EVENTS_N = 20


@dataclass
class IndependenceClusterStats:
    """Raw (today's) vs. independence-corrected win rate for one
    (label, horizon_candles) group."""

    label: str
    direction: str | None
    horizon_candles: int
    # What calibration.summarize()/validate_out_of_sample() report today,
    # repeated here so the two can be compared side by side.
    n_raw: int
    win_rate_raw_pct: float
    win_rate_raw_ci_low_pct: float | None
    # After collapsing time-overlapping cases (any symbol) into one event.
    n_independent_events: int
    win_rate_independent_pct: float | None
    win_rate_independent_ci_low_pct: float | None
    # True only when the corrected CI's lower bound itself clears 50% --
    # mirrors OutOfSampleResult.holds_up's "conservative bound, not point
    # estimate" philosophy.
    holds_up_independent: bool
    warning: str | None = None


def _case_window(case: LabeledCase) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Conservative (upper-bound) estimate of the time range ``case`` was
    "live" in the market. See the module docstring's last bullet for why
    this is deliberately the longest plausible window, not the true one."""
    start = pd.Timestamp(case.entry_time)
    minutes_per_candle = TIMEFRAME_MINUTES.get(case.timeframe, 0)
    end = start + pd.Timedelta(minutes=minutes_per_candle * case.horizon_candles)
    return start, end


def cluster_independent_events(cases: list[LabeledCase]) -> list[int]:
    """Assigns each case in ``cases`` a 0-based cluster id such that two
    cases land in the same cluster whenever their approximate "live"
    windows (see ``_case_window``) overlap in time -- *regardless of
    symbol*. Deliberately cross-symbol: the entire point is to catch one
    market-wide move (e.g. a BTC pump) firing the same label on several
    correlated altcoins near-simultaneously, which is one event, not many.

    Returns a list parallel to ``cases`` (same order, same length) giving
    each case's cluster id. Clustering is transitive: if A overlaps B and
    B overlaps C, all three land in one cluster even if A and C don't
    directly overlap -- the standard definition of a connected component
    of the interval-overlap graph.

    O(n log n): a single sort by window start, then one linear sweep
    merging a case into the running cluster whenever its start precedes the
    running cluster's latest end (the standard interval-merging algorithm).
    """
    if not cases:
        return []
    windows = [_case_window(c) for c in cases]
    order = sorted(range(len(cases)), key=lambda i: windows[i][0])

    cluster_ids = [-1] * len(cases)
    current_cluster = -1
    current_end: pd.Timestamp | None = None
    for i in order:
        start, end = windows[i]
        if current_end is None or start > current_end:
            current_cluster += 1
            current_end = end
        else:
            current_end = max(current_end, end)
        cluster_ids[i] = current_cluster
    return cluster_ids


def summarize_with_independence(
    cases: list[LabeledCase],
    cost_pct: float,
    min_independent_n: int = MIN_INDEPENDENT_EVENTS_N,
) -> list[IndependenceClusterStats]:
    """Per (label, horizon_candles) raw win rate (same numbers
    ``calibration.summarize()`` reports) next to an independence-corrected
    one, computed by collapsing time-overlapping cases into single events
    (mean net return across each cluster's members) before re-running the
    Wilson interval over the independent-event count.

    ``cost_pct`` must match whatever was used to grade these cases
    elsewhere in the same report (callers should pass the same flat
    ``round_trip_cost_pct`` that ``validate_out_of_sample`` receives --
    mirroring that function's existing simplification of not applying
    per-symbol cost tiers here either).

    Undirected labels (no long/short win-loss concept) are skipped: there
    is nothing to cluster a win rate for.
    """
    if not cases:
        return []
    df = pd.DataFrame([c.__dict__ for c in cases])
    results: list[IndependenceClusterStats] = []
    for (label, horizon), group in df.groupby(["label", "horizon_candles"]):
        direction = LABEL_DIRECTION.get(label)
        net_return = net_return_pct_series(direction, group, cost_pct)
        if net_return is None:
            continue  # undirected: no win/loss to correct

        n_raw = len(group)
        wins_raw = net_return > 0
        win_rate_raw = round(wins_raw.mean() * 100, 1)
        ci_raw_low, _ = wilson_score_interval(win_rate_raw, n_raw)

        group_cases = [cases[i] for i in group.index]
        cluster_ids = cluster_independent_events(group_cases)
        cluster_returns: dict[int, list[float]] = {}
        for net, cid in zip(net_return, cluster_ids, strict=True):
            cluster_returns.setdefault(cid, []).append(net)
        cluster_mean_returns = [sum(values) / len(values) for values in cluster_returns.values()]

        n_independent = len(cluster_mean_returns)
        wins_independent = sum(1 for r in cluster_mean_returns if r > 0)
        win_rate_independent = round(wins_independent / n_independent * 100, 1) if n_independent else None
        ci_independent_low, _ = wilson_score_interval(win_rate_independent, n_independent)

        warning: str | None = None
        holds_up = False
        if n_independent < min_independent_n:
            warning = (
                f"Samo {n_independent} nezavisnih događaja (od {n_raw} sirovih slučajeva) -- premalo za "
                f"pouzdanu provjeru (potrebno \u2265{min_independent_n})."
            )
        elif ci_independent_low is not None and ci_independent_low > 50.0:
            holds_up = True
        else:
            ci_txt = f"{ci_independent_low}%" if ci_independent_low is not None else "N/A"
            warning = (
                f"Edge se NE potvrđuje kad se vremenski preklapajući trejdovi (bilo kog simbola) broje kao "
                f"jedan događaj: donja granica 95% CI je {ci_txt} (sirovo, bez korekcije: {win_rate_raw}%, "
                f"n={n_raw})."
            )

        results.append(
            IndependenceClusterStats(
                label=label,
                direction=direction,
                horizon_candles=int(horizon),
                n_raw=n_raw,
                win_rate_raw_pct=win_rate_raw,
                win_rate_raw_ci_low_pct=ci_raw_low,
                n_independent_events=n_independent,
                win_rate_independent_pct=win_rate_independent,
                win_rate_independent_ci_low_pct=ci_independent_low,
                holds_up_independent=holds_up,
                warning=warning,
            )
        )
    return sorted(results, key=lambda r: (-r.n_raw, r.label, r.horizon_candles))
