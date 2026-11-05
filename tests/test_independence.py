"""Unit tests for Phase 8 Korak 6's sample-independence correction.

All tests build synthetic ``LabeledCase`` lists directly -- no network
calls -- so the overlap/clustering logic and the corrected Wilson interval
can be checked against hand-computed expectations.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from scanner.calibration import LabeledCase
from scanner.validation.independence import (
    MIN_INDEPENDENT_EVENTS_N,
    cluster_independent_events,
    summarize_with_independence,
)

pytestmark = pytest.mark.unit

_T0 = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)


def _case(
    label: str,
    minutes_after_t0: int,
    forward_return_pct: float,
    *,
    symbol: str = "BTCUSDT",
    timeframe: str = "5m",
    horizon_candles: int = 12,
) -> LabeledCase:
    entry_time = (_T0 + timedelta(minutes=minutes_after_t0)).isoformat()
    return LabeledCase(
        symbol=symbol,
        timeframe=timeframe,
        label=label,
        entry_time=entry_time,
        entry_close=100.0,
        forward_return_pct=forward_return_pct,
        horizon_candles=horizon_candles,
        funding_cost_pct=0.0,
    )


def _cluster_groups(cases: list[LabeledCase]) -> list[set[int]]:
    """Helper: cluster ids are arbitrary integers assigned in sweep order,
    so tests compare *groupings* (which indices share a cluster), not the
    raw id values themselves."""
    ids = cluster_independent_events(cases)
    groups: dict[int, set[int]] = {}
    for i, cid in enumerate(ids):
        groups.setdefault(cid, set()).add(i)
    return sorted(groups.values(), key=lambda s: min(s))


class TestClusterIndependentEvents:
    def test_empty_input_returns_empty_list(self):
        assert cluster_independent_events([]) == []

    def test_single_case_is_its_own_cluster(self):
        cases = [_case("trend:bullish", 0, 5.0)]
        assert cluster_independent_events(cases) == [0]

    def test_non_overlapping_cases_get_separate_clusters(self):
        # 5m timeframe, horizon=12 candles -> ~60 minute window. Entries 10
        # hours apart can never overlap.
        cases = [
            _case("trend:bullish", 0, 5.0),
            _case("trend:bullish", 600, 5.0),
        ]
        assert _cluster_groups(cases) == [{0}, {1}]

    def test_overlapping_windows_on_different_symbols_merge_into_one_cluster(self):
        # Entries 5 minutes apart, both with ~60 minute windows -- these
        # overlap even though they're different symbols. Clustering is
        # deliberately cross-symbol (see module docstring): one BTC pump
        # firing the same label on several altcoins near-simultaneously is
        # one market event, not several independent confirmations.
        cases = [
            _case("trend:bullish", 0, 5.0, symbol="BTCUSDT"),
            _case("trend:bullish", 5, 5.0, symbol="ETHUSDT"),
        ]
        assert _cluster_groups(cases) == [{0, 1}]

    def test_transitive_chain_merges_into_one_cluster_even_if_ends_dont_overlap(self):
        # A (t=0, window ~60min ending t=60) overlaps B (t=50, ending
        # t=110). B overlaps C (t=100, ending t=160). A and C alone
        # (t=0 vs t=100) do not overlap directly, but they must still land
        # in the same cluster via B -- the connected-components definition.
        cases = [
            _case("trend:bullish", 0, 5.0, symbol="A"),
            _case("trend:bullish", 50, 5.0, symbol="B"),
            _case("trend:bullish", 100, 5.0, symbol="C"),
        ]
        assert _cluster_groups(cases) == [{0, 1, 2}]

    def test_input_order_does_not_affect_grouping(self):
        cases = [
            _case("trend:bullish", 100, 5.0, symbol="C"),
            _case("trend:bullish", 0, 5.0, symbol="A"),
            _case("trend:bullish", 50, 5.0, symbol="B"),
        ]
        groups = _cluster_groups(cases)
        assert groups == [{0, 1, 2}]

    def test_longer_timeframe_widens_the_overlap_window(self):
        # Same 65-minute gap: on 5m/horizon=12 (~60 min window) these do NOT
        # overlap, but on 1h/horizon=12 (~12 hour window) they do -- the
        # conservative longest-plausible-duration approximation (see module
        # docstring) scales with timeframe, as intended.
        short_tf_cases = [
            _case("trend:bullish", 0, 5.0, timeframe="5m", horizon_candles=12),
            _case("trend:bullish", 65, 5.0, timeframe="5m", horizon_candles=12),
        ]
        assert _cluster_groups(short_tf_cases) == [{0}, {1}]

        long_tf_cases = [
            _case("trend:bullish", 0, 5.0, timeframe="1h", horizon_candles=12),
            _case("trend:bullish", 65, 5.0, timeframe="1h", horizon_candles=12),
        ]
        assert _cluster_groups(long_tf_cases) == [{0, 1}]


class TestSummarizeWithIndependence:
    def test_undirected_label_is_skipped(self):
        cases = [_case("trend:neutral", m * 600, 1.0) for m in range(5)]
        assert summarize_with_independence(cases, cost_pct=0.1) == []

    def test_empty_input_returns_empty_list(self):
        assert summarize_with_independence([], cost_pct=0.1) == []

    def test_clustered_wins_reduce_independent_n_without_changing_raw_n(self):
        # 10 cases, all winners, but arriving in two tight clusters of 5
        # (same market move, five different symbols each time) ten hours
        # apart -- the raw count is 10, but there are only 2 independent
        # events.
        cases = []
        for cluster_start_minutes in (0, 600):
            for j, symbol in enumerate(["A", "B", "C", "D", "E"]):
                cases.append(
                    _case("trend:bullish", cluster_start_minutes + j, 5.0, symbol=symbol)
                )
        [stats] = summarize_with_independence(cases, cost_pct=0.1, min_independent_n=1)

        assert stats.label == "trend:bullish"
        assert stats.direction == "long"
        assert stats.n_raw == 10
        assert stats.win_rate_raw_pct == 100.0
        assert stats.n_independent_events == 2
        assert stats.win_rate_independent_pct == 100.0
        # Raw CI (n=10) must be tighter (higher lower-bound) than the
        # independence-corrected one (n=2) for the same 100% win rate --
        # fewer independent trials means less confidence, exactly the
        # point of this feature.
        assert stats.win_rate_raw_ci_low_pct > stats.win_rate_independent_ci_low_pct

    def test_mean_net_return_per_cluster_determines_cluster_outcome(self):
        # One cluster of 2 cases: one clear winner (net +4.9% after cost),
        # one narrow loser (net -0.1% after cost) -- the cluster's mean net
        # return should still be positive, so it counts as ONE independent
        # win, not a wash.
        win_case = _case("trend:bullish", 0, 5.0, symbol="A")
        loss_case = _case("trend:bullish", 1, 0.0, symbol="B")  # 0% move - cost = a loss
        [stats] = summarize_with_independence([win_case, loss_case], cost_pct=0.1, min_independent_n=1)

        assert stats.n_raw == 2
        assert stats.n_independent_events == 1
        assert stats.win_rate_independent_pct == 100.0  # the one cluster's mean return is positive

    def test_holds_up_independent_false_when_corrected_ci_low_at_or_below_50(self):
        # 30 raw cases, all wins, but clustered into just 3 independent
        # events -- not enough for the corrected Wilson CI lower bound to
        # clear 50%, even though the raw (uncorrected) one would.
        cases = []
        for cluster_start_minutes in (0, 600, 1200):
            for j, symbol in enumerate(["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]):
                cases.append(
                    _case("trend:bullish", cluster_start_minutes + j, 5.0, symbol=symbol)
                )
        [stats] = summarize_with_independence(cases, cost_pct=0.1, min_independent_n=1)

        assert stats.n_raw == 30
        assert stats.n_independent_events == 3
        assert stats.holds_up_independent is False
        assert stats.warning is not None

    def test_too_few_independent_events_sets_min_n_warning_not_holds_up(self):
        cases = [_case("trend:bullish", m * 600, 5.0, symbol=f"SYM{m}") for m in range(5)]
        [stats] = summarize_with_independence(cases, cost_pct=0.1, min_independent_n=MIN_INDEPENDENT_EVENTS_N)

        assert stats.n_independent_events == 5
        assert stats.holds_up_independent is False
        assert "premalo" in stats.warning

    def test_cost_pct_is_applied_to_the_independent_win_rate_too(self):
        # A 0.05% move is a win before cost but a loss after a 0.1% cost --
        # the independence-corrected win rate must reflect the same
        # cost-adjusted sign flip as the raw one, not an unadjusted one.
        cases = [_case("trend:bullish", m * 600, 0.05, symbol=f"SYM{m}") for m in range(3)]
        [stats] = summarize_with_independence(cases, cost_pct=0.1, min_independent_n=1)

        assert stats.win_rate_raw_pct == 0.0
        assert stats.win_rate_independent_pct == 0.0
