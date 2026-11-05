"""Unit tests for Phase 8 Korak 7 Test 2's walk-forward validation.

All tests build synthetic ``LabeledCase`` lists directly -- no network
calls -- so the fold-splitting and per-fold verdict logic can be checked
against hand-computed expectations.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from scanner.calibration import LabeledCase
from scanner.validation.walk_forward import (
    WALK_FORWARD_MIN_TEST_N,
    validate_walk_forward,
    walk_forward_folds,
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


class TestWalkForwardFolds:
    def test_empty_input_returns_empty_list(self):
        assert walk_forward_folds([], n_folds=4) == []

    def test_rejects_non_positive_n_folds(self):
        with pytest.raises(ValueError):
            walk_forward_folds([_case("trend:bullish", 0, 1.0)], n_folds=0)

    def test_too_few_cases_for_requested_folds_returns_empty_list(self):
        # n_folds=4 needs 5 chunks; 4 cases can't even give one per chunk.
        cases = [_case("trend:bullish", i, 1.0) for i in range(4)]
        assert walk_forward_folds(cases, n_folds=4) == []

    def test_exact_chunk_boundaries_and_expanding_window(self):
        # 10 cases, n_folds=4 -> 5 equal chunks of 2 each.
        cases = [_case("trend:bullish", i, 1.0) for i in range(10)]
        folds = walk_forward_folds(cases, n_folds=4)
        assert len(folds) == 4
        # Fold 1: train = chunk0 (indices 0-1), test = chunk1 (indices 2-3).
        train1, test1 = folds[0]
        assert [c.entry_time for c in train1] == [cases[0].entry_time, cases[1].entry_time]
        assert [c.entry_time for c in test1] == [cases[2].entry_time, cases[3].entry_time]
        # Fold 2: train = chunks0-1 (indices 0-3), test = chunk2 (indices 4-5).
        train2, test2 = folds[1]
        assert len(train2) == 4
        assert [c.entry_time for c in test2] == [cases[4].entry_time, cases[5].entry_time]
        # Fold 4 (last): train = chunks0-3 (indices 0-7), test = chunk4 (indices 8-9).
        train4, test4 = folds[3]
        assert len(train4) == 8
        assert [c.entry_time for c in test4] == [cases[8].entry_time, cases[9].entry_time]

    def test_folds_are_chronologically_ordered_regardless_of_input_order(self):
        cases = [_case("trend:bullish", i, 1.0) for i in range(10)]
        shuffled = [cases[3], cases[0], cases[9], cases[1], cases[5], cases[2], cases[8], cases[4], cases[7], cases[6]]
        folds = walk_forward_folds(shuffled, n_folds=4)
        train1, test1 = folds[0]
        assert [c.entry_time for c in train1] == [cases[0].entry_time, cases[1].entry_time]
        assert [c.entry_time for c in test1] == [cases[2].entry_time, cases[3].entry_time]


class TestValidateWalkForward:
    def test_empty_input_returns_empty_list(self):
        assert validate_walk_forward([], cost_pct=0.1) == []

    def test_too_little_history_returns_empty_list(self):
        cases = [_case("trend:bullish", i, 1.0) for i in range(4)]
        assert validate_walk_forward(cases, cost_pct=0.1, n_folds=4) == []

    def test_undirected_label_is_skipped(self):
        # trend:neutral has no LABEL_DIRECTION entry -> no win/loss concept.
        cases = [_case("trend:neutral", i * 10, 1.0) for i in range(200)]
        results = validate_walk_forward(cases, cost_pct=0.1, n_folds=4, min_test_n=5, min_folds_evaluated=2)
        assert results == []

    def test_consistent_winner_holds_up_in_every_fold(self):
        # 200 "long" cases, each clearly profitable net of cost -> every
        # fold's test window should be a clean, high win-rate sample.
        cases = [_case("trend:bullish", i * 10, 5.0) for i in range(200)]
        results = validate_walk_forward(cases, cost_pct=0.1, n_folds=4, min_test_n=5, min_folds_evaluated=2)
        assert len(results) == 1
        result = results[0]
        assert result.label == "trend:bullish"
        assert result.direction == "long"
        assert result.n_folds_total == 4
        assert result.n_folds_evaluated == 4
        assert result.n_folds_holding_up == 4
        assert result.holds_up_walk_forward is True
        assert result.warning is None
        assert len(result.folds) == 4
        for fold in result.folds:
            assert fold.evaluated
            assert fold.holds_up

    def test_consistent_loser_never_holds_up(self):
        # All losing trades -> every fold's CI lower bound should sit well
        # under 50%.
        cases = [_case("trend:bearish", i * 10, 5.0) for i in range(200)]
        results = validate_walk_forward(cases, cost_pct=0.1, n_folds=4, min_test_n=5, min_folds_evaluated=2)
        assert len(results) == 1
        result = results[0]
        assert result.n_folds_evaluated == 4
        assert result.n_folds_holding_up == 0
        assert result.holds_up_walk_forward is False
        assert result.warning is not None
        assert "NE drži" in result.warning

    def test_edge_only_in_later_folds_fails_overall_verdict(self):
        # First half of history (folds 1-2's test windows) is a loser;
        # second half (folds 3-4's test windows) is a clean winner. A real
        # edge must hold in EVERY evaluated fold, not just recent ones.
        losing = [_case("trend:bullish", i * 10, -5.0) for i in range(100)]
        winning = [_case("trend:bullish", 1000 + i * 10, 5.0) for i in range(100)]
        cases = losing + winning
        results = validate_walk_forward(cases, cost_pct=0.1, n_folds=4, min_test_n=5, min_folds_evaluated=2)
        assert len(results) == 1
        result = results[0]
        assert result.n_folds_evaluated == 4
        assert result.n_folds_holding_up < result.n_folds_evaluated
        assert result.holds_up_walk_forward is False

    def test_insufficient_evaluated_folds_gives_no_verdict(self):
        # Only enough cases for 1 fold to clear min_test_n; default
        # min_folds_evaluated=2 means no verdict should be given.
        cases = [_case("trend:bullish", i * 10, 5.0) for i in range(6)]
        results = validate_walk_forward(cases, cost_pct=0.1, n_folds=4, min_test_n=5, min_folds_evaluated=2)
        assert len(results) == 1
        result = results[0]
        assert result.n_folds_evaluated < 2
        assert result.holds_up_walk_forward is False
        assert result.warning is not None
        assert "premalo" in result.warning

    def test_fold_below_min_test_n_is_reported_but_not_evaluated(self):
        cases = [_case("trend:bullish", i * 10, 5.0) for i in range(200)]
        results = validate_walk_forward(cases, cost_pct=0.1, n_folds=4, min_test_n=1000, min_folds_evaluated=2)
        result = results[0]
        assert result.n_folds_evaluated == 0
        for fold in result.folds:
            assert fold.evaluated is False
            assert fold.holds_up is False

    def test_default_min_test_n_constant_is_exported(self):
        assert WALK_FORWARD_MIN_TEST_N == 20
