"""Tests for scanner.backtesting.rolling_holdout -- the Layer 2 boundary
tracker. The single most important property under test:
``advance_boundary`` must NEVER allow a confirmation window to be reused
(the boundary can only move forward)."""

from __future__ import annotations

import pytest

from scanner.backtesting.rolling_holdout import (
    RollingHoldoutState,
    advance_boundary,
    fresh_window_available_ms,
    initial_boundary_ms,
)


class TestInitialBoundaryMs:
    def test_uses_now_when_later_than_shared_holdout(self) -> None:
        shared_holdout_end = 1_000_000
        now = 5_000_000
        assert initial_boundary_ms(shared_holdout_end, now_ms=now) == now

    def test_uses_shared_holdout_end_when_later_than_now(self) -> None:
        # Pathological/defensive case: should not happen in practice (the
        # locked holdout is always in the past), but never silently set a
        # boundary that is behind the locked holdout's own end.
        shared_holdout_end = 9_000_000
        now = 1_000_000
        assert initial_boundary_ms(shared_holdout_end, now_ms=now) == shared_holdout_end


class TestFreshWindowAvailableMs:
    def test_positive_elapsed_time(self) -> None:
        state = RollingHoldoutState(boundary_timestamp_ms=1_000_000)
        assert fresh_window_available_ms(state, now_ms=1_500_000) == 500_000

    def test_never_negative(self) -> None:
        state = RollingHoldoutState(boundary_timestamp_ms=2_000_000)
        assert fresh_window_available_ms(state, now_ms=1_000_000) == 0


class TestAdvanceBoundary:
    def test_moves_forward(self) -> None:
        state = RollingHoldoutState(boundary_timestamp_ms=1_000_000)
        new_state = advance_boundary(state, consumed_until_ms=2_000_000)
        assert new_state.boundary_timestamp_ms == 2_000_000

    def test_rejects_backward_move(self) -> None:
        state = RollingHoldoutState(boundary_timestamp_ms=2_000_000)
        with pytest.raises(ValueError):
            advance_boundary(state, consumed_until_ms=1_000_000)

    def test_rejects_no_op_move(self) -> None:
        state = RollingHoldoutState(boundary_timestamp_ms=2_000_000)
        with pytest.raises(ValueError):
            advance_boundary(state, consumed_until_ms=2_000_000)

    def test_cannot_reuse_a_consumed_window(self) -> None:
        """Simulates two confirmations in sequence: the second must not be
        able to start before the first one's consumed end."""
        state = RollingHoldoutState(boundary_timestamp_ms=1_000_000)
        state_after_first = advance_boundary(state, consumed_until_ms=1_500_000)
        # A second confirmation attempting to reuse any part of
        # [1_000_000, 1_500_000) must fail.
        with pytest.raises(ValueError):
            advance_boundary(state_after_first, consumed_until_ms=1_200_000)
        # Advancing past the first window is fine.
        state_after_second = advance_boundary(state_after_first, consumed_until_ms=2_000_000)
        assert state_after_second.boundary_timestamp_ms == 2_000_000
