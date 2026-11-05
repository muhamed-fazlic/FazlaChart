"""Tests for scanner.backtesting.power -- the extracted n_required / power
calculation, cross-checked against the numbers already frozen into the two
closed hypothesis families (funding-extreme: sigma=2.0614 -> n_required=371;
cross-sectional momentum: sigma=1.1838 -> n_required=123), so this module is
provably a faithful extraction, not a new formula."""

from __future__ import annotations

import math

import pytest

from scanner.backtesting.power import (
    DEFAULT_TARGET_EDGE_PCT,
    DEFAULT_Z_ALPHA,
    DEFAULT_Z_BETA,
    estimate_days_needed,
    required_sample_size,
)


class TestRequiredSampleSize:
    def test_matches_funding_extreme_family(self) -> None:
        result = required_sample_size(2.0614)
        assert result.n_required == 371

    def test_matches_cross_sectional_momentum_family(self) -> None:
        result = required_sample_size(1.1838)
        assert result.n_required == 123

    def test_defaults_match_frozen_constants(self) -> None:
        result = required_sample_size(1.0)
        assert result.z_alpha == DEFAULT_Z_ALPHA
        assert result.z_beta == DEFAULT_Z_BETA
        assert result.target_edge_pct == DEFAULT_TARGET_EDGE_PCT

    def test_manual_formula_cross_check(self) -> None:
        sigma = 1.5
        edge = 0.25
        expected = math.ceil(((1.96 + 0.84) ** 2) * (sigma**2) / (edge**2))
        result = required_sample_size(sigma, target_edge_pct=edge)
        assert result.n_required == expected

    def test_larger_sigma_requires_more_events(self) -> None:
        small = required_sample_size(1.0)
        large = required_sample_size(2.0)
        assert large.n_required > small.n_required

    def test_tighter_target_edge_requires_more_events(self) -> None:
        loose = required_sample_size(1.0, target_edge_pct=0.5)
        tight = required_sample_size(1.0, target_edge_pct=0.1)
        assert tight.n_required > loose.n_required

    def test_rejects_non_positive_sigma(self) -> None:
        with pytest.raises(ValueError):
            required_sample_size(0.0)
        with pytest.raises(ValueError):
            required_sample_size(-1.0)

    def test_rejects_non_positive_target_edge(self) -> None:
        with pytest.raises(ValueError):
            required_sample_size(1.0, target_edge_pct=0.0)

    def test_is_sufficient(self) -> None:
        result = required_sample_size(2.0614)
        assert result.is_sufficient(448) is True
        assert result.is_sufficient(371) is True
        assert result.is_sufficient(370) is False


class TestEstimateDaysNeeded:
    def test_already_sufficient_returns_zero(self) -> None:
        assert estimate_days_needed(n_required=100, n_independent_so_far=150, events_per_day=1.0) == 0.0
        assert estimate_days_needed(n_required=100, n_independent_so_far=100, events_per_day=1.0) == 0.0

    def test_linear_projection(self) -> None:
        # 50 more events needed, 2/day -> 25 days.
        days = estimate_days_needed(n_required=100, n_independent_so_far=50, events_per_day=2.0)
        assert days == pytest.approx(25.0)

    def test_zero_event_rate_is_infinite(self) -> None:
        days = estimate_days_needed(n_required=100, n_independent_so_far=0, events_per_day=0.0)
        assert days == math.inf

    def test_negative_event_rate_is_infinite(self) -> None:
        days = estimate_days_needed(n_required=100, n_independent_so_far=0, events_per_day=-1.0)
        assert days == math.inf
