"""Unit tests for Phase 8 Korak 3's statistical validation helpers:
Deflated Sharpe Ratio, Probability of Backtest Overfitting (CSCV), and
Benjamini-Hochberg FDR correction. All pure-maths, no network/filesystem
calls."""

from __future__ import annotations

import math
import random

import pytest

from scanner.validation.statistical_validation import (
    benjamini_hochberg,
    block_bootstrap_confidence_interval,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probability_of_backtest_overfitting,
)

pytestmark = pytest.mark.unit


class TestExpectedMaxSharpe:
    def test_single_trial_has_no_deflation(self):
        assert expected_max_sharpe(1, sharpe_variance_across_trials=1.0) == 0.0

    def test_zero_variance_has_no_deflation(self):
        assert expected_max_sharpe(50, sharpe_variance_across_trials=0.0) == 0.0

    def test_more_trials_inflates_the_benchmark(self):
        few = expected_max_sharpe(5, sharpe_variance_across_trials=1.0)
        many = expected_max_sharpe(500, sharpe_variance_across_trials=1.0)
        assert many > few > 0

    def test_extremely_large_n_does_not_raise(self):
        # Regression guard for the inv_cdf(p=1.0) edge case.
        value = expected_max_sharpe(10_000_000, sharpe_variance_across_trials=1.0)
        assert math.isfinite(value)


class TestDeflatedSharpeRatio:
    def test_empty_returns_raises(self):
        with pytest.raises(ValueError):
            deflated_sharpe_ratio([], n_trials=1)

    def test_zero_variance_returns_raises(self):
        with pytest.raises(ValueError):
            deflated_sharpe_ratio([0.01, 0.01, 0.01], n_trials=1)

    def test_single_trial_degenerates_to_plain_psr(self):
        # With n_trials=1 the deflation benchmark is forced to 0 regardless
        # of what trial_sharpe_ratios would imply.
        returns = [0.01, 0.02, -0.005, 0.015, 0.008, -0.002, 0.012] * 5
        result = deflated_sharpe_ratio(returns, n_trials=1)
        assert result.deflated_sharpe_benchmark == 0.0
        assert 0.0 <= result.probabilistic_sharpe_ratio <= 1.0
        assert result.n_observations == len(returns)

    def test_more_trials_lowers_the_dsr_for_the_same_returns(self):
        rng = random.Random(42)
        returns = [rng.gauss(0.01, 0.02) for _ in range(200)]
        sharpe_pool = [rng.gauss(0.3, 0.5) for _ in range(500)]

        few_trials = deflated_sharpe_ratio(returns, n_trials=2, trial_sharpe_ratios=sharpe_pool[:2])
        many_trials = deflated_sharpe_ratio(returns, n_trials=500, trial_sharpe_ratios=sharpe_pool)

        assert many_trials.deflated_sharpe_benchmark >= few_trials.deflated_sharpe_benchmark
        assert many_trials.probabilistic_sharpe_ratio <= few_trials.probabilistic_sharpe_ratio

    def test_likely_skill_property_thresholds_at_point_95(self):
        rng = random.Random(7)
        # Strong, consistent positive edge with low variance and a single
        # trial should clear the 0.95 bar comfortably.
        returns = [rng.gauss(0.02, 0.005) for _ in range(300)]
        result = deflated_sharpe_ratio(returns, n_trials=1)
        assert result.likely_skill is True
        assert result.probabilistic_sharpe_ratio > 0.95

    def test_no_edge_does_not_claim_skill(self):
        rng = random.Random(11)
        returns = [rng.gauss(0.0, 0.02) for _ in range(300)]
        result = deflated_sharpe_ratio(returns, n_trials=1)
        assert result.likely_skill is False


class TestProbabilityOfBacktestOverfitting:
    def test_rejects_odd_n_splits(self):
        matrix = [[0.1, 0.2] for _ in range(20)]
        with pytest.raises(ValueError):
            probability_of_backtest_overfitting(matrix, n_splits=5)

    def test_rejects_single_trial_column(self):
        matrix = [[0.1] for _ in range(20)]
        with pytest.raises(ValueError):
            probability_of_backtest_overfitting(matrix, n_splits=4)

    def test_rejects_too_few_periods_for_splits(self):
        matrix = [[0.1, 0.2] for _ in range(3)]
        with pytest.raises(ValueError):
            probability_of_backtest_overfitting(matrix, n_splits=10)

    def test_genuinely_skillful_trial_has_low_pbo(self):
        # One trial column is consistently better in every single period;
        # it should win in-sample AND out-of-sample on virtually every
        # split, so PBO should be low.
        rng = random.Random(3)
        n_periods = 40
        n_trials = 6
        matrix = []
        for _ in range(n_periods):
            row = [rng.gauss(0.0, 0.01) for _ in range(n_trials)]
            row[0] += 0.05  # trial 0 is reliably best, every period
            matrix.append(row)

        result = probability_of_backtest_overfitting(matrix, n_splits=8)
        assert result["pbo"] < 0.3
        assert result["n_splits"] == 8
        assert result["n_combinations"] == math.comb(8, 4)

    def test_pure_noise_trials_average_pbo_near_one_half(self):
        # All trial columns are pure noise with identical distribution --
        # whichever looks best in-sample is essentially a coin flip
        # out-of-sample, so PBO should hover around 0.5 (not reliably
        # skillful). A single seed's PBO estimate is noisy on its own
        # (small sample of splits), so average across several independent
        # matrices rather than asserting on one seed.
        pbos = []
        for seed in range(10):
            rng = random.Random(seed)
            n_periods = 60
            n_trials = 10
            matrix = [[rng.gauss(0.0, 0.02) for _ in range(n_trials)] for _ in range(n_periods)]
            pbos.append(probability_of_backtest_overfitting(matrix, n_splits=10)["pbo"])

        assert 0.35 < (sum(pbos) / len(pbos)) < 0.65


class TestBenjaminiHochberg:
    def test_empty_input_returns_empty_list(self):
        assert benjamini_hochberg([]) == []

    def test_no_correction_needed_when_all_clearly_significant(self):
        p_values = [0.001, 0.002, 0.0005, 0.003]
        assert benjamini_hochberg(p_values, alpha=0.05) == [True, True, True, True]

    def test_no_correction_passes_when_all_clearly_insignificant(self):
        p_values = [0.9, 0.8, 0.95, 0.7]
        assert benjamini_hochberg(p_values, alpha=0.05) == [False, False, False, False]

    def test_mixed_p_values_only_survive_the_low_end(self):
        # Classic textbook-style example: many hypotheses tested, only a
        # few should survive FDR correction even though some individually
        # clear the uncorrected 0.05 bar.
        p_values = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.9]
        significant = benjamini_hochberg(p_values, alpha=0.05)
        # The uncorrected count below 0.05 is 5; BH should keep at most that many.
        assert sum(significant) <= sum(1 for p in p_values if p < 0.05)
        assert significant[0] is True  # the smallest p-value always survives if any does
        assert significant[-1] is False  # the largest p-value never survives

    def test_order_independent_of_input_order(self):
        p_values = [0.9, 0.001, 0.5, 0.01]
        significant = benjamini_hochberg(p_values, alpha=0.05)
        # Result must be index-aligned to the ORIGINAL (unsorted) order.
        assert significant[1] is True  # 0.001
        assert significant[0] is False  # 0.9


class TestBlockBootstrapConfidenceInterval:
    def test_rejects_too_few_values(self):
        with pytest.raises(ValueError):
            block_bootstrap_confidence_interval([0.1, 0.2, 0.3])

    def test_rejects_non_positive_block_size(self):
        with pytest.raises(ValueError):
            block_bootstrap_confidence_interval([0.1] * 20, block_size=0)

    def test_rejects_block_size_too_large(self):
        values = [0.1] * 20
        with pytest.raises(ValueError):
            block_bootstrap_confidence_interval(values, block_size=len(values))

    def test_rejects_invalid_confidence_pct(self):
        values = [0.1] * 20
        with pytest.raises(ValueError):
            block_bootstrap_confidence_interval(values, confidence_pct=100.0)
        with pytest.raises(ValueError):
            block_bootstrap_confidence_interval(values, confidence_pct=0.0)

    def test_observed_statistic_matches_plain_computation_on_raw_values(self):
        rng = random.Random(1)
        values = [rng.gauss(0.01, 0.02) for _ in range(100)]
        result = block_bootstrap_confidence_interval(values, seed=5, n_resamples=200)
        assert result["observed"] == pytest.approx(sum(values) / len(values))

    def test_ci_contains_observed_for_a_clear_positive_edge(self):
        # Consistent, low-variance positive returns: the CI should sit
        # comfortably away from zero on both sides.
        rng = random.Random(2)
        values = [rng.gauss(0.02, 0.003) for _ in range(300)]
        result = block_bootstrap_confidence_interval(values, seed=9, n_resamples=500)
        assert result["ci_low"] <= result["observed"] <= result["ci_high"]
        assert result["ci_low"] > 0.0  # edge survives even the lower bound

    def test_ci_straddles_zero_for_pure_noise(self):
        rng = random.Random(3)
        values = [rng.gauss(0.0, 0.02) for _ in range(300)]
        result = block_bootstrap_confidence_interval(values, seed=4, n_resamples=500)
        assert result["ci_low"] < 0.0 < result["ci_high"]

    def test_wider_confidence_level_widens_the_interval(self):
        rng = random.Random(6)
        values = [rng.gauss(0.01, 0.02) for _ in range(200)]
        narrow = block_bootstrap_confidence_interval(values, seed=1, n_resamples=500, confidence_pct=80.0)
        wide = block_bootstrap_confidence_interval(values, seed=1, n_resamples=500, confidence_pct=99.0)
        assert (wide["ci_high"] - wide["ci_low"]) > (narrow["ci_high"] - narrow["ci_low"])

    def test_default_block_size_is_reported_and_sane(self):
        values = [0.01] * 64
        result = block_bootstrap_confidence_interval(values, seed=1, n_resamples=50)
        assert 2 <= result["block_size"] < len(values)

    def test_explicit_block_size_and_statistic_are_honored(self):
        rng = random.Random(8)
        values = [rng.gauss(0.01, 0.02) for _ in range(150)]
        result = block_bootstrap_confidence_interval(
            values,
            statistic=max,
            block_size=5,
            seed=2,
            n_resamples=300,
        )
        assert result["block_size"] == 5
        assert result["observed"] == max(values)

    def test_deterministic_with_fixed_seed(self):
        rng = random.Random(10)
        values = [rng.gauss(0.01, 0.02) for _ in range(120)]
        first = block_bootstrap_confidence_interval(values, seed=42, n_resamples=100)
        second = block_bootstrap_confidence_interval(values, seed=42, n_resamples=100)
        assert first == second
