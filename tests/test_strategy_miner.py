"""Tests for scanner.backtesting.strategy_miner -- Layer 1 of Strategy Lab.

Network-free: the three per-template run functions are monkeypatched to
fake, deterministic data so ``run_mining_cycle``'s combination-count,
ranking, and stability-scoring logic can be exercised without ever calling
a real backtest (those are covered by the existing
test_funding_extreme_backtest.py / test_cross_sectional_momentum_backtest.py
/ test_portfolio_backtest.py suites)."""

from __future__ import annotations

import logging

import pytest

from scanner.backtesting import strategy_miner as miner
from scanner.backtesting.strategy_miner import (
    ALL_TEMPLATE_VARIANTS,
    CROSS_SECTIONAL_MOMENTUM_VARIANTS,
    FUNDING_EXTREME_VARIANTS,
    TREND_ANOMALY_VARIANTS,
    _composite_score,
    _patched_module_constants,
    compute_stability_score,
    run_mining_cycle,
)


class _FakeModule:
    """Tiny stand-in for a frozen backtest module, used to prove
    ``_patched_module_constants`` restores originals even when the body
    raises."""

    CONST_A = 1
    CONST_B = "x"


class TestPatchedModuleConstants:
    def test_overrides_then_restores(self) -> None:
        mod = _FakeModule()
        with _patched_module_constants(mod, CONST_A=99):
            assert mod.CONST_A == 99
        assert mod.CONST_A == 1

    def test_restores_even_on_exception(self) -> None:
        mod = _FakeModule()
        with pytest.raises(RuntimeError), _patched_module_constants(mod, CONST_A=99):
            raise RuntimeError("boom")
        assert mod.CONST_A == 1

    def test_restores_multiple_overrides(self) -> None:
        mod = _FakeModule()
        with _patched_module_constants(mod, CONST_A=2, CONST_B="y"):
            assert mod.CONST_A == 2
            assert mod.CONST_B == "y"
        assert mod.CONST_A == 1
        assert mod.CONST_B == "x"


class TestPreDeclaredGrid:
    """The grid is bounded and explicit -- no more than 5 variants per
    template, and the total count must be exactly computable up front."""

    def test_trend_anomaly_has_two_variants(self) -> None:
        assert len(TREND_ANOMALY_VARIANTS) == 2

    def test_funding_extreme_has_at_most_five_variants(self) -> None:
        assert 1 <= len(FUNDING_EXTREME_VARIANTS) <= 5

    def test_cross_sectional_momentum_has_at_most_five_variants(self) -> None:
        assert 1 <= len(CROSS_SECTIONAL_MOMENTUM_VARIANTS) <= 5

    def test_all_template_variants_is_the_union(self) -> None:
        assert len(ALL_TEMPLATE_VARIANTS) == (
            len(TREND_ANOMALY_VARIANTS) + len(FUNDING_EXTREME_VARIANTS) + len(CROSS_SECTIONAL_MOMENTUM_VARIANTS)
        )


class TestComputeStabilityScore:
    def test_none_for_too_few_events(self) -> None:
        assert compute_stability_score([1.0, 2.0, 3.0]) is None

    def test_full_agreement_scores_one(self) -> None:
        returns = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        assert compute_stability_score(returns) == 1.0

    def test_sign_disagreement_scores_less_than_one(self) -> None:
        returns = [5.0, 5.0, -5.0, -5.0, -5.0, -5.0]
        score = compute_stability_score(returns)
        assert score is not None
        assert score < 1.0


class TestCompositeScore:
    def test_none_net_scores_negative_infinity(self) -> None:
        assert _composite_score(None, 1.0) == float("-inf")

    def test_unstable_candidate_scores_lower_than_stable(self) -> None:
        stable = _composite_score(1.0, 1.0)
        unstable = _composite_score(1.0, 0.0)
        assert stable > unstable

    def test_missing_stability_uses_neutral_weight(self) -> None:
        assert _composite_score(2.0, None) == pytest.approx(1.0)


class TestRunMiningCycle:
    def test_combination_count_matches_template_variants_times_terciles(self, monkeypatch, caplog) -> None:
        def fake_funding(symbols, as_of_ts, cache, variant):
            return {"n_independent": 10, "net_block_bootstrap": {"observed_pct": 0.1}, "gross_block_bootstrap": None}, [
                0.1
            ] * 10

        def fake_momentum(symbols, as_of_ts, cache, variant):
            return {"n_independent": 10, "net_block_bootstrap": {"observed_pct": -0.1}, "gross_block_bootstrap": None}, [
                -0.1
            ] * 10

        def fake_trend(symbols, as_of_ts, variant):
            return {"metrics": {"n_independent_episodes": 5, "net_profit_pct": 1.5}}, [1.5] * 5

        monkeypatch.setattr(miner, "_run_funding_extreme_variant", fake_funding)
        monkeypatch.setattr(miner, "_run_cross_sectional_variant", fake_momentum)
        monkeypatch.setattr(miner, "_run_trend_anomaly_variant", fake_trend)

        with caplog.at_level(logging.INFO):
            candidates = run_mining_cycle(1_000, symbols=("BTCUSDT", "ETHUSDT"))

        expected_total = len(ALL_TEMPLATE_VARIANTS) * 1  # single "unsliced" tercile fallback
        assert len(candidates) == expected_total
        assert any(f"{expected_total} total combinations" in msg for msg in caplog.messages)

    def test_candidates_sorted_by_composite_score_descending(self, monkeypatch) -> None:
        def fake_funding(symbols, as_of_ts, cache, variant):
            return {"n_independent": 10, "net_block_bootstrap": {"observed_pct": 0.1}, "gross_block_bootstrap": None}, [
                0.1
            ] * 10

        def fake_momentum(symbols, as_of_ts, cache, variant):
            return {"n_independent": 10, "net_block_bootstrap": {"observed_pct": 5.0}, "gross_block_bootstrap": None}, [
                5.0
            ] * 10

        def fake_trend(symbols, as_of_ts, variant):
            return {"metrics": {"n_independent_episodes": 5, "net_profit_pct": -3.0}}, [-3.0] * 5

        monkeypatch.setattr(miner, "_run_funding_extreme_variant", fake_funding)
        monkeypatch.setattr(miner, "_run_cross_sectional_variant", fake_momentum)
        monkeypatch.setattr(miner, "_run_trend_anomaly_variant", fake_trend)

        candidates = run_mining_cycle(1_000, symbols=("BTCUSDT",))
        scores = [c.composite_score for c in candidates]
        assert scores == sorted(scores, reverse=True)
        # The highest score should come from a cross_sectional_momentum variant (net=5.0).
        assert candidates[0].template == "cross_sectional_momentum"

    def test_requires_symbols_when_no_ranked_cache_given(self) -> None:
        with pytest.raises(ValueError):
            run_mining_cycle(1_000)

    def test_a_failing_combination_does_not_abort_the_cycle(self, monkeypatch) -> None:
        def failing_funding(symbols, as_of_ts, cache, variant):
            raise RuntimeError("simulated backtest failure")

        def fake_momentum(symbols, as_of_ts, cache, variant):
            return {"n_independent": 10, "net_block_bootstrap": {"observed_pct": 0.1}, "gross_block_bootstrap": None}, [
                0.1
            ] * 10

        def fake_trend(symbols, as_of_ts, variant):
            return {"metrics": {"n_independent_episodes": 5, "net_profit_pct": 1.5}}, [1.5] * 5

        monkeypatch.setattr(miner, "_run_funding_extreme_variant", failing_funding)
        monkeypatch.setattr(miner, "_run_cross_sectional_variant", fake_momentum)
        monkeypatch.setattr(miner, "_run_trend_anomaly_variant", fake_trend)

        candidates = run_mining_cycle(1_000, symbols=("BTCUSDT",))
        # All funding_extreme_reversion variants failed and were skipped;
        # the other two templates' variants still produced candidates.
        assert all(c.template != "funding_extreme_reversion" for c in candidates)
        assert len(candidates) == len(TREND_ANOMALY_VARIANTS) + len(CROSS_SECTIONAL_MOMENTUM_VARIANTS)
