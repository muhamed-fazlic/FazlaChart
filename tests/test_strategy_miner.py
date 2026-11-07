"""Tests for scanner.backtesting.strategy_miner -- Layer 1 of Strategy Lab.

Network-free: the three per-template run functions are monkeypatched to
fake, deterministic data so ``run_mining_cycle``'s combination-count,
ranking, and stability-scoring logic can be exercised without ever calling
a real backtest (those are covered by the existing
test_funding_extreme_backtest.py / test_cross_sectional_momentum_backtest.py
/ test_portfolio_backtest.py suites)."""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from scanner.backtesting import strategy_miner as miner
from scanner.backtesting.historical_universe import (
    RankedUniverseWindow,
    load_universe_cache,
    save_ranked_universe_cache,
)
from scanner.backtesting.strategy_miner import (
    ALL_TEMPLATE_VARIANTS,
    CROSS_SECTIONAL_MOMENTUM_VARIANTS,
    FUNDING_CARRY_XS_VARIANTS,
    FUNDING_EXTREME_VARIANTS,
    LOW_VOL_VARIANTS,
    MOMENTUM_30D_VARIANTS,
    ROLLING_AS_OF_TEMPLATES,
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
            len(TREND_ANOMALY_VARIANTS)
            + len(FUNDING_EXTREME_VARIANTS)
            + len(CROSS_SECTIONAL_MOMENTUM_VARIANTS)
            + len(MOMENTUM_30D_VARIANTS)
            + len(LOW_VOL_VARIANTS)
            + len(FUNDING_CARRY_XS_VARIANTS)
        )

    def test_korak10_rolling_templates_have_two_variants_each(self) -> None:
        assert len(MOMENTUM_30D_VARIANTS) == 2
        assert len(LOW_VOL_VARIANTS) == 2
        assert len(FUNDING_CARRY_XS_VARIANTS) == 2

    def test_rolling_as_of_templates_matches_korak10_template_names(self) -> None:
        assert ROLLING_AS_OF_TEMPLATES == frozenset({"momentum_30d", "low_vol", "funding_carry_xs"})


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

        def fake_engine(symbols, as_of_ts, cache, variant, bootstrap_confidence_pct=None, start_timestamp_ms=None):
            return {"n_independent": 10, "net_block_bootstrap": {"observed_pct": 0.05}, "gross_block_bootstrap": None}, [
                0.05
            ] * 10

        def fake_trend(symbols, as_of_ts, variant):
            return {"metrics": {"n_independent_episodes": 5, "net_profit_pct": 1.5}}, [1.5] * 5

        monkeypatch.setattr(miner, "_run_funding_extreme_variant", fake_funding)
        monkeypatch.setattr(miner, "_run_cross_sectional_variant", fake_momentum)
        monkeypatch.setattr(miner, "_run_cross_sectional_engine_variant", fake_engine)
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

        def fake_engine(symbols, as_of_ts, cache, variant, bootstrap_confidence_pct=None, start_timestamp_ms=None):
            return {"n_independent": 10, "net_block_bootstrap": {"observed_pct": 0.2}, "gross_block_bootstrap": None}, [
                0.2
            ] * 10

        def fake_trend(symbols, as_of_ts, variant):
            return {"metrics": {"n_independent_episodes": 5, "net_profit_pct": -3.0}}, [-3.0] * 5

        monkeypatch.setattr(miner, "_run_funding_extreme_variant", fake_funding)
        monkeypatch.setattr(miner, "_run_cross_sectional_variant", fake_momentum)
        monkeypatch.setattr(miner, "_run_cross_sectional_engine_variant", fake_engine)
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

        def fake_engine(symbols, as_of_ts, cache, variant, bootstrap_confidence_pct=None, start_timestamp_ms=None):
            return {"n_independent": 10, "net_block_bootstrap": {"observed_pct": 0.1}, "gross_block_bootstrap": None}, [
                0.1
            ] * 10

        def fake_trend(symbols, as_of_ts, variant):
            return {"metrics": {"n_independent_episodes": 5, "net_profit_pct": 1.5}}, [1.5] * 5

        monkeypatch.setattr(miner, "_run_funding_extreme_variant", failing_funding)
        monkeypatch.setattr(miner, "_run_cross_sectional_variant", fake_momentum)
        monkeypatch.setattr(miner, "_run_cross_sectional_engine_variant", fake_engine)
        monkeypatch.setattr(miner, "_run_trend_anomaly_variant", fake_trend)

        candidates = run_mining_cycle(1_000, symbols=("BTCUSDT",))
        # All funding_extreme_reversion variants failed and were skipped;
        # the other templates' variants still produced candidates.
        assert all(c.template != "funding_extreme_reversion" for c in candidates)
        assert len(candidates) == (
            len(TREND_ANOMALY_VARIANTS)
            + len(CROSS_SECTIONAL_MOMENTUM_VARIANTS)
            + len(MOMENTUM_30D_VARIANTS)
            + len(LOW_VOL_VARIANTS)
            + len(FUNDING_CARRY_XS_VARIANTS)
        )

    def test_already_mined_keys_are_skipped_entirely(self, monkeypatch) -> None:
        """Regression coverage for a real production bug: once real time
        passes the frozen SHARED_HOLDOUT_END_MS boundary, as_of_ts is
        clamped to the same value forever, so re-running an
        already-mined combination against it is guaranteed to reproduce
        the identical result -- a pure waste of CPU that also pollutes
        storage with duplicate rows every cycle. already_mined_keys lets
        callers skip those combinations without even invoking the
        (potentially expensive) backtest for them."""
        calls: list[str] = []

        def fake_funding(symbols, as_of_ts, cache, variant):
            calls.append(f"funding:{variant.variant_label}")
            return {"n_independent": 10, "net_block_bootstrap": {"observed_pct": 0.1}, "gross_block_bootstrap": None}, [
                0.1
            ] * 10

        def fake_momentum(symbols, as_of_ts, cache, variant):
            calls.append(f"momentum:{variant.variant_label}")
            return {"n_independent": 10, "net_block_bootstrap": {"observed_pct": 0.1}, "gross_block_bootstrap": None}, [
                0.1
            ] * 10

        def fake_engine(symbols, as_of_ts, cache, variant, bootstrap_confidence_pct=None, start_timestamp_ms=None):
            calls.append(f"engine:{variant.variant_label}")
            return {"n_independent": 10, "net_block_bootstrap": {"observed_pct": 0.1}, "gross_block_bootstrap": None}, [
                0.1
            ] * 10

        def fake_trend(symbols, as_of_ts, variant):
            calls.append(f"trend:{variant.variant_label}")
            return {"metrics": {"n_independent_episodes": 5, "net_profit_pct": 1.5}}, [1.5] * 5

        monkeypatch.setattr(miner, "_run_funding_extreme_variant", fake_funding)
        monkeypatch.setattr(miner, "_run_cross_sectional_variant", fake_momentum)
        monkeypatch.setattr(miner, "_run_cross_sectional_engine_variant", fake_engine)
        monkeypatch.setattr(miner, "_run_trend_anomaly_variant", fake_trend)

        already_mined = frozenset(
            (variant.template, "unsliced", variant.variant_label) for variant in TREND_ANOMALY_VARIANTS
        )
        candidates = run_mining_cycle(1_000, symbols=("BTCUSDT",), already_mined_keys=already_mined)

        # The already-mined trend_anomaly variants were never even
        # invoked (not just filtered out after the fact).
        assert all(not call.startswith("trend:") for call in calls)
        assert all(c.template != "trend_anomaly" for c in candidates)
        # Everything else still ran normally.
        assert len(candidates) == (
            len(FUNDING_EXTREME_VARIANTS)
            + len(CROSS_SECTIONAL_MOMENTUM_VARIANTS)
            + len(MOMENTUM_30D_VARIANTS)
            + len(LOW_VOL_VARIANTS)
            + len(FUNDING_CARRY_XS_VARIANTS)
        )

    def test_all_combinations_already_mined_produces_no_candidates(self, monkeypatch) -> None:
        def fail_if_called(*args, **kwargs):
            raise AssertionError("should not be called -- everything was already mined")

        monkeypatch.setattr(miner, "_run_funding_extreme_variant", fail_if_called)
        monkeypatch.setattr(miner, "_run_cross_sectional_variant", fail_if_called)
        monkeypatch.setattr(miner, "_run_cross_sectional_engine_variant", fail_if_called)
        monkeypatch.setattr(miner, "_run_trend_anomaly_variant", fail_if_called)

        already_mined = frozenset(
            (variant.template, "unsliced", variant.variant_label) for variant in ALL_TEMPLATE_VARIANTS
        )
        candidates = run_mining_cycle(1_000, symbols=("BTCUSDT",), already_mined_keys=already_mined)
        assert candidates == []


class TestRankedUniverseCacheTranslation:
    """Regression coverage for a real production bug: funding_extreme/
    cross_sectional variants used to receive the RANKED universe cache
    path (``{"ranked_symbols": [...]}`` schema) completely unchanged as
    their ``point_in_time_universe_cache`` argument, even though their
    frozen backtest modules only understand the older
    ``{"symbols": [...]}`` schema -- causing a ``KeyError: 'symbols'`` on
    every single funding_extreme_reversion/cross_sectional_momentum
    combination, every mining cycle, while only trend_anomaly (which never
    takes that argument at all) ever succeeded."""

    @staticmethod
    def _build_ranked_cache(tmp_path: Path) -> Path:
        base = datetime(2024, 1, 1, tzinfo=UTC)
        # 170 symbols so the default tercile rank ranges (1-57 / 58-113 /
        # 114-170) each get a non-empty slice, same shape as the real
        # production cache.
        symbols = tuple(f"SYM{i}USDT" for i in range(170))
        windows = [
            RankedUniverseWindow(start=base, end=base + timedelta(days=30), ranked_symbols=symbols),
            RankedUniverseWindow(
                start=base + timedelta(days=30), end=base + timedelta(days=60), ranked_symbols=symbols
            ),
        ]
        cache_path = tmp_path / "ranked_universe.json"
        save_ranked_universe_cache(windows, cache_path)
        return cache_path

    def test_funding_extreme_and_cross_sectional_receive_old_schema_cache(self, tmp_path, monkeypatch) -> None:
        ranked_cache_path = self._build_ranked_cache(tmp_path)
        seen_caches: list[str | None] = []
        loaded_window_sets: list[tuple] = []

        def _check_and_record(cache: str | None) -> None:
            seen_caches.append(cache)
            assert cache is not None
            assert cache != str(ranked_cache_path)
            # Must be loadable through the OLD schema's loader, while the
            # file still exists (it's deleted once the cycle ends) -- the
            # real bug: the ranked cache's "ranked_symbols" key raises
            # KeyError('symbols') here instead.
            windows = load_universe_cache(cache)
            assert windows
            assert all(w.symbols for w in windows)
            loaded_window_sets.append(windows)

        def fake_funding(symbols, as_of_ts, cache, variant):
            _check_and_record(cache)
            return {"n_independent": 10, "net_block_bootstrap": {"observed_pct": 0.1}, "gross_block_bootstrap": None}, [
                0.1
            ] * 10

        def fake_momentum(symbols, as_of_ts, cache, variant):
            _check_and_record(cache)
            return {"n_independent": 10, "net_block_bootstrap": {"observed_pct": 0.1}, "gross_block_bootstrap": None}, [
                0.1
            ] * 10

        def fake_engine(symbols, as_of_ts, cache, variant, bootstrap_confidence_pct=None, start_timestamp_ms=None):
            _check_and_record(cache)
            return {"n_independent": 10, "net_block_bootstrap": {"observed_pct": 0.1}, "gross_block_bootstrap": None}, [
                0.1
            ] * 10

        def fake_trend(symbols, as_of_ts, variant):
            return {"metrics": {"n_independent_episodes": 5, "net_profit_pct": 1.5}}, [1.5] * 5

        monkeypatch.setattr(miner, "_run_funding_extreme_variant", fake_funding)
        monkeypatch.setattr(miner, "_run_cross_sectional_variant", fake_momentum)
        monkeypatch.setattr(miner, "_run_cross_sectional_engine_variant", fake_engine)
        monkeypatch.setattr(miner, "_run_trend_anomaly_variant", fake_trend)

        candidates = run_mining_cycle(1_000, ranked_universe_cache_path=str(ranked_cache_path))

        # All 3 terciles x (funding_extreme + cross_sectional + the three
        # Korak 10 rolling templates) variants ran -- none silently
        # skipped due to a cache-schema mismatch.
        expected_non_trend = (
            len(FUNDING_EXTREME_VARIANTS)
            + len(CROSS_SECTIONAL_MOMENTUM_VARIANTS)
            + len(MOMENTUM_30D_VARIANTS)
            + len(LOW_VOL_VARIANTS)
            + len(FUNDING_CARRY_XS_VARIANTS)
        ) * 3
        assert len(seen_caches) == expected_non_trend
        assert len(loaded_window_sets) == expected_non_trend
        assert len(candidates) == len(ALL_TEMPLATE_VARIANTS) * 3

    def test_temp_tercile_cache_files_are_cleaned_up(self, tmp_path, monkeypatch) -> None:
        ranked_cache_path = self._build_ranked_cache(tmp_path)
        seen_caches: list[str] = []

        def fake_funding(symbols, as_of_ts, cache, variant):
            seen_caches.append(cache)
            return {"n_independent": 10, "net_block_bootstrap": {"observed_pct": 0.1}, "gross_block_bootstrap": None}, [
                0.1
            ] * 10

        def fake_engine(symbols, as_of_ts, cache, variant, bootstrap_confidence_pct=None, start_timestamp_ms=None):
            seen_caches.append(cache)
            return {"n_independent": 10, "net_block_bootstrap": {"observed_pct": 0.1}, "gross_block_bootstrap": None}, [
                0.1
            ] * 10

        monkeypatch.setattr(miner, "_run_funding_extreme_variant", fake_funding)
        monkeypatch.setattr(miner, "_run_cross_sectional_variant", fake_funding)
        monkeypatch.setattr(miner, "_run_cross_sectional_engine_variant", fake_engine)
        monkeypatch.setattr(
            miner, "_run_trend_anomaly_variant", lambda symbols, as_of_ts, variant: ({"metrics": {}}, [])
        )

        run_mining_cycle(1_000, ranked_universe_cache_path=str(ranked_cache_path))

        assert seen_caches
        for cache_path in seen_caches:
            assert not os.path.exists(cache_path)

    def test_tercile_universe_cache_path_context_manager_cleans_up(self, tmp_path) -> None:
        ranked_cache_path = self._build_ranked_cache(tmp_path)
        with miner.tercile_universe_cache_path(str(ranked_cache_path), "top") as cache_path:
            assert os.path.exists(cache_path)
            windows = load_universe_cache(cache_path)
            assert windows
            assert len(windows[-1].symbols) == 57
        assert not os.path.exists(cache_path)


class TestSelectVariantsWithinUniqueConfigCap:
    """Korak 10 item 4: a hard, non-self-widening 300-config cap."""

    def test_all_variants_admitted_when_budget_is_generous(self) -> None:
        admitted = miner.select_variants_within_unique_config_cap(
            miner.ALL_TEMPLATE_VARIANTS, ("top", "mid", "bottom"), frozenset(), max_unique_configurations=300
        )
        assert admitted == miner.ALL_TEMPLATE_VARIANTS

    def test_already_seen_variants_always_pass_regardless_of_budget(self) -> None:
        variant = miner.ALL_TEMPLATE_VARIANTS[0]
        ever_mined = frozenset({(variant.template, "top", variant.variant_label)})
        admitted = miner.select_variants_within_unique_config_cap(
            (variant,), ("top",), ever_mined, max_unique_configurations=0
        )
        assert admitted == (variant,)

    def test_brand_new_variant_rejected_once_budget_exhausted(self) -> None:
        variant = miner.ALL_TEMPLATE_VARIANTS[0]
        admitted = miner.select_variants_within_unique_config_cap(
            (variant,), ("top",), frozenset(), max_unique_configurations=0
        )
        assert admitted == ()

    def test_partial_budget_is_all_or_nothing_per_variant(self) -> None:
        # Needs 3 new slots (one per tercile) but only 2 remain -- the
        # whole variant is rejected this cycle, not partially admitted.
        variant = miner.ALL_TEMPLATE_VARIANTS[0]
        admitted = miner.select_variants_within_unique_config_cap(
            (variant,), ("top", "mid", "bottom"), frozenset(), max_unique_configurations=2
        )
        assert admitted == ()

    def test_does_not_mutate_input_tuple(self) -> None:
        original = miner.ALL_TEMPLATE_VARIANTS
        miner.select_variants_within_unique_config_cap(original, ("top",), frozenset(), max_unique_configurations=1)
        assert miner.ALL_TEMPLATE_VARIANTS == original

    def test_budget_consumed_across_multiple_variants_in_call_order(self) -> None:
        v1, v2, v3 = miner.ALL_TEMPLATE_VARIANTS[:3]
        # Exactly enough budget for the first two variants' single tercile
        # each (1 slot apiece), nothing left for the third.
        admitted = miner.select_variants_within_unique_config_cap(
            (v1, v2, v3), ("top",), frozenset(), max_unique_configurations=2
        )
        assert admitted == (v1, v2)


class TestEvaluateVariantDispatchForRollingTemplates:
    """Confirms ``evaluate_variant`` routes the three Korak 10 templates
    through ``_run_cross_sectional_engine_variant`` and threads an
    optional ``bootstrap_confidence_pct`` all the way through."""

    @pytest.mark.parametrize("template", ["momentum_30d", "low_vol", "funding_carry_xs"])
    def test_dispatches_to_cross_sectional_engine_runner(self, template, monkeypatch) -> None:
        seen_kwargs: dict = {}

        def fake_engine(symbols, as_of_ts, cache, variant, bootstrap_confidence_pct=None, start_timestamp_ms=None):
            seen_kwargs["bootstrap_confidence_pct"] = bootstrap_confidence_pct
            return {
                "n_independent": 7,
                "net_block_bootstrap": {"observed_pct": 1.0},
                "gross_block_bootstrap": {"observed_pct": 2.0},
            }, [1.0] * 7

        monkeypatch.setattr(miner, "_run_cross_sectional_engine_variant", fake_engine)
        variant = next(v for v in miner.ALL_TEMPLATE_VARIANTS if v.template == template)

        evaluation = miner.evaluate_variant(
            variant, ("BTCUSDT",), 1_000, None, bootstrap_confidence_pct=90.0
        )

        assert seen_kwargs["bootstrap_confidence_pct"] == 90.0
        assert evaluation.n_independent == 7
        assert evaluation.net_observed_pct == 1.0
        assert evaluation.gross_observed_pct == 2.0

    def test_defaults_to_none_confidence_when_unspecified(self, monkeypatch) -> None:
        seen_kwargs: dict = {}

        def fake_engine(symbols, as_of_ts, cache, variant, bootstrap_confidence_pct=None, start_timestamp_ms=None):
            seen_kwargs["bootstrap_confidence_pct"] = bootstrap_confidence_pct
            return {"n_independent": 1, "net_block_bootstrap": None, "gross_block_bootstrap": None}, []

        monkeypatch.setattr(miner, "_run_cross_sectional_engine_variant", fake_engine)
        variant = next(v for v in miner.ALL_TEMPLATE_VARIANTS if v.template == "low_vol")

        miner.evaluate_variant(variant, ("BTCUSDT",), 1_000, None)

        assert seen_kwargs["bootstrap_confidence_pct"] is None
