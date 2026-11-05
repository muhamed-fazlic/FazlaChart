"""Unit tests for scanner.backtesting.strategy_variants (Phase 8 "strategy
variants" framework)."""

from __future__ import annotations

import pytest

from scanner.backtesting.strategy_variants import (
    DEFAULT_VARIANT_ID,
    STRATEGY_VARIANTS,
    family_suffix,
    get_variant,
    report_path_for_variant,
)


class TestGetVariant:
    def test_none_defaults_to_original(self):
        assert get_variant(None).variant_id == DEFAULT_VARIANT_ID

    def test_empty_string_defaults_to_original(self):
        assert get_variant("").variant_id == DEFAULT_VARIANT_ID

    def test_original_has_no_backtest_symbol_kwargs(self):
        assert get_variant("original").backtest_symbol_kwargs == {}

    def test_inverted_sets_invert_signals(self):
        assert get_variant("inverted").backtest_symbol_kwargs == {"invert_signals": True}

    def test_unknown_variant_raises_key_error(self):
        with pytest.raises(KeyError, match="unknown_variant_xyz"):
            get_variant("unknown_variant_xyz")

    def test_every_registered_variant_has_a_label_and_description(self):
        for variant in STRATEGY_VARIANTS.values():
            assert variant.label
            assert variant.description


class TestFamilySuffix:
    def test_original_has_no_suffix(self):
        assert family_suffix("original") == ""

    def test_non_default_variant_gets_an_explicit_suffix(self):
        assert family_suffix("inverted") == "__inverted"

    def test_suffix_round_trips_into_a_distinct_family_name(self):
        base = "portfolio_backtest_20symbol_warmup1500_full"
        assert f"{base}{family_suffix('inverted')}" == f"{base}__inverted"
        assert f"{base}{family_suffix('original')}" == base


class TestReportPathForVariant:
    def test_original_keeps_the_base_path_unchanged(self):
        assert report_path_for_variant("scanner_data/portfolio_backtest_report.json", "original") == (
            "scanner_data/portfolio_backtest_report.json"
        )

    def test_non_default_variant_gets_a_json_suffix(self):
        assert report_path_for_variant("scanner_data/portfolio_backtest_report.json", "inverted") == (
            "scanner_data/portfolio_backtest_report__inverted.json"
        )

    def test_handles_a_path_without_a_json_extension(self):
        assert report_path_for_variant("scanner_data/report", "inverted") == "scanner_data/report__inverted"
