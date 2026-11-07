"""Tests for scanner.backtesting.layer2_entry -- Korak 10 item 5/6's Layer
2 entry-threshold gate and Holm/Bonferroni confidence correction."""

from __future__ import annotations

import pytest

from scanner.backtesting.layer2_entry import (
    BASE_ALPHA,
    ENTRY_WINDOW,
    MIN_N,
    check_entry_threshold,
    holm_corrected_confidence_pct,
)


def _result(net: float = 1.0, gross: float = 2.0, n: int = MIN_N) -> dict:
    return {"net_observed_pct": net, "gross_observed_pct": gross, "n_independent": n}


class TestCheckEntryThreshold:
    def test_qualifies_when_all_conditions_met(self) -> None:
        recent = [_result() for _ in range(ENTRY_WINDOW)]
        result = check_entry_threshold(recent, latest_btc_beta=0.1)
        assert result.qualifies is True
        assert result.windows_positive == ENTRY_WINDOW
        assert result.windows_checked == ENTRY_WINDOW

    def test_fails_when_fewer_windows_than_required(self) -> None:
        recent = [_result() for _ in range(ENTRY_WINDOW - 1)]
        result = check_entry_threshold(recent, latest_btc_beta=0.1)
        assert result.qualifies is False
        assert "windows observed so far" in result.reason

    def test_fails_when_one_of_the_recent_windows_is_negative(self) -> None:
        recent = [_result(net=1.0), _result(net=-0.5), _result(net=1.0)]
        result = check_entry_threshold(recent, latest_btc_beta=0.1)
        assert result.qualifies is False
        assert result.windows_positive == 2
        assert "net-positive" in result.reason

    def test_fails_when_latest_n_below_min_n(self) -> None:
        recent = [_result(n=MIN_N - 1), _result(), _result()]
        result = check_entry_threshold(recent, latest_btc_beta=0.1)
        assert result.qualifies is False
        assert "MIN_N" in result.reason

    def test_fails_when_latest_gross_not_positive(self) -> None:
        recent = [_result(gross=0.0), _result(), _result()]
        result = check_entry_threshold(recent, latest_btc_beta=0.1)
        assert result.qualifies is False
        assert "gross_observed_pct" in result.reason

    def test_fails_when_gross_is_none(self) -> None:
        recent = [{"net_observed_pct": 1.0, "gross_observed_pct": None, "n_independent": MIN_N} for _ in range(3)]
        result = check_entry_threshold(recent, latest_btc_beta=0.1)
        assert result.qualifies is False

    def test_fails_when_beta_missing(self) -> None:
        """A config whose report never carries a beta (e.g. trend_anomaly)
        fails CLOSED -- it never silently passes by default."""
        recent = [_result() for _ in range(ENTRY_WINDOW)]
        result = check_entry_threshold(recent, latest_btc_beta=None)
        assert result.qualifies is False
        assert "beta" in result.reason

    def test_fails_when_beta_at_or_above_threshold(self) -> None:
        recent = [_result() for _ in range(ENTRY_WINDOW)]
        result = check_entry_threshold(recent, latest_btc_beta=0.3)
        assert result.qualifies is False

    def test_negative_beta_is_also_checked_by_absolute_value(self) -> None:
        recent = [_result() for _ in range(ENTRY_WINDOW)]
        result = check_entry_threshold(recent, latest_btc_beta=-0.35)
        assert result.qualifies is False

    def test_custom_thresholds_are_honored(self) -> None:
        recent = [_result(n=5) for _ in range(2)]
        result = check_entry_threshold(recent, latest_btc_beta=0.1, min_n=5, entry_window=2)
        assert result.qualifies is True

    def test_more_history_than_required_only_checks_the_most_recent_window(self) -> None:
        # 5 results, only the most recent 3 matter; an old negative one
        # further back should not block qualification.
        recent = [_result(), _result(), _result(), _result(net=-5.0), _result()]
        result = check_entry_threshold(recent, latest_btc_beta=0.1)
        assert result.qualifies is True
        assert result.windows_checked == ENTRY_WINDOW


class TestHolmCorrectedConfidencePct:
    def test_single_config_uses_base_alpha_unadjusted(self) -> None:
        pct = holm_corrected_confidence_pct(1)
        assert pct == pytest.approx(100.0 * (1.0 - BASE_ALPHA))

    def test_more_configs_tightens_the_corrected_confidence(self) -> None:
        pct_few = holm_corrected_confidence_pct(10)
        pct_many = holm_corrected_confidence_pct(300)
        assert pct_many > pct_few

    def test_zero_or_negative_count_treated_as_one(self) -> None:
        assert holm_corrected_confidence_pct(0) == holm_corrected_confidence_pct(1)
        assert holm_corrected_confidence_pct(-5) == holm_corrected_confidence_pct(1)

    def test_never_exceeds_a_sane_upper_bound(self) -> None:
        # Even an absurdly large config count must not push alpha to
        # exactly zero / confidence to exactly 100%.
        pct = holm_corrected_confidence_pct(10_000_000)
        assert pct < 100.0

    def test_custom_base_alpha_is_honored(self) -> None:
        pct = holm_corrected_confidence_pct(1, base_alpha=0.1)
        assert pct == pytest.approx(90.0)
