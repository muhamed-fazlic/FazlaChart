"""Unit tests for the calibration cache — mocks the (network-bound) backtest
itself so these never touch Binance."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agents_crypto.calibration_lookup import (
    find_oos_for_labels,
    find_stats_for_labels,
    get_calibration_stats,
    get_cached_calibration_stats,
    get_oos_validation,
)
from scanner.calibration import CalibrationStats, OutOfSampleResult

pytestmark = pytest.mark.unit

_FAKE_STATS = [
    CalibrationStats(
        label="trend:bearish",
        direction="short",
        horizon_candles=12,
        n=387,
        win_rate_pct=37.0,
        avg_net_return_pct=-0.173,
        avg_win_return_pct=0.637,
        avg_loss_return_pct=-0.647,
    ),
    CalibrationStats(
        label="anomaly:failed_breakout",
        direction="short",
        horizon_candles=12,
        n=41,
        win_rate_pct=34.1,
        avg_net_return_pct=-1.283,
        avg_win_return_pct=0.953,
        avg_loss_return_pct=-2.442,
    ),
]


def test_computes_and_caches_on_first_call(tmp_path):
    cache_path = str(tmp_path / "cal.json")
    with (
        patch("agents_crypto.calibration_lookup.backtest_symbol", return_value=["case"]) as mock_backtest,
        patch("agents_crypto.calibration_lookup.summarize", return_value=_FAKE_STATS) as mock_summarize,
    ):
        stats = get_calibration_stats("BTCUSDT", "1h", cache_path=cache_path)
        assert mock_backtest.called
        assert mock_summarize.called
        assert len(stats) == 2

        # Second call within TTL should hit the cache, not recompute.
        mock_backtest.reset_mock()
        stats_again = get_calibration_stats("BTCUSDT", "1h", cache_path=cache_path)
        assert not mock_backtest.called
        assert stats_again == stats


def test_force_refresh_recomputes(tmp_path):
    cache_path = str(tmp_path / "cal.json")
    with (
        patch("agents_crypto.calibration_lookup.backtest_symbol", return_value=["case"]),
        patch("agents_crypto.calibration_lookup.summarize", return_value=_FAKE_STATS),
    ):
        get_calibration_stats("BTCUSDT", "1h", cache_path=cache_path)

    with (
        patch("agents_crypto.calibration_lookup.backtest_symbol", return_value=["case"]) as mock_backtest,
        patch("agents_crypto.calibration_lookup.summarize", return_value=_FAKE_STATS),
    ):
        get_calibration_stats("BTCUSDT", "1h", cache_path=cache_path, force_refresh=True)
        assert mock_backtest.called


def test_stale_cache_recomputes(tmp_path):
    cache_path = str(tmp_path / "cal.json")
    with (
        patch("agents_crypto.calibration_lookup.backtest_symbol", return_value=["case"]),
        patch("agents_crypto.calibration_lookup.summarize", return_value=_FAKE_STATS),
    ):
        get_calibration_stats("BTCUSDT", "1h", cache_path=cache_path, ttl_seconds=0)

    with (
        patch("agents_crypto.calibration_lookup.backtest_symbol", return_value=["case"]) as mock_backtest,
        patch("agents_crypto.calibration_lookup.summarize", return_value=_FAKE_STATS),
    ):
        get_calibration_stats("BTCUSDT", "1h", cache_path=cache_path, ttl_seconds=0)
        assert mock_backtest.called


def test_find_stats_for_labels_matches_label_and_horizon():
    matches = find_stats_for_labels(_FAKE_STATS, labels=["trend:bearish", "trend:neutral"], horizon_candles=12)
    assert len(matches) == 1
    assert matches[0].label == "trend:bearish"


def test_find_stats_for_labels_filters_by_horizon():
    matches = find_stats_for_labels(_FAKE_STATS, labels=["trend:bearish"], horizon_candles=4)
    assert matches == []


def test_get_cached_calibration_stats_never_backtests_on_miss(tmp_path):
    """A cold/empty cache must return [] immediately — never trigger a
    backtest — since this is the cache-only path used by endpoints that
    rank many symbols at once and can't afford to block on one."""
    cache_path = str(tmp_path / "cal.json")
    with patch("agents_crypto.calibration_lookup.backtest_symbol") as mock_backtest:
        stats = get_cached_calibration_stats("BTCUSDT", "1h", cache_path=cache_path)
        assert stats == []
        assert not mock_backtest.called


def test_get_cached_calibration_stats_reads_warm_cache(tmp_path):
    cache_path = str(tmp_path / "cal.json")
    with (
        patch("agents_crypto.calibration_lookup.backtest_symbol", return_value=["case"]),
        patch("agents_crypto.calibration_lookup.summarize", return_value=_FAKE_STATS),
    ):
        get_calibration_stats("BTCUSDT", "1h", cache_path=cache_path)  # warms the cache

    stats = get_cached_calibration_stats("BTCUSDT", "1h", cache_path=cache_path)
    assert len(stats) == 2
    assert stats[0].label == "trend:bearish"


def test_get_cached_calibration_stats_ignores_expired_entry(tmp_path):
    cache_path = str(tmp_path / "cal.json")
    with (
        patch("agents_crypto.calibration_lookup.backtest_symbol", return_value=["case"]),
        patch("agents_crypto.calibration_lookup.summarize", return_value=_FAKE_STATS),
    ):
        get_calibration_stats("BTCUSDT", "1h", cache_path=cache_path, ttl_seconds=0)

    # ttl_seconds=0 means the entry is already stale by the time this reads it.
    stats = get_cached_calibration_stats("BTCUSDT", "1h", cache_path=cache_path, ttl_seconds=0)
    assert stats == []


_FAKE_OOS = [
    OutOfSampleResult(
        label="trend:bearish",
        direction="short",
        horizon_candles=12,
        train_n=70,
        train_win_rate_pct=60.0,
        test_n=30,
        test_win_rate_pct=55.0,
        test_win_rate_ci_low_pct=42.0,
        holds_up=True,
        warning=None,
    ),
]


def test_get_oos_validation_computes_and_caches_on_first_call(tmp_path):
    cache_path = str(tmp_path / "cal.json")
    with (
        patch("agents_crypto.calibration_lookup.backtest_symbol", return_value=["case"]),
        patch("agents_crypto.calibration_lookup.validate_out_of_sample", return_value=_FAKE_OOS) as mock_validate,
    ):
        results = get_oos_validation("BTCUSDT", "1h", cache_path=cache_path)
        assert mock_validate.called
        assert len(results) == 1
        assert results[0].holds_up is True

        # Second call within TTL should hit the cache, not recompute.
        mock_validate.reset_mock()
        results_again = get_oos_validation("BTCUSDT", "1h", cache_path=cache_path)
        assert not mock_validate.called
        assert results_again == results


def test_get_oos_validation_force_refresh_recomputes(tmp_path):
    cache_path = str(tmp_path / "cal.json")
    with (
        patch("agents_crypto.calibration_lookup.backtest_symbol", return_value=["case"]),
        patch("agents_crypto.calibration_lookup.validate_out_of_sample", return_value=_FAKE_OOS),
    ):
        get_oos_validation("BTCUSDT", "1h", cache_path=cache_path)

    with (
        patch("agents_crypto.calibration_lookup.backtest_symbol", return_value=["case"]),
        patch("agents_crypto.calibration_lookup.validate_out_of_sample", return_value=_FAKE_OOS) as mock_validate,
    ):
        get_oos_validation("BTCUSDT", "1h", cache_path=cache_path, force_refresh=True)
        assert mock_validate.called


def test_get_oos_validation_stale_cache_recomputes(tmp_path):
    cache_path = str(tmp_path / "cal.json")
    with (
        patch("agents_crypto.calibration_lookup.backtest_symbol", return_value=["case"]),
        patch("agents_crypto.calibration_lookup.validate_out_of_sample", return_value=_FAKE_OOS),
    ):
        get_oos_validation("BTCUSDT", "1h", cache_path=cache_path, ttl_seconds=0)

    with (
        patch("agents_crypto.calibration_lookup.backtest_symbol", return_value=["case"]),
        patch("agents_crypto.calibration_lookup.validate_out_of_sample", return_value=_FAKE_OOS) as mock_validate,
    ):
        get_oos_validation("BTCUSDT", "1h", cache_path=cache_path, ttl_seconds=0)
        assert mock_validate.called


def test_get_oos_validation_does_not_disturb_cached_stats(tmp_path):
    """Computing OOS results must not clobber an already-cached stats entry
    for the same (symbol, timeframe) — the two are merged into one cache
    entry, written independently."""
    cache_path = str(tmp_path / "cal.json")
    with (
        patch("agents_crypto.calibration_lookup.backtest_symbol", return_value=["case"]),
        patch("agents_crypto.calibration_lookup.summarize", return_value=_FAKE_STATS),
    ):
        get_calibration_stats("BTCUSDT", "1h", cache_path=cache_path)

    with (
        patch("agents_crypto.calibration_lookup.backtest_symbol", return_value=["case"]),
        patch("agents_crypto.calibration_lookup.validate_out_of_sample", return_value=_FAKE_OOS),
    ):
        get_oos_validation("BTCUSDT", "1h", cache_path=cache_path)

    stats = get_cached_calibration_stats("BTCUSDT", "1h", cache_path=cache_path)
    assert len(stats) == 2


def test_find_oos_for_labels_matches_label_and_horizon():
    matches = find_oos_for_labels(_FAKE_OOS, labels=["trend:bearish", "trend:neutral"], horizon_candles=12)
    assert len(matches) == 1
    assert matches[0].label == "trend:bearish"


def test_find_oos_for_labels_filters_by_horizon():
    matches = find_oos_for_labels(_FAKE_OOS, labels=["trend:bearish"], horizon_candles=4)
    assert matches == []
