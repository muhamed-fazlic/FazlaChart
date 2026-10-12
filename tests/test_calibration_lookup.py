"""Unit tests for the calibration cache — mocks the (network-bound) backtest
itself so these never touch Binance."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agents_crypto.calibration_lookup import find_stats_for_labels, get_calibration_stats
from scanner.calibration import CalibrationStats

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
