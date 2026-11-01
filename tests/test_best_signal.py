"""Unit tests for the Top Signal Now ranking, in particular the correlated-
exposure filter — mocks calibration stats so this never touches Binance."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agents_crypto.best_signal import compute_candidates
from scanner.calibration import CalibrationStats

pytestmark = pytest.mark.unit

_STRONG_SHORT = CalibrationStats(
    label="trend:bearish",
    direction="short",
    horizon_candles=12,
    n=200,
    win_rate_pct=65.0,
    avg_net_return_pct=0.8,
    avg_win_return_pct=1.5,
    avg_loss_return_pct=-0.9,
    win_rate_ci_low_pct=60.0,
    win_rate_ci_high_pct=70.0,
)

_STRONG_LONG = CalibrationStats(
    label="trend:bullish",
    direction="long",
    horizon_candles=12,
    n=200,
    win_rate_pct=65.0,
    avg_net_return_pct=0.8,
    avg_win_return_pct=1.5,
    avg_loss_return_pct=-0.9,
    win_rate_ci_low_pct=60.0,
    win_rate_ci_high_pct=70.0,
)


def _row(symbol: str, trend: str, scanned_at: str = "2026-01-01T00:00:00+00:00") -> dict:
    return {
        "symbol": symbol,
        "scanned_at": scanned_at,
        "reference_timeframe": "1h",
        "close_price": 100.0,
        "trend": trend,
        "anomalies": [],
        "snapshot": {"timeframes": {"1h": {"trend": trend, "close": 100.0}}},
    }


def test_two_bearish_altcoins_aligned_with_bearish_btc_are_flagged_correlated():
    rows = [_row("BTCUSDT", "bearish"), _row("AAAUSDT", "bearish"), _row("BBBUSDT", "bearish")]
    with patch("agents_crypto.calibration_lookup.get_cached_calibration_stats", return_value=[_STRONG_SHORT]):
        result = compute_candidates(rows, limit=5)

    symbols = {s["symbol"] for s in result["signals"]}
    assert symbols == {"BTCUSDT", "AAAUSDT", "BBBUSDT"}
    alt_signals = [s for s in result["signals"] if s["symbol"] in ("AAAUSDT", "BBBUSDT")]
    for sig in alt_signals:
        assert sig["correlation"] is not None
        assert sig["correlation"]["group_size"] == 3
        assert sig["correlation"]["diverges_from_btc"] is False
        assert sig["symbol"] not in sig["correlation"]["symbols"]


def test_lone_signal_has_no_correlation_group():
    rows = [_row("BTCUSDT", "neutral"), _row("AAAUSDT", "bearish")]
    with patch("agents_crypto.calibration_lookup.get_cached_calibration_stats", return_value=[_STRONG_SHORT]):
        result = compute_candidates(rows, limit=5)

    assert len(result["signals"]) == 1
    assert result["signals"][0]["correlation"] is None


def test_signal_diverging_from_btc_is_grouped_separately_from_aligned_ones():
    # BTC bullish; AAA/BBB bearish (diverging from BTC); CCC bearish but BTC
    # itself is bearish too in a second scenario would collide — here keep it
    # simple: two bearish alts against a bullish BTC are a "diverging together"
    # group, not merged with anything BTC-aligned (there's nothing else here).
    rows = [_row("BTCUSDT", "bullish"), _row("AAAUSDT", "bearish"), _row("BBBUSDT", "bearish")]
    with patch("agents_crypto.calibration_lookup.get_cached_calibration_stats", return_value=[_STRONG_SHORT]):
        result = compute_candidates(rows, limit=5)

    assert len(result["signals"]) == 2
    for sig in result["signals"]:
        assert sig["correlation"]["diverges_from_btc"] is True
        assert "odstupaju od BTC" in sig["correlation"]["note"]


def test_missing_btc_row_does_not_crash_and_still_groups_by_direction():
    rows = [_row("AAAUSDT", "bearish"), _row("BBBUSDT", "bearish")]
    with patch("agents_crypto.calibration_lookup.get_cached_calibration_stats", return_value=[_STRONG_SHORT]):
        result = compute_candidates(rows, limit=5)

    assert len(result["signals"]) == 2
    assert all(sig["correlation"]["group_size"] == 2 for sig in result["signals"])


def test_muted_bucket_is_excluded_from_signals():
    rows = [_row("AAAUSDT", "bearish")]
    with patch("agents_crypto.calibration_lookup.get_cached_calibration_stats", return_value=[_STRONG_SHORT]):
        result = compute_candidates(rows, limit=5, muted_buckets={("1h", "short")})

    assert result["signals"] == []
    assert result["muted_buckets"] == [("1h", "short")]


def test_unmuted_direction_is_unaffected_by_a_different_muted_bucket():
    rows = [_row("AAAUSDT", "bearish")]
    with patch("agents_crypto.calibration_lookup.get_cached_calibration_stats", return_value=[_STRONG_SHORT]):
        result = compute_candidates(rows, limit=5, muted_buckets={("4h", "short")})

    assert len(result["signals"]) == 1
    assert result["signals"][0]["symbol"] == "AAAUSDT"


def test_no_muted_buckets_behaves_exactly_as_before():
    rows = [_row("AAAUSDT", "bearish")]
    with patch("agents_crypto.calibration_lookup.get_cached_calibration_stats", return_value=[_STRONG_SHORT]):
        result = compute_candidates(rows, limit=5)

    assert len(result["signals"]) == 1
    assert result["muted_buckets"] == []


def test_different_directions_are_never_grouped_together():
    rows = [_row("BTCUSDT", "neutral"), _row("AAAUSDT", "bearish"), _row("BBBUSDT", "bullish")]

    def _stats(symbol, _timeframe):
        return [_STRONG_SHORT] if symbol == "AAAUSDT" else [_STRONG_LONG]

    with patch("agents_crypto.calibration_lookup.get_cached_calibration_stats", side_effect=_stats):
        result = compute_candidates(rows, limit=5)

    assert len(result["signals"]) == 2
    assert all(sig["correlation"] is None for sig in result["signals"])
