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


def _multi_tf_row(symbol: str, trend_by_tf: dict[str, str], scanned_at: str = "2026-01-01T00:00:00+00:00") -> dict:
    """A row whose snapshot carries a different trend per timeframe --
    exercises the multi-timeframe candidate search directly, instead of
    every timeframe collapsing to the one flattened ``trend`` field."""
    return {
        "symbol": symbol,
        "scanned_at": scanned_at,
        "reference_timeframe": "1h",
        "close_price": 100.0,
        "trend": trend_by_tf.get("1h", "neutral"),
        "anomalies": [],
        "snapshot": {"timeframes": {tf: {"trend": trend, "close": 100.0} for tf, trend in trend_by_tf.items()}},
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


class TestMultiTimeframe:
    """A symbol can qualify on more than one timeframe at once -- each
    timeframe is searched and calibrated independently of the others."""

    def test_one_symbol_qualifying_on_two_timeframes_surfaces_both(self):
        rows = [_multi_tf_row("AAAUSDT", {"1h": "bearish", "4h": "bullish"})]

        def _stats(_symbol, timeframe):
            return [_STRONG_SHORT] if timeframe == "1h" else [_STRONG_LONG]

        with patch("agents_crypto.calibration_lookup.get_cached_calibration_stats", side_effect=_stats):
            result = compute_candidates(rows, limit=5)

        by_tf = {sig["timeframe"]: sig for sig in result["signals"]}
        assert set(by_tf) == {"1h", "4h"}
        assert by_tf["1h"]["direction"] == "short"
        assert by_tf["4h"]["direction"] == "long"

    def test_signals_by_timeframe_groups_each_timeframe_independently(self):
        rows = [
            _multi_tf_row("AAAUSDT", {"1h": "bearish"}),
            _multi_tf_row("BBBUSDT", {"4h": "bearish"}),
        ]
        with patch("agents_crypto.calibration_lookup.get_cached_calibration_stats", return_value=[_STRONG_SHORT]):
            result = compute_candidates(rows, limit=5)

        assert [s["symbol"] for s in result["signals_by_timeframe"]["1h"]] == ["AAAUSDT"]
        assert [s["symbol"] for s in result["signals_by_timeframe"]["4h"]] == ["BBBUSDT"]
        assert result["signals_by_timeframe"]["5m"] == []
        assert result["signals_by_timeframe"]["15m"] == []

    def test_limit_caps_each_timeframe_independently_not_the_overall_total(self):
        rows = [
            _multi_tf_row("AAAUSDT", {"1h": "bearish", "4h": "bearish"}),
            _multi_tf_row("BBBUSDT", {"1h": "bearish", "4h": "bearish"}),
            _multi_tf_row("CCCUSDT", {"1h": "bearish", "4h": "bearish"}),
        ]
        with patch("agents_crypto.calibration_lookup.get_cached_calibration_stats", return_value=[_STRONG_SHORT]):
            result = compute_candidates(rows, limit=2)

        assert len(result["signals_by_timeframe"]["1h"]) == 2
        assert len(result["signals_by_timeframe"]["4h"]) == 2
        assert len(result["signals"]) == 4  # 2 per timeframe x 2 timeframes, not capped at 2 overall

    def test_muting_one_timeframes_bucket_does_not_affect_another_timeframe(self):
        rows = [_multi_tf_row("AAAUSDT", {"1h": "bearish", "4h": "bearish"})]
        with patch("agents_crypto.calibration_lookup.get_cached_calibration_stats", return_value=[_STRONG_SHORT]):
            result = compute_candidates(rows, limit=5, muted_buckets={("1h", "short")})

        assert result["signals_by_timeframe"]["1h"] == []
        assert [s["timeframe"] for s in result["signals_by_timeframe"]["4h"]] == ["4h"]

    def test_anomaly_label_is_looked_up_per_timeframe_unprefixed(self):
        """Each timeframe's own anomaly list (unprefixed) must be used to
        build labels, not the flattened cross-timeframe list (which would be
        prefixed like "1h:failed_breakout" and could never match a
        calibration label such as "anomaly:failed_breakout")."""
        row = _multi_tf_row("AAAUSDT", {"1h": "bearish"})
        row["snapshot"]["timeframes"]["1h"]["anomalies"] = ["failed_breakout"]
        row["anomalies"] = ["1h:failed_breakout"]  # flattened/prefixed -- must be ignored for label-building

        seen_labels = {}

        def _find_stats(_stats, labels, _horizon):
            seen_labels["labels"] = labels
            return [_STRONG_SHORT]

        with (
            patch("agents_crypto.calibration_lookup.get_cached_calibration_stats", return_value=[_STRONG_SHORT]),
            patch("agents_crypto.calibration_lookup.find_stats_for_labels", side_effect=_find_stats),
        ):
            compute_candidates([row], limit=5)

        assert "anomaly:failed_breakout" in seen_labels["labels"]
        assert "anomaly:1h:failed_breakout" not in seen_labels["labels"]
