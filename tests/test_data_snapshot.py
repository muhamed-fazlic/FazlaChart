"""Unit tests for scanner.backtesting.data_snapshot -- the frozen-snapshot
candle cache + dataset-fingerprint helpers used to make backtests
reproducible (see its module docstring for the "why": identical-config runs
against the live, 24/7 scanner were measured to differ by ~9 percentage
points of net profit purely because of data drift between runs)."""

from __future__ import annotations

import pandas as pd
import pytest

from scanner.backtesting.data_snapshot import (
    cached_load_extended_history,
    combined_dataset_fingerprint,
    dataset_fingerprint,
    parse_end_timestamp,
)

pytestmark = pytest.mark.unit


def _history(n: int = 5, close: float = 100.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open_time": pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC"),
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 10.0,
        }
    )


class TestParseEndTimestamp:
    def test_parses_iso_with_z_suffix_as_utc(self):
        assert parse_end_timestamp("2026-01-01T00:00:00Z") == 1767225600000

    def test_parses_iso_with_explicit_offset(self):
        assert parse_end_timestamp("2026-01-01T00:00:00+00:00") == 1767225600000

    def test_parses_naive_iso_as_utc(self):
        assert parse_end_timestamp("2026-01-01T00:00:00") == 1767225600000

    def test_parses_raw_epoch_milliseconds_string(self):
        assert parse_end_timestamp("1767225600000") == 1767225600000


class TestDatasetFingerprint:
    def test_same_data_gives_same_fingerprint(self):
        assert dataset_fingerprint(_history()) == dataset_fingerprint(_history())

    def test_different_data_gives_different_fingerprint(self):
        assert dataset_fingerprint(_history(close=100.0)) != dataset_fingerprint(_history(close=101.0))

    def test_empty_history_does_not_raise(self):
        empty = pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume"])
        assert dataset_fingerprint(empty)  # just must not raise, and return something truthy


class TestCombinedDatasetFingerprint:
    def test_order_independent(self):
        # Two runs that fetched the same per-pair fingerprints in a
        # different order (e.g. a different --symbols ordering) must still
        # be recognised as "the same overall dataset".
        a = combined_dataset_fingerprint(["aaa", "bbb", "ccc"])
        b = combined_dataset_fingerprint(["ccc", "aaa", "bbb"])
        assert a == b

    def test_different_fingerprints_give_different_combined_hash(self):
        a = combined_dataset_fingerprint(["aaa", "bbb"])
        b = combined_dataset_fingerprint(["aaa", "ccc"])
        assert a != b


class TestCachedLoadExtendedHistory:
    def test_none_end_timestamp_bypasses_cache_and_calls_through(self, monkeypatch, tmp_path):
        from scanner.backtesting import data_snapshot as module

        calls: list[tuple] = []

        def fake_load_extended_history(symbol, timeframe, total_candles, end_time=None):
            calls.append((symbol, timeframe, total_candles, end_time))
            return _history()

        monkeypatch.setattr(module, "load_extended_history", fake_load_extended_history)
        result = cached_load_extended_history("BTCUSDT", "1h", 100, None, cache_dir=str(tmp_path))

        assert len(calls) == 1
        assert not list(tmp_path.iterdir()), "live (end_timestamp_ms=None) path must never write to the cache"
        assert len(result) == 5

    def test_locked_timestamp_fetches_once_then_serves_from_cache(self, monkeypatch, tmp_path):
        from scanner.backtesting import data_snapshot as module

        calls: list[tuple] = []

        def fake_load_extended_history(symbol, timeframe, total_candles, end_time=None):
            calls.append((symbol, timeframe, total_candles, end_time))
            return _history()

        monkeypatch.setattr(module, "load_extended_history", fake_load_extended_history)

        first = cached_load_extended_history("BTCUSDT", "1h", 100, 1700000000000, cache_dir=str(tmp_path))
        second = cached_load_extended_history("BTCUSDT", "1h", 100, 1700000000000, cache_dir=str(tmp_path))

        assert len(calls) == 1, "the second call with the same key must be served from disk, not refetched"
        assert calls[0] == ("BTCUSDT", "1h", 100, 1700000000000)
        pd.testing.assert_frame_equal(first, second)

    def test_different_end_timestamp_is_a_different_cache_entry(self, monkeypatch, tmp_path):
        from scanner.backtesting import data_snapshot as module

        calls: list[tuple] = []

        def fake_load_extended_history(symbol, timeframe, total_candles, end_time=None):
            calls.append(end_time)
            return _history()

        monkeypatch.setattr(module, "load_extended_history", fake_load_extended_history)
        cached_load_extended_history("BTCUSDT", "1h", 100, 1700000000000, cache_dir=str(tmp_path))
        cached_load_extended_history("BTCUSDT", "1h", 100, 1700000099999, cache_dir=str(tmp_path))

        assert calls == [1700000000000, 1700000099999]
