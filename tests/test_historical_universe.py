"""Unit tests for Phase 8 Korak 4's point-in-time universe builder
(``scanner.backtesting.historical_universe``). No real network calls --
``client.get_klines``/``get_usdt_perpetual_symbols`` are monkeypatched with
synthetic data so every test is deterministic and precisely controls what
"no look-ahead" would mean for the volume ranking."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from scanner.backtesting import historical_universe as module
from scanner.backtesting.historical_universe import (
    PointInTimeUniverseConfig,
    RankedUniverseWindow,
    UniverseWindow,
    build_point_in_time_universe,
    build_ranked_point_in_time_universe,
    is_symbol_in_universe_at,
    load_ranked_universe_cache,
    load_universe_cache,
    save_ranked_universe_cache,
    save_universe_cache,
    symbols_at,
    windows_for_rank_range,
)
from tradingagents.dataflows.errors import NoMarketDataError

pytestmark = pytest.mark.unit

_T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _daily_klines(n: int, quote_volume: float) -> list[dict]:
    """``n`` daily candles, each with the same ``quote_volume``, ending just
    before whatever ``end_time`` the caller requested (the fake client below
    ignores the exact candle timestamps -- only the count/volume matter for
    these tests)."""
    return [{"quote_volume": str(quote_volume)} for _ in range(n)]


class TestAvgDailyQuoteVolume:
    def test_insufficient_history_returns_none(self, monkeypatch):
        monkeypatch.setattr(module.client, "get_klines", lambda *a, **kw: _daily_klines(3, 1000.0))
        result = module._avg_daily_quote_volume("NEWUSDT", _T0, lookback_days=30, min_history_days=7)
        assert result is None

    def test_enough_history_returns_average(self, monkeypatch):
        monkeypatch.setattr(module.client, "get_klines", lambda *a, **kw: _daily_klines(10, 500.0))
        result = module._avg_daily_quote_volume("BTCUSDT", _T0, lookback_days=30, min_history_days=7)
        assert result == pytest.approx(500.0)

    def test_vendor_error_returns_none(self, monkeypatch):
        def raise_error(*a, **kw):
            raise NoMarketDataError("GONEUSDT")

        monkeypatch.setattr(module.client, "get_klines", raise_error)
        result = module._avg_daily_quote_volume("GONEUSDT", _T0, lookback_days=30, min_history_days=7)
        assert result is None


class TestBuildPointInTimeUniverse:
    def test_ranks_by_trailing_volume_and_respects_top_n(self, monkeypatch):
        monkeypatch.setattr(module.client, "get_usdt_perpetual_symbols", lambda: ["BTCUSDT", "ETHUSDT", "DOGEUSDT"])

        volumes = {"BTCUSDT": 3000.0, "ETHUSDT": 2000.0, "DOGEUSDT": 1000.0}

        def fake_get_klines(symbol, interval, limit, end_time=None):
            return _daily_klines(limit, volumes[symbol])

        monkeypatch.setattr(module.client, "get_klines", fake_get_klines)

        config = PointInTimeUniverseConfig(
            start=_T0, end=_T0 + timedelta(days=30), top_n=2, rebalance_days=30, min_history_days=1
        )
        windows = build_point_in_time_universe(config)

        assert len(windows) == 1
        assert windows[0].symbols == frozenset({"BTCUSDT", "ETHUSDT"})

    def test_excludes_symbols_without_enough_history(self, monkeypatch):
        monkeypatch.setattr(module.client, "get_usdt_perpetual_symbols", lambda: ["BTCUSDT", "TOOUSDT"])

        def fake_get_klines(symbol, interval, limit, end_time=None):
            if symbol == "TOOUSDT":
                return _daily_klines(2, 999_999.0)  # huge volume, but not enough history
            return _daily_klines(30, 100.0)

        monkeypatch.setattr(module.client, "get_klines", fake_get_klines)

        config = PointInTimeUniverseConfig(
            start=_T0, end=_T0 + timedelta(days=30), top_n=5, rebalance_days=30, min_history_days=7
        )
        windows = build_point_in_time_universe(config)

        assert windows[0].symbols == frozenset({"BTCUSDT"})

    def test_multiple_rebalance_windows_span_the_full_range(self, monkeypatch):
        monkeypatch.setattr(module.client, "get_usdt_perpetual_symbols", lambda: ["BTCUSDT"])
        monkeypatch.setattr(module.client, "get_klines", lambda *a, **kw: _daily_klines(30, 100.0))

        config = PointInTimeUniverseConfig(
            start=_T0, end=_T0 + timedelta(days=65), top_n=5, rebalance_days=30, min_history_days=1
        )
        windows = build_point_in_time_universe(config)

        assert len(windows) == 3  # 30 + 30 + 5 day remainder
        assert windows[0].start == _T0
        assert windows[-1].end == _T0 + timedelta(days=65)

    def test_candidate_symbols_overrides_auto_discovery(self, monkeypatch):
        def fail_if_called():
            raise AssertionError("should not auto-discover when candidate_symbols is given")

        monkeypatch.setattr(module.client, "get_usdt_perpetual_symbols", lambda: fail_if_called())
        monkeypatch.setattr(module.client, "get_klines", lambda *a, **kw: _daily_klines(10, 100.0))

        config = PointInTimeUniverseConfig(
            start=_T0,
            end=_T0 + timedelta(days=30),
            candidate_symbols=("BTCUSDT",),
            min_history_days=1,
        )
        windows = build_point_in_time_universe(config)
        assert windows[0].symbols == frozenset({"BTCUSDT"})

    def test_include_delisted_adds_to_candidate_set(self, monkeypatch):
        monkeypatch.setattr(module.client, "get_usdt_perpetual_symbols", lambda: ["BTCUSDT"])
        monkeypatch.setattr(module.client, "get_klines", lambda *a, **kw: _daily_klines(10, 100.0))

        config = PointInTimeUniverseConfig(
            start=_T0,
            end=_T0 + timedelta(days=30),
            top_n=5,
            include_delisted=("DELISTEDUSDT",),
            min_history_days=1,
        )
        windows = build_point_in_time_universe(config)
        assert windows[0].symbols == frozenset({"BTCUSDT", "DELISTEDUSDT"})


class TestSymbolLookup:
    def _windows(self) -> list[UniverseWindow]:
        return [
            UniverseWindow(start=_T0, end=_T0 + timedelta(days=30), symbols=frozenset({"BTCUSDT"})),
            UniverseWindow(
                start=_T0 + timedelta(days=30),
                end=_T0 + timedelta(days=60),
                symbols=frozenset({"ETHUSDT"}),
            ),
        ]

    def test_symbols_at_returns_the_covering_window(self):
        windows = self._windows()
        assert symbols_at(windows, _T0 + timedelta(days=5)) == frozenset({"BTCUSDT"})
        assert symbols_at(windows, _T0 + timedelta(days=35)) == frozenset({"ETHUSDT"})

    def test_symbols_at_outside_every_window_is_none(self):
        windows = self._windows()
        assert symbols_at(windows, _T0 - timedelta(days=1)) is None
        assert symbols_at(windows, _T0 + timedelta(days=90)) is None

    def test_is_symbol_in_universe_at_true_and_false(self):
        windows = self._windows()
        assert is_symbol_in_universe_at(windows, "BTCUSDT", _T0 + timedelta(days=5)) is True
        assert is_symbol_in_universe_at(windows, "ETHUSDT", _T0 + timedelta(days=5)) is False

    def test_is_symbol_in_universe_at_outside_range_is_false(self):
        windows = self._windows()
        assert is_symbol_in_universe_at(windows, "BTCUSDT", _T0 + timedelta(days=90)) is False


class TestCacheRoundTrip:
    def test_save_and_load_preserves_windows(self, tmp_path):
        windows = [
            UniverseWindow(start=_T0, end=_T0 + timedelta(days=30), symbols=frozenset({"BTCUSDT", "ETHUSDT"})),
        ]
        path = tmp_path / "cache.json"
        save_universe_cache(windows, path)

        loaded = load_universe_cache(path)
        assert loaded == windows

    def test_cache_file_is_human_readable_json(self, tmp_path):
        windows = [UniverseWindow(start=_T0, end=_T0 + timedelta(days=30), symbols=frozenset({"BTCUSDT"}))]
        path = tmp_path / "cache.json"
        save_universe_cache(windows, path)

        raw = json.loads(path.read_text(encoding="utf-8"))
        assert raw == [
            {
                "start": _T0.isoformat(),
                "end": (_T0 + timedelta(days=30)).isoformat(),
                "symbols": ["BTCUSDT"],
            }
        ]


class TestBuildRankedPointInTimeUniverse:
    """Korak 10: the rank-preserving builder (used to slice a wide top-N
    universe, e.g. top-170, into liquidity terciles) must rank identically
    to the unordered ``build_point_in_time_universe`` -- only the output
    shape (ordered tuple vs. frozenset) differs."""

    def test_preserves_full_rank_order_best_first(self, monkeypatch):
        monkeypatch.setattr(
            module.client, "get_usdt_perpetual_symbols", lambda: ["BTCUSDT", "ETHUSDT", "DOGEUSDT", "ADAUSDT"]
        )
        volumes = {"BTCUSDT": 4000.0, "ETHUSDT": 3000.0, "DOGEUSDT": 2000.0, "ADAUSDT": 1000.0}

        def fake_get_klines(symbol, interval, limit, end_time=None):
            return _daily_klines(limit, volumes[symbol])

        monkeypatch.setattr(module.client, "get_klines", fake_get_klines)

        config = PointInTimeUniverseConfig(
            start=_T0, end=_T0 + timedelta(days=30), top_n=3, rebalance_days=30, min_history_days=1
        )
        windows = build_ranked_point_in_time_universe(config)

        assert len(windows) == 1
        assert windows[0].ranked_symbols == ("BTCUSDT", "ETHUSDT", "DOGEUSDT")

    def test_does_not_mutate_or_replace_existing_unordered_builder(self, monkeypatch):
        """Backward-compat guard: the new ranked builder must not change
        what the original unordered builder returns for the same config."""
        monkeypatch.setattr(module.client, "get_usdt_perpetual_symbols", lambda: ["BTCUSDT", "ETHUSDT"])
        monkeypatch.setattr(module.client, "get_klines", lambda *a, **kw: _daily_klines(10, 100.0))

        config = PointInTimeUniverseConfig(
            start=_T0, end=_T0 + timedelta(days=30), top_n=5, rebalance_days=30, min_history_days=1
        )
        unordered = build_point_in_time_universe(config)
        ranked = build_ranked_point_in_time_universe(config)

        assert unordered[0].symbols == frozenset(ranked[0].ranked_symbols)


class TestWindowsForRankRange:
    def _ranked_windows(self) -> list[RankedUniverseWindow]:
        return [
            RankedUniverseWindow(
                start=_T0,
                end=_T0 + timedelta(days=30),
                ranked_symbols=("BTCUSDT", "ETHUSDT", "DOGEUSDT", "ADAUSDT", "SOLUSDT"),
            )
        ]

    def test_slices_a_middle_tier_inclusive_1_indexed(self):
        windows = self._ranked_windows()
        tier = windows_for_rank_range(windows, 2, 4)
        assert len(tier) == 1
        assert isinstance(tier[0], UniverseWindow)
        assert tier[0].symbols == frozenset({"ETHUSDT", "DOGEUSDT", "ADAUSDT"})
        assert tier[0].start == _T0
        assert tier[0].end == _T0 + timedelta(days=30)

    def test_top_tier_matches_rank_1_to_n(self):
        windows = self._ranked_windows()
        tier = windows_for_rank_range(windows, 1, 2)
        assert tier[0].symbols == frozenset({"BTCUSDT", "ETHUSDT"})

    def test_rank_range_beyond_available_candidates_is_partial_not_error(self):
        windows = self._ranked_windows()
        tier = windows_for_rank_range(windows, 4, 10)
        assert tier[0].symbols == frozenset({"ADAUSDT", "SOLUSDT"})

    def test_invalid_rank_range_raises(self):
        windows = self._ranked_windows()
        with pytest.raises(ValueError):
            windows_for_rank_range(windows, 0, 5)
        with pytest.raises(ValueError):
            windows_for_rank_range(windows, 5, 2)

    def test_three_non_overlapping_tiers_partition_the_full_top_n(self):
        windows = self._ranked_windows()
        high = windows_for_rank_range(windows, 1, 2)
        mid = windows_for_rank_range(windows, 3, 4)
        low = windows_for_rank_range(windows, 5, 5)
        union = high[0].symbols | mid[0].symbols | low[0].symbols
        assert union == frozenset(windows[0].ranked_symbols)
        assert not (high[0].symbols & mid[0].symbols)
        assert not (mid[0].symbols & low[0].symbols)


class TestRankedCacheRoundTrip:
    def test_save_and_load_preserves_ranked_windows(self, tmp_path):
        windows = [
            RankedUniverseWindow(
                start=_T0, end=_T0 + timedelta(days=30), ranked_symbols=("BTCUSDT", "ETHUSDT")
            ),
        ]
        path = tmp_path / "ranked_cache.json"
        save_ranked_universe_cache(windows, path)

        loaded = load_ranked_universe_cache(path)
        assert loaded == windows

    def test_ranked_cache_file_is_human_readable_json_with_order_preserved(self, tmp_path):
        windows = [
            RankedUniverseWindow(
                start=_T0, end=_T0 + timedelta(days=30), ranked_symbols=("ETHUSDT", "BTCUSDT")
            )
        ]
        path = tmp_path / "ranked_cache.json"
        save_ranked_universe_cache(windows, path)

        raw = json.loads(path.read_text(encoding="utf-8"))
        assert raw == [
            {
                "start": _T0.isoformat(),
                "end": (_T0 + timedelta(days=30)).isoformat(),
                "ranked_symbols": ["ETHUSDT", "BTCUSDT"],
            }
        ]
