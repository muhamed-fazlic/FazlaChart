"""Unit tests for the generalized cross-sectional engine
(``scanner.backtesting.cross_sectional_engine``) -- Strategy Lab Layer 1
infrastructure (Korak 10). No network calls: everything is built from
small hand-crafted candle DataFrames, funding events, and in-memory dicts.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from scanner.backtesting.cross_sectional_engine import (
    CrossSectionalConfig,
    build_rebalance_grid,
    form_basket,
    funding_carry_ranking_fn,
    low_vol_ranking_fn,
    momentum_ranking_fn,
)
from scanner.backtesting.funding_simulator import FundingEvent

pytestmark = pytest.mark.unit

_T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _dt(hours: float) -> datetime:
    return _T0 + timedelta(hours=hours)


def _history(prices_by_hour: dict[float, float]) -> pd.DataFrame:
    rows = sorted(prices_by_hour.items())
    return pd.DataFrame(
        {
            "open_time": [pd.Timestamp(_dt(h)) for h, _ in rows],
            "close": [c for _, c in rows],
        }
    )


def _momentum_config(**overrides) -> CrossSectionalConfig:
    defaults = {
        "hypothesis": "momentum_30d",
        "ranking_fn": momentum_ranking_fn,
        "lookback_hours": 30 * 24,
        "rebalance_hours": 7 * 24,
        "basket_size": 5,
    }
    defaults.update(overrides)
    return CrossSectionalConfig(**defaults)


class TestMomentumRankingFn:
    def test_higher_trailing_return_scores_higher(self):
        histories = {
            "AAAUSDT": _history({0.0: 100.0, 24.0: 110.0}),
            "BBBUSDT": _history({0.0: 100.0, 24.0: 95.0}),
        }
        scores = momentum_ranking_fn(_dt(24.0), histories, {}, lookback_hours=24)
        assert scores["AAAUSDT"] == pytest.approx(10.0)
        assert scores["BBBUSDT"] == pytest.approx(-5.0)

    def test_no_look_ahead_past_t(self):
        histories = {"AAAUSDT": _history({0.0: 100.0, 24.0: 110.0, 30.0: 1_000.0})}
        scores = momentum_ranking_fn(_dt(24.0), histories, {}, lookback_hours=24)
        assert scores["AAAUSDT"] == pytest.approx(10.0)


class TestLowVolRankingFn:
    def test_calm_symbol_scores_higher_than_turbulent_one(self):
        # AAA barely moves hour to hour; BBB swings wildly -- "low vol"
        # direction means AAA (calm) must score HIGHER (long leg).
        calm = _history({h: 100.0 + (0.1 if h % 2 == 0 else -0.1) for h in range(0, 25)})
        turbulent = _history({h: 100.0 + (20.0 if h % 2 == 0 else -20.0) for h in range(0, 25)})
        histories = {"AAAUSDT": calm, "BBBUSDT": turbulent}
        scores = low_vol_ranking_fn(_dt(24.0), histories, {}, lookback_hours=24)
        assert scores["AAAUSDT"] > scores["BBBUSDT"]

    def test_too_few_candles_excludes_symbol(self):
        histories = {"AAAUSDT": _history({24.0: 100.0})}
        scores = low_vol_ranking_fn(_dt(24.0), histories, {}, lookback_hours=24)
        assert "AAAUSDT" not in scores


class TestFundingCarryRankingFn:
    def test_most_negative_avg_funding_scores_highest(self):
        histories = {"AAAUSDT": _history({24.0: 100.0}), "BBBUSDT": _history({24.0: 100.0})}
        funding_by_symbol = {
            # Negative funding -> longs are PAID -> most attractive long.
            "AAAUSDT": [FundingEvent(timestamp=_dt(12.0), rate=-0.0005)],
            # Positive funding -> shorts are PAID -> most attractive short.
            "BBBUSDT": [FundingEvent(timestamp=_dt(12.0), rate=0.0005)],
        }
        scores = funding_carry_ranking_fn(_dt(24.0), histories, funding_by_symbol, lookback_hours=24)
        assert scores["AAAUSDT"] > scores["BBBUSDT"]

    def test_settlement_outside_lookback_window_excluded(self):
        histories = {"AAAUSDT": _history({24.0: 100.0})}
        funding_by_symbol = {"AAAUSDT": [FundingEvent(timestamp=_dt(-100.0), rate=-0.0005)]}
        scores = funding_carry_ranking_fn(_dt(24.0), histories, funding_by_symbol, lookback_hours=24)
        assert "AAAUSDT" not in scores


class TestFormBasketRespectsConfiguredSize:
    def test_fewer_than_min_eligible_skips(self):
        config = _momentum_config(basket_size=5)
        scores = {f"SYM{i}USDT": float(i) for i in range(config.min_eligible_symbols - 1)}
        assert form_basket(scores, config) is None

    def test_forms_long_and_short_legs_of_configured_size(self):
        config = _momentum_config(basket_size=3)
        scores = {f"SYM{i}USDT": float(i) for i in range(20)}
        basket = form_basket(scores, config)
        longs = [s for s, leg in basket.items() if leg == "long"]
        shorts = [s for s, leg in basket.items() if leg == "short"]
        assert len(longs) == 3
        assert len(shorts) == 3


class TestBuildRebalanceGrid:
    def test_steps_by_configured_rebalance_hours(self):
        histories = {"AAAUSDT": _history({0.0: 100.0, 48.0: 100.0})}
        grid = build_rebalance_grid(histories, rebalance_hours=24)
        assert grid == [_dt(0.0), _dt(24.0), _dt(48.0)]


def _linear_price_history(hours: int, start_price: float, slope_per_hour: float) -> pd.DataFrame:
    """Hourly candle series ``start_price + slope_per_hour * h`` for
    ``h in [0, hours]`` -- deterministic enough that ranking order between
    symbols is known in advance (needed to pin down exactly which basket
    ``run_backtest`` should pick at a given rebalance time)."""
    open_times = [_dt(float(h)) for h in range(hours + 1)]
    closes = [start_price + slope_per_hour * h for h in range(hours + 1)]
    return pd.DataFrame(
        {
            "open_time": open_times,
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": 10.0,
        }
    )


class TestRunBacktestIsolationBoundary:
    """Korak 10 item 2/6 leakage fix: ``start_timestamp_ms`` must (a) let
    every rebalance's ranking signal use lookback data from before the
    boundary (legitimate warm-up), while (b) never reporting a period
    whose own rebalance time is before the boundary. A regression that
    instead truncated the loaded candle history itself at the boundary
    (the "obvious but wrong" fix) would silently swap in the wrong basket
    for any period whose lookback dips before the boundary -- exactly what
    ``test_warmup_history_before_boundary_still_drives_ranking`` below
    would catch.
    """

    SYMBOLS = ("AUSDT", "BUSDT", "CUSDT", "DUSDT")

    def _histories(self) -> dict[str, pd.DataFrame]:
        # A/B: tiny, slow drift (never the strongest momentum signal).
        # C/D: strong, opposite-signed trends across the whole series --
        # the clear long/short pick whenever their full trailing history
        # is visible to the ranking function.
        return {
            "AUSDT": _linear_price_history(199, 1000.0, 0.1),
            "BUSDT": _linear_price_history(199, 1000.0, -0.1),
            "CUSDT": _linear_price_history(199, 1000.0, 3.0),
            "DUSDT": _linear_price_history(199, 1000.0, -3.0),
        }

    def _patch_data_loading(self, monkeypatch, histories: dict[str, pd.DataFrame]):
        from scanner.backtesting import cross_sectional_engine as module

        def fake_load_history(symbol, timeframe, total_candles, end_timestamp_ms, cache_dir):
            return histories[symbol]

        def fake_load_funding(symbol, end_timestamp_ms, cache_dir):
            return []

        monkeypatch.setattr(module, "cached_load_extended_history", fake_load_history)
        monkeypatch.setattr(module, "cached_fetch_full_funding_history", fake_load_funding)

    def _config(self) -> CrossSectionalConfig:
        return CrossSectionalConfig(
            hypothesis="test isolation boundary",
            ranking_fn=momentum_ranking_fn,
            lookback_hours=48,
            rebalance_hours=24,
            basket_size=1,
            min_eligible_multiplier=2,
        )

    def test_no_returned_period_starts_before_the_boundary(self, monkeypatch):
        from scanner.backtesting.cross_sectional_engine import run_backtest

        histories = self._histories()
        self._patch_data_loading(monkeypatch, histories)
        boundary = _dt(120.0)

        report = run_backtest(
            self._config(),
            self.SYMBOLS,
            end_timestamp_ms=int(_dt(199.0).timestamp() * 1000),
            start_timestamp_ms=int(boundary.timestamp() * 1000),
        )

        assert report["rebalance_periods"], "expected at least one period at/after the boundary"
        for period in report["rebalance_periods"]:
            assert datetime.fromisoformat(period["rebalance_time"]) >= boundary

    def test_warmup_history_before_boundary_still_drives_ranking(self, monkeypatch):
        """The first included period (rebalance_time == boundary, 48h
        lookback) must rank symbols using its FULL 48h lookback -- which
        dips 48h before the boundary -- not just data at/after it. CUSDT
        (strong uptrend) must win the long leg and DUSDT (strong
        downtrend) the short leg; if lookback data before the boundary
        were wrongly excluded, C/D would have no valid trailing price at
        all and the weak A/B drift would be picked instead."""
        from scanner.backtesting.cross_sectional_engine import run_backtest

        histories = self._histories()
        self._patch_data_loading(monkeypatch, histories)
        boundary = _dt(120.0)

        report = run_backtest(
            self._config(),
            self.SYMBOLS,
            end_timestamp_ms=int(_dt(199.0).timestamp() * 1000),
            start_timestamp_ms=int(boundary.timestamp() * 1000),
        )

        first_period = next(
            p for p in report["rebalance_periods"] if datetime.fromisoformat(p["rebalance_time"]) == boundary
        )
        assert first_period["basket"] == {"CUSDT": "long", "DUSDT": "short"}

    def test_none_start_timestamp_matches_manually_filtered_full_history(self, monkeypatch):
        """Regression check: running with ``start_timestamp_ms=None``
        (Layer 1's own call shape, unchanged) must reproduce exactly the
        periods a full, unfiltered run would have produced for the same
        window -- i.e. filtering by boundary changes nothing about Layer
        1's own full-history results."""
        from scanner.backtesting.cross_sectional_engine import run_backtest

        histories = self._histories()
        self._patch_data_loading(monkeypatch, histories)
        end_ms = int(_dt(199.0).timestamp() * 1000)

        full_report = run_backtest(self._config(), self.SYMBOLS, end_timestamp_ms=end_ms, start_timestamp_ms=None)
        boundary = _dt(120.0)
        filtered_report = run_backtest(
            self._config(), self.SYMBOLS, end_timestamp_ms=end_ms, start_timestamp_ms=int(boundary.timestamp() * 1000)
        )

        full_periods_after_boundary = [
            p for p in full_report["rebalance_periods"] if datetime.fromisoformat(p["rebalance_time"]) >= boundary
        ]
        assert full_periods_after_boundary == filtered_report["rebalance_periods"]
        assert full_report["start_timestamp_ms"] is None
        assert filtered_report["start_timestamp_ms"] == int(boundary.timestamp() * 1000)
