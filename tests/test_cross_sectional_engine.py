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
