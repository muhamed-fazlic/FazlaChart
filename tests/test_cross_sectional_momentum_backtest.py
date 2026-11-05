"""Unit tests for the cross-sectional momentum backtest
(``scanner.backtesting.cross_sectional_momentum_backtest``) -- see
``PREREGISTRATION_CROSS_SECTIONAL_MOMENTUM.md`` for the frozen spec these
tests check conformance to. No network calls: everything is built from
small hand-crafted candle DataFrames and in-memory dicts.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from scanner.backtesting.cross_sectional_momentum_backtest import (
    BASKET_SIZE,
    MIN_ELIGIBLE_SYMBOLS,
    build_rebalance_period,
    changed_symbols_and_turnover,
    eligible_symbols_at,
    form_basket,
    pearson_correlation,
)
from scanner.calibration import DEFAULT_ROUND_TRIP_COST_PCT, round_trip_cost_pct_for

pytestmark = pytest.mark.unit

_T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _dt(hours: float) -> datetime:
    return _T0 + timedelta(hours=hours)


def _history(prices_by_hour: dict[float, float]) -> pd.DataFrame:
    """Builds a minimal candle DataFrame with just the columns this module
    reads (``open_time``, ``close``), one row per ``{hour_offset: close}``
    entry, sorted chronologically."""
    rows = sorted(prices_by_hour.items())
    return pd.DataFrame(
        {
            "open_time": [pd.Timestamp(_dt(h)) for h, _ in rows],
            "close": [c for _, c in rows],
        }
    )


class TestEligibleSymbolsAt24hNoLookAhead:
    def test_computes_24h_trailing_return_using_only_past_prices(self):
        # Symbol goes 100 -> 110 over the 24h window ending at t=24.
        histories = {"AAAUSDT": _history({0.0: 100.0, 24.0: 110.0})}
        momentum = eligible_symbols_at(_dt(24.0), histories, None)
        assert momentum["AAAUSDT"] == pytest.approx(10.0)

    def test_ignores_price_data_after_t_even_if_present(self):
        # A price spike AFTER t (hour 30) must not leak into the 24h
        # return computed AT t=24 -- confirms no look-ahead.
        histories = {"AAAUSDT": _history({0.0: 100.0, 24.0: 110.0, 30.0: 1000.0})}
        momentum = eligible_symbols_at(_dt(24.0), histories, None)
        assert momentum["AAAUSDT"] == pytest.approx(10.0)

    def test_symbol_missing_t_minus_24h_price_is_excluded(self):
        # No candle at/before hour 0 -- symbol listed too late to rank at
        # t=24 (only its hour-24 price exists).
        histories = {"AAAUSDT": _history({24.0: 110.0})}
        momentum = eligible_symbols_at(_dt(24.0), histories, None)
        assert "AAAUSDT" not in momentum


class TestMinimumEligibleSkipRule:
    def test_fewer_than_ten_eligible_skips_the_rebalance(self):
        momentum = {f"SYM{i}USDT": float(i) for i in range(MIN_ELIGIBLE_SYMBOLS - 1)}
        assert form_basket(momentum) is None

    def test_exactly_ten_eligible_forms_a_basket(self):
        momentum = {f"SYM{i}USDT": float(i) for i in range(MIN_ELIGIBLE_SYMBOLS)}
        basket = form_basket(momentum)
        assert basket is not None
        assert len(basket) == MIN_ELIGIBLE_SYMBOLS

    def test_never_shrinks_basket_size_below_frozen_n(self):
        # 20 eligible symbols -> still exactly N long + N short, not more.
        momentum = {f"SYM{i}USDT": float(i) for i in range(20)}
        basket = form_basket(momentum)
        longs = [s for s, leg in basket.items() if leg == "long"]
        shorts = [s for s, leg in basket.items() if leg == "short"]
        assert len(longs) == BASKET_SIZE
        assert len(shorts) == BASKET_SIZE


class TestBasketFormationTopBottom:
    def test_top_n_go_long_bottom_n_go_short(self):
        # 10 symbols ranked 0..9 by momentum value -- top 5 (5..9) long,
        # bottom 5 (0..4) short.
        momentum = {f"SYM{i}USDT": float(i) for i in range(10)}
        basket = form_basket(momentum)
        for i in range(5, 10):
            assert basket[f"SYM{i}USDT"] == "long"
        for i in range(0, 5):
            assert basket[f"SYM{i}USDT"] == "short"

    def test_middle_symbols_excluded_from_basket(self):
        # With exactly MIN_ELIGIBLE_SYMBOLS (10) == 2*N, no symbol is
        # left out; with MORE than 10 eligible, the middle ones should be
        # excluded entirely (neither long nor short).
        momentum = {f"SYM{i}USDT": float(i) for i in range(14)}
        basket = form_basket(momentum)
        assert len(basket) == 2 * BASKET_SIZE
        # Ranked 0 (lowest) .. 13 (highest): bottom 5 (0-4) short, top 5
        # (9-13) long, middle 4 (5-8) excluded from the basket entirely.
        for i in range(5, 9):
            assert f"SYM{i}USDT" not in basket


class TestTurnoverCostLogic:
    def test_symbol_staying_in_same_leg_pays_no_cost_and_is_not_turnover(self):
        prev = {"AAAUSDT": "long", "BBBUSDT": "short"}
        curr = {"AAAUSDT": "long", "BBBUSDT": "short"}
        changed, turnover = changed_symbols_and_turnover(prev, curr)
        assert changed == set()
        assert turnover == 0.0

    def test_symbol_entering_a_leg_pays_full_cost(self):
        prev = {"AAAUSDT": "long"}
        curr = {"AAAUSDT": "long", "BBBUSDT": "short"}
        changed, turnover = changed_symbols_and_turnover(prev, curr)
        assert changed == {"BBBUSDT"}
        assert turnover == pytest.approx(0.5)  # 1 of 2 current slots changed

    def test_symbol_exiting_a_leg_entirely_pays_full_cost(self):
        # BBBUSDT was short at t-1, absent entirely at t.
        prev = {"AAAUSDT": "long", "BBBUSDT": "short"}
        curr = {"AAAUSDT": "long"}
        changed, turnover = changed_symbols_and_turnover(prev, curr)
        assert changed == {"BBBUSDT"}
        # turnover is measured over TODAY's basket slots -- today has
        # only 1 slot and it did not change.
        assert turnover == 0.0

    def test_symbol_flipping_legs_pays_cost_exactly_once(self):
        prev = {"AAAUSDT": "long"}
        curr = {"AAAUSDT": "short"}
        changed, turnover = changed_symbols_and_turnover(prev, curr)
        assert changed == {"AAAUSDT"}
        assert turnover == pytest.approx(1.0)

    def test_first_rebalance_has_no_prior_basket_everything_is_new(self):
        curr = {"AAAUSDT": "long", "BBBUSDT": "short"}
        changed, turnover = changed_symbols_and_turnover(None, curr)
        assert changed == {"AAAUSDT", "BBBUSDT"}
        assert turnover == pytest.approx(1.0)

    def test_cost_scaling_matches_round_trip_cost_pct_for(self):
        # Sanity check: the per-symbol cost used by build_rebalance_period
        # is round_trip_cost_pct_for(symbol), scaled by weight -- confirm
        # the calibration helper itself returns the expected base rate for
        # an arbitrary (non-tiered) symbol.
        assert round_trip_cost_pct_for("ZZZUSDT", DEFAULT_ROUND_TRIP_COST_PCT) >= DEFAULT_ROUND_TRIP_COST_PCT


class TestBuildRebalancePeriodCostAndReturn:
    def test_same_leg_continuation_pays_zero_cost(self):
        # Two symbols, both staying in the same leg between prev and
        # current basket -- cost_pct must be exactly 0.
        histories = {
            "AAAUSDT": _history({0.0: 100.0, 24.0: 110.0}),
            "BBBUSDT": _history({0.0: 50.0, 24.0: 45.0}),
        }
        basket = {"AAAUSDT": "long", "BBBUSDT": "short"}
        period = build_rebalance_period(
            _dt(0.0), _dt(24.0), {}, basket, basket, histories, {}, DEFAULT_ROUND_TRIP_COST_PCT
        )
        assert period is not None
        assert period.cost_pct == 0.0
        assert period.turnover == 0.0

    def test_new_basket_pays_full_scaled_cost(self):
        histories = {
            "AAAUSDT": _history({0.0: 100.0, 24.0: 110.0}),
        }
        basket = {"AAAUSDT": "long"}
        period = build_rebalance_period(
            _dt(0.0), _dt(24.0), {}, basket, None, histories, {}, DEFAULT_ROUND_TRIP_COST_PCT
        )
        assert period is not None
        expected_weight = 0.5 / BASKET_SIZE
        expected_cost = expected_weight * round_trip_cost_pct_for("AAAUSDT", DEFAULT_ROUND_TRIP_COST_PCT)
        assert period.cost_pct == pytest.approx(expected_cost, rel=1e-6)

    def test_no_candle_data_at_or_before_entry_returns_none(self):
        # Entry time (hour 0) is before the symbol's earliest available
        # candle (hour 40) -- no price can be found without look-ahead,
        # so the whole period must be rejected rather than silently
        # substituting a future price.
        histories = {"AAAUSDT": _history({40.0: 100.0})}
        basket = {"AAAUSDT": "long"}
        period = build_rebalance_period(
            _dt(0.0), _dt(24.0), {}, basket, None, histories, {}, DEFAULT_ROUND_TRIP_COST_PCT
        )
        assert period is None


class TestBetaCorrelation:
    def test_perfectly_correlated_series_returns_one(self):
        xs = [1.0, 2.0, 3.0, 4.0]
        ys = [2.0, 4.0, 6.0, 8.0]
        assert pearson_correlation(xs, ys) == pytest.approx(1.0)

    def test_perfectly_anticorrelated_series_returns_minus_one(self):
        xs = [1.0, 2.0, 3.0, 4.0]
        ys = [-1.0, -2.0, -3.0, -4.0]
        assert pearson_correlation(xs, ys) == pytest.approx(-1.0)

    def test_zero_variance_series_returns_none(self):
        xs = [1.0, 1.0, 1.0]
        ys = [1.0, 2.0, 3.0]
        assert pearson_correlation(xs, ys) is None

    def test_too_few_points_returns_none(self):
        assert pearson_correlation([1.0], [1.0]) is None


class TestNIndependentCountsRebalancePeriodsNotPositions:
    def test_one_rebalance_with_ten_symbol_positions_counts_as_one_period(self):
        # A single rebalance period involves 2*N=10 individual symbol
        # positions (5 long + 5 short), but n_independent (built by the
        # caller from len(periods)) must count ONE period, not ten
        # positions. This test directly documents/checks that invariant
        # at the RebalancePeriod level: one call to build_rebalance_period
        # for a 10-symbol basket produces exactly one period object,
        # regardless of how many symbols it holds.
        histories = {f"SYM{i}USDT": _history({0.0: 100.0, 24.0: 100.0 + i}) for i in range(10)}
        basket = form_basket({f"SYM{i}USDT": float(i) for i in range(10)})
        assert basket is not None
        period = build_rebalance_period(
            _dt(0.0), _dt(24.0), {}, basket, None, histories, {}, DEFAULT_ROUND_TRIP_COST_PCT
        )
        assert period is not None
        assert len(basket) == 2 * BASKET_SIZE
        # The unit of observation is this ONE period object, not its 10
        # underlying symbol positions.
        assert isinstance(period.net_return_pct, float)
