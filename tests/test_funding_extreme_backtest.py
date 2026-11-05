"""Unit tests for the funding-extreme-reversion backtest
(``scanner.backtesting.funding_extreme_backtest``) -- see
``PREREGISTRATION_FUNDING_EXTREME_REVERSION.md`` for the frozen spec these
tests check conformance to. No network calls: signals/trades are built
directly from synthetic ``FundingEvent``/``FundingSignal`` lists and a
small hand-built candle DataFrame.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from scanner.backtesting.funding_extreme_backtest import (
    FUNDING_WINDOW_MIN_OBSERVATIONS,
    FundingSignal,
    Trade,
    apply_cooldown,
    build_trade,
    dedupe_correlated_signals,
    filter_signals_to_tradable_range,
    generate_raw_signals,
    rolling_percentile,
)
from scanner.backtesting.funding_simulator import FundingEvent
from scanner.calibration import DEFAULT_ROUND_TRIP_COST_PCT

pytestmark = pytest.mark.unit

_T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _dt(hours: float) -> datetime:
    return _T0 + timedelta(hours=hours)


def _funding_events(rates: list[float], start_hour: float = 0.0) -> list[FundingEvent]:
    """Evenly-spaced 8h-apart funding settlements, oldest first."""
    return [FundingEvent(timestamp=_dt(start_hour + 8 * i), rate=r) for i, r in enumerate(rates)]


def _signal(symbol: str, direction: str, entry_hour: float, percentile: float = 96.0) -> FundingSignal:
    return FundingSignal(
        symbol=symbol, direction=direction, entry_time=_dt(entry_hour), percentile=percentile, funding_rate=0.001
    )


class TestRollingPercentile:
    def test_empty_window_returns_neutral_midpoint(self):
        assert rolling_percentile([], 0.5) == 50.0

    def test_value_above_all_prior_is_100th_percentile(self):
        assert rolling_percentile([0.1, 0.2, 0.3], 1.0) == 100.0

    def test_value_below_all_prior_is_0th_percentile(self):
        assert rolling_percentile([0.1, 0.2, 0.3], -1.0) == 0.0

    def test_value_in_the_middle(self):
        # 2 of 4 window values are strictly less than 0.25.
        assert rolling_percentile([0.1, 0.2, 0.3, 0.4], 0.25) == 50.0

    def test_ties_do_not_count_as_less(self):
        # strict "<" convention: a value tied with every window member is
        # the 0th percentile, not the 100th.
        assert rolling_percentile([0.5, 0.5, 0.5], 0.5) == 0.0


class TestGenerateRawSignals:
    def test_no_signal_before_minimum_observations(self):
        # 89 strictly-prior observations is one short of the frozen
        # 90-observation minimum -- the 90th event (index 89) must not
        # emit a signal even though its rate is an extreme outlier.
        rates = [0.0001] * (FUNDING_WINDOW_MIN_OBSERVATIONS - 1) + [10.0]
        events = _funding_events(rates)
        signals = generate_raw_signals("BTCUSDT", events)
        assert signals == []

    def test_signal_emitted_at_exactly_minimum_observations(self):
        # The 91st event (index 90) has exactly 90 strictly-prior
        # observations behind it -- the frozen minimum -- and should now
        # be eligible to emit a signal.
        rates = [0.0001] * FUNDING_WINDOW_MIN_OBSERVATIONS + [10.0]
        events = _funding_events(rates)
        signals = generate_raw_signals("BTCUSDT", events)
        assert len(signals) == 1
        assert signals[0].direction == "short"
        assert signals[0].entry_time == events[-1].timestamp

    def test_no_look_ahead_current_value_excluded_from_its_own_window(self):
        # If the current (extreme) rate were included in its own
        # comparison window, it could never score the 100th percentile
        # (it would tie with itself). Confirm it does.
        rates = [0.0001] * FUNDING_WINDOW_MIN_OBSERVATIONS + [10.0]
        events = _funding_events(rates)
        signals = generate_raw_signals("BTCUSDT", events)
        assert signals[0].percentile == 100.0

    def test_low_extreme_emits_long_signal(self):
        rates = [0.0001] * FUNDING_WINDOW_MIN_OBSERVATIONS + [-10.0]
        events = _funding_events(rates)
        signals = generate_raw_signals("BTCUSDT", events)
        assert len(signals) == 1
        assert signals[0].direction == "long"

    def test_window_caps_at_270_observations(self):
        # A single far-outlier (-1000.0) is the very first observation,
        # followed by exactly 270 zeros. By the time the 270th zero is
        # appended, the window (maxlen=270) is forced to evict the
        # original -1000.0 outlier -- it is strictly older than the 270
        # most recent observations. A final test event (-500.0) then
        # lands strictly between -1000.0 and 0.0: if the window correctly
        # holds only the most recent 270 observations (270 zeros, no
        # -1000.0), its percentile is EXACTLY 0.0 (nothing in the window
        # is less than -500.0). If the cap were broken (e.g. an
        # unbounded list), -1000.0 would still be present and the
        # percentile would instead be a small but nonzero value
        # (~0.37%), which this test rules out.
        events = (
            [FundingEvent(timestamp=_dt(0.0), rate=-1000.0)]
            + [FundingEvent(timestamp=_dt(8.0 * (i + 1)), rate=0.0) for i in range(270)]
            + [FundingEvent(timestamp=_dt(8.0 * 271), rate=-500.0)]
        )
        signals = generate_raw_signals("BTCUSDT", events)
        last_signal = signals[-1]
        assert last_signal.entry_time == events[-1].timestamp
        assert last_signal.percentile == 0.0


class TestCooldown:
    def test_blocks_entry_within_16h_of_previous_same_symbol_direction(self):
        signals = [_signal("BTCUSDT", "short", 0.0), _signal("BTCUSDT", "short", 15.9)]
        kept = apply_cooldown(signals)
        assert len(kept) == 1
        assert kept[0].entry_time == _dt(0.0)

    def test_allows_entry_at_exactly_16h(self):
        signals = [_signal("BTCUSDT", "short", 0.0), _signal("BTCUSDT", "short", 16.0)]
        kept = apply_cooldown(signals)
        assert [s.entry_time for s in kept] == [_dt(0.0), _dt(16.0)]

    def test_allows_entry_after_16h(self):
        signals = [_signal("BTCUSDT", "short", 0.0), _signal("BTCUSDT", "short", 20.0)]
        kept = apply_cooldown(signals)
        assert len(kept) == 2

    def test_cooldown_is_independent_per_symbol_direction_key(self):
        # Same symbol, opposite direction: not subject to the same-key
        # cooldown at all.
        signals = [_signal("BTCUSDT", "short", 0.0), _signal("BTCUSDT", "long", 1.0)]
        kept = apply_cooldown(signals)
        assert len(kept) == 2

    def test_cooldown_applies_sequentially_not_just_to_first(self):
        # Third entry measured against the second KEPT entry's time, not
        # the first.
        signals = [
            _signal("BTCUSDT", "short", 0.0),
            _signal("BTCUSDT", "short", 16.0),
            _signal("BTCUSDT", "short", 30.0),  # only 14h after the 2nd kept entry
        ]
        kept = apply_cooldown(signals)
        assert [s.entry_time for s in kept] == [_dt(0.0), _dt(16.0)]


class TestCrossSymbolDedup:
    def test_overlapping_entries_same_direction_different_symbols_collapse(self):
        # Both entries' [entry_time, entry_time+8h] windows overlap.
        signals = [
            _signal("BTCUSDT", "short", 0.0, percentile=96.0),
            _signal("ETHUSDT", "short", 2.0, percentile=99.0),
        ]
        kept, n_removed = dedupe_correlated_signals(signals)
        assert n_removed == 1
        assert len(kept) == 1
        # Most-extreme-percentile representative wins.
        assert kept[0].symbol == "ETHUSDT"

    def test_non_overlapping_entries_stay_separate(self):
        signals = [
            _signal("BTCUSDT", "short", 0.0),
            _signal("ETHUSDT", "short", 20.0),  # 20h later, well past the 8h window
        ]
        kept, n_removed = dedupe_correlated_signals(signals)
        assert n_removed == 0
        assert len(kept) == 2

    def test_different_directions_never_collapse_even_if_overlapping_in_time(self):
        signals = [_signal("BTCUSDT", "short", 0.0), _signal("ETHUSDT", "long", 0.0)]
        kept, n_removed = dedupe_correlated_signals(signals)
        assert n_removed == 0
        assert len(kept) == 2

    def test_transitive_merge_across_three_entries(self):
        # A (0-8h) overlaps B (6-14h); B overlaps C (12-20h); A and C do
        # not directly overlap (0-8h vs 12-20h) but must still land in one
        # cluster via B.
        signals = [
            _signal("BTCUSDT", "short", 0.0, percentile=96.0),
            _signal("ETHUSDT", "short", 6.0, percentile=97.0),
            _signal("SOLUSDT", "short", 12.0, percentile=99.5),
        ]
        kept, n_removed = dedupe_correlated_signals(signals)
        assert n_removed == 2
        assert len(kept) == 1
        assert kept[0].symbol == "SOLUSDT"


class TestFilterSignalsToTradableRange:
    def test_drops_signal_entirely_before_candle_history(self):
        signals = [_signal("BTCUSDT", "short", -100.0)]
        kept = filter_signals_to_tradable_range(signals, _dt(0.0), _dt(1000.0))
        assert kept == []

    def test_drops_signal_whose_exit_is_after_candle_history_ends(self):
        signals = [_signal("BTCUSDT", "short", 998.0)]  # exit at 1006h > 1000h max
        kept = filter_signals_to_tradable_range(signals, _dt(0.0), _dt(1000.0))
        assert kept == []

    def test_keeps_signal_fully_within_range(self):
        signals = [_signal("BTCUSDT", "short", 500.0)]
        kept = filter_signals_to_tradable_range(signals, _dt(0.0), _dt(1000.0))
        assert len(kept) == 1


def _candle_history(prices: dict[float, float]) -> pd.DataFrame:
    """Builds a minimal 1h-candle-shaped DataFrame from {hour_offset: close}."""
    rows = [
        {
            "open_time": pd.Timestamp(_dt(hour)),
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": 1.0,
        }
        for hour, price in sorted(prices.items())
    ]
    return pd.DataFrame(rows)


class TestBuildTrade:
    def test_long_gross_return_matches_price_move(self):
        signal = _signal("BTCUSDT", "long", 0.0)
        history = _candle_history({0.0: 100.0, 8.0: 110.0})
        trade = build_trade(signal, history, [], base_cost_pct=0.0)
        assert trade is not None
        assert trade.gross_return_pct == pytest.approx(10.0)

    def test_short_gross_return_is_mirrored(self):
        signal = _signal("BTCUSDT", "short", 0.0)
        history = _candle_history({0.0: 100.0, 8.0: 110.0})
        trade = build_trade(signal, history, [], base_cost_pct=0.0)
        assert trade is not None
        assert trade.gross_return_pct == pytest.approx(-10.0)

    def test_fee_cost_reduces_net_return_vs_gross(self):
        signal = _signal("BTCUSDT", "long", 0.0)
        history = _candle_history({0.0: 100.0, 8.0: 110.0})
        trade = build_trade(signal, history, [], base_cost_pct=DEFAULT_ROUND_TRIP_COST_PCT)
        assert trade is not None
        assert trade.net_return_pct == pytest.approx(trade.gross_return_pct - trade.cost_pct)
        assert trade.net_return_pct < trade.gross_return_pct

    def test_funding_cost_reduces_net_return_for_long_paying_positive_funding(self):
        signal = _signal("BTCUSDT", "long", 0.0)
        history = _candle_history({0.0: 100.0, 8.0: 110.0})
        funding_events = [FundingEvent(timestamp=_dt(4.0), rate=0.001)]  # settles mid-hold
        trade = build_trade(signal, history, funding_events, base_cost_pct=0.0)
        assert trade is not None
        assert trade.funding_cost_pct == pytest.approx(0.1)
        assert trade.net_return_pct == pytest.approx(trade.gross_return_pct - 0.1)

    def test_returns_none_when_price_unavailable(self):
        signal = _signal("BTCUSDT", "long", -50.0)  # entry before any candle exists
        history = _candle_history({0.0: 100.0, 8.0: 110.0})
        assert build_trade(signal, history, [], base_cost_pct=0.0) is None


def test_trade_dataclass_fields_roundtrip():
    # Sanity check that Trade is a plain, inspectable dataclass (used by
    # the report's JSON serialization).
    trade = Trade(
        symbol="BTCUSDT",
        direction="long",
        entry_time=_dt(0.0),
        exit_time=_dt(8.0),
        percentile=99.0,
        entry_price=100.0,
        exit_price=105.0,
        gross_return_pct=5.0,
        cost_pct=0.1,
        funding_cost_pct=0.0,
        net_return_pct=4.9,
    )
    assert trade.net_return_pct == pytest.approx(trade.gross_return_pct - trade.cost_pct - trade.funding_cost_pct)
