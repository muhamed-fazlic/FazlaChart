"""Unit tests for live forward-outcome tracking (scanner/signal_tracking.py).

Uses a real on-disk sqlite db (via scanner.storage.init_db) so the dedup /
resolve queries are exercised against actual SQL, but never touches Binance:
calibration lookups are monkeypatched, same pattern as test_webapp.py.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from scanner.calibration import CalibrationStats
from scanner.signal_tracking import _net_return_pct, log_new_signals, resolve_open_signals
from scanner.storage import fetch_recent_signal_outcomes, init_db, save_scan_result

pytestmark = pytest.mark.unit


def _snapshot(symbol: str, trend: str, close: float, atr: float = 0.2, timeframe: str = "1h") -> dict:
    """Single-timeframe snapshot for the common-case tests below. For
    multi-timeframe scenarios see ``_multi_tf_snapshot``."""
    return {
        "symbol": symbol,
        "timeframes": {
            timeframe: {
                "symbol": symbol,
                "timeframe": timeframe,
                "trend": trend,
                "close": close,
                "atr": atr,
                "rsi": 45.0,
                "ema20": close,
                "ema50": close,
                "anomalies": [],
            },
        },
        "derivatives": {},
    }


def _multi_tf_snapshot(symbol: str, closes_by_tf: dict[str, float], trend: str = "bearish", atr: float = 0.2) -> dict:
    """A snapshot carrying several timeframes at once, each with its own
    close price -- used to prove a tracked signal resolves against its own
    timeframe's close, not whichever timeframe happens to be first/default."""
    return {
        "symbol": symbol,
        "timeframes": {
            tf: {
                "symbol": symbol,
                "timeframe": tf,
                "trend": trend,
                "close": close,
                "atr": atr,
                "rsi": 45.0,
                "ema20": close,
                "ema50": close,
                "anomalies": [],
            }
            for tf, close in closes_by_tf.items()
        },
        "derivatives": {},
    }


_STRONG_SHORT_MATCH = [
    CalibrationStats(
        label="trend:bearish",
        direction="short",
        horizon_candles=12,
        n=387,
        win_rate_pct=65.0,
        avg_net_return_pct=0.8,
        avg_win_return_pct=1.5,
        avg_loss_return_pct=-0.9,
        win_rate_ci_low_pct=60.1,
        win_rate_ci_high_pct=69.8,
    ),
]


@pytest.fixture
def conn(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "agents_crypto.calibration_lookup.get_cached_calibration_stats", lambda *a, **k: _STRONG_SHORT_MATCH
    )
    db_path = tmp_path / "tracking_test.db"
    connection = init_db(str(db_path))
    yield connection
    connection.close()


def _rows(conn):
    from scanner.storage import fetch_latest_scan_per_symbol

    return fetch_latest_scan_per_symbol(conn)


class TestLogNewSignals:
    def test_logs_a_qualifying_candidate(self, conn):
        save_scan_result(conn, "INJUSDT", _snapshot("INJUSDT", "bearish", 10.0), [])
        logged = log_new_signals(conn, _rows(conn))
        assert logged == 1
        outcomes = fetch_recent_signal_outcomes(conn)
        assert len(outcomes) == 1
        assert outcomes[0]["symbol"] == "INJUSDT"
        assert outcomes[0]["direction"] == "short"
        assert outcomes[0]["status"] == "open"
        assert outcomes[0]["promised_win_rate_ci_low_pct"] == 60.1

    def test_persists_pattern_quality_score_and_regime_fallback_from_the_candidate(self, conn):
        # Analytics-breakdown fields (Phase 3/4/5) must flow from the live
        # Top Signal Now candidate all the way into the tracked DB row, not
        # just exist on the dashboard -- that's what makes the by-pattern/
        # by-regime/by-quality forward-tracking breakdowns meaningful.
        save_scan_result(conn, "INJUSDT", _snapshot("INJUSDT", "bearish", 10.0), [])
        assert log_new_signals(conn, _rows(conn)) == 1
        outcome = fetch_recent_signal_outcomes(conn)[0]
        assert outcome["pattern"] in (None, "trending", "breakout", "reversal")
        assert outcome["quality_score"] is None or isinstance(outcome["quality_score"], float)
        assert outcome["regime_fallback"] in (None, 0, 1)

    def test_does_not_duplicate_an_already_open_signal(self, conn):
        save_scan_result(conn, "INJUSDT", _snapshot("INJUSDT", "bearish", 10.0), [])
        rows = _rows(conn)
        assert log_new_signals(conn, rows) == 1
        assert log_new_signals(conn, rows) == 0  # still open -- not logged again
        assert len(fetch_recent_signal_outcomes(conn)) == 1

    def test_skips_candidate_with_no_computable_entry(self, conn):
        # No "atr" field at all -> compute_risk_levels can't run -> entry is None.
        snapshot = _snapshot("INJUSDT", "bearish", 10.0)
        del snapshot["timeframes"]["1h"]["atr"]
        save_scan_result(conn, "INJUSDT", snapshot, [])
        assert log_new_signals(conn, _rows(conn)) == 0
        assert fetch_recent_signal_outcomes(conn) == []

    def test_on_new_signal_callback_fires_once_per_new_candidate(self, conn):
        save_scan_result(conn, "INJUSDT", _snapshot("INJUSDT", "bearish", 10.0), [])
        seen = []
        rows = _rows(conn)
        assert log_new_signals(conn, rows, on_new_signal=seen.append) == 1
        assert len(seen) == 1
        assert seen[0]["symbol"] == "INJUSDT"

        # Still open on the next cycle -- the callback must not fire again.
        assert log_new_signals(conn, rows, on_new_signal=seen.append) == 0
        assert len(seen) == 1

    def test_on_new_signal_callback_failure_does_not_block_logging(self, conn):
        def _boom(_candidate):
            raise RuntimeError("telegram is down")

        save_scan_result(conn, "INJUSDT", _snapshot("INJUSDT", "bearish", 10.0), [])
        logged = log_new_signals(conn, _rows(conn), on_new_signal=_boom)
        assert logged == 1
        assert len(fetch_recent_signal_outcomes(conn)) == 1


class TestResolveOpenSignals:
    def test_resolves_take_profit_when_price_reaches_target(self, conn):
        # 15m is single-target (not in MULTI_TARGET_TIMEFRAMES) -- the
        # multi-target 1h/4h partial-close path is covered separately in
        # TestMultiTargetResolution below.
        save_scan_result(conn, "INJUSDT", _snapshot("INJUSDT", "bearish", 10.0, timeframe="15m"), [])
        log_new_signals(conn, _rows(conn))
        outcome = fetch_recent_signal_outcomes(conn)[0]
        take_profit = outcome["take_profit"]

        save_scan_result(conn, "INJUSDT", _snapshot("INJUSDT", "bearish", take_profit - 0.01, timeframe="15m"), [])
        resolved = resolve_open_signals(conn, _rows(conn))

        assert resolved == 1
        updated = fetch_recent_signal_outcomes(conn)[0]
        assert updated["status"] == "take_profit"
        assert updated["realized_return_pct"] > 0  # a short that hit its target nets a profit, net of cost

    def test_resolves_a_4h_signal_using_the_4h_close_not_the_1h_close(self, conn):
        # Both timeframes qualify (same mocked calibration match for any
        # timeframe) -- only the 4h close moves here, proving resolution
        # looks up the outcome's own timeframe's close, not the 1h one.
        # 4h is multi-target (see MULTI_TARGET_TIMEFRAMES) so driving the
        # close straight past TP3 is what fully resolves it in one step.
        save_scan_result(conn, "INJUSDT", _multi_tf_snapshot("INJUSDT", {"1h": 10.0, "4h": 10.0}), [])
        log_new_signals(conn, _rows(conn))
        outcomes = {o["timeframe"]: o for o in fetch_recent_signal_outcomes(conn)}
        assert set(outcomes) == {"1h", "4h"}
        take_profit_extended_4h = outcomes["4h"]["take_profit_extended"]

        save_scan_result(
            conn, "INJUSDT", _multi_tf_snapshot("INJUSDT", {"1h": 10.0, "4h": take_profit_extended_4h - 0.01}), []
        )
        resolved = resolve_open_signals(conn, _rows(conn))

        assert resolved == 1
        updated = {o["timeframe"]: o for o in fetch_recent_signal_outcomes(conn)}
        assert updated["4h"]["status"] == "full_run"
        assert updated["1h"]["status"] == "open"  # unchanged 1h close -- that signal must stay open

    def test_resolves_stop_loss_when_price_reaches_stop(self, conn):
        save_scan_result(conn, "INJUSDT", _snapshot("INJUSDT", "bearish", 10.0), [])
        log_new_signals(conn, _rows(conn))
        outcome = fetch_recent_signal_outcomes(conn)[0]
        stop_loss = outcome["stop_loss"]

        save_scan_result(conn, "INJUSDT", _snapshot("INJUSDT", "bearish", stop_loss + 0.01), [])
        resolved = resolve_open_signals(conn, _rows(conn))

        assert resolved == 1
        updated = fetch_recent_signal_outcomes(conn)[0]
        assert updated["status"] == "stop_loss"
        assert updated["realized_return_pct"] < 0  # a short whose stop got hit nets a loss

    def test_leaves_signal_open_when_neither_level_is_touched(self, conn):
        save_scan_result(conn, "INJUSDT", _snapshot("INJUSDT", "bearish", 10.0), [])
        log_new_signals(conn, _rows(conn))

        save_scan_result(conn, "INJUSDT", _snapshot("INJUSDT", "bearish", 10.0), [])  # unchanged price
        resolved = resolve_open_signals(conn, _rows(conn))

        assert resolved == 0
        assert fetch_recent_signal_outcomes(conn)[0]["status"] == "open"

    def test_times_out_after_horizon_elapses_without_a_hit(self, conn):
        save_scan_result(conn, "INJUSDT", _snapshot("INJUSDT", "bearish", 10.0), [])
        log_new_signals(conn, _rows(conn))

        # Backdate logged_at well past the 12-candle (12h, on a 1h timeframe) horizon.
        stale_time = (datetime.now(UTC) - timedelta(hours=20)).isoformat()
        conn.execute("UPDATE signal_outcomes SET logged_at = ?", (stale_time,))
        conn.commit()

        save_scan_result(conn, "INJUSDT", _snapshot("INJUSDT", "bearish", 10.0), [])  # still no SL/TP hit
        resolved = resolve_open_signals(conn, _rows(conn))

        assert resolved == 1
        assert fetch_recent_signal_outcomes(conn)[0]["status"] == "timeout"

    def test_skips_symbol_no_longer_present_this_cycle(self, conn):
        save_scan_result(conn, "INJUSDT", _snapshot("INJUSDT", "bearish", 10.0), [])
        log_new_signals(conn, _rows(conn))

        resolved = resolve_open_signals(conn, [])  # INJUSDT absent from this cycle's rows
        assert resolved == 0
        assert fetch_recent_signal_outcomes(conn)[0]["status"] == "open"


class TestMultiTargetResolution:
    """4h signals are managed through TP1/TP2/TP3 partial closes instead of
    one single exit (see signal_tracking.MULTI_TARGET_TIMEFRAMES) -- banking
    a target moves the stop and keeps the trade open; it only resolves once
    the (moved) stop is hit, all three targets bank, or the horizon times
    out."""

    def _log_4h_short(self, conn):
        save_scan_result(conn, "INJUSDT", _snapshot("INJUSDT", "bearish", 10.0, timeframe="4h"), [])
        log_new_signals(conn, _rows(conn))
        return fetch_recent_signal_outcomes(conn)[0]

    def test_logs_a_4h_signal_as_multi_target_with_an_extended_third_target(self, conn):
        outcome = self._log_4h_short(conn)
        assert outcome["multi_target"] == 1
        assert outcome["targets_hit"] == 0
        assert outcome["effective_stop_loss"] == outcome["stop_loss"]
        assert outcome["take_profit_extended"] is not None
        # TP3 (extended) should be further from entry than TP2 (take_profit)
        # for a short -- i.e. a lower price.
        assert outcome["take_profit_extended"] < outcome["take_profit"]

    def test_1h_signals_are_also_managed_as_multi_target(self, conn):
        # Any timeframe of 1h or longer gets TP1/TP2/TP3 -- not just 4h.
        save_scan_result(conn, "INJUSDT", _snapshot("INJUSDT", "bearish", 10.0, timeframe="1h"), [])
        log_new_signals(conn, _rows(conn))
        outcome = fetch_recent_signal_outcomes(conn)[0]
        assert outcome["multi_target"] == 1
        assert outcome["take_profit_extended"] is not None

    def test_banking_tp1_moves_the_stop_to_breakeven_and_keeps_the_trade_open(self, conn):
        outcome = self._log_4h_short(conn)
        tp1 = outcome["take_profit_conservative"]

        save_scan_result(conn, "INJUSDT", _snapshot("INJUSDT", "bearish", tp1 - 0.01, timeframe="4h"), [])
        resolved = resolve_open_signals(conn, _rows(conn))

        assert resolved == 0  # progress only -- not a final resolution
        updated = fetch_recent_signal_outcomes(conn)[0]
        assert updated["status"] == "open"
        assert updated["targets_hit"] == 1
        assert updated["effective_stop_loss"] == updated["entry_price"]

    def test_stopped_at_breakeven_after_tp1_resolves_as_tp1_be_with_a_net_positive_return(self, conn):
        outcome = self._log_4h_short(conn)
        tp1 = outcome["take_profit_conservative"]
        entry = outcome["entry_price"]

        save_scan_result(conn, "INJUSDT", _snapshot("INJUSDT", "bearish", tp1 - 0.01, timeframe="4h"), [])
        resolve_open_signals(conn, _rows(conn))  # banks TP1, moves stop to breakeven

        save_scan_result(conn, "INJUSDT", _snapshot("INJUSDT", "bearish", entry, timeframe="4h"), [])
        resolved = resolve_open_signals(conn, _rows(conn))

        assert resolved == 1
        updated = fetch_recent_signal_outcomes(conn)[0]
        assert updated["status"] == "tp1_be"
        # 40% banked a real profit at TP1, 60% closed flat at breakeven --
        # net should still be positive (not treated as a straight loss).
        assert updated["realized_return_pct"] > 0

    def test_banking_tp1_then_tp2_trails_the_stop_to_the_tp1_price(self, conn):
        outcome = self._log_4h_short(conn)
        tp1, tp2 = outcome["take_profit_conservative"], outcome["take_profit"]

        save_scan_result(conn, "INJUSDT", _snapshot("INJUSDT", "bearish", tp1 - 0.01, timeframe="4h"), [])
        resolve_open_signals(conn, _rows(conn))  # banks TP1

        save_scan_result(conn, "INJUSDT", _snapshot("INJUSDT", "bearish", tp2 - 0.01, timeframe="4h"), [])
        resolved = resolve_open_signals(conn, _rows(conn))

        assert resolved == 0
        updated = fetch_recent_signal_outcomes(conn)[0]
        assert updated["status"] == "open"
        assert updated["targets_hit"] == 2
        assert updated["effective_stop_loss"] == pytest.approx(tp1)

    def test_pullback_to_tp1_level_after_tp2_resolves_as_tp2_be(self, conn):
        outcome = self._log_4h_short(conn)
        tp1, tp2 = outcome["take_profit_conservative"], outcome["take_profit"]

        save_scan_result(conn, "INJUSDT", _snapshot("INJUSDT", "bearish", tp1 - 0.01, timeframe="4h"), [])
        resolve_open_signals(conn, _rows(conn))  # banks TP1
        save_scan_result(conn, "INJUSDT", _snapshot("INJUSDT", "bearish", tp2 - 0.01, timeframe="4h"), [])
        resolve_open_signals(conn, _rows(conn))  # banks TP2, trails stop to TP1

        save_scan_result(conn, "INJUSDT", _snapshot("INJUSDT", "bearish", tp1, timeframe="4h"), [])
        resolved = resolve_open_signals(conn, _rows(conn))

        assert resolved == 1
        updated = fetch_recent_signal_outcomes(conn)[0]
        assert updated["status"] == "tp2_be"
        assert updated["realized_return_pct"] > 0

    def test_a_single_cycle_jump_past_all_three_targets_resolves_as_full_run(self, conn):
        outcome = self._log_4h_short(conn)
        tp3 = outcome["take_profit_extended"]

        save_scan_result(conn, "INJUSDT", _snapshot("INJUSDT", "bearish", tp3 - 0.01, timeframe="4h"), [])
        resolved = resolve_open_signals(conn, _rows(conn))

        assert resolved == 1
        updated = fetch_recent_signal_outcomes(conn)[0]
        assert updated["status"] == "full_run"
        assert updated["realized_return_pct"] > 0

    def test_stop_loss_before_any_target_resolves_exactly_like_a_single_target_signal(self, conn):
        outcome = self._log_4h_short(conn)
        stop_loss = outcome["stop_loss"]

        save_scan_result(conn, "INJUSDT", _snapshot("INJUSDT", "bearish", stop_loss + 0.01, timeframe="4h"), [])
        resolved = resolve_open_signals(conn, _rows(conn))

        assert resolved == 1
        updated = fetch_recent_signal_outcomes(conn)[0]
        assert updated["status"] == "stop_loss"
        assert updated["realized_return_pct"] < 0

    def test_timeout_with_tp1_banked_marks_the_remainder_at_the_current_close(self, conn):
        outcome = self._log_4h_short(conn)
        tp1 = outcome["take_profit_conservative"]

        save_scan_result(conn, "INJUSDT", _snapshot("INJUSDT", "bearish", tp1 - 0.01, timeframe="4h"), [])
        resolve_open_signals(conn, _rows(conn))  # banks TP1

        stale_time = (datetime.now(UTC) - timedelta(hours=60)).isoformat()
        conn.execute("UPDATE signal_outcomes SET logged_at = ?", (stale_time,))
        conn.commit()

        save_scan_result(conn, "INJUSDT", _snapshot("INJUSDT", "bearish", tp1, timeframe="4h"), [])  # unchanged
        resolved = resolve_open_signals(conn, _rows(conn))

        assert resolved == 1
        updated = fetch_recent_signal_outcomes(conn)[0]
        assert updated["status"] == "timeout"
        assert updated["realized_return_pct"] > 0  # TP1's banked profit still counts toward the final return


class TestNetReturnPct:
    def test_long_profits_when_price_rises(self):
        assert _net_return_pct("long", 100.0, 110.0) == pytest.approx(10.0 - 0.10)

    def test_short_profits_when_price_falls(self):
        assert _net_return_pct("short", 100.0, 90.0) == pytest.approx(10.0 - 0.10)

    def test_long_loses_when_price_falls(self):
        assert _net_return_pct("long", 100.0, 95.0) == pytest.approx(-5.0 - 0.10)

