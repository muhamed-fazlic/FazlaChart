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
        save_scan_result(conn, "INJUSDT", _snapshot("INJUSDT", "bearish", 10.0), [])
        log_new_signals(conn, _rows(conn))
        outcome = fetch_recent_signal_outcomes(conn)[0]
        take_profit = outcome["take_profit"]

        save_scan_result(conn, "INJUSDT", _snapshot("INJUSDT", "bearish", take_profit - 0.01), [])
        resolved = resolve_open_signals(conn, _rows(conn))

        assert resolved == 1
        updated = fetch_recent_signal_outcomes(conn)[0]
        assert updated["status"] == "take_profit"
        assert updated["realized_return_pct"] > 0  # a short that hit its target nets a profit, net of cost

    def test_resolves_a_4h_signal_using_the_4h_close_not_the_1h_close(self, conn):
        # Both timeframes qualify (same mocked calibration match for any
        # timeframe) -- only the 4h close moves here, proving resolution
        # looks up the outcome's own timeframe's close, not the 1h one.
        save_scan_result(conn, "INJUSDT", _multi_tf_snapshot("INJUSDT", {"1h": 10.0, "4h": 10.0}), [])
        log_new_signals(conn, _rows(conn))
        outcomes = {o["timeframe"]: o for o in fetch_recent_signal_outcomes(conn)}
        assert set(outcomes) == {"1h", "4h"}
        take_profit_4h = outcomes["4h"]["take_profit"]

        save_scan_result(conn, "INJUSDT", _multi_tf_snapshot("INJUSDT", {"1h": 10.0, "4h": take_profit_4h - 0.01}), [])
        resolved = resolve_open_signals(conn, _rows(conn))

        assert resolved == 1
        updated = {o["timeframe"]: o for o in fetch_recent_signal_outcomes(conn)}
        assert updated["4h"]["status"] == "take_profit"
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


class TestNetReturnPct:
    def test_long_profits_when_price_rises(self):
        assert _net_return_pct("long", 100.0, 110.0) == pytest.approx(10.0 - 0.10)

    def test_short_profits_when_price_falls(self):
        assert _net_return_pct("short", 100.0, 90.0) == pytest.approx(10.0 - 0.10)

    def test_long_loses_when_price_falls(self):
        assert _net_return_pct("long", 100.0, 95.0) == pytest.approx(-5.0 - 0.10)
