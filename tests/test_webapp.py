"""Unit tests for the webapp API — points SCANNER_STORAGE_PATH at a throwaway
SQLite database seeded directly via scanner.storage, so these tests never
touch Binance or the real scanner_data/ directory."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from scanner.storage import init_db, save_scan_result

pytestmark = pytest.mark.integration
# fastapi.testclient.TestClient spins up a real asyncio event loop; on Windows
# the ProactorEventLoop's internal self-pipe wakeup mechanism opens a genuine
# loopback socket, which the suite's global `_no_network` fixture (by design)
# refuses for anything not marked `integration`. No external network or
# Binance/LLM call happens here — it's a same-machine ASGI test harness detail.

_BTC_SNAPSHOT = {
    "symbol": "BTCUSDT",
    "timeframes": {
        "1h": {
            "symbol": "BTCUSDT",
            "timeframe": "1h",
            "trend": "bullish",
            "close": 50000.0,
            "rsi": 60.0,
            "ema20": 49500.0,
            "ema50": 49000.0,
            "volume_change_pct": 12.0,
            "relative_volume": 1.1,
            "support": 48000.0,
            "resistance": 51000.0,
            "momentum_fading": False,
            "anomalies": [],
        },
    },
    "derivatives": {
        "funding_rate": 0.0001,
        "open_interest_change_pct_60m": 2.5,
    },
}

_ALT_SNAPSHOT = {
    "symbol": "INJUSDT",
    "timeframes": {
        "1h": {
            "symbol": "INJUSDT",
            "timeframe": "1h",
            "trend": "bearish",
            "close": 10.0,
            "rsi": 35.0,
            "ema20": 10.2,
            "ema50": 10.5,
            "volume_change_pct": -20.0,
            "relative_volume": 0.5,
            "support": 9.5,
            "resistance": 10.8,
            "momentum_fading": True,
            "anomalies": ["failed_breakout"],
        },
    },
    "derivatives": {
        "funding_rate": -0.0002,
        "open_interest_change_pct_60m": -1.0,
    },
}


@pytest.fixture
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "scanner_test.db"
    monkeypatch.setenv("SCANNER_STORAGE_PATH", str(db_path))

    conn = init_db(str(db_path))
    save_scan_result(conn, "BTCUSDT", _BTC_SNAPSHOT, [])
    save_scan_result(conn, "INJUSDT", _ALT_SNAPSHOT, ["1h:failed_breakout"])
    conn.close()

    from webapp.main import app

    return TestClient(app)


def test_health_reports_storage_path(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["storage_exists"] is True
    assert body["status"] == "healthy"
    assert body["staleness_seconds"] is not None
    assert body["symbols_ever_tracked"] == 2


def test_health_reports_stale_when_last_scan_is_old(tmp_path, monkeypatch):
    import sqlite3
    from datetime import UTC, datetime, timedelta

    db_path = tmp_path / "scanner_test_stale.db"
    monkeypatch.setenv("SCANNER_STORAGE_PATH", str(db_path))
    monkeypatch.setenv("SCANNER_HEALTH_MAX_STALENESS_SECONDS", "60")

    conn = init_db(str(db_path))
    save_scan_result(conn, "BTCUSDT", _BTC_SNAPSHOT, [])
    stale_time = (datetime.now(UTC) - timedelta(seconds=600)).isoformat()
    conn.execute("UPDATE scans SET scanned_at = ?", (stale_time,))
    conn.commit()
    conn.close()

    from webapp.main import app

    resp = TestClient(app).get("/api/health")
    body = resp.json()
    assert body["ok"] is False
    assert body["status"] == "stale"
    assert body["staleness_seconds"] > 60


def test_health_reports_no_data_for_empty_database(tmp_path, monkeypatch):
    db_path = tmp_path / "scanner_test_empty.db"
    monkeypatch.setenv("SCANNER_STORAGE_PATH", str(db_path))
    init_db(str(db_path)).close()

    from webapp.main import app

    resp = TestClient(app).get("/api/health")
    body = resp.json()
    assert body["status"] == "no_data"
    assert body["ok"] is False


def test_overview_returns_btc_summary_and_symbols(client):
    resp = client.get("/api/overview")
    assert resp.status_code == 200
    body = resp.json()

    assert body["btc"]["symbol"] == "BTCUSDT"
    assert body["btc"]["trend_by_timeframe"]["1h"] == "bullish"

    symbols_by_name = {s["symbol"]: s for s in body["symbols"]}
    assert set(symbols_by_name) == {"BTCUSDT", "INJUSDT"}
    assert symbols_by_name["INJUSDT"]["anomalies"] == ["1h:failed_breakout"]
    assert symbols_by_name["INJUSDT"]["trend"] == "bearish"


def test_overview_sorts_anomalies_first(client):
    resp = client.get("/api/overview")
    symbols = resp.json()["symbols"]
    # INJUSDT (has anomalies) should be ordered before BTCUSDT (none) given
    # the sort key (has_no_anomalies, symbol).
    assert symbols[0]["symbol"] == "INJUSDT"


def test_symbol_history_returns_404_for_unknown_symbol(client):
    resp = client.get("/api/symbol/DOESNOTEXISTUSDT/history")
    assert resp.status_code == 404


def test_symbol_history_returns_points(client):
    resp = client.get("/api/symbol/BTCUSDT/history")
    assert resp.status_code == 200
    body = resp.json()
    assert body["symbol"] == "BTCUSDT"
    assert len(body["history"]) == 1
    assert body["history"][0]["close_price"] == 50000.0


def test_index_serves_dashboard_html(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "FazlaChart" in resp.text


def test_analyze_endpoint_returns_structured_payload(client, monkeypatch):
    from agents_crypto.derivatives_risk_agent import AssessmentResult, DerivativesRiskAssessment

    fake_result = AssessmentResult(
        assessment=DerivativesRiskAssessment(
            symbol="INJUSDT",
            bias="short",
            historical_basis="trend:bearish, n=387, win_rate=37.0%",
            key_observations=["RSI below 40"],
            risk_factors=["Low sample size"],
            risk_reward_commentary="R:R is 1.67.",
            caveats="Doesn't account for news.",
        ),
        risk_levels={"entry": 10.0, "stop_loss": 10.3, "take_profit": 9.5, "risk_pct": 3.0, "reward_pct": 5.0, "risk_reward_ratio": 1.67},
        multi_timeframe_risk_levels={},
        calibration_matches=[{"label": "trend:bearish", "direction": "short", "n": 387, "win_rate_pct": 37.0, "avg_net_return_pct": -0.17}],
        diverges_from_btc=True,
        liquidity_heatmap={},
    )

    captured = {}

    def _fake_assess_symbol(symbol, timeframe="1h", horizon_candles=12, provider=None, model=None, **_kwargs):
        assert symbol == "INJUSDT"
        captured["provider"] = provider
        captured["model"] = model
        return fake_result

    monkeypatch.setattr("agents_crypto.derivatives_risk_agent.assess_symbol", _fake_assess_symbol)

    resp = client.post("/api/analyze/injusdt?timeframe=1h&provider=groq&model=openai%2Fgpt-oss-120b")
    assert resp.status_code == 200
    body = resp.json()
    assert body["symbol"] == "INJUSDT"
    assert body["assessment"]["bias"] == "short"
    assert body["diverges_from_btc"] is True
    assert body["risk_levels"]["take_profit"] == 9.5
    assert captured["provider"] == "groq"
    assert captured["model"] == "openai/gpt-oss-120b"


def test_calibration_endpoint_returns_matches_without_llm(client, monkeypatch):
    from scanner.calibration import CalibrationStats

    fake_stats = [
        CalibrationStats(
            label="trend:bearish",
            direction="short",
            horizon_candles=12,
            n=387,
            win_rate_pct=37.0,
            avg_net_return_pct=-0.173,
            avg_win_return_pct=0.637,
            avg_loss_return_pct=-0.647,
        ),
        CalibrationStats(
            label="anomaly:failed_breakout",
            direction="short",
            horizon_candles=12,
            n=41,
            win_rate_pct=34.1,
            avg_net_return_pct=-1.283,
            avg_win_return_pct=0.953,
            avg_loss_return_pct=-2.442,
        ),
    ]

    captured = {}

    def _fake_get_calibration_stats(symbol, timeframe, horizons=(12,), **_kwargs):
        captured["symbol"] = symbol
        captured["timeframe"] = timeframe
        captured["horizons"] = horizons
        return fake_stats

    monkeypatch.setattr("agents_crypto.calibration_lookup.get_calibration_stats", _fake_get_calibration_stats)
    monkeypatch.setattr("agents_crypto.calibration_lookup.get_oos_validation", lambda *a, **k: [])

    resp = client.get("/api/symbol/injusdt/calibration")
    assert resp.status_code == 200
    body = resp.json()
    assert body["symbol"] == "INJUSDT"
    assert body["active_labels"] == ["trend:bearish", "anomaly:failed_breakout"]
    assert len(body["matches"]) == 2
    assert body["matches"][0]["win_rate_pct"] == 37.0
    assert body["out_of_sample"] == []
    assert captured["symbol"] == "INJUSDT"
    assert captured["timeframe"] == "1h"


def test_calibration_endpoint_includes_out_of_sample_validation(client, monkeypatch):
    from scanner.calibration import CalibrationStats, OutOfSampleResult

    monkeypatch.setattr(
        "agents_crypto.calibration_lookup.get_calibration_stats",
        lambda *a, **k: [
            CalibrationStats(
                label="trend:bearish",
                direction="short",
                horizon_candles=12,
                n=100,
                win_rate_pct=55.0,
                avg_net_return_pct=0.3,
                avg_win_return_pct=0.8,
                avg_loss_return_pct=-0.5,
            )
        ],
    )
    monkeypatch.setattr(
        "agents_crypto.calibration_lookup.get_oos_validation",
        lambda *a, **k: [
            OutOfSampleResult(
                label="trend:bearish",
                direction="short",
                horizon_candles=12,
                train_n=70,
                train_win_rate_pct=60.0,
                test_n=30,
                test_win_rate_pct=40.0,
                test_win_rate_ci_low_pct=25.0,
                holds_up=False,
                warning="Edge se NE potvrđuje na najnovijem periodu.",
            )
        ],
    )

    resp = client.get("/api/symbol/injusdt/calibration")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["out_of_sample"]) == 1
    assert body["out_of_sample"][0]["holds_up"] is False
    assert body["out_of_sample"][0]["test_win_rate_ci_low_pct"] == 25.0


def test_calibration_endpoint_returns_404_for_unknown_symbol(client):
    resp = client.get("/api/symbol/DOESNOTEXISTUSDT/calibration")
    assert resp.status_code == 404


def test_analyze_endpoint_returns_502_on_failure(client, monkeypatch):
    def _raise(*_args, **_kwargs):
        raise RuntimeError("Ollama is not reachable")

    monkeypatch.setattr("agents_crypto.derivatives_risk_agent.assess_symbol", _raise)

    resp = client.post("/api/analyze/INJUSDT")
    assert resp.status_code == 502
    assert "Ollama is not reachable" in resp.json()["detail"]


def test_best_signal_requires_statistically_significant_edge(client, monkeypatch):
    """A match below the sample-size floor or whose 95% CI lower bound
    doesn't clear 50% must never surface as a "best signal", no matter how
    high its raw win_rate_pct looks."""
    from scanner.calibration import CalibrationStats

    weak_match = [
        CalibrationStats(
            label="trend:bearish",
            direction="short",
            horizon_candles=12,
            n=10,  # below the minimum sample size
            win_rate_pct=80.0,
            avg_net_return_pct=1.0,
            avg_win_return_pct=2.0,
            avg_loss_return_pct=-1.0,
            win_rate_ci_low_pct=44.0,  # straddles 50% anyway
            win_rate_ci_high_pct=97.0,
        ),
    ]
    monkeypatch.setattr(
        "agents_crypto.calibration_lookup.get_cached_calibration_stats", lambda *a, **k: weak_match
    )

    resp = client.get("/api/scan/best-signal")
    assert resp.status_code == 200
    body = resp.json()
    assert body["signals"] == []
    assert body["candidates_considered"] == 2  # both BTCUSDT (bullish) and INJUSDT (bearish) are directional


def test_best_signal_surfaces_confirmed_edge_with_entry_zone(client, monkeypatch):
    from scanner.calibration import CalibrationStats

    strong_match = [
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
    monkeypatch.setattr(
        "agents_crypto.calibration_lookup.get_cached_calibration_stats", lambda *a, **k: strong_match
    )

    resp = client.get("/api/scan/best-signal")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["signals"]) == 1
    signal = body["signals"][0]
    assert signal["symbol"] == "INJUSDT"
    assert signal["direction"] == "short"
    assert signal["win_rate_ci_low_pct"] == 60.1
    # INJUSDT's fixture snapshot has no "atr" field, so no entry zone can be
    # computed — confirms the endpoint degrades gracefully instead of crashing.
    assert signal["entry"] is None


def test_best_signal_recenters_entry_zone_on_patient_entry_when_overextended(tmp_path, monkeypatch):
    """A short on a symbol whose price has already run far below EMA20 with
    deeply oversold RSI must not hand back an entry zone glued to the
    current (already-extended) price — it should recenter on the suggested
    patient/retracement entry instead, so the user isn't told to short right
    at the bottom of a move that's statistically due for a bounce."""
    from scanner.calibration import CalibrationStats

    db_path = tmp_path / "scanner_test.db"
    monkeypatch.setenv("SCANNER_STORAGE_PATH", str(db_path))

    extended_snapshot = {
        "symbol": "EXTUSDT",
        "timeframes": {
            "1h": {
                "symbol": "EXTUSDT",
                "timeframe": "1h",
                "trend": "bearish",
                "close": 90.0,
                "atr": 2.0,
                "rsi": 25.0,
                "ema20": 100.0,
                "ema50": 105.0,
                "volume_change_pct": -10.0,
                "relative_volume": 0.8,
                "support": 85.0,
                "resistance": 105.0,
                "momentum_fading": False,
                "anomalies": [],
            },
        },
        "derivatives": {"funding_rate": -0.0001, "open_interest_change_pct_60m": -1.0},
    }

    conn = init_db(str(db_path))
    save_scan_result(conn, "EXTUSDT", extended_snapshot, [])
    conn.close()

    from webapp.main import app

    strong_match = [
        CalibrationStats(
            label="trend:bearish",
            direction="short",
            horizon_candles=12,
            n=200,
            win_rate_pct=60.0,
            avg_net_return_pct=0.5,
            avg_win_return_pct=1.0,
            avg_loss_return_pct=-0.8,
            win_rate_ci_low_pct=53.0,
            win_rate_ci_high_pct=66.5,
        ),
    ]
    monkeypatch.setattr(
        "agents_crypto.calibration_lookup.get_cached_calibration_stats", lambda *a, **k: strong_match
    )

    resp = TestClient(app).get("/api/scan/best-signal")
    assert resp.status_code == 200
    signal = resp.json()["signals"][0]
    entry = signal["entry"]
    assert entry is not None
    assert entry["extension_warning"] is not None
    assert entry["take_profit_conservative"] is not None
    # The zone must sit above the current close (90.0), recentered toward
    # the EMA20 retracement, not glued to the already-extended current price.
    assert entry["entry_zone_low"] > 90.0


def test_forward_tracking_summary_reports_empty_state_with_no_logged_signals(client):
    resp = client.get("/api/forward-tracking/summary")
    assert resp.status_code == 200
    body = resp.json()
    assert body["open_count"] == 0
    assert body["resolved_count"] == 0
    assert body["realized_win_rate_pct"] is None


def test_forward_tracking_summary_separates_open_from_resolved(tmp_path, monkeypatch):
    from scanner.storage import log_signal_outcome

    db_path = tmp_path / "scanner_test.db"
    monkeypatch.setenv("SCANNER_STORAGE_PATH", str(db_path))
    conn = init_db(str(db_path))
    log_signal_outcome(
        conn,
        symbol="AAAUSDT",
        timeframe="1h",
        direction="short",
        label="trend:bearish",
        entry_price=10.0,
        stop_loss=10.5,
        take_profit=9.0,
        take_profit_conservative=9.25,
        promised_win_rate_pct=65.0,
        promised_win_rate_ci_low_pct=60.0,
        promised_n=200,
        horizon_candles=12,
    )
    log_signal_outcome(
        conn,
        symbol="BBBUSDT",
        timeframe="1h",
        direction="long",
        label="trend:bullish",
        entry_price=10.0,
        stop_loss=9.5,
        take_profit=11.0,
        take_profit_conservative=10.75,
        promised_win_rate_pct=70.0,
        promised_win_rate_ci_low_pct=62.0,
        promised_n=150,
        horizon_candles=12,
    )
    from scanner.storage import resolve_signal_outcome

    resolve_signal_outcome(conn, 2, "take_profit", 11.0, 0.9)
    conn.close()

    from webapp.main import app

    resp = TestClient(app).get("/api/forward-tracking/summary")
    assert resp.status_code == 200
    body = resp.json()
    assert body["open_count"] == 1
    assert body["resolved_count"] == 1
    assert body["resolved_by_status"]["take_profit"] == 1
    assert body["realized_win_rate_pct"] == 100.0

