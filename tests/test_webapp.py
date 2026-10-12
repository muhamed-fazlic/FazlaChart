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
        calibration_matches=[{"label": "trend:bearish", "direction": "short", "n": 387, "win_rate_pct": 37.0, "avg_net_return_pct": -0.17}],
        diverges_from_btc=True,
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


def test_analyze_endpoint_returns_502_on_failure(client, monkeypatch):
    def _raise(*_args, **_kwargs):
        raise RuntimeError("Ollama is not reachable")

    monkeypatch.setattr("agents_crypto.derivatives_risk_agent.assess_symbol", _raise)

    resp = client.post("/api/analyze/INJUSDT")
    assert resp.status_code == 502
    assert "Ollama is not reachable" in resp.json()["detail"]
