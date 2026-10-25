"""FastAPI dashboard: a read-only web view over the Market Scanner's SQLite
database (``scanner_data/scanner.db`` by default).

Run with:
    uvicorn webapp.main:app --reload --port 8000
or:
    .\\tool\\run_dashboard.ps1   /   ./tool/run_dashboard.sh

This process never calls Binance or an LLM on its own — it only reads what
the scanner (``python -m scanner``, running separately) has already written
to disk, so it is safe to start/stop/restart independently of the scanner
itself. The one exception is ``POST /api/analyze/{symbol}``, which is only
ever triggered by an explicit user click (the "Analyze with AI" button) and
calls the Derivatives/Risk agent (live Binance snapshot + local LLM).
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from scanner.storage import (
    fetch_latest_scan,
    fetch_latest_scan_per_symbol,
    fetch_symbol_history,
    init_db,
)

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="FazlaChart Scanner Dashboard")


@app.middleware("http")
async def _no_cache_dashboard_assets(request, call_next):
    """The dashboard's HTML/CSS/JS change far more often during active
    development than a browser's default heuristic caching assumes, and
    Chrome will otherwise silently serve a stale copy from its disk cache
    on the next launch (e.g. the desktop shortcut reopening an old tab)
    without even revalidating. Force a conditional request every time for
    the index page and everything under /static/ so a server-side change
    is never masked by a stale local cache."""
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
    return response


def _storage_path() -> str:
    """Read lazily (not cached at import time) so tests can point this at a
    throwaway database via the SCANNER_STORAGE_PATH env var."""
    return os.environ.get("SCANNER_STORAGE_PATH", "scanner_data/scanner.db")


def _btc_symbol() -> str:
    return os.environ.get("SCANNER_BTC_SYMBOL", "BTCUSDT")


def _get_conn() -> sqlite3.Connection:
    # A short-lived connection per request is simplest and safe for SQLite's
    # single-writer/many-reader model; the scanner process holds its own
    # separate connection for writes.
    return init_db(_storage_path())


def _age_seconds(iso_timestamp: str) -> float:
    from datetime import UTC, datetime

    ts = datetime.fromisoformat(iso_timestamp)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return (datetime.now(UTC) - ts).total_seconds()


@app.get("/api/health")
def health() -> dict[str, Any]:
    storage_path = _storage_path()
    exists = Path(storage_path).exists()
    return {"ok": True, "storage_path": storage_path, "storage_exists": exists}


@app.get("/api/overview")
def overview() -> dict[str, Any]:
    """Latest known state of every symbol the scanner has ever scanned, plus
    a BTC summary block (derived from BTC's own latest row, so no extra
    Binance calls are made here)."""
    conn = _get_conn()
    try:
        rows = fetch_latest_scan_per_symbol(conn)
    finally:
        conn.close()

    if not rows:
        return {"btc": None, "symbols": [], "generated_at": None}

    btc_symbol = _btc_symbol()
    btc_row = next((r for r in rows if r["symbol"] == btc_symbol), None)
    btc_summary = None
    if btc_row is not None:
        tfs = btc_row["snapshot"].get("timeframes", {})
        btc_summary = {
            "symbol": btc_symbol,
            "close": btc_row["close_price"],
            "trend_by_timeframe": {tf: data["trend"] for tf, data in tfs.items()},
            "funding_rate": btc_row["snapshot"].get("derivatives", {}).get("funding_rate"),
            "open_interest_change_pct_60m": btc_row["snapshot"]
            .get("derivatives", {})
            .get("open_interest_change_pct_60m"),
            "anomalies": btc_row["anomalies"],
            "scanned_at": btc_row["scanned_at"],
        }

    symbols = []
    for row in rows:
        tfs = row["snapshot"].get("timeframes", {})
        ref_tf = row["reference_timeframe"]
        ref = tfs.get(ref_tf, {})
        derivatives = row["snapshot"].get("derivatives", {})
        symbols.append(
            {
                "symbol": row["symbol"],
                "trend": row["trend"],
                "close": row["close_price"],
                "reference_timeframe": ref_tf,
                "rsi": ref.get("rsi"),
                "ema20": ref.get("ema20"),
                "ema50": ref.get("ema50"),
                "volume_change_pct": ref.get("volume_change_pct"),
                "relative_volume": ref.get("relative_volume"),
                "support": ref.get("support"),
                "resistance": ref.get("resistance"),
                "momentum_fading": ref.get("momentum_fading"),
                "funding_rate": derivatives.get("funding_rate"),
                "open_interest_change_pct_60m": derivatives.get("open_interest_change_pct_60m"),
                "anomalies": row["anomalies"],
                "scanned_at": row["scanned_at"],
                "age_seconds": round(_age_seconds(row["scanned_at"]), 1),
                "snapshot": row["snapshot"],
            }
        )

    # Surface the most "interesting" symbols first: anomalies, then recency.
    symbols.sort(key=lambda s: (len(s["anomalies"]) == 0, s["symbol"]), reverse=False)

    generated_at = max(r["scanned_at"] for r in rows)
    return {"btc": btc_summary, "symbols": symbols, "generated_at": generated_at}


def _active_labels(trend: str, anomalies: list[str]) -> list[str]:
    """Same label-naming scheme ``agents_crypto.derivatives_risk_agent``
    uses, duplicated here (it's two lines) instead of imported, so this
    free endpoint never has to pull in the LLM client stack just to compute
    which calibration labels are active right now."""
    return [f"trend:{trend}"] + [f"anomaly:{a}" for a in anomalies]


@app.get("/api/symbol/{symbol}/calibration")
def symbol_calibration(symbol: str, horizon_candles: int = 12) -> dict[str, Any]:
    """The historical win-rate for whatever label(s) are active on this
    symbol *right now*, straight from the cached calibration backtest — the
    exact same ground truth ``POST /api/analyze/{symbol}`` grounds its LLM
    call in, but with no LLM call and no tokens spent. Meant to be checked
    first, so an "Analyze with AI" click only happens once this already
    looks worth spending a call on.
    """
    symbol = symbol.upper()
    conn = _get_conn()
    try:
        row = fetch_latest_scan(conn, symbol)
    finally:
        conn.close()
    if row is None:
        raise HTTPException(status_code=404, detail=f"No scan history for {symbol} yet")

    timeframe = row["reference_timeframe"]
    tf_snapshot = row["snapshot"].get("timeframes", {}).get(timeframe, {})
    labels = _active_labels(row["trend"], tf_snapshot.get("anomalies", []))

    # Imported lazily so merely running the dashboard never pulls in the
    # backtest/pandas machinery unless this endpoint is actually hit.
    from agents_crypto.calibration_lookup import find_stats_for_labels, get_calibration_stats

    calibration_stats = get_calibration_stats(symbol, timeframe, horizons=(horizon_candles,))
    matches = find_stats_for_labels(calibration_stats, labels, horizon_candles)

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "horizon_candles": horizon_candles,
        "active_labels": labels,
        "matches": [m.__dict__ for m in matches],
    }


@app.get("/api/symbol/{symbol}/history")
def symbol_history(symbol: str, limit: int = 200) -> dict[str, Any]:
    conn = _get_conn()
    try:
        history = fetch_symbol_history(conn, symbol.upper(), limit=limit)
    finally:
        conn.close()
    if not history:
        raise HTTPException(status_code=404, detail=f"No scan history for {symbol.upper()} yet")
    return {"symbol": symbol.upper(), "history": history}


@app.post("/api/analyze/{symbol}")
def analyze_symbol(
    symbol: str,
    timeframe: str = "1h",
    horizon_candles: int = 12,
    model: str | None = None,
    provider: str | None = None,
) -> dict[str, Any]:
    """On-demand only: calls the Derivatives/Risk LLM agent (live Binance
    snapshot + cached calibration stats + an LLM — local Ollama or a cloud
    provider such as Groq) for one symbol. This is the one endpoint in this
    dashboard that leaves the "read-only, no LLM" world — it's a deliberate,
    explicit user action (clicking "Analyze"), never triggered automatically
    by the auto-refresh loop.

    Runs synchronously in FastAPI's threadpool, which can take anywhere from
    a couple of seconds (cloud provider) to several minutes (local CPU-only
    Ollama) depending on the chosen provider/model — there is no
    streaming/progress here.
    """
    # Imported lazily so merely running the dashboard never pulls in the LLM
    # client stack (langchain, etc.) unless this endpoint is actually hit.
    from agents_crypto.derivatives_risk_agent import DEFAULT_MODEL, DEFAULT_PROVIDER, assess_symbol

    try:
        result = assess_symbol(
            symbol.upper(),
            timeframe=timeframe,
            horizon_candles=horizon_candles,
            provider=provider or os.environ.get("DASHBOARD_LLM_PROVIDER", DEFAULT_PROVIDER),
            model=model or os.environ.get("DASHBOARD_LLM_MODEL", DEFAULT_MODEL),
        )
    except Exception as exc:  # noqa: BLE001 - surface any failure (Ollama down, bad symbol, etc.) to the UI
        raise HTTPException(status_code=502, detail=f"Analysis failed: {exc}") from exc

    return {
        "symbol": symbol.upper(),
        "timeframe": timeframe,
        "assessment": result.assessment.model_dump(),
        "risk_levels": result.risk_levels,
        "calibration_matches": result.calibration_matches,
        "diverges_from_btc": result.diverges_from_btc,
    }



# Static dashboard (HTML/CSS/JS) — mounted last so it doesn't shadow /api/*.
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))
