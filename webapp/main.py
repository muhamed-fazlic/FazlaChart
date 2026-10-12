"""FastAPI dashboard: a read-only web view over the Market Scanner's SQLite
database (``scanner_data/scanner.db`` by default).

Run with:
    uvicorn webapp.main:app --reload --port 8000
or:
    .\\tool\\run_dashboard.ps1   /   ./tool/run_dashboard.sh

This process never calls Binance or an LLM — it only reads what the scanner
(``python -m scanner``, running separately) has already written to disk, so
it is safe to start/stop/restart independently of the scanner itself.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from scanner.storage import fetch_latest_scan_per_symbol, fetch_symbol_history, init_db

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="FazlaChart Scanner Dashboard")


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


# Static dashboard (HTML/CSS/JS) — mounted last so it doesn't shadow /api/*.
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))
