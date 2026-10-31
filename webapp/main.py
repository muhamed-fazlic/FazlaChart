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
    connect_for_reads,
    fetch_latest_scan,
    fetch_latest_scan_per_symbol,
    fetch_symbol_history,
    init_db,
)

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="FazlaChart Scanner Dashboard")


@app.on_event("startup")
async def _raise_threadpool_capacity() -> None:
    """``GET /api/symbol/{symbol}/calibration`` is a sync endpoint, so each
    call occupies one of anyio's worker-thread pool slots (default 40) for
    its full duration. On a cold cache (e.g. right after a restart) that
    call runs a real backtest — a few seconds of network + computation —
    so a burst of many cold-cache calibration requests can otherwise fill
    the pool and make unrelated fast endpoints (like ``/api/overview``)
    queue up behind them. Raising the capacity gives slow, rare cold-cache
    lookups room without starving the rest of the dashboard."""
    from anyio import to_thread

    to_thread.current_default_thread_limiter().total_tokens = 200


@app.on_event("startup")
async def _warm_up_llm_client_stack() -> None:
    """``POST /api/analyze`` imports the LLM client stack (``langchain_openai``
    and friends) lazily, on its very first call, so merely running the
    dashboard never pays for it. That first import reads dozens of files off
    disk and is genuinely slow (seconds) even in isolation — and if it
    happens to land while the scanner/win-rate calibration backtests (pandas,
    CPU-bound, running on several worker threads at once) are mid-cycle, the
    GIL starves that one import thread of CPU time, stretching it out
    dramatically. Every *other* concurrent ``/api/analyze`` call then also
    blocks (Python serializes concurrent imports of the same module with a
    per-module lock), so a single unlucky first click can make every
    "Analyze with AI" button appear to hang indefinitely — with no error,
    because nothing has actually failed, it's just queued behind a slow
    import. Paying this cost once at startup, before the server is under
    load, avoids that entirely; this does not call Binance or an LLM, it
    only loads the Python modules."""
    import logging

    import anyio

    def _import_llm_stack() -> None:
        # Merely importing the modules isn't enough: the ``openai`` SDK's
        # ``chat``/``completions`` submodules are themselves lazily imported
        # the first time ``ChatOpenAI.__init__`` touches ``client.chat`` (a
        # lazy property), so only *constructing* a client triggers that
        # deeper import. This makes no network call — building the object is
        # pure local setup.
        from tradingagents.llm_clients.factory import create_llm_client

        create_llm_client(provider="groq", model="openai/gpt-oss-120b").get_llm()

    try:
        await anyio.to_thread.run_sync(_import_llm_stack)
    except Exception:
        # Best-effort warm-up only (e.g. missing GROQ_API_KEY): worst case we
        # fall back to the original lazy-import-on-first-click behavior, so
        # a misconfiguration here must never prevent the dashboard from
        # starting.
        logging.getLogger(__name__).warning("LLM client warm-up failed; will import lazily on first use", exc_info=True)


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
    # separate connection for writes. Re-running the schema-creation writes
    # of init_db() on every single request used to add needless write-lock
    # contention against the scanner's own writes (this is what caused the
    # "database is locked" crash once the win-rate column started firing
    # dozens of near-simultaneous requests per page load) — so once the
    # database file actually exists, just open a plain read connection.
    path = _storage_path()
    if not Path(path).exists():
        return init_db(path)
    return connect_for_reads(path)


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


_symbols_cache: dict[str, Any] = {"symbols": None, "fetched_at": 0.0}
_SYMBOLS_CACHE_TTL_SECONDS = 3600  # the tradable-symbol list barely changes; no point re-fetching every keystroke


@app.get("/api/symbols")
def list_symbols() -> dict[str, Any]:
    """Every USDT-margined perpetual futures symbol Binance currently lists —
    not just the ~75 the scanner actively tracks — so the dashboard's symbol
    filter can suggest/autocomplete (and later look up) any coin, including
    ones outside the scanner's top-N-by-volume universe (e.g. PAXGUSDT,
    1000PEPEUSDT). Cached in-process for an hour: this list is effectively
    static minute-to-minute, so it's cheap to reuse instead of re-querying
    Binance on every keystroke."""
    import time as _time

    from tradingagents.dataflows.vendors.binance import client

    now = _time.time()
    if _symbols_cache["symbols"] is None or (now - _symbols_cache["fetched_at"]) > _SYMBOLS_CACHE_TTL_SECONDS:
        _symbols_cache["symbols"] = sorted(client.get_usdt_perpetual_symbols())
        _symbols_cache["fetched_at"] = now
    return {"symbols": _symbols_cache["symbols"]}


def _row_to_summary(row: dict[str, Any]) -> dict[str, Any]:
    """Shapes one scanner DB row into the flat per-symbol dict the frontend
    table (and the single-symbol lookup endpoint) render. Shared so both
    callers stay in sync instead of duplicating this field mapping."""
    tfs = row["snapshot"].get("timeframes", {})
    ref_tf = row["reference_timeframe"]
    ref = tfs.get(ref_tf, {})
    derivatives = row["snapshot"].get("derivatives", {})
    return {
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


def _btc_summary(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """BTC summary block derived from BTC's own latest scan row — shared by
    ``/api/overview`` and ``/api/scan/top`` so both views give the same BTC
    reference context without any extra Binance calls."""
    btc_symbol = _btc_symbol()
    btc_row = next((r for r in rows if r["symbol"] == btc_symbol), None)
    if btc_row is None:
        return None
    tfs = btc_row["snapshot"].get("timeframes", {})
    return {
        "symbol": btc_symbol,
        "close": btc_row["close_price"],
        "trend_by_timeframe": {tf: data["trend"] for tf, data in tfs.items()},
        "funding_rate": btc_row["snapshot"].get("derivatives", {}).get("funding_rate"),
        "open_interest_change_pct_60m": btc_row["snapshot"].get("derivatives", {}).get("open_interest_change_pct_60m"),
        "anomalies": btc_row["anomalies"],
        "scanned_at": btc_row["scanned_at"],
    }


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

    btc_summary = _btc_summary(rows)
    symbols = [_row_to_summary(row) for row in rows]

    # Surface the most "interesting" symbols first: anomalies, then recency.
    symbols.sort(key=lambda s: (len(s["anomalies"]) == 0, s["symbol"]), reverse=False)

    generated_at = max(r["scanned_at"] for r in rows)
    return {"btc": btc_summary, "symbols": symbols, "generated_at": generated_at}


def _structure_strength(summary: dict[str, Any]) -> float:
    """Heuristic ranking score only — never a prediction — for how strongly a
    symbol's *current* trend structure is expressed, used to pick the default
    "top movers" view. Rewards: EMA20/EMA50 separation relative to price
    (trend conviction), RSI distance from the neutral 50 midpoint, confirming
    anomaly flags, and above-average relative volume."""
    close = summary.get("close") or 0
    ema20, ema50 = summary.get("ema20"), summary.get("ema50")
    ema_spread_pct = abs(ema20 - ema50) / close * 100 if close and ema20 is not None and ema50 is not None else 0.0
    rsi = summary.get("rsi")
    rsi_distance = abs(rsi - 50) if rsi is not None else 0.0
    volume_bonus = max((summary.get("relative_volume") or 1.0) - 1.0, 0.0)
    anomaly_bonus = len(summary.get("anomalies") or [])
    return ema_spread_pct * 20 + rsi_distance * 0.5 + anomaly_bonus * 8 + volume_bonus * 5


@app.get("/api/scan/top")
def top_movers(limit: int = 10) -> dict[str, Any]:
    """Default dashboard view: the ``limit`` tracked symbols *currently*
    labeled bullish or bearish (neutral excluded — no directional structure
    to rank) with the strongest trend structure per :func:`_structure_strength`.
    Reads straight from the scanner's existing DB, same as ``/api/overview``
    — no live Binance calls, so this stays cheap even though it's the
    default page load."""
    conn = _get_conn()
    try:
        rows = fetch_latest_scan_per_symbol(conn)
    finally:
        conn.close()

    if not rows:
        return {"btc": None, "symbols": [], "generated_at": None}

    btc_summary = _btc_summary(rows)
    directional_rows = [r for r in rows if r["trend"] in ("bullish", "bearish")]
    summaries = [_row_to_summary(row) for row in directional_rows]
    summaries.sort(key=_structure_strength, reverse=True)

    generated_at = max(r["scanned_at"] for r in rows)
    return {"btc": btc_summary, "symbols": summaries[: max(1, limit)], "generated_at": generated_at}


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

    calibration_stats = get_calibration_stats(symbol, timeframe, horizons=(horizon_candles,), use_process=True)
    matches = find_stats_for_labels(calibration_stats, labels, horizon_candles)

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "horizon_candles": horizon_candles,
        "active_labels": labels,
        "matches": [m.__dict__ for m in matches],
    }


@app.get("/api/symbol/{symbol}/heatmap")
def symbol_liquidity_heatmap(symbol: str, timeframe: str = "1h", bins: int = 24) -> dict[str, Any]:
    """Approximate volume-profile / likely stop-order heatmap for one
    symbol+timeframe — see
    ``tradingagents.dataflows.vendors.binance.liquidity_heatmap`` for the
    method. Not real order-book data; a proxy to give the dashboard (and the
    AI analysis prompt) a sense of where TP/SL clusters likely sit."""
    symbol = symbol.upper()
    from tradingagents.dataflows.vendors.binance.snapshot import build_liquidity_heatmap_snapshot

    try:
        return build_liquidity_heatmap_snapshot(symbol, timeframe=timeframe, bins=bins)
    except Exception as exc:  # noqa: BLE001 - surface bad/unknown symbols clearly
        raise HTTPException(status_code=404, detail=f"Could not fetch heatmap for {symbol}: {exc}") from exc


@app.get("/api/symbol/{symbol}/lookup")
def symbol_lookup(symbol: str, timeframe: str = "1h", horizon_candles: int = 12) -> dict[str, Any]:
    """On-demand, single-symbol version of ``/api/overview`` + its win-rate
    cell, bundled into one call — the default way the dashboard now looks
    things up, instead of rendering/computing win-rates for every tracked
    symbol up front.

    Works even for a symbol the scanner has never tracked (outside its
    top-N-by-volume universe): falls back to a live Binance snapshot for the
    row fields in that case (``"tracked": false`` in the response), since
    none of those fields need scanner history to compute.
    """
    symbol = symbol.upper()
    conn = _get_conn()
    try:
        row = fetch_latest_scan(conn, symbol)
    finally:
        conn.close()

    if row is not None:
        summary = _row_to_summary(row)
        lookup_timeframe = summary["reference_timeframe"]
        labels = _active_labels(row["trend"], row["anomalies"])
        summary["tracked"] = True
    else:
        # Imported lazily: a free/no-LLM lookup shouldn't pay for pulling in
        # the LLM client stack, only the Binance data layer.
        from tradingagents.dataflows.vendors.binance.snapshot import build_symbol_snapshot

        try:
            snapshot = build_symbol_snapshot(symbol, timeframes=(timeframe,))
        except Exception as exc:  # noqa: BLE001 - surface bad/unknown symbols clearly
            raise HTTPException(status_code=404, detail=f"Could not fetch live data for {symbol}: {exc}") from exc
        tf = snapshot["timeframes"][timeframe]
        derivatives = snapshot.get("derivatives", {})
        summary = {
            "symbol": symbol,
            "trend": tf["trend"],
            "close": tf["close"],
            "reference_timeframe": timeframe,
            "rsi": tf.get("rsi"),
            "ema20": tf.get("ema20"),
            "ema50": tf.get("ema50"),
            "volume_change_pct": tf.get("volume_change_pct"),
            "relative_volume": tf.get("relative_volume"),
            "support": tf.get("support"),
            "resistance": tf.get("resistance"),
            "momentum_fading": tf.get("momentum_fading"),
            "funding_rate": derivatives.get("funding_rate"),
            "open_interest_change_pct_60m": derivatives.get("open_interest_change_pct_60m"),
            "anomalies": tf.get("anomalies", []),
            "scanned_at": None,
            "age_seconds": 0,
            "snapshot": snapshot,
            "tracked": False,
        }
        lookup_timeframe = timeframe
        labels = _active_labels(tf["trend"], tf.get("anomalies", []))

    # Imported lazily so a page that never looks anything up never pulls in
    # the backtest/pandas machinery.
    from agents_crypto.calibration_lookup import find_stats_for_labels, get_calibration_stats

    calibration_stats = get_calibration_stats(symbol, lookup_timeframe, horizons=(horizon_candles,), use_process=True)
    matches = find_stats_for_labels(calibration_stats, labels, horizon_candles)
    summary["calibration_matches"] = [m.__dict__ for m in matches]
    return summary


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
        "multi_timeframe_risk_levels": result.multi_timeframe_risk_levels,
        "calibration_matches": result.calibration_matches,
        "diverges_from_btc": result.diverges_from_btc,
        "liquidity_heatmap": result.liquidity_heatmap,
    }



# Static dashboard (HTML/CSS/JS) — mounted last so it doesn't shadow /api/*.
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))
