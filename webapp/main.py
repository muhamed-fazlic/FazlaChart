"""FastAPI dashboard: a read-only web view over the Market Scanner's SQLite
database (``scanner_data/scanner.db`` by default).

Run with:
    uvicorn webapp.main:app --reload --port 8000
or:
    .\\tool\\run_dashboard.ps1   /   ./tool/run_dashboard.sh

This process never calls Binance or an LLM on its own — it only reads what
the scanner (``python -m scanner``, running separately) has already written
to disk, so it is safe to start/stop/restart independently of the scanner
itself. Two exceptions, both only ever triggered by an explicit user click:
``POST /api/analyze/{symbol}`` (the "Analyze with AI" button; calls the
Derivatives/Risk agent — live Binance snapshot + local LLM), and
``POST /api/backtest/refresh`` (the "🔄 Osvježi" button on the Backtest tab;
launches ``tool/run_full_portfolio_backtest.sh`` as a detached, nice+cpulimit
throttled background process and returns immediately — never runs it inline).
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from scanner.drift_detector import active_muted_bucket_set
from scanner.storage import (
    connect_for_reads,
    fetch_all_active_muted_buckets,
    fetch_latest_scan,
    fetch_latest_scan_per_symbol,
    fetch_recent_signal_outcomes,
    fetch_symbol_history,
    init_db,
)
from scanner.validation.experiment_registry import (
    fetch_experiments,
    init_db as init_experiment_registry,
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


_CALIBRATION_WARMUP_BATCH_SIZE = 2  # matches calibration_lookup's backtest-worker-process cap
_CALIBRATION_WARMUP_INTERVAL_SECONDS = 3


@app.on_event("startup")
async def _start_calibration_warmup_loop() -> None:
    """``/api/scan/best-signal`` only ever reads the on-disk calibration
    cache (see ``get_cached_calibration_stats``) — it never backtests
    on-demand, to stay fast even when ranking 100+ symbols across 4
    independent timeframes each. Without this loop that cache would stay
    empty for any (symbol, timeframe) pair a human never happened to click
    on, making "best signal" silently blind to most of the tracked
    universe. This fills it in the background, a couple of pairs every
    ~20s, so it's self-sustaining without ever blocking a real request."""
    import asyncio

    asyncio.create_task(_calibration_warmup_loop())


_HEALTH_CHECK_INTERVAL_SECONDS = 120
_HEALTH_ALERT_COOLDOWN_SECONDS = 3600  # one "scanner looks stuck" alert per hour, not one per 2-minute check
_health_alert_state: dict[str, Any] = {"last_alert_at": 0.0, "was_stale": False}


@app.on_event("startup")
async def _start_health_watchdog_loop() -> None:
    """The dashboard process is simpler and far less likely to hang than the
    scanner (it does no Binance/LLM calls of its own on this path), so it is
    the natural place to notice "the scanner stopped writing fresh data" and
    tell a human via Telegram -- rather than relying on someone to notice the
    numbers on screen stopped moving. Runs independently of whether anyone
    has the dashboard tab open."""
    import asyncio

    asyncio.create_task(_health_watchdog_loop())


async def _health_watchdog_loop() -> None:
    import logging
    import time as _time

    import anyio

    from scanner.health import check_scanner_health
    from scanner.telegram_alerts import TelegramAlertConfig, send_telegram_message

    logger = logging.getLogger(__name__)
    telegram = TelegramAlertConfig(
        bot_token=os.environ.get("TELEGRAM_BOT_TOKEN"),
        chat_id=os.environ.get("TELEGRAM_CHAT_ID"),
    )
    while True:
        try:
            if Path(_storage_path()).exists() and telegram.enabled:
                # Open (and close) the connection inside the worker-thread
                # callback itself: sqlite3 connections are thread-affine by
                # default, so one opened here on the event-loop thread and
                # then handed across to anyio's worker thread would raise
                # "SQLite objects created in a thread can only be used in
                # that same thread" on every single cycle.
                def _check_health() -> Any:
                    conn = _get_conn()
                    try:
                        return check_scanner_health(conn, max_staleness_seconds=_health_max_staleness_seconds())
                    finally:
                        conn.close()

                result = await anyio.to_thread.run_sync(_check_health)
                now = _time.time()
                if result.status == "stale":
                    cooldown_elapsed = now - _health_alert_state["last_alert_at"] > _HEALTH_ALERT_COOLDOWN_SECONDS
                    if cooldown_elapsed:
                        minutes = round((result.staleness_seconds or 0) / 60, 1)
                        message = (
                            f"\u26a0\ufe0f FazlaChart scanner izgleda zaglavljen — nema novih podataka {minutes} min "
                            f"(zadnji scan: {result.last_scanned_at}). Provjeri servis na VPS-u."
                        )
                        if send_telegram_message(telegram.bot_token, telegram.chat_id, message):
                            _health_alert_state["last_alert_at"] = now
                            logger.warning("Sent scanner-staleness Telegram alert (%s min stale)", minutes)
                elif _health_alert_state["was_stale"]:
                    # Recovered -- let the next stale episode alert immediately
                    # rather than waiting out the old cooldown window.
                    _health_alert_state["last_alert_at"] = 0.0
                _health_alert_state["was_stale"] = result.status == "stale"
        except Exception:
            logger.warning("Health watchdog cycle failed", exc_info=True)
        await anyio.sleep(_HEALTH_CHECK_INTERVAL_SECONDS)



async def _calibration_warmup_loop() -> None:
    import asyncio
    import logging

    import anyio

    from agents_crypto.best_signal import TRACKED_TIMEFRAMES
    from agents_crypto.calibration_lookup import get_cached_calibration_stats, get_calibration_stats

    logger = logging.getLogger(__name__)

    async def _warm_one(symbol: str, timeframe: str) -> None:
        try:
            await anyio.to_thread.run_sync(
                lambda: get_calibration_stats(
                    symbol, timeframe, horizons=(_BEST_SIGNAL_HORIZON_CANDLES,), use_process=True
                )
            )
        except Exception:
            logger.warning("Calibration warm-up failed for %s/%s", symbol, timeframe, exc_info=True)

    while True:
        try:
            conn = _get_conn()
            try:
                rows = fetch_latest_scan_per_symbol(conn)
            finally:
                conn.close()
            # Every (symbol, timeframe) combination compute_candidates can
            # surface needs its own warm calibration cache entry, not just
            # the scan row's flattened reference timeframe -- see
            # TRACKED_TIMEFRAMES's docstring for why each timeframe is an
            # independent search.
            stale = [
                (r["symbol"], tf)
                for r in rows
                for tf in TRACKED_TIMEFRAMES
                if (tf_snapshot := r.get("snapshot", {}).get("timeframes", {}).get(tf))
                and tf_snapshot.get("trend") in ("bullish", "bearish")
                and not get_cached_calibration_stats(r["symbol"], tf)
            ]
            # Dispatch the whole batch concurrently instead of awaiting one
            # pair at a time -- the shared ProcessPoolExecutor already caps
            # real parallelism at _MAX_BACKTEST_WORKERS, so gathering here
            # costs nothing extra but keeps that pool actually busy instead
            # of sitting half-idle waiting on a strictly sequential loop.
            await asyncio.gather(
                *(_warm_one(symbol, tf) for symbol, tf in stale[:_CALIBRATION_WARMUP_BATCH_SIZE])
            )
        except Exception:
            logger.warning("Calibration warm-up cycle failed", exc_info=True)
        await anyio.sleep(_CALIBRATION_WARMUP_INTERVAL_SECONDS)


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


def _backtest_report_path() -> str:
    """Read lazily so tests can point this at a throwaway file via the
    BACKTEST_REPORT_PATH env var — mirrors ``_storage_path()``'s convention.
    This file is written on demand by
    ``python -m scanner.backtesting.portfolio_backtest --report-json ...``;
    it is NOT regenerated automatically by this webapp process."""
    return os.environ.get("BACKTEST_REPORT_PATH", "scanner_data/portfolio_backtest_report.json")


def _experiment_registry_path() -> str:
    """Read lazily so tests can point this at a throwaway file via the
    EXPERIMENT_REGISTRY_PATH env var — mirrors ``_backtest_report_path()``'s
    convention. Written on demand by
    ``python -m scanner.backtesting.portfolio_backtest --experiment-family ...``;
    this webapp process only ever reads it."""
    return os.environ.get("EXPERIMENT_REGISTRY_PATH", "scanner_data/experiment_registry.db")


def _backtest_refresh_status_path() -> str:
    """Mirrors ``_backtest_report_path()``'s env-var convention. Tracks the
    state of the background job launched by ``POST /api/backtest/refresh``
    so a page reload / a second browser tab can see whether a refresh is
    already in progress, instead of this webapp process holding that state
    only in memory (which a restart would silently lose while the backtest
    keeps running)."""
    return os.environ.get("BACKTEST_REFRESH_STATUS_PATH", "scanner_data/backtest_refresh_status.json")


def _backtest_refresh_log_path() -> str:
    return os.environ.get("BACKTEST_REFRESH_LOG_PATH", "scanner_data/backtest_refresh.log")


def _backtest_refresh_script_path() -> Path:
    return Path(__file__).resolve().parent.parent / "tool" / "run_full_portfolio_backtest.sh"


def _read_refresh_status() -> dict[str, Any]:
    path = Path(_backtest_refresh_status_path())
    if not path.exists():
        return {"state": "idle"}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"state": "idle"}


def _write_refresh_status(status: dict[str, Any]) -> None:
    path = Path(_backtest_refresh_status_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status, indent=2), encoding="utf-8")


def _pid_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    except AttributeError:
        # os.kill(pid, 0) isn't available on Windows -- this whole feature
        # is VPS-only in practice, but don't hard-crash local dev.
        return True
    return True


def _refresh_in_progress(status: dict[str, Any]) -> bool:
    return status.get("state") == "running" and isinstance(status.get("pid"), int) and _pid_is_running(status["pid"])


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
    if not exists:
        return {"ok": False, "storage_path": storage_path, "storage_exists": False, "status": "no_data"}

    from scanner.health import check_scanner_health

    conn = _get_conn()
    try:
        result = check_scanner_health(conn, max_staleness_seconds=_health_max_staleness_seconds())
    finally:
        conn.close()
    return {
        "ok": result.status == "healthy",
        "storage_path": storage_path,
        "storage_exists": True,
        "status": result.status,
        "last_scanned_at": result.last_scanned_at,
        "staleness_seconds": result.staleness_seconds,
        "symbols_ever_tracked": result.symbols_ever_tracked,
        "max_staleness_seconds": result.max_staleness_seconds,
    }


def _health_max_staleness_seconds() -> float:
    return float(os.environ.get("SCANNER_HEALTH_MAX_STALENESS_SECONDS", "900"))


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


_BEST_SIGNAL_HORIZON_CANDLES = 12
_BEST_SIGNAL_MIN_SAMPLE_SIZE = 30  # below this, even a high win rate is too noisy to call a real edge


@app.get("/api/scan/best-signal")
def best_signal(limit: int = 5) -> dict[str, Any]:
    """Statistically-grounded "what's actually worth trading right now"
    shortlist — NOT a heuristic ranking like ``/api/scan/top``. Ranks every
    currently-tracked directional symbol by the conservative (95%-CI
    lower-bound) win rate of its active calibration label(s), so a lucky
    small sample can never outrank a well-tested edge, and requires a
    minimum sample size before a label counts as "significant" at all.

    Read-only and always fast: it only ever reads whatever's already in the
    on-disk calibration cache (populated by the background warm-up loop and
    by users' own lookups) and never triggers a fresh backtest inline — a
    symbol simply has no entry here until its cache is warm.

    The ranking itself lives in ``agents_crypto.best_signal.compute_candidates``
    so the scanner's live forward-outcome tracker can log exactly what this
    endpoint would have shown at the same instant (see
    ``scanner.signal_tracking``), instead of risking the two silently
    drifting apart over time."""
    conn = _get_conn()
    try:
        rows = fetch_latest_scan_per_symbol(conn)
        muted_buckets = active_muted_bucket_set(conn)
    finally:
        conn.close()

    from agents_crypto.best_signal import compute_candidates

    return compute_candidates(rows, limit=limit, muted_buckets=muted_buckets)


_TIME_RANGE_PRESETS: dict[str, timedelta] = {
    "1h": timedelta(hours=1),
    "4h": timedelta(hours=4),
    "12h": timedelta(hours=12),
    "1d": timedelta(days=1),
    "1w": timedelta(days=7),
    "1M": timedelta(days=30),
}


def _resolve_time_range(range_: str | None, since: str | None, until: str | None) -> tuple[str | None, str | None]:
    """Translate the "Analitika trejdova" tab's time-range filter into
    ``(since, until)`` ISO-8601 timestamps for
    :func:`scanner.storage.fetch_recent_signal_outcomes`.

    Explicit ``since``/``until`` (ISO timestamps) always win -- they cover
    the "custom range" picker, letting a user pick a specific past window
    instead of "now minus N". Otherwise ``range_`` is resolved against one
    of the preset tokens (``1h``/``4h``/``12h``/``1d``/``1w``/``1M``)
    relative to now. An unrecognized/missing ``range_`` with no explicit
    bounds means "no filter, all history" -- never an error, since this is
    a dashboard convenience filter, not a strict API contract."""
    if since is not None or until is not None:
        return since, until
    if range_ is not None and range_ in _TIME_RANGE_PRESETS:
        cutoff = datetime.now(UTC) - _TIME_RANGE_PRESETS[range_]
        return cutoff.isoformat(), None
    return None, None


def _quality_bucket(quality_score: float | None) -> str | None:
    """Bucket a signal's Quality Score (0-100) into the same tiers shown on
    the dashboard badge (see ``qualityScoreBadgeHtml`` in app.js), so the
    trade-analytics breakdown answers the question the score exists for:
    does a higher score actually realize a better win rate, empirically?"""
    if quality_score is None:
        return None
    if quality_score >= 70:
        return "70-100 (jak)"
    if quality_score >= 50:
        return "50-69 (osrednji)"
    return "0-49 (slab)"


def _bucket_stats(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Shared win-rate/avg-net/gap-vs-promised aggregation used by every
    "group resolved trades by X" breakdown endpoint, so by-bucket,
    by-pattern, by-regime, and by-quality-bucket all compute the exact same
    statistics the exact same way."""
    net_returns = [o["realized_return_pct"] for o in items if o["realized_return_pct"] is not None]
    wins = sum(1 for r in net_returns if r > 0)
    promised = [o["promised_win_rate_ci_low_pct"] for o in items if o["promised_win_rate_ci_low_pct"] is not None]
    win_rate = round(wins / len(net_returns) * 100, 1) if net_returns else None
    avg_net = round(sum(net_returns) / len(net_returns), 2) if net_returns else None
    avg_promised = round(sum(promised) / len(promised), 1) if promised else None
    gap = round(win_rate - avg_promised, 1) if win_rate is not None and avg_promised is not None else None
    by_status = {"take_profit": 0, "stop_loss": 0, "timeout": 0, "tp1_be": 0, "tp2_be": 0, "full_run": 0}
    for o in items:
        by_status[o["status"]] = by_status.get(o["status"], 0) + 1
    return {
        "resolved_count": len(items),
        "by_status": by_status,
        "win_rate_pct": win_rate,
        "avg_net_return_pct": avg_net,
        "avg_promised_win_rate_ci_low_pct": avg_promised,
        "gap_pp": gap,
    }


@app.get("/api/drift-status")
def drift_status() -> dict[str, Any]:
    """Which ``(timeframe, direction)`` buckets the live calibration-drift
    detector currently has muted (see ``scanner.drift_detector``), and why
    — surfaced here so a user can see *why* a bucket is missing from Top
    Signal Now instead of it silently disappearing. Empty list is the
    normal/healthy state."""
    conn = _get_conn()
    try:
        muted = fetch_all_active_muted_buckets(conn)
    finally:
        conn.close()
    return {"muted_buckets": muted, "count": len(muted)}


@app.get("/api/backtest/report")
def backtest_report() -> dict[str, Any]:
    """Reads the cached retrospective portfolio-backtest report (Phase 8
    Korak 2 — see PHASE8_PROFITABILITY_RESEARCH_TODO.md) for the "🧪
    Backtest" tab. This is a **counterfactual simulation** replayed over
    history, not a live result — deliberately kept separate from
    ``/api/forward-tracking/*`` (which tracks real signals the live system
    actually surfaced). The report file is generated on demand by
    ``python -m scanner.backtesting.portfolio_backtest --report-json ...``
    (a multi-minute job that fetches klines over the network), never by
    this request handler, since the VPS has no spare CPU/network budget to
    run it inline on every page load.

    Returns ``{"available": False}`` rather than a 404/500 if no report has
    been generated yet, so the frontend can render a friendly empty state."""
    path = Path(_backtest_report_path())
    if not path.exists():
        return {"available": False}
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=f"Could not read backtest report: {exc}") from None
    report["available"] = True
    return report


@app.get("/api/backtest/experiments")
def backtest_experiments(limit: int = 50) -> dict[str, Any]:
    """Reads the Phase 8 Korak 3 experiment registry (see
    ``scanner.validation.experiment_registry``) for the "🧪 Backtest"
    tab's trial-history table — every recorded backtest/validation attempt,
    most recent first, so the dashboard shows honestly how many trials
    preceded any given result (the same count the Deflated Sharpe Ratio in
    ``/api/backtest/report`` is corrected against), not just the winner.

    Returns ``{"available": False}`` rather than a 404/500 if no registry
    database has been created yet (nothing has ever been recorded)."""
    path = Path(_experiment_registry_path())
    if not path.exists():
        return {"available": False}
    try:
        conn = init_experiment_registry(str(path), timeout=5.0)
        try:
            experiments = fetch_experiments(conn, limit=limit)
        finally:
            conn.close()
    except sqlite3.Error as exc:
        raise HTTPException(status_code=500, detail=f"Could not read experiment registry: {exc}") from None
    return {"available": True, "experiments": experiments}


@app.post("/api/backtest/refresh")
def backtest_refresh() -> dict[str, Any]:
    """Explicit, on-demand only (same spirit as ``/api/analyze/{symbol}``):
    launches ``tool/run_full_portfolio_backtest.sh`` -- the one canonical
    command that combines the point-in-time universe filter, real funding
    cost, per-symbol cost tiers, and OOS validation together (see that
    script and PHASE8_PROFITABILITY_RESEARCH_TODO.md for why this one
    command exists instead of hand-typed flags) -- as a detached background
    process and returns immediately. ``nice``+``cpulimit`` throttle it to
    ~30% of one CPU so it doesn't starve the scanner/dashboard on the VPS's
    limited vCPU budget; falls back to un-throttled (nice only) if
    ``cpulimit`` isn't installed, rather than failing the request.

    Refuses to start a second run while one is already in progress (checked
    via the PID recorded in ``_backtest_refresh_status_path()``). The
    frontend polls ``GET /api/backtest/refresh/status`` for progress and
    reloads the report once the state flips to "done"."""
    status = _read_refresh_status()
    if _refresh_in_progress(status):
        return {"started": False, "reason": "already_running", "status": status}

    script_path = _backtest_refresh_script_path()
    if not script_path.exists():
        raise HTTPException(status_code=500, detail=f"Backtest script not found: {script_path}")

    log_path = Path(_backtest_refresh_log_path())
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(UTC).isoformat()

    with open(log_path, "a", encoding="utf-8") as log_fh:
        log_fh.write(f"\n=== refresh triggered via dashboard button {started_at} ===\n")
        log_fh.flush()
        try:
            proc = subprocess.Popen(
                ["nice", "-n", "19", "bash", str(script_path)],
                cwd=str(script_path.parent.parent),
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"Could not launch backtest: {exc}") from None

        cpulimit_bin = shutil.which("cpulimit")
        if cpulimit_bin:
            subprocess.Popen(
                [cpulimit_bin, "-p", str(proc.pid), "-l", "30", "-b"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )

    status = {
        "state": "running",
        "pid": proc.pid,
        "started_at": started_at,
        "finished_at": None,
        "log_path": str(log_path),
    }
    _write_refresh_status(status)
    return {"started": True, "status": status}


@app.get("/api/backtest/refresh/status")
def backtest_refresh_status() -> dict[str, Any]:
    """Polled by the "🔄 Osvježi" button's status text while a refresh is
    running. There's no separate watcher process -- the transition out of
    "running" happens lazily, the first time anyone calls this endpoint
    after the backgrounded process has actually exited. Distinguishes
    "done" from "error" by checking whether ``report.json``'s mtime moved
    forward after the run started (the script can exit 0 having skipped
    writing the report on, e.g., every symbol failing to fetch)."""
    status = _read_refresh_status()
    if status.get("state") == "running" and isinstance(status.get("pid"), int) and not _pid_is_running(status["pid"]):
        report_path = Path(_backtest_report_path())
        succeeded = False
        started_at = status.get("started_at")
        if report_path.exists() and started_at:
            try:
                started_dt = datetime.fromisoformat(started_at)
                mtime = datetime.fromtimestamp(report_path.stat().st_mtime, tz=UTC)
                succeeded = mtime >= started_dt
            except (OSError, ValueError):
                succeeded = False
        status["state"] = "done" if succeeded else "error"
        status["finished_at"] = datetime.now(UTC).isoformat()
        _write_refresh_status(status)
    return status


@app.get("/api/forward-tracking/summary")
def forward_tracking_summary(
    limit: int = 500, range: str | None = None, since: str | None = None, until: str | None = None
) -> dict[str, Any]:
    """Closed-loop validation: compares each Top Signal Now candidate's
    *promised* calibration win rate against what it actually, live did once
    resolved (hit its stop, its target, or timed out) — not just its own
    historical backtest. See ``scanner.signal_tracking`` for how these rows
    get logged and resolved every scan cycle.

    Resolved-only statistics are reported separately from the still-open
    count: an in-progress signal's eventual outcome is unknown, so counting
    it as a loss (or a win) either way would quietly bias the comparison.

    ``range``/``since``/``until`` filter to a time window on ``logged_at``
    (when the trade was opened) — see ``_resolve_time_range``."""
    since_iso, until_iso = _resolve_time_range(range, since, until)
    conn = _get_conn()
    try:
        outcomes = fetch_recent_signal_outcomes(conn, limit=limit, since=since_iso, until=until_iso)
    finally:
        conn.close()

    open_count = sum(1 for o in outcomes if o["status"] == "open")
    resolved = [o for o in outcomes if o["status"] != "open"]
    by_status = {"take_profit": 0, "stop_loss": 0, "timeout": 0, "tp1_be": 0, "tp2_be": 0, "full_run": 0}
    for o in resolved:
        by_status[o["status"]] = by_status.get(o["status"], 0) + 1

    realized_win_rate_pct = None
    avg_promised_win_rate_ci_low_pct = None
    if resolved:
        wins = sum(1 for o in resolved if o["realized_return_pct"] is not None and o["realized_return_pct"] > 0)
        realized_win_rate_pct = round(wins / len(resolved) * 100, 1)
        promised = [o["promised_win_rate_ci_low_pct"] for o in resolved if o["promised_win_rate_ci_low_pct"] is not None]
        if promised:
            avg_promised_win_rate_ci_low_pct = round(sum(promised) / len(promised), 1)

    return {
        "open_count": open_count,
        "resolved_count": len(resolved),
        "resolved_by_status": by_status,
        "realized_win_rate_pct": realized_win_rate_pct,
        "avg_promised_win_rate_ci_low_pct": avg_promised_win_rate_ci_low_pct,
        "recent": outcomes[:20],
    }


@app.get("/api/forward-tracking/trades")
def forward_tracking_trades(
    status: str | None = None,
    limit: int = 1000,
    range: str | None = None,
    since: str | None = None,
    until: str | None = None,
) -> dict[str, Any]:
    """Every individual tracked signal (open and resolved), with full
    entry/SL/TP/exit detail — the per-trade data behind the summary counts
    shown by ``/api/forward-tracking/summary``. Powers the "Analitika
    trejdova" tab so a user can see exactly *which* symbol/timeframe/
    direction produced each take-profit, stop-loss, or timeout, not just an
    aggregate count. Each row also carries ``pattern``, ``quality_score``,
    ``regime`` and ``regime_fallback`` (Phases 3-5) so a resolved trade's
    outcome can be read against the features that ranked it, not just its
    (timeframe, direction) bucket.

    ``status``, if given, filters to one of ``open``, ``take_profit``,
    ``stop_loss``, ``timeout``, or -- for multi-target (4h) signals, see
    ``scanner.signal_tracking.MULTI_TARGET_TIMEFRAMES`` -- ``tp1_be``
    (TP1 banked, then stopped at breakeven), ``tp2_be`` (TP1+TP2 banked,
    then stopped at the TP1 price) or ``full_run`` (all three targets hit).
    Unknown values are ignored (no filter applied) rather than erroring,
    since this is a convenience filter, not a strict API contract.

    ``range``/``since``/``until`` filter to a time window on ``logged_at``
    (when the trade was opened) — see ``_resolve_time_range``."""
    since_iso, until_iso = _resolve_time_range(range, since, until)
    conn = _get_conn()
    try:
        outcomes = fetch_recent_signal_outcomes(conn, limit=limit, since=since_iso, until=until_iso)
    finally:
        conn.close()

    valid_statuses = {"open", "take_profit", "stop_loss", "timeout", "tp1_be", "tp2_be", "full_run"}
    if status in valid_statuses:
        outcomes = [o for o in outcomes if o["status"] == status]

    return {"trades": outcomes, "count": len(outcomes)}


@app.get("/api/forward-tracking/by-bucket")
def forward_tracking_by_bucket(
    limit: int = 2000, range: str | None = None, since: str | None = None, until: str | None = None
) -> dict[str, Any]:
    """Resolved trade outcomes grouped by ``(timeframe, direction)`` bucket
    — lets a user see whether the aggregate win-rate/net-return picture from
    ``/api/forward-tracking/summary`` is spread evenly across buckets or
    driven (positively or negatively) by just one or two of them, which a
    single combined average can hide.

    ``range``/``since``/``until`` filter to a time window on ``logged_at``
    (when the trade was opened) — see ``_resolve_time_range``."""
    since_iso, until_iso = _resolve_time_range(range, since, until)
    conn = _get_conn()
    try:
        outcomes = fetch_recent_signal_outcomes(conn, limit=limit, since=since_iso, until=until_iso)
    finally:
        conn.close()

    resolved = [o for o in outcomes if o["status"] != "open"]
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for o in resolved:
        buckets.setdefault((o["timeframe"], o["direction"]), []).append(o)

    rows = []
    for (timeframe, direction), items in buckets.items():
        rows.append({"timeframe": timeframe, "direction": direction, **_bucket_stats(items)})
    rows.sort(key=lambda r: r["resolved_count"], reverse=True)
    return {"buckets": rows}


@app.get("/api/forward-tracking/by-pattern")
def forward_tracking_by_pattern(
    limit: int = 2000, range: str | None = None, since: str | None = None, until: str | None = None
) -> dict[str, Any]:
    """Resolved trade outcomes grouped by TRENDING/BREAKOUT/REVERSAL pattern
    classification (Phase 4) — empirically answers "does a BREAKOUT-pattern
    signal actually realize a better win rate than a REVERSAL one", instead
    of assuming the classification matters just because it exists. Trades
    logged before the ``pattern`` column existed have ``pattern is None``
    and are grouped under ``"unknown"``."""
    since_iso, until_iso = _resolve_time_range(range, since, until)
    conn = _get_conn()
    try:
        outcomes = fetch_recent_signal_outcomes(conn, limit=limit, since=since_iso, until=until_iso)
    finally:
        conn.close()

    resolved = [o for o in outcomes if o["status"] != "open"]
    buckets: dict[str, list[dict[str, Any]]] = {}
    for o in resolved:
        buckets.setdefault(o.get("pattern") or "unknown", []).append(o)

    rows = [{"pattern": pattern, **_bucket_stats(items)} for pattern, items in buckets.items()]
    rows.sort(key=lambda r: r["resolved_count"], reverse=True)
    return {"patterns": rows}


@app.get("/api/forward-tracking/by-regime")
def forward_tracking_by_regime(
    limit: int = 2000, range: str | None = None, since: str | None = None, until: str | None = None
) -> dict[str, Any]:
    """Resolved trade outcomes grouped by volatility regime (Phase 5:
    ``mirno``/``normalno``/``volatilno``) *and* split by whether that
    signal's win rate was regime-matched or fell back to the all-regime
    blend — empirically answers the question Phase 5 exists to answer:
    does regime-matched calibration actually pay off versus the blend, not
    just in theory."""
    since_iso, until_iso = _resolve_time_range(range, since, until)
    conn = _get_conn()
    try:
        outcomes = fetch_recent_signal_outcomes(conn, limit=limit, since=since_iso, until=until_iso)
    finally:
        conn.close()

    resolved = [o for o in outcomes if o["status"] != "open"]
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for o in resolved:
        regime = o.get("regime") or "unknown"
        match_status = "unknown"
        if o.get("regime_fallback") is not None:
            match_status = "blend" if o["regime_fallback"] else "regime-matched"
        buckets.setdefault((regime, match_status), []).append(o)

    rows = [
        {"regime": regime, "match_status": match_status, **_bucket_stats(items)}
        for (regime, match_status), items in buckets.items()
    ]
    rows.sort(key=lambda r: r["resolved_count"], reverse=True)
    return {"regimes": rows}


@app.get("/api/forward-tracking/by-quality")
def forward_tracking_by_quality(
    limit: int = 2000, range: str | None = None, since: str | None = None, until: str | None = None
) -> dict[str, Any]:
    """Resolved trade outcomes grouped by Signal Quality Score tier
    (Phase 3) — the single most important empirical check in this whole
    analytics tab: does a higher Quality Score actually realize a better
    win rate, or is it just a nicer-looking number? Trades logged before
    the ``quality_score`` column existed are grouped under ``"unknown"``."""
    since_iso, until_iso = _resolve_time_range(range, since, until)
    conn = _get_conn()
    try:
        outcomes = fetch_recent_signal_outcomes(conn, limit=limit, since=since_iso, until=until_iso)
    finally:
        conn.close()

    resolved = [o for o in outcomes if o["status"] != "open"]
    buckets: dict[str, list[dict[str, Any]]] = {}
    for o in resolved:
        bucket = _quality_bucket(o.get("quality_score")) or "unknown"
        buckets.setdefault(bucket, []).append(o)

    rows = [{"quality_bucket": bucket, **_bucket_stats(items)} for bucket, items in buckets.items()]
    rows.sort(key=lambda r: r["resolved_count"], reverse=True)
    return {"quality_buckets": rows}


def _regime_matched_calibration(
    symbol: str, timeframe: str, labels: list[str], horizon_candles: int
) -> tuple[list[Any], str | None, bool]:
    """Calibration matches scoped to the symbol's *current* volatility regime
    when possible, falling back to the all-regime blend when there isn't
    enough regime-specific history yet (new/thin label) or the regime can't
    be determined. Returns ``(matches, regime, used_fallback)`` so the caller
    can tell the UI whether a result is regime-matched or a blended fallback.
    """
    from agents_crypto.calibration_lookup import find_regime_matched_stats
    from scanner.calibration import fetch_current_regime

    regime = fetch_current_regime(symbol, timeframe)
    matches, used_fallback = find_regime_matched_stats(symbol, timeframe, labels, horizon_candles, regime)
    return matches, regime, used_fallback


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

    matches, regime, used_fallback = _regime_matched_calibration(symbol, timeframe, labels, horizon_candles)

    from agents_crypto.calibration_lookup import find_oos_for_labels, get_oos_validation
    from tradingagents.dataflows.errors import VendorError

    try:
        oos_results = get_oos_validation(symbol, timeframe, horizons=(horizon_candles,), use_process=True)
    except VendorError:
        # Same graceful-degrade pattern as _regime_matched_calibration's
        # regime lookup: a transient data-vendor failure here shouldn't
        # 500 the whole endpoint, it should just mean "no OOS data yet".
        oos_results = []
    oos_matches = find_oos_for_labels(oos_results, labels, horizon_candles)

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "horizon_candles": horizon_candles,
        "active_labels": labels,
        "matches": [m.__dict__ for m in matches],
        "current_regime": regime,
        "regime_fallback": used_fallback,
        "out_of_sample": [r.__dict__ for r in oos_matches],
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

    matches, regime, used_fallback = _regime_matched_calibration(symbol, lookup_timeframe, labels, horizon_candles)
    summary["calibration_matches"] = [m.__dict__ for m in matches]
    summary["current_regime"] = regime
    summary["regime_fallback"] = used_fallback
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
        "sentiment": result.sentiment,
    }



# Static dashboard (HTML/CSS/JS) — mounted last so it doesn't shadow /api/*.
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))
