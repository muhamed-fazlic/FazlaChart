"""SQLite persistence for scanner output.

Every full cycle writes one row per symbol: the raw structured snapshot, the
anomalies that fired, and the reference close price. This is intentionally
dumb storage with no interpretation — its purpose is to let a later process
join "what the scanner saw at time T" against "what price actually did after
T" so prediction/agent calibration (point B of the project brief) has
ground truth to check against, instead of trusting a model's self-reported
confidence.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS scans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    scanned_at TEXT NOT NULL,
    reference_timeframe TEXT NOT NULL,
    close_price REAL NOT NULL,
    trend TEXT NOT NULL,
    anomalies_json TEXT NOT NULL,
    snapshot_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_scans_symbol_time ON scans (symbol, scanned_at);

-- Closed-loop validation: every time a symbol+direction first qualifies for
-- the Top Signal Now panel, one row is logged here with exactly the entry/
-- SL/TP and calibration-promised win rate a user would have seen at that
-- moment. A background resolver then watches it every scan cycle and marks
-- it take_profit/stop_loss/timeout once its outcome is known, so the
-- promised win rate can be checked against what the signal actually did
-- live — not just against its own historical backtest.
CREATE TABLE IF NOT EXISTS signal_outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    direction TEXT NOT NULL,
    label TEXT NOT NULL,
    logged_at TEXT NOT NULL,
    entry_price REAL NOT NULL,
    stop_loss REAL NOT NULL,
    take_profit REAL NOT NULL,
    take_profit_conservative REAL,
    promised_win_rate_pct REAL,
    promised_win_rate_ci_low_pct REAL,
    promised_n INTEGER,
    horizon_candles INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    resolved_at TEXT,
    exit_price REAL,
    realized_return_pct REAL,
    -- Captured from the candidate at log time (Phases 3-5: Signal Quality
    -- Score, TRENDING/BREAKOUT/REVERSAL pattern classification, volatility
    -- regime matching) so resolved trades can be analyzed by these, not
    -- just by (timeframe, direction) -- e.g. "do high-Quality-Score signals
    -- actually realize a better win rate", not just assumed.
    pattern TEXT,
    quality_score REAL,
    regime_fallback INTEGER,
    -- Multi-target (TP1/TP2/TP3) partial-close management for timeframes
    -- whose moves run further/longer (see signal_tracking.MULTI_TARGET_TIMEFRAMES).
    -- take_profit_conservative/take_profit double as TP1/TP2; this column
    -- adds TP3. targets_hit (0-3) and effective_stop_loss track live
    -- progress: 0/original SL, 1/breakeven after TP1, 2/TP1 price after TP2
    -- -- so a trade that banks TP1 then gets stopped at breakeven is
    -- recorded as the partial win it is, not lumped in with a straight loss.
    take_profit_extended REAL,
    multi_target INTEGER NOT NULL DEFAULT 0,
    targets_hit INTEGER NOT NULL DEFAULT 0,
    effective_stop_loss REAL
);
CREATE INDEX IF NOT EXISTS idx_signal_outcomes_status ON signal_outcomes (status);
CREATE INDEX IF NOT EXISTS idx_signal_outcomes_symbol_open ON signal_outcomes (symbol, timeframe, direction, status);

-- Calibration drift guardrail: when a (timeframe, direction) bucket's
-- *live, realized* win rate falls well short of what it promised over
-- enough resolved signals, it's muted here so the scanner stops
-- recommending it until the live evidence improves (see
-- scanner.drift_detector). This is never touched by the calibration
-- backtest itself -- only by what actually happened to real tracked
-- signals in `signal_outcomes`.
CREATE TABLE IF NOT EXISTS muted_buckets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timeframe TEXT NOT NULL,
    direction TEXT NOT NULL,
    muted_at TEXT NOT NULL,
    unmuted_at TEXT,
    reason TEXT NOT NULL,
    realized_win_rate_pct REAL,
    promised_win_rate_ci_low_pct_avg REAL,
    n_resolved INTEGER
);
CREATE INDEX IF NOT EXISTS idx_muted_buckets_active ON muted_buckets (timeframe, direction, unmuted_at);
"""


def init_db(storage_path: str, timeout: float = 10.0) -> sqlite3.Connection:
    """Open (creating if needed) the scanner database and ensure its schema.

    WAL journal mode lets readers (the dashboard, potentially several at
    once) proceed concurrently while the scanner holds a write transaction,
    instead of the default rollback-journal mode's writer-blocks-everyone
    behaviour; ``timeout`` makes a connection retry-and-wait on a transient
    lock rather than raising ``database is locked`` immediately, which
    becomes likely the moment more than one connection touches the database
    around the same instant (e.g. a scan cycle writing while the dashboard's
    UI fires several read requests at once).
    """
    path = Path(storage_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=timeout)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_SCHEMA)
    _migrate_schema(conn)
    conn.commit()
    return conn


def _migrate_schema(conn: sqlite3.Connection) -> None:
    """Add columns introduced after a database file already existed.
    ``CREATE TABLE IF NOT EXISTS`` in ``_SCHEMA`` only creates a table the
    very first time -- it never adds a new column to a table that already
    exists on disk, so any such column needs an explicit, idempotent
    ``ALTER TABLE`` here instead."""
    columns = {row[1] for row in conn.execute("PRAGMA table_info(signal_outcomes)")}
    if "regime" not in columns:
        conn.execute("ALTER TABLE signal_outcomes ADD COLUMN regime TEXT")
    if "pattern" not in columns:
        conn.execute("ALTER TABLE signal_outcomes ADD COLUMN pattern TEXT")
    if "quality_score" not in columns:
        conn.execute("ALTER TABLE signal_outcomes ADD COLUMN quality_score REAL")
    if "regime_fallback" not in columns:
        conn.execute("ALTER TABLE signal_outcomes ADD COLUMN regime_fallback INTEGER")
    if "take_profit_extended" not in columns:
        conn.execute("ALTER TABLE signal_outcomes ADD COLUMN take_profit_extended REAL")
    if "multi_target" not in columns:
        conn.execute("ALTER TABLE signal_outcomes ADD COLUMN multi_target INTEGER NOT NULL DEFAULT 0")
    if "targets_hit" not in columns:
        conn.execute("ALTER TABLE signal_outcomes ADD COLUMN targets_hit INTEGER NOT NULL DEFAULT 0")
    if "effective_stop_loss" not in columns:
        conn.execute("ALTER TABLE signal_outcomes ADD COLUMN effective_stop_loss REAL")


def connect_for_reads(storage_path: str, timeout: float = 10.0) -> sqlite3.Connection:
    """A plain connection for read-only endpoints that doesn't repeat the
    schema-creation write statements ``init_db`` runs on every call — those
    writes are pointless once the schema already exists, and the fewer
    write-journal touches happen on a hot read path, the less lock
    contention there is against the scanner's own writer connection.

    Deliberately does NOT (re-)issue ``PRAGMA journal_mode=WAL`` here:
    switching journal mode itself needs a brief exclusive lock, and once
    some writer connection has switched the database file to WAL mode that
    setting is persisted in the file itself, so every later connection
    (read or write) automatically uses WAL without needing to ask again.
    Asking again on every single read connection only reintroduces the
    lock contention this function exists to avoid.
    """
    return sqlite3.connect(str(storage_path), timeout=timeout)


def save_scan_result(
    conn: sqlite3.Connection,
    symbol: str,
    snapshot: dict[str, Any],
    anomalies: list[str],
    reference_timeframe: str = "1h",
) -> None:
    tf_snapshot = snapshot["timeframes"].get(reference_timeframe) or next(iter(snapshot["timeframes"].values()))
    conn.execute(
        "INSERT INTO scans (symbol, scanned_at, reference_timeframe, close_price, trend, anomalies_json, snapshot_json)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            symbol,
            datetime.now(UTC).isoformat(),
            reference_timeframe,
            tf_snapshot["close"],
            tf_snapshot["trend"],
            json.dumps(anomalies),
            json.dumps(snapshot),
        ),
    )
    conn.commit()


def fetch_recent_scans(conn: sqlite3.Connection, symbol: str, limit: int = 50) -> list[dict[str, Any]]:
    """Latest scans for one symbol, most-recent first — e.g. to compute a
    realized forward return against an earlier snapshot's close price."""
    cursor = conn.execute(
        "SELECT symbol, scanned_at, reference_timeframe, close_price, trend, anomalies_json"
        " FROM scans WHERE symbol = ? ORDER BY scanned_at DESC LIMIT ?",
        (symbol, limit),
    )
    columns = [description[0] for description in cursor.description]
    return [dict(zip(columns, row, strict=False)) for row in cursor.fetchall()]


def fetch_latest_scan(conn: sqlite3.Connection, symbol: str) -> dict[str, Any] | None:
    """The single most recent scan row for one symbol, or ``None`` if it has
    never been scanned — e.g. for a cheap per-symbol lookup that doesn't need
    every symbol like :func:`fetch_latest_scan_per_symbol` does."""
    cursor = conn.execute(
        "SELECT symbol, scanned_at, reference_timeframe, close_price, trend, anomalies_json, snapshot_json"
        " FROM scans WHERE symbol = ? ORDER BY scanned_at DESC LIMIT 1",
        (symbol,),
    )
    row = cursor.fetchone()
    if row is None:
        return None
    columns = [description[0] for description in cursor.description]
    result = dict(zip(columns, row, strict=False))
    result["anomalies"] = json.loads(result.pop("anomalies_json"))
    result["snapshot"] = json.loads(result.pop("snapshot_json"))
    return result


def fetch_latest_scan_per_symbol(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """One row per symbol — its most recent scan — for a dashboard/overview.
    Includes the full parsed snapshot (indicators, derivatives, anomalies
    per timeframe), not just the flattened reference-timeframe columns."""
    cursor = conn.execute(
        """
        SELECT s.symbol, s.scanned_at, s.reference_timeframe, s.close_price, s.trend,
               s.anomalies_json, s.snapshot_json
        FROM scans s
        INNER JOIN (
            SELECT symbol, MAX(scanned_at) AS max_time FROM scans GROUP BY symbol
        ) latest ON s.symbol = latest.symbol AND s.scanned_at = latest.max_time
        ORDER BY s.symbol
        """
    )
    columns = [description[0] for description in cursor.description]
    rows = [dict(zip(columns, row, strict=False)) for row in cursor.fetchall()]
    for row in rows:
        row["anomalies"] = json.loads(row.pop("anomalies_json"))
        row["snapshot"] = json.loads(row.pop("snapshot_json"))
    return rows


def fetch_known_symbols(conn: sqlite3.Connection) -> list[str]:
    """Every symbol ever persisted, most-recently-scanned first. Used to keep
    refreshing a symbol that has since fallen out of the current top-N by
    volume, instead of letting its last snapshot go stale forever once it
    drops out of rotation."""
    cursor = conn.execute(
        "SELECT symbol FROM scans GROUP BY symbol ORDER BY MAX(scanned_at) DESC"
    )
    return [row[0] for row in cursor.fetchall()]


def fetch_scan_health(conn: sqlite3.Connection) -> dict[str, Any]:
    """The single latest ``scanned_at`` timestamp across every symbol, plus
    how many distinct symbols have ever been scanned -- the minimal facts
    needed to answer "is the scanner actually still running?" without
    pulling every row's full snapshot JSON."""
    cursor = conn.execute("SELECT MAX(scanned_at), COUNT(DISTINCT symbol) FROM scans")
    last_scanned_at, symbol_count = cursor.fetchone()
    return {"last_scanned_at": last_scanned_at, "symbols_ever_tracked": symbol_count or 0}


def fetch_symbol_history(conn: sqlite3.Connection, symbol: str, limit: int = 200) -> list[dict[str, Any]]:
    """Chronological (oldest-first) close price + trend history for one
    symbol, for a sparkline/chart on the dashboard."""
    cursor = conn.execute(
        "SELECT scanned_at, close_price, trend FROM scans"
        " WHERE symbol = ? ORDER BY scanned_at ASC LIMIT ?",
        (symbol, limit),
    )
    columns = [description[0] for description in cursor.description]
    return [dict(zip(columns, row, strict=False)) for row in cursor.fetchall()]


def prune_old_scans(conn: sqlite3.Connection, retain_days: int = 60) -> int:
    """Delete `scans` rows older than ``retain_days`` and reclaim their disk
    space, so the database doesn't grow without bound over months of 24/7
    operation. Safe to call anytime: calibration always re-derives its
    stats fresh from live Binance klines rather than this table (see
    scanner.calibration's module docstring), and every dashboard read only
    looks at recent rows (``fetch_latest_scan*``, ``fetch_symbol_history``
    with a row limit). ``signal_outcomes`` (the live forward-tracking
    record) is untouched here -- that table is the actual track record and
    stays small regardless of how long the scanner runs. Returns the number
    of rows deleted.
    """
    cutoff = (datetime.now(UTC) - timedelta(days=retain_days)).isoformat()
    cursor = conn.execute("DELETE FROM scans WHERE scanned_at < ?", (cutoff,))
    deleted = cursor.rowcount
    conn.commit()
    if deleted:
        conn.execute("VACUUM")
    return deleted


def log_signal_outcome(
    conn: sqlite3.Connection,
    *,
    symbol: str,
    timeframe: str,
    direction: str,
    label: str,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    take_profit_conservative: float | None,
    promised_win_rate_pct: float | None,
    promised_win_rate_ci_low_pct: float | None,
    promised_n: int | None,
    horizon_candles: int,
    regime: str | None = None,
    pattern: str | None = None,
    quality_score: float | None = None,
    regime_fallback: bool | None = None,
    take_profit_extended: float | None = None,
    multi_target: bool = False,
) -> None:
    """Record one new live signal to track to its outcome. Callers are
    responsible for not calling this twice for the same still-open trade
    (see :func:`fetch_open_signal_outcome`) — this function itself has no
    dedup logic, it just inserts.

    ``multi_target=True`` (set by the caller for timeframes in
    ``signal_tracking.MULTI_TARGET_TIMEFRAMES``) manages the trade through
    TP1/TP2/TP3 partial closes instead of one single exit --
    ``effective_stop_loss`` starts equal to ``stop_loss`` and moves as each
    target is banked (see ``signal_tracking.resolve_open_signals``)."""
    conn.execute(
        """
        INSERT INTO signal_outcomes (
            symbol, timeframe, direction, label, logged_at, entry_price,
            stop_loss, take_profit, take_profit_conservative,
            promised_win_rate_pct, promised_win_rate_ci_low_pct, promised_n,
            horizon_candles, status, regime, pattern, quality_score, regime_fallback,
            take_profit_extended, multi_target, targets_hit, effective_stop_loss
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?, ?, ?, ?, ?, ?, 0, ?)
        """,
        (
            symbol,
            timeframe,
            direction,
            label,
            datetime.now(UTC).isoformat(),
            entry_price,
            stop_loss,
            take_profit,
            take_profit_conservative,
            promised_win_rate_pct,
            promised_win_rate_ci_low_pct,
            promised_n,
            horizon_candles,
            regime,
            pattern,
            quality_score,
            None if regime_fallback is None else int(regime_fallback),
            take_profit_extended,
            int(multi_target),
            stop_loss,
        ),
    )
    conn.commit()


def update_signal_outcome_progress(
    conn: sqlite3.Connection, outcome_id: int, *, targets_hit: int, effective_stop_loss: float
) -> None:
    """Persist multi-target partial-close progress without resolving the
    trade -- called when a TP1/TP2 is banked (see
    ``signal_tracking.resolve_open_signals``) so the next scan cycle checks
    the moved stop and the next un-banked target, and the trade stays
    ``open`` until the final target or the (now moved) stop is hit."""
    conn.execute(
        "UPDATE signal_outcomes SET targets_hit = ?, effective_stop_loss = ? WHERE id = ?",
        (targets_hit, effective_stop_loss, outcome_id),
    )
    conn.commit()


def fetch_open_signal_outcome(
    conn: sqlite3.Connection, symbol: str, timeframe: str, direction: str
) -> dict[str, Any] | None:
    """The still-open tracked signal for this symbol+timeframe+direction, if
    any — used to avoid logging a duplicate row every scan cycle while the
    same opportunity keeps qualifying for the Top Signal Now panel. Only one
    open trade is tracked per (symbol, timeframe, direction) at a time."""
    cursor = conn.execute(
        "SELECT * FROM signal_outcomes WHERE symbol = ? AND timeframe = ? AND direction = ? AND status = 'open'"
        " ORDER BY logged_at DESC LIMIT 1",
        (symbol, timeframe, direction),
    )
    row = cursor.fetchone()
    if row is None:
        return None
    columns = [description[0] for description in cursor.description]
    return dict(zip(columns, row, strict=False))


def fetch_open_signal_outcomes(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Every currently-open tracked signal, for the resolver to check each
    scan cycle against the latest known price."""
    cursor = conn.execute("SELECT * FROM signal_outcomes WHERE status = 'open'")
    columns = [description[0] for description in cursor.description]
    return [dict(zip(columns, row, strict=False)) for row in cursor.fetchall()]


def resolve_signal_outcome(
    conn: sqlite3.Connection, outcome_id: int, status: str, exit_price: float, realized_return_pct: float
) -> None:
    """Close out a tracked signal with its real, observed outcome."""
    conn.execute(
        "UPDATE signal_outcomes SET status = ?, resolved_at = ?, exit_price = ?, realized_return_pct = ?"
        " WHERE id = ?",
        (status, datetime.now(UTC).isoformat(), exit_price, realized_return_pct, outcome_id),
    )
    conn.commit()


def fetch_recent_signal_outcomes(
    conn: sqlite3.Connection,
    limit: int = 200,
    since: str | None = None,
    until: str | None = None,
) -> list[dict[str, Any]]:
    """Most-recently-logged tracked signals (open and resolved), for a
    live-vs-promised win-rate comparison endpoint. ``since``/``until`` are
    optional ISO-8601 timestamps that filter on ``logged_at`` (when the
    trade was opened), used for the dashboard's time-range analytics
    filter -- e.g. "only trades opened in the last 4 hours"."""
    clauses = []
    params: list[Any] = []
    if since is not None:
        clauses.append("logged_at >= ?")
        params.append(since)
    if until is not None:
        clauses.append("logged_at <= ?")
        params.append(until)
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    params.append(limit)
    cursor = conn.execute(f"SELECT * FROM signal_outcomes{where} ORDER BY logged_at DESC LIMIT ?", params)
    columns = [description[0] for description in cursor.description]
    return [dict(zip(columns, row, strict=False)) for row in cursor.fetchall()]


def fetch_resolved_outcomes_by_bucket(
    conn: sqlite3.Connection, timeframe: str, direction: str, limit: int = 60
) -> list[dict[str, Any]]:
    """The most recently *resolved* (``take_profit``/``stop_loss``/
    ``timeout``) tracked signals for one ``(timeframe, direction)`` bucket,
    newest first. Capped at ``limit`` so the drift check looks at a rolling
    recent window rather than all-time history -- a bucket that was bad six
    months ago but has since recovered should read as recovered, not stay
    judged by stale evidence forever."""
    cursor = conn.execute(
        "SELECT * FROM signal_outcomes WHERE timeframe = ? AND direction = ? AND status != 'open'"
        " ORDER BY resolved_at DESC LIMIT ?",
        (timeframe, direction, limit),
    )
    columns = [description[0] for description in cursor.description]
    return [dict(zip(columns, row, strict=False)) for row in cursor.fetchall()]


def fetch_distinct_resolved_buckets(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    """Every ``(timeframe, direction)`` combination that has at least one
    resolved tracked signal -- the universe of buckets the drift detector
    needs to evaluate."""
    cursor = conn.execute("SELECT DISTINCT timeframe, direction FROM signal_outcomes WHERE status != 'open'")
    return [(row[0], row[1]) for row in cursor.fetchall()]


def fetch_active_muted_bucket(conn: sqlite3.Connection, timeframe: str, direction: str) -> dict[str, Any] | None:
    """The currently-active mute row for this bucket, if any (``unmuted_at``
    still ``NULL``)."""
    cursor = conn.execute(
        "SELECT * FROM muted_buckets WHERE timeframe = ? AND direction = ? AND unmuted_at IS NULL"
        " ORDER BY muted_at DESC LIMIT 1",
        (timeframe, direction),
    )
    row = cursor.fetchone()
    if row is None:
        return None
    columns = [description[0] for description in cursor.description]
    return dict(zip(columns, row, strict=False))


def fetch_all_active_muted_buckets(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Every bucket currently muted -- used by
    :func:`agents_crypto.best_signal.compute_candidates` (via its callers)
    to filter them out of Top Signal Now, and by the dashboard to show the
    user which buckets are currently suppressed and why."""
    cursor = conn.execute("SELECT * FROM muted_buckets WHERE unmuted_at IS NULL ORDER BY muted_at DESC")
    columns = [description[0] for description in cursor.description]
    return [dict(zip(columns, row, strict=False)) for row in cursor.fetchall()]


def mute_bucket(
    conn: sqlite3.Connection,
    *,
    timeframe: str,
    direction: str,
    reason: str,
    realized_win_rate_pct: float,
    promised_win_rate_ci_low_pct_avg: float,
    n_resolved: int,
) -> None:
    """Suppress a ``(timeframe, direction)`` bucket from being recommended.
    Callers must check :func:`fetch_active_muted_bucket` first so a bucket
    already muted doesn't get a second, redundant active row."""
    conn.execute(
        """
        INSERT INTO muted_buckets (
            timeframe, direction, muted_at, reason,
            realized_win_rate_pct, promised_win_rate_ci_low_pct_avg, n_resolved
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            timeframe,
            direction,
            datetime.now(UTC).isoformat(),
            reason,
            realized_win_rate_pct,
            promised_win_rate_ci_low_pct_avg,
            n_resolved,
        ),
    )
    conn.commit()


def unmute_bucket(conn: sqlite3.Connection, mute_id: int) -> None:
    """Mark a mute row as lifted (the bucket's recent live performance has
    recovered)."""
    conn.execute(
        "UPDATE muted_buckets SET unmuted_at = ? WHERE id = ?",
        (datetime.now(UTC).isoformat(), mute_id),
    )
    conn.commit()
