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
from datetime import UTC, datetime
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
    realized_return_pct REAL
);
CREATE INDEX IF NOT EXISTS idx_signal_outcomes_status ON signal_outcomes (status);
CREATE INDEX IF NOT EXISTS idx_signal_outcomes_symbol_open ON signal_outcomes (symbol, timeframe, direction, status);
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
    conn.commit()
    return conn


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
) -> None:
    """Record one new live signal to track to its outcome. Callers are
    responsible for not calling this twice for the same still-open trade
    (see :func:`fetch_open_signal_outcome`) — this function itself has no
    dedup logic, it just inserts."""
    conn.execute(
        """
        INSERT INTO signal_outcomes (
            symbol, timeframe, direction, label, logged_at, entry_price,
            stop_loss, take_profit, take_profit_conservative,
            promised_win_rate_pct, promised_win_rate_ci_low_pct, promised_n,
            horizon_candles, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open')
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
        ),
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


def fetch_recent_signal_outcomes(conn: sqlite3.Connection, limit: int = 200) -> list[dict[str, Any]]:
    """Most-recently-logged tracked signals (open and resolved), for a
    live-vs-promised win-rate comparison endpoint."""
    cursor = conn.execute("SELECT * FROM signal_outcomes ORDER BY logged_at DESC LIMIT ?", (limit,))
    columns = [description[0] for description in cursor.description]
    return [dict(zip(columns, row, strict=False)) for row in cursor.fetchall()]
