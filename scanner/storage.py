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
"""


def init_db(storage_path: str) -> sqlite3.Connection:
    path = Path(storage_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


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
