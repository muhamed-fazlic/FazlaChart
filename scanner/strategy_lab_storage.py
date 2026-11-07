"""SQLite persistence for the Strategy Lab feature (Layer 1 continuous
mining + Layer 2 power-gated confirmation), following the exact same
pattern as ``scanner.storage`` (WAL mode, ``init_db``/``connect_for_reads``
split, idempotent ``CREATE TABLE IF NOT EXISTS`` schema plus an explicit
``_migrate_schema`` for later column additions) rather than inventing a new
DB-access style.

Deliberately a SEPARATE database file (``scanner_data/strategy_lab.db``,
not ``scanner_data/scanner.db``) -- this data has nothing to do with the
live scanner's own signal tracking, and keeping it separate means an
import/export or a `rm` during development can never accidentally touch
the live scanner's ground-truth data.

Four tables, one per concern:

* ``mining_candidates`` -- Layer 1's continuously-updated, purely
  descriptive leaderboard. One row per (template, tercile, variant,
  mining run). Never carries a pass/fail verdict.
* ``weekly_snapshots`` -- the #1-ranked candidate's full backtest-engine
  stats, recomputed weekly on train-only data. Explicitly exploratory
  (selection bias from Layer 1's ranking).
* ``confirmations`` -- Layer 2's rare, power-gated, REAL confirmatory runs
  against fresh (never-before-used) calendar time. The ONLY place in this
  whole feature allowed to carry ``edge_survives_lower_bound`` as a
  genuine pass/fail verdict.
* ``rolling_holdout_state`` -- single-row table tracking the
  forward-advancing boundary of calendar time not yet consumed by any
  confirmation (see ``scanner.backtesting.rolling_holdout``).
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_STRATEGY_LAB_DB_PATH = "scanner_data/strategy_lab.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS mining_candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    mined_at TEXT NOT NULL,
    as_of_timestamp_ms INTEGER NOT NULL,
    template TEXT NOT NULL,
    tercile TEXT NOT NULL,
    variant_label TEXT NOT NULL,
    params_json TEXT NOT NULL,
    n_independent INTEGER NOT NULL,
    gross_observed_pct REAL,
    net_observed_pct REAL,
    stability_score REAL,
    composite_score REAL NOT NULL,
    report_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_mining_candidates_run ON mining_candidates (mined_at);
CREATE INDEX IF NOT EXISTS idx_mining_candidates_as_of ON mining_candidates (as_of_timestamp_ms);
CREATE INDEX IF NOT EXISTS idx_mining_candidates_score ON mining_candidates (composite_score DESC);

CREATE TABLE IF NOT EXISTS weekly_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_at TEXT NOT NULL,
    candidate_id INTEGER,
    template TEXT NOT NULL,
    tercile TEXT NOT NULL,
    variant_label TEXT NOT NULL,
    n_independent INTEGER NOT NULL,
    net_observed_pct REAL,
    net_ci_low_pct REAL,
    net_ci_high_pct REAL,
    gross_observed_pct REAL,
    win_rate_pct REAL,
    telegram_sent INTEGER NOT NULL DEFAULT 0,
    report_json TEXT NOT NULL,
    FOREIGN KEY (candidate_id) REFERENCES mining_candidates (id)
);
CREATE INDEX IF NOT EXISTS idx_weekly_snapshots_time ON weekly_snapshots (snapshot_at);

CREATE TABLE IF NOT EXISTS confirmations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    attempted_at TEXT NOT NULL,
    candidate_id INTEGER,
    template TEXT NOT NULL,
    tercile TEXT NOT NULL,
    variant_label TEXT NOT NULL,
    fresh_window_start_ms INTEGER NOT NULL,
    fresh_window_end_ms INTEGER NOT NULL,
    n_independent INTEGER NOT NULL,
    n_required INTEGER NOT NULL,
    net_observed_pct REAL,
    net_ci_low_pct REAL,
    net_ci_high_pct REAL,
    edge_survives_lower_bound INTEGER NOT NULL,
    is_current_best INTEGER NOT NULL DEFAULT 0,
    telegram_sent INTEGER NOT NULL DEFAULT 0,
    report_json TEXT NOT NULL,
    FOREIGN KEY (candidate_id) REFERENCES mining_candidates (id)
);
CREATE INDEX IF NOT EXISTS idx_confirmations_time ON confirmations (attempted_at);
CREATE INDEX IF NOT EXISTS idx_confirmations_current_best ON confirmations (is_current_best);

-- Single-row table (id always 1) -- see rolling_holdout.py for the
-- monotonic-advancement guarantee this state backs.
CREATE TABLE IF NOT EXISTS rolling_holdout_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    boundary_timestamp_ms INTEGER NOT NULL,
    updated_at TEXT NOT NULL
);
"""


def init_db(storage_path: str = DEFAULT_STRATEGY_LAB_DB_PATH, timeout: float = 10.0) -> sqlite3.Connection:
    """Opens (creating if needed) the Strategy Lab database and ensures its
    schema -- same WAL-mode/timeout rationale as ``scanner.storage.init_db``."""
    path = Path(storage_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=timeout)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


def connect_for_reads(storage_path: str = DEFAULT_STRATEGY_LAB_DB_PATH, timeout: float = 10.0) -> sqlite3.Connection:
    """Plain read connection for the dashboard API endpoint -- mirrors
    ``scanner.storage.connect_for_reads``' no-schema-write rationale."""
    return sqlite3.connect(str(storage_path), timeout=timeout)


def database_exists(storage_path: str = DEFAULT_STRATEGY_LAB_DB_PATH) -> bool:
    """Whether the database file is already on disk -- lets the dashboard
    endpoint return a graceful ``{"available": False}`` before Layer 1 has
    ever run a mining cycle, instead of ``init_db`` silently creating an
    empty database just because the endpoint was polled."""
    return Path(storage_path).exists()


def insert_mining_candidate(
    conn: sqlite3.Connection,
    *,
    as_of_timestamp_ms: int,
    template: str,
    tercile: str,
    variant_label: str,
    params: dict[str, Any],
    n_independent: int,
    gross_observed_pct: float | None,
    net_observed_pct: float | None,
    stability_score: float | None,
    composite_score: float,
    report: dict[str, Any],
) -> int:
    cursor = conn.execute(
        "INSERT INTO mining_candidates (mined_at, as_of_timestamp_ms, template, tercile, variant_label,"
        " params_json, n_independent, gross_observed_pct, net_observed_pct, stability_score, composite_score,"
        " report_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            datetime.now(UTC).isoformat(),
            as_of_timestamp_ms,
            template,
            tercile,
            variant_label,
            json.dumps(params),
            n_independent,
            gross_observed_pct,
            net_observed_pct,
            stability_score,
            composite_score,
            json.dumps(report),
        ),
    )
    conn.commit()
    return int(cursor.lastrowid)


def fetch_leaderboard(conn: sqlite3.Connection, limit: int = 10) -> list[dict[str, Any]]:
    """Top ``limit`` candidates from the MOST RECENT mining run only (not
    pooled across every historical run ever stored), ranked by
    ``composite_score`` descending.

    Grouped by ``as_of_timestamp_ms`` (the single value shared by every
    candidate within one mining cycle), NOT ``mined_at`` -- ``mined_at`` is
    ``datetime.now(UTC).isoformat()`` captured individually per row inside
    ``insert_mining_candidate``'s insert loop, so it differs (down to the
    microsecond) between candidates of the SAME cycle. Grouping by it was a
    real bug: ``WHERE mined_at = MAX(mined_at)`` only ever matched the
    single last-inserted row of the latest cycle, not all of that cycle's
    candidates, making this table silently show just one row instead of
    one per (template, tercile, variant) combination actually mined."""
    latest = conn.execute("SELECT MAX(as_of_timestamp_ms) FROM mining_candidates").fetchone()[0]
    if latest is None:
        return []
    cursor = conn.execute(
        "SELECT id, mined_at, template, tercile, variant_label, params_json, n_independent,"
        " gross_observed_pct, net_observed_pct, stability_score, composite_score"
        " FROM mining_candidates WHERE as_of_timestamp_ms = ? ORDER BY composite_score DESC LIMIT ?",
        (latest, limit),
    )
    columns = [d[0] for d in cursor.description]
    rows = [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]
    for row in rows:
        row["params"] = json.loads(row.pop("params_json"))
    return rows


def fetch_mined_combo_keys(conn: sqlite3.Connection, as_of_timestamp_ms: int) -> frozenset[tuple[str, str, str]]:
    """``(template, tercile, variant_label)`` combinations already stored
    for this exact ``as_of_timestamp_ms``. The scheduler uses this to skip
    re-mining a combination whose result is guaranteed to be byte-identical
    to what's already stored -- ``as_of_ts`` is clamped to a frozen
    train/holdout boundary once real time passes it, so without this check
    every scheduled cycle re-runs (and re-inserts) the exact same
    combinations forever, wasting CPU and filling the leaderboard with
    duplicate rows (a real bug this function exists to prevent)."""
    cursor = conn.execute(
        "SELECT DISTINCT template, tercile, variant_label FROM mining_candidates WHERE as_of_timestamp_ms = ?",
        (as_of_timestamp_ms,),
    )
    return frozenset((row[0], row[1], row[2]) for row in cursor.fetchall())


def fetch_total_candidates_tested(conn: sqlite3.Connection) -> int:
    """Total row count in ``mining_candidates`` across EVERY mining cycle
    ever run (not just the most recent one -- that's ``fetch_leaderboard``'s
    job). Purely a "how much exploring has Layer 1 done so far" counter,
    not a statistical statement."""
    return int(conn.execute("SELECT COUNT(*) FROM mining_candidates").fetchone()[0])


def _candidate_all_time_rank(conn: sqlite3.Connection, composite_score: float) -> int:
    """1-based rank of ``composite_score`` among every candidate ever
    mined (higher score = better rank), ties broken arbitrarily by SQLite's
    row order (acceptable here -- this is a descriptive position, not a
    statistical claim)."""
    better_count = conn.execute(
        "SELECT COUNT(*) FROM mining_candidates WHERE composite_score > ?", (composite_score,)
    ).fetchone()[0]
    return int(better_count) + 1


def fetch_latest_candidate(conn: sqlite3.Connection) -> dict[str, Any] | None:
    """The single most-recently-inserted row in ``mining_candidates``
    (highest ``id``), with its all-time rank by ``composite_score``
    attached -- lets the dashboard show "what did Layer 1 just test, and
    how does it compare to the best ever found" without a second
    round-trip."""
    cursor = conn.execute(
        "SELECT id, mined_at, as_of_timestamp_ms, template, tercile, variant_label, params_json,"
        " n_independent, gross_observed_pct, net_observed_pct, stability_score, composite_score"
        " FROM mining_candidates ORDER BY id DESC LIMIT 1"
    )
    row = cursor.fetchone()
    if row is None:
        return None
    columns = [d[0] for d in cursor.description]
    result = dict(zip(columns, row, strict=True))
    result["params"] = json.loads(result.pop("params_json"))
    result["all_time_rank"] = _candidate_all_time_rank(conn, result["composite_score"])
    return result


def fetch_best_candidate_all_time(conn: sqlite3.Connection) -> dict[str, Any] | None:
    """The single highest-``composite_score`` row across EVERY mining
    cycle ever run (not just the latest one). This is still Layer 1 --
    purely descriptive, NEVER a confirmed edge. Use ``fetch_current_best``
    (Layer 2 confirmations) for the only field in this whole feature that
    carries a real pass/fail verdict."""
    cursor = conn.execute(
        "SELECT id, mined_at, as_of_timestamp_ms, template, tercile, variant_label, params_json,"
        " n_independent, gross_observed_pct, net_observed_pct, stability_score, composite_score"
        " FROM mining_candidates ORDER BY composite_score DESC LIMIT 1"
    )
    row = cursor.fetchone()
    if row is None:
        return None
    columns = [d[0] for d in cursor.description]
    result = dict(zip(columns, row, strict=True))
    result["params"] = json.loads(result.pop("params_json"))
    result["all_time_rank"] = 1
    return result


def fetch_mining_candidate(conn: sqlite3.Connection, candidate_id: int) -> dict[str, Any] | None:
    cursor = conn.execute(
        "SELECT id, mined_at, as_of_timestamp_ms, template, tercile, variant_label, params_json,"
        " n_independent, gross_observed_pct, net_observed_pct, stability_score, composite_score"
        " FROM mining_candidates WHERE id = ?",
        (candidate_id,),
    )
    row = cursor.fetchone()
    if row is None:
        return None
    columns = [d[0] for d in cursor.description]
    result = dict(zip(columns, row, strict=True))
    result["params"] = json.loads(result.pop("params_json"))
    return result


def insert_weekly_snapshot(
    conn: sqlite3.Connection,
    *,
    candidate_id: int | None,
    template: str,
    tercile: str,
    variant_label: str,
    n_independent: int,
    net_observed_pct: float | None,
    net_ci_low_pct: float | None,
    net_ci_high_pct: float | None,
    gross_observed_pct: float | None,
    win_rate_pct: float | None,
    telegram_sent: bool,
    report: dict[str, Any],
) -> int:
    cursor = conn.execute(
        "INSERT INTO weekly_snapshots (snapshot_at, candidate_id, template, tercile, variant_label,"
        " n_independent, net_observed_pct, net_ci_low_pct, net_ci_high_pct, gross_observed_pct, win_rate_pct,"
        " telegram_sent, report_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            datetime.now(UTC).isoformat(),
            candidate_id,
            template,
            tercile,
            variant_label,
            n_independent,
            net_observed_pct,
            net_ci_low_pct,
            net_ci_high_pct,
            gross_observed_pct,
            win_rate_pct,
            int(telegram_sent),
            json.dumps(report),
        ),
    )
    conn.commit()
    return int(cursor.lastrowid)


def fetch_weekly_snapshots(conn: sqlite3.Connection, limit: int = 52) -> list[dict[str, Any]]:
    cursor = conn.execute(
        "SELECT id, snapshot_at, candidate_id, template, tercile, variant_label, n_independent,"
        " net_observed_pct, net_ci_low_pct, net_ci_high_pct, gross_observed_pct, win_rate_pct, telegram_sent"
        " FROM weekly_snapshots ORDER BY snapshot_at DESC LIMIT ?",
        (limit,),
    )
    columns = [d[0] for d in cursor.description]
    return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]


def insert_confirmation(
    conn: sqlite3.Connection,
    *,
    candidate_id: int | None,
    template: str,
    tercile: str,
    variant_label: str,
    fresh_window_start_ms: int,
    fresh_window_end_ms: int,
    n_independent: int,
    n_required: int,
    net_observed_pct: float | None,
    net_ci_low_pct: float | None,
    net_ci_high_pct: float | None,
    edge_survives_lower_bound: bool,
    telegram_sent: bool,
    report: dict[str, Any],
) -> int:
    """Inserts a confirmation attempt. If ``edge_survives_lower_bound`` is
    True, clears ``is_current_best`` on every prior row first and marks
    this one -- "at most one current best" (prior passes remain in history
    with ``is_current_best = 0``, never deleted)."""
    if edge_survives_lower_bound:
        conn.execute("UPDATE confirmations SET is_current_best = 0")
    cursor = conn.execute(
        "INSERT INTO confirmations (attempted_at, candidate_id, template, tercile, variant_label,"
        " fresh_window_start_ms, fresh_window_end_ms, n_independent, n_required, net_observed_pct,"
        " net_ci_low_pct, net_ci_high_pct, edge_survives_lower_bound, is_current_best, telegram_sent,"
        " report_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            datetime.now(UTC).isoformat(),
            candidate_id,
            template,
            tercile,
            variant_label,
            fresh_window_start_ms,
            fresh_window_end_ms,
            n_independent,
            n_required,
            net_observed_pct,
            net_ci_low_pct,
            net_ci_high_pct,
            int(edge_survives_lower_bound),
            int(edge_survives_lower_bound),
            int(telegram_sent),
            json.dumps(report),
        ),
    )
    conn.commit()
    return int(cursor.lastrowid)


def fetch_confirmations(conn: sqlite3.Connection, limit: int = 100) -> list[dict[str, Any]]:
    cursor = conn.execute(
        "SELECT id, attempted_at, template, tercile, variant_label, fresh_window_start_ms,"
        " fresh_window_end_ms, n_independent, n_required, net_observed_pct, net_ci_low_pct, net_ci_high_pct,"
        " edge_survives_lower_bound, is_current_best, telegram_sent"
        " FROM confirmations ORDER BY attempted_at DESC LIMIT ?",
        (limit,),
    )
    columns = [d[0] for d in cursor.description]
    return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]


def fetch_current_best(conn: sqlite3.Connection) -> dict[str, Any] | None:
    cursor = conn.execute(
        "SELECT id, attempted_at, template, tercile, variant_label, fresh_window_start_ms,"
        " fresh_window_end_ms, n_independent, n_required, net_observed_pct, net_ci_low_pct, net_ci_high_pct,"
        " edge_survives_lower_bound, telegram_sent"
        " FROM confirmations WHERE is_current_best = 1 ORDER BY attempted_at DESC LIMIT 1"
    )
    row = cursor.fetchone()
    if row is None:
        return None
    columns = [d[0] for d in cursor.description]
    return dict(zip(columns, row, strict=True))


def fetch_rolling_holdout_state(conn: sqlite3.Connection) -> dict[str, Any] | None:
    cursor = conn.execute("SELECT boundary_timestamp_ms, updated_at FROM rolling_holdout_state WHERE id = 1")
    row = cursor.fetchone()
    if row is None:
        return None
    return {"boundary_timestamp_ms": row[0], "updated_at": row[1]}


def save_rolling_holdout_state(conn: sqlite3.Connection, boundary_timestamp_ms: int) -> None:
    conn.execute(
        "INSERT INTO rolling_holdout_state (id, boundary_timestamp_ms, updated_at) VALUES (1, ?, ?)"
        " ON CONFLICT (id) DO UPDATE SET boundary_timestamp_ms = excluded.boundary_timestamp_ms,"
        " updated_at = excluded.updated_at",
        (boundary_timestamp_ms, datetime.now(UTC).isoformat()),
    )
    conn.commit()
