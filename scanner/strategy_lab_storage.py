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

Five tables, one per concern:

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
* ``layer2_frozen_candidates`` -- the Layer 2 "confirmation queue" (Korak
  10 item 5): configs that have satisfied the entry threshold, with their
  configuration and ``as_of_ts`` (T) frozen forever the first time they
  qualify. Empty most of the time -- that is the normal, expected state.
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

-- Korak 10 widening: the Layer 2 "confirmation queue". One row is
-- inserted the FIRST time a (template, tercile, variant_label) config
-- ever satisfies the entry threshold (layer2_entry.check_entry_threshold)
-- -- its configuration and the as_of_ts at that moment (T) are then
-- FROZEN here forever (see UNIQUE constraint: a config can only ever be
-- frozen once, even if it keeps re-qualifying on later cycles).
-- ``consumed`` flips to 1 the first time run_layer2_check actually spends
-- a real confirmatory run on it (pass or fail -- both consume the single
-- shot; see STRATEGY_LAB_DESIGN.md Korak 10 item 6).
CREATE TABLE IF NOT EXISTS layer2_frozen_candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    template TEXT NOT NULL,
    tercile TEXT NOT NULL,
    variant_label TEXT NOT NULL,
    frozen_as_of_ts INTEGER NOT NULL,
    frozen_at TEXT NOT NULL,
    source_candidate_id INTEGER,
    windows_checked INTEGER NOT NULL,
    windows_positive INTEGER NOT NULL,
    net_observed_pct REAL,
    gross_observed_pct REAL,
    n_independent INTEGER,
    btc_beta REAL,
    consumed INTEGER NOT NULL DEFAULT 0,
    UNIQUE (template, tercile, variant_label),
    FOREIGN KEY (source_candidate_id) REFERENCES mining_candidates (id)
);
CREATE INDEX IF NOT EXISTS idx_layer2_frozen_consumed ON layer2_frozen_candidates (consumed);
"""


def _migrate_schema(conn: sqlite3.Connection) -> None:
    """Adds columns introduced after a database file already existed.
    ``CREATE TABLE IF NOT EXISTS`` in ``_SCHEMA`` only creates a table the
    very first time -- it never adds a new column to a table that already
    exists on disk, so any such column needs an explicit, idempotent
    ``ALTER TABLE`` here instead (same pattern as ``scanner.storage``'s
    own ``_migrate_schema``)."""
    columns = {row[1] for row in conn.execute("PRAGMA table_info(mining_candidates)")}
    if "dataset_hash" not in columns:
        # Korak 10 widening: lets dedup v2 key on (template, tercile,
        # variant_label, dataset_hash) instead of just as_of_timestamp_ms --
        # see fetch_mined_dataset_hashes. NULL for rows mined before this
        # column existed and for templates whose frozen backtest report
        # never carried a dataset_hash (trend_anomaly) -- both fall back
        # to the older as_of_timestamp_ms-based dedup key, unaffected.
        conn.execute("ALTER TABLE mining_candidates ADD COLUMN dataset_hash TEXT")


def init_db(storage_path: str = DEFAULT_STRATEGY_LAB_DB_PATH, timeout: float = 10.0) -> sqlite3.Connection:
    """Opens (creating if needed) the Strategy Lab database and ensures its
    schema -- same WAL-mode/timeout rationale as ``scanner.storage.init_db``."""
    path = Path(storage_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=timeout)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_SCHEMA)
    _migrate_schema(conn)
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
    dataset_hash: str | None = None,
) -> int:
    cursor = conn.execute(
        "INSERT INTO mining_candidates (mined_at, as_of_timestamp_ms, template, tercile, variant_label,"
        " params_json, n_independent, gross_observed_pct, net_observed_pct, stability_score, composite_score,"
        " report_json, dataset_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
            dataset_hash,
        ),
    )
    conn.commit()
    return int(cursor.lastrowid)


def fetch_leaderboard(conn: sqlite3.Connection, limit: int = 10) -> list[dict[str, Any]]:
    """Top ``limit`` DISTINCT (template, tercile, variant_label) configs,
    each represented by its MOST RECENT evaluation only, ranked by that
    evaluation's ``composite_score`` descending.

    Korak 10 widening note: before the rolling ``as_of_ts`` for the new
    cross-sectional templates was introduced, every config in the table
    shared the exact same ``as_of_timestamp_ms`` (the single frozen
    ``SHARED_HOLDOUT_END_MS``), so "latest mining run" and "latest per
    config" were the same thing and grouping by ``MAX(as_of_timestamp_ms)``
    worked. Now the closed/frozen families (trend_anomaly,
    funding_extreme_reversion, cross_sectional_momentum) stay pinned to
    that one fixed ``as_of_ts`` forever, while the new rolling templates
    (momentum_30d, low_vol, funding_carry_xs) get a NEW ``as_of_ts``/
    ``dataset_hash`` roughly weekly -- so a single global "latest" no
    longer identifies "the current result for every config". Instead this
    groups by config and takes each one's own highest-``id`` (most
    recently inserted) row, which handles both cases correctly."""
    cursor = conn.execute(
        "SELECT mc.id, mc.mined_at, mc.as_of_timestamp_ms, mc.template, mc.tercile, mc.variant_label,"
        " mc.params_json, mc.n_independent, mc.gross_observed_pct, mc.net_observed_pct, mc.stability_score,"
        " mc.composite_score, mc.dataset_hash"
        " FROM mining_candidates mc"
        " INNER JOIN ("
        "   SELECT template, tercile, variant_label, MAX(id) AS max_id"
        "   FROM mining_candidates GROUP BY template, tercile, variant_label"
        " ) latest ON mc.id = latest.max_id"
        " ORDER BY mc.composite_score DESC LIMIT ?",
        (limit,),
    )
    columns = [d[0] for d in cursor.description]
    rows = [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]
    for row in rows:
        row["params"] = json.loads(row.pop("params_json"))
    return rows


def fetch_mined_combo_keys(conn: sqlite3.Connection, as_of_timestamp_ms: int) -> frozenset[tuple[str, str, str]]:
    """``(template, tercile, variant_label)`` combinations already stored
    for this exact ``as_of_timestamp_ms``. Used ONLY for the closed/frozen
    families (whose ``as_of_ts`` is permanently clamped to
    ``SHARED_HOLDOUT_END_MS``) to skip re-mining a combination whose
    result is guaranteed to be byte-identical to what's already stored --
    without this check every scheduled cycle re-runs (and re-inserts) the
    exact same combinations forever, wasting CPU and filling the
    leaderboard with duplicate rows (a real bug this function exists to
    prevent). The new rolling templates use ``fetch_mined_dataset_hashes``
    instead (their ``as_of_ts`` moves, so pre-filtering on it would skip
    genuinely new data)."""
    cursor = conn.execute(
        "SELECT DISTINCT template, tercile, variant_label FROM mining_candidates WHERE as_of_timestamp_ms = ?",
        (as_of_timestamp_ms,),
    )
    return frozenset((row[0], row[1], row[2]) for row in cursor.fetchall())


def fetch_ever_mined_combo_keys(conn: sqlite3.Connection) -> frozenset[tuple[str, str, str]]:
    """``(template, tercile, variant_label)`` combinations ever stored, at
    ANY ``as_of_timestamp_ms`` -- i.e. the set of configs that already
    count towards the unique-configuration cap (Korak 10 item 4). Re-
    testing one of these at a new rolling ``as_of_ts`` never grows the
    space; only a combination NOT in this set does."""
    cursor = conn.execute("SELECT DISTINCT template, tercile, variant_label FROM mining_candidates")
    return frozenset((row[0], row[1], row[2]) for row in cursor.fetchall())


def fetch_mined_dataset_hashes(conn: sqlite3.Connection) -> frozenset[tuple[str, str, str, str]]:
    """``(template, tercile, variant_label, dataset_hash)`` combinations
    already stored, for every row that HAS a ``dataset_hash`` (the rolling
    templates; closed/frozen-family rows and rows mined before this column
    existed have ``dataset_hash IS NULL`` and are excluded). Dedup key v2
    (Korak 10 item 2): the rolling templates are ALWAYS re-run every
    cycle (their ``as_of_ts`` moves, so pre-filtering on it would wrongly
    skip genuinely fresh data), but if the exact same underlying dataset
    was already stored for this config (e.g. a restart re-running the same
    still-current week), this lets the caller skip the duplicate INSERT."""
    cursor = conn.execute(
        "SELECT DISTINCT template, tercile, variant_label, dataset_hash FROM mining_candidates"
        " WHERE dataset_hash IS NOT NULL"
    )
    return frozenset((row[0], row[1], row[2], row[3]) for row in cursor.fetchall())


def count_unique_configurations(conn: sqlite3.Connection) -> int:
    """Count of DISTINCT (template, tercile, variant_label) configs ever
    mined -- the "x / 300" counter (Korak 10 item 4). Deliberately counts
    by CONFIG, not by row/``as_of_ts``: re-scoring an existing config on a
    new rolling window must never inflate this number (that would wrongly
    tighten the Holm correction over time even though the hypothesis space
    itself hasn't grown)."""
    row = conn.execute(
        "SELECT COUNT(*) FROM (SELECT DISTINCT template, tercile, variant_label FROM mining_candidates)"
    ).fetchone()
    return int(row[0])


def fetch_recent_results_for_config(
    conn: sqlite3.Connection, template: str, tercile: str, variant_label: str, limit: int = 3
) -> list[dict[str, Any]]:
    """The ``limit`` most recent (highest ``as_of_timestamp_ms``) stored
    results for one exact config, newest first -- used both for the
    dashboard's "windows positive" column and for
    ``layer2_entry.check_entry_threshold``'s consecutive-positive-windows
    check."""
    cursor = conn.execute(
        "SELECT id, as_of_timestamp_ms, n_independent, gross_observed_pct, net_observed_pct, dataset_hash"
        " FROM mining_candidates WHERE template = ? AND tercile = ? AND variant_label = ?"
        " ORDER BY as_of_timestamp_ms DESC, id DESC LIMIT ?",
        (template, tercile, variant_label, limit),
    )
    columns = [d[0] for d in cursor.description]
    return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]


def fetch_first_mined_at_for_config(
    conn: sqlite3.Connection, template: str, tercile: str, variant_label: str
) -> int | None:
    """Earliest ``as_of_timestamp_ms`` ever stored for one exact config --
    used as a "training period started here" marker for rolling-template
    candidates in Layer 2's power estimate (they don't share the old
    closed families' fixed ``SHARED_HOLDOUT_END_MS`` training start)."""
    row = conn.execute(
        "SELECT MIN(as_of_timestamp_ms) FROM mining_candidates WHERE template = ? AND tercile = ? AND variant_label = ?",
        (template, tercile, variant_label),
    ).fetchone()
    return int(row[0]) if row and row[0] is not None else None


def insert_layer2_frozen_candidate(
    conn: sqlite3.Connection,
    *,
    template: str,
    tercile: str,
    variant_label: str,
    frozen_as_of_ts: int,
    source_candidate_id: int | None,
    windows_checked: int,
    windows_positive: int,
    net_observed_pct: float | None,
    gross_observed_pct: float | None,
    n_independent: int | None,
    btc_beta: float | None,
) -> int | None:
    """Freezes a (template, tercile, variant_label) config into the Layer 2
    confirmation queue the FIRST time it satisfies the entry threshold
    (Korak 10 item 5). ``INSERT OR IGNORE`` + the ``UNIQUE`` schema
    constraint means a config already in the queue is never re-frozen --
    its original configuration and T (``frozen_as_of_ts``) stay exactly as
    first recorded, even if it keeps re-qualifying on later cycles.
    Returns the new row's id, or ``None`` if it was already frozen."""
    cursor = conn.execute(
        "INSERT OR IGNORE INTO layer2_frozen_candidates (template, tercile, variant_label, frozen_as_of_ts,"
        " frozen_at, source_candidate_id, windows_checked, windows_positive, net_observed_pct,"
        " gross_observed_pct, n_independent, btc_beta) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            template,
            tercile,
            variant_label,
            frozen_as_of_ts,
            datetime.now(UTC).isoformat(),
            source_candidate_id,
            windows_checked,
            windows_positive,
            net_observed_pct,
            gross_observed_pct,
            n_independent,
            btc_beta,
        ),
    )
    conn.commit()
    return int(cursor.lastrowid) if cursor.rowcount else None


def fetch_layer2_candidate_queue(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Every frozen Layer 2 candidate, oldest-frozen first (consumed or
    not) -- backs the dashboard's "confirmation queue" section."""
    cursor = conn.execute(
        "SELECT id, template, tercile, variant_label, frozen_as_of_ts, frozen_at, source_candidate_id,"
        " windows_checked, windows_positive, net_observed_pct, gross_observed_pct, n_independent, btc_beta,"
        " consumed FROM layer2_frozen_candidates ORDER BY frozen_at ASC"
    )
    columns = [d[0] for d in cursor.description]
    return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]


def fetch_next_unconsumed_layer2_candidate(conn: sqlite3.Connection) -> dict[str, Any] | None:
    """Oldest-frozen candidate that hasn't yet had its one Layer 2
    confirmation shot (FIFO -- the first config to qualify gets checked
    first). ``None`` means the queue is empty, the normal, expected state
    most of the time."""
    cursor = conn.execute(
        "SELECT id, template, tercile, variant_label, frozen_as_of_ts, frozen_at, source_candidate_id,"
        " windows_checked, windows_positive, net_observed_pct, gross_observed_pct, n_independent, btc_beta"
        " FROM layer2_frozen_candidates WHERE consumed = 0 ORDER BY frozen_at ASC LIMIT 1"
    )
    row = cursor.fetchone()
    if row is None:
        return None
    columns = [d[0] for d in cursor.description]
    return dict(zip(columns, row, strict=True))


def mark_layer2_candidate_consumed(conn: sqlite3.Connection, candidate_id: int) -> None:
    conn.execute("UPDATE layer2_frozen_candidates SET consumed = 1 WHERE id = ?", (candidate_id,))
    conn.commit()


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


def fetch_latest_weekly_snapshot_at(conn: sqlite3.Connection) -> datetime | None:
    """UTC timestamp of the most recently stored weekly snapshot, or
    ``None`` if Layer 1 has never produced one. Lets the scheduler persist
    "when did the weekly snapshot/Layer-2 check last run" in the database
    instead of an in-process ``time.monotonic()`` value -- the in-memory
    version resets to "never" on every process restart, so a routine
    deployment restart used to immediately re-trigger an out-of-schedule
    weekly snapshot (and its Telegram message) regardless of how recently
    the real last one happened (a real production bug this function exists
    to prevent)."""
    row = conn.execute("SELECT MAX(snapshot_at) FROM weekly_snapshots").fetchone()
    if row is None or row[0] is None:
        return None
    return datetime.fromisoformat(row[0])


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
