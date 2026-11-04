"""Phase 8 Korak 3: permanent experiment trial-count ledger.

Deflated Sharpe Ratio and multiple-testing correction are only honest if
``n_trials`` (how many strategy variants were tried before this one) is
itself honest. Without a durable record, it is trivially easy to
under-report trials after the fact ("we only really tried 3 things") once
a profitable-looking configuration has already been found -- this module
exists so that number is never a guess.

Every backtest/validation run that is meant to inform a real decision
(parameter choice, timeframe choice, label/pattern choice, SL/TP
configuration, train/validation/test split) should call
:func:`record_experiment` BEFORE looking at whether the result is good,
exactly the way a pre-registered clinical trial locks its analysis plan in
before unblinding -- recording only after cherry-picking the winner
defeats the entire purpose. :func:`count_prior_trials` then gives the
honest ``n_trials``/``n_trials_before`` figures that feed
:func:`scanner.validation.statistical_validation.deflated_sharpe_ratio` and
:func:`scanner.validation.statistical_validation.benjamini_hochberg`.

Intentionally a separate, standalone SQLite database (not the live
``scanner.db`` schema in :mod:`scanner.storage`) -- this is an offline
research ledger, never touched by the live scanner/dashboard processes,
same philosophy as :mod:`scanner.backtesting.portfolio_backtest` writing
its own JSON files rather than reading/writing the production database.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recorded_at TEXT NOT NULL,
    strategy_family TEXT NOT NULL,
    strategy_version TEXT,
    timeframe TEXT,
    label TEXT,
    horizon_candles INTEGER,
    sl_tp_config_json TEXT,
    train_period_start TEXT,
    train_period_end TEXT,
    validation_period_start TEXT,
    validation_period_end TEXT,
    test_period_start TEXT,
    test_period_end TEXT,
    n_trades INTEGER,
    sharpe_ratio REAL,
    net_profit_pct REAL,
    result_json TEXT,
    notes TEXT
);
CREATE INDEX IF NOT EXISTS idx_experiments_family ON experiments (strategy_family, recorded_at);
"""


def init_db(storage_path: str, timeout: float = 10.0) -> sqlite3.Connection:
    """Open (creating if needed) the experiment registry database and
    ensure its schema -- mirrors ``scanner.storage.init_db``'s WAL/timeout
    conventions, applied to this module's own standalone database file."""
    path = Path(storage_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=timeout)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


def record_experiment(
    conn: sqlite3.Connection,
    *,
    strategy_family: str,
    strategy_version: str | None = None,
    timeframe: str | None = None,
    label: str | None = None,
    horizon_candles: int | None = None,
    sl_tp_config: dict[str, Any] | None = None,
    train_period: tuple[str | None, str | None] | None = None,
    validation_period: tuple[str | None, str | None] | None = None,
    test_period: tuple[str | None, str | None] | None = None,
    n_trades: int | None = None,
    sharpe_ratio: float | None = None,
    net_profit_pct: float | None = None,
    result: dict[str, Any] | None = None,
    notes: str | None = None,
) -> int:
    """Permanently record one experiment/trial. ``strategy_family`` groups
    trials that should be counted together for multiple-testing purposes
    (e.g. ``"best_signal_v1"`` or ``"portfolio_backtest_5m15m1h4h"``) --
    pick a family name that matches the actual scope of "things I tried
    while looking for a profitable configuration", not something narrower
    that would under-count real trials.

    ``train_period``/``validation_period``/``test_period`` are each an
    optional ``(start, end)`` pair of ISO timestamps (or any string label);
    omit a period entirely (leave as ``None``) if that split was not used
    for this experiment. Returns the new row's ``id``.
    """
    train_period = train_period or (None, None)
    validation_period = validation_period or (None, None)
    test_period = test_period or (None, None)
    cursor = conn.execute(
        """
        INSERT INTO experiments (
            recorded_at, strategy_family, strategy_version, timeframe, label,
            horizon_candles, sl_tp_config_json,
            train_period_start, train_period_end,
            validation_period_start, validation_period_end,
            test_period_start, test_period_end,
            n_trades, sharpe_ratio, net_profit_pct, result_json, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now(UTC).isoformat(),
            strategy_family,
            strategy_version,
            timeframe,
            label,
            horizon_candles,
            json.dumps(sl_tp_config) if sl_tp_config is not None else None,
            train_period[0],
            train_period[1],
            validation_period[0],
            validation_period[1],
            test_period[0],
            test_period[1],
            n_trades,
            sharpe_ratio,
            net_profit_pct,
            json.dumps(result) if result is not None else None,
            notes,
        ),
    )
    conn.commit()
    return int(cursor.lastrowid)


def count_prior_trials(conn: sqlite3.Connection, strategy_family: str, *, before: str | None = None) -> int:
    """How many experiments in ``strategy_family`` were recorded at or
    before ``before`` (an ISO timestamp; defaults to "right now"). This IS
    the honest ``n_trials`` figure to feed into
    ``statistical_validation.deflated_sharpe_ratio``/``benjamini_hochberg``
    -- counting every recorded attempt, not just the one that ended up
    looking best.
    """
    before = before or datetime.now(UTC).isoformat()
    row = conn.execute(
        "SELECT COUNT(*) FROM experiments WHERE strategy_family = ? AND recorded_at <= ?",
        (strategy_family, before),
    ).fetchone()
    return int(row[0]) if row else 0


def fetch_experiments(
    conn: sqlite3.Connection,
    strategy_family: str | None = None,
    limit: int = 200,
) -> list[dict[str, Any]]:
    """Fetch recorded experiments, most recent first. Pass
    ``strategy_family`` to filter to one trial family; omit it to see
    everything (e.g. for an overview of total research activity)."""
    columns = [
        "id",
        "recorded_at",
        "strategy_family",
        "strategy_version",
        "timeframe",
        "label",
        "horizon_candles",
        "sl_tp_config_json",
        "train_period_start",
        "train_period_end",
        "validation_period_start",
        "validation_period_end",
        "test_period_start",
        "test_period_end",
        "n_trades",
        "sharpe_ratio",
        "net_profit_pct",
        "result_json",
        "notes",
    ]
    if strategy_family is not None:
        rows = conn.execute(
            f"SELECT {', '.join(columns)} FROM experiments WHERE strategy_family = ? "
            "ORDER BY recorded_at DESC LIMIT ?",
            (strategy_family, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            f"SELECT {', '.join(columns)} FROM experiments ORDER BY recorded_at DESC LIMIT ?",
            (limit,),
        ).fetchall()

    results = []
    for row in rows:
        entry = dict(zip(columns, row, strict=True))
        if entry["sl_tp_config_json"]:
            entry["sl_tp_config"] = json.loads(entry.pop("sl_tp_config_json"))
        else:
            entry["sl_tp_config"] = None
            entry.pop("sl_tp_config_json")
        if entry["result_json"]:
            entry["result"] = json.loads(entry.pop("result_json"))
        else:
            entry["result"] = None
            entry.pop("result_json")
        results.append(entry)
    return results
