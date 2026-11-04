"""Phase 8 Korak 1: persistent archive of every candidate that cleared the
Top Signal Now statistical significance gate, whether or not it was
actually selected.

``signal_outcomes`` (see :mod:`scanner.storage`) only ever records the
winners -- the candidates that made it into the live top-N shown on the
dashboard. That's enough to check "did what we chose actually work", but it
can never answer "would taking MORE of what we rejected also have worked",
because the rejected candidates simply vanish once a cycle ends. This
module closes that gap: every cycle, every candidate in
``agents_crypto.best_signal.compute_candidates(..., include_rejected=True)``'s
``all_eligible_by_timeframe`` is persisted to the ``candidate_archive``
table, tagged with exactly why it was or wasn't selected (outranked within
its own timeframe's ``limit``, or excluded by a muted bucket).

This is intentionally dumb, additive storage with no interpretation of its
own -- same philosophy as ``scans`` (see that table's docstring in
``scanner.storage``). The interpretation -- replaying what would have
happened to every rejected candidate too, not just the selected ones -- is
Korak 2's job (portfolio-level counterfactual backtest), which this
archive exists to make possible.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from typing import Any


def record_candidates(conn: sqlite3.Connection, result: dict[str, Any]) -> int:
    """Persist every candidate in ``result["all_eligible_by_timeframe"]``
    (the output of :func:`agents_crypto.best_signal.compute_candidates`
    called with ``include_rejected=True``) into the ``candidate_archive``
    table -- one row per (symbol, timeframe, direction, scanned_at),
    selected or not.

    ``INSERT OR IGNORE`` against the table's
    ``(scanned_at, symbol, timeframe, direction)`` unique constraint makes
    this safe to call more than once for the same cycle (e.g. after a
    crash-restart that re-processes the same scan) -- a duplicate call
    simply inserts nothing new instead of double-counting. Returns how many
    rows were newly inserted.

    A missing or empty ``all_eligible_by_timeframe`` (e.g. a caller that
    didn't pass ``include_rejected=True`` to ``compute_candidates``) is
    treated as "nothing to archive" rather than an error, so this can be
    called defensively without every caller needing to check first.
    """
    all_by_timeframe = result.get("all_eligible_by_timeframe") or {}
    archived_at = datetime.now(UTC).isoformat()
    inserted = 0
    for candidates in all_by_timeframe.values():
        for candidate in candidates:
            entry = candidate.get("entry") or {}
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO candidate_archive (
                    scanned_at, symbol, timeframe, direction, rank, selected, muted,
                    rejection_reason, label, pattern, n, win_rate_pct, win_rate_ci_low_pct,
                    avg_net_return_pct, regime, regime_fallback, quality_score,
                    entry_price, stop_loss, take_profit, risk_reward_ratio,
                    candidate_json, archived_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    candidate["scanned_at"],
                    candidate["symbol"],
                    candidate["timeframe"],
                    candidate["direction"],
                    candidate["rank"],
                    1 if candidate.get("selected") else 0,
                    1 if candidate.get("muted") else 0,
                    candidate.get("rejection_reason"),
                    candidate.get("label"),
                    candidate.get("pattern"),
                    candidate.get("n"),
                    candidate.get("win_rate_pct"),
                    candidate.get("win_rate_ci_low_pct"),
                    candidate.get("avg_net_return_pct"),
                    candidate.get("regime"),
                    1 if candidate.get("regime_fallback") else 0,
                    (candidate.get("quality_score") or {}).get("total"),
                    entry.get("price"),
                    entry.get("stop_loss"),
                    entry.get("take_profit"),
                    entry.get("risk_reward_ratio"),
                    json.dumps(candidate),
                    archived_at,
                ),
            )
            inserted += max(cursor.rowcount, 0)
    conn.commit()
    return inserted
