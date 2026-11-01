"""Unit tests for pruning old `scans` rows so the database doesn't grow
without bound over months of 24/7 operation — SQLite only, no network calls."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from scanner.storage import fetch_known_symbols, init_db, prune_old_scans, save_scan_result

pytestmark = pytest.mark.unit

_SNAPSHOT = {
    "symbol": "BTCUSDT",
    "timeframes": {"1h": {"trend": "bullish", "close": 50000.0}},
}


def test_prune_old_scans_deletes_rows_past_retention_and_keeps_recent(tmp_path):
    conn = init_db(str(tmp_path / "scans.db"))
    save_scan_result(conn, "BTCUSDT", _SNAPSHOT, [])
    save_scan_result(conn, "ETHUSDT", _SNAPSHOT, [])

    old_timestamp = (datetime.now(UTC) - timedelta(days=90)).isoformat()
    conn.execute("UPDATE scans SET scanned_at = ? WHERE symbol = ?", (old_timestamp, "BTCUSDT"))
    conn.commit()

    deleted = prune_old_scans(conn, retain_days=60)

    assert deleted == 1
    assert fetch_known_symbols(conn) == ["ETHUSDT"]


def test_prune_old_scans_is_a_no_op_when_nothing_is_old(tmp_path):
    conn = init_db(str(tmp_path / "scans.db"))
    save_scan_result(conn, "BTCUSDT", _SNAPSHOT, [])

    deleted = prune_old_scans(conn, retain_days=60)

    assert deleted == 0
    assert fetch_known_symbols(conn) == ["BTCUSDT"]


def test_prune_old_scans_on_empty_db_returns_zero(tmp_path):
    conn = init_db(str(tmp_path / "empty.db"))
    assert prune_old_scans(conn, retain_days=60) == 0
