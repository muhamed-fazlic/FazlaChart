"""Unit tests for the scanner-staleness health check — SQLite only, no
network calls."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from scanner.health import check_scanner_health
from scanner.storage import fetch_scan_health, init_db, save_scan_result

pytestmark = pytest.mark.unit

_SNAPSHOT = {
    "symbol": "BTCUSDT",
    "timeframes": {"1h": {"trend": "bullish", "close": 50000.0}},
}


def test_fetch_scan_health_returns_none_timestamp_when_empty(tmp_path):
    conn = init_db(str(tmp_path / "empty.db"))
    info = fetch_scan_health(conn)
    assert info["last_scanned_at"] is None
    assert info["symbols_ever_tracked"] == 0


def test_fetch_scan_health_counts_distinct_symbols(tmp_path):
    conn = init_db(str(tmp_path / "scans.db"))
    save_scan_result(conn, "BTCUSDT", _SNAPSHOT, [])
    save_scan_result(conn, "ETHUSDT", _SNAPSHOT, [])
    info = fetch_scan_health(conn)
    assert info["last_scanned_at"] is not None
    assert info["symbols_ever_tracked"] == 2


def test_check_scanner_health_no_data_when_never_scanned(tmp_path):
    conn = init_db(str(tmp_path / "empty.db"))
    result = check_scanner_health(conn)
    assert result.status == "no_data"
    assert result.last_scanned_at is None
    assert result.staleness_seconds is None


def test_check_scanner_health_is_healthy_right_after_a_scan(tmp_path):
    conn = init_db(str(tmp_path / "fresh.db"))
    save_scan_result(conn, "BTCUSDT", _SNAPSHOT, [])
    result = check_scanner_health(conn, max_staleness_seconds=900)
    assert result.status == "healthy"
    assert result.staleness_seconds < 10


def test_check_scanner_health_is_stale_past_the_threshold(tmp_path):
    conn = init_db(str(tmp_path / "stale.db"))
    save_scan_result(conn, "BTCUSDT", _SNAPSHOT, [])
    old_timestamp = (datetime.now(UTC) - timedelta(seconds=1000)).isoformat()
    conn.execute("UPDATE scans SET scanned_at = ?", (old_timestamp,))
    conn.commit()

    result = check_scanner_health(conn, max_staleness_seconds=900)
    assert result.status == "stale"
    assert result.staleness_seconds > 900
