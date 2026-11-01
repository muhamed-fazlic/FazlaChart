"""Scanner health / staleness check.

The scanner's own ``run_forever`` loop already survives a single cycle
throwing (it logs and moves on), which is the right behavior for a bad
symbol or a flaky vendor response -- but it means the process can also
silently wedge (hang inside a call that never raises, or get OOM-killed
and not restarted) without anything in its own log screaming about it.
This module answers one question from the *outside*: "when did the scanner
last actually write fresh data?" -- so the dashboard (a separate, simpler
process far less likely to hang) can surface that to a human and/or push a
Telegram alert, instead of a user only finding out because the numbers on
screen stopped moving.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from scanner.storage import fetch_scan_health

HealthStatus = Literal["healthy", "stale", "no_data"]


@dataclass
class ScannerHealth:
    status: HealthStatus
    last_scanned_at: str | None
    staleness_seconds: float | None
    symbols_ever_tracked: int
    max_staleness_seconds: float


def check_scanner_health(conn: sqlite3.Connection, max_staleness_seconds: float = 900.0) -> ScannerHealth:
    """``max_staleness_seconds`` should comfortably exceed one real poll
    interval (default scanner config polls every 300s) so a single slow
    cycle never false-positives; the default of 900s (3x) tolerates one
    cycle running long without masking an actually-stuck scanner."""
    info = fetch_scan_health(conn)
    last_scanned_at = info["last_scanned_at"]
    symbols_ever_tracked = info["symbols_ever_tracked"]

    if last_scanned_at is None:
        return ScannerHealth(
            status="no_data",
            last_scanned_at=None,
            staleness_seconds=None,
            symbols_ever_tracked=symbols_ever_tracked,
            max_staleness_seconds=max_staleness_seconds,
        )

    last_dt = datetime.fromisoformat(last_scanned_at)
    if last_dt.tzinfo is None:
        last_dt = last_dt.replace(tzinfo=UTC)
    staleness_seconds = (datetime.now(UTC) - last_dt).total_seconds()
    status: HealthStatus = "healthy" if staleness_seconds <= max_staleness_seconds else "stale"
    return ScannerHealth(
        status=status,
        last_scanned_at=last_scanned_at,
        staleness_seconds=round(staleness_seconds, 1),
        symbols_ever_tracked=symbols_ever_tracked,
        max_staleness_seconds=max_staleness_seconds,
    )
