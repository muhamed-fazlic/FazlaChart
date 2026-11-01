"""Unit tests for the calibration-drift mute/unmute logic
(scanner/drift_detector.py). Uses a real on-disk sqlite db (via
scanner.storage.init_db) and writes signal_outcomes rows directly so each
test can control exactly how many resolved signals exist and what their
realized/promised numbers are, without going through the full scan/track
pipeline."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from scanner.drift_detector import (
    DRIFT_THRESHOLD_PP,
    MIN_RESOLVED_FOR_DRIFT_CHECK,
    active_muted_bucket_set,
    compute_bucket_drift,
    evaluate_and_apply_drift_mutes,
)
from scanner.storage import fetch_active_muted_bucket, init_db

pytestmark = pytest.mark.unit


@pytest.fixture
def conn(tmp_path):
    db_path = tmp_path / "drift_test.db"
    connection = init_db(str(db_path))
    yield connection
    connection.close()


def _insert_resolved(conn, *, timeframe="1h", direction="short", promised_ci_low=60.0, is_win=True, when=None):
    """Writes one resolved signal_outcomes row with just the fields the
    drift detector reads."""
    resolved_at = (when or datetime.now(UTC)).isoformat()
    conn.execute(
        """
        INSERT INTO signal_outcomes (
            symbol, timeframe, direction, label, logged_at, entry_price,
            stop_loss, take_profit, promised_win_rate_ci_low_pct, promised_n,
            horizon_candles, status, resolved_at, exit_price, realized_return_pct
        ) VALUES ('TESTUSDT', ?, ?, 'trend:bearish', ?, 10.0, 11.0, 9.0, ?, 200, 12, ?, ?, 9.5, ?)
        """,
        (
            timeframe,
            direction,
            resolved_at,
            promised_ci_low,
            "take_profit" if is_win else "stop_loss",
            resolved_at,
            0.5 if is_win else -0.5,
        ),
    )
    conn.commit()


class TestComputeBucketDrift:
    def test_returns_none_below_minimum_sample_size(self, conn):
        for _ in range(MIN_RESOLVED_FOR_DRIFT_CHECK - 1):
            _insert_resolved(conn, is_win=True)
        assert compute_bucket_drift(conn, "1h", "short") is None

    def test_computes_realized_vs_promised_once_enough_resolved(self, conn):
        # 50% realized win rate against a 60% promised CI-low -> 10pp drift.
        for i in range(MIN_RESOLVED_FOR_DRIFT_CHECK):
            _insert_resolved(conn, promised_ci_low=60.0, is_win=(i % 2 == 0))

        stats = compute_bucket_drift(conn, "1h", "short")
        assert stats is not None
        assert stats.n_resolved == MIN_RESOLVED_FOR_DRIFT_CHECK
        assert stats.realized_win_rate_pct == pytest.approx(50.0)
        assert stats.promised_win_rate_ci_low_pct_avg == pytest.approx(60.0)
        assert stats.drift_pp == pytest.approx(10.0)

    def test_only_considers_most_recent_window(self, conn):
        # 60 old losses, then 30 new wins -- the rolling window (60) should
        # be dominated by recent results once there are more than 60 total.
        old_time = datetime.now(UTC) - timedelta(days=30)
        for _ in range(60):
            _insert_resolved(conn, is_win=False, when=old_time)
        for _ in range(30):
            _insert_resolved(conn, is_win=True)

        stats = compute_bucket_drift(conn, "1h", "short")
        assert stats is not None
        assert stats.n_resolved == 60  # DRIFT_WINDOW_SIZE
        # Window = 30 new wins + 30 most-recent old losses (ordered by resolved_at desc).
        assert stats.realized_win_rate_pct == pytest.approx(50.0)


class TestEvaluateAndApplyDriftMutes:
    def test_mutes_a_bucket_that_drifted_badly(self, conn):
        for i in range(MIN_RESOLVED_FOR_DRIFT_CHECK):
            _insert_resolved(conn, promised_ci_low=65.0, is_win=(i < 10))  # 10/30 = ~33% realized

        muted_events = []
        result = evaluate_and_apply_drift_mutes(conn, on_mute=muted_events.append)

        assert result["muted"] == ["1h/short"]
        assert len(muted_events) == 1
        assert fetch_active_muted_bucket(conn, "1h", "short") is not None
        assert active_muted_bucket_set(conn) == {("1h", "short")}

    def test_does_not_mute_a_bucket_within_tolerance(self, conn):
        for i in range(MIN_RESOLVED_FOR_DRIFT_CHECK):
            _insert_resolved(conn, promised_ci_low=60.0, is_win=(i % 2 == 0))  # 50% vs 60% promised = 10pp < 12pp threshold

        result = evaluate_and_apply_drift_mutes(conn)

        assert result["muted"] == []
        assert fetch_active_muted_bucket(conn, "1h", "short") is None

    def test_does_not_mute_with_insufficient_sample(self, conn):
        for _ in range(MIN_RESOLVED_FOR_DRIFT_CHECK - 5):
            _insert_resolved(conn, promised_ci_low=80.0, is_win=False)  # terrible, but too few samples

        result = evaluate_and_apply_drift_mutes(conn)

        assert result["muted"] == []
        assert active_muted_bucket_set(conn) == set()

    def test_restores_a_muted_bucket_once_it_recovers(self, conn):
        old_time = datetime.now(UTC) - timedelta(days=5)
        for _ in range(MIN_RESOLVED_FOR_DRIFT_CHECK):
            _insert_resolved(conn, promised_ci_low=65.0, is_win=False, when=old_time)  # 0% realized -> mute

        evaluate_and_apply_drift_mutes(conn)
        assert fetch_active_muted_bucket(conn, "1h", "short") is not None

        # Enough new, recent wins to fully push the stale losses out of the
        # rolling window (DRIFT_WINDOW_SIZE=60), not just dilute them.
        for _ in range(60):
            _insert_resolved(conn, promised_ci_low=65.0, is_win=True)

        unmuted_events = []
        result = evaluate_and_apply_drift_mutes(conn, on_unmute=unmuted_events.append)

        assert result["unmuted"] == ["1h/short"]
        assert len(unmuted_events) == 1
        assert fetch_active_muted_bucket(conn, "1h", "short") is None

    def test_stays_muted_if_recovery_is_only_partial(self, conn):
        old_time = datetime.now(UTC) - timedelta(days=5)
        for _ in range(MIN_RESOLVED_FOR_DRIFT_CHECK):
            _insert_resolved(conn, promised_ci_low=65.0, is_win=False, when=old_time)
        evaluate_and_apply_drift_mutes(conn)
        assert fetch_active_muted_bucket(conn, "1h", "short") is not None

        # Still well short of recovery threshold (needs drift < 6pp; this keeps it near 65pp promised vs ~50% realized = 15pp).
        for i in range(MIN_RESOLVED_FOR_DRIFT_CHECK):
            _insert_resolved(conn, promised_ci_low=65.0, is_win=(i % 2 == 0))

        result = evaluate_and_apply_drift_mutes(conn)
        assert result["unmuted"] == []
        assert fetch_active_muted_bucket(conn, "1h", "short") is not None

    def test_mute_callback_failure_does_not_block_other_buckets(self, conn):
        for i in range(MIN_RESOLVED_FOR_DRIFT_CHECK):
            _insert_resolved(conn, timeframe="1h", direction="short", promised_ci_low=70.0, is_win=(i < 5))
            _insert_resolved(conn, timeframe="4h", direction="long", promised_ci_low=70.0, is_win=(i < 5))

        def _boom(_stats):
            raise RuntimeError("telegram is down")

        result = evaluate_and_apply_drift_mutes(conn, on_mute=_boom)

        assert set(result["muted"]) == {"1h/short", "4h/long"}
        assert active_muted_bucket_set(conn) == {("1h", "short"), ("4h", "long")}

    def test_ignores_buckets_with_no_resolved_history(self, conn):
        result = evaluate_and_apply_drift_mutes(conn)
        assert result == {"muted": [], "unmuted": []}


def test_drift_threshold_constant_is_sane():
    # Sanity guard so a future edit can't silently make this a hair-trigger.
    assert DRIFT_THRESHOLD_PP >= 5.0
