"""Unit tests for Korak 2's MFE/MAE-based SL/TP calibration
(scanner/analysis/sl_tp_calibration.py). Uses a real on-disk sqlite db (via
scanner.storage.init_db) and writes signal_outcomes rows directly, same
pattern as tests/test_drift_detector.py, so each test fully controls the
resolved sample it's computing a recommendation from."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from agents_crypto.risk_calc import DEFAULT_SL_ATR_MULTIPLIER, DEFAULT_TP_ATR_MULTIPLIER
from scanner.analysis.sl_tp_calibration import (
    MAX_ATR_MULTIPLIER,
    MIN_ATR_MULTIPLIER,
    compute_label_sl_tp_calibration,
    fetch_resolved_outcomes_with_excursion,
    run_sl_tp_calibration,
)
from scanner.storage import init_db

pytestmark = pytest.mark.unit


@pytest.fixture
def conn(tmp_path):
    db_path = tmp_path / "sl_tp_calibration_test.db"
    connection = init_db(str(db_path))
    yield connection
    connection.close()


def _insert_resolved(
    conn,
    *,
    label="trend:bullish",
    direction="long",
    entry_price=100.0,
    stop_loss=90.0,
    take_profit=120.0,
    mfe_pct=15.0,
    mae_pct=-5.0,
    status="take_profit",
):
    """Writes one resolved signal_outcomes row with just the fields
    sl_tp_calibration reads. Defaults describe a long trade whose logged
    stop/target are 10%/20% away from entry (i.e. the original ATR-based
    distances), with a mild favorable/adverse excursion."""
    now = datetime.now(UTC).isoformat()
    conn.execute(
        """
        INSERT INTO signal_outcomes (
            symbol, timeframe, direction, label, logged_at, entry_price,
            stop_loss, take_profit, horizon_candles, status, resolved_at,
            exit_price, realized_return_pct, mfe_pct, mfe_price, mae_pct, mae_price
        ) VALUES ('TESTUSDT', '1h', ?, ?, ?, ?, ?, ?, 12, ?, ?, ?, 0.0, ?, ?, ?, ?)
        """,
        (
            direction,
            label,
            now,
            entry_price,
            stop_loss,
            take_profit,
            status,
            now,
            entry_price,
            mfe_pct,
            entry_price * (1 + mfe_pct / 100.0),
            mae_pct,
            entry_price * (1 + mae_pct / 100.0),
        ),
    )
    conn.commit()


class TestFetchResolvedOutcomesWithExcursion:
    def test_excludes_open_rows(self, conn):
        conn.execute(
            "INSERT INTO signal_outcomes (symbol, timeframe, direction, label, logged_at, entry_price, "
            "stop_loss, take_profit, horizon_candles, status) VALUES "
            "('TESTUSDT', '1h', 'long', 'trend:bullish', ?, 100.0, 90.0, 120.0, 12, 'open')",
            (datetime.now(UTC).isoformat(),),
        )
        conn.commit()
        assert fetch_resolved_outcomes_with_excursion(conn) == []

    def test_excludes_backfilled_zero_excursion_placeholders(self, conn):
        _insert_resolved(conn, mfe_pct=0.0, mae_pct=0.0)
        assert fetch_resolved_outcomes_with_excursion(conn) == []

    def test_includes_genuinely_zero_mae_with_nonzero_mfe(self, conn):
        # A trade that only ever moved in its favor -- a real (if unusual)
        # outcome, not a migration placeholder, since mfe_pct isn't also 0.
        _insert_resolved(conn, mfe_pct=12.0, mae_pct=0.0)
        rows = fetch_resolved_outcomes_with_excursion(conn)
        assert len(rows) == 1

    def test_includes_resolved_rows_with_usable_excursion(self, conn):
        _insert_resolved(conn)
        _insert_resolved(conn)
        rows = fetch_resolved_outcomes_with_excursion(conn)
        assert len(rows) == 2


class TestComputeLabelSlTpCalibration:
    def _rows(self, n, **kwargs):
        return [
            {
                "label": kwargs.get("label", "trend:bullish"),
                "direction": kwargs.get("direction", "long"),
                "entry_price": 100.0,
                "stop_loss": kwargs.get("stop_loss", 90.0),
                "take_profit": kwargs.get("take_profit", 120.0),
                "mfe_pct": kwargs.get("mfe_pct", 15.0),
                "mae_pct": kwargs.get("mae_pct", -5.0),
            }
            for _ in range(n)
        ]

    def test_below_min_n_is_skipped_with_a_reason(self):
        report = compute_label_sl_tp_calibration(self._rows(5), min_n=30)
        assert "trend:bullish" not in report.calibrations
        assert "trend:bullish" in report.skipped

    def test_undirected_label_is_skipped(self):
        report = compute_label_sl_tp_calibration(self._rows(40, label="anomaly:volume_spike"), min_n=30)
        assert "anomaly:volume_spike" not in report.calibrations
        assert "anomaly:volume_spike" in report.skipped

    def test_enough_samples_produces_a_calibration(self):
        report = compute_label_sl_tp_calibration(self._rows(40), min_n=30)
        assert "trend:bullish" in report.calibrations
        c = report.calibrations["trend:bullish"]
        assert c.n == 40
        assert c.direction == "long"
        assert MIN_ATR_MULTIPLIER <= c.sl_atr_multiplier <= MAX_ATR_MULTIPLIER
        assert MIN_ATR_MULTIPLIER <= c.tp_atr_multiplier <= MAX_ATR_MULTIPLIER

    def test_identical_rows_recover_the_default_multipliers(self):
        # Every row's logged sl/tp distance already came from the live
        # default multipliers, and every row's excursion exactly matches
        # that logged distance (mae == -sl_distance, mfe == tp_distance) --
        # so the calibrated multiplier should come back as the default.
        sl_distance_pct = -10.0  # (90 - 100) / 100 * 100
        tp_distance_pct = 20.0  # (120 - 100) / 100 * 100
        report = compute_label_sl_tp_calibration(
            self._rows(40, mae_pct=sl_distance_pct, mfe_pct=tp_distance_pct), min_n=30
        )
        c = report.calibrations["trend:bullish"]
        assert c.sl_atr_multiplier == pytest.approx(DEFAULT_SL_ATR_MULTIPLIER, rel=1e-6)
        assert c.tp_atr_multiplier == pytest.approx(DEFAULT_TP_ATR_MULTIPLIER, rel=1e-6)

    def test_wider_adverse_excursion_recommends_a_wider_stop(self):
        tighter = compute_label_sl_tp_calibration(self._rows(40, mae_pct=-5.0), min_n=30)
        wider = compute_label_sl_tp_calibration(self._rows(40, mae_pct=-9.0), min_n=30)
        assert (
            wider.calibrations["trend:bullish"].sl_atr_multiplier
            > tighter.calibrations["trend:bullish"].sl_atr_multiplier
        )

    def test_overrides_dict_is_directly_usable_by_backtest_symbol(self):
        report = compute_label_sl_tp_calibration(self._rows(40), min_n=30)
        overrides = report.overrides()
        assert overrides["trend:bullish"] == (
            report.calibrations["trend:bullish"].sl_atr_multiplier,
            report.calibrations["trend:bullish"].tp_atr_multiplier,
        )

    def test_empty_input_calibrates_nothing(self):
        report = compute_label_sl_tp_calibration([], min_n=30)
        assert report.calibrations == {}
        assert report.skipped == {}


class TestRunSlTpCalibration:
    def test_end_to_end_against_a_real_sqlite_db(self, tmp_path):
        db_path = tmp_path / "e2e.db"
        connection = init_db(str(db_path))
        for _ in range(35):
            _insert_resolved(connection)
        connection.close()

        report = run_sl_tp_calibration(str(db_path), min_n=30)
        assert "trend:bullish" in report.calibrations
        assert report.calibrations["trend:bullish"].n == 35
