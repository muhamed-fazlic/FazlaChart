"""Unit tests for Phase 8 Korak 3's experiment registry — the permanent
trial-count ledger that feeds honest ``n_trials`` figures into the
Deflated Sharpe Ratio / Benjamini-Hochberg corrections. SQLite only, no
network calls."""

from __future__ import annotations

import pytest

from scanner.validation.experiment_registry import (
    count_prior_trials,
    fetch_experiments,
    init_db,
    record_experiment,
)

pytestmark = pytest.mark.unit


def test_record_and_fetch_round_trips_all_fields(tmp_path):
    conn = init_db(str(tmp_path / "experiments.db"))

    experiment_id = record_experiment(
        conn,
        strategy_family="portfolio_backtest_v1",
        strategy_version="v1",
        timeframe="1h",
        label="trend:bearish",
        horizon_candles=12,
        sl_tp_config={"sl_atr_multiplier": 1.5, "tp_atr_multiplier": 3.0},
        train_period=("2025-01-01", "2025-06-01"),
        validation_period=("2025-06-01", "2025-09-01"),
        test_period=("2025-09-01", "2025-12-01"),
        n_trades=302,
        sharpe_ratio=-3.92,
        net_profit_pct=-45.15,
        result={"max_drawdown_pct": 46.98},
        notes="initial 20-symbol dashboard report run",
    )

    assert experiment_id == 1
    rows = fetch_experiments(conn, strategy_family="portfolio_backtest_v1")
    assert len(rows) == 1
    row = rows[0]
    assert row["strategy_family"] == "portfolio_backtest_v1"
    assert row["timeframe"] == "1h"
    assert row["sl_tp_config"] == {"sl_atr_multiplier": 1.5, "tp_atr_multiplier": 3.0}
    assert row["train_period_start"] == "2025-01-01"
    assert row["train_period_end"] == "2025-06-01"
    assert row["n_trades"] == 302
    assert row["result"] == {"max_drawdown_pct": 46.98}
    assert row["notes"] == "initial 20-symbol dashboard report run"


def test_record_experiment_with_minimal_fields(tmp_path):
    conn = init_db(str(tmp_path / "experiments.db"))
    experiment_id = record_experiment(conn, strategy_family="quick_probe")
    assert experiment_id == 1
    row = fetch_experiments(conn, strategy_family="quick_probe")[0]
    assert row["sl_tp_config"] is None
    assert row["result"] is None
    assert row["train_period_start"] is None


def test_count_prior_trials_scopes_by_family(tmp_path):
    conn = init_db(str(tmp_path / "experiments.db"))
    record_experiment(conn, strategy_family="family_a")
    record_experiment(conn, strategy_family="family_a")
    record_experiment(conn, strategy_family="family_b")

    assert count_prior_trials(conn, "family_a") == 2
    assert count_prior_trials(conn, "family_b") == 1
    assert count_prior_trials(conn, "family_c") == 0


def test_count_prior_trials_respects_before_cutoff(tmp_path):
    conn = init_db(str(tmp_path / "experiments.db"))
    record_experiment(conn, strategy_family="family_a")
    cutoff = conn.execute("SELECT recorded_at FROM experiments").fetchone()[0]
    record_experiment(conn, strategy_family="family_a")

    assert count_prior_trials(conn, "family_a", before=cutoff) == 1
    assert count_prior_trials(conn, "family_a") == 2


def test_fetch_experiments_without_family_returns_everything_most_recent_first(tmp_path):
    conn = init_db(str(tmp_path / "experiments.db"))
    record_experiment(conn, strategy_family="family_a", notes="first")
    record_experiment(conn, strategy_family="family_b", notes="second")

    rows = fetch_experiments(conn)
    assert [r["notes"] for r in rows] == ["second", "first"]


def test_fetch_experiments_respects_limit(tmp_path):
    conn = init_db(str(tmp_path / "experiments.db"))
    for i in range(5):
        record_experiment(conn, strategy_family="family_a", notes=f"trial-{i}")

    rows = fetch_experiments(conn, strategy_family="family_a", limit=2)
    assert len(rows) == 2
