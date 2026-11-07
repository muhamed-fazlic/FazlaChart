"""Tests for scanner.strategy_lab_scheduler -- focused on StrategyLabScheduler.run_mining_once
not re-mining combinations already stored for the current as_of_ts.

Regression coverage for a real production bug: once real time passes the
frozen SHARED_HOLDOUT_END_MS boundary, as_of_ts is clamped to that same
value every single cycle forever. Without skipping already-mined
combinations, every scheduled mining cycle re-ran (and re-inserted) the
exact same (template, tercile, variant) results over and over -- wasting
CPU on guaranteed-identical backtests and filling the dashboard's "Layer 1
top 10" leaderboard with duplicate rows of the same candidate."""

from __future__ import annotations

from scanner import strategy_lab_scheduler as scheduler_module, strategy_lab_storage as storage
from scanner.backtesting.strategy_miner import MiningCandidate


def _make_candidate(template: str, tercile: str, variant_label: str) -> MiningCandidate:
    return MiningCandidate(
        template=template,
        tercile=tercile,
        variant_label=variant_label,
        params={},
        n_independent=50,
        gross_observed_pct=0.5,
        net_observed_pct=0.3,
        stability_score=1.0,
        composite_score=0.3,
        report={},
    )


class TestRunMiningOnceSkipsAlreadyMinedCombos:
    def test_passes_already_mined_keys_from_storage_to_run_mining_cycle(self, tmp_path, monkeypatch) -> None:
        db_path = str(tmp_path / "strategy_lab.db")
        conn = storage.init_db(db_path)
        as_of_ts = scheduler_module.SHARED_HOLDOUT_END_MS
        storage.insert_mining_candidate(
            conn,
            as_of_timestamp_ms=as_of_ts,
            template="trend_anomaly",
            tercile="unsliced",
            variant_label="Original signal",
            params={},
            n_independent=50,
            gross_observed_pct=0.5,
            net_observed_pct=0.3,
            stability_score=1.0,
            composite_score=0.3,
            report={},
        )
        conn.close()

        captured: dict[str, object] = {}

        def fake_run_mining_cycle(as_of_ts_arg, *, symbols, ranked_universe_cache_path, template_variants, already_mined_keys):
            captured["already_mined_keys"] = already_mined_keys
            # Only the NEW combination should be produced -- the
            # already-mined one must not be re-run.
            return [_make_candidate("trend_anomaly", "unsliced", "Fade (invert) signal")]

        monkeypatch.setattr(scheduler_module, "run_mining_cycle", fake_run_mining_cycle)
        # Freeze "now" past the holdout boundary so as_of_ts clamps to the
        # exact same value as the pre-seeded row above.
        monkeypatch.setattr(
            scheduler_module,
            "datetime",
            type("FrozenDatetime", (), {"now": staticmethod(lambda tz=None: __import__("datetime").datetime(2030, 1, 1, tzinfo=tz))}),
        )

        sched = scheduler_module.StrategyLabScheduler(db_path=db_path, symbols=("BTCUSDT",))
        stored_count = sched.run_mining_once()

        assert ("trend_anomaly", "unsliced", "Original signal") in captured["already_mined_keys"]
        assert stored_count == 1

        conn = storage.init_db(db_path)
        try:
            rows = conn.execute(
                "SELECT variant_label FROM mining_candidates WHERE as_of_timestamp_ms = ? ORDER BY variant_label",
                (as_of_ts,),
            ).fetchall()
        finally:
            conn.close()
        # The pre-seeded "Original signal" row must still be there exactly
        # once (not duplicated), plus the one genuinely new candidate.
        assert [r[0] for r in rows] == ["Fade (invert) signal", "Original signal"]


class TestWeeklyRunStateSurvivesRestart:
    """Regression coverage for a real production bug: the scheduler used to
    track "when did the weekly snapshot last run" only in an in-process
    ``time.monotonic()`` field, starting at ``None`` every time. That meant
    every routine deployment restart of the systemd service immediately
    re-triggered an out-of-schedule weekly snapshot (and its Telegram
    message), no matter how recently the real last one had actually run."""

    def test_init_seeds_last_weekly_run_at_from_existing_snapshot(self, tmp_path) -> None:
        db_path = str(tmp_path / "strategy_lab.db")
        conn = storage.init_db(db_path)
        storage.insert_weekly_snapshot(
            conn,
            candidate_id=None,
            template="trend_anomaly",
            tercile="unsliced",
            variant_label="Original signal",
            n_independent=50,
            net_observed_pct=0.3,
            net_ci_low_pct=None,
            net_ci_high_pct=None,
            gross_observed_pct=0.5,
            win_rate_pct=50.0,
            telegram_sent=True,
            report={},
        )
        conn.close()

        # Simulating a process restart: a brand new scheduler instance must
        # pick up the already-persisted snapshot time instead of starting
        # from "never ran" (which would immediately re-fire).
        sched = scheduler_module.StrategyLabScheduler(db_path=db_path, symbols=("BTCUSDT",))

        assert sched._last_weekly_run_at is not None

    def test_init_leaves_last_weekly_run_at_none_when_no_snapshot_exists_yet(self, tmp_path) -> None:
        db_path = str(tmp_path / "strategy_lab.db")
        storage.init_db(db_path).close()

        sched = scheduler_module.StrategyLabScheduler(db_path=db_path, symbols=("BTCUSDT",))

        # First-ever run of the service genuinely has no prior weekly
        # snapshot, so it is correct (and desired) for it to run once.
        assert sched._last_weekly_run_at is None
