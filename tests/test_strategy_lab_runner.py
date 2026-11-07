"""Tests for scanner.backtesting.strategy_lab_runner -- the weekly
snapshot and Layer 2 confirmation gate orchestration.

Network-free and backtest-free: monkeypatches ``miner.evaluate_variant``
directly (one seam, since both ``run_weekly_snapshot`` and
``run_layer2_check`` call through it) so these tests exercise the
orchestration logic (storage round-trips, power gating, rolling-holdout
boundary advancement, Telegram send wiring) without touching any frozen
backtest module.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scanner import strategy_lab_storage as storage
from scanner.backtesting import strategy_lab_runner as runner, strategy_miner as miner
from scanner.telegram_alerts import TelegramAlertConfig

SHARED_HOLDOUT_END_MS = 1_790_253_820_000
DAY_MS = 86_400_000


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    return str(tmp_path / "strategy_lab_test.db")


def _seed_boundary(db_path: str, boundary_timestamp_ms: int) -> None:
    """Pre-seeds the rolling holdout boundary to a FIXED point in the past,
    simulating "this feature started a while ago" so a later ``now_ms`` in
    a test genuinely represents accumulated fresh time -- ``now_ms``
    passed to ``run_layer2_check`` is a simulated clock, not real wall
    time, so it must not be conflated with ``initial_boundary_ms``'s own
    ``max(shared_holdout_end_ms, now)`` seeding logic."""
    conn = storage.init_db(db_path)
    try:
        storage.save_rolling_holdout_state(conn, boundary_timestamp_ms)
    finally:
        conn.close()


def _seed_candidate(db_path: str, *, net_observed_pct: float = 1.0, n_independent: int = 50) -> int:
    conn = storage.init_db(db_path)
    try:
        variant = miner.FUNDING_EXTREME_VARIANTS[0]
        candidate_id = storage.insert_mining_candidate(
            conn,
            as_of_timestamp_ms=SHARED_HOLDOUT_END_MS - 180 * DAY_MS,
            template=variant.template,
            tercile="unsliced",
            variant_label=variant.variant_label,
            params=variant.params,
            n_independent=n_independent,
            gross_observed_pct=net_observed_pct + 0.2,
            net_observed_pct=net_observed_pct,
            stability_score=1.0,
            composite_score=net_observed_pct,
            report={"n_independent": n_independent},
        )
        return candidate_id
    finally:
        conn.close()


def _seed_frozen_candidate(
    db_path: str,
    *,
    net_observed_pct: float = 1.0,
    n_independent: int = 50,
    windows_checked: int = 3,
    windows_positive: int = 3,
    btc_beta: float = 0.1,
    frozen_as_of_ts: int | None = None,
) -> int:
    """Seeds a mining_candidate row AND freezes it straight into the Layer
    2 confirmation queue -- bypassing ``check_entry_threshold`` so these
    orchestration tests can control exactly which candidate
    ``run_layer2_check`` sees, independent of the entry-threshold logic
    (which has its own dedicated tests in ``test_layer2_entry.py``).
    Returns the frozen queue row's id."""
    candidate_id = _seed_candidate(db_path, net_observed_pct=net_observed_pct, n_independent=n_independent)
    variant = miner.FUNDING_EXTREME_VARIANTS[0]
    conn = storage.init_db(db_path)
    try:
        frozen_id = storage.insert_layer2_frozen_candidate(
            conn,
            template=variant.template,
            tercile="unsliced",
            variant_label=variant.variant_label,
            frozen_as_of_ts=(
                frozen_as_of_ts if frozen_as_of_ts is not None else SHARED_HOLDOUT_END_MS - 180 * DAY_MS
            ),
            source_candidate_id=candidate_id,
            windows_checked=windows_checked,
            windows_positive=windows_positive,
            net_observed_pct=net_observed_pct,
            gross_observed_pct=net_observed_pct + 0.2,
            n_independent=n_independent,
            btc_beta=btc_beta,
        )
        assert frozen_id is not None
        return frozen_id
    finally:
        conn.close()


class TestRunWeeklySnapshot:
    def test_returns_none_when_no_candidates_yet(self, db_path: str) -> None:
        result = runner.run_weekly_snapshot(db_path=db_path, end_timestamp_ms=SHARED_HOLDOUT_END_MS)
        assert result is None

    def test_stores_and_returns_snapshot(self, db_path: str, monkeypatch) -> None:
        _seed_candidate(db_path)

        def fake_evaluate(variant, symbols, end_timestamp_ms, cache_path):
            return miner.VariantEvaluation(
                report={
                    "net_block_bootstrap": {"ci_low_pct": -0.2, "ci_high_pct": 1.8, "edge_survives_lower_bound": False},
                    "win_rate_pct": 55.0,
                },
                event_net_returns=[1.0] * 10,
                n_independent=55,
                net_observed_pct=1.1,
                gross_observed_pct=1.3,
            )

        monkeypatch.setattr(miner, "evaluate_variant", fake_evaluate)
        result = runner.run_weekly_snapshot(
            db_path=db_path, end_timestamp_ms=SHARED_HOLDOUT_END_MS, symbols=("BTCUSDT", "ETHUSDT")
        )

        assert result is not None
        assert result["net_observed_pct"] == 1.1
        assert result["telegram_sent"] is False  # no telegram_config passed

        conn = storage.connect_for_reads(db_path)
        try:
            snapshots = storage.fetch_weekly_snapshots(conn)
        finally:
            conn.close()
        assert len(snapshots) == 1
        assert snapshots[0]["n_independent"] == 55

    def test_sends_telegram_when_configured_and_enabled(self, db_path: str, monkeypatch) -> None:
        _seed_candidate(db_path)

        def fake_evaluate(variant, symbols, end_timestamp_ms, cache_path):
            return miner.VariantEvaluation(
                report={"net_block_bootstrap": None, "win_rate_pct": None},
                event_net_returns=[],
                n_independent=55,
                net_observed_pct=1.1,
                gross_observed_pct=None,
            )

        sent_messages = []
        monkeypatch.setattr(miner, "evaluate_variant", fake_evaluate)
        monkeypatch.setattr(runner, "send_telegram_message", lambda token, chat, text: sent_messages.append(text) or True)

        config = TelegramAlertConfig(bot_token="t", chat_id="c")
        result = runner.run_weekly_snapshot(
            db_path=db_path,
            end_timestamp_ms=SHARED_HOLDOUT_END_MS,
            symbols=("BTCUSDT",),
            telegram_config=config,
        )
        assert result["telegram_sent"] is True
        assert len(sent_messages) == 1
        assert "eksploratorno" in sent_messages[0].lower()


class TestRunLayer2Check:
    def test_returns_waiting_when_no_candidate(self, db_path: str) -> None:
        result = runner.run_layer2_check(
            db_path=db_path, shared_holdout_end_ms=SHARED_HOLDOUT_END_MS, now_ms=SHARED_HOLDOUT_END_MS + 5 * DAY_MS
        )
        assert result.ran_confirmation is False
        assert result.confirmation is None
        assert result.candidate is None

    def test_waits_when_insufficient_fresh_power(self, db_path: str) -> None:
        # Small net effect -> large sigma assumption -> huge n_required,
        # while only 5 days have passed -> clearly insufficient.
        _seed_frozen_candidate(db_path, net_observed_pct=10.0, n_independent=100)
        _seed_boundary(db_path, SHARED_HOLDOUT_END_MS)
        result = runner.run_layer2_check(
            db_path=db_path, shared_holdout_end_ms=SHARED_HOLDOUT_END_MS, now_ms=SHARED_HOLDOUT_END_MS + 5 * DAY_MS
        )
        assert result.ran_confirmation is False
        assert result.ready_for_manual_trigger is False
        assert result.candidate is not None
        assert result.n_required is not None
        assert result.estimated_days_needed is not None and result.estimated_days_needed > 0

        # Boundary must have been initialized but NOT advanced (nothing consumed).
        conn = storage.connect_for_reads(db_path)
        try:
            state = storage.fetch_rolling_holdout_state(conn)
        finally:
            conn.close()
        assert state is not None
        assert state["boundary_timestamp_ms"] == SHARED_HOLDOUT_END_MS

    def test_reports_ready_for_manual_trigger_but_does_not_spend_without_it(self, db_path: str, monkeypatch) -> None:
        # Large effect/high event rate in training -> small n_required,
        # easily met by a long fresh window -- but manual_trigger defaults
        # to False, so NOTHING should be spent/consumed/advanced yet.
        _seed_frozen_candidate(db_path, net_observed_pct=5.0, n_independent=2000)
        _seed_boundary(db_path, SHARED_HOLDOUT_END_MS)
        now_ms = SHARED_HOLDOUT_END_MS + 400 * DAY_MS

        def fake_evaluate(variant, symbols, end_timestamp_ms, cache_path, bootstrap_confidence_pct=None, start_timestamp_ms=None):
            raise AssertionError("evaluate_variant must not be called when manual_trigger is False")

        monkeypatch.setattr(miner, "evaluate_variant", fake_evaluate)
        result = runner.run_layer2_check(
            db_path=db_path,
            shared_holdout_end_ms=SHARED_HOLDOUT_END_MS,
            now_ms=now_ms,
            symbols=("BTCUSDT", "ETHUSDT"),
        )

        assert result.ran_confirmation is False
        assert result.ready_for_manual_trigger is True
        assert result.candidate is not None

        conn = storage.connect_for_reads(db_path)
        try:
            confirmations = storage.fetch_confirmations(conn)
            state = storage.fetch_rolling_holdout_state(conn)
            queue = storage.fetch_layer2_candidate_queue(conn)
        finally:
            conn.close()
        assert confirmations == []
        assert state["boundary_timestamp_ms"] == SHARED_HOLDOUT_END_MS  # untouched
        assert queue[0]["consumed"] == 0  # untouched

    def test_runs_confirmation_and_advances_boundary_when_power_sufficient(self, db_path: str, monkeypatch) -> None:
        # Large effect/high event rate in training -> small n_required,
        # easily met by a long fresh window.
        _seed_frozen_candidate(db_path, net_observed_pct=5.0, n_independent=2000)
        _seed_boundary(db_path, SHARED_HOLDOUT_END_MS)
        now_ms = SHARED_HOLDOUT_END_MS + 400 * DAY_MS

        def fake_evaluate(variant, symbols, end_timestamp_ms, cache_path, bootstrap_confidence_pct=None, start_timestamp_ms=None):
            return miner.VariantEvaluation(
                report={"net_block_bootstrap": {"ci_low_pct": 0.5, "ci_high_pct": 2.0, "edge_survives_lower_bound": True}},
                event_net_returns=[1.0] * 500,
                n_independent=500,
                net_observed_pct=1.2,
                gross_observed_pct=1.4,
            )

        monkeypatch.setattr(miner, "evaluate_variant", fake_evaluate)
        result = runner.run_layer2_check(
            db_path=db_path,
            shared_holdout_end_ms=SHARED_HOLDOUT_END_MS,
            now_ms=now_ms,
            symbols=("BTCUSDT", "ETHUSDT"),
            manual_trigger=True,
        )

        assert result.ran_confirmation is True
        assert result.confirmation is not None
        assert result.confirmation["edge_survives_lower_bound"] is True

        conn = storage.connect_for_reads(db_path)
        try:
            confirmations = storage.fetch_confirmations(conn)
            state = storage.fetch_rolling_holdout_state(conn)
            best = storage.fetch_current_best(conn)
            queue = storage.fetch_layer2_candidate_queue(conn)
        finally:
            conn.close()
        assert len(confirmations) == 1
        assert state["boundary_timestamp_ms"] == now_ms
        assert best is not None
        assert best["edge_survives_lower_bound"] == 1
        assert queue[0]["consumed"] == 1  # spent exactly once, win or lose

    def test_second_confirmation_cannot_reuse_consumed_window(self, db_path: str, monkeypatch) -> None:
        _seed_frozen_candidate(db_path, net_observed_pct=5.0, n_independent=2000)
        _seed_boundary(db_path, SHARED_HOLDOUT_END_MS)
        now_ms = SHARED_HOLDOUT_END_MS + 400 * DAY_MS

        def fake_evaluate(variant, symbols, end_timestamp_ms, cache_path, bootstrap_confidence_pct=None, start_timestamp_ms=None):
            return miner.VariantEvaluation(
                report={"net_block_bootstrap": {"ci_low_pct": 0.5, "ci_high_pct": 2.0, "edge_survives_lower_bound": True}},
                event_net_returns=[1.0] * 500,
                n_independent=500,
                net_observed_pct=1.2,
                gross_observed_pct=1.4,
            )

        monkeypatch.setattr(miner, "evaluate_variant", fake_evaluate)
        first = runner.run_layer2_check(
            db_path=db_path,
            shared_holdout_end_ms=SHARED_HOLDOUT_END_MS,
            now_ms=now_ms,
            symbols=("BTCUSDT",),
            manual_trigger=True,
        )
        assert first.ran_confirmation is True

        # The one frozen candidate is now consumed -- a second call finds
        # an empty queue (no second config qualified), rather than
        # re-running a confirmation over an already-fully-consumed
        # (zero-length) fresh window.
        second = runner.run_layer2_check(
            db_path=db_path,
            shared_holdout_end_ms=SHARED_HOLDOUT_END_MS,
            now_ms=now_ms,
            symbols=("BTCUSDT",),
            manual_trigger=True,
        )
        assert second.ran_confirmation is False
        assert second.candidate is None
        assert second.fresh_days_available == 0.0

    def test_confirmation_isolation_boundary_uses_candidates_own_later_t_not_the_stale_global_boundary(
        self, db_path: str, monkeypatch
    ) -> None:
        """Korak 10 item 2/6 leakage fix: the confirmatory backtest's
        isolation boundary must be ``max(boundary_ms, candidate's own
        frozen_as_of_ts)``, per-candidate -- never the bare global
        rolling-holdout boundary alone. Here the global boundary is
        seeded OLDER than the candidate's own ``frozen_as_of_ts`` (e.g.
        because an earlier, different candidate's confirmation attempt
        left a stale, earlier marker); using the global boundary alone
        would let this candidate's confirmation see calendar time its own
        training data already used -- exactly the leak the fix closes."""
        frozen_as_of_ts = SHARED_HOLDOUT_END_MS + 50 * DAY_MS
        _seed_frozen_candidate(db_path, net_observed_pct=5.0, n_independent=2000, frozen_as_of_ts=frozen_as_of_ts)
        _seed_boundary(db_path, SHARED_HOLDOUT_END_MS)  # stale: older than the candidate's own T
        now_ms = SHARED_HOLDOUT_END_MS + 400 * DAY_MS

        captured: dict[str, object] = {}

        def fake_evaluate(variant, symbols, end_timestamp_ms, cache_path, bootstrap_confidence_pct=None, start_timestamp_ms=None):
            captured["start_timestamp_ms"] = start_timestamp_ms
            return miner.VariantEvaluation(
                report={"net_block_bootstrap": {"ci_low_pct": 0.5, "ci_high_pct": 2.0, "edge_survives_lower_bound": True}},
                event_net_returns=[1.0] * 500,
                n_independent=500,
                net_observed_pct=1.2,
                gross_observed_pct=1.4,
            )

        monkeypatch.setattr(miner, "evaluate_variant", fake_evaluate)
        result = runner.run_layer2_check(
            db_path=db_path,
            shared_holdout_end_ms=SHARED_HOLDOUT_END_MS,
            now_ms=now_ms,
            symbols=("BTCUSDT",),
            manual_trigger=True,
        )

        assert result.ran_confirmation is True
        # The boundary actually passed through to the backtest must be the
        # candidate's own (later) T, not the stale (earlier) global one.
        assert captured["start_timestamp_ms"] == frozen_as_of_ts

        conn = storage.connect_for_reads(db_path)
        try:
            confirmations = storage.fetch_confirmations(conn)
        finally:
            conn.close()
        assert len(confirmations) == 1
        assert confirmations[0]["fresh_window_start_ms"] == frozen_as_of_ts
