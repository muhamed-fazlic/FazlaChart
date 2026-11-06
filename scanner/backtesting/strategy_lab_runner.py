"""Strategy Lab orchestration: the weekly snapshot (descriptive, no
holdout spend) and Layer 2's power-gated confirmation gate. See
``STRATEGY_LAB_DESIGN.md`` for the full design rationale.

Both functions are plain, directly-callable Python functions (not a tight
``while True`` loop) so a periodic trigger can call them on its own
schedule -- see ``scanner.strategy_lab_scheduler`` for the actual cadence
(mining every few hours, weekly snapshot/Layer-2 check once a week),
mirroring ``scanner.market_scanner.MarketScanner.run_forever``'s existing
scan-cycle convention rather than inventing a second one.
"""

from __future__ import annotations

import logging
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from scanner import strategy_lab_storage as storage
from scanner.backtesting import strategy_miner as miner
from scanner.backtesting.power import required_sample_size
from scanner.backtesting.rolling_holdout import (
    RollingHoldoutState,
    advance_boundary,
    fresh_window_available_ms,
    initial_boundary_ms,
)
from scanner.telegram_alerts import (
    TelegramAlertConfig,
    format_strategy_lab_confirmation_report,
    format_strategy_lab_weekly_report,
    send_telegram_message,
)

logger = logging.getLogger(__name__)

_MS_PER_DAY = 86_400_000


def _variant_for_candidate(candidate: dict[str, Any]) -> miner.TemplateVariant:
    """Reconstructs the exact ``TemplateVariant`` a stored candidate row
    came from, by matching on (template, variant_label) against the
    pre-declared grid -- so the weekly snapshot / confirmation run uses the
    IDENTICAL parameterization Layer 1 scored, not a re-derived one."""
    for variant in miner.ALL_TEMPLATE_VARIANTS:
        if variant.template == candidate["template"] and variant.variant_label == candidate["variant_label"]:
            return variant
    raise ValueError(
        f"no matching template variant for stored candidate "
        f"(template={candidate['template']!r}, variant_label={candidate['variant_label']!r}) -- "
        "the pre-declared grid may have changed since this candidate was mined"
    )


def run_weekly_snapshot(
    *,
    db_path: str = storage.DEFAULT_STRATEGY_LAB_DB_PATH,
    end_timestamp_ms: int,
    symbols: tuple[str, ...] | None = None,
    ranked_universe_cache_path: str | None = None,
    telegram_config: TelegramAlertConfig | None = None,
) -> dict[str, Any] | None:
    """Re-runs the current #1-ranked Layer 1 candidate through the SAME
    frozen backtest engine (via ``miner.evaluate_variant`` -- no
    reimplemented backtest/CI logic) on train-only data
    (``end_timestamp_ms``), to get full stats including the block-bootstrap
    CI. Stores a timestamped snapshot row and sends a Telegram report.

    Explicitly exploratory: the candidate was selected by looking at
    Layer 1's own ranking, so this snapshot's CI is NOT a valid
    confirmatory test (selection bias) -- see
    ``format_strategy_lab_weekly_report``'s disclaimer, always included.

    Returns ``None`` (nothing to snapshot) if Layer 1 has never produced a
    leaderboard yet.
    """
    conn = storage.init_db(db_path)
    try:
        leaderboard = storage.fetch_leaderboard(conn, limit=1)
        if not leaderboard:
            logger.info("Strategy Lab weekly snapshot: no Layer 1 candidates yet -- nothing to snapshot")
            return None
        top = leaderboard[0]
        variant = _variant_for_candidate(top)
        tercile_symbols = symbols
        evaluation_cache_path: str | None = None
        with ExitStack() as stack:
            if ranked_universe_cache_path is not None:
                tercile_symbols = _resolve_tercile_symbols(ranked_universe_cache_path, top["tercile"])
                # Never forward the raw ranked cache to evaluate_variant --
                # funding_extreme/cross_sectional's frozen modules expect the
                # OLD schema, not the ranked one (see tercile_universe_cache_path).
                evaluation_cache_path = stack.enter_context(
                    miner.tercile_universe_cache_path(ranked_universe_cache_path, top["tercile"])
                )
            if not tercile_symbols:
                raise ValueError("could not resolve a symbol set for the #1 candidate's tercile")

            evaluation = miner.evaluate_variant(
                variant, tuple(tercile_symbols), end_timestamp_ms, evaluation_cache_path
            )
            ci = evaluation.report.get("net_block_bootstrap")
            win_rate_pct = evaluation.report.get("win_rate_pct")

            telegram_sent = False
            snapshot_payload = {
                "template": top["template"],
                "tercile": top["tercile"],
                "variant_label": top["variant_label"],
                "net_observed_pct": evaluation.net_observed_pct,
                "net_ci_low_pct": ci["ci_low_pct"] if ci else None,
                "net_ci_high_pct": ci["ci_high_pct"] if ci else None,
                "n_independent": evaluation.n_independent,
                "win_rate_pct": win_rate_pct,
            }
            if telegram_config is not None and telegram_config.enabled:
                telegram_sent = send_telegram_message(
                    telegram_config.bot_token,
                    telegram_config.chat_id,
                    format_strategy_lab_weekly_report(snapshot_payload),
                )

            snapshot_id = storage.insert_weekly_snapshot(
                conn,
                candidate_id=top["id"],
                template=top["template"],
                tercile=top["tercile"],
                variant_label=top["variant_label"],
                n_independent=evaluation.n_independent,
                net_observed_pct=evaluation.net_observed_pct,
                net_ci_low_pct=ci["ci_low_pct"] if ci else None,
                net_ci_high_pct=ci["ci_high_pct"] if ci else None,
                gross_observed_pct=evaluation.gross_observed_pct,
                win_rate_pct=win_rate_pct,
                telegram_sent=telegram_sent,
                report=evaluation.report,
            )
            snapshot_payload["id"] = snapshot_id
            snapshot_payload["telegram_sent"] = telegram_sent
            return snapshot_payload
    finally:
        conn.close()


def _resolve_tercile_symbols(ranked_universe_cache_path: str, tercile_name: str) -> tuple[str, ...]:
    from scanner.backtesting.historical_universe import (
        load_ranked_universe_cache,
        windows_for_rank_range,
    )

    start_rank, end_rank = miner.DEFAULT_TERCILE_RANK_RANGES[tercile_name]
    ranked_windows = load_ranked_universe_cache(ranked_universe_cache_path)
    sliced = windows_for_rank_range([ranked_windows[-1]], start_rank, end_rank)
    return tuple(sorted(sliced[-1].symbols)) if sliced else ()


@dataclass(frozen=True)
class Layer2CheckResult:
    """What ``run_layer2_check`` decided this week -- either "still
    waiting" (``confirmation`` is ``None``) or a completed confirmation
    attempt (pass or fail, both stored either way)."""

    ran_confirmation: bool
    fresh_days_available: float
    n_required: int | None = None
    estimated_days_needed: float | None = None
    confirmation: dict[str, Any] | None = None


def run_layer2_check(
    *,
    db_path: str = storage.DEFAULT_STRATEGY_LAB_DB_PATH,
    shared_holdout_end_ms: int,
    now_ms: int | None = None,
    symbols: tuple[str, ...] | None = None,
    ranked_universe_cache_path: str | None = None,
    telegram_config: TelegramAlertConfig | None = None,
) -> Layer2CheckResult:
    """Weekly Layer 2 check: is there enough accumulated FRESH (never
    before used) calendar time for the current #1 candidate to meet its
    power-calculated ``n_required``? If not, just reports how much longer
    to wait (spends nothing). If yes, runs exactly ONE real confirmatory
    backtest against that fresh window, advances the rolling holdout
    boundary past it (so it can never be reused -- see
    ``scanner.backtesting.rolling_holdout.advance_boundary``), computes the
    block-bootstrap CI via the same frozen engine, records a genuine
    pass/fail verdict, and sends a Telegram report.
    """
    if now_ms is None:
        now_ms = int(datetime.now(UTC).timestamp() * 1000)

    conn = storage.init_db(db_path)
    try:
        state_row = storage.fetch_rolling_holdout_state(conn)
        if state_row is None:
            boundary_ms = initial_boundary_ms(shared_holdout_end_ms, now_ms=now_ms)
            storage.save_rolling_holdout_state(conn, boundary_ms)
        else:
            boundary_ms = state_row["boundary_timestamp_ms"]
        state = RollingHoldoutState(boundary_timestamp_ms=boundary_ms)
        fresh_ms = fresh_window_available_ms(state, now_ms=now_ms)
        fresh_days = fresh_ms / _MS_PER_DAY

        leaderboard = storage.fetch_leaderboard(conn, limit=1)
        if not leaderboard:
            logger.info("Strategy Lab Layer 2 check: no Layer 1 candidate yet -- nothing to confirm")
            return Layer2CheckResult(ran_confirmation=False, fresh_days_available=fresh_days)
        top = leaderboard[0]

        if top.get("net_observed_pct") is None or not top["n_independent"]:
            return Layer2CheckResult(ran_confirmation=False, fresh_days_available=fresh_days)

        # Project the candidate's own training-period event rate forward to
        # estimate how many fresh days are needed -- see power.estimate_days_needed.
        training_candidate = storage.fetch_mining_candidate(conn, top["id"])
        as_of_ms = training_candidate["as_of_timestamp_ms"] if training_candidate else None
        train_period_days = max((as_of_ms - shared_holdout_end_ms) / _MS_PER_DAY, 1.0) if as_of_ms else 180.0
        events_per_day = top["n_independent"] / train_period_days if train_period_days > 0 else 0.0

        sigma_pct = abs(top.get("net_observed_pct") or 0.0) or 1.0
        try:
            power = required_sample_size(sigma_pct)
        except ValueError:
            return Layer2CheckResult(ran_confirmation=False, fresh_days_available=fresh_days)

        # How many fresh events can already be expected to have accumulated.
        n_independent_so_far_estimate = int(events_per_day * fresh_days)
        from scanner.backtesting.power import estimate_days_needed

        days_needed = estimate_days_needed(
            n_required=power.n_required, n_independent_so_far=n_independent_so_far_estimate, events_per_day=events_per_day
        )

        if n_independent_so_far_estimate < power.n_required:
            logger.info(
                "Strategy Lab Layer 2: not enough power yet for %s/%s/%s -- ~%.1f more days needed "
                "(n_required=%d, estimated so far=%d)",
                top["template"],
                top["tercile"],
                top["variant_label"],
                days_needed,
                power.n_required,
                n_independent_so_far_estimate,
            )
            return Layer2CheckResult(
                ran_confirmation=False,
                fresh_days_available=fresh_days,
                n_required=power.n_required,
                estimated_days_needed=days_needed,
            )

        # Enough power: run exactly ONE real confirmatory backtest against
        # the fresh window [boundary_ms, now_ms].
        variant = _variant_for_candidate(top)
        tercile_symbols = symbols
        evaluation_cache_path: str | None = None
        with ExitStack() as stack:
            if ranked_universe_cache_path is not None:
                tercile_symbols = _resolve_tercile_symbols(ranked_universe_cache_path, top["tercile"])
                evaluation_cache_path = stack.enter_context(
                    miner.tercile_universe_cache_path(ranked_universe_cache_path, top["tercile"])
                )
            if not tercile_symbols:
                raise ValueError("could not resolve a symbol set for the #1 candidate's tercile")

            evaluation = miner.evaluate_variant(variant, tuple(tercile_symbols), now_ms, evaluation_cache_path)
            ci = evaluation.report.get("net_block_bootstrap")
            edge_survives = bool(ci and ci["edge_survives_lower_bound"])

            confirmation_payload = {
                "template": top["template"],
                "tercile": top["tercile"],
                "variant_label": top["variant_label"],
                "fresh_window_start_iso": datetime.fromtimestamp(boundary_ms / 1000.0, tz=UTC).isoformat(),
                "fresh_window_end_iso": datetime.fromtimestamp(now_ms / 1000.0, tz=UTC).isoformat(),
                "net_observed_pct": evaluation.net_observed_pct,
                "net_ci_low_pct": ci["ci_low_pct"] if ci else None,
                "net_ci_high_pct": ci["ci_high_pct"] if ci else None,
                "n_independent": evaluation.n_independent,
                "n_required": power.n_required,
                "edge_survives_lower_bound": edge_survives,
            }
            telegram_sent = False
            if telegram_config is not None and telegram_config.enabled:
                telegram_sent = send_telegram_message(
                    telegram_config.bot_token,
                    telegram_config.chat_id,
                    format_strategy_lab_confirmation_report(confirmation_payload),
                )

            storage.insert_confirmation(
                conn,
                candidate_id=top["id"],
                template=top["template"],
                tercile=top["tercile"],
                variant_label=top["variant_label"],
                fresh_window_start_ms=boundary_ms,
                fresh_window_end_ms=now_ms,
                n_independent=evaluation.n_independent,
                n_required=power.n_required,
                net_observed_pct=evaluation.net_observed_pct,
                net_ci_low_pct=ci["ci_low_pct"] if ci else None,
                net_ci_high_pct=ci["ci_high_pct"] if ci else None,
                edge_survives_lower_bound=edge_survives,
                telegram_sent=telegram_sent,
                report=evaluation.report,
            )

            new_state = advance_boundary(state, consumed_until_ms=now_ms)
            storage.save_rolling_holdout_state(conn, new_state.boundary_timestamp_ms)

            confirmation_payload["telegram_sent"] = telegram_sent
            return Layer2CheckResult(
                ran_confirmation=True,
                fresh_days_available=fresh_days,
                n_required=power.n_required,
                estimated_days_needed=0.0,
                confirmation=confirmation_payload,
            )
    finally:
        conn.close()
