"""Strategy Lab's scheduler: drives Layer 1 mining every few hours and the
weekly snapshot + Layer 2 confirmation check once a week, mirroring
``scanner.market_scanner.MarketScanner.run_forever``'s poll-loop convention
(single always-running process, log-and-continue on per-cycle errors,
sleep for the remaining interval) rather than introducing a second
scheduling mechanism.

Run continuously with ``python -m scanner.strategy_lab_scheduler`` as its
OWN systemd unit, separate from the main Market Scanner service, since the
two have unrelated cadences and failure domains.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import UTC, datetime

from scanner import strategy_lab_storage as storage
from scanner.backtesting import layer2_entry
from scanner.backtesting.strategy_lab_runner import run_layer2_check, run_weekly_snapshot
from scanner.backtesting.strategy_miner import (
    ALL_TEMPLATE_VARIANTS,
    CLOSED_PRIOR_TEMPLATES,
    MAX_UNIQUE_CONFIGURATIONS,
    ROLLING_AS_OF_TEMPLATES,
    run_mining_cycle,
    select_variants_within_unique_config_cap,
)
from scanner.telegram_alerts import TelegramAlertConfig

logger = logging.getLogger(__name__)

# The frozen train/holdout boundary already locked in for the three closed
# research families (``scanner_data/research_registry.json``'s
# ``shared_holdout.end_timestamp_ms``) -- Layer 1 mining NEVER looks past
# this for those three; Layer 2's rolling holdout boundary starts no
# earlier than it. The three CLOSED_PRIOR_TEMPLATES stay pinned here
# forever (Korak 10): they are never re-opened, never given new variants,
# never Layer 2 eligible.
SHARED_HOLDOUT_END_MS = 1_790_253_820_000

# Korak 10 item 1: the three ROLLING_AS_OF_TEMPLATES are mined against a
# forward-moving ``as_of_ts`` instead, computed fresh each cycle by
# ``compute_rolling_as_of_ts`` -- a 14-day embargo (so the most recent
# days, which may still have incomplete candle/funding data, are never
# used) snapped down to the nearest whole week. Snapping to whole weeks
# means repeated mining cycles WITHIN the same week reproduce the
# identical ``as_of_ts``/``dataset_hash`` (correctly deduped, see
# ``strategy_lab_storage.fetch_mined_dataset_hashes``), while a new week
# produces a genuinely new one -- this is what lets these templates'
# leaderboard rows actually change over time instead of discovering the
# same combinations forever.
ROLLING_AS_OF_EMBARGO_MS = 14 * 86_400_000
ROLLING_AS_OF_SNAP_MS = 7 * 86_400_000


def compute_rolling_as_of_ts(now_ms: int | None = None) -> int:
    """Pure function of wall-clock time -- no persisted state needed. See
    ``ROLLING_AS_OF_EMBARGO_MS``/``ROLLING_AS_OF_SNAP_MS`` above."""
    if now_ms is None:
        now_ms = int(datetime.now(UTC).timestamp() * 1000)
    return ((now_ms - ROLLING_AS_OF_EMBARGO_MS) // ROLLING_AS_OF_SNAP_MS) * ROLLING_AS_OF_SNAP_MS


DEFAULT_MINING_INTERVAL_SECONDS = 6 * 3600  # every few hours
DEFAULT_WEEKLY_INTERVAL_SECONDS = 7 * 24 * 3600  # once a week


class StrategyLabScheduler:
    def __init__(
        self,
        *,
        db_path: str = storage.DEFAULT_STRATEGY_LAB_DB_PATH,
        symbols: tuple[str, ...] | None = None,
        ranked_universe_cache_path: str | None = None,
        mining_interval_seconds: int = DEFAULT_MINING_INTERVAL_SECONDS,
        weekly_interval_seconds: int = DEFAULT_WEEKLY_INTERVAL_SECONDS,
        telegram_config: TelegramAlertConfig | None = None,
    ) -> None:
        self.db_path = db_path
        self.symbols = symbols
        self.ranked_universe_cache_path = ranked_universe_cache_path
        self.mining_interval_seconds = mining_interval_seconds
        self.weekly_interval_seconds = weekly_interval_seconds
        self.telegram_config = telegram_config or TelegramAlertConfig(
            bot_token=os.environ.get("TELEGRAM_BOT_TOKEN") or None,
            chat_id=os.environ.get("TELEGRAM_CHAT_ID") or None,
        )
        # Seeded from storage (NOT left as None / a fresh time.monotonic()
        # clock) so a process restart can never by itself re-trigger an
        # out-of-schedule weekly snapshot + Telegram send -- see
        # storage.fetch_latest_weekly_snapshot_at's docstring for the real
        # bug this prevents. Wall-clock (UTC datetime), not monotonic,
        # since monotonic clocks cannot be compared across process restarts.
        conn = storage.init_db(self.db_path)
        try:
            self._last_weekly_run_at: datetime | None = storage.fetch_latest_weekly_snapshot_at(conn)
            # Korak 10 item 2/6 leakage fix: the rolling holdout boundary's
            # first-ever value is bootstrapped HERE, at this
            # owning-process's startup -- never lazily from
            # ``run_layer2_check``'s read-only path (the dashboard's
            # overview endpoint calls it with ``manual_trigger=False`` on
            # every page load, which must never have a database side
            # effect). This keeps the anchor fixed and stable even if the
            # dashboard is the very first thing to ever query it.
            if storage.fetch_rolling_holdout_state(conn) is None:
                from scanner.backtesting.rolling_holdout import initial_boundary_ms

                storage.save_rolling_holdout_state(
                    conn, initial_boundary_ms(SHARED_HOLDOUT_END_MS, now_ms=int(datetime.now(UTC).timestamp() * 1000))
                )
        finally:
            conn.close()


    def run_mining_once(self) -> int:
        """Runs one Layer 1 mining cycle, persisting every evaluated
        candidate, then checks the freshly-mined rolling-template configs
        against the Layer 2 entry threshold (Korak 10 item 5), freezing
        any that newly qualify. Returns the number of NEW candidate rows
        stored (closed-family skips + rolling-family dataset_hash dedup
        skips are not counted).

        Two separate ``run_mining_cycle`` calls, not one:
        * ``CLOSED_PRIOR_TEMPLATES`` at the fixed ``SHARED_HOLDOUT_END_MS``
          -- unchanged behaviour, pre-filtered via
          ``fetch_mined_combo_keys`` so an already-mined combination is
          never even re-run (its result is guaranteed byte-identical).
        * ``ROLLING_AS_OF_TEMPLATES`` at ``compute_rolling_as_of_ts()`` --
          ALWAYS re-run every cycle (cheap: disk-cached data, and a new
          week needs genuinely fresh results), but POST-HOC deduped by
          ``(template, tercile, variant_label, dataset_hash)`` before
          insert, so re-running within the same week never produces
          duplicate leaderboard rows.

        Both calls are first passed through
        ``select_variants_within_unique_config_cap`` against the TOTAL
        unique-configuration budget (Korak 10 item 4) -- a brand new
        variant is silently skipped (logged, not inserted, not mined) once
        the 300 cap is reached; already-seen configs are never gated by
        this.
        """
        conn = storage.init_db(self.db_path)
        try:
            ever_mined_keys = storage.fetch_ever_mined_combo_keys(conn)
            admitted_variants = select_variants_within_unique_config_cap(
                ALL_TEMPLATE_VARIANTS,
                tercile_names=self._tercile_names(),
                ever_mined_keys=ever_mined_keys,
                max_unique_configurations=MAX_UNIQUE_CONFIGURATIONS,
            )
            closed_variants = tuple(v for v in admitted_variants if v.template in CLOSED_PRIOR_TEMPLATES)
            rolling_variants = tuple(v for v in admitted_variants if v.template in ROLLING_AS_OF_TEMPLATES)

            stored_count = 0

            if closed_variants:
                closed_as_of_ts = min(int(datetime.now(UTC).timestamp() * 1000), SHARED_HOLDOUT_END_MS)
                already_mined_keys = storage.fetch_mined_combo_keys(conn, closed_as_of_ts)
                closed_candidates = run_mining_cycle(
                    closed_as_of_ts,
                    symbols=self.symbols,
                    ranked_universe_cache_path=self.ranked_universe_cache_path,
                    template_variants=closed_variants,
                    already_mined_keys=already_mined_keys,
                )
                for candidate in closed_candidates:
                    storage.insert_mining_candidate(
                        conn,
                        as_of_timestamp_ms=closed_as_of_ts,
                        template=candidate.template,
                        tercile=candidate.tercile,
                        variant_label=candidate.variant_label,
                        params=candidate.params,
                        n_independent=candidate.n_independent,
                        gross_observed_pct=candidate.gross_observed_pct,
                        net_observed_pct=candidate.net_observed_pct,
                        stability_score=candidate.stability_score,
                        composite_score=candidate.composite_score,
                        report=candidate.report,
                        dataset_hash=candidate.report.get("dataset_hash"),
                    )
                    stored_count += 1

            if rolling_variants:
                rolling_as_of_ts = compute_rolling_as_of_ts()
                rolling_candidates = run_mining_cycle(
                    rolling_as_of_ts,
                    symbols=self.symbols,
                    ranked_universe_cache_path=self.ranked_universe_cache_path,
                    template_variants=rolling_variants,
                    already_mined_keys=frozenset(),  # always re-run; dedup is post-hoc below
                )
                already_dataset_hashes = storage.fetch_mined_dataset_hashes(conn)
                for candidate in rolling_candidates:
                    dataset_hash = candidate.report.get("dataset_hash")
                    dedup_key = (candidate.template, candidate.tercile, candidate.variant_label, dataset_hash)
                    if dataset_hash is not None and dedup_key in already_dataset_hashes:
                        logger.debug(
                            "Strategy Lab rolling dedup: %s/%s/%s already stored for dataset_hash=%s, skipping insert",
                            candidate.template,
                            candidate.tercile,
                            candidate.variant_label,
                            dataset_hash,
                        )
                    else:
                        storage.insert_mining_candidate(
                            conn,
                            as_of_timestamp_ms=rolling_as_of_ts,
                            template=candidate.template,
                            tercile=candidate.tercile,
                            variant_label=candidate.variant_label,
                            params=candidate.params,
                            n_independent=candidate.n_independent,
                            gross_observed_pct=candidate.gross_observed_pct,
                            net_observed_pct=candidate.net_observed_pct,
                            stability_score=candidate.stability_score,
                            composite_score=candidate.composite_score,
                            report=candidate.report,
                            dataset_hash=dataset_hash,
                        )
                        stored_count += 1
                        already_dataset_hashes = already_dataset_hashes | {dedup_key}

                self._check_and_freeze_layer2_entries(conn, rolling_candidates)
        finally:
            conn.close()
        logger.info("Strategy Lab mining cycle stored %d new candidates", stored_count)
        return stored_count

    def _tercile_names(self) -> tuple[str, ...]:
        """The tercile names a mining cycle will actually use -- mirrors
        ``run_mining_cycle``'s own fallback logic exactly (the ranked-cache
        tercile set if configured, else the single ``"unsliced"`` fallback)
        so the unique-config cap counts the SAME keys the mining cycle
        itself will produce."""
        if self.ranked_universe_cache_path is not None:
            from scanner.backtesting.strategy_miner import DEFAULT_TERCILE_RANK_RANGES

            return tuple(DEFAULT_TERCILE_RANK_RANGES.keys())
        return ("unsliced",)

    def _check_and_freeze_layer2_entries(self, conn, rolling_candidates) -> None:
        """Korak 10 item 5: after a rolling-template mining pass, checks
        each freshly-evaluated config against the Layer 2 entry threshold
        and freezes any that newly qualify into the confirmation queue.
        Uses THIS cycle's own freshly-computed report for
        ``latest_btc_beta`` (correct even when the row itself was a
        dataset_hash dedup-skip, since a dedup match means the stored
        value is identical anyway) rather than re-querying storage for it."""
        seen_configs: set[tuple[str, str, str]] = set()
        for candidate in rolling_candidates:
            key = (candidate.template, candidate.tercile, candidate.variant_label)
            if key in seen_configs:
                continue
            seen_configs.add(key)

            recent_results = storage.fetch_recent_results_for_config(
                conn,
                candidate.template,
                candidate.tercile,
                candidate.variant_label,
                limit=layer2_entry.ENTRY_WINDOW,
            )
            beta_check = candidate.report.get("beta_check") or {}
            latest_btc_beta = beta_check.get("correlation_with_btc")

            threshold_result = layer2_entry.check_entry_threshold(recent_results, latest_btc_beta=latest_btc_beta)
            if not threshold_result.qualifies:
                continue

            source_candidate_id = recent_results[0]["id"] if recent_results else None
            frozen_id = storage.insert_layer2_frozen_candidate(
                conn,
                template=candidate.template,
                tercile=candidate.tercile,
                variant_label=candidate.variant_label,
                frozen_as_of_ts=recent_results[0]["as_of_timestamp_ms"] if recent_results else compute_rolling_as_of_ts(),
                source_candidate_id=source_candidate_id,
                windows_checked=threshold_result.windows_checked,
                windows_positive=threshold_result.windows_positive,
                net_observed_pct=candidate.net_observed_pct,
                gross_observed_pct=candidate.gross_observed_pct,
                n_independent=candidate.n_independent,
                btc_beta=latest_btc_beta,
            )
            if frozen_id is not None:
                logger.info(
                    "Strategy Lab Layer 2 entry: %s/%s/%s newly qualified and frozen into the confirmation "
                    "queue (%s)",
                    candidate.template,
                    candidate.tercile,
                    candidate.variant_label,
                    threshold_result.reason,
                )

    def run_weekly_once(self, *, manual_trigger_layer2: bool = False) -> None:
        """Runs the weekly snapshot, then the Layer 2 power-gated check,
        in that order (the snapshot is always safe/descriptive; Layer 2
        only spends fresh data when power is sufficient AND
        ``manual_trigger_layer2=True`` -- the scheduled/automatic path
        (``run_forever``) always leaves this ``False``, per Korak 10 item
        6's "ne automatski": the scheduler may only ever REPORT power-gate
        readiness, never itself spend the one-shot confirmatory window.
        Only an explicit owner-invoked ``--confirm-layer2`` CLI run passes
        ``True``)."""
        snapshot = run_weekly_snapshot(
            db_path=self.db_path,
            end_timestamp_ms=SHARED_HOLDOUT_END_MS,
            symbols=self.symbols,
            ranked_universe_cache_path=self.ranked_universe_cache_path,
            telegram_config=self.telegram_config,
        )
        if snapshot is None:
            logger.info("Strategy Lab weekly: no Layer 1 candidate yet, skipping snapshot")
        else:
            logger.info("Strategy Lab weekly snapshot stored: %s", snapshot)

        layer2_result = run_layer2_check(
            db_path=self.db_path,
            shared_holdout_end_ms=SHARED_HOLDOUT_END_MS,
            symbols=self.symbols,
            ranked_universe_cache_path=self.ranked_universe_cache_path,
            telegram_config=self.telegram_config,
            manual_trigger=manual_trigger_layer2,
        )
        if layer2_result.ran_confirmation:
            logger.info("Strategy Lab Layer 2 confirmation ran: %s", layer2_result.confirmation)
        elif layer2_result.ready_for_manual_trigger:
            logger.info(
                "Strategy Lab Layer 2: %s is power-ready and awaiting the owner's manual "
                "--confirm-layer2 trigger (nothing spent yet)",
                layer2_result.candidate,
            )
        elif layer2_result.candidate is None:
            logger.info("Strategy Lab Layer 2: confirmation queue is empty, nothing to check")
        else:
            logger.info(
                "Strategy Lab Layer 2: waiting on %s (~%.1f fresh days available, ~%.1f more days needed)",
                layer2_result.candidate,
                layer2_result.fresh_days_available,
                layer2_result.estimated_days_needed or 0.0,
            )

    def run_forever(self) -> None:
        logger.info(
            "Strategy Lab scheduler starting: mining every %ss, weekly snapshot/Layer-2 every %ss",
            self.mining_interval_seconds,
            self.weekly_interval_seconds,
        )
        while True:
            cycle_start_monotonic = time.monotonic()
            cycle_start_wall = datetime.now(UTC)
            try:
                self.run_mining_once()
            except Exception:
                logger.exception("Strategy Lab mining cycle failed; continuing on the next interval")

            if (
                self._last_weekly_run_at is None
                or (cycle_start_wall - self._last_weekly_run_at).total_seconds() >= self.weekly_interval_seconds
            ):
                try:
                    self.run_weekly_once()
                except Exception:
                    logger.exception("Strategy Lab weekly snapshot/Layer-2 check failed; continuing")
                self._last_weekly_run_at = cycle_start_wall

            elapsed = time.monotonic() - cycle_start_monotonic
            sleep_for = max(0.0, self.mining_interval_seconds - elapsed)
            time.sleep(sleep_for)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strategy Lab scheduler (Layer 1 mining + weekly snapshot/Layer 2)")
    parser.add_argument("--symbols", type=str, default="", help="Comma-separated symbols, e.g. BTCUSDT,ETHUSDT")
    parser.add_argument(
        "--ranked-universe-cache",
        type=str,
        default="",
        help="Path to a ranked point-in-time universe cache (see scanner.backtesting.historical_universe) for"
        " the real 3-tercile mining run; falls back to --symbols as a single flat tercile if omitted",
    )
    parser.add_argument("--mining-interval", type=int, default=DEFAULT_MINING_INTERVAL_SECONDS)
    parser.add_argument("--weekly-interval", type=int, default=DEFAULT_WEEKLY_INTERVAL_SECONDS)
    parser.add_argument("--storage", type=str, default=storage.DEFAULT_STRATEGY_LAB_DB_PATH)
    parser.add_argument("--once", action="store_true", help="Run one mining cycle (+ weekly check) and exit")
    parser.add_argument(
        "--confirm-layer2",
        action="store_true",
        help="Owner-only: if the Layer 2 confirmation queue's oldest candidate is power-ready, actually run its"
        " ONE confirmatory backtest, spend the fresh holdout window, and record pass/fail. Does nothing else"
        " (no mining, no weekly snapshot). This is the ONLY way a Layer 2 confirmation is ever actually run --"
        " the scheduled/automatic path never does this itself (Korak 10 item 6).",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    symbols = tuple(s.strip().upper() for s in args.symbols.split(",") if s.strip()) or None
    scheduler = StrategyLabScheduler(
        db_path=args.storage,
        symbols=symbols,
        ranked_universe_cache_path=args.ranked_universe_cache or None,
        mining_interval_seconds=args.mining_interval,
        weekly_interval_seconds=args.weekly_interval,
    )
    if args.confirm_layer2:
        result = run_layer2_check(
            db_path=scheduler.db_path,
            shared_holdout_end_ms=SHARED_HOLDOUT_END_MS,
            symbols=scheduler.symbols,
            ranked_universe_cache_path=scheduler.ranked_universe_cache_path,
            telegram_config=scheduler.telegram_config,
            manual_trigger=True,
        )
        if result.ran_confirmation:
            print(f"Layer 2 confirmation ran: {result.confirmation}")
        elif result.candidate is None:
            print("Layer 2 confirmation queue is empty -- nothing to confirm.")
        else:
            print(
                f"Candidate {result.candidate} is not power-ready yet "
                f"(~{result.estimated_days_needed or 0.0:.1f} more days needed)."
            )
        return
    if args.once:
        scheduler.run_mining_once()
        scheduler.run_weekly_once()
        return
    scheduler.run_forever()


if __name__ == "__main__":
    main()
