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
from scanner.backtesting.strategy_lab_runner import run_layer2_check, run_weekly_snapshot
from scanner.backtesting.strategy_miner import ALL_TEMPLATE_VARIANTS, run_mining_cycle
from scanner.telegram_alerts import TelegramAlertConfig

logger = logging.getLogger(__name__)

# The frozen train/holdout boundary already locked in for the three closed
# research families (``scanner_data/research_registry.json``'s
# ``shared_holdout.end_timestamp_ms``) -- Layer 1 mining NEVER looks past
# this; Layer 2's rolling holdout boundary starts no earlier than it.
SHARED_HOLDOUT_END_MS = 1_790_253_820_000

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
        self._last_weekly_run_at_monotonic: float | None = None

    def run_mining_once(self) -> int:
        """Runs one Layer 1 mining cycle, persisting every evaluated
        candidate. Returns the number of candidates stored."""
        as_of_ts = min(int(datetime.now(UTC).timestamp() * 1000), SHARED_HOLDOUT_END_MS)
        candidates = run_mining_cycle(
            as_of_ts,
            symbols=self.symbols,
            ranked_universe_cache_path=self.ranked_universe_cache_path,
            template_variants=ALL_TEMPLATE_VARIANTS,
        )
        conn = storage.init_db(self.db_path)
        try:
            for candidate in candidates:
                storage.insert_mining_candidate(
                    conn,
                    as_of_timestamp_ms=as_of_ts,
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
                )
        finally:
            conn.close()
        logger.info("Strategy Lab mining cycle stored %d candidates", len(candidates))
        return len(candidates)

    def run_weekly_once(self) -> None:
        """Runs the weekly snapshot, then the Layer 2 power-gated check,
        in that order (the snapshot is always safe/descriptive; Layer 2
        only spends fresh data when power is sufficient)."""
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
        )
        if layer2_result.ran_confirmation:
            logger.info("Strategy Lab Layer 2 confirmation ran: %s", layer2_result.confirmation)
        else:
            logger.info(
                "Strategy Lab Layer 2: waiting (~%.1f fresh days available, ~%.1f more days needed)",
                layer2_result.fresh_days_available / 86_400_000,
                layer2_result.estimated_days_needed or 0.0,
            )

    def run_forever(self) -> None:
        logger.info(
            "Strategy Lab scheduler starting: mining every %ss, weekly snapshot/Layer-2 every %ss",
            self.mining_interval_seconds,
            self.weekly_interval_seconds,
        )
        while True:
            cycle_start = time.monotonic()
            try:
                self.run_mining_once()
            except Exception:
                logger.exception("Strategy Lab mining cycle failed; continuing on the next interval")

            if (
                self._last_weekly_run_at_monotonic is None
                or cycle_start - self._last_weekly_run_at_monotonic >= self.weekly_interval_seconds
            ):
                try:
                    self.run_weekly_once()
                except Exception:
                    logger.exception("Strategy Lab weekly snapshot/Layer-2 check failed; continuing")
                self._last_weekly_run_at_monotonic = cycle_start

            elapsed = time.monotonic() - cycle_start
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
    if args.once:
        scheduler.run_mining_once()
        scheduler.run_weekly_once()
        return
    scheduler.run_forever()


if __name__ == "__main__":
    main()
