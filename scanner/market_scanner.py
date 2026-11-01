"""Market Scanner orchestration: one cycle = refresh BTC context, scan every
symbol in the configured universe, flag anomalies, and persist the result.

Run continuously with ``python -m scanner`` (see :mod:`scanner.__main__`).
"""

from __future__ import annotations

import logging
import time
from typing import Any

from scanner.anomaly import collect_all_anomalies
from scanner.btc_context import build_btc_context
from scanner.config import ScannerConfig
from scanner.drift_detector import active_muted_bucket_set, evaluate_and_apply_drift_mutes
from scanner.signal_tracking import log_new_signals, resolve_open_signals
from scanner.storage import fetch_known_symbols, fetch_latest_scan_per_symbol, init_db, prune_old_scans, save_scan_result
from scanner.telegram_alerts import TelegramAlertConfig, maybe_send_alert, maybe_send_drift_alert
from tradingagents.dataflows.errors import VendorError
from tradingagents.dataflows.vendors.binance import client
from tradingagents.dataflows.vendors.binance.snapshot import build_symbol_snapshot

logger = logging.getLogger(__name__)


class MarketScanner:
    def __init__(self, config: ScannerConfig | None = None):
        self.config = config or ScannerConfig()
        self.conn = init_db(self.config.storage_path)
        self._telegram = TelegramAlertConfig(
            bot_token=self.config.telegram_bot_token,
            chat_id=self.config.telegram_chat_id,
            min_win_rate_ci_low_pct=self.config.telegram_min_win_rate_ci_low_pct,
            min_n=self.config.telegram_min_n,
            require_btc_alignment=self.config.telegram_require_btc_alignment,
        )
        self._last_prune_at: float = 0.0

    def resolve_universe(self) -> list[str]:
        """The symbols to scan this cycle: explicit list, else top-N by
        volume unioned with every symbol already known from past cycles (so
        a symbol that drops out of the current top-N keeps getting
        refreshed instead of its dashboard row going stale forever), capped
        at `max_universe_size` with the freshest top-N ranking taking
        priority over older known symbols when trimming to fit."""
        if self.config.symbols:
            universe = list(self.config.symbols)
        else:
            current_top_n = client.get_top_symbols_by_quote_volume(top_n=self.config.top_n_by_volume)
            if self.config.keep_known_symbols:
                known = fetch_known_symbols(self.conn)
                seen = set(current_top_n)
                universe = list(current_top_n) + [s for s in known if s not in seen]
                universe = universe[: self.config.max_universe_size]
            else:
                universe = current_top_n
        return [s for s in universe if s not in self.config.symbols_blocklist]

    def scan_symbol(self, symbol: str, btc_context: dict[str, Any]) -> dict[str, Any] | None:
        """Build one symbol's snapshot, flag anomalies, persist, and return it.
        Returns None (and logs) if the vendor could not answer for this symbol,
        so one bad symbol never stops the cycle."""
        try:
            snapshot = build_symbol_snapshot(symbol, timeframes=self.config.timeframes)
        except VendorError as exc:
            logger.warning("Skipping %s this cycle: %s", symbol, exc)
            return None

        anomalies = collect_all_anomalies(snapshot, btc_context, self.config)
        snapshot["anomalies"] = anomalies
        save_scan_result(self.conn, symbol, snapshot, anomalies)
        if anomalies:
            logger.info("%s anomalies: %s", symbol, anomalies)
        return snapshot

    def run_once(self) -> dict[str, Any]:
        """One full scan cycle. Returns {"btc": ..., "symbols": {sym: snapshot}}."""
        btc_context = build_btc_context(self.config.btc_timeframes)
        universe = self.resolve_universe()
        logger.info("Scanning %d symbols (BTC reference trend: %s)", len(universe), btc_context["reference_trend"])

        results: dict[str, Any] = {}
        for symbol in universe:
            snapshot = self.scan_symbol(symbol, btc_context)
            if snapshot is not None:
                results[symbol] = snapshot

        self._track_signal_outcomes()
        self._maybe_prune_old_scans()
        return {"btc": btc_context, "symbols": results}

    def _track_signal_outcomes(self) -> None:
        """Resolve any open tracked signals against this cycle's fresh
        prices, re-evaluate the calibration-drift mute state, then log any
        newly-qualifying Top Signal Now candidates — in that order, so a
        signal that both resolves and re-qualifies in the same cycle starts
        a genuinely new tracked trade rather than being treated as still
        open, and a bucket that just got muted this cycle is respected
        immediately rather than one cycle late. Never lets a tracking bug
        take down the scan cycle itself."""
        try:
            rows = fetch_latest_scan_per_symbol(self.conn)
            resolved = resolve_open_signals(self.conn, rows)
            mute_changes = evaluate_and_apply_drift_mutes(self.conn, on_mute=self._on_bucket_muted, on_unmute=self._on_bucket_unmuted)
            muted_buckets = active_muted_bucket_set(self.conn)
            logged = log_new_signals(self.conn, rows, on_new_signal=self._on_new_signal, muted_buckets=muted_buckets)
            if resolved or logged or mute_changes["muted"] or mute_changes["unmuted"]:
                logger.info(
                    "Signal tracking: resolved %d, logged %d new, muted %s, unmuted %s",
                    resolved,
                    logged,
                    mute_changes["muted"],
                    mute_changes["unmuted"],
                )
        except Exception:
            logger.exception("Signal outcome tracking failed this cycle; continuing")

    def _on_new_signal(self, candidate: dict[str, Any]) -> None:
        """Pushes a Telegram alert for a newly-qualifying candidate when
        alerting is enabled and the candidate clears the configured quality
        threshold. A no-op (and never raises) when Telegram isn't configured."""
        if maybe_send_alert(candidate, self._telegram):
            logger.info("Telegram alert sent for %s %s %s", candidate["symbol"], candidate["timeframe"], candidate["direction"])

    def _on_bucket_muted(self, stats: Any) -> None:
        """Pushes a Telegram notice when the live calibration-drift detector
        mutes a bucket (see :mod:`scanner.drift_detector`)."""
        maybe_send_drift_alert(stats, self._telegram, muted=True)
        logger.warning(
            "Calibration drift: muted %s/%s (realized %.1f%% vs promised %.1f%%, n=%d)",
            stats.timeframe, stats.direction, stats.realized_win_rate_pct, stats.promised_win_rate_ci_low_pct_avg, stats.n_resolved,
        )

    def _on_bucket_unmuted(self, stats: Any) -> None:
        """Pushes a Telegram notice when a previously-muted bucket's live
        results have recovered enough to be restored."""
        maybe_send_drift_alert(stats, self._telegram, muted=False)
        logger.info(
            "Calibration drift: restored %s/%s (realized %.1f%% vs promised %.1f%%, n=%d)",
            stats.timeframe, stats.direction, stats.realized_win_rate_pct, stats.promised_win_rate_ci_low_pct_avg, stats.n_resolved,
        )

    def _maybe_prune_old_scans(self) -> None:
        """Delete `scans` rows older than `scan_retention_days`, at most
        once every `retention_check_interval_seconds` -- keeps the database
        from growing without bound over months of 24/7 operation. Never lets
        a pruning bug take down the scan cycle itself."""
        now = time.monotonic()
        if now - self._last_prune_at < self.config.retention_check_interval_seconds:
            return
        self._last_prune_at = now
        try:
            deleted = prune_old_scans(self.conn, retain_days=self.config.scan_retention_days)
            if deleted:
                logger.info("Pruned %d scan rows older than %d days", deleted, self.config.scan_retention_days)
        except Exception:
            logger.exception("Scan retention pruning failed; continuing")

    def run_forever(self) -> None:
        """Loop ``run_once`` every ``poll_interval_seconds``, logging and
        continuing past any unexpected per-cycle error so the 24/7 service
        stays up even if one cycle fails entirely."""
        logger.info(
            "Market Scanner starting: poll every %ss, %s",
            self.config.poll_interval_seconds,
            f"{len(self.config.symbols)} fixed symbols" if self.config.symbols else f"top {self.config.top_n_by_volume} by volume",
        )
        while True:
            cycle_start = time.monotonic()
            try:
                result = self.run_once()
                logger.info("Cycle complete: %d symbols scanned", len(result["symbols"]))
            except Exception:
                logger.exception("Scan cycle failed; continuing on the next interval")
            elapsed = time.monotonic() - cycle_start
            sleep_for = max(0.0, self.config.poll_interval_seconds - elapsed)
            time.sleep(sleep_for)
