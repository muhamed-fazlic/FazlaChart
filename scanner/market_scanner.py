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
from scanner.signal_tracking import log_new_signals, resolve_open_signals
from scanner.storage import fetch_known_symbols, fetch_latest_scan_per_symbol, init_db, save_scan_result
from tradingagents.dataflows.errors import VendorError
from tradingagents.dataflows.vendors.binance import client
from tradingagents.dataflows.vendors.binance.snapshot import build_symbol_snapshot

logger = logging.getLogger(__name__)


class MarketScanner:
    def __init__(self, config: ScannerConfig | None = None):
        self.config = config or ScannerConfig()
        self.conn = init_db(self.config.storage_path)

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
        return {"btc": btc_context, "symbols": results}

    def _track_signal_outcomes(self) -> None:
        """Resolve any open tracked signals against this cycle's fresh
        prices, then log any newly-qualifying Top Signal Now candidates —
        in that order, so a signal that both resolves and re-qualifies in
        the same cycle starts a genuinely new tracked trade rather than
        being treated as still open. Never lets a tracking bug take down
        the scan cycle itself."""
        try:
            rows = fetch_latest_scan_per_symbol(self.conn)
            resolved = resolve_open_signals(self.conn, rows)
            logged = log_new_signals(self.conn, rows)
            if resolved or logged:
                logger.info("Signal tracking: resolved %d, logged %d new", resolved, logged)
        except Exception:
            logger.exception("Signal outcome tracking failed this cycle; continuing")

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
