"""Cached access to the calibration backtest, keyed by (symbol, timeframe).

The full backtest (``scanner.calibration.backtest_symbol``) fetches up to
1500 historical candles and walks every one of them — a few seconds of
network + computation, almost all of it single-threaded pandas work. An
interpreting agent calling this on demand doesn't need millisecond
freshness, so results are cached to disk with a TTL and only recomputed
when stale or missing.

On a cold cache, the dashboard process runs this backtest on-demand (the
webapp's win-rate column and ``POST /api/analyze`` both call into
``get_calibration_stats``). Because it's CPU-bound pandas code, running it
on a thread inside the main server process still holds the GIL for its
whole duration, starving every other request on that same process —
including an in-flight ``POST /api/analyze`` whose *own* calibration lookup
would otherwise be an instant cache hit. ``use_process=True`` (the default
for real endpoints) runs the backtest in a short-lived worker *process*
instead of a thread, so it genuinely runs in parallel with the main
process rather than just appearing to.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from threading import Lock
from typing import Any

from scanner.calibration import CalibrationStats, backtest_symbol, summarize

DEFAULT_CACHE_PATH = "scanner_data/calibration_cache.json"
DEFAULT_TTL_SECONDS = 6 * 3600  # 6 hours: labels/backtests don't meaningfully drift faster than this

# Capped small: this is CPU-bound backtest work, not I/O, so more workers than
# cores just thrashes. 2 lets at most two cold-cache symbols backtest truly in
# parallel without competing with the scanner/dashboard for every core.
_MAX_BACKTEST_WORKERS = 2
_executor: ProcessPoolExecutor | None = None
_executor_lock = Lock()

# In-flight de-duplication: if two requests ask for the same (symbol,
# timeframe) while it's already backtesting, the second one piggybacks on the
# first's future instead of starting a duplicate backtest in another worker
# process (observed happening in practice when the win-rate column's own
# background refresh overlaps itself).
_inflight: dict[str, Future] = {}
_inflight_lock = Lock()


def _get_executor() -> ProcessPoolExecutor:
    global _executor
    with _executor_lock:
        if _executor is None:
            _executor = ProcessPoolExecutor(max_workers=_MAX_BACKTEST_WORKERS)
        return _executor


def _run_backtest(symbol: str, timeframe: str, horizons: tuple[int, ...]) -> list[CalibrationStats]:
    """Module-level (picklable) entry point executed in the worker process."""
    cases = backtest_symbol(symbol, timeframe, horizons)
    return summarize(cases)


def _cache_key(symbol: str, timeframe: str) -> str:
    return f"{symbol.upper()}:{timeframe}"


def _load_cache(cache_path: str) -> dict[str, Any]:
    path = Path(cache_path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save_cache(cache_path: str, cache: dict[str, Any]) -> None:
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2))


def get_calibration_stats(
    symbol: str,
    timeframe: str = "1h",
    horizons: tuple[int, ...] = (4, 12, 24),
    cache_path: str = DEFAULT_CACHE_PATH,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
    force_refresh: bool = False,
    use_process: bool = False,
) -> list[CalibrationStats]:
    """Calibration stats for ``symbol``/``timeframe``, from cache if fresh.

    Recomputes (and re-caches) the full backtest when the cached entry is
    missing, stale, or ``force_refresh`` is set.

    ``use_process=True`` runs a cold-cache backtest in a separate worker
    process (see module docstring) instead of inline on the calling thread.
    Real dashboard endpoints pass this; it defaults to ``False`` so unit
    tests can keep mocking ``backtest_symbol``/``summarize`` directly without
    the process-boundary pickling that would otherwise hide the mock.
    """
    cache = _load_cache(cache_path)
    key = _cache_key(symbol, timeframe)
    entry = cache.get(key)

    if not force_refresh and entry and (time.time() - entry["computed_at"]) < ttl_seconds:
        return [CalibrationStats(**row) for row in entry["stats"]]

    if use_process:
        stats = _run_backtest_deduped(symbol, timeframe, horizons)
    else:
        cases = backtest_symbol(symbol, timeframe, horizons)
        stats = summarize(cases)

    cache[key] = {"computed_at": time.time(), "stats": [asdict(s) for s in stats]}
    _save_cache(cache_path, cache)
    return stats


def _run_backtest_deduped(symbol: str, timeframe: str, horizons: tuple[int, ...]) -> list[CalibrationStats]:
    """Runs ``_run_backtest`` in the shared worker-process pool, collapsing
    concurrent callers for the same (symbol, timeframe) onto one future
    instead of each starting its own duplicate backtest."""
    key = _cache_key(symbol, timeframe)
    with _inflight_lock:
        future = _inflight.get(key)
        if future is None:
            future = _get_executor().submit(_run_backtest, symbol, timeframe, horizons)
            _inflight[key] = future
    try:
        return future.result()
    finally:
        with _inflight_lock:
            if _inflight.get(key) is future:
                del _inflight[key]


def find_stats_for_labels(
    stats: list[CalibrationStats],
    labels: list[str],
    horizon_candles: int,
) -> list[CalibrationStats]:
    """The subset of ``stats`` matching any of ``labels`` at ``horizon_candles``
    — i.e. "what does history say about the label(s) active right now?"."""
    return [s for s in stats if s.label in labels and s.horizon_candles == horizon_candles]
