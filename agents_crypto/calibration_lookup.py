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

from scanner.calibration import CalibrationStats, OutOfSampleResult, backtest_symbol, summarize, validate_out_of_sample

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


_REGIME_KEYS = ("all", "low", "normal", "high")


def _run_backtest(symbol: str, timeframe: str, horizons: tuple[int, ...]) -> dict[str, list[CalibrationStats]]:
    """Module-level (picklable) entry point executed in the worker process.

    Runs the (expensive) backtest once and summarizes it four ways — overall
    plus each volatility regime — so a caller asking for a specific regime
    later doesn't pay for another full backtest, just a cheap re-aggregation
    of cases already computed here.
    """
    cases = backtest_symbol(symbol, timeframe, horizons)
    return {
        "all": summarize(cases),
        "low": summarize(cases, regime_filter="low"),
        "normal": summarize(cases, regime_filter="normal"),
        "high": summarize(cases, regime_filter="high"),
    }


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
    regime: str | None = None,
) -> list[CalibrationStats]:
    """Calibration stats for ``symbol``/``timeframe``, from cache if fresh.

    Recomputes (and re-caches) the full backtest when the cached entry is
    missing, stale, or ``force_refresh`` is set.

    ``use_process=True`` runs a cold-cache backtest in a separate worker
    process (see module docstring) instead of inline on the calling thread.
    Real dashboard endpoints pass this; it defaults to ``False`` so unit
    tests can keep mocking ``backtest_symbol``/``summarize`` directly without
    the process-boundary pickling that would otherwise hide the mock.

    ``regime``, when given ("low"/"normal"/"high"), returns stats scoped to
    that volatility regime only instead of the all-history blend — pass the
    symbol's *current* regime (see :func:`scanner.calibration.current_volatility_regime`)
    to answer "how has this label done in conditions like right now".
    """
    regime_key = regime if regime in ("low", "normal", "high") else "all"
    cache = _load_cache(cache_path)
    key = _cache_key(symbol, timeframe)
    entry = cache.get(key)

    if not force_refresh and entry and (time.time() - entry["computed_at"]) < ttl_seconds:
        regime_rows = entry["stats"].get(regime_key, [])
        return [CalibrationStats(**row) for row in regime_rows]

    if use_process:
        regime_stats = _run_backtest_deduped(symbol, timeframe, horizons)
    else:
        regime_stats = _run_backtest(symbol, timeframe, horizons)

    cache[key] = {
        "computed_at": time.time(),
        "stats": {rk: [asdict(s) for s in rows] for rk, rows in regime_stats.items()},
    }
    _save_cache(cache_path, cache)
    return regime_stats.get(regime_key, [])


def _run_backtest_deduped(symbol: str, timeframe: str, horizons: tuple[int, ...]) -> dict[str, list[CalibrationStats]]:
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


def _run_oos_validation(symbol: str, timeframe: str, horizons: tuple[int, ...]) -> list[OutOfSampleResult]:
    """Module-level (picklable) entry point executed in the worker process —
    see :func:`scanner.calibration.validate_out_of_sample`."""
    cases = backtest_symbol(symbol, timeframe, horizons)
    return validate_out_of_sample(cases)


# Separate in-flight de-dup map from the main stats backtest's: the two are
# independent computations over the same underlying cases (recomputed
# separately, not shared, to keep this additive and not risk disturbing the
# well-tested stats caching path above) and could otherwise collide on the
# same cache key if both happened to run concurrently for the same symbol.
_inflight_oos: dict[str, Future] = {}
_inflight_oos_lock = Lock()


def _run_oos_deduped(symbol: str, timeframe: str, horizons: tuple[int, ...]) -> list[OutOfSampleResult]:
    key = _cache_key(symbol, timeframe)
    with _inflight_oos_lock:
        future = _inflight_oos.get(key)
        if future is None:
            future = _get_executor().submit(_run_oos_validation, symbol, timeframe, horizons)
            _inflight_oos[key] = future
    try:
        return future.result()
    finally:
        with _inflight_oos_lock:
            if _inflight_oos.get(key) is future:
                del _inflight_oos[key]


def get_oos_validation(
    symbol: str,
    timeframe: str = "1h",
    horizons: tuple[int, ...] = (4, 12, 24),
    cache_path: str = DEFAULT_CACHE_PATH,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
    force_refresh: bool = False,
    use_process: bool = False,
) -> list[OutOfSampleResult]:
    """Out-of-sample validation for ``symbol``/``timeframe``, cached
    alongside (but independently refreshed from) the main calibration stats
    — see :func:`scanner.calibration.validate_out_of_sample`. Answers "does
    this label's edge survive on data more recent than what it was measured
    on", not just "what was the historical win rate".
    """
    cache = _load_cache(cache_path)
    key = _cache_key(symbol, timeframe)
    entry = cache.get(key) or {}

    if not force_refresh and "oos" in entry and (time.time() - entry.get("oos_computed_at", 0)) < ttl_seconds:
        return [OutOfSampleResult(**row) for row in entry["oos"]]

    if use_process:
        results = _run_oos_deduped(symbol, timeframe, horizons)
    else:
        results = _run_oos_validation(symbol, timeframe, horizons)

    entry["oos"] = [asdict(r) for r in results]
    entry["oos_computed_at"] = time.time()
    cache[key] = entry
    _save_cache(cache_path, cache)
    return results


def find_stats_for_labels(
    stats: list[CalibrationStats],
    labels: list[str],
    horizon_candles: int,
) -> list[CalibrationStats]:
    """The subset of ``stats`` matching any of ``labels`` at ``horizon_candles``
    — i.e. "what does history say about the label(s) active right now?"."""
    return [s for s in stats if s.label in labels and s.horizon_candles == horizon_candles]


def find_oos_for_labels(
    results: list[OutOfSampleResult],
    labels: list[str],
    horizon_candles: int,
) -> list[OutOfSampleResult]:
    """The subset of out-of-sample validation ``results`` matching any of
    ``labels`` at ``horizon_candles`` — mirrors :func:`find_stats_for_labels`."""
    return [r for r in results if r.label in labels and r.horizon_candles == horizon_candles]


def find_regime_matched_stats(
    symbol: str,
    timeframe: str,
    labels: list[str],
    horizon_candles: int,
    current_regime: str | None,
    use_process: bool = True,
) -> tuple[list[CalibrationStats], bool]:
    """Calibration matches scoped to ``current_regime`` ("low"/"normal"/
    "high"/None, computed by the caller — see
    :func:`scanner.calibration.current_volatility_regime`), falling back to
    the all-regime blend whenever there isn't enough regime-specific history
    yet for the active label(s), or the regime itself is unknown.

    Returns ``(matches, used_fallback)`` so callers can tell the user whether
    the result is regime-matched evidence or a blended, less-specific one.
    """
    if current_regime is not None:
        regime_stats = get_calibration_stats(
            symbol, timeframe, horizons=(horizon_candles,), use_process=use_process, regime=current_regime
        )
        regime_matches = find_stats_for_labels(regime_stats, labels, horizon_candles)
        if regime_matches:
            return regime_matches, False

    all_stats = get_calibration_stats(symbol, timeframe, horizons=(horizon_candles,), use_process=use_process)
    return find_stats_for_labels(all_stats, labels, horizon_candles), True


def get_cached_calibration_stats(
    symbol: str,
    timeframe: str = "1h",
    regime: str | None = None,
    cache_path: str = DEFAULT_CACHE_PATH,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
) -> list[CalibrationStats]:
    """Like :func:`get_calibration_stats`, but NEVER triggers a fresh backtest
    on a cache miss/stale entry — it only ever reads what's already on disk,
    returning ``[]`` immediately otherwise.

    Used by endpoints that need to cheaply rank many symbols at once (e.g.
    "best signal right now" across the whole tracked universe): spawning a
    cold-cache backtest per symbol there would make the request take
    minutes. Those symbols just get skipped from the ranking until their
    cache is populated by the normal on-demand path (a user viewing that
    row, or the background warm-up loop)."""
    regime_key = regime if regime in ("low", "normal", "high") else "all"
    cache = _load_cache(cache_path)
    entry = cache.get(_cache_key(symbol, timeframe))
    if not entry or (time.time() - entry["computed_at"]) >= ttl_seconds:
        return []
    regime_rows = entry["stats"].get(regime_key, [])
    return [CalibrationStats(**row) for row in regime_rows]
