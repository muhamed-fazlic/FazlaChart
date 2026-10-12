"""Cached access to the calibration backtest, keyed by (symbol, timeframe).

The full backtest (``scanner.calibration.backtest_symbol``) fetches up to
1500 historical candles and walks every one of them — a few seconds of
network + computation. An interpreting agent calling this on demand doesn't
need millisecond freshness, so results are cached to disk with a TTL and
only recomputed when stale or missing.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from scanner.calibration import CalibrationStats, backtest_symbol, summarize

DEFAULT_CACHE_PATH = "scanner_data/calibration_cache.json"
DEFAULT_TTL_SECONDS = 6 * 3600  # 6 hours: labels/backtests don't meaningfully drift faster than this


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
) -> list[CalibrationStats]:
    """Calibration stats for ``symbol``/``timeframe``, from cache if fresh.

    Recomputes (and re-caches) the full backtest when the cached entry is
    missing, stale, or ``force_refresh`` is set.
    """
    cache = _load_cache(cache_path)
    key = _cache_key(symbol, timeframe)
    entry = cache.get(key)

    if not force_refresh and entry and (time.time() - entry["computed_at"]) < ttl_seconds:
        return [CalibrationStats(**row) for row in entry["stats"]]

    cases = backtest_symbol(symbol, timeframe, horizons)
    stats = summarize(cases)

    cache[key] = {"computed_at": time.time(), "stats": [asdict(s) for s in stats]}
    _save_cache(cache_path, cache)
    return stats


def find_stats_for_labels(
    stats: list[CalibrationStats],
    labels: list[str],
    horizon_candles: int,
) -> list[CalibrationStats]:
    """The subset of ``stats`` matching any of ``labels`` at ``horizon_candles``
    — i.e. "what does history say about the label(s) active right now?"."""
    return [s for s in stats if s.label in labels and s.horizon_candles == horizon_candles]
