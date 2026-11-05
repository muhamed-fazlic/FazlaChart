"""Frozen-snapshot data loading for reproducible backtests.

Why this exists: ``scanner.calibration.load_extended_history`` always pages
backwards from "now" (``end_time=None``) when no explicit anchor is given.
Since the live scanner runs 24/7 and keeps adding fresh candles, two runs of
the *exact same* backtest config minutes or hours apart silently see two
different datasets -- this was empirically measured to cause ~9 percentage
points of pure noise in the "original" strategy variant's net profit
across identical-config runs (see PHASE8_PROFITABILITY_RESEARCH_TODO.md).
That makes it impossible to tell whether a change in the result came from a
code/parameter change or just from the data having moved underneath it.

:func:`cached_load_extended_history` fixes this for any caller that pins an
explicit ``end_timestamp_ms``: the exact same (symbol, timeframe,
total_candles, end_timestamp_ms) request always returns the exact same
candles, served from an on-disk cache after the first fetch (which also
makes repeated runs against a locked snapshot much faster, since nothing
needs to hit the network again). Leaving ``end_timestamp_ms=None`` preserves
the old "always fetch the freshest available data" behaviour used by live
monitoring/dashboard use cases that are not trying to be reproducible.

:func:`dataset_fingerprint` computes a short hash identifying exactly which
candles (open_time/close values) were actually used, so two experiment-
registry trials can be compared to tell whether they really ran over the
same data or not -- without needing to diff raw kline dumps by hand.
"""

from __future__ import annotations

import hashlib
import pickle
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from scanner.calibration import load_extended_history

DEFAULT_KLINE_CACHE_DIR = "scanner_data/kline_cache"


def parse_end_timestamp(value: str) -> int:
    """Parse a CLI-supplied end-timestamp string into epoch milliseconds.

    Accepts a bare ISO 8601 datetime (``2026-11-04T00:00:00`` or with a
    trailing ``Z``/offset) or a raw integer epoch-milliseconds string.
    Naive datetimes are assumed UTC, matching Binance's own timestamps.
    """
    value = value.strip()
    if value.isdigit():
        return int(value)
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp() * 1000)


def _cache_path(cache_dir: str, symbol: str, timeframe: str, total_candles: int, end_timestamp_ms: int) -> Path:
    key = f"{symbol}_{timeframe}_{total_candles}_{end_timestamp_ms}"
    return Path(cache_dir) / f"{key}.pkl"


def cached_load_extended_history(
    symbol: str,
    timeframe: str,
    total_candles: int,
    end_timestamp_ms: int | None,
    cache_dir: str = DEFAULT_KLINE_CACHE_DIR,
) -> pd.DataFrame:
    """Like :func:`scanner.calibration.load_extended_history`, but anchored
    to a frozen ``end_timestamp_ms`` and cached to disk so repeated runs
    against the same locked snapshot are both deterministic and fast.

    ``end_timestamp_ms=None`` bypasses the cache entirely and always fetches
    the freshest available data (the old, non-reproducible behaviour) --
    this is the right choice for live/ad-hoc dashboard refreshes that are
    not trying to be a locked, comparable trial.
    """
    if end_timestamp_ms is None:
        return load_extended_history(symbol, timeframe, total_candles)

    path = _cache_path(cache_dir, symbol, timeframe, total_candles, end_timestamp_ms)
    if path.exists():
        with open(path, "rb") as fh:
            return pickle.load(fh)

    history = load_extended_history(symbol, timeframe, total_candles, end_time=end_timestamp_ms)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(history, fh)
    return history


def dataset_fingerprint(history: pd.DataFrame) -> str:
    """A short, stable hash of the candles actually used, so two runs can
    be compared to confirm (or rule out) that they saw the same data."""
    if history.empty:
        return hashlib.sha256(b"empty").hexdigest()[:16]
    payload = pd.util.hash_pandas_object(history[["open_time", "open", "high", "low", "close", "volume"]]).values
    return hashlib.sha256(payload.tobytes()).hexdigest()[:16]


def combined_dataset_fingerprint(fingerprints: list[str]) -> str:
    """Combine several per-(symbol, timeframe) fingerprints (in a stable,
    caller-chosen order) into one overall hash for the whole run."""
    joined = "|".join(sorted(fingerprints))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]
