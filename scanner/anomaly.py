"""Cross-symbol anomaly checks that need more than one snapshot to evaluate —
currently just the BTC-divergence check. Single-symbol anomalies (volume not
confirming price, failed breakouts, OI/funding divergence, volatility
expansion) live in
:mod:`tradingagents.dataflows.vendors.binance.anomaly` and are already
attached to each timeframe snapshot by the data layer.
"""

from __future__ import annotations

from typing import Any

from scanner.btc_context import diverges_from_btc


def altcoin_vs_btc_divergence(
    symbol_snapshot: dict[str, Any], btc_context: dict[str, Any], min_steps: int = 2, reference_timeframe: str = "1h"
) -> list[str]:
    """Flag when an altcoin's trend strongly disagrees with BTC's trend on
    the same timeframe — a move that is not confirmed by the broader market."""
    tf_snapshot = symbol_snapshot["timeframes"].get(reference_timeframe)
    if not tf_snapshot:
        return []
    btc_trend = btc_context["reference_trend"]
    if diverges_from_btc(tf_snapshot["trend"], btc_trend, min_steps):
        return [f"altcoin_diverges_from_btc_on_{reference_timeframe}"]
    return []


def collect_all_anomalies(symbol_snapshot: dict[str, Any], btc_context: dict[str, Any], config) -> list[str]:
    """Merge the per-timeframe anomalies already on the snapshot with the
    cross-symbol (BTC-relative) ones into one flat, de-duplicated list."""
    flags: list[str] = []
    for tf, tf_snapshot in symbol_snapshot["timeframes"].items():
        for anomaly in tf_snapshot.get("anomalies", []):
            flags.append(f"{tf}:{anomaly}")
    flags.extend(altcoin_vs_btc_divergence(symbol_snapshot, btc_context, config.btc_divergence_min_steps))
    return flags
