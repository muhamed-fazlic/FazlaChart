"""Shared "what's actually worth trading right now" ranking logic.

Extracted out of the webapp so both the read-only dashboard endpoint
(``webapp.main.best_signal``) and the scanner's own live forward-outcome
logger (``scanner.signal_tracking``) rank candidates identically — the
tracker must log exactly what a user would have seen on the dashboard at
that moment, not a subtly different recomputation that could silently
diverge from it over time.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

BEST_SIGNAL_HORIZON_CANDLES = 12
BEST_SIGNAL_MIN_SAMPLE_SIZE = 30  # below this, even a high win rate is too noisy to call a real edge


def _active_labels(trend: str, anomalies: list[str]) -> list[str]:
    """Same label-naming scheme ``agents_crypto.derivatives_risk_agent``
    uses, duplicated here (it's two lines) instead of imported, so this free
    code path never has to pull in the LLM client stack just to compute
    which calibration labels are active right now."""
    return [f"trend:{trend}"] + [f"anomaly:{a}" for a in anomalies]


def _age_seconds(iso_timestamp: str) -> float:
    ts = datetime.fromisoformat(iso_timestamp)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return (datetime.now(UTC) - ts).total_seconds()


def compute_candidates(rows: list[dict[str, Any]], limit: int = 5) -> dict[str, Any]:
    """Rank every currently-tracked directional symbol by the conservative
    (95%-CI lower-bound) win rate of its active calibration label(s), so a
    lucky small sample can never outrank a well-tested edge, and require a
    minimum sample size before a label counts as "significant" at all.

    ``rows`` is the output of :func:`scanner.storage.fetch_latest_scan_per_symbol`
    (or an equivalent list of per-symbol scan dicts). Read-only and always
    fast: it only ever reads whatever's already in the on-disk calibration
    cache and never triggers a fresh backtest inline — a symbol simply has
    no entry here until its cache is warm.
    """
    if not rows:
        return {"signals": [], "generated_at": None, "candidates_considered": 0, "cache_coverage": None}

    from agents_crypto.calibration_lookup import find_stats_for_labels, get_cached_calibration_stats
    from agents_crypto.risk_calc import compute_risk_levels

    candidates: list[dict[str, Any]] = []
    considered = 0
    for row in rows:
        if row["trend"] not in ("bullish", "bearish"):
            continue
        considered += 1
        ref_tf = row["reference_timeframe"]
        ref = row["snapshot"].get("timeframes", {}).get(ref_tf, {})
        labels = _active_labels(row["trend"], row["anomalies"])

        stats = get_cached_calibration_stats(row["symbol"], ref_tf)
        matches = find_stats_for_labels(stats, labels, BEST_SIGNAL_HORIZON_CANDLES)
        significant = [
            m
            for m in matches
            if m.direction in ("long", "short")
            and m.n >= BEST_SIGNAL_MIN_SAMPLE_SIZE
            and m.win_rate_ci_low_pct is not None
            and m.win_rate_ci_low_pct > 50.0
        ]
        if not significant:
            continue
        # Rank by the CI's conservative lower bound, not the point estimate —
        # rewards both a high win rate AND enough samples to be confident in it.
        best = max(significant, key=lambda m: m.win_rate_ci_low_pct)

        close, atr = row["close_price"], ref.get("atr")
        entry = None
        if close and atr:
            try:
                levels = compute_risk_levels(
                    close, atr, best.direction, rsi=ref.get("rsi"), ema20=ref.get("ema20")
                )
            except ValueError:
                levels = None
            if levels:
                # A narrow confirmation band around the current price, not a
                # single exact tick — ATR-sized, biased slightly toward a
                # pullback/retest entry in the trade direction rather than
                # chasing the current price outright. If the move already
                # looks extended (see risk_calc._extension_check), recenter
                # the zone on the suggested patient/retracement entry instead
                # of the current (already-stretched) price.
                band = atr * 0.25
                zone_center = levels.patient_entry if levels.patient_entry is not None else close
                zone = (
                    (zone_center - band, zone_center + band * 0.4)
                    if best.direction == "long"
                    else (zone_center - band * 0.4, zone_center + band)
                )
                entry = {
                    "price": round(close, 8),
                    "entry_zone_low": round(min(zone), 8),
                    "entry_zone_high": round(max(zone), 8),
                    "stop_loss": levels.stop_loss,
                    "take_profit": levels.take_profit,
                    "take_profit_conservative": levels.take_profit_conservative,
                    "risk_reward_ratio": levels.risk_reward_ratio,
                    "volatility_warning": levels.volatility_warning,
                    "extension_warning": levels.extension_warning,
                }

        candidates.append(
            {
                "symbol": row["symbol"],
                "timeframe": ref_tf,
                "trend": row["trend"],
                "label": best.label,
                "direction": best.direction,
                "n": best.n,
                "win_rate_pct": best.win_rate_pct,
                "win_rate_ci_low_pct": best.win_rate_ci_low_pct,
                "win_rate_ci_high_pct": best.win_rate_ci_high_pct,
                "avg_net_return_pct": best.avg_net_return_pct,
                "regime": best.regime,
                "entry": entry,
                "scanned_at": row["scanned_at"],
                "age_seconds": round(_age_seconds(row["scanned_at"]), 1),
            }
        )

    # Primary sort: the CI's conservative lower bound (never the raw point
    # estimate) — see the module docstring. Ties there (rounded to 1 decimal,
    # so distinct symbols land on the same number often) are broken by the
    # raw win rate, then by sample size, instead of falling back to whatever
    # order rows happened to come out of the database in.
    candidates.sort(
        key=lambda c: (c["win_rate_ci_low_pct"], c["win_rate_pct"], c["n"]),
        reverse=True,
    )
    generated_at = max(r["scanned_at"] for r in rows)
    return {
        "signals": candidates[: max(1, limit)],
        "generated_at": generated_at,
        "candidates_considered": considered,
        "cache_coverage": f"{len(candidates)}/{considered}",
    }
