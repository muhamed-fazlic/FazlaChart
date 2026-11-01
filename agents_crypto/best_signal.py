"""Shared "what's actually worth trading right now" ranking logic.

Extracted out of the webapp so both the read-only dashboard endpoint
(``webapp.main.best_signal``) and the scanner's own live forward-outcome
logger (``scanner.signal_tracking``) rank candidates identically — the
tracker must log exactly what a user would have seen on the dashboard at
that moment, not a subtly different recomputation that could silently
diverge from it over time.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

from scanner.btc_context import BTC_SYMBOL, diverges_from_btc

BEST_SIGNAL_HORIZON_CANDLES = 12
BEST_SIGNAL_MIN_SAMPLE_SIZE = 30  # below this, even a high win rate is too noisy to call a real edge


def _tag_correlated_exposure(signals: list[dict[str, Any]], btc_trend: str | None) -> None:
    """Mutates ``signals`` in place, attaching a ``correlation`` sub-dict to
    each candidate when 2+ of the surfaced signals likely represent the same
    underlying bet rather than independent opportunities.

    This doesn't need a live price-correlation computation (which would mean
    fetching fresh klines per pair, just to rank a shortlist): in crypto
    futures the dominant source of cross-symbol correlation is simply "the
    whole market moves with BTC". Two altcoin signals that share a direction
    AND aren't diverging from BTC's own trend are, in practice, almost always
    just the same macro bet on BTC's direction wearing two different tickers
    — if BTC reverses, both move together. Signals that *diverge* from BTC
    are flagged as more idiosyncratic even when they share a direction, since
    that divergence is itself evidence of an independent, symbol-specific
    driver.
    """
    groups: dict[tuple[str, bool], list[dict[str, Any]]] = defaultdict(list)
    for sig in signals:
        diverges = diverges_from_btc(sig["trend"], btc_trend) if btc_trend else False
        sig["diverges_from_btc"] = diverges
        groups[(sig["direction"], diverges)].append(sig)

    for (direction, diverges), members in groups.items():
        if len(members) < 2:
            for sig in members:
                sig["correlation"] = None
            continue
        symbols = [m["symbol"] for m in members]
        if diverges:
            note = (
                f"{len(members)} signala ({', '.join(symbols)}) dijele {direction.upper()} smjer i podjednako "
                "odstupaju od BTC trenda — vjerovatno dijele zajednički uzrok, tretiraj kao jednu "
                "efektivnu poziciju, ne kao nezavisne prilike."
            )
        else:
            note = (
                f"{len(members)} signala ({', '.join(symbols)}) dijele {direction.upper()} smjer i prate BTC "
                f"trend — ovo je vjerovatno jedna efektivna oklada na smjer tržišta, ne {len(members)} "
                "nezavisnih prilika. Razmisli o manjem ukupnom izlaganju ako uzmeš više od jednog."
            )
        for sig in members:
            sig["correlation"] = {
                "group_size": len(members),
                "symbols": [s for s in symbols if s != sig["symbol"]],
                "diverges_from_btc": diverges,
                "note": note,
            }


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


def compute_candidates(
    rows: list[dict[str, Any]],
    limit: int = 5,
    muted_buckets: set[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Rank every currently-tracked directional symbol by the conservative
    (95%-CI lower-bound) win rate of its active calibration label(s), so a
    lucky small sample can never outrank a well-tested edge, and require a
    minimum sample size before a label counts as "significant" at all.

    ``rows`` is the output of :func:`scanner.storage.fetch_latest_scan_per_symbol`
    (or an equivalent list of per-symbol scan dicts). Read-only and always
    fast: it only ever reads whatever's already in the on-disk calibration
    cache and never triggers a fresh backtest inline — a symbol simply has
    no entry here until its cache is warm.

    ``muted_buckets``, if given, is a set of ``(timeframe, direction)`` pairs
    that the live calibration-drift detector (see :mod:`scanner.drift_detector`)
    has found to be underperforming their promised win rate badly enough,
    over enough resolved live signals, to stop recommending — see
    :func:`scanner.storage.fetch_all_active_muted_buckets`. A muted bucket's
    candidates are excluded entirely from both the ranking and the returned
    count of "significant" matches, so the dashboard and the live tracker
    (:func:`scanner.signal_tracking.log_new_signals`) never show or track an
    opportunity this project's own live track record says isn't holding up.
    """
    if not rows:
        return {
            "signals": [],
            "generated_at": None,
            "candidates_considered": 0,
            "cache_coverage": None,
            "muted_buckets": sorted(muted_buckets or set()),
        }

    muted_buckets = muted_buckets or set()
    from agents_crypto.calibration_lookup import find_stats_for_labels, get_cached_calibration_stats
    from agents_crypto.risk_calc import compute_kelly_fraction, compute_risk_levels

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
        if (ref_tf, best.direction) in muted_buckets:
            continue  # live track record says this bucket isn't holding up -- see muted_buckets docstring above

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

        # Conservative (CI lower-bound) win rate feeds Kelly sizing, same
        # philosophy as the ranking itself: never let a lucky point estimate
        # suggest a bigger bet than the evidence actually supports.
        kelly = compute_kelly_fraction(best.win_rate_ci_low_pct, best.avg_win_return_pct, best.avg_loss_return_pct)

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
                "kelly_sizing": {
                    "recommended_fraction_pct": round(kelly.recommended_fraction * 100, 2),
                    "full_kelly_fraction_pct": round(kelly.full_kelly_fraction * 100, 2),
                    "warning": kelly.warning,
                },
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
    top_signals = candidates[: max(1, limit)]

    btc_row = next((r for r in rows if r["symbol"] == BTC_SYMBOL), None)
    _tag_correlated_exposure(top_signals, btc_row["trend"] if btc_row else None)

    return {
        "signals": top_signals,
        "generated_at": generated_at,
        "candidates_considered": considered,
        "cache_coverage": f"{len(candidates)}/{considered}",
        "muted_buckets": sorted(muted_buckets),
    }
