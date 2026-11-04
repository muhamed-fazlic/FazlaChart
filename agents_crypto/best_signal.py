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

# Every timeframe the scanner snapshots gets its own independent candidate
# search and its own calibration lookup (never mixed with another
# timeframe's stats) -- a symbol can be a 1h short AND a 4h long
# simultaneously, and each is ranked purely against its own track record.
TRACKED_TIMEFRAMES: tuple[str, ...] = ("5m", "15m", "1h", "4h")

# Below this many backtest cases, even a high win rate is too noisy to call a
# real edge -- but "enough" isn't the same number on every timeframe. The
# backtest always walks the same fixed ~1500-candle window (see
# ``scanner.calibration.backtest_symbol``), so a slower timeframe spans far
# more calendar time per case (1500 4h candles is ~250 days vs. ~5 days for
# 1500 5m candles): its label occurrences are rarer but each one reflects a
# more distinct, less autocorrelated market situation, while a fast
# timeframe's cases are noisier and more clustered in time, so the same raw
# count buys less real statistical confidence. 1h keeps the original bar (30)
# unchanged -- that's the one timeframe this project already had a live
# track record against before the multi-timeframe engine shipped.
BEST_SIGNAL_MIN_SAMPLE_SIZE: dict[str, int] = {
    "5m": 40,
    "15m": 35,
    "1h": 30,
    "4h": 20,
}
# Fallback for any timeframe not explicitly listed above (shouldn't happen
# for anything in TRACKED_TIMEFRAMES, but keeps lookups total).
_DEFAULT_MIN_SAMPLE_SIZE = 30


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


def _rank_key(candidate: dict[str, Any]) -> tuple[float, float, float, int]:
    # Primary: the Signal Quality Score (see agents_crypto.signal_quality) --
    # live trend/momentum/volume/multi-timeframe-alignment/volatility
    # confirmation among candidates that have ALL already cleared the
    # statistical gate (min sample size + CI-low win rate > 50%, computed in
    # compute_candidates below). This never lets a candidate with no proven
    # historical edge outrank one that has one; it only decides order among
    # candidates that already do. Ties on quality score (rounded to 1
    # decimal) fall back to the CI's conservative lower bound, then the raw
    # win rate, then sample size, instead of database row order.
    return (
        candidate["quality_score"]["total"],
        candidate["win_rate_ci_low_pct"],
        candidate["win_rate_pct"],
        candidate["n"],
    )


def compute_candidates(
    rows: list[dict[str, Any]],
    limit: int = 5,
    muted_buckets: set[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Rank every currently-tracked directional (symbol, timeframe) pair by
    the conservative (95%-CI lower-bound) win rate of its active calibration
    label(s), so a lucky small sample can never outrank a well-tested edge,
    and require a minimum sample size before a label counts as
    "significant" at all.

    Every timeframe in :data:`TRACKED_TIMEFRAMES` is searched and calibrated
    independently per symbol (see that constant's docstring) — a symbol can
    surface as, say, a 1h short and a 4h long at the same time, each ranked
    purely against its own (symbol, timeframe) track record, never a
    different timeframe's.

    Each label's stats are additionally scoped to the symbol's *current*
    volatility regime ("low"/"normal"/"high", carried on each timeframe's
    snapshot as ``volatility_regime`` — see
    ``tradingagents.dataflows.vendors.binance.indicators.current_volatility_regime``)
    whenever there's enough regime-specific backtest history for that label,
    falling back to the all-regime blend otherwise (see
    :func:`agents_crypto.calibration_lookup.find_cached_regime_matched_stats`)
    — a candidate's ``regime_fallback`` field tells you which one happened.
    "Market Regime Detection": a label's edge during a calm, grinding trend
    and the same label during a wild, whipsawing market are not the same bet,
    so the ranking should answer "how has this done in conditions LIKE RIGHT
    NOW", not just an all-time average.

    ``rows`` is the output of :func:`scanner.storage.fetch_latest_scan_per_symbol`
    (or an equivalent list of per-symbol scan dicts) — its full ``snapshot``
    (every timeframe's trend/anomalies/indicators, not just the flattened
    reference-timeframe columns) is what gets walked here. Read-only and
    always fast: it only ever reads whatever's already in the on-disk
    calibration cache and never triggers a fresh backtest inline — a
    (symbol, timeframe) pair simply has no entry here until its cache is
    warm.

    ``limit`` caps how many candidates are kept **per timeframe** (see
    ``signals_by_timeframe`` below), not overall.

    ``muted_buckets``, if given, is a set of ``(timeframe, direction)`` pairs
    that the live calibration-drift detector (see :mod:`scanner.drift_detector`)
    has found to be underperforming their promised win rate badly enough,
    over enough resolved live signals, to stop recommending — see
    :func:`scanner.storage.fetch_all_active_muted_buckets`. A muted bucket's
    candidates are excluded entirely from both the ranking and the returned
    count of "significant" matches, so the dashboard and the live tracker
    (:func:`scanner.signal_tracking.log_new_signals`) never show or track an
    opportunity this project's own live track record says isn't holding up.

    Returns a dict with:
    - ``signals``: every surfaced candidate across all timeframes, flattened
      into one list and sorted by the same conservative ranking (for the
      live tracker, and any caller that just wants "everything that
      qualified" regardless of timeframe).
    - ``signals_by_timeframe``: the same candidates grouped by timeframe,
      each list independently capped at ``limit`` and sorted the same way —
      what the dashboard's per-timeframe Top N panel renders.
    """
    if not rows:
        return {
            "signals": [],
            "signals_by_timeframe": {tf: [] for tf in TRACKED_TIMEFRAMES},
            "generated_at": None,
            "candidates_considered": 0,
            "cache_coverage": None,
            "muted_buckets": sorted(muted_buckets or set()),
        }

    muted_buckets = muted_buckets or set()
    from agents_crypto.calibration_lookup import find_cached_regime_matched_stats
    from agents_crypto.risk_calc import compute_kelly_fraction, compute_risk_levels
    from agents_crypto.signal_pattern import classify_pattern
    from agents_crypto.signal_quality import compute_quality_score

    candidates: list[dict[str, Any]] = []
    considered = 0
    for row in rows:
        timeframes_snapshot = row.get("snapshot", {}).get("timeframes", {})
        for tf in TRACKED_TIMEFRAMES:
            tf_snapshot = timeframes_snapshot.get(tf)
            if not tf_snapshot:
                continue  # this cycle's snapshot didn't include this timeframe for this symbol
            trend = tf_snapshot.get("trend")
            anomalies_for_tf = tf_snapshot.get("anomalies", [])
            if trend not in ("bullish", "bearish") and not anomalies_for_tf:
                continue  # neutral trend with no anomaly-based setup either -- nothing directional could possibly match
            considered += 1
            # Anomalies here come from this timeframe's own (unprefixed)
            # list, not the flattened/cross-timeframe ``row["anomalies"]``
            # (which prefixes each entry with its timeframe, e.g.
            # "1h:failed_breakout") — using the per-timeframe list both
            # keeps each timeframe's labels scoped to itself AND matches the
            # unprefixed label format the historical backtest in
            # scanner.calibration actually trained on, so anomaly-based
            # labels can match calibration stats at all.
            labels = _active_labels(trend, anomalies_for_tf)

            # Match against the symbol's *current* volatility regime
            # ("low"/"normal"/"high", from this same cycle's snapshot --
            # see tradingagents...indicators.current_volatility_regime) when
            # there's enough regime-specific backtest history for the active
            # label(s), falling back to the all-regime blend otherwise --
            # same "how has this label done in conditions LIKE RIGHT NOW"
            # philosophy as the single-symbol analyze endpoint
            # (webapp.main._regime_matched_calibration), but regime-free
            # here since it's already in the snapshot at zero extra cost.
            current_regime = tf_snapshot.get("volatility_regime")
            matches, regime_fallback = find_cached_regime_matched_stats(
                row["symbol"], tf, labels, BEST_SIGNAL_HORIZON_CANDLES, current_regime
            )
            min_n = BEST_SIGNAL_MIN_SAMPLE_SIZE.get(tf, _DEFAULT_MIN_SAMPLE_SIZE)
            significant = [
                m
                for m in matches
                if m.direction in ("long", "short")
                and m.n >= min_n
                and m.win_rate_ci_low_pct is not None
                and m.win_rate_ci_low_pct > 50.0
            ]
            if not significant:
                continue
            # Rank by the CI's conservative lower bound, not the point
            # estimate — rewards both a high win rate AND enough samples to
            # be confident in it.
            best = max(significant, key=lambda m: m.win_rate_ci_low_pct)
            if (tf, best.direction) in muted_buckets:
                continue  # live track record says this bucket isn't holding up -- see muted_buckets docstring above

            close, atr = tf_snapshot.get("close"), tf_snapshot.get("atr")
            entry = None
            if close and atr:
                try:
                    levels = compute_risk_levels(
                        close, atr, best.direction, rsi=tf_snapshot.get("rsi"), ema20=tf_snapshot.get("ema20")
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

            # Signal Quality Score: re-ranks candidates that already cleared
            # the statistical gate above by how strongly LIVE conditions
            # (trend strength, momentum, volume, multi-timeframe alignment,
            # volatility, plus this same historical win rate) confirm the
            # label right now -- see agents_crypto.signal_quality for why.
            quality = compute_quality_score(
                row, tf_snapshot, tf, best.direction, best.win_rate_ci_low_pct, best.n
            )

            candidates.append(
                {
                    "symbol": row["symbol"],
                    "timeframe": tf,
                    "trend": trend,
                    "label": best.label,
                    "pattern": classify_pattern(best.label),
                    "direction": best.direction,
                    "n": best.n,
                    "win_rate_pct": best.win_rate_pct,
                    "win_rate_ci_low_pct": best.win_rate_ci_low_pct,
                    "win_rate_ci_high_pct": best.win_rate_ci_high_pct,
                    "avg_net_return_pct": best.avg_net_return_pct,
                    "regime": best.regime,
                    "current_market_regime": current_regime,
                    "regime_fallback": regime_fallback,
                    "entry": entry,
                    "kelly_sizing": {
                        "recommended_fraction_pct": round(kelly.recommended_fraction * 100, 2),
                        "full_kelly_fraction_pct": round(kelly.full_kelly_fraction * 100, 2),
                        "warning": kelly.warning,
                    },
                    "quality_score": {
                        "total": quality.total,
                        "trend_strength": quality.trend_strength,
                        "momentum": quality.momentum,
                        "volume_confirmation": quality.volume_confirmation,
                        "multi_timeframe_alignment": quality.multi_timeframe_alignment,
                        "volatility_conditions": quality.volatility_conditions,
                        "historical_performance": quality.historical_performance,
                    },
                    "scanned_at": row["scanned_at"],
                    "age_seconds": round(_age_seconds(row["scanned_at"]), 1),
                }
            )


    generated_at = max(r["scanned_at"] for r in rows)

    # Group by timeframe first, rank + cap each group independently at
    # `limit`, then flatten back into one globally-ranked list -- so a
    # timeframe with fewer qualifying candidates never gets crowded out by
    # another timeframe having more.
    by_timeframe: dict[str, list[dict[str, Any]]] = {tf: [] for tf in TRACKED_TIMEFRAMES}
    for c in candidates:
        by_timeframe[c["timeframe"]].append(c)

    signals_by_timeframe: dict[str, list[dict[str, Any]]] = {}
    flattened: list[dict[str, Any]] = []
    for tf in TRACKED_TIMEFRAMES:
        ranked = sorted(by_timeframe[tf], key=_rank_key, reverse=True)[: max(1, limit)]
        signals_by_timeframe[tf] = ranked
        flattened.extend(ranked)
    flattened.sort(key=_rank_key, reverse=True)

    btc_row = next((r for r in rows if r["symbol"] == BTC_SYMBOL), None)
    _tag_correlated_exposure(flattened, btc_row["trend"] if btc_row else None)

    return {
        "signals": flattened,
        "signals_by_timeframe": signals_by_timeframe,
        "generated_at": generated_at,
        "candidates_considered": considered,
        "cache_coverage": f"{len(candidates)}/{considered}",
        "muted_buckets": sorted(muted_buckets),
    }
