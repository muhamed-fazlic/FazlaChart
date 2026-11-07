"""Strategy Lab Layer 1 -- the continuous, train-data-only, purely
DESCRIPTIVE strategy miner (see ``STRATEGY_LAB_DESIGN.md``).

Evaluates a small, EXPLICITLY-DECLARED grid of variants built from the
three existing frozen signal templates:

* Template A -- trend/anomaly (family #1, ``PHASE8_PROFITABILITY_RESEARCH_TODO.md``
  Korak 7): reuses ``scanner.backtesting.portfolio_backtest.run_portfolio_backtest``
  and ``scanner.backtesting.strategy_variants`` AS-IS, with its two already
  -frozen variants (``original``/``inverted``) as the grid -- no new knobs
  invented on this template.
* Template B -- funding-extreme reversion (family #2): reuses
  ``scanner.backtesting.funding_extreme_backtest.run_backtest`` with its
  percentile-threshold/hold-hours constants TEMPORARILY overridden (see
  ``_patched_module_constants``) for a small, pre-declared grid around the
  frozen 95/5/8h baseline.
* Template C -- cross-sectional momentum (family #3): reuses
  ``scanner.backtesting.cross_sectional_momentum_backtest.run_backtest``
  with its lookback/basket-size constants similarly overridden for a small
  grid around the frozen 24h/N=5 baseline.

``_patched_module_constants`` does NOT edit those frozen modules on disk --
it temporarily reassigns module-level attributes (which those modules'
functions look up as globals at call time) for the duration of one call,
then restores the originals in a ``finally`` block. This is the mechanism
that lets Layer 1 explore nearby parameterizations WITHOUT duplicating a
single line of the frozen backtest/signal-generation logic, and without
ever permanently retuning the closed, frozen hypotheses those modules
still serve (their own CLI/tests/default behaviour are completely
unaffected -- the patch is undone before this module's caller ever sees
control again). Single-process, sequential use only (not safe to call
concurrently from multiple threads/processes against the same module).

Every combination this module evaluates operates ONLY on data at or before
``end_timestamp_ms`` (passed through unchanged to the frozen backtest
functions' own ``end_timestamp_ms`` argument, which already guarantees no
look-ahead past that point) -- the same frozen train/validation boundary
used by the three closed hypothesis families
(``research_registry.json``'s ``shared_holdout.end_timestamp_ms``). The
existing locked 6-week holdout AFTER that boundary is never read here.
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
import statistics
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from scanner.backtesting import (
    cross_sectional_momentum_backtest as csm,
    funding_extreme_backtest as feb,
)
from scanner.backtesting.historical_universe import windows_for_rank_range
from scanner.backtesting.portfolio_backtest import PortfolioBacktestConfig, run_portfolio_backtest
from scanner.backtesting.strategy_variants import STRATEGY_VARIANTS

logger = logging.getLogger(__name__)

# --- Liquidity terciles (Korak 10 infra) -------------------------------
# 1-indexed, inclusive rank ranges over a single ranked point-in-time
# universe build (see scanner.backtesting.historical_universe). The
# production-scale build (170 symbols) is the separate "Korak 10
# exploratory scan" effort; this module only consumes an already-built
# ranked cache (via ``tercile_rank_ranges``/``windows_for_rank_range``) or,
# absent one, falls back to treating the whole supplied symbol list as a
# single flat "tercile" (documented below in ``run_mining_cycle``).
DEFAULT_TERCILE_RANK_RANGES: dict[str, tuple[int, int]] = {
    "top": (1, 57),
    "mid": (58, 113),
    "bottom": (114, 170),
}


@contextlib.contextmanager
def _patched_module_constants(module: Any, **overrides: Any):
    """Temporarily overrides module-level attributes in ``module``,
    restoring the originals on exit (success or exception) -- see this
    file's module docstring for why this is the chosen mechanism instead
    of editing the frozen backtest modules."""
    originals = {name: getattr(module, name) for name in overrides}
    try:
        for name, value in overrides.items():
            setattr(module, name, value)
        yield
    finally:
        for name, value in originals.items():
            setattr(module, name, value)


@dataclass(frozen=True)
class TemplateVariant:
    """One pre-declared grid point: a named parameterization of one
    template. ``overrides`` is empty for a variant that needs no module
    patching (e.g. trend/anomaly's variants, which are selected via
    ``strategy_variant`` rather than constant-patching)."""

    template: str
    variant_label: str
    params: dict[str, Any]
    overrides: dict[str, Any] = field(default_factory=dict)


# --- Pre-declared parameter grids (bounded, explicit -- NOT a search) ---
# Template B bounds (documented): percentile threshold within [93, 97] of
# the frozen 95/5 baseline, hold time within [4, 12]h of the frozen 8h
# baseline. 3 variants (<= 5 per template, per Strategy Lab's hard limit).
FUNDING_EXTREME_VARIANTS: tuple[TemplateVariant, ...] = (
    TemplateVariant(
        "funding_extreme_reversion",
        "baseline (95/5, 8h hold)",
        {"short_percentile_threshold": 95.0, "long_percentile_threshold": 5.0, "hold_hours": 8},
        {},
    ),
    TemplateVariant(
        "funding_extreme_reversion",
        "tighter percentile (97/3, 8h hold)",
        {"short_percentile_threshold": 97.0, "long_percentile_threshold": 3.0, "hold_hours": 8},
        {"SHORT_PERCENTILE_THRESHOLD": 97.0, "LONG_PERCENTILE_THRESHOLD": 3.0},
    ),
    TemplateVariant(
        "funding_extreme_reversion",
        "shorter hold (95/5, 4h hold)",
        {"short_percentile_threshold": 95.0, "long_percentile_threshold": 5.0, "hold_hours": 4},
        {"HOLD_HOURS": 4},
    ),
)

# Template C bounds (documented): lookback within [12, 48]h of the frozen
# 24h baseline, basket size (N) within [3, 8] of the frozen N=5 baseline
# (MIN_ELIGIBLE_SYMBOLS = 2*N is kept consistent with BASKET_SIZE in every
# overridden variant, mirroring the frozen module's own invariant).
CROSS_SECTIONAL_MOMENTUM_VARIANTS: tuple[TemplateVariant, ...] = (
    TemplateVariant(
        "cross_sectional_momentum",
        "baseline (24h lookback, N=5)",
        {"lookback_hours": 24, "basket_size": 5},
        {},
    ),
    TemplateVariant(
        "cross_sectional_momentum",
        "shorter lookback (12h, N=5)",
        {"lookback_hours": 12, "basket_size": 5},
        {"LOOKBACK_HOURS": 12},
    ),
    TemplateVariant(
        "cross_sectional_momentum",
        "wider basket (24h, N=8)",
        {"lookback_hours": 24, "basket_size": 8},
        {"BASKET_SIZE": 8, "MIN_ELIGIBLE_SYMBOLS": 16},
    ),
)

# Template A: the strategy_variants.py framework's own already-frozen
# variants -- no new parameterization invented here.
TREND_ANOMALY_VARIANTS: tuple[TemplateVariant, ...] = tuple(
    TemplateVariant(
        "trend_anomaly",
        STRATEGY_VARIANTS[variant_id].label,
        {"strategy_variant": variant_id},
        {},
    )
    for variant_id in STRATEGY_VARIANTS
)

ALL_TEMPLATE_VARIANTS: tuple[TemplateVariant, ...] = (
    TREND_ANOMALY_VARIANTS + FUNDING_EXTREME_VARIANTS + CROSS_SECTIONAL_MOMENTUM_VARIANTS
)


@dataclass(frozen=True)
class MiningCandidate:
    """One (template, tercile, variant) result from a single mining cycle
    -- purely descriptive. Deliberately has NO pass/fail field: Layer 1
    output is "where to look", never a finding (see module docstring)."""

    template: str
    tercile: str
    variant_label: str
    params: dict[str, Any]
    n_independent: int
    gross_observed_pct: float | None
    net_observed_pct: float | None
    stability_score: float | None
    composite_score: float
    report: dict[str, Any]


def compute_stability_score(event_net_returns: list[float]) -> float | None:
    """Simple, documented (not a statistical claim) cross-sub-period
    consistency check: splits the chronologically-ordered
    ``event_net_returns`` into 3 equal-ish thirds and scores how many of
    them agree in SIGN with the overall mean (0/3, 1/3, 2/3, or 3/3,
    normalized to 0.0-1.0). Returns ``None`` if fewer than 6 events (too
    few to split into 3 meaningful groups).

    This is intentionally crude -- a real stability measure would need its
    own validation; its only job here is to down-rank candidates whose
    apparent effect only exists in one sub-period (a classic overfitting
    smell), not to certify anything.
    """
    n = len(event_net_returns)
    if n < 6:
        return None
    overall_mean = statistics.mean(event_net_returns)
    overall_sign = 1 if overall_mean >= 0 else -1
    third = n // 3
    thirds = [
        event_net_returns[:third],
        event_net_returns[third : 2 * third],
        event_net_returns[2 * third :],
    ]
    agree = 0
    for group in thirds:
        if not group:
            continue
        group_mean = statistics.mean(group)
        group_sign = 1 if group_mean >= 0 else -1
        if group_sign == overall_sign:
            agree += 1
    return agree / 3.0


def _composite_score(net_observed_pct: float | None, stability_score: float | None) -> float:
    """Documented ranking heuristic (NOT a CI, NOT a pass/fail signal):
    net effect size scaled down for instability. A candidate with no
    usable net observation (e.g. zero events) always scores
    ``-math.inf`` so it never outranks anything with real data."""
    if net_observed_pct is None:
        return -math.inf
    weight = stability_score if stability_score is not None else 0.5
    return net_observed_pct * weight


def _run_funding_extreme_variant(
    symbols: tuple[str, ...], end_timestamp_ms: int, point_in_time_universe_cache: str | None, variant: TemplateVariant
) -> tuple[dict[str, Any], list[float]]:
    with _patched_module_constants(feb, **variant.overrides):
        report = feb.run_backtest(
            symbols, end_timestamp_ms=end_timestamp_ms, point_in_time_universe_cache=point_in_time_universe_cache
        )
    event_net_returns = [t["net_return_pct"] for t in report["trades"]]
    return report, event_net_returns


def _run_cross_sectional_variant(
    symbols: tuple[str, ...], end_timestamp_ms: int, point_in_time_universe_cache: str | None, variant: TemplateVariant
) -> tuple[dict[str, Any], list[float]]:
    with _patched_module_constants(csm, **variant.overrides):
        report = csm.run_backtest(
            symbols, end_timestamp_ms=end_timestamp_ms, point_in_time_universe_cache=point_in_time_universe_cache
        )
    event_net_returns = [p["net_return_pct"] for p in report["rebalance_periods"]]
    return report, event_net_returns


def _run_trend_anomaly_variant(
    symbols: tuple[str, ...], end_timestamp_ms: int, variant: TemplateVariant
) -> tuple[dict[str, Any], list[float]]:
    config = PortfolioBacktestConfig(
        symbols=symbols,
        end_timestamp_ms=end_timestamp_ms,
        strategy_variant=variant.params["strategy_variant"],
    )
    result = run_portfolio_backtest(config)
    event_net_returns = [t.net_return_pct for t in result.trades]
    report = {
        "hypothesis": "trend_anomaly",
        "variant": variant.params["strategy_variant"],
        "metrics": result.metrics,
        "n_trades": len(result.trades),
    }
    return report, event_net_returns


@dataclass(frozen=True)
class VariantEvaluation:
    """Generic result of running ONE template variant against ONE symbol
    set/time window -- shared by Layer 1's mining cycle, the weekly
    snapshot, and Layer 2's confirmation run so none of them duplicate the
    per-template dispatch logic below."""

    report: dict[str, Any]
    event_net_returns: list[float]
    n_independent: int
    net_observed_pct: float | None
    gross_observed_pct: float | None


def evaluate_variant(
    variant: TemplateVariant,
    symbols: tuple[str, ...],
    end_timestamp_ms: int,
    point_in_time_universe_cache: str | None,
) -> VariantEvaluation:
    """Runs ``variant`` against ``symbols``/``end_timestamp_ms`` through
    the correct frozen backtest module, normalizing the result into
    ``VariantEvaluation``. Shared by ``run_mining_cycle`` (Layer 1),
    ``run_weekly_snapshot``, and Layer 2's confirmation run -- the ONE
    place that knows how to dispatch by ``variant.template``."""
    if variant.template == "funding_extreme_reversion":
        report, event_net_returns = _run_funding_extreme_variant(
            symbols, end_timestamp_ms, point_in_time_universe_cache, variant
        )
        n_independent = report["n_independent"]
        net_observed_pct = report["net_block_bootstrap"]["observed_pct"] if report["net_block_bootstrap"] else None
        gross_observed_pct = (
            report["gross_block_bootstrap"]["observed_pct"] if report["gross_block_bootstrap"] else None
        )
    elif variant.template == "cross_sectional_momentum":
        report, event_net_returns = _run_cross_sectional_variant(
            symbols, end_timestamp_ms, point_in_time_universe_cache, variant
        )
        n_independent = report["n_independent"]
        net_observed_pct = report["net_block_bootstrap"]["observed_pct"] if report["net_block_bootstrap"] else None
        gross_observed_pct = (
            report["gross_block_bootstrap"]["observed_pct"] if report["gross_block_bootstrap"] else None
        )
    elif variant.template == "trend_anomaly":
        report, event_net_returns = _run_trend_anomaly_variant(symbols, end_timestamp_ms, variant)
        n_independent = report["metrics"].get("n_independent_episodes", 0)
        net_observed_pct = report["metrics"].get("net_profit_pct")
        gross_observed_pct = None  # not generically available for this template's equity metric
    else:
        raise ValueError(f"unknown template {variant.template!r}")
    return VariantEvaluation(
        report=report,
        event_net_returns=event_net_returns,
        n_independent=n_independent,
        net_observed_pct=net_observed_pct,
        gross_observed_pct=gross_observed_pct,
    )


def _write_tercile_universe_cache(windows: list[Any], tercile_name: str) -> Path:
    """Writes ``windows`` (already tercile-sliced ``UniverseWindow``s) out
    in the OLD ``save_universe_cache`` schema (``{"symbols": [...]}`` per
    window) to a fresh temp file, and returns its path. Caller owns
    deleting the file once done with it (``Path.unlink(missing_ok=True)``)."""
    from scanner.backtesting.historical_universe import save_universe_cache

    fd, raw_path = tempfile.mkstemp(prefix=f"strategy_lab_tercile_{tercile_name}_", suffix=".json")
    os.close(fd)  # mkstemp leaves the fd open; close it so the file can be written/deleted (notably on Windows)
    tmp_file = Path(raw_path)
    save_universe_cache(windows, tmp_file)
    return tmp_file


@contextlib.contextmanager
def tercile_universe_cache_path(
    ranked_universe_cache_path: str,
    tercile_name: str,
    tercile_rank_ranges: dict[str, tuple[int, int]] | None = None,
):
    """Context manager yielding a path to a TEMPORARY old-schema universe
    cache file scoped to one tercile, derived from a ranked universe cache
    (``historical_universe.save_ranked_universe_cache`` output). Always use
    this (never forward a ranked cache path directly) before calling
    ``evaluate_variant`` for ``funding_extreme_reversion``/
    ``cross_sectional_momentum`` -- their frozen backtest modules only
    understand the old ``{"symbols": [...]}`` schema, not the ranked
    cache's ``{"ranked_symbols": [...]}`` one, and forwarding the wrong one
    raises ``KeyError: 'symbols'`` (a real production bug this helper
    exists to prevent recurring). The temp file is deleted on exit."""
    from scanner.backtesting.historical_universe import load_ranked_universe_cache

    tercile_rank_ranges = tercile_rank_ranges or DEFAULT_TERCILE_RANK_RANGES
    start_rank, end_rank = tercile_rank_ranges[tercile_name]
    ranked_windows = load_ranked_universe_cache(ranked_universe_cache_path)
    full_tercile_windows = windows_for_rank_range(ranked_windows, start_rank, end_rank)
    tmp_file = _write_tercile_universe_cache(full_tercile_windows, tercile_name)
    try:
        yield str(tmp_file)
    finally:
        tmp_file.unlink(missing_ok=True)


def run_mining_cycle(
    as_of_ts: int,
    *,
    symbols: tuple[str, ...] | None = None,
    tercile_rank_ranges: dict[str, tuple[int, int]] | None = None,
    ranked_universe_cache_path: str | None = None,
    template_variants: tuple[TemplateVariant, ...] = ALL_TEMPLATE_VARIANTS,
    already_mined_keys: frozenset[tuple[str, str, str]] | None = None,
) -> list[MiningCandidate]:
    """Runs exactly one Layer 1 mining cycle: every (template, tercile,
    variant) combination in ``template_variants`` x ``tercile_rank_ranges``,
    using only data at or before ``as_of_ts``. Prints/logs the EXACT total
    combination count up front (same "count must be known upfront"
    discipline as the Korak 10 exploratory scan) before running anything.

    ``already_mined_keys`` is an optional set of ``(template, tercile,
    variant_label)`` tuples to SKIP entirely (no backtest re-run, no
    candidate produced for them). Pass in every combination already stored
    for this exact ``as_of_ts`` to avoid pointlessly re-computing a result
    that is mathematically guaranteed to be byte-identical to last time --
    ``as_of_ts`` is clamped to a frozen train/holdout boundary
    (``SHARED_HOLDOUT_END_MS``) once real time passes it, so re-running the
    same combination against the same frozen cutoff forever produces the
    same answer and only wastes CPU and pollutes storage with duplicate
    rows (a real bug this parameter exists to prevent).

    ``ranked_universe_cache_path`` should point at a Korak-10-style ranked
    point-in-time universe cache (``historical_universe.save_ranked_universe_cache``
    output) covering the full wide universe (e.g. 170 symbols); this
    function then slices it into the requested terciles via
    ``windows_for_rank_range`` and runs each template/variant once per
    tercile's symbol set. If no such cache is supplied (e.g. before the
    separate Korak 10 exploratory-scan effort has produced one), callers
    MUST instead supply an explicit ``symbols`` tuple, which is then used
    unsliced as a SINGLE flat tercile named ``"unsliced"`` -- a documented,
    smaller-scale fallback for engineering verification, NOT a substitute
    for the real 170-symbol/3-tercile run once that cache exists.

    Internally, each tercile also gets its OWN point-in-time universe cache
    file written in the older ``save_universe_cache``/``load_universe_cache``
    schema (``{"symbols": [...]}`` per window) -- ``funding_extreme_reversion``
    and ``cross_sectional_momentum``'s frozen backtest modules only understand
    that schema, not the ranked cache's ``{"ranked_symbols": [...]}`` one, so
    the raw ``ranked_universe_cache_path`` must never be forwarded to them
    directly (doing so raises ``KeyError: 'symbols'`` for every one of their
    combinations, every cycle -- this was a real bug, not a hypothetical one).
    These per-tercile files are temporary and removed once the cycle ends.
    """
    tercile_rank_ranges = tercile_rank_ranges or DEFAULT_TERCILE_RANK_RANGES

    tercile_symbol_sets: dict[str, tuple[str, ...]] = {}
    tercile_universe_cache_paths: dict[str, str | None] = {}
    tmp_cache_files: list[Path] = []
    if ranked_universe_cache_path is not None:
        from scanner.backtesting.historical_universe import load_ranked_universe_cache

        ranked_windows = load_ranked_universe_cache(ranked_universe_cache_path)
        if not ranked_windows:
            raise ValueError(f"ranked universe cache at {ranked_universe_cache_path!r} is empty")
        latest_window = ranked_windows[-1]
        for tercile_name, (start_rank, end_rank) in tercile_rank_ranges.items():
            sliced = windows_for_rank_range([latest_window], start_rank, end_rank)
            tercile_symbol_sets[tercile_name] = tuple(sorted(sliced[-1].symbols)) if sliced else ()

            # Full multi-window slice (not just the latest window) so
            # funding_extreme/cross_sectional still get real point-in-time
            # survivorship-bias protection across the whole backtest period,
            # same as trend_anomaly's Korak-10 runs did.
            full_tercile_windows = windows_for_rank_range(ranked_windows, start_rank, end_rank)
            tmp_file = _write_tercile_universe_cache(full_tercile_windows, tercile_name)
            tmp_cache_files.append(tmp_file)
            tercile_universe_cache_paths[tercile_name] = str(tmp_file)
    else:
        if not symbols:
            raise ValueError("symbols must be provided when ranked_universe_cache_path is not set")
        tercile_symbol_sets["unsliced"] = tuple(symbols)
        logger.warning(
            "No ranked_universe_cache_path supplied -- falling back to a single flat 'unsliced' tercile "
            "over the %d symbols provided. This is a documented engineering fallback, not the real "
            "170-symbol / 3-tercile mining run.",
            len(symbols),
        )

    total_combinations = len(template_variants) * len(tercile_symbol_sets)
    already_mined_keys = already_mined_keys or frozenset()
    skip_count = sum(
        1
        for tercile_name in tercile_symbol_sets
        for variant in template_variants
        if (variant.template, tercile_name, variant.variant_label) in already_mined_keys
    )
    logger.info(
        "Strategy Lab mining cycle: %d template variants x %d terciles = %d total combinations "
        "(as_of_ts=%d) -- %d already mined at this as_of_ts and will be skipped, %d new",
        len(template_variants),
        len(tercile_symbol_sets),
        total_combinations,
        as_of_ts,
        skip_count,
        total_combinations - skip_count,
    )
    print(
        f"Strategy Lab mining cycle: {len(template_variants)} template variants x "
        f"{len(tercile_symbol_sets)} terciles = {total_combinations} total combinations "
        f"(as_of_ts={as_of_ts}) -- {skip_count} already mined and skipped, {total_combinations - skip_count} new"
    )

    candidates: list[MiningCandidate] = []
    try:
        for tercile_name, tercile_symbols in tercile_symbol_sets.items():
            if not tercile_symbols:
                logger.warning(
                    "Tercile %r has no symbols at this as_of_ts -- skipping all variants for it", tercile_name
                )
                continue
            tercile_cache_path = tercile_universe_cache_paths.get(tercile_name)
            for variant in template_variants:
                if (variant.template, tercile_name, variant.variant_label) in already_mined_keys:
                    continue
                try:
                    evaluation = evaluate_variant(variant, tercile_symbols, as_of_ts, tercile_cache_path)
                except Exception:
                    logger.exception(
                        "Mining combination failed (template=%s tercile=%s variant=%s) -- skipping, not aborting "
                        "the whole cycle",
                        variant.template,
                        tercile_name,
                        variant.variant_label,
                    )
                    continue

                stability_score = compute_stability_score(evaluation.event_net_returns)
                composite = _composite_score(evaluation.net_observed_pct, stability_score)
                candidates.append(
                    MiningCandidate(
                        template=variant.template,
                        tercile=tercile_name,
                        variant_label=variant.variant_label,
                        params=variant.params,
                        n_independent=evaluation.n_independent,
                        gross_observed_pct=evaluation.gross_observed_pct,
                        net_observed_pct=evaluation.net_observed_pct,
                        stability_score=stability_score,
                        composite_score=composite,
                        report=evaluation.report,
                    )
                )
    finally:
        for tmp_file in tmp_cache_files:
            tmp_file.unlink(missing_ok=True)

    candidates.sort(key=lambda c: c.composite_score, reverse=True)
    return candidates
