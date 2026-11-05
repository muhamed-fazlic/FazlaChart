"""Phase 8 "strategy variants" framework: a small, explicit registry mapping
a short ``variant_id`` to the keyword overrides it needs on
:func:`scanner.calibration.backtest_symbol`, so every future strategy idea
(not just the original signal) can be run through the *exact same*
validation pipeline — OOS split, sample-independence correction,
walk-forward, Deflated Sharpe Ratio, block bootstrap, experiment registry —
without any of that code needing to know variants exist at all.

Why this exists instead of a one-off ``invert_direction`` flag wired only
into ``portfolio_backtest.py``: a naive "negate the final return" transform
is wrong whenever SL and TP distances are asymmetric (the live system's
ATR multipliers are, by default), because the actually-taken trade can exit
on a completely different candle than the original signal did. A true
variant has to change what gets *walked* at the labeling layer
(:func:`scanner.calibration.backtest_symbol`/``_walk_forward_exit``), not
just post-process an already-resolved return. Every variant below is
therefore defined purely in terms of ``backtest_symbol`` kwargs — adding a
new strategy idea later (e.g. a wider/narrower ATR multiplier, a different
tie-break) means adding one entry here, nothing else.

Each variant also gets its own ``experiment_family`` suffix (see
:func:`family_suffix`) so its trial history / Deflated Sharpe Ratio in the
experiment registry is never pooled with a different variant's attempts --
comparing "original" trials against "inverted" trials under one family name
would understate both (Korak 3's whole point). Note this does NOT give a
true cross-variant multiple-testing budget: DSR/PBO still only correct for
trials *within* a family. Treating "we tried 2 variants" itself as 2
additional trials is a known, open gap -- see
PHASE8_PROFITABILITY_RESEARCH_TODO.md's Korak 7 Test 3 section.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class StrategyVariant:
    variant_id: str
    label: str
    description: str
    backtest_symbol_kwargs: dict[str, Any] = field(default_factory=dict)


STRATEGY_VARIANTS: dict[str, StrategyVariant] = {
    "original": StrategyVariant(
        variant_id="original",
        label="Original signal",
        description="The scanner's signal taken as-is (long on bullish labels, short on bearish).",
        backtest_symbol_kwargs={},
    ),
    "inverted": StrategyVariant(
        variant_id="inverted",
        label="Fade (invert) signal",
        description=(
            "Every directional label is traded in the OPPOSITE direction, with the "
            "stop/target ATR distances swapped (the original signal's target distance "
            "becomes the faded trade's stop, and vice versa)."
        ),
        backtest_symbol_kwargs={"invert_signals": True},
    ),
}

DEFAULT_VARIANT_ID = "original"


def get_variant(variant_id: str | None) -> StrategyVariant:
    """Looks up a variant by id, defaulting to ``"original"`` for ``None``
    or an empty string (keeps every existing caller that doesn't know about
    variants working unchanged). Raises ``KeyError`` for an unknown id
    rather than silently falling back, since that almost always means a
    typo'd CLI flag or stale query param, not an intentional choice."""
    key = variant_id or DEFAULT_VARIANT_ID
    if key not in STRATEGY_VARIANTS:
        known = ", ".join(sorted(STRATEGY_VARIANTS))
        raise KeyError(f"Unknown strategy variant {key!r}. Known variants: {known}")
    return STRATEGY_VARIANTS[key]


def family_suffix(variant_id: str) -> str:
    """"original" keeps the family name unchanged (so it stays comparable
    with every pre-existing experiment-registry row recorded before this
    framework existed); every other variant gets an explicit ``__<id>``
    suffix so its trial history is never pooled with a different variant's."""
    return "" if variant_id == DEFAULT_VARIANT_ID else f"__{variant_id}"


def report_path_for_variant(base_path: str, variant_id: str) -> str:
    """Derives a variant's own file path from a base path (e.g.
    ``scanner_data/portfolio_backtest_report.json`` ->
    ``scanner_data/portfolio_backtest_report__inverted.json``, or
    ``scanner_data/backtest_refresh.log`` ->
    ``scanner_data/backtest_refresh__inverted.log``), so each variant's
    report/status/log file lives on its own instead of overwriting a
    shared one -- ``"original"`` keeps the base path unchanged so every
    existing deployment/env-var keeps working without reconfiguration."""
    if variant_id == DEFAULT_VARIANT_ID:
        return base_path
    root, ext = os.path.splitext(base_path)
    return f"{root}__{variant_id}{ext}"
