"""Korak 2 -- data-driven SL/TP calibration from the live system's own
tracked outcomes, instead of trusting one flat ATR multiplier pair for
every label regardless of how far price actually tends to move for or
against that specific signal.

Every ``signal_outcomes`` row the live tracker resolves already carries
``mfe_pct``/``mae_pct`` (see :mod:`scanner.signal_tracking`): the best and
worst directional excursion price actually reached during the trade's
life, sampled every scan cycle. That is real, lived evidence of how far a
*specific label* tends to run in its favor and against it -- evidence the
flat :data:`agents_crypto.risk_calc.DEFAULT_SL_ATR_MULTIPLIER`/
:data:`agents_crypto.risk_calc.DEFAULT_TP_ATR_MULTIPLIER` never looks at.

This module turns that evidence into a *recommended* per-label
``(sl_atr_multiplier, tp_atr_multiplier)`` pair, expressed in the exact
same ATR-multiple units the live system and
:func:`scanner.calibration.backtest_symbol` already use (not raw
percentages), so the result can be fed straight back into
``backtest_symbol(sl_tp_overrides=...)`` to test whether trading that
label with its own calibrated stop/target actually improves on the flat
default -- which is the entire point: a number that can't be re-tested
against the backtest is just a guess with extra steps.

Deliberately NOT wired into the live scanner or the default backtest run
automatically. A label's calibrated multiplier is only as good as the
resolved sample it came from (see ``min_n``), and silently swapping live
risk parameters out from under a running system without a human comparing
before/after backtest numbers first would be exactly the kind of
unverified "the model says so" decision this whole project set out to
avoid.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass, field
from typing import Any

from scanner.calibration import LABEL_DIRECTION
from scanner.config import ScannerConfig
from scanner.storage import connect_for_reads

# Below this many resolved samples for a label, its empirical MFE/MAE
# distribution is too thin to trust over the live system's flat default --
# matches the project's existing "don't trust a win rate below this many
# samples" convention (agents_crypto.best_signal.BEST_SIGNAL_MIN_SAMPLE_SIZE
# uses the same order of magnitude for the same reason).
DEFAULT_MIN_N = 30

# Percentile of the (ATR-normalized) adverse-excursion distribution used as
# the recommended stop distance. Higher = a wider stop that survives more
# of the label's own historical noise before giving up on a trade; lower =
# a tighter stop that cuts losers faster but risks being clipped by normal
# noise more often. 70 is a deliberately conservative starting point (only
# the worst 30% of historical adverse excursions would have breached it).
DEFAULT_SL_PERCENTILE = 70.0

# Percentile of the (ATR-normalized) favorable-excursion distribution used
# as the recommended target distance. Lower = a target reachable by more
# of the label's historical trades (hits more often, but banks less per
# winner); higher = a more ambitious target. 40 means the target is set at
# a level the label's own history reached or exceeded 60% of the time.
DEFAULT_TP_PERCENTILE = 40.0

# Sanity bounds so a thin or pathological sample can't recommend an
# unusable multiplier (e.g. a near-zero stop that would fire on every tick
# of noise, or a 20x-ATR target no trade could realistically reach).
MIN_ATR_MULTIPLIER = 0.3
MAX_ATR_MULTIPLIER = 5.0


@dataclass
class LabelSlTpCalibration:
    """One label's recommended SL/TP, plus the evidence behind it so a
    human reviewing the output isn't just asked to trust a number."""

    label: str
    direction: str
    n: int
    sl_atr_multiplier: float
    tp_atr_multiplier: float
    median_mae_atr_units: float
    median_mfe_atr_units: float
    sl_percentile: float = DEFAULT_SL_PERCENTILE
    tp_percentile: float = DEFAULT_TP_PERCENTILE

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "direction": self.direction,
            "n": self.n,
            "sl_atr_multiplier": round(self.sl_atr_multiplier, 3),
            "tp_atr_multiplier": round(self.tp_atr_multiplier, 3),
            "median_mae_atr_units": round(self.median_mae_atr_units, 3),
            "median_mfe_atr_units": round(self.median_mfe_atr_units, 3),
            "sl_percentile": self.sl_percentile,
            "tp_percentile": self.tp_percentile,
        }


@dataclass
class SlTpCalibrationReport:
    """Every label with enough resolved evidence to recommend from, plus
    the labels that were skipped and why -- so "no recommendation" is
    always distinguishable from "forgot to check"."""

    calibrations: dict[str, LabelSlTpCalibration] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)

    def overrides(self) -> dict[str, tuple[float, float]]:
        """The ``sl_tp_overrides`` mapping :func:`scanner.calibration.backtest_symbol`
        expects: ``label -> (sl_atr_multiplier, tp_atr_multiplier)``."""
        return {label: (c.sl_atr_multiplier, c.tp_atr_multiplier) for label, c in self.calibrations.items()}

    def as_dict(self) -> dict[str, Any]:
        return {
            "calibrations": {label: c.as_dict() for label, c in self.calibrations.items()},
            "skipped": self.skipped,
        }


def _percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile, no numpy/scipy dependency needed for
    this one-off calculation. ``q`` is 0-100, matching the rest of this
    module's convention."""
    if not values:
        raise ValueError("_percentile() needs at least one value")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (q / 100.0) * (len(ordered) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def _median(values: list[float]) -> float:
    return _percentile(values, 50.0)


def fetch_resolved_outcomes_with_excursion(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Every resolved (``status != 'open'``) tracked signal that has usable
    MFE/MAE data -- i.e. excludes rows logged before those columns existed
    and were only backfilled to a zero-excursion placeholder (see
    ``storage._migrate_schema``'s backfill comment): a real zero-excursion
    trade is indistinguishable from a backfilled placeholder by value
    alone, but a placeholder's ``mfe_pct``/``mae_pct`` are *both* exactly
    0.0 together, which a genuinely resolved trade essentially never is."""
    cursor = conn.execute(
        "SELECT label, direction, entry_price, stop_loss, take_profit, mfe_pct, mae_pct"
        " FROM signal_outcomes"
        " WHERE status != 'open' AND mfe_pct IS NOT NULL AND mae_pct IS NOT NULL"
        " AND NOT (mfe_pct = 0.0 AND mae_pct = 0.0)"
    )
    columns = [c[0] for c in cursor.description]
    return [dict(zip(columns, row, strict=False)) for row in cursor.fetchall()]


def compute_label_sl_tp_calibration(
    rows: list[dict[str, Any]],
    *,
    min_n: int = DEFAULT_MIN_N,
    sl_percentile: float = DEFAULT_SL_PERCENTILE,
    tp_percentile: float = DEFAULT_TP_PERCENTILE,
) -> SlTpCalibrationReport:
    """Turn raw resolved ``signal_outcomes`` rows (see
    :func:`fetch_resolved_outcomes_with_excursion`) into a per-label
    recommended ``(sl_atr_multiplier, tp_atr_multiplier)`` pair.

    Each row's own originally-logged ``stop_loss``/``take_profit`` already
    encode that trade's ATR distance at the live default multiplier (the
    live system always logs with
    :data:`agents_crypto.risk_calc.DEFAULT_SL_ATR_MULTIPLIER`/
    :data:`agents_crypto.risk_calc.DEFAULT_TP_ATR_MULTIPLIER`), so dividing
    the row's actual excursion by that row's own logged distance and
    multiplying back by the default multiplier converts "how far did price
    move, in percent" into "how many ATR-multiples deep was that", in the
    exact same units ``backtest_symbol(sl_tp_overrides=...)`` consumes --
    without needing the row's raw ATR value at all (which isn't stored on
    ``signal_outcomes``).
    """
    from agents_crypto.risk_calc import DEFAULT_SL_ATR_MULTIPLIER, DEFAULT_TP_ATR_MULTIPLIER

    by_label: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_label.setdefault(row["label"], []).append(row)

    report = SlTpCalibrationReport()
    for label, label_rows in sorted(by_label.items()):
        direction = LABEL_DIRECTION.get(label)
        if direction is None:
            report.skipped[label] = "undirected label -- no SL/TP side to calibrate"
            continue

        mae_atr_units: list[float] = []
        mfe_atr_units: list[float] = []
        for row in label_rows:
            entry_price = row["entry_price"]
            sl_distance_pct = abs(entry_price - row["stop_loss"]) / entry_price * 100.0
            tp_distance_pct = abs(row["take_profit"] - entry_price) / entry_price * 100.0
            if sl_distance_pct > 0:
                mae_atr_units.append(abs(row["mae_pct"]) / sl_distance_pct * DEFAULT_SL_ATR_MULTIPLIER)
            if tp_distance_pct > 0:
                mfe_atr_units.append(max(row["mfe_pct"], 0.0) / tp_distance_pct * DEFAULT_TP_ATR_MULTIPLIER)

        n = len(label_rows)
        if n < min_n or not mae_atr_units or not mfe_atr_units:
            report.skipped[label] = f"only {n} resolved sample(s), need >= {min_n} with usable excursion data"
            continue

        sl_mult = min(max(_percentile(mae_atr_units, sl_percentile), MIN_ATR_MULTIPLIER), MAX_ATR_MULTIPLIER)
        tp_mult = min(max(_percentile(mfe_atr_units, tp_percentile), MIN_ATR_MULTIPLIER), MAX_ATR_MULTIPLIER)
        report.calibrations[label] = LabelSlTpCalibration(
            label=label,
            direction=direction,
            n=n,
            sl_atr_multiplier=sl_mult,
            tp_atr_multiplier=tp_mult,
            median_mae_atr_units=_median(mae_atr_units),
            median_mfe_atr_units=_median(mfe_atr_units),
            sl_percentile=sl_percentile,
            tp_percentile=tp_percentile,
        )
    return report


def run_sl_tp_calibration(
    storage_path: str,
    *,
    min_n: int = DEFAULT_MIN_N,
    sl_percentile: float = DEFAULT_SL_PERCENTILE,
    tp_percentile: float = DEFAULT_TP_PERCENTILE,
) -> SlTpCalibrationReport:
    """Connect to the scanner database, pull every resolved tracked signal
    with usable excursion data, and compute the calibration report."""
    conn = connect_for_reads(storage_path)
    try:
        rows = fetch_resolved_outcomes_with_excursion(conn)
    finally:
        conn.close()
    return compute_label_sl_tp_calibration(rows, min_n=min_n, sl_percentile=sl_percentile, tp_percentile=tp_percentile)


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage-path", default=None, help="Defaults to ScannerConfig()'s storage_path.")
    parser.add_argument("--min-n", type=int, default=DEFAULT_MIN_N)
    parser.add_argument("--sl-percentile", type=float, default=DEFAULT_SL_PERCENTILE)
    parser.add_argument("--tp-percentile", type=float, default=DEFAULT_TP_PERCENTILE)
    parser.add_argument("--out", default=None, help="Optional path to write the full report as JSON.")
    args = parser.parse_args()

    storage_path = args.storage_path or ScannerConfig().storage_path
    report = run_sl_tp_calibration(
        storage_path, min_n=args.min_n, sl_percentile=args.sl_percentile, tp_percentile=args.tp_percentile
    )

    print(f"Resolved-outcome SL/TP calibration ({storage_path}):\n")
    if report.calibrations:
        print(f"{'label':<28}{'dir':<7}{'n':>6}{'sl_mult':>10}{'tp_mult':>10}{'R:R':>8}")
        for label, c in sorted(report.calibrations.items()):
            rr = c.tp_atr_multiplier / c.sl_atr_multiplier
            print(f"{label:<28}{c.direction:<7}{c.n:>6}{c.sl_atr_multiplier:>10.2f}{c.tp_atr_multiplier:>10.2f}{rr:>8.2f}")
    else:
        print("(no label had enough resolved samples to calibrate)")
    if report.skipped:
        print("\nSkipped:")
        for label, reason in sorted(report.skipped.items()):
            print(f"  {label}: {reason}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report.as_dict(), fh, indent=2)
        print(f"\nWrote full report to {args.out}")


if __name__ == "__main__":
    _main()
