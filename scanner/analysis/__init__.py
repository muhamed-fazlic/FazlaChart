"""Offline analysis tools that mine data the live system already collects
(live-tracked ``signal_outcomes``, backtest reports, ...) to recommend
strategy corrections -- as opposed to :mod:`scanner.backtesting` and
:mod:`scanner.validation`, which measure whether a strategy variant is
good, this package answers "given what actually happened, how should the
strategy change"."""

from __future__ import annotations
