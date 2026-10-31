"""Deterministic (no-LLM) stop-loss / take-profit sizing from ATR.

Kept separate from the LLM agent on purpose: entry/exit levels are a
calculation, not an opinion. The agent is given these numbers as ground
truth and may discuss or critique them, but it never invents its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Bias = Literal["long", "short"]

DEFAULT_SL_ATR_MULTIPLIER = 1.5
DEFAULT_TP_ATR_MULTIPLIER = 2.5

# Stop-loss distance (risk_pct) thresholds above which ATR-based sizing stops
# being reliable: in a violently volatile/crashing regime, volatility itself
# is accelerating faster than ATR (a trailing average of recent ranges) can
# track, so a stop placed at a fixed ATR multiple gets clipped by ordinary
# noise as often as it catches a real reversal. This never changes the
# computed levels — it only flags them so the UI/agent can say so instead of
# presenting a precise-looking number as reliable.
EXTREME_VOLATILITY_RISK_PCT = 8.0
ELEVATED_VOLATILITY_RISK_PCT = 4.0


@dataclass
class RiskLevels:
    bias: Bias
    entry: float
    stop_loss: float
    take_profit: float
    risk_pct: float
    reward_pct: float
    risk_reward_ratio: float
    sl_atr_multiplier: float
    tp_atr_multiplier: float
    volatility_warning: str | None = None


def compute_risk_levels(
    close: float,
    atr: float,
    bias: Bias,
    sl_atr_multiplier: float = DEFAULT_SL_ATR_MULTIPLIER,
    tp_atr_multiplier: float = DEFAULT_TP_ATR_MULTIPLIER,
) -> RiskLevels:
    """ATR-based stop-loss/take-profit for a hypothetical position.

    Long: stop below entry, target above. Short: mirrored. Both multipliers
    must be positive; ``atr``/``close`` must be positive, real market values.
    """
    if close <= 0:
        raise ValueError(f"close must be > 0, got {close}")
    if atr <= 0:
        raise ValueError(f"atr must be > 0, got {atr}")
    if sl_atr_multiplier <= 0 or tp_atr_multiplier <= 0:
        raise ValueError("SL/TP ATR multipliers must be > 0")
    if bias not in ("long", "short"):
        raise ValueError(f"bias must be 'long' or 'short', got {bias!r}")

    sl_distance = atr * sl_atr_multiplier
    tp_distance = atr * tp_atr_multiplier

    if bias == "long":
        stop_loss = close - sl_distance
        take_profit = close + tp_distance
    else:
        stop_loss = close + sl_distance
        take_profit = close - tp_distance

    risk_pct = round(sl_distance / close * 100.0, 3)
    reward_pct = round(tp_distance / close * 100.0, 3)
    risk_reward_ratio = round(tp_distance / sl_distance, 3)

    volatility_warning = None
    if risk_pct >= EXTREME_VOLATILITY_RISK_PCT:
        volatility_warning = (
            f"EKSTREMNA volatilnost (SL na {risk_pct:.1f}% od ulaza): ATR-sizing je ovdje nepouzdan — "
            "volatilnost raste brže nego što prosjek zadnjih svijeća stigne uhvatiti, pa se stop može "
            "okidati nasumično i na normalan šum. Razmisli o manjem position size-u ili preskoči."
        )
    elif risk_pct >= ELEVATED_VOLATILITY_RISK_PCT:
        volatility_warning = (
            f"Povišena volatilnost (SL na {risk_pct:.1f}% od ulaza) — stop je širok, provjeri position sizing."
        )

    return RiskLevels(
        bias=bias,
        entry=round(close, 8),
        stop_loss=round(stop_loss, 8),
        take_profit=round(take_profit, 8),
        risk_pct=risk_pct,
        reward_pct=reward_pct,
        risk_reward_ratio=risk_reward_ratio,
        sl_atr_multiplier=sl_atr_multiplier,
        tp_atr_multiplier=tp_atr_multiplier,
        volatility_warning=volatility_warning,
    )
