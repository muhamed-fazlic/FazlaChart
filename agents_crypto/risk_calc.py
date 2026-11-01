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

# Take-profit "pull-in" fraction: raw ATR-multiple targets sit exactly at a
# precise tick with nothing behind them, so a limit order resting there can
# get wicked without actually filling (no resting counter-liquidity at that
# exact price) and price snaps back before a full fill confirms. The
# conservative target asks for less of the move, which is reached (and
# actually fillable) far more often in practice than the full stretch target.
CONSERVATIVE_TP_FRACTION = 0.75

# "Already extended" detection: entering a short after RSI is already deep
# in oversold territory (or a long after it's deep in overbought) means
# chasing a move that is statistically more likely to see a relief bounce
# than to keep running — exactly the "entry at the bottom of the cycle,
# then it recovers" failure mode. Flagged via RSI extremity *and* distance
# from EMA20 in ATR units (how far price has already travelled from its
# short-term mean), so a merely-low RSI in a slow grind doesn't trigger it.
OVEREXTENDED_RSI_SHORT = 32.0
OVEREXTENDED_RSI_LONG = 68.0
OVEREXTENDED_EMA_DISTANCE_ATR = 2.0
# How far back toward EMA20 the suggested "patient" entry pulls — a 38.2%
# Fibonacci retracement toward the mean, consistent with the retracement
# levels already shown elsewhere in the app rather than an arbitrary number.
PATIENT_ENTRY_RETRACEMENT = 0.382


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
    # Nearer, more-likely-to-actually-fill target (see CONSERVATIVE_TP_FRACTION).
    take_profit_conservative: float | None = None
    # Set when price looks already extended away from its short-term mean in
    # the trade's direction (see OVEREXTENDED_* constants above).
    extension_warning: str | None = None
    # Suggested retracement entry to wait for instead of chasing `entry`,
    # only populated alongside extension_warning.
    patient_entry: float | None = None


def _extension_check(
    close: float, atr: float, bias: Bias, rsi: float | None, ema20: float | None
) -> tuple[str | None, float | None]:
    if rsi is None or ema20 is None or ema20 <= 0:
        return None, None
    ema_distance_atr = abs(close - ema20) / atr
    if ema_distance_atr < OVEREXTENDED_EMA_DISTANCE_ATR:
        return None, None

    if bias == "short" and rsi <= OVEREXTENDED_RSI_SHORT and close < ema20:
        patient_entry = round(close + (ema20 - close) * PATIENT_ENTRY_RETRACEMENT, 8)
        warning = (
            f"Cijena je već {ema_distance_atr:.1f} ATR ispod EMA20, a RSI ({rsi:.0f}) je duboko prekupljen "
            "(oversold) — ulazak ovdje znači juriti pokret koji je već istegnut i statistički sklon "
            f"privremenom odbijanju (bounce) prije nastavka. Razmisli o čekanju povratka cijene ka ~{patient_entry} "
            "(38.2% retracement ka EMA20) prije ulaska, umjesto ulaska po trenutnoj cijeni."
        )
        return warning, patient_entry
    if bias == "long" and rsi >= OVEREXTENDED_RSI_LONG and close > ema20:
        patient_entry = round(close - (close - ema20) * PATIENT_ENTRY_RETRACEMENT, 8)
        warning = (
            f"Cijena je već {ema_distance_atr:.1f} ATR iznad EMA20, a RSI ({rsi:.0f}) je duboko prekupljen "
            "(overbought) — ulazak ovdje znači juriti pokret koji je već istegnut i statistički sklon "
            f"privremenom povlačenju prije nastavka. Razmisli o čekanju povratka cijene ka ~{patient_entry} "
            "(38.2% retracement ka EMA20) prije ulaska, umjesto ulaska po trenutnoj cijeni."
        )
        return warning, patient_entry
    return None, None


def compute_risk_levels(
    close: float,
    atr: float,
    bias: Bias,
    sl_atr_multiplier: float = DEFAULT_SL_ATR_MULTIPLIER,
    tp_atr_multiplier: float = DEFAULT_TP_ATR_MULTIPLIER,
    rsi: float | None = None,
    ema20: float | None = None,
) -> RiskLevels:
    """ATR-based stop-loss/take-profit for a hypothetical position.

    Long: stop below entry, target above. Short: mirrored. Both multipliers
    must be positive; ``atr``/``close`` must be positive, real market values.

    ``rsi``/``ema20`` are optional; when given, they're used purely to flag
    (never to change the deterministic SL/TP math) when the proposed entry
    looks like it's chasing an already-extended move, and to compute a
    suggested "patient" retracement entry instead.
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

    take_profit_conservative = round(close + (take_profit - close) * CONSERVATIVE_TP_FRACTION, 8)
    extension_warning, patient_entry = _extension_check(close, atr, bias, rsi, ema20)

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
        take_profit_conservative=take_profit_conservative,
        extension_warning=extension_warning,
        patient_entry=patient_entry,
    )
