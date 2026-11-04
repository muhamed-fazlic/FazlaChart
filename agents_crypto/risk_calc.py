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

# Extended third target: for timeframes whose moves tend to run further and
# for longer (currently 4h -- see signal_tracking.MULTI_TARGET_TIMEFRAMES),
# a trade is managed with three partial targets instead of one single exit.
# TP1 = take_profit_conservative (the existing 0.75x target), TP2 =
# take_profit (the existing full 1.0x target), TP3 = this extra "let it run"
# target for genuine trend continuation. Multiplies the *full* tp_distance,
# same reference point as CONSERVATIVE_TP_FRACTION above.
EXTENDED_TP_FRACTION = 1.6

# Position-size weights for the three-target management scheme above:
# front-loaded on purpose -- bank the easy/likely target first to eliminate
# downside risk on most of the size, then let progressively smaller
# fractions ride for the less-likely-but-bigger continuation targets. Must
# sum to 1.0 (checked by tests, not enforced at runtime since these are
# fixed module constants, not user input).
TP1_WEIGHT = 0.40
TP2_WEIGHT = 0.35
TP3_WEIGHT = 0.25

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
    # "Let it run" third target for the 3-target management scheme (see
    # EXTENDED_TP_FRACTION) -- only meaningful for timeframes that actually
    # use it; harmless to compute and ignore otherwise.
    take_profit_extended: float | None = None
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


@dataclass
class KellySizing:
    """Fractional-Kelly position-size suggestion derived from a *calibrated*
    (backtested, not guessed) win rate and average win/loss size.

    Deliberately conservative in three ways: (1) the caller is expected to
    pass the win rate's 95%-CI *lower bound*, never the raw point estimate,
    so a lucky small sample can't inflate the suggested size; (2) the result
    is scaled by ``fraction_multiplier`` (half-Kelly by default — full Kelly
    is mathematically optimal only for a perfectly known edge, and real
    edges are estimated with error, so full Kelly tends to oversize and
    produce brutal drawdowns on real, noisy data); (3) hard-capped at
    ``KELLY_MAX_FRACTION`` regardless of what the formula says, because no
    single calibration label — however good it looks — should ever justify
    betting a huge chunk of capital on one trade.
    """

    win_rate_used_pct: float
    full_kelly_fraction: float
    recommended_fraction: float
    fraction_multiplier: float
    warning: str | None = None


# Half-Kelly: the standard practitioner compromise between Kelly's optimal
# long-run growth rate and its extreme sensitivity to estimation error.
DEFAULT_KELLY_FRACTION_MULTIPLIER = 0.5
# Never suggest betting more than this fraction of capital on one signal,
# no matter how good the formula says the edge is.
KELLY_MAX_FRACTION = 0.25
# Below this win/loss size, the "edge" is an artifact of rounding/near-zero
# denominators rather than a real average outcome — treat it as unknown.
_MIN_MEANINGFUL_RETURN_PCT = 1e-6


def compute_kelly_fraction(
    win_rate_pct: float,
    avg_win_return_pct: float | None,
    avg_loss_return_pct: float | None,
    fraction_multiplier: float = DEFAULT_KELLY_FRACTION_MULTIPLIER,
) -> KellySizing:
    """Fractional-Kelly suggested bet size (as a fraction of capital, 0-1).

    ``win_rate_pct`` should be the conservative (95%-CI lower-bound) win
    rate from calibration, not the raw point estimate — see class docstring.
    ``avg_win_return_pct`` must be positive, ``avg_loss_return_pct`` must be
    negative (as stored in ``CalibrationStats``); either missing/zero means
    there isn't enough resolved history to size off of yet.

    Classic Kelly for a binary bet: f* = p - q/b, where p = win probability,
    q = 1-p, and b = avg_win / avg_loss (in absolute terms) — the "odds"
    received on a win relative to what's risked on a loss.
    """
    if (
        avg_win_return_pct is None
        or avg_loss_return_pct is None
        or avg_win_return_pct <= _MIN_MEANINGFUL_RETURN_PCT
        or avg_loss_return_pct >= -_MIN_MEANINGFUL_RETURN_PCT
    ):
        return KellySizing(
            win_rate_used_pct=win_rate_pct,
            full_kelly_fraction=0.0,
            recommended_fraction=0.0,
            fraction_multiplier=fraction_multiplier,
            warning="Nedovoljno podataka o prosječnom dobitku/gubitku za Kelly proračun — veličina pozicije se ne predlaže.",
        )

    p = win_rate_pct / 100.0
    q = 1.0 - p
    b = avg_win_return_pct / abs(avg_loss_return_pct)
    full_kelly = p - q / b

    if full_kelly <= 0:
        return KellySizing(
            win_rate_used_pct=win_rate_pct,
            full_kelly_fraction=round(full_kelly, 4),
            recommended_fraction=0.0,
            fraction_multiplier=fraction_multiplier,
            warning=(
                "Kelly formula daje nultu ili negativnu vrijednost čak i uz konzervativnu (donja CI granica) "
                "procjenu win rate-a — matematički nema prednosti (edge) po trenutnoj kalibraciji. Preskoči ovaj "
                "signal ili trguj minimalnom veličinom."
            ),
        )

    recommended = min(full_kelly * fraction_multiplier, KELLY_MAX_FRACTION)
    warning = None
    if full_kelly * fraction_multiplier > KELLY_MAX_FRACTION:
        warning = (
            f"Kelly formula predlaže veći udio kapitala ({full_kelly * fraction_multiplier * 100:.1f}%) — "
            f"ograničeno na maksimalnih {KELLY_MAX_FRACTION * 100:.0f}% po jednom signalu radi sigurnosti."
        )

    return KellySizing(
        win_rate_used_pct=win_rate_pct,
        full_kelly_fraction=round(full_kelly, 4),
        recommended_fraction=round(recommended, 4),
        fraction_multiplier=fraction_multiplier,
        warning=warning,
    )


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
    take_profit_extended = round(close + (take_profit - close) * EXTENDED_TP_FRACTION, 8)
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
        take_profit_extended=take_profit_extended,
        extension_warning=extension_warning,
        patient_entry=patient_entry,
    )
