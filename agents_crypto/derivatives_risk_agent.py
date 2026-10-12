"""Derivatives/Risk agent: interprets the scanner's objective facts plus the
calibration backtest's historical win-rate, and produces a structured
assessment. It never invents a probability — if no calibration history
exists for the active label, the schema forces it to say so explicitly.

Usage:
    python -m agents_crypto --symbol INJUSDT --timeframe 1h
"""

from __future__ import annotations

import os
from typing import Any, Literal

from pydantic import BaseModel, Field

from agents_crypto.calibration_lookup import find_stats_for_labels, get_calibration_stats
from agents_crypto.risk_calc import compute_risk_levels
from scanner.btc_context import build_btc_context, diverges_from_btc
from tradingagents.dataflows.vendors.binance.snapshot import build_symbol_snapshot
from tradingagents.llm_clients.factory import create_llm_client

DEFAULT_PROVIDER = os.environ.get("CRYPTO_AGENT_LLM_PROVIDER", "ollama")
DEFAULT_MODEL = os.environ.get("CRYPTO_AGENT_LLM_MODEL", "llama3.1:8b")


class DerivativesRiskAssessment(BaseModel):
    """Structured output the LLM must fill in — every field is grounded in
    the input facts it was given; nothing here is a free-floating prediction."""

    symbol: str
    bias: Literal["long", "short", "neutral"] = Field(
        description="Directional lean implied by the CURRENT trend/anomaly labels given, not a guess."
    )
    historical_basis: str = Field(
        description="Which calibration stat(s) this bias is grounded in (label, n, win rate), "
        "or 'no historical calibration data for the active label(s)' if none were provided."
    )
    key_observations: list[str] = Field(
        description="2-5 short bullet facts drawn directly from the provided snapshot/derivatives data."
    )
    risk_factors: list[str] = Field(
        description="2-5 short bullet risks: e.g. BTC context conflict, low n in calibration, high funding, thin volume."
    )
    risk_reward_commentary: str = Field(
        description="Commentary on the given ATR-based stop-loss/take-profit levels — do not invent new levels."
    )
    caveats: str = Field(
        description="What this assessment does NOT account for (e.g. news, orderbook depth, broader macro)."
    )


def _build_llm(provider: str = DEFAULT_PROVIDER, model: str = DEFAULT_MODEL, base_url: str | None = None):
    client = create_llm_client(provider=provider, model=model, base_url=base_url)
    return client.get_llm()


def _active_labels(timeframe_snapshot: dict[str, Any]) -> list[str]:
    labels = [f"trend:{timeframe_snapshot['trend']}"]
    labels += [f"anomaly:{a}" for a in timeframe_snapshot.get("anomalies", [])]
    return labels


def _build_prompt(
    symbol: str,
    timeframe: str,
    snapshot: dict[str, Any],
    btc_context: dict[str, Any],
    calibration_matches: list[dict[str, Any]],
    risk_levels: dict[str, Any] | None,
    diverges: bool,
) -> str:
    return f"""You are a derivatives/risk analyst for crypto futures. You are given ALREADY-MEASURED,
objective facts below — you do not have tools, and you must not invent numbers that are not present
in this input. Your job is strictly to INTERPRET what is given, grounding any directional bias and
confidence in the historical calibration data provided (not your own intuition about what "should"
happen).

Rules:
- If `calibration_matches` is empty, you MUST say in `historical_basis` that there is no historical
  calibration data for the active label(s) — do not substitute a confidence number of your own.
- `bias` must follow the CURRENT trend/anomaly labels given, not a contrarian guess.
- Do not propose your own stop-loss/take-profit; only comment on the ones given in `risk_levels`.
- Treat a calibration sample size (`n`) under ~30 as too small to be reliable, and say so.

SYMBOL: {symbol}
TIMEFRAME: {timeframe}

SNAPSHOT (objective facts from the Market Scanner):
{snapshot}

BTC REFERENCE CONTEXT:
{btc_context}
DIVERGES FROM BTC (opposite trend to BTC's 1h reference): {diverges}

CALIBRATION BACKTEST MATCHES (historical win-rate/avg-return for the labels active right now,
computed from real Binance history, net of trading costs — this is ground truth, not a guess):
{calibration_matches if calibration_matches else "NONE — no historical calibration available for the active label(s)."}

DETERMINISTIC RISK LEVELS (ATR-based, already calculated — do not invent different numbers):
{risk_levels if risk_levels else "N/A — trend is neutral, no directional bias to size a trade for."}

Produce your structured assessment now."""


def assess_symbol(
    symbol: str,
    timeframe: str = "1h",
    horizon_candles: int = 12,
    provider: str = DEFAULT_PROVIDER,
    model: str = DEFAULT_MODEL,
    base_url: str | None = None,
    llm: Any = None,
) -> DerivativesRiskAssessment:
    """Fetch live scanner facts + cached calibration stats for ``symbol``,
    compute deterministic risk levels, and ask the LLM to interpret them
    into a structured, grounded assessment."""
    snapshot = build_symbol_snapshot(symbol, timeframes=(timeframe,))
    tf_snapshot = snapshot["timeframes"][timeframe]
    btc_context = build_btc_context()

    labels = _active_labels(tf_snapshot)
    calibration_stats = get_calibration_stats(symbol, timeframe, horizons=(horizon_candles,))
    matches = find_stats_for_labels(calibration_stats, labels, horizon_candles)
    matches_json = [m.__dict__ for m in matches]

    risk_levels = None
    bias_hint = tf_snapshot["trend"]
    if bias_hint in ("bullish", "bearish") and tf_snapshot.get("atr"):
        levels = compute_risk_levels(
            close=tf_snapshot["close"],
            atr=tf_snapshot["atr"],
            bias="long" if bias_hint == "bullish" else "short",
        )
        risk_levels = levels.__dict__

    prompt = _build_prompt(
        symbol,
        timeframe,
        snapshot,
        btc_context,
        matches_json,
        risk_levels,
        diverges=diverges_from_btc(tf_snapshot["trend"], btc_context["reference_trend"]),
    )

    chat_model = llm or _build_llm(provider, model, base_url)
    structured_llm = chat_model.with_structured_output(DerivativesRiskAssessment)
    result = structured_llm.invoke(prompt)
    if isinstance(result, DerivativesRiskAssessment):
        return result
    return DerivativesRiskAssessment.model_validate(result)
