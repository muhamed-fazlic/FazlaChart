"""Derivatives/Risk agent: interprets the scanner's objective facts plus the
calibration backtest's historical win-rate, and produces a structured
assessment. It never invents a probability — if no calibration history
exists for the active label, the schema forces it to say so explicitly.

Usage:
    python -m agents_crypto --symbol INJUSDT --timeframe 1h
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field

from agents_crypto.calibration_lookup import find_stats_for_labels, get_calibration_stats
from agents_crypto.risk_calc import compute_risk_levels
from scanner.btc_context import build_btc_context, diverges_from_btc
from tradingagents.dataflows.vendors.binance.liquidity_heatmap import compute_liquidity_heatmap
from tradingagents.dataflows.vendors.binance.indicators import klines_to_dataframe
from tradingagents.dataflows.vendors.binance import client as binance_client
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
        default="",
        description="Which calibration stat(s) this bias is grounded in (label, n, win rate), "
        "or 'no historical calibration data for the active label(s)' if none were provided.",
    )
    key_observations: list[str] = Field(
        default_factory=list,
        description="2-5 short bullet facts drawn directly from the provided snapshot/derivatives data.",
    )
    risk_factors: list[str] = Field(
        default_factory=list,
        description="2-5 short bullet risks: e.g. BTC context conflict, low n in calibration, high funding, thin volume.",
    )
    risk_reward_commentary: str = Field(
        default="",
        description="Commentary on the given ATR-based stop-loss/take-profit levels — do not invent new levels.",
    )
    caveats: str = Field(
        default="",
        description="What this assessment does NOT account for (e.g. news, orderbook depth, broader macro).",
    )


@dataclass
class AssessmentResult:
    """Everything a caller (CLI or dashboard) needs to show, not just the
    LLM's interpretation — the deterministic risk levels and calibration
    matches it was grounded in are surfaced too, so a UI can display them
    side by side instead of only trusting the model's prose about them."""

    assessment: DerivativesRiskAssessment
    risk_levels: dict[str, Any] | None
    multi_timeframe_risk_levels: dict[str, dict[str, Any]]
    calibration_matches: list[dict[str, Any]]
    diverges_from_btc: bool
    liquidity_heatmap: dict[str, Any]


# Without an explicit timeout, a stuck/hanging connection (dead keep-alive,
# provider-side stall, ...) blocks the LLM call forever: the dashboard's
# "Analyze with AI" button stays disabled until the page is reloaded, with no
# error ever surfaced. Local CPU-only Ollama can legitimately take minutes, so
# it gets a much longer budget than cloud providers (which should normally
# answer in a few seconds).
_CLOUD_LLM_TIMEOUT_SECONDS = 60.0
_LOCAL_LLM_TIMEOUT_SECONDS = 300.0


def _build_llm(provider: str = DEFAULT_PROVIDER, model: str = DEFAULT_MODEL, base_url: str | None = None):
    timeout = _LOCAL_LLM_TIMEOUT_SECONDS if provider.lower() == "ollama" else _CLOUD_LLM_TIMEOUT_SECONDS
    client = create_llm_client(provider=provider, model=model, base_url=base_url, timeout=timeout)
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
    heatmap_summary: dict[str, Any],
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
- Use `liquidity_heatmap` to judge whether the given stop-loss/take-profit sit inside a crowded
  stop-hunt zone or high-volume node (riskier/more likely to get wicked) and mention that in
  `risk_reward_commentary` if relevant — it is an approximation from public volume, not real
  order-book data, so say so if you rely on it.
- If `risk_levels` includes a `chop_zone_warning`, you MUST mention it in `risk_reward_commentary` —
  it means the given stop-loss sits inside an unusually dense trading zone where price tends to
  whipsaw both directions before breaking out, independent of normal ATR-based sizing.

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

LIQUIDITY HEATMAP (approximate volume-profile proxy for where TP/SL orders likely cluster —
NOT real order-book data, just a structural hint):
{heatmap_summary}

Produce your structured assessment now."""


def _heatmap_prompt_summary(heatmap: dict[str, Any], top_bins: int = 5) -> dict[str, Any]:
    """Condensed view of the heatmap for the LLM prompt — the top-N highest
    volume bins plus POC/value-area/stop-zones, not every bin (keeps the
    prompt short and cheap)."""
    bins_sorted = sorted(heatmap.get("price_bins", []), key=lambda b: b["volume"], reverse=True)[:top_bins]
    return {
        "current_price": heatmap.get("current_price"),
        "point_of_control": heatmap.get("poc_price"),
        "value_area_low": heatmap.get("value_area_low"),
        "value_area_high": heatmap.get("value_area_high"),
        "top_volume_zones": [{"price_low": b["price_low"], "price_high": b["price_high"], "pct_of_total": b["pct_of_total"]} for b in bins_sorted],
        "likely_stop_zones": heatmap.get("likely_stop_zones", []),
        "note": heatmap.get("note"),
    }


def _sl_dense_zone_warning(stop_loss: float | None, heatmap: dict[str, Any]) -> str | None:
    """If the given stop-loss sits inside an unusually crowded volume-profile
    bin (a "chop zone" where price tends to whipsaw both ways before it
    breaks out), flag it — a stop placed in the middle of the heaviest
    traffic is more likely to get wicked than one placed just outside it."""
    bins = heatmap.get("price_bins", [])
    if stop_loss is None or not bins:
        return None
    avg_pct = 100.0 / len(bins)
    dense_threshold = avg_pct * 1.75
    for b in bins:
        if b["price_low"] <= stop_loss <= b["price_high"] and b["pct_of_total"] >= dense_threshold:
            return (
                f"SL ({stop_loss}) je unutar neuobičajeno guste zone trgovanja ({b['pct_of_total']:.1f}% volumena, "
                f"{b['price_low']}-{b['price_high']}) — tu cijena tipično mlati na obje strane prije probijanja. "
                "Razmisli o SL van te zone ili čekanju potvrde zatvaranja svijeće."
            )
    return None


def _risk_levels_by_timeframe(timeframe_snapshots: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Entry/SL/TP for every timeframe. Directional (bullish/bearish)
    timeframes get real ATR-based levels; neutral ones get an explicit
    ``bias: "neutral"`` placeholder instead of being silently dropped, so the
    UI can show "no direction" rather than make the timeframe vanish."""
    levels_by_tf: dict[str, dict[str, Any]] = {}
    for tf, tf_snapshot in timeframe_snapshots.items():
        trend = tf_snapshot.get("trend")
        if trend not in ("bullish", "bearish") or not tf_snapshot.get("atr"):
            levels_by_tf[tf] = {"bias": "neutral"}
            continue
        levels = compute_risk_levels(
            close=tf_snapshot["close"],
            atr=tf_snapshot["atr"],
            bias="long" if trend == "bullish" else "short",
        )
        levels_by_tf[tf] = levels.__dict__
    return levels_by_tf


def assess_symbol(
    symbol: str,
    timeframe: str = "1h",
    horizon_candles: int = 12,
    provider: str = DEFAULT_PROVIDER,
    model: str = DEFAULT_MODEL,
    base_url: str | None = None,
    llm: Any = None,
) -> AssessmentResult:
    """Fetch live scanner facts + cached calibration stats for ``symbol``,
    compute deterministic risk levels, and ask the LLM to interpret them
    into a structured, grounded assessment. Returns the assessment bundled
    with the risk levels/calibration matches it was grounded in."""
    snapshot = build_symbol_snapshot(symbol)  # all default timeframes (5m/15m/1h/4h), for per-timeframe risk levels
    tf_snapshot = snapshot["timeframes"][timeframe]
    btc_context = build_btc_context()

    labels = _active_labels(tf_snapshot)
    calibration_stats = get_calibration_stats(symbol, timeframe, horizons=(horizon_candles,), use_process=True)
    matches = find_stats_for_labels(calibration_stats, labels, horizon_candles)
    matches_json = [m.__dict__ for m in matches]

    multi_timeframe_risk_levels = _risk_levels_by_timeframe(snapshot["timeframes"])

    diverges = diverges_from_btc(tf_snapshot["trend"], btc_context["reference_trend"])

    klines = binance_client.get_klines(symbol, timeframe, limit=200)
    heatmap = compute_liquidity_heatmap(klines_to_dataframe(klines))
    heatmap["symbol"] = symbol
    heatmap["timeframe"] = timeframe
    heatmap_summary = _heatmap_prompt_summary(heatmap)

    # Cross-check every timeframe's stop-loss against this (single, selected-
    # timeframe) heatmap: a dense-volume "chop zone" doesn't move much between
    # timeframes for the same coin, so this is a useful proxy even for the
    # other timeframes' levels, not just the selected one.
    for levels in multi_timeframe_risk_levels.values():
        if levels.get("bias") == "neutral":
            continue
        chop_warning = _sl_dense_zone_warning(levels.get("stop_loss"), heatmap)
        if chop_warning:
            levels["chop_zone_warning"] = chop_warning

    risk_levels = multi_timeframe_risk_levels.get(timeframe)
    if risk_levels and risk_levels.get("bias") == "neutral":
        risk_levels = None  # keep the prompt's existing "N/A — neutral" branch working

    prompt = _build_prompt(
        symbol, timeframe, snapshot, btc_context, matches_json, risk_levels, diverges=diverges, heatmap_summary=heatmap_summary
    )

    chat_model = llm or _build_llm(provider, model, base_url)
    structured_llm = chat_model.with_structured_output(DerivativesRiskAssessment)

    # Smaller/local models occasionally fail to emit a valid tool call (returns
    # None) or emit one missing/malformed fields (raises a validation error) -
    # this is a model-quality quirk, not a code bug, so retry a couple of
    # times before giving up.
    attempts = 3
    assessment: DerivativesRiskAssessment | None = None
    last_error: Exception | None = None
    for _attempt in range(attempts):
        try:
            raw_result = structured_llm.invoke(prompt)
            if raw_result is None:
                raise ValueError("LLM returned no structured output (failed tool call)")
            assessment = (
                raw_result
                if isinstance(raw_result, DerivativesRiskAssessment)
                else DerivativesRiskAssessment.model_validate(raw_result)
            )
            break
        except Exception as exc:  # noqa: BLE001 - retry on any parse/validation failure
            last_error = exc
            continue
    if assessment is None:
        raise ValueError(
            f"LLM ({provider}/{model}) failed to return a valid structured assessment "
            f"after {attempts} attempts: {last_error}. Try a larger/more capable model."
        ) from last_error
    return AssessmentResult(
        assessment=assessment,
        risk_levels=risk_levels,
        multi_timeframe_risk_levels=multi_timeframe_risk_levels,
        calibration_matches=matches_json,
        diverges_from_btc=diverges,
        liquidity_heatmap=heatmap,
    )

