"""Unit tests for the Derivatives/Risk agent's assembly logic — snapshot,
calibration, and risk-level fetching are mocked; the LLM is a fake object
so these never touch a real Ollama server or the network."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agents_crypto.derivatives_risk_agent import DerivativesRiskAssessment, assess_symbol
from scanner.calibration import CalibrationStats

pytestmark = pytest.mark.unit

_FAKE_SNAPSHOT = {
    "symbol": "INJUSDT",
    "timeframes": {
        "1h": {
            "trend": "bearish",
            "close": 3.42,
            "atr": 0.08,
            "rsi": 43.5,
            "anomalies": ["failed_breakout"],
        }
    },
    "derivatives": {"funding_rate": -0.0002, "open_interest_change_pct": 5.8},
}

_FAKE_BTC_CONTEXT = {
    "symbol": "BTCUSDT",
    "timeframes": {"1h": {"trend": "bullish"}},
    "reference_trend": "bullish",
}

_FAKE_CALIBRATION = [
    CalibrationStats(
        label="trend:bearish",
        direction="short",
        horizon_candles=12,
        n=387,
        win_rate_pct=37.0,
        avg_net_return_pct=-0.173,
        avg_win_return_pct=0.637,
        avg_loss_return_pct=-0.647,
    ),
]


class _FakeStructuredLLM:
    """Stand-in for `chat_model.with_structured_output(Schema)` — records the
    prompt it was called with and returns a canned, schema-valid assessment."""

    def __init__(self):
        self.last_prompt = None

    def invoke(self, prompt):
        self.last_prompt = prompt
        return DerivativesRiskAssessment(
            symbol="INJUSDT",
            bias="short",
            historical_basis="trend:bearish, n=387, win_rate=37.0%",
            key_observations=["RSI 43.5", "Failed breakout anomaly active"],
            risk_factors=["BTC trend is bullish — divergence risk", "Calibration win rate below 50%"],
            risk_reward_commentary="Given R:R, the win rate does not clear breakeven.",
            caveats="Does not account for news or orderbook depth.",
        )


class _FakeChatModel:
    def __init__(self):
        self.structured = _FakeStructuredLLM()

    def with_structured_output(self, schema):
        assert schema is DerivativesRiskAssessment
        return self.structured


def test_assess_symbol_returns_grounded_assessment():
    fake_llm = _FakeChatModel()
    with (
        patch("agents_crypto.derivatives_risk_agent.build_symbol_snapshot", return_value=_FAKE_SNAPSHOT),
        patch("agents_crypto.derivatives_risk_agent.build_btc_context", return_value=_FAKE_BTC_CONTEXT),
        patch("agents_crypto.derivatives_risk_agent.get_calibration_stats", return_value=_FAKE_CALIBRATION),
    ):
        result = assess_symbol("INJUSDT", timeframe="1h", horizon_candles=12, llm=fake_llm)

    assert isinstance(result, DerivativesRiskAssessment)
    assert result.bias == "short"
    assert "37.0" in result.historical_basis

    prompt = fake_llm.structured.last_prompt
    assert "INJUSDT" in prompt
    assert "win_rate_pct" in prompt or "win_rate" in prompt
    assert "DIVERGES FROM BTC" in prompt
    assert "True" in prompt  # bearish alt vs bullish BTC => diverges


def test_assess_symbol_states_no_calibration_when_empty():
    fake_llm = _FakeChatModel()
    with (
        patch("agents_crypto.derivatives_risk_agent.build_symbol_snapshot", return_value=_FAKE_SNAPSHOT),
        patch("agents_crypto.derivatives_risk_agent.build_btc_context", return_value=_FAKE_BTC_CONTEXT),
        patch("agents_crypto.derivatives_risk_agent.get_calibration_stats", return_value=[]),
    ):
        assess_symbol("INJUSDT", timeframe="1h", horizon_candles=12, llm=fake_llm)

    prompt = fake_llm.structured.last_prompt
    assert "NONE — no historical calibration available" in prompt


def test_assess_symbol_includes_risk_levels_for_directional_trend():
    fake_llm = _FakeChatModel()
    with (
        patch("agents_crypto.derivatives_risk_agent.build_symbol_snapshot", return_value=_FAKE_SNAPSHOT),
        patch("agents_crypto.derivatives_risk_agent.build_btc_context", return_value=_FAKE_BTC_CONTEXT),
        patch("agents_crypto.derivatives_risk_agent.get_calibration_stats", return_value=_FAKE_CALIBRATION),
    ):
        assess_symbol("INJUSDT", timeframe="1h", horizon_candles=12, llm=fake_llm)

    prompt = fake_llm.structured.last_prompt
    assert "stop_loss" in prompt
    assert "take_profit" in prompt


def test_assess_symbol_omits_risk_levels_for_neutral_trend():
    neutral_snapshot = {
        **_FAKE_SNAPSHOT,
        "timeframes": {"1h": {**_FAKE_SNAPSHOT["timeframes"]["1h"], "trend": "neutral", "anomalies": []}},
    }
    fake_llm = _FakeChatModel()
    with (
        patch("agents_crypto.derivatives_risk_agent.build_symbol_snapshot", return_value=neutral_snapshot),
        patch("agents_crypto.derivatives_risk_agent.build_btc_context", return_value=_FAKE_BTC_CONTEXT),
        patch("agents_crypto.derivatives_risk_agent.get_calibration_stats", return_value=[]),
    ):
        assess_symbol("INJUSDT", timeframe="1h", horizon_candles=12, llm=fake_llm)

    prompt = fake_llm.structured.last_prompt
    assert "N/A — trend is neutral" in prompt
