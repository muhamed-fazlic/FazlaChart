"""Unit tests for the Derivatives/Risk agent's assembly logic — snapshot,
calibration, and risk-level fetching are mocked; the LLM is a fake object
so these never touch a real Ollama server or the network."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agents_crypto.derivatives_risk_agent import (
    DerivativesRiskAssessment,
    _sl_near_wall_warning,
    assess_symbol,
)
from scanner.calibration import CalibrationStats

pytestmark = pytest.mark.unit


_FAKE_WALLS = {
    "bid_walls": [{"side": "bid", "price": 3.40, "notional_usdt": 50000.0}],
    "ask_walls": [{"side": "ask", "price": 3.50, "notional_usdt": 42000.0}],
}


def test_sl_near_wall_warning_fires_when_stop_sits_on_a_real_wall():
    warning = _sl_near_wall_warning(3.405, _FAKE_WALLS, tolerance_pct=0.5)
    assert warning is not None
    assert "3.4" in warning
    assert "bid" in warning


def test_sl_near_wall_warning_is_none_when_stop_is_far_from_any_wall():
    assert _sl_near_wall_warning(3.0, _FAKE_WALLS, tolerance_pct=0.5) is None


def test_sl_near_wall_warning_handles_missing_walls_and_stop():
    assert _sl_near_wall_warning(None, _FAKE_WALLS) is None
    assert _sl_near_wall_warning(3.4, None) is None
    assert _sl_near_wall_warning(3.4, {"bid_walls": [], "ask_walls": []}) is None

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

    assert isinstance(result.assessment, DerivativesRiskAssessment)
    assert result.assessment.bias == "short"
    assert "37.0" in result.assessment.historical_basis
    assert result.diverges_from_btc is True
    assert len(result.calibration_matches) == 1
    assert result.risk_levels is not None

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


class _FlakyStructuredLLM:
    """Fails the first N calls (simulating a local model's unreliable tool
    calling), then succeeds — used to verify the retry logic in assess_symbol."""

    def __init__(self, fail_times, failure_mode="none"):
        self.fail_times = fail_times
        self.failure_mode = failure_mode
        self.call_count = 0
        self.last_prompt = None

    def invoke(self, prompt):
        self.last_prompt = prompt
        self.call_count += 1
        if self.call_count <= self.fail_times:
            if self.failure_mode == "none":
                return None
            raise ValueError("missing required field")
        return DerivativesRiskAssessment(
            symbol="INJUSDT",
            bias="short",
            historical_basis="trend:bearish, n=387, win_rate=37.0%",
            key_observations=["RSI 43.5"],
            risk_factors=["Calibration win rate below 50%"],
            risk_reward_commentary="Commentary.",
            caveats="Caveat.",
        )


class _FlakyChatModel:
    def __init__(self, structured):
        self.structured = structured

    def with_structured_output(self, schema):
        assert schema is DerivativesRiskAssessment
        return self.structured


def test_assess_symbol_retries_when_llm_returns_none():
    fake_llm = _FlakyChatModel(_FlakyStructuredLLM(fail_times=2, failure_mode="none"))
    with (
        patch("agents_crypto.derivatives_risk_agent.build_symbol_snapshot", return_value=_FAKE_SNAPSHOT),
        patch("agents_crypto.derivatives_risk_agent.build_btc_context", return_value=_FAKE_BTC_CONTEXT),
        patch("agents_crypto.derivatives_risk_agent.get_calibration_stats", return_value=_FAKE_CALIBRATION),
    ):
        result = assess_symbol("INJUSDT", timeframe="1h", horizon_candles=12, llm=fake_llm)

    assert result.assessment.bias == "short"
    assert fake_llm.structured.call_count == 3


def test_assess_symbol_retries_when_llm_raises_validation_error():
    fake_llm = _FlakyChatModel(_FlakyStructuredLLM(fail_times=1, failure_mode="raise"))
    with (
        patch("agents_crypto.derivatives_risk_agent.build_symbol_snapshot", return_value=_FAKE_SNAPSHOT),
        patch("agents_crypto.derivatives_risk_agent.build_btc_context", return_value=_FAKE_BTC_CONTEXT),
        patch("agents_crypto.derivatives_risk_agent.get_calibration_stats", return_value=_FAKE_CALIBRATION),
    ):
        result = assess_symbol("INJUSDT", timeframe="1h", horizon_candles=12, llm=fake_llm)

    assert result.assessment.bias == "short"
    assert fake_llm.structured.call_count == 2


def test_assess_symbol_raises_after_exhausting_retries():
    fake_llm = _FlakyChatModel(_FlakyStructuredLLM(fail_times=99, failure_mode="none"))
    with (
        patch("agents_crypto.derivatives_risk_agent.build_symbol_snapshot", return_value=_FAKE_SNAPSHOT),
        patch("agents_crypto.derivatives_risk_agent.build_btc_context", return_value=_FAKE_BTC_CONTEXT),
        patch("agents_crypto.derivatives_risk_agent.get_calibration_stats", return_value=_FAKE_CALIBRATION),
        pytest.raises(ValueError, match="failed to return a valid structured assessment"),
    ):
        assess_symbol("INJUSDT", timeframe="1h", horizon_candles=12, llm=fake_llm)

    assert fake_llm.structured.call_count == 3


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
