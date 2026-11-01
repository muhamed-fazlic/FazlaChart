"""Unit tests for scanner/telegram_alerts.py — mocks requests.post so these
never touch the real Telegram API."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from scanner.telegram_alerts import (
    TelegramAlertConfig,
    format_alert_message,
    maybe_send_alert,
    send_telegram_message,
    should_alert,
)

pytestmark = pytest.mark.unit

_CANDIDATE = {
    "symbol": "INJUSDT",
    "timeframe": "1h",
    "direction": "short",
    "trend": "bearish",
    "label": "trend:bearish",
    "n": 200,
    "win_rate_pct": 65.0,
    "win_rate_ci_low_pct": 60.0,
    "win_rate_ci_high_pct": 70.0,
    "avg_net_return_pct": 0.8,
    "diverges_from_btc": False,
    "correlation": None,
    "entry": {
        "price": 10.0,
        "entry_zone_low": 9.9,
        "entry_zone_high": 10.1,
        "stop_loss": 10.6,
        "take_profit": 8.9,
        "take_profit_conservative": 9.3,
        "risk_reward_ratio": 1.67,
        "volatility_warning": None,
        "extension_warning": None,
    },
    "kelly_sizing": {"recommended_fraction_pct": 4.2, "full_kelly_fraction_pct": 8.4, "warning": None},
}


class TestTelegramAlertConfig:
    def test_disabled_without_both_token_and_chat_id(self):
        assert TelegramAlertConfig().enabled is False
        assert TelegramAlertConfig(bot_token="abc").enabled is False
        assert TelegramAlertConfig(chat_id="123").enabled is False

    def test_enabled_with_both(self):
        assert TelegramAlertConfig(bot_token="abc", chat_id="123").enabled is True


class TestShouldAlert:
    def test_passes_a_strong_candidate(self):
        config = TelegramAlertConfig(bot_token="t", chat_id="c", min_win_rate_ci_low_pct=60.0, min_n=50)
        assert should_alert(_CANDIDATE, config) is True

    def test_rejects_below_win_rate_threshold(self):
        config = TelegramAlertConfig(bot_token="t", chat_id="c", min_win_rate_ci_low_pct=70.0)
        assert should_alert(_CANDIDATE, config) is False

    def test_rejects_below_sample_size_threshold(self):
        config = TelegramAlertConfig(bot_token="t", chat_id="c", min_n=500)
        assert should_alert(_CANDIDATE, config) is False

    def test_rejects_btc_divergence_when_alignment_required(self):
        config = TelegramAlertConfig(bot_token="t", chat_id="c", require_btc_alignment=True)
        divergent = {**_CANDIDATE, "diverges_from_btc": True}
        assert should_alert(divergent, config) is False
        assert should_alert(_CANDIDATE, config) is True

    def test_rejects_missing_calibration_fields(self):
        config = TelegramAlertConfig(bot_token="t", chat_id="c")
        assert should_alert({**_CANDIDATE, "win_rate_ci_low_pct": None}, config) is False


class TestFormatAlertMessage:
    def test_includes_core_fields(self):
        text = format_alert_message(_CANDIDATE)
        assert "INJUSDT" in text
        assert "SHORT" in text
        assert "65.0%" in text
        assert "n=200" in text
        assert "Kelly" in text

    def test_omits_entry_block_when_no_entry(self):
        text = format_alert_message({**_CANDIDATE, "entry": None})
        assert "Entry" not in text


class TestSendTelegramMessage:
    def test_returns_true_on_http_200(self):
        with patch("scanner.telegram_alerts.requests.post", return_value=MagicMock(status_code=200)) as mock_post:
            assert send_telegram_message("tok", "chat", "hi") is True
            assert mock_post.called

    def test_returns_false_on_non_200(self):
        with patch(
            "scanner.telegram_alerts.requests.post", return_value=MagicMock(status_code=400, text="bad request")
        ):
            assert send_telegram_message("tok", "chat", "hi") is False

    def test_returns_false_and_swallows_exception_on_network_failure(self):
        with patch("scanner.telegram_alerts.requests.post", side_effect=ConnectionError("no network")):
            assert send_telegram_message("tok", "chat", "hi") is False


class TestMaybeSendAlert:
    def test_does_not_call_requests_when_disabled(self):
        config = TelegramAlertConfig()  # no token/chat_id -> disabled
        with patch("scanner.telegram_alerts.requests.post") as mock_post:
            assert maybe_send_alert(_CANDIDATE, config) is False
            assert not mock_post.called

    def test_does_not_call_requests_when_below_threshold(self):
        config = TelegramAlertConfig(bot_token="t", chat_id="c", min_n=10_000)
        with patch("scanner.telegram_alerts.requests.post") as mock_post:
            assert maybe_send_alert(_CANDIDATE, config) is False
            assert not mock_post.called

    def test_sends_when_enabled_and_qualifying(self):
        config = TelegramAlertConfig(bot_token="t", chat_id="c")
        with patch("scanner.telegram_alerts.requests.post", return_value=MagicMock(status_code=200)) as mock_post:
            assert maybe_send_alert(_CANDIDATE, config) is True
            assert mock_post.called
