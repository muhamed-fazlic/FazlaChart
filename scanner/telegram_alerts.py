"""Telegram push alerts for high-quality Top Signal Now candidates.

Lets a user keep a phone's Telegram app as their only "is it worth looking
at the dashboard right now" check instead of manually refreshing it. Fires
once per newly-qualifying signal (de-duplication is handled upstream by
:func:`scanner.signal_tracking.log_new_signals`, which only logs — and thus
only alerts on — a symbol+direction the very first time it crosses the Top
Signal Now bar), not every scan cycle it stays open.

A Telegram outage, missing credentials, or malformed response must never
take down the scanner cycle: every function here is best-effort and either
returns ``False``/logs a warning instead of raising.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import requests

logger = logging.getLogger(__name__)

TELEGRAM_API_URL = "https://api.telegram.org/bot{token}/sendMessage"


@dataclass(frozen=True)
class TelegramAlertConfig:
    """Alerting is opt-in: leaving ``bot_token``/``chat_id`` unset (the
    default) disables it entirely, so a plain ``ScannerConfig()`` never
    tries to reach Telegram. The quality thresholds mirror the project's
    broader "never overstate the edge" philosophy — alerts require a higher
    bar than the dashboard's own Top Signal Now panel, since a push
    notification demands attention in a way a quiet dashboard row doesn't.
    """

    bot_token: str | None = None
    chat_id: str | None = None
    min_win_rate_ci_low_pct: float = 60.0
    min_n: int = 50
    require_btc_alignment: bool = False

    @property
    def enabled(self) -> bool:
        return bool(self.bot_token and self.chat_id)


def should_alert(candidate: dict[str, Any], config: TelegramAlertConfig) -> bool:
    """Whether ``candidate`` (one entry from
    :func:`agents_crypto.best_signal.compute_candidates`'s ``signals`` list)
    clears the configured alert threshold."""
    ci_low = candidate.get("win_rate_ci_low_pct")
    n = candidate.get("n")
    if ci_low is None or n is None:
        return False
    if ci_low < config.min_win_rate_ci_low_pct:
        return False
    if n < config.min_n:
        return False
    if config.require_btc_alignment and candidate.get("diverges_from_btc"):
        return False
    return True


def format_alert_message(candidate: dict[str, Any]) -> str:
    """Plain-text Telegram message for one qualifying candidate — same
    numbers the dashboard's Top Signal Now panel would show, just pushed
    instead of requiring a manual check."""
    direction_word = "LONG 🟢" if candidate["direction"] == "long" else "SHORT 🔴"
    lines = [
        f"🎯 Novi potvrđen signal: {candidate['symbol']} {direction_word} ({candidate['timeframe']})",
        f"Win rate: {candidate['win_rate_pct']:.1f}% [CI-low {candidate['win_rate_ci_low_pct']:.1f}%] (n={candidate['n']})",
        f"Label: {candidate['label']}",
    ]
    entry = candidate.get("entry")
    if entry:
        lines.append(
            f"Entry: {entry['entry_zone_low']:.6g}–{entry['entry_zone_high']:.6g} | "
            f"SL: {entry['stop_loss']:.6g} | TP: {entry['take_profit']:.6g} | R:R {entry['risk_reward_ratio']:.2f}"
        )
    kelly = candidate.get("kelly_sizing")
    if kelly and kelly.get("recommended_fraction_pct"):
        lines.append(f"💰 Kelly: {kelly['recommended_fraction_pct']:.1f}% kapitala")
    correlation = candidate.get("correlation")
    if correlation:
        lines.append(f"🔗 {correlation['note']}")
    return "\n".join(lines)


def send_telegram_message(bot_token: str, chat_id: str, text: str, timeout: float = 10.0) -> bool:
    """Best-effort push to a Telegram chat. Returns ``True`` on an HTTP 200
    response, ``False`` on anything else (including a request exception) —
    never raises, so a Telegram outage can't take the scanner down with it.
    """
    try:
        resp = requests.post(
            TELEGRAM_API_URL.format(token=bot_token),
            json={"chat_id": chat_id, "text": text},
            timeout=timeout,
        )
        if resp.status_code != 200:
            logger.warning("Telegram alert failed (HTTP %s): %s", resp.status_code, resp.text[:200])
            return False
        return True
    except Exception:
        logger.exception("Telegram alert failed to send")
        return False


def maybe_send_alert(candidate: dict[str, Any], config: TelegramAlertConfig) -> bool:
    """Sends an alert for ``candidate`` only when alerting is enabled and it
    clears the configured threshold. Returns whether a message was actually
    sent, for logging/testing purposes."""
    if not config.enabled or not should_alert(candidate, config):
        return False
    return send_telegram_message(config.bot_token, config.chat_id, format_alert_message(candidate))
