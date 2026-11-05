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


def format_drift_mute_message(stats: Any) -> str:
    """Plain-text Telegram notice for a bucket the drift detector just
    muted (see :mod:`scanner.drift_detector`). ``stats`` is a
    ``scanner.drift_detector.BucketDriftStats``."""
    return (
        f"🔇 Bucket privremeno isključen: {stats.timeframe} {stats.direction.upper()}\n"
        f"Realizovan win rate (zadnjih {stats.n_resolved} signala): {stats.realized_win_rate_pct:.1f}%\n"
        f"Obećan win rate (CI-low): {stats.promised_win_rate_ci_low_pct_avg:.1f}%\n"
        f"Razlika: {stats.drift_pp:.1f}pp ispod obećanog -- ovaj bucket se ne preporučuje dok se "
        "live rezultati ne poprave."
    )


def format_drift_unmute_message(stats: Any) -> str:
    """Plain-text Telegram notice for a bucket the drift detector just
    restored after recovering. ``stats`` is a
    ``scanner.drift_detector.BucketDriftStats``."""
    return (
        f"🔊 Bucket vraćen: {stats.timeframe} {stats.direction.upper()}\n"
        f"Realizovan win rate (zadnjih {stats.n_resolved} signala): {stats.realized_win_rate_pct:.1f}%\n"
        f"Obećan win rate (CI-low): {stats.promised_win_rate_ci_low_pct_avg:.1f}%\n"
        "Live rezultati su se oporavili -- bucket je ponovo uključen u preporuke."
    )


def format_strategy_lab_weekly_report(snapshot: dict[str, Any]) -> str:
    """Plain-text Telegram message for Strategy Lab's weekly snapshot (see
    ``scanner.backtesting.strategy_miner``/``run_weekly_snapshot``). Always
    carries the "eksploratorno, nije potvrđeno" disclaimer -- the snapshot
    candidate was selected by looking at Layer 1's own ranking (selection
    bias), so it must never read like a confirmed finding, matching
    :func:`format_alert_message`'s tone of never overstating the edge."""
    ci_low = snapshot.get("net_ci_low_pct")
    ci_high = snapshot.get("net_ci_high_pct")
    ci_text = f" CI [{ci_low:.2f}%, {ci_high:.2f}%]" if ci_low is not None and ci_high is not None else ""
    lines = [
        "🔬 Strategy Lab — sedmični snapshot",
        f"Kandidat: {snapshot['template']} / {snapshot['tercile']} / {snapshot['variant_label']}",
        f"Neto povrat: {snapshot.get('net_observed_pct', 'n/a')}%{ci_text} (n={snapshot.get('n_independent', 'n/a')})",
    ]
    if snapshot.get("win_rate_pct") is not None:
        lines.append(f"Win rate: {snapshot['win_rate_pct']:.1f}%")
    lines.append(
        "⚠️ Eksploratorno, nije potvrđeno — ovaj kandidat je izabran gledanjem u Layer 1 rang-listu "
        "(selection bias), ne pre-registrovana potvrđena hipoteza."
    )
    return "\n".join(lines)


def format_strategy_lab_confirmation_report(confirmation: dict[str, Any]) -> str:
    """Plain-text Telegram message for a Layer 2 confirmation attempt — the
    ONLY place in Strategy Lab allowed to report a genuine pass/fail
    verdict (``edge_survives_lower_bound``)."""
    passed = bool(confirmation.get("edge_survives_lower_bound"))
    verdict_word = "✅ EDGE PREŽIVIO CI DONJU GRANICU" if passed else "❌ Nema edge-a (CI ne prelazi nulu)"
    ci_low = confirmation.get("net_ci_low_pct")
    ci_high = confirmation.get("net_ci_high_pct")
    ci_text = f"[{ci_low:.2f}%, {ci_high:.2f}%]" if ci_low is not None and ci_high is not None else "n/a"
    lines = [
        "🧪 Strategy Lab — Layer 2 potvrda (confirmatory run)",
        f"Kandidat: {confirmation['template']} / {confirmation['tercile']} / {confirmation['variant_label']}",
        f"Svježi prozor: {confirmation.get('fresh_window_start_iso', 'n/a')} → {confirmation.get('fresh_window_end_iso', 'n/a')}",
        f"Neto povrat: {confirmation.get('net_observed_pct', 'n/a')}% CI {ci_text} (n={confirmation.get('n_independent', 'n/a')}, potrebno={confirmation.get('n_required', 'n/a')})",
        verdict_word,
    ]
    if not passed:
        lines.append("Ovo je normalan, neutralan ishod (kao i prethodne tri zatvorene porodice) — nije greška.")
    else:
        lines.append(
            "Kandidat za dalji ljudski pregled (execution layer, forward/paper test) — NIJE automatski "
            "pokretanje stvarnog novca."
        )
    return "\n".join(lines)


def maybe_send_drift_alert(stats: Any, config: TelegramAlertConfig, *, muted: bool) -> bool:
    """Best-effort push of a drift mute/unmute notice. Unlike
    :func:`maybe_send_alert`, this ignores the quality thresholds (
    ``min_win_rate_ci_low_pct``/``min_n``) since a drift event is itself
    always worth notifying about when alerting is enabled at all."""
    if not config.enabled:
        return False
    text = format_drift_mute_message(stats) if muted else format_drift_unmute_message(stats)
    return send_telegram_message(config.bot_token, config.chat_id, text)
