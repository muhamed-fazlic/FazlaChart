"""Unit tests for Phase 8 Korak 1's candidate archive — persisting every
candidate that cleared the significance gate, selected or not. SQLite only,
no network calls."""

from __future__ import annotations

import json

import pytest

from scanner.candidate_archive import record_candidates
from scanner.storage import fetch_candidate_archive, init_db, prune_old_candidate_archive

pytestmark = pytest.mark.unit


def _candidate(
    symbol: str,
    timeframe: str,
    direction: str,
    rank: int,
    selected: bool,
    muted: bool = False,
    rejection_reason: str | None = None,
    scanned_at: str = "2026-01-01T00:00:00+00:00",
) -> dict:
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "direction": direction,
        "label": "trend:bearish",
        "pattern": "trending",
        "n": 200,
        "win_rate_pct": 65.0,
        "win_rate_ci_low_pct": 60.0,
        "avg_net_return_pct": 0.8,
        "regime": "normal",
        "regime_fallback": False,
        "quality_score": {"total": 72.5},
        "entry": {"price": 100.0, "stop_loss": 95.0, "take_profit": 110.0, "risk_reward_ratio": 2.0},
        "scanned_at": scanned_at,
        "rank": rank,
        "selected": selected,
        "muted": muted,
        "rejection_reason": rejection_reason,
    }


def _result(candidates_by_tf: dict[str, list[dict]]) -> dict:
    return {"all_eligible_by_timeframe": candidates_by_tf}


def test_record_candidates_persists_both_selected_and_rejected(tmp_path):
    conn = init_db(str(tmp_path / "scanner.db"))
    result = _result(
        {
            "1h": [
                _candidate("AAAUSDT", "1h", "short", rank=1, selected=True),
                _candidate("BBBUSDT", "1h", "short", rank=2, selected=False, rejection_reason="rank_exceeds_limit"),
            ]
        }
    )

    inserted = record_candidates(conn, result)

    assert inserted == 2
    rows = fetch_candidate_archive(conn)
    assert {r["symbol"] for r in rows} == {"AAAUSDT", "BBBUSDT"}
    selected_row = next(r for r in rows if r["symbol"] == "AAAUSDT")
    rejected_row = next(r for r in rows if r["symbol"] == "BBBUSDT")
    assert selected_row["selected"] == 1
    assert selected_row["rejection_reason"] is None
    assert rejected_row["selected"] == 0
    assert rejected_row["rejection_reason"] == "rank_exceeds_limit"
    assert rejected_row["entry_price"] == 100.0
    assert rejected_row["quality_score"] == 72.5
    assert json.loads(json.dumps(rejected_row["candidate"]))["symbol"] == "BBBUSDT"


def test_record_candidates_tags_muted_candidates(tmp_path):
    conn = init_db(str(tmp_path / "scanner.db"))
    result = _result(
        {"1h": [_candidate("AAAUSDT", "1h", "short", rank=1, selected=False, muted=True, rejection_reason="muted_bucket")]}
    )

    record_candidates(conn, result)

    row = fetch_candidate_archive(conn)[0]
    assert row["muted"] == 1
    assert row["rejection_reason"] == "muted_bucket"


def test_record_candidates_with_missing_all_eligible_key_is_a_no_op(tmp_path):
    conn = init_db(str(tmp_path / "scanner.db"))
    assert record_candidates(conn, {}) == 0
    assert fetch_candidate_archive(conn) == []


def test_record_candidates_is_idempotent_for_the_same_cycle(tmp_path):
    """A crash-restart that re-processes the same scan cycle must not
    double-count the same candidate twice."""
    conn = init_db(str(tmp_path / "scanner.db"))
    result = _result({"1h": [_candidate("AAAUSDT", "1h", "short", rank=1, selected=True)]})

    first = record_candidates(conn, result)
    second = record_candidates(conn, result)

    assert first == 1
    assert second == 0
    assert len(fetch_candidate_archive(conn)) == 1


def test_fetch_candidate_archive_filters_by_symbol_and_timeframe(tmp_path):
    conn = init_db(str(tmp_path / "scanner.db"))
    result = _result(
        {
            "1h": [_candidate("AAAUSDT", "1h", "short", rank=1, selected=True)],
            "4h": [_candidate("AAAUSDT", "4h", "long", rank=1, selected=True)],
        }
    )
    record_candidates(conn, result)

    assert len(fetch_candidate_archive(conn, symbol="AAAUSDT")) == 2
    assert len(fetch_candidate_archive(conn, symbol="AAAUSDT", timeframe="1h")) == 1
    assert len(fetch_candidate_archive(conn, symbol="ZZZUSDT")) == 0


def test_prune_old_candidate_archive_deletes_rows_past_retention(tmp_path):
    conn = init_db(str(tmp_path / "scanner.db"))
    old_result = _result(
        {"1h": [_candidate("AAAUSDT", "1h", "short", rank=1, selected=True, scanned_at="2020-01-01T00:00:00+00:00")]}
    )
    recent_result = _result(
        {"1h": [_candidate("BBBUSDT", "1h", "short", rank=1, selected=True, scanned_at="2099-01-01T00:00:00+00:00")]}
    )
    record_candidates(conn, old_result)
    record_candidates(conn, recent_result)

    deleted = prune_old_candidate_archive(conn, retain_days=90)

    assert deleted == 1
    remaining = fetch_candidate_archive(conn)
    assert len(remaining) == 1
    assert remaining[0]["symbol"] == "BBBUSDT"


def test_prune_old_candidate_archive_on_empty_db_returns_zero(tmp_path):
    conn = init_db(str(tmp_path / "empty.db"))
    assert prune_old_candidate_archive(conn, retain_days=90) == 0
