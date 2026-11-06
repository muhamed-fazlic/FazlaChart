"""Tests for scanner.strategy_lab_storage -- focused on fetch_leaderboard's
"most recent mining cycle" grouping logic.

Regression coverage for a real production bug: insert_mining_candidate
stamps ``mined_at`` with ``datetime.now(UTC).isoformat()`` individually for
EACH row (microsecond precision), so within one mining cycle (many
candidates inserted in a tight loop, sharing one ``as_of_timestamp_ms``)
every row gets a slightly different ``mined_at``. fetch_leaderboard used to
group by ``MAX(mined_at)``, which only ever matched the single
last-inserted row of the latest cycle -- silently showing just one
candidate in the "Layer 1 top 10" dashboard table instead of every
candidate actually mined that cycle."""

from __future__ import annotations

from scanner import strategy_lab_storage as storage


def _insert(conn, *, as_of_timestamp_ms: int, variant_label: str, composite_score: float) -> int:
    return storage.insert_mining_candidate(
        conn,
        as_of_timestamp_ms=as_of_timestamp_ms,
        template="trend_anomaly",
        tercile="top",
        variant_label=variant_label,
        params={},
        n_independent=50,
        gross_observed_pct=None,
        net_observed_pct=-1.0,
        stability_score=1.0,
        composite_score=composite_score,
        report={},
    )


class TestFetchLeaderboard:
    def test_returns_every_candidate_from_the_latest_cycle(self, tmp_path) -> None:
        conn = storage.init_db(str(tmp_path / "strategy_lab.db"))
        try:
            # Older cycle: must never leak into the leaderboard.
            _insert(conn, as_of_timestamp_ms=1_000, variant_label="old cycle only", composite_score=99.0)

            # Latest cycle: several candidates sharing one as_of_timestamp_ms,
            # but each gets its OWN mined_at (real insert-loop behavior).
            latest_as_of = 2_000
            _insert(conn, as_of_timestamp_ms=latest_as_of, variant_label="Original signal", composite_score=-57.88)
            _insert(conn, as_of_timestamp_ms=latest_as_of, variant_label="Fade (invert) signal", composite_score=-14.25)
            _insert(conn, as_of_timestamp_ms=latest_as_of, variant_label="third variant", composite_score=-30.0)

            leaderboard = storage.fetch_leaderboard(conn, limit=10)
        finally:
            conn.close()

        # All 3 of the latest cycle's candidates must appear -- not just
        # the single last-inserted row.
        assert len(leaderboard) == 3
        assert all(row["variant_label"] != "old cycle only" for row in leaderboard)
        # Ranked by composite_score descending: the best (least negative)
        # candidate must be #1, regardless of insertion order.
        assert leaderboard[0]["variant_label"] == "Fade (invert) signal"
        assert leaderboard[1]["variant_label"] == "third variant"
        assert leaderboard[2]["variant_label"] == "Original signal"

    def test_respects_limit(self, tmp_path) -> None:
        conn = storage.init_db(str(tmp_path / "strategy_lab.db"))
        try:
            for i in range(5):
                _insert(conn, as_of_timestamp_ms=1_000, variant_label=f"v{i}", composite_score=float(i))
            leaderboard = storage.fetch_leaderboard(conn, limit=2)
        finally:
            conn.close()
        assert len(leaderboard) == 2

    def test_empty_db_returns_empty_list(self, tmp_path) -> None:
        conn = storage.init_db(str(tmp_path / "strategy_lab.db"))
        try:
            leaderboard = storage.fetch_leaderboard(conn)
        finally:
            conn.close()
        assert leaderboard == []
