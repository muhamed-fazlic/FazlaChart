"""Layer 2's rolling holdout boundary -- the forward-advancing marker of
"calendar time not yet used for any confirmation", kept strictly SEPARATE
from (and never derived from or touching) the project's existing locked
6-week holdout (``research_registry.json``'s ``shared_holdout``, reserved
for a future pre-registered confirmatory hypothesis per the funding-extreme
and cross-sectional-momentum closing notes).

Why a new boundary instead of spending the existing locked holdout: the
existing holdout is a FIXED, already-decided-to-be-reserved resource whose
use is "a separate decision about resources, not an automatic default" (see
``research_registry.json``'s ``multiple_testing_counter.note``). Strategy
Lab instead rides the fact that this is a LIVE market: every day that
passes after this boundary is first initialized produces genuinely fresh
candles nobody (including this feature) has looked at yet. The boundary
starts at "now" (or later, if the shared locked holdout's end is somehow
still in the future) and only ever moves forward, each confirmation
consuming the then-available fresh window and pushing the boundary past
it -- so no byte of data is ever evaluated by more than one confirmation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime


@dataclass(frozen=True)
class RollingHoldoutState:
    """``boundary_timestamp_ms`` is exclusive on the "already consumed or
    reserved" side: any confirmation window must start at or after it."""

    boundary_timestamp_ms: int


def initial_boundary_ms(shared_holdout_end_ms: int, now_ms: int | None = None) -> int:
    """The rolling holdout boundary's starting value, the first time this
    feature ever runs: ``max(shared_holdout_end_ms, now)`` -- i.e. never
    before the existing locked holdout's own end (that data is reserved,
    not fresh), and never in the future relative to "now" (there is no
    fresh data to protect yet if ``now`` hasn't arrived).
    """
    if now_ms is None:
        now_ms = int(datetime.now(UTC).timestamp() * 1000)
    return max(shared_holdout_end_ms, now_ms)


def fresh_window_available_ms(state: RollingHoldoutState, now_ms: int | None = None) -> int:
    """Milliseconds of fresh (never-consumed) calendar time available right
    now -- ``0`` if ``now`` has not yet advanced past the boundary (should
    not happen in practice since the boundary is never set ahead of "now",
    but guarded rather than returning a negative number)."""
    if now_ms is None:
        now_ms = int(datetime.now(UTC).timestamp() * 1000)
    return max(0, now_ms - state.boundary_timestamp_ms)


def advance_boundary(state: RollingHoldoutState, consumed_until_ms: int) -> RollingHoldoutState:
    """Returns a new state with the boundary moved forward to
    ``consumed_until_ms`` -- the end of the fresh window a confirmation run
    just consumed. Raises ``ValueError`` if ``consumed_until_ms`` would move
    the boundary BACKWARD (or leave it unchanged), which would mean some
    span of time is being (re-)marked as consumed when it already was, or a
    caller is trying to "give back" already-spent time -- both indicate a
    bug, not a valid state transition. This is the single piece of logic
    that guarantees a confirmation window can never be reused.
    """
    if consumed_until_ms <= state.boundary_timestamp_ms:
        raise ValueError(
            f"consumed_until_ms ({consumed_until_ms}) must be strictly after the current boundary "
            f"({state.boundary_timestamp_ms}) -- the rolling holdout boundary can only move forward"
        )
    return RollingHoldoutState(boundary_timestamp_ms=consumed_until_ms)
