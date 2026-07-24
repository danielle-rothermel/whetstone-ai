"""Budget guard + credits snapshot tests (no network)."""

from __future__ import annotations

import pytest

from whetstone.runner.budget import (
    DEFAULT_EXPECTED_CELL_USD,
    RESERVE_USD,
    BudgetGuard,
    ReserveError,
    StopLossError,
    credits_from_payload,
)


def test_reserve_and_stop_loss_constants() -> None:
    assert RESERVE_USD == 18.60
    assert DEFAULT_EXPECTED_CELL_USD == 2.0
    assert BudgetGuard().stop_loss_usd == 4.0


def test_reserve_refuses_canonical_below_reserve() -> None:
    guard = BudgetGuard()
    with pytest.raises(ReserveError, match="reserve"):
        guard.check_start(canonical=True, remaining_usd=10.0)


def test_reserve_allows_canonical_above_reserve() -> None:
    BudgetGuard().check_start(canonical=True, remaining_usd=50.0)


def test_reserve_allows_rerun_below_reserve() -> None:
    # A rerun (fixing plumbing) may proceed below the reserve.
    BudgetGuard().check_start(canonical=True, remaining_usd=5.0, is_rerun=True)


def test_reserve_ignores_non_canonical_and_missing_remaining() -> None:
    guard = BudgetGuard()
    guard.check_start(canonical=False, remaining_usd=1.0)
    guard.check_start(canonical=True, remaining_usd=None)


def test_stop_loss_triggers_above_two_x_expected() -> None:
    guard = BudgetGuard()
    guard.check_stop_loss(3.99)  # under 2x $2 -> ok
    assert not guard.would_halt(4.0)
    assert guard.would_halt(4.01)
    with pytest.raises(StopLossError, match="stop-loss"):
        guard.check_stop_loss(4.5)


def test_credits_from_nested_payload() -> None:
    snap = credits_from_payload(
        {"data": {"total_credits": 710.0, "total_usage": 616.97}}
    )
    assert snap.remaining_usd == pytest.approx(93.03)


def test_credits_from_flat_payload() -> None:
    snap = credits_from_payload({"total_credits": 100.0, "total_usage": 40.0})
    assert snap.remaining_usd == pytest.approx(60.0)


def test_credits_missing_fields_remaining_none() -> None:
    snap = credits_from_payload({"total_credits": None})
    assert snap.remaining_usd is None
