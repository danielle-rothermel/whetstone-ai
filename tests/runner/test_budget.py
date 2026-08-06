"""Budget guard and credits snapshot tests (no network)."""

from __future__ import annotations

import pytest
from dr_serialize import StrictJsonDecodeError

from whetstone.runner.budget import (
    DEFAULT_EXPECTED_CELL_USD,
    RESERVE_USD,
    BudgetGuard,
    ReserveError,
    StopLossError,
    credits_from_payload,
    openrouter_credits_fetcher,
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
    BudgetGuard().check_start(canonical=True, remaining_usd=5.0, is_rerun=True)


def test_reserve_ignores_non_canonical_and_missing_remaining() -> None:
    guard = BudgetGuard()
    guard.check_start(canonical=False, remaining_usd=1.0)
    guard.check_start(canonical=True, remaining_usd=None)


def test_stop_loss_triggers_above_two_x_expected() -> None:
    guard = BudgetGuard()
    guard.check_stop_loss(3.99)
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


@pytest.mark.parametrize(
    "content",
    [
        b'{"data":{},"data":{}}',
        b'{"data":{"total_credits":NaN}}',
        b"\xff",
    ],
)
def test_live_credits_fetcher_rejects_non_strict_json(
    content: bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Response:
        def __init__(self, body: bytes) -> None:
            self.content = body

        def raise_for_status(self) -> None:
            pass

    response = Response(content)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr("httpx.get", lambda *args, **kwargs: response)

    with pytest.raises(StrictJsonDecodeError):
        openrouter_credits_fetcher()()


def test_credits_missing_fields_remaining_none() -> None:
    snap = credits_from_payload({"total_credits": None})
    assert snap.remaining_usd is None


@pytest.mark.parametrize("bad", ["NaN", "Infinity", "-Infinity", "nonsense"])
def test_credits_reject_non_finite_amounts(bad: str) -> None:
    """A non-finite amount reads as unreported, never as a number.

    A parsed ``NaN`` propagates into ``remaining_usd`` and makes every guard
    comparison false, so admitting one would silently disarm the reserve and
    the stop-loss on a paid run.
    """
    snap = credits_from_payload({"total_credits": bad, "total_usage": 1.0})
    assert snap.total_credits is None
    assert snap.remaining_usd is None


def test_reserve_refuses_a_non_finite_balance() -> None:
    """NaN is an unreadable balance, not a balance above the reserve."""
    guard = BudgetGuard()
    with pytest.raises(ReserveError, match="not a finite amount"):
        guard.check_start(canonical=True, remaining_usd=float("nan"))


def test_stop_loss_halts_on_a_non_finite_spend() -> None:
    """``nan > stop_loss`` is False, so the guard must reject it explicitly."""
    guard = BudgetGuard()
    with pytest.raises(StopLossError, match="not a finite amount"):
        guard.check_stop_loss(float("nan"))
    assert guard.would_halt(float("nan"))


def test_would_halt_reads_unknown_spend_as_no_halt() -> None:
    """Unknown spend is not evidence of a halt; a real overspend is."""
    guard = BudgetGuard()
    assert not guard.would_halt(None)
    assert guard.would_halt(4.01)
