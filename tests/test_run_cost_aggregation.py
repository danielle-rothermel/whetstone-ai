"""Aggregating run spend from observed usage.

The rule under test is that ``usd`` reports a complete total or nothing at
all. dr-providers surfaces a price only when the provider returns one, so a
sum over the priced subset would understate real spend while presenting
itself as the run total.
"""

from __future__ import annotations

import pytest

from whetstone.optim.cost import (
    ProposerCallUsage,
    RoleCost,
    RunCostReport,
    UsageObservation,
    aggregate_role_cost,
)


def test_no_calls_reports_zeros_and_no_price() -> None:
    """A role that made no call has no price to report, not a zero price."""
    cost = aggregate_role_cost(())
    assert cost.calls == 0
    assert cost.input_tokens == 0
    assert cost.output_tokens == 0
    assert cost.priced_calls == 0
    assert cost.unpriced_calls == 0
    assert cost.usd is None


def test_fully_priced_calls_total_tokens_and_usd() -> None:
    cost = aggregate_role_cost(
        (
            UsageObservation(input_tokens=10, output_tokens=4, usd=0.25),
            UsageObservation(input_tokens=6, output_tokens=2, usd=0.75),
        )
    )
    assert cost.calls == 2
    assert cost.input_tokens == 16
    assert cost.output_tokens == 6
    assert cost.priced_calls == 2
    assert cost.unpriced_calls == 0
    assert cost.usd == pytest.approx(1.0)


def test_fully_unpriced_calls_report_tokens_without_usd() -> None:
    """Tokens are still authoritative when no provider reported a price."""
    cost = aggregate_role_cost(
        (
            UsageObservation(input_tokens=10, output_tokens=4),
            UsageObservation(input_tokens=6, output_tokens=2),
        )
    )
    assert cost.calls == 2
    assert cost.input_tokens == 16
    assert cost.output_tokens == 6
    assert cost.priced_calls == 0
    assert cost.unpriced_calls == 2
    assert cost.usd is None


def test_mixed_priced_and_unpriced_calls_withhold_usd() -> None:
    """A partial sum is never presented as the total."""
    cost = aggregate_role_cost(
        (
            UsageObservation(input_tokens=10, output_tokens=4, usd=0.25),
            UsageObservation(input_tokens=6, output_tokens=2),
        )
    )
    assert cost.calls == 2
    assert cost.input_tokens == 16
    assert cost.output_tokens == 6
    # The split stays visible so a reader can see what a price would cover.
    assert cost.priced_calls == 1
    assert cost.unpriced_calls == 1
    assert cost.usd is None


def test_a_zero_price_is_a_price_not_a_missing_one() -> None:
    """A genuinely free call must not suppress the run total."""
    cost = aggregate_role_cost(
        (
            UsageObservation(input_tokens=5, output_tokens=1, usd=0.0),
            UsageObservation(input_tokens=5, output_tokens=1, usd=0.5),
        )
    )
    assert cost.priced_calls == 2
    assert cost.unpriced_calls == 0
    assert cost.usd == pytest.approx(0.5)


def test_role_cost_rejects_a_usd_total_that_omits_unpriced_calls() -> None:
    """The contract itself refuses a partial total, not just the aggregator."""
    with pytest.raises(ValueError, match="usd must be absent"):
        RoleCost(
            calls=2,
            priced_calls=1,
            unpriced_calls=1,
            usd=0.25,
        )


def test_role_cost_requires_call_counts_to_reconcile() -> None:
    with pytest.raises(ValueError, match="must sum to calls"):
        RoleCost(calls=3, priced_calls=1, unpriced_calls=1)


def test_proposer_call_usage_projects_onto_an_observation() -> None:
    observation = ProposerCallUsage(
        prompt_tokens=7,
        completion_tokens=3,
        usd=0.125,
    ).observation()
    assert observation == UsageObservation(
        input_tokens=7,
        output_tokens=3,
        usd=0.125,
    )


@pytest.mark.parametrize(
    ("prompt_tokens", "completion_tokens"),
    [(11, None), (None, 11)],
)
def test_either_absent_token_direction_is_an_incomplete_breakdown(
    prompt_tokens: int | None, completion_tokens: int | None
) -> None:
    """One known direction is not a token breakdown.

    The absent side is carried into the totals as zero, so a call reporting
    only one direction publishes a token total that understates the call.
    That has to increment ``rows_missing_token_breakdown`` or the
    understatement is invisible.
    """
    observation = ProposerCallUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        usd=0.5,
    ).observation()

    assert observation.missing_token_breakdown is True
    assert aggregate_role_cost((observation,)).rows_missing_token_breakdown == 1


def test_both_token_directions_present_is_a_complete_breakdown() -> None:
    observation = ProposerCallUsage(prompt_tokens=11, completion_tokens=4).observation()

    assert observation.missing_token_breakdown is False


def test_report_defaults_to_empty_roles() -> None:
    report = RunCostReport()
    assert report.task_model == RoleCost()
    assert report.proposer == RoleCost()
