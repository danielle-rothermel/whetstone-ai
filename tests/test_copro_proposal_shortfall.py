"""COPRO tolerates a stochastic proposal shortfall and records it.

A COPRO round asks the proposer for ``breadth`` (or ``breadth - 1``)
instructions. Some of those drafts do not become usable candidates: a
provider call fails in the infrastructure, a returned draft violates the
proposal contract, or a draft duplicates a template already proposed.
Requiring every requested draft to land turns one bad draft out of many
into a whole-run terminal failure, which is a perfection requirement on
stochastic infrastructure.

These tests pin the tolerance and its measurement: a round that realizes at
least one usable proposal continues on what it has, records requested versus
realized counts, and only a malformed protocol interaction -- a transport
returning more than the round paid for -- stays
``copro_proposal_cardinality``.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from whetstone.coordination.runtime_bootstrap import copro_run_request
from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
from whetstone.optim.contracts import (
    OPTIM_RESULT_SCHEMA,
    OptimResult,
    OutputContract,
    StepStatus,
)
from whetstone.optim.copro.adapter import CoproAttempt, CoproConfig, CoproDriver
from whetstone.optim.proposal.proposer import ProposalDraft
from whetstone.testing.fakes.proposer import DummyProposerTransport
from whetstone.testing.runtime import (
    build_toy_copro_control,
    prepare_toy_copro_run,
    register_toy_runtime,
)

# A template the toy proposal contract rejects, so the draft comes back from
# the transport intact and is dropped at ``validate_instruction`` instead --
# the "model answered, answer unusable" case, distinct from an infra failure.
REJECTED_TEMPLATE = "  "

GOOD_BODIES = (
    "Reply briefly to: {prompt} with a concise greeting.",
    "Answer {prompt} in one short friendly sentence.",
    "Respond to {prompt} in a single clear sentence.",
    "Give {prompt} one short helpful answer.",
    "Answer {prompt} plainly in one sentence.",
)


def _runtime_with_bodies(store, bodies, *, breadth, depth=1):
    """A toy COPRO runtime whose proposer is scripted to ``bodies``.

    ``FakeProposerTransport`` fills every requested slot: a slot with no
    scripted body becomes a ``failed=True`` draft (the infra-failure case)
    and an empty-string body becomes a billed failed draft. Both are dropped
    downstream, which is exactly the shortfall under test.
    """
    engine = ReferenceEvalRuntimeConfig().build_engine(store)
    control = build_toy_copro_control(
        breadth=breadth, depth=depth, engine=engine
    )
    runtime = register_toy_runtime(
        store=store,
        engine=engine,
        copro_control=control,
        proposal_bodies=bodies,
    )
    return runtime, control


def _drive(runtime, control):
    launch = prepare_toy_copro_run(
        runtime,
        run_id=f"shortfall-{uuid4().hex[:8]}",
        control=control,
        terminal_top_k=1,
    )
    result_ref = runtime.controller.drive(
        copro_run_request(
            launch,
            controller_identity_hash=runtime.controller.runtime_hash,
        )
    )
    assert result_ref.schema_name == OPTIM_RESULT_SCHEMA
    return OptimResult.model_validate(runtime.store.get(result_ref.reference))


def _shortfall(runtime, step_result):
    """The shortfall record a step persisted in its state snapshot."""
    state_ref = step_result.record.state_ref
    assert state_ref is not None
    return runtime.store.get(state_ref.reference)["proposal_shortfall"]


def _proposal_step(result):
    """The first proposal round of a driven run."""
    return result.step_results[0]


# --- a dropped draft no longer kills the run ------------------------------


def test_a_validation_rejected_draft_proceeds_with_one_fewer_proposal(
    sqlite_store,
) -> None:
    """breadth 6 asks for 5 drafts; one is rejected and 4 continue.

    Fails before the shortfall tolerance: the round realized 5 occurrences
    against a requested breadth of 6 and terminalized the whole run with
    ``copro_proposal_cardinality``.
    """
    bodies = (GOOD_BODIES[0], REJECTED_TEMPLATE) + GOOD_BODIES[1:4]
    runtime, control = _runtime_with_bodies(sqlite_store, bodies, breadth=6)

    result = _drive(runtime, control)

    assert result.terminal_failure is None
    step = _proposal_step(result)
    assert step.record.status is StepStatus.CONTINUE
    shortfall = _shortfall(runtime, step)
    assert shortfall["requested_occurrences"] == 6
    assert shortfall["realized_occurrences"] == 5
    assert shortfall["dropped_occurrences"] == 1
    # The rejection is attributed, not silently absorbed.
    assert [entry["disposition"] for entry in shortfall["rejected"]] == [
        "rejected"
    ]


def test_an_infra_failed_proposer_call_proceeds_with_one_fewer_proposal(
    sqlite_store,
) -> None:
    """A transport slot that made no call is dropped, not fatal.

    Scripting fewer bodies than the round requests makes the fake transport
    fill the last slot with a failed draft carrying no call identity, which
    is the infrastructure-failure shape.
    """
    runtime, control = _runtime_with_bodies(
        sqlite_store, GOOD_BODIES[:4], breadth=6
    )

    result = _drive(runtime, control)

    assert result.terminal_failure is None
    step = _proposal_step(result)
    assert step.record.status is StepStatus.CONTINUE
    shortfall = _shortfall(runtime, step)
    assert shortfall["requested_proposal_count"] == 5
    assert shortfall["requested_occurrences"] == 6
    assert shortfall["realized_occurrences"] == 5
    assert [entry["disposition"] for entry in shortfall["rejected"]] == [
        "provider_failed"
    ]


def test_a_full_round_records_no_dropped_occurrences(sqlite_store) -> None:
    """The shortfall record is present and zero when nothing was dropped."""
    runtime, control = _runtime_with_bodies(
        sqlite_store, GOOD_BODIES, breadth=6
    )

    result = _drive(runtime, control)

    shortfall = _shortfall(runtime, _proposal_step(result))
    assert shortfall["requested_occurrences"] == 6
    assert shortfall["realized_occurrences"] == 6
    assert shortfall["dropped_occurrences"] == 0
    assert shortfall["rejected"] == []


def test_the_requested_breadth_is_unchanged_by_a_shortfall(
    sqlite_store,
) -> None:
    """Pre-registered breadth is design; realized count is measurement."""
    bodies = (GOOD_BODIES[0], REJECTED_TEMPLATE) + GOOD_BODIES[1:4]
    runtime, control = _runtime_with_bodies(sqlite_store, bodies, breadth=6)

    result = _drive(runtime, control)

    contract = _proposal_step(result).record.request.record.step_output_contract
    assert contract.returned_proposal_count == control.breadth - 1
    assert contract.min_returned_proposal_count == 1


# --- a round that realizes nothing terminalizes honestly ------------------


class _SeedOnlyTransport(DummyProposerTransport):
    """Answers the seed round, then fails every later round's slots.

    Models a proposer that stops producing usable instructions once the run
    has measured history -- the round realizes nothing new, which is a
    legitimate optimizer outcome rather than a broken contract.
    """

    def draft(self, config, request, count):
        if request.proposal_mode == "seed_proposal":
            return super().draft(config, request, count)
        return tuple(
            ProposalDraft.failure(detail="no instruction available")
            for _ in range(count)
        )


def test_a_zero_valid_round_after_measured_history_retains_best_so_far(
    sqlite_store,
) -> None:
    """A history round realizes nothing, so the run keeps its best-so-far.

    The seed round measures a candidate; the history round's drafts all
    fail. The run terminalizes COMPLETE on what it already measured instead
    of dying as a contract violation, and records the empty round.
    """
    engine = ReferenceEvalRuntimeConfig().build_engine(sqlite_store)
    control = build_toy_copro_control(breadth=2, depth=2, engine=engine)
    runtime = register_toy_runtime(
        store=sqlite_store,
        engine=engine,
        copro_control=control,
        proposer_transport=_SeedOnlyTransport(
            scripted_bodies=GOOD_BODIES[:1],
            execution_policy_hash=engine.execution_policy_identity_hash(),
            prompt_adapter_identity_hash=(
                control.prompt_adapter_identity_hash
            ),
        ),
    )

    result = _drive(runtime, control)

    assert result.terminal_failure is None
    assert len(result.proposals) == 1
    final = result.step_results[-1]
    assert final.record.status is StepStatus.COMPLETE
    shortfall = _shortfall(runtime, final)
    assert shortfall["realized_occurrences"] == 0
    assert shortfall["dropped_occurrences"] == control.breadth


def test_a_zero_valid_seed_round_fails_without_measured_history(
    sqlite_store,
) -> None:
    """Nothing proposed and nothing measured: a genuine terminal failure.

    Distinguished from the cardinality violation by its own code -- there is
    no protocol breach here, the search simply has nothing to report.
    """
    runtime, control = _runtime_with_bodies(
        sqlite_store, (REJECTED_TEMPLATE,), breadth=2
    )

    result = _drive(runtime, control)

    assert result.terminal_failure is not None
    assert result.terminal_failure.code == "copro_proposal_round_empty"


# --- malformed protocol still violates the cardinality contract -----------


class _OverfillingTransport(DummyProposerTransport):
    """A transport that returns more drafts than the round paid for."""

    def draft(self, config, request, count):
        drafts = super().draft(config, request, count)
        extra = ProposalDraft(
            template="An unrequested extra instruction for {prompt}.",
            request_evidence={"logical_call_id": "overfill"},
            response_evidence={"draft_index": count},
            usage={"proposer_calls": 1},
            cost=0.0,
        )
        return (*drafts, extra)


def test_a_transport_returning_extra_drafts_still_violates_cardinality(
    sqlite_store,
) -> None:
    """Overfill is a malformed protocol interaction, not a stochastic one."""
    engine = ReferenceEvalRuntimeConfig().build_engine(sqlite_store)
    control = build_toy_copro_control(breadth=3, depth=1, engine=engine)
    runtime = register_toy_runtime(
        store=sqlite_store,
        engine=engine,
        copro_control=control,
        proposer_transport=_OverfillingTransport(
            scripted_bodies=GOOD_BODIES[:2],
            execution_policy_hash=engine.execution_policy_identity_hash(),
            prompt_adapter_identity_hash=(
                control.prompt_adapter_identity_hash
            ),
        ),
    )

    result = _drive(runtime, control)

    assert result.terminal_failure is not None
    assert result.terminal_failure.code == "copro_proposal_cardinality"


# --- the driver folds a short round as one complete round -----------------


def test_fold_round_rejects_a_round_exceeding_breadth() -> None:
    """More occurrences than breadth remains a contract violation."""
    driver = CoproDriver(CoproConfig(breadth=2, depth=2))
    with pytest.raises(ValueError, match="exceeded breadth"):
        driver.fold_round(
            driver.initial_state(
                _driver_candidate(), mutation_field="user_prompt_template"
            ),
            _measured_round(count=3, round_index=0),
        )


def test_fold_round_rejects_an_empty_round() -> None:
    """A round with no measured occurrence cannot be folded."""
    driver = CoproDriver(CoproConfig(breadth=2, depth=2))
    with pytest.raises(ValueError, match="at least one measured occurrence"):
        driver.fold_round(
            driver.initial_state(
                _driver_candidate(), mutation_field="user_prompt_template"
            ),
            (),
        )


def _driver_candidate():
    from whetstone.testing.toy.experiment import build_toy_experiment

    return build_toy_experiment(num_seeds=1).initial_candidate


def _measured_round(*, count: int, round_index: int):
    """``count`` distinct measured occurrences for one round.

    Built with ``model_construct`` because these exercise ``fold_round``'s
    cardinality guard alone, which reads only the occurrence count.
    """
    return tuple(
        CoproAttempt.model_construct(
            occurrence_ordinal=index,
            round_index=round_index,
            run_id="r",
            step_index=round_index,
            intent_id=f"i{index}",
            reward=0.5,
        )
        for index in range(count)
    )


def test_output_contract_rejects_a_floor_above_the_requested_count() -> None:
    """A floor cannot exceed what the round asked for."""
    with pytest.raises(ValueError, match="cannot exceed"):
        OutputContract(
            returned_proposal_count=2,
            min_returned_proposal_count=3,
        )


def test_output_contract_rejects_a_zero_floor() -> None:
    """A continuing round must realize something to continue on."""
    with pytest.raises(ValueError, match="at least 1"):
        OutputContract(
            returned_proposal_count=2,
            min_returned_proposal_count=0,
        )


def test_a_shortfall_floor_does_not_relax_terminal_cardinality() -> None:
    """Terminal selection stays exact; the floor is continuing-only."""
    contract = OutputContract(
        returned_proposal_count=4,
        min_returned_proposal_count=1,
    )
    assert contract.admits_accepted_count(StepStatus.CONTINUE, 1)
    assert contract.admits_accepted_count(StepStatus.CONTINUE, 4)
    assert not contract.admits_accepted_count(StepStatus.CONTINUE, 0)
    assert not contract.admits_accepted_count(StepStatus.CONTINUE, 5)
    # COMPLETE has no floor: it must hit the terminal count exactly.
    assert contract.admits_accepted_count(StepStatus.COMPLETE, 4)
    assert not contract.admits_accepted_count(StepStatus.COMPLETE, 1)


def test_an_exact_contract_admits_only_the_requested_count() -> None:
    """Without a floor the contract binds exactly, as before."""
    contract = OutputContract(returned_proposal_count=3)
    assert contract.admits_accepted_count(StepStatus.CONTINUE, 3)
    assert not contract.admits_accepted_count(StepStatus.CONTINUE, 2)


