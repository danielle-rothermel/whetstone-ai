from __future__ import annotations

from typing import cast

import pytest

from tests.optimization.miprov2.support import (
    make_minimal_miprov2_runtime,
    resolve_miprov2_eval_config_binding,
)
from tests.optimization.support import make_harness, make_store, registry
from whetstone.core.effects.authority import EffectAuthority
from whetstone.core.effects.models import ReplayPolicy
from whetstone.experiment.candidate import candidate_reference
from whetstone.optimization.contracts import (
    BudgetState,
    IntentOutcome,
    IntentResolution,
    OptimizationRun,
    OutputContract,
    ResolutionClass,
    ResolutionDetail,
    StepMode,
    StepStatus,
    optimization_run_reference,
)
from whetstone.optimization.miprov2.adapter import (
    MIPROV2_INTENT_REJECTED_CODE,
    MIPROV2_PROPOSAL_FAILED_CODE,
    MIPROV2_STATE_KEY,
    Miprov2Adapter,
)
from whetstone.optimization.miprov2.eval_config import (
    Miprov2EvalConfigBinding,
    Miprov2EvalConfigBindingRequest,
    Miprov2EvalConfigResolver,
)
from whetstone.optimization.miprov2.evidence import Miprov2EvidenceResolver
from whetstone.optimization.miprov2.proposal import Miprov2ProposalResponse
from whetstone.optimization.miprov2.runtime import Miprov2State
from whetstone.optimization.proposal.proposer import (
    DurableProposalExecutor,
    FakeProposerTransport,
    ProposalExecutorDurabilityContract,
    _durable_proposal_executor,
)

PROPOSER_GAVE_UP = "scripted proposer produced an empty draft"
REJECTION_MESSAGE = "intent target Eval Config is not the engine's binding"


class _StaticEvalConfigResolver:
    def resolve(
        self,
        request: Miprov2EvalConfigBindingRequest,
    ) -> Miprov2EvalConfigBinding:
        return resolve_miprov2_eval_config_binding(request)


def _pass_through_executor() -> DurableProposalExecutor:
    return _durable_proposal_executor(
        durability_contract=ProposalExecutorDurabilityContract(
            recovery_policy=ReplayPolicy.DURABLE_WORKFLOW,
            policy_identity_hash="c" * 64,
        ),
        execute=lambda *, config, request, transport, count: transport.draft(
            config, request, count
        ),
    )


def _adapter_case(tmp_path, *, templates: tuple[str, ...]):

    driver, state = make_minimal_miprov2_runtime()
    store = make_store(tmp_path)
    transport = FakeProposerTransport(
        {},
        default=templates,
        execution_policy_hash=state.control.provider_execution_policy_hash,
        prompt_adapter_identity_hash=(
            state.control.prompt_adapter_identity_hash
        ),
    )
    adapter = Miprov2Adapter(
        store=store,
        proposer_config=state.control.prompt_model,
        transport=transport,
        eval_config_resolver=cast(
            Miprov2EvalConfigResolver,
            _StaticEvalConfigResolver(),
        ),
        proposal_executor=_pass_through_executor(),
        driver=driver,
    )
    run = optimization_run_reference(
        OptimizationRun(
            run_id=state.run_id,
            optimizer_config=state.control.reference(),
            adapter_key=adapter.key,
            mode=StepMode.PROPOSAL_ONLY,
            terminal_output_contract=OutputContract(returned_proposal_count=1),
            template_render_contract=state.control.template_render_contract,
            reward_policy=state.control.reward_policy,
        )
    )
    budget = BudgetState(
        remaining={
            "bootstrap_rollouts": 0,
            "proposal_calls": 2,
            "evaluations": 2,
            "task_rows": 6,
        }
    )
    request = adapter.build_step_request(
        run=run,
        step_index=0,
        initial_state=state,
        initial_budget=budget,
    )
    harness = make_harness(
        store=store,
        adapter_registry=registry(adapter),
        run=run,
        effect_authority=EffectAuthority.memory(),
        adapter_replay_policy=ReplayPolicy.DURABLE_WORKFLOW,
    )
    return harness, request, adapter, store, driver, state


def _folded_state(store, result) -> Miprov2State:
    assert result.state_ref is not None
    snapshot = store.get(result.state_ref.reference)
    return Miprov2State.model_validate(snapshot[MIPROV2_STATE_KEY])


def test_failed_proposal_persists_a_terminal_failed_step(tmp_path) -> None:
    harness, request, _adapter, store, driver, _state = _adapter_case(
        tmp_path, templates=("",)
    )

    result, _ref = harness.run_step(request)

    assert result.status is StepStatus.FAILED
    assert result.terminal_failure is not None
    assert result.terminal_failure.code == MIPROV2_PROPOSAL_FAILED_CODE
    assert PROPOSER_GAVE_UP in result.terminal_failure.message
    assert result.accepted_candidates == ()
    assert result.resolved_intents == ()
    assert result.budget_delta.consumed["proposal_calls"] == 1

    folded = _folded_state(store, result)
    assert folded.phase == "failed"
    assert folded.failure is not None
    assert PROPOSER_GAVE_UP in folded.failure
    assert folded.proposal_state is not None
    assert folded.proposal_state.stage == "failed"
    with pytest.raises(ValueError, match=PROPOSER_GAVE_UP):
        driver.plan(folded)


def test_successful_proposal_step_is_not_terminalized(tmp_path) -> None:
    harness, request, _adapter, store, _driver, _state = _adapter_case(
        tmp_path, templates=("Instruction: durable {query}.",)
    )

    result, _ref = harness.run_step(request)

    assert result.status is StepStatus.CONTINUE
    assert result.terminal_failure is None
    folded = _folded_state(store, result)
    assert folded.phase == "proposal"
    assert folded.failure is None


def _proposal_response(request) -> Miprov2ProposalResponse:
    return Miprov2ProposalResponse(
        request_identity_hash=request.identity_hash,
        text="Instruction: improved {query}.",
        evidence={"ordinal": request.effect_ordinal},
    )


def _state_pending_baseline_intent(driver, state) -> Miprov2State:

    transition_budget = state.budget.proposal_calls + 2
    trace: list[tuple[str, str, str | None]] = []
    for _transition in range(transition_budget):
        plan = driver.plan(state)
        proposal_stage = (
            plan.state.proposal_state.stage
            if plan.state.proposal_state is not None
            else None
        )
        trace.append((plan.kind, plan.state.phase, proposal_stage))
        if plan.kind == "eval_config_binding":
            assert plan.eval_config_binding is not None
            state = driver.fold_eval_config_binding(
                plan.state,
                resolve_miprov2_eval_config_binding(plan.eval_config_binding),
            )
            continue
        if plan.kind == "proposal_model":
            assert plan.proposal_request is not None
            state = driver.fold_proposal(
                plan.state,
                _proposal_response(plan.proposal_request),
            )
            continue
        assert plan.kind == "baseline_evaluation"
        return plan.state
    raise AssertionError(
        "MIPROv2 did not reach its baseline intent within the proposal-call "
        f"budget plus binding and terminal transitions; trace={trace!r}; "
        f"state={state.model_dump_json()}"
    )


def _request_for(request, state: Miprov2State):

    consumed = {
        label: state.effect_counts[label]
        for label in ("bootstrap_rollouts", "proposal_calls", "evaluations")
    }
    consumed["task_rows"] = state.effect_counts["task_rows"]
    remaining = {
        label: getattr(state.budget, label) - count
        for label, count in consumed.items()
        if label != "task_rows"
    }
    remaining["task_rows"] = (
        request.budget.remaining["task_rows"] - consumed["task_rows"]
    )
    return request.model_copy(
        update={
            "pools": {MIPROV2_STATE_KEY: state.model_dump(mode="json")},
            "budget": BudgetState(consumed=consumed, remaining=remaining),
        }
    )


def _rejected(intent) -> IntentResolution:
    return IntentResolution(
        schema_version=2,
        intent=intent,
        outcome=IntentOutcome.REJECTED,
        detail=ResolutionDetail(
            classification=ResolutionClass.VALIDATION,
            message=REJECTION_MESSAGE,
        ),
        resolved_eval_config=intent.target_eval_config,
    )


def _issue_baseline_intent(adapter, driver, request, state):

    pending = _state_pending_baseline_intent(driver, state)
    output = adapter.invoke(_request_for(request, pending), ())
    assert len(output.evaluation_intents) == 1
    issued = Miprov2State.model_validate(output.state_delta[MIPROV2_STATE_KEY])
    return output.evaluation_intents[0], issued


def test_baseline_intent_uses_exact_request_candidate_without_proposal(
    tmp_path,
) -> None:
    _harness, request, adapter, _store, driver, state = _adapter_case(
        tmp_path,
        templates=("Instruction: durable {query}.",),
    )
    pending = _state_pending_baseline_intent(driver, state)
    exact_request = _request_for(request, pending)

    output = adapter.invoke(exact_request, ())

    assert output.proposed_candidates == ()
    assert len(output.evaluation_intents) == 1
    intent = output.evaluation_intents[0]
    assert intent.purpose == "miprov2_baseline"
    assert intent.candidate == state.control.base_candidate
    assert intent.candidate in tuple(
        candidate_reference(item) for item in exact_request.candidates
    )


def test_rejected_resolution_folds_to_a_terminal_failed_step(
    tmp_path,
) -> None:
    _harness, request, adapter, _store, driver, state = _adapter_case(
        tmp_path, templates=("Instruction: durable {query}.",)
    )
    intent, issued = _issue_baseline_intent(adapter, driver, request, state)

    folded = adapter.fold_resolution(issued, _rejected(intent))

    assert folded.phase == "failed"
    assert folded.failure is not None
    assert intent.intent_id in folded.failure
    assert "rejected before execution" in folded.failure
    assert "validation" in folded.failure
    assert REJECTION_MESSAGE in folded.failure
    assert folded.study_transcript == issued.study_transcript
    assert folded.completed_effects == issued.completed_effects
    assert folded.effect_counts == issued.effect_counts
    assert issued.pending_evaluation is not None
    assert folded.pending_evaluation is None


def test_rejected_resolution_leaves_the_evidence_resolver_untouched(
    tmp_path,
) -> None:
    _harness, request, adapter, store, driver, state = _adapter_case(
        tmp_path, templates=("Instruction: durable {query}.",)
    )
    intent, issued = _issue_baseline_intent(adapter, driver, request, state)
    rejection = _rejected(intent)

    with pytest.raises(ValueError, match="failed resolution"):
        Miprov2EvidenceResolver(store).resolve_evaluation_failure(rejection)

    assert adapter.fold_resolution(issued, rejection).phase == "failed"


def test_a_terminal_state_invokes_as_a_failed_step(tmp_path) -> None:
    _harness, request, adapter, _store, driver, state = _adapter_case(
        tmp_path, templates=("Instruction: durable {query}.",)
    )
    intent, issued = _issue_baseline_intent(adapter, driver, request, state)
    folded = adapter.fold_resolution(issued, _rejected(intent))

    output = adapter.invoke(_request_for(request, folded), ())

    assert output.proposed_status is StepStatus.FAILED
    assert output.terminal_failure is not None
    assert output.terminal_failure.code == MIPROV2_INTENT_REJECTED_CODE
    assert intent.intent_id in output.terminal_failure.message
    assert output.accepted_candidates == ()
    assert output.evaluation_intents == ()
    assert (
        Miprov2State.model_validate(
            output.state_delta[MIPROV2_STATE_KEY]
        ).phase
        == "failed"
    )
