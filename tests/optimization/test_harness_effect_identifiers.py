from __future__ import annotations

import pytest

from whetstone.core.effects.authority import EffectAuthority
from whetstone.core.effects.models import ReplayPolicy
from whetstone.experiment.candidate import candidate_reference
from whetstone.optimization.adapters import AdapterOutput
from whetstone.optimization.contracts import (
    BudgetDelta,
    OptimizationStepResult,
    StepMode,
    StepStatus,
    step_request_reference,
    step_result_reference,
)
from whetstone.optimization.harness import (
    INTENT_EFFECT_KEY_PREFIX,
    INTENT_EFFECT_KEY_SCHEMA_VERSION,
)
from whetstone.optimization.run_store import (
    OPTIMIZATION_RESULT_BINDING_PREFIX,
    STEP_RESULT_BINDING_PREFIX,
)
from whetstone.optimization.tools.contracts import (
    RuntimeToolHandle,
    ToolCall,
)

from .support import (
    CountingProposalAdapter,
    RecordingEvaluationService,
    RecordingToolExecutor,
    make_harness,
    make_intent,
    make_store,
    output_contract,
    proposal_request,
    proposal_run,
    proposed_candidate,
    registry,
    tool_request,
)


class DuplicateIntentAdapter:
    invocations = 0

    @property
    def key(self) -> str:
        return "proposal-test"

    @property
    def mode(self) -> StepMode:
        return StepMode.PROPOSAL_ONLY

    @property
    def required_replay_policy(self) -> ReplayPolicy:
        return ReplayPolicy.IDEMPOTENT

    def invoke(self, request, handles) -> AdapterOutput:
        del handles
        self.invocations += 1
        proposed = proposed_candidate(
            request.candidates[0], "duplicate-intent", text="new"
        )
        intent = make_intent(
            proposed,
            run_id=request.run_id,
            step_index=request.step_index,
        )
        return AdapterOutput(
            proposed_candidates=(proposed,),
            accepted_candidates=(proposed,),
            evaluation_intents=(intent, intent),
            budget_delta=BudgetDelta(consumed={"generations": 2}),
            proposed_status=StepStatus.COMPLETE,
        )


class SampleedCandidateAdapter:
    @property
    def key(self) -> str:
        return "proposal-test"

    @property
    def mode(self) -> StepMode:
        return StepMode.PROPOSAL_ONLY

    @property
    def required_replay_policy(self) -> ReplayPolicy:
        return ReplayPolicy.IDEMPOTENT

    def invoke(self, request, handles) -> AdapterOutput:
        del handles
        proposed = proposed_candidate(
            request.candidates[0], "repeated-candidate", text="new"
        )
        first = make_intent(
            proposed,
            run_id=request.run_id,
            step_index=request.step_index,
        )
        second = first.model_copy(
            update={"intent_id": f"{first.intent_id}-repeat"}
        )
        return AdapterOutput(
            proposed_candidates=(proposed, proposed),
            accepted_candidates=(proposed, proposed),
            evaluation_intents=(first, second),
            budget_delta=BudgetDelta(consumed={"generations": 2}),
            proposed_status=StepStatus.COMPLETE,
        )


class DuplicateRuntimeToolAdapter:
    invocations = 0

    @property
    def key(self) -> str:
        return "tool-test"

    @property
    def mode(self) -> StepMode:
        return StepMode.TOOL_USING

    @property
    def required_replay_policy(self) -> ReplayPolicy:
        return ReplayPolicy.IDEMPOTENT

    def invoke(
        self,
        request,
        handles: tuple[RuntimeToolHandle, ...],
    ) -> AdapterOutput:
        self.invocations += 1
        handle = handles[0]
        call = ToolCall(
            call_id="duplicate-call",
            tool_config=handle.tool_config_ref,
            capacity_binding=handle.binding,
            args={"model_route": "r0", "template": "duplicate"},
        )
        handle(call)
        handle(call)
        raise AssertionError("duplicate Tool Call must not execute twice")


class OmittingToolOutputAdapter:
    invocations = 0

    @property
    def key(self) -> str:
        return "tool-test"

    @property
    def mode(self) -> StepMode:
        return StepMode.TOOL_USING

    @property
    def required_replay_policy(self) -> ReplayPolicy:
        return ReplayPolicy.IDEMPOTENT

    def invoke(
        self,
        request,
        handles: tuple[RuntimeToolHandle, ...],
    ) -> AdapterOutput:
        self.invocations += 1
        handle = handles[0]
        call = ToolCall(
            call_id="forged-duplicate",
            tool_config=handle.tool_config_ref,
            capacity_binding=handle.binding,
            args={"model_route": "r0", "template": "forged"},
        )
        handle(call)
        proposed = proposed_candidate(
            request.candidates[0], "tool-proposal", text="tool"
        )
        return AdapterOutput(
            proposed_candidates=(proposed,),
            accepted_candidates=(proposed,),
            proposed_status=StepStatus.COMPLETE,
        )


def test_intent_effect_and_result_namespaces_are_v2(tmp_path) -> None:
    store = make_store(tmp_path)
    request = proposal_request()
    harness = make_harness(
        store=store,
        adapter_registry=registry(),
        run=request.run,
        evaluation_service=RecordingEvaluationService(store),
    )
    intent = make_intent(
        request.candidates[0],
        run_id=request.run_id,
        step_index=request.step_index,
    )

    effect_request = harness._intent_effect_request(request, intent)

    assert INTENT_EFFECT_KEY_SCHEMA_VERSION == 2
    assert effect_request.semantic_key == (
        f"{INTENT_EFFECT_KEY_PREFIX}"
        "6a73a06854c98d56d6516a8e9d0e96f95ffd771be65508309a78cc6717e842e0"
    )
    assert harness._result_binding_key(request.run_id, 0) == (
        f"{STEP_RESULT_BINDING_PREFIX}{request.run_id}#0"
    )
    assert harness._terminal_binding_key(request.run_id) == (
        f"{OPTIMIZATION_RESULT_BINDING_PREFIX}{request.run_id}"
    )
    assert ":v2:" in STEP_RESULT_BINDING_PREFIX
    assert ":v2:" in OPTIMIZATION_RESULT_BINDING_PREFIX

    store.bind(
        f"whetstone.optimization_step_result:{request.run_id}#0",
        request.run.record_ref.reference,
    )
    store.bind(
        f"whetstone.optimization_result:{request.run_id}",
        request.run.record_ref.reference,
    )
    assert harness.resolve_step_result(request.run_id, 0) is None
    assert harness.resolve_optimization_result(request.run_id) is None


def test_prior_result_must_embed_the_current_exact_run(tmp_path) -> None:
    run_id = "same-logical-run"
    foreign_run = proposal_run(run_id=run_id)
    current_run = proposal_run(
        run_id=run_id,
        contract=output_contract(distinct_bases=True),
    )
    assert foreign_run.record_ref != current_run.record_ref

    foreign_request = proposal_request(run=foreign_run)
    proposed = proposed_candidate(
        foreign_request.candidates[0],
        "foreign-prior",
        text="foreign",
    )
    proposed_ref = candidate_reference(proposed)
    delta = BudgetDelta(consumed={"generations": 1})
    foreign_result = OptimizationStepResult(
        request=step_request_reference(foreign_request),
        proposed_candidates=(proposed_ref,),
        accepted_candidates=(proposed_ref,),
        budget_delta=delta,
        budget=foreign_request.budget.debit(delta),
        status=StepStatus.CONTINUE,
    )
    foreign_result_ref = step_result_reference(foreign_result)

    store = make_store(tmp_path)
    persisted, _ = store.put(
        foreign_result_ref.record_ref.schema_name,
        foreign_result.record_content(),
    )
    assert persisted == foreign_result_ref.record_ref.reference
    store.bind(
        f"{STEP_RESULT_BINDING_PREFIX}{run_id}#0",
        foreign_result_ref.record_ref.reference,
    )

    adapter = CountingProposalAdapter()
    service = RecordingEvaluationService(store)
    harness = make_harness(
        store=store,
        adapter_registry=registry(adapter),
        run=current_run,
        evaluation_service=service,
    )
    request = proposal_request(
        run=current_run,
        step_index=1,
        prior_step_result_ref=foreign_result_ref.record_ref,
        budget=foreign_result.budget,
        contract=current_run.record.terminal_output_contract,
    )

    with pytest.raises(ValueError, match="another exact Optimization Run"):
        harness.run_step(request)

    assert adapter.invocations == 0
    assert service.calls == []
    assert harness.resolve_step_result(run_id, 1) is None


def test_duplicate_intent_ids_are_rejected_before_service_execution(
    tmp_path,
) -> None:
    store = make_store(tmp_path)
    adapter = DuplicateIntentAdapter()
    service = RecordingEvaluationService(store)
    request = proposal_request()
    harness = make_harness(
        store=store,
        adapter_registry=registry(adapter),
        run=request.run,
        evaluation_service=service,
    )

    with pytest.raises(
        ValueError, match="Evaluation Intent IDs must be unique"
    ):
        harness.run_step(request)

    assert adapter.invocations == 1
    assert service.calls == []
    assert harness.resolve_step_result(request.run_id, 0) is None


def test_unique_intents_preserve_repeated_candidate_multisets(
    tmp_path,
) -> None:
    store = make_store(tmp_path)
    service = RecordingEvaluationService(store)
    request = proposal_request(contract=output_contract(2))
    harness = make_harness(
        store=store,
        adapter_registry=registry(SampleedCandidateAdapter()),
        run=request.run,
        evaluation_service=service,
    )

    result, _ = harness.run_step(request)

    assert len(result.proposed_candidates) == 2
    assert result.proposed_candidates[0] == result.proposed_candidates[1]
    assert len(result.accepted_candidates) == 2
    assert len(service.calls) == 2


def test_duplicate_runtime_tool_call_stops_before_second_execution(
    tmp_path,
) -> None:
    store = make_store(tmp_path)
    adapter = DuplicateRuntimeToolAdapter()
    authority = EffectAuthority.memory()
    executor = RecordingToolExecutor(authority)
    request = tool_request()
    harness = make_harness(
        store=store,
        adapter_registry=registry(adapter),
        run=request.run,
        effect_authority=authority,
        tool_executor=executor,
    )

    with pytest.raises(ValueError, match="Tool Call IDs must be unique"):
        harness.run_step(request)

    assert adapter.invocations == 1
    assert [call.call_id for call in executor.calls] == ["duplicate-call"]
    assert harness.resolve_step_result(request.run_id, 0) is None


def test_adapter_cannot_omit_issued_tool_evidence(tmp_path) -> None:
    store = make_store(tmp_path)
    adapter = OmittingToolOutputAdapter()
    authority = EffectAuthority.memory()
    executor = RecordingToolExecutor(authority)
    request = tool_request()
    harness = make_harness(
        store=store,
        adapter_registry=registry(adapter),
        run=request.run,
        effect_authority=authority,
        tool_executor=executor,
    )

    result, _ = harness.run_step(request)

    assert adapter.invocations == 1
    assert [call.call_id for call in executor.calls] == ["forged-duplicate"]
    assert len(result.tool_evidence) == 1
    assert result.tool_evidence[0].result.record.call.record.call_id == (
        "forged-duplicate"
    )
    assert result.budget_delta.consumed == {"tool_calls": 1}
