"""Hostile serialized-input regressions for Step and Intent semantics."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from whetstone.core.effects.models import (
    EffectTerminal,
    TerminalOutcome,
)
from whetstone.core.identity import (
    NonEmptyId,
    TerminalFailure,
    typed_ref_for_record,
)
from whetstone.core.roles import EvaluationRole
from whetstone.evaluation.schema_names import (
    EVALUATION_EVIDENCE_SCHEMA,
    EVALUATION_FAILURE_SCHEMA,
)
from whetstone.experiment.binding import EvaluationBinding
from whetstone.experiment.candidate import (
    Candidate,
    candidate_reference,
)
from whetstone.experiment.reward import (
    Reward,
    RewardPolicy,
    RewardTerm,
    apply_reward_policy,
    reward_reference,
)
from whetstone.optimization.contracts import (
    INTENT_RESOLUTION_SCHEMA_VERSION,
    IntentOutcome,
    IntentResolution,
    OptimizationStepResult,
    OutputContract,
    ResolutionClass,
    ResolutionDetail,
    StepStatus,
    ToolEvidence,
    step_request_reference,
)
from whetstone.optimization.proposal.mutation import candidate_from_draft
from whetstone.optimization.proposal.proposer import ProposalDraft
from whetstone.optimization.tools.admission import (
    ToolCallState,
    ToolCallStoreEntry,
    tool_effect_request,
)
from whetstone.optimization.tools.contracts import (
    RefusalClass,
    ToolCall,
    ToolCapacityScope,
    ToolRefusal,
    ToolResult,
    tool_call_reference,
    tool_capacity_binding,
    tool_result_reference,
)

from .support import (
    candidate,
    evaluation_binding,
    make_intent,
    proposal_request,
    pure_request,
    tool_request,
)


def _resolution(
    *,
    request=None,
    resolved_candidate=None,
    outcome: IntentOutcome = IntentOutcome.COMPLETED,
    failure: TerminalFailure | None = None,
    binding: EvaluationBinding | None = None,
) -> IntentResolution:
    exact_request = request or proposal_request()
    exact_candidate = (
        resolved_candidate or _valid_proposed(exact_request).record
    )
    intent = make_intent(
        exact_candidate,
        run_id=exact_request.run_id,
        step_index=exact_request.step_index,
        binding=binding,
    )
    evaluation_result_ref = (
        None
        if outcome is IntentOutcome.REJECTED
        else typed_ref_for_record(
            (
                EVALUATION_EVIDENCE_SCHEMA
                if outcome is IntentOutcome.COMPLETED
                else EVALUATION_FAILURE_SCHEMA
            ),
            {"intent_id": intent.intent_id, "outcome": outcome.value},
        )
    )
    reward_evidence_refs = (
        tuple(
            typed_ref_for_record(
                "whetstone.test.reward_evidence",
                {"intent_id": intent.intent_id, "ordinal": ordinal},
            )
            for ordinal in range(2)
        )
        if (
            outcome is IntentOutcome.COMPLETED
            and intent.evaluation_binding.role is EvaluationRole.INTERNAL
        )
        else ()
    )
    reward_ref = None
    if (
        outcome is IntentOutcome.COMPLETED
        and intent.evaluation_binding.role is EvaluationRole.INTERNAL
    ):
        policy = exact_request.run.record.reward_policy
        assert policy is not None
        reward_ref = reward_reference(
            apply_reward_policy(
                policy,
                aggregates={term.name: 0.75 for term in policy.terms},
                evidence_role=EvaluationRole.INTERNAL,
                evidence_refs=reward_evidence_refs,
            )
        )
    return IntentResolution(
        schema_version=INTENT_RESOLUTION_SCHEMA_VERSION,
        intent=intent,
        outcome=outcome,
        detail=ResolutionDetail(
            classification=(
                ResolutionClass.MEASURED
                if outcome is IntentOutcome.COMPLETED
                else ResolutionClass.UNSCORABLE
            ),
            message=outcome.value,
        ),
        evaluation_result_ref=evaluation_result_ref,
        reward_evidence_refs=reward_evidence_refs,
        resolved_eval_config=intent.target_eval_config,
        reward_ref=reward_ref,
        terminal_failure=failure,
    )


def _proposal_result(
    *,
    request=None,
    resolution: IntentResolution | None = None,
    status: StepStatus = StepStatus.COMPLETE,
    failure: TerminalFailure | None = None,
) -> OptimizationStepResult:
    exact_request = request or proposal_request()
    proposed = _valid_proposed(exact_request)
    return OptimizationStepResult(
        request=step_request_reference(exact_request),
        proposed_candidates=(proposed,),
        accepted_candidates=() if status is StepStatus.FAILED else (proposed,),
        resolved_intents=(
            (resolution,)
            if resolution is not None
            else (_resolution(request=exact_request),)
        ),
        budget=exact_request.budget,
        status=status,
        terminal_failure=failure,
    )


def _valid_proposed(request, candidate_id: str = "P1"):
    return candidate_reference(
        candidate_from_draft(
            base=request.candidates[0],
            candidate_id=candidate_id,
            draft=ProposalDraft(template=f"{candidate_id} {{query}}"),
            run=request.run,
        )
    )


def _tool_evidence(
    request,
    *,
    failure: TerminalFailure | None = None,
) -> ToolEvidence:
    config = request.tool_configs[0]
    binding = tool_capacity_binding(
        ToolCapacityScope.RUN,
        request.run.record_ref,
    )
    call = ToolCall(
        call_id="call-1",
        tool_config=config,
        capacity_binding=binding,
        args={"model_route": "r0", "template": "prompt"},
    )
    result = ToolResult(
        call=tool_call_reference(call),
        output=(
            None
            if failure is not None
            else {"rollout_refs": [], "accepted_ordinal": 1}
        ),
        terminal_failure=failure,
        provenance_ordinal=1,
    )
    result_ref = tool_result_reference(result)
    effect_terminal = EffectTerminal(
        request=tool_effect_request(call),
        outcome=(
            TerminalOutcome.FAILED
            if failure is not None
            else TerminalOutcome.SUCCEEDED
        ),
        owner_id=NonEmptyId("schema-test-owner"),
        attempt_id=NonEmptyId("schema-test-attempt"),
        fence=1,
        result_ref=result_ref.record_ref,
        failure=failure,
    )
    entry = ToolCallStoreEntry(
        tool_call=tool_call_reference(call),
        tool_config=config,
        store_namespace_key=call.store_namespace_key,
        capacity_scope=call.capacity_scope,
        capacity_scope_id=call.capacity_scope_id,
        state=ToolCallState.COMPLETED,
        capacity_debit_ordinal=1,
        tool_result_ref=result_ref.record_ref,
        effect_terminal=effect_terminal,
    )
    return ToolEvidence(result=result_ref, store_entry=entry)


def _refused_tool_evidence(request) -> ToolEvidence:
    config = request.tool_configs[0]
    call = ToolCall(
        call_id="refused-call",
        tool_config=config,
        capacity_binding=tool_capacity_binding(
            ToolCapacityScope.RUN,
            request.run.record_ref,
        ),
        args={"model_route": "r0", "template": "prompt"},
    )
    refusal = ToolRefusal(
        refusal_class=RefusalClass.CAPACITY,
        reason="capacity exhausted",
    )
    result_ref = tool_result_reference(
        ToolResult(
            call=tool_call_reference(call),
            refusal=refusal,
        )
    )
    entry = ToolCallStoreEntry(
        tool_call=tool_call_reference(call),
        tool_config=config,
        store_namespace_key=call.store_namespace_key,
        capacity_scope=call.capacity_scope,
        capacity_scope_id=call.capacity_scope_id,
        state=ToolCallState.REFUSED,
        refusal=refusal,
        tool_result_ref=result_ref.record_ref,
    )
    return ToolEvidence(result=result_ref, store_entry=entry)


def test_tool_evidence_rejects_serialized_success_as_failed() -> None:
    evidence = _tool_evidence(tool_request())
    payload = evidence.model_dump(mode="json")
    terminal = payload["store_entry"]["effect_terminal"]
    assert terminal is not None
    terminal["outcome"] = TerminalOutcome.FAILED.value
    terminal["failure"] = TerminalFailure(
        code="forged_failure",
        message="successful result claimed as failed",
    ).model_dump(mode="json")

    with pytest.raises(
        ValidationError,
        match=r"failed EffectTerminal failure.*exactly equal",
    ):
        ToolEvidence.model_validate_json(json.dumps(payload))


def test_tool_evidence_rejects_serialized_failure_as_success() -> None:
    failure = TerminalFailure(code="tool_failed", message="tool failed")
    evidence = _tool_evidence(tool_request(), failure=failure)
    payload = evidence.model_dump(mode="json")
    terminal = payload["store_entry"]["effect_terminal"]
    assert terminal is not None
    terminal["outcome"] = TerminalOutcome.SUCCEEDED.value
    terminal["failure"] = None

    with pytest.raises(
        ValidationError,
        match=r"succeeded EffectTerminal.*successful Tool Result",
    ):
        ToolEvidence.model_validate_json(json.dumps(payload))


def test_tool_evidence_rejects_serialized_refusal_divergence() -> None:
    evidence = _refused_tool_evidence(tool_request())
    payload = evidence.model_dump(mode="json")
    refusal = payload["store_entry"]["refusal"]
    assert refusal is not None
    refusal["reason"] = "different refusal"

    with pytest.raises(
        ValidationError,
        match=r"exact Tool Result refusal",
    ):
        ToolEvidence.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize("terminal_variant", ["success", "failure"])
def test_refused_tool_evidence_rejects_serialized_result_terminal_variant(
    terminal_variant: str,
) -> None:
    evidence = _refused_tool_evidence(tool_request())
    result = ToolResult(
        call=evidence.result.record.call,
        output=(
            {"rollout_refs": [], "accepted_ordinal": 1}
            if terminal_variant == "success"
            else None
        ),
        terminal_failure=(
            TerminalFailure(code="tool_failed", message="tool failed")
            if terminal_variant == "failure"
            else None
        ),
        provenance_ordinal=1,
    )
    result_ref = tool_result_reference(result)
    payload = evidence.model_dump(mode="json")
    payload["result"] = result_ref.model_dump(mode="json")
    payload["store_entry"]["tool_result_ref"] = (
        result_ref.record_ref.model_dump(mode="json")
    )

    with pytest.raises(
        ValidationError,
        match=r"result has no output or terminal failure",
    ):
        ToolEvidence.model_validate_json(json.dumps(payload))


def test_tool_evidence_rejects_serialized_ordinal_divergence() -> None:
    evidence = _tool_evidence(tool_request())
    payload = evidence.model_dump(mode="json")
    payload["store_entry"]["capacity_debit_ordinal"] = 2

    with pytest.raises(
        ValidationError,
        match=r"capacity debit ordinal.*exactly equal",
    ):
        ToolEvidence.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize("terminal_variant", ["success", "failure"])
@pytest.mark.parametrize("hostile_ordinal", ["missing", 0])
def test_tool_evidence_rejects_serialized_non_positive_result_ordinal(
    terminal_variant: str,
    hostile_ordinal: str | int,
) -> None:
    failure = (
        TerminalFailure(code="tool_failed", message="tool failed")
        if terminal_variant == "failure"
        else None
    )
    evidence = _tool_evidence(tool_request(), failure=failure)
    payload = evidence.model_dump(mode="json")
    result_record = payload["result"]["record"]
    if hostile_ordinal == "missing":
        result_record.pop("provenance_ordinal")
    else:
        result_record["provenance_ordinal"] = hostile_ordinal

    with pytest.raises(
        ValidationError,
        match="non-refused Tool Result requires a positive provenance ordinal",
    ):
        ToolEvidence.model_validate_json(json.dumps(payload))


def test_resolution_composes_reward_and_official_forbids_it() -> None:
    request = proposal_request()
    policy = request.run.record.reward_policy
    assert policy is not None
    evidence_refs = _resolution(request=request).reward_evidence_refs
    reward = apply_reward_policy(
        policy,
        aggregates={term.name: 0.75 for term in policy.terms},
        evidence_role=EvaluationRole.INTERNAL,
        evidence_refs=evidence_refs,
    )
    resolution = _resolution(request=request)
    payload = resolution.model_dump(mode="json")
    payload["reward_ref"] = reward_reference(reward).model_dump(mode="json")
    validated = IntentResolution.model_validate(payload)
    assert validated.reward_ref is not None
    assert validated.reward_ref.record == reward
    payload["reward_ref"] = reward_reference(reward).record_ref.model_dump(
        mode="json"
    )
    with pytest.raises(ValidationError):
        IntentResolution.model_validate(payload)

    exact_binding = evaluation_binding()
    official_payload = exact_binding.model_dump(mode="json")
    official_payload["role"] = EvaluationRole.OFFICIAL.value
    official_payload["authority_principal"] = "official-publisher"
    official = EvaluationBinding.model_validate(official_payload)
    official_resolution = _resolution(request=request, binding=official)
    assert official_resolution.evaluation_result_ref is not None
    assert official_resolution.reward_ref is None
    assert official_resolution.reward_evidence_refs == ()
    official_payload = official_resolution.model_dump(mode="json")
    official_payload["reward_ref"] = reward_reference(reward).model_dump(
        mode="json"
    )
    with pytest.raises(ValidationError, match=r"official.*must not"):
        IntentResolution.model_validate(official_payload)


@pytest.mark.parametrize(
    "outcome",
    [IntentOutcome.COMPLETED, IntentOutcome.FAILED],
)
def test_executed_resolution_requires_evaluation_result(
    outcome: IntentOutcome,
) -> None:
    resolution = _resolution(
        outcome=outcome,
        failure=(
            TerminalFailure(code="evaluation_failed", message="failed")
            if outcome is IntentOutcome.FAILED
            else None
        ),
    )
    payload = resolution.model_dump(mode="json")
    payload["evaluation_result_ref"] = None

    with pytest.raises(ValidationError, match="requires an Evaluation Result"):
        IntentResolution.model_validate(payload)


def test_rejected_resolution_forbids_evaluation_result() -> None:
    resolution = _resolution(outcome=IntentOutcome.REJECTED)
    payload = resolution.model_dump(mode="json")
    payload["evaluation_result_ref"] = typed_ref_for_record(
        EVALUATION_EVIDENCE_SCHEMA, {"result": "forbidden"}
    ).model_dump(mode="json")

    with pytest.raises(
        ValidationError, match="must not carry an Evaluation Result"
    ):
        IntentResolution.model_validate(payload)


@pytest.mark.parametrize(
    ("outcome", "wrong_schema"),
    [
        (IntentOutcome.COMPLETED, EVALUATION_FAILURE_SCHEMA),
        (IntentOutcome.FAILED, EVALUATION_EVIDENCE_SCHEMA),
        (IntentOutcome.COMPLETED, "whetstone.evaluation_binding"),
        (IntentOutcome.FAILED, "whetstone.evaluation_binding"),
        (IntentOutcome.COMPLETED, "whetstone.evaluation_outputs"),
        (IntentOutcome.FAILED, "whetstone.evaluation_outputs"),
        (IntentOutcome.COMPLETED, "whetstone.evaluation_intent_claim"),
        (IntentOutcome.FAILED, "whetstone.evaluation_intent_claim"),
    ],
)
def test_executed_resolution_rejects_non_evaluation_result_schema(
    outcome: IntentOutcome,
    wrong_schema: str,
) -> None:
    resolution = _resolution(
        outcome=outcome,
        failure=(
            TerminalFailure(code="evaluation_failed", message="failed")
            if outcome is IntentOutcome.FAILED
            else None
        ),
    )
    payload = resolution.model_dump(mode="json")
    payload["evaluation_result_ref"] = typed_ref_for_record(
        wrong_schema, {"result": "wrong schema"}
    ).model_dump(mode="json")

    with pytest.raises(
        ValidationError, match="evaluation_result_ref must use schema"
    ):
        IntentResolution.model_validate(payload)


def test_completed_internal_resolution_requires_reward() -> None:
    resolution = _resolution()
    payload = resolution.model_dump(mode="json")
    payload["reward_ref"] = None
    payload["reward_evidence_refs"] = []

    with pytest.raises(ValidationError, match="requires a Reward"):
        IntentResolution.model_validate(payload)


@pytest.mark.parametrize("kind", ["official_completed", "failed"])
def test_rewardless_resolution_requires_empty_reward_evidence_refs(
    kind: str,
) -> None:
    if kind == "official_completed":
        binding_payload = evaluation_binding().model_dump(mode="json")
        binding_payload["role"] = EvaluationRole.OFFICIAL.value
        binding_payload["authority_principal"] = "official-publisher"
        resolution = _resolution(
            binding=EvaluationBinding.model_validate(binding_payload)
        )
    else:
        resolution = _resolution(
            outcome=IntentOutcome.FAILED,
            failure=TerminalFailure(
                code="evaluation_failed", message="failed"
            ),
        )
    payload = resolution.model_dump(mode="json")
    payload["reward_evidence_refs"] = [
        typed_ref_for_record(
            "whetstone.test.reward_evidence", {"forbidden": kind}
        ).model_dump(mode="json")
    ]

    with pytest.raises(ValidationError, match=r"rewardless.*must not carry"):
        IntentResolution.model_validate(payload)


@pytest.mark.parametrize(
    "outcome",
    [IntentOutcome.REJECTED, IntentOutcome.FAILED],
)
def test_only_completed_intent_resolution_may_carry_reward(
    outcome: IntentOutcome,
) -> None:
    request = proposal_request()
    policy = request.run.record.reward_policy
    assert policy is not None
    evidence_refs = (
        typed_ref_for_record(
            "whetstone.test.evaluation_evidence",
            {"intent_id": "forbidden-reward"},
        ),
    )
    reward = apply_reward_policy(
        policy,
        aggregates={term.name: 0.75 for term in policy.terms},
        evidence_role=EvaluationRole.INTERNAL,
        evidence_refs=evidence_refs,
    )
    failure = (
        TerminalFailure(code="provider", message="failed")
        if outcome is IntentOutcome.FAILED
        else None
    )
    if outcome is IntentOutcome.REJECTED:
        intent = make_intent(
            candidate("P1"),
            run_id=request.run_id,
            step_index=request.step_index,
        )
        resolution = IntentResolution(
            schema_version=INTENT_RESOLUTION_SCHEMA_VERSION,
            intent=intent,
            outcome=outcome,
            detail=ResolutionDetail(
                classification=ResolutionClass.VALIDATION,
                message="rejected",
            ),
            resolved_eval_config=intent.target_eval_config,
        )
    else:
        resolution = _resolution(
            request=request,
            outcome=outcome,
            failure=failure,
        )
    payload = resolution.model_dump(mode="json")
    payload["reward_ref"] = reward_reference(reward).model_dump(mode="json")

    with pytest.raises(ValidationError, match="only a completed"):
        IntentResolution.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("run_id", "another-run", "exact request run"),
        ("step_index", 1, "exact request step"),
    ],
)
def test_step_result_rejects_intent_from_another_request_position(
    field: str,
    value: str | int,
    message: str,
) -> None:
    result = _proposal_result()
    payload = result.model_dump(mode="json")
    payload["resolved_intents"][0]["intent"][field] = value

    with pytest.raises(ValidationError, match=message):
        OptimizationStepResult.model_validate(payload)


def test_step_result_rejects_intent_for_an_unbound_candidate() -> None:
    result = _proposal_result()
    payload = result.model_dump(mode="json")
    payload["resolved_intents"][0]["intent"]["candidate"] = (
        candidate_reference(candidate("outside")).model_dump(mode="json")
    )

    with pytest.raises(ValidationError, match="exact request, or proposed"):
        OptimizationStepResult.model_validate(payload)


def test_step_result_allows_intent_for_exact_request_candidate() -> None:
    request = proposal_request(candidates=(candidate("A"),))
    resolution = _resolution(
        request=request,
        resolved_candidate=request.candidates[0],
    )
    result = _proposal_result(request=request, resolution=resolution)
    assert (
        result.resolved_intents[0].intent.candidate.record.candidate_id == "A"
    )


def test_step_result_enforces_mode_specific_evidence_shape() -> None:
    proposal = _proposal_result()
    pure = pure_request()
    pure_candidate = candidate_reference(pure.candidates[0])
    pure_payload = OptimizationStepResult(
        request=step_request_reference(pure),
        proposed_candidates=(pure_candidate,),
        accepted_candidates=(pure_candidate,),
        budget=pure.budget,
        status=StepStatus.COMPLETE,
    ).model_dump(mode="json")
    pure_payload["resolved_intents"] = proposal.model_dump(mode="json")[
        "resolved_intents"
    ]
    with pytest.raises(ValidationError, match=r"pure.*no execution evidence"):
        OptimizationStepResult.model_validate(pure_payload)

    tool = tool_request()
    proposed_candidate = candidate_reference(
        candidate_from_draft(
            base=tool.candidates[0],
            candidate_id="TP",
            draft=ProposalDraft(template="tool proposed {query}"),
            run=tool.run,
        )
    )
    tool_payload = OptimizationStepResult(
        request=step_request_reference(tool),
        proposed_candidates=(proposed_candidate,),
        accepted_candidates=(proposed_candidate,),
        budget=tool.budget,
        status=StepStatus.COMPLETE,
    ).model_dump(mode="json")
    tool_payload["resolved_intents"] = proposal.model_dump(mode="json")[
        "resolved_intents"
    ]
    with pytest.raises(ValidationError, match=r"tool-using.*Tool Evidence"):
        OptimizationStepResult.model_validate(tool_payload)

    proposal_payload = proposal.model_dump(mode="json")
    proposal_payload["resolved_intents"] = []
    proposal_payload["tool_evidence"] = [
        _tool_evidence(tool).model_dump(mode="json")
    ]
    with pytest.raises(ValidationError, match=r"proposal-only.*Intent"):
        OptimizationStepResult.model_validate_json(
            json.dumps(proposal_payload)
        )


def test_failed_step_requires_exact_nested_tool_failure() -> None:
    request = tool_request()
    nested = TerminalFailure(code="tool_failed", message="tool failed")
    result = OptimizationStepResult(
        request=step_request_reference(request),
        tool_evidence=(_tool_evidence(request, failure=nested),),
        budget=request.budget,
        status=StepStatus.FAILED,
        terminal_failure=nested,
    )
    payload = result.model_dump(mode="json")
    payload["terminal_failure"] = TerminalFailure(
        code="outer_failed",
        message="different failure",
    ).model_dump(mode="json")

    with pytest.raises(ValidationError, match="exact outer Step failure"):
        OptimizationStepResult.model_validate_json(json.dumps(payload))


def test_failed_step_requires_exact_nested_intent_failure() -> None:
    request = proposal_request()
    nested = TerminalFailure(code="intent_failed", message="intent failed")
    resolution = _resolution(
        request=request,
        outcome=IntentOutcome.FAILED,
        failure=nested,
    )
    result = _proposal_result(
        request=request,
        resolution=resolution,
        status=StepStatus.FAILED,
        failure=nested,
    )
    payload = result.model_dump(mode="json")
    payload["terminal_failure"] = TerminalFailure(
        code="outer_failed",
        message="different failure",
    ).model_dump(mode="json")

    with pytest.raises(ValidationError, match="exact outer Step failure"):
        OptimizationStepResult.model_validate(payload)


def test_nonfailed_step_may_retain_candidate_local_intent_failure() -> None:
    request = proposal_request()
    local_failure = TerminalFailure(
        code="candidate_unscorable",
        message="candidate could not be scored",
    )
    resolution = _resolution(
        request=request,
        outcome=IntentOutcome.FAILED,
        failure=local_failure,
    )
    result = _proposal_result(request=request, resolution=resolution)
    assert result.status is StepStatus.COMPLETE
    assert result.resolved_intents[0].terminal_failure == local_failure


def test_pure_result_preserves_exact_request_multiset() -> None:
    exact = candidate("A")
    request = pure_request(candidates=(exact, exact))
    exact_ref = candidate_reference(exact)
    result = OptimizationStepResult(
        request=step_request_reference(request),
        proposed_candidates=(exact_ref, exact_ref),
        accepted_candidates=(exact_ref, exact_ref),
        budget=request.budget,
        status=StepStatus.COMPLETE,
    )
    assert result.proposed_candidates == (exact_ref, exact_ref)

    payload = result.model_dump(mode="json")
    payload["proposed_candidates"].pop()
    with pytest.raises(ValidationError, match="exactly equal"):
        OptimizationStepResult.model_validate(payload)


@pytest.mark.parametrize("forgery", ["base", "surface"])
def test_effectful_result_revalidates_exact_base_and_mutation_surface(
    forgery: str,
) -> None:
    request = proposal_request()
    valid = _valid_proposed(request)
    record = valid.record
    forged = Candidate(
        candidate_id=record.candidate_id,
        base_ref=(
            typed_ref_for_record("whetstone.test.other_base", {"id": "other"})
            if forgery == "base"
            else record.base_ref
        ),
        payload={
            **record.payload.to_json(),
            "fixed": "forged" if forgery == "surface" else "same",
        },
    )
    with pytest.raises(
        ValidationError,
        match=(
            "exact request base"
            if forgery == "base"
            else "canonical Mutation Surface"
        ),
    ):
        OptimizationStepResult(
            request=step_request_reference(request),
            proposed_candidates=(candidate_reference(forged),),
            accepted_candidates=(candidate_reference(forged),),
            budget=request.budget,
            status=StepStatus.COMPLETE,
        )


def test_accepted_candidate_subset_preserves_legitimate_multiplicity() -> None:
    request = proposal_request(
        contract=OutputContract(returned_proposal_count=2)
    )
    proposed = _valid_proposed(request)
    result = OptimizationStepResult(
        request=step_request_reference(request),
        proposed_candidates=(proposed, proposed),
        accepted_candidates=(proposed, proposed),
        budget=request.budget,
        status=StepStatus.COMPLETE,
    )
    assert result.accepted_candidates == (proposed, proposed)

    with pytest.raises(ValidationError, match="multiset must be contained"):
        OptimizationStepResult(
            request=step_request_reference(request),
            proposed_candidates=(proposed,),
            accepted_candidates=(proposed, proposed),
            budget=request.budget,
            status=StepStatus.COMPLETE,
        )


def test_step_result_rejects_duplicate_intent_resolutions() -> None:
    result = _proposal_result()
    payload = result.model_dump(mode="json")
    payload["resolved_intents"].append(payload["resolved_intents"][0])

    with pytest.raises(
        ValidationError, match="duplicate Evaluation Intent IDs"
    ):
        OptimizationStepResult.model_validate(payload)


def test_request_and_result_candidates_obey_run_template_contract() -> None:
    request = proposal_request()
    request_payload = request.model_dump(mode="json")
    request_payload["candidates"][0]["payload"]["user_prompt_template"] = (
        "{unavailable}"
    )
    with pytest.raises(
        ValidationError, match=r"Step Request candidate.*template"
    ):
        type(request).model_validate(request_payload)

    base = request.candidates[0]
    invalid = Candidate(
        candidate_id="P-invalid",
        base_ref=candidate_reference(base).record_ref,
        payload={
            **base.payload.to_json(),
            "user_prompt_template": "{unavailable}",
        },
    )
    with pytest.raises(ValidationError, match=r"proposed candidate.*template"):
        OptimizationStepResult(
            request=step_request_reference(request),
            proposed_candidates=(candidate_reference(invalid),),
            accepted_candidates=(candidate_reference(invalid),),
            budget=request.budget,
            status=StepStatus.COMPLETE,
        )


def test_resolution_reward_binds_evidence_and_run_policy() -> None:
    request = proposal_request()
    resolution = _resolution(request=request)
    assert resolution.reward_ref is not None
    assert len(resolution.reward_evidence_refs) == 2
    assert (
        resolution.reward_ref.record.evidence_refs
        == resolution.reward_evidence_refs
    )

    order_payload = resolution.model_dump(mode="json")
    order_payload["reward_evidence_refs"] = list(
        reversed(order_payload["reward_evidence_refs"])
    )
    with pytest.raises(ValidationError, match="exactly equal"):
        IntentResolution.model_validate(order_payload)

    evidence_payload = resolution.model_dump(mode="json")
    changed_reward = resolution.reward_ref.record.model_dump(mode="json")
    changed_reward["evidence_refs"] = [
        typed_ref_for_record(
            "whetstone.test.evaluation_evidence",
            {"intent_id": "different"},
        ).model_dump(mode="json")
    ]
    evidence_payload["reward_ref"] = reward_reference(
        Reward.model_validate(changed_reward)
    ).model_dump(mode="json")
    with pytest.raises(ValidationError, match="exactly equal"):
        IntentResolution.model_validate(evidence_payload)

    alternate = RewardPolicy(
        policy_name="alternate/v1",
        terms=(RewardTerm(name="score", weight=2.0),),
    )
    reward = apply_reward_policy(
        alternate,
        aggregates={"score": 0.75},
        evidence_role=EvaluationRole.INTERNAL,
        evidence_refs=resolution.reward_evidence_refs,
    )
    policy_payload = resolution.model_dump(mode="json")
    policy_payload["reward_ref"] = reward_reference(reward).model_dump(
        mode="json"
    )
    with pytest.raises(ValidationError, match="expected Reward Policy hash"):
        IntentResolution.model_validate(policy_payload)
