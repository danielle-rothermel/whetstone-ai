"""Typed optimization serialization-boundary contracts."""

import pickle
from typing import Any

import pytest
from dr_providers import ProviderTransportPolicy
from pydantic import ValidationError

from whetstone.evaluation_role import EvaluationRole
from whetstone.optimization import (
    EVALUATION_EVIDENCE_SCHEMA,
    EVALUATION_FAILURE_SCHEMA,
    INTENT_RESOLUTION_SCHEMA,
    INTENT_RESOLUTION_SCHEMA_VERSION,
    BudgetState,
    Candidate,
    CandidateRef,
    EvalConfigRef,
    EvaluationIntent,
    IdentityRef,
    ImmutableJsonObject,
    IntentOutcome,
    IntentResolution,
    OptimizationRun,
    OptimizationStepRequest,
    OptimizationStepResult,
    OutputContract,
    ProposalDraft,
    ResolutionClass,
    ResolutionDetail,
    RewardPolicy,
    RewardTerm,
    StepKind,
    StepMode,
    StepStatus,
    TemplateRenderContract,
    TemplateRenderKind,
    TerminalFailure,
    ToolCall,
    ToolCapacityScope,
    ToolDefinitionRef,
    ToolRefusal,
    ToolResult,
    TypedRef,
    candidate_from_draft,
    candidate_reference,
    canonical_json_equal,
    compute_identity_hash,
    eval_config_reference,
    optimization_run_reference,
    tool_call_reference,
    tool_config_reference,
    tool_result_reference,
    typed_ref_for_record,
)
from whetstone.optimization.schema import (
    CANDIDATE_IDENTITY_SCHEMA,
    CANDIDATE_IDENTITY_SCHEMA_VERSION,
    EVALUATION_BINDING_SCHEMA,
    EVALUATION_BINDING_SCHEMA_VERSION,
    OPTIMIZATION_RUN_SCHEMA,
    OPTIMIZATION_RUN_SCHEMA_VERSION,
    STEP_RESULT_SCHEMA,
    EvaluationBinding,
    ExecutionEnvironmentFingerprint,
    OptimizationProposal,
    OptimizationResult,
    OptimizationStepRequestRef,
    OptimizationStepResultRef,
    step_request_reference,
    step_result_reference,
)
from whetstone.optimization.tools import tool_capacity_binding
from whetstone.provider.policy import (
    PROVIDER_EXECUTION_POLICY_SCHEMA,
    ProviderExecutionPolicy,
)

from .support import (
    FULL_A,
    candidate,
    eval_config,
    make_tool_definition_config,
    python_format_contract,
    tool_run,
)


def test_candidate_ref_binds_exact_record_content_and_identity() -> None:
    record = candidate()
    ref = candidate_reference(record)
    assert ref.record == record
    assert ref.identity_hash == record.identity_hash()
    with pytest.raises(ValidationError, match="exact candidate"):
        CandidateRef(
            record=record,
            record_ref=TypedRef(schema_name="wrong", content_hash=FULL_A),
            identity_hash=record.identity_hash(),
        )


def test_eval_config_ref_binds_exact_typed_record_and_identity() -> None:
    record = eval_config()
    ref = eval_config_reference(record)
    assert ref.record == record
    assert ref.identity_hash == record.config_identity_hash
    with pytest.raises(ValidationError, match="identity_hash"):
        EvalConfigRef(
            record=record,
            record_ref=ref.record_ref,
            identity_hash=FULL_A,
        )


def test_intent_has_exact_refs_and_no_loose_identity_fields() -> None:
    proposed = candidate("P1")
    intent = _evaluation_intent(proposed)
    dumped = intent.model_dump()
    assert dumped["candidate"]["record"]["candidate_id"] == "P1"
    assert dumped["target_eval_config"]["record"]["config_identity_hash"]
    assert (
        dumped["evaluation_binding"]["eval_config"]["identity_hash"]
        == intent.target_eval_config.identity_hash
    )
    assert "candidate_id" not in dumped
    assert "target_eval_config_ref" not in dumped
    assert "target_eval_config_hash" not in dumped
    with pytest.raises(ValidationError):
        EvaluationIntent.model_validate({**dumped, "candidate_id": "P1"})


def _evaluation_binding(
    *,
    role: EvaluationRole = EvaluationRole.INTERNAL,
    authority_principal: str | None = None,
    config: EvalConfigRef | None = None,
) -> EvaluationBinding:
    return EvaluationBinding(
        schema_version=EVALUATION_BINDING_SCHEMA_VERSION,
        eval_config=config or eval_config_reference(eval_config()),
        role=role,
        authority_principal=authority_principal,
        campaign="schema-tests",
        provider_execution_policy_ref=_provider_execution_policy_ref(),
        retry_policy_ref=typed_ref_for_record(
            "whetstone.test.retry_policy",
            {"max_retries": 1},
        ),
        operational_policy_refs=(
            typed_ref_for_record(
                "whetstone.test.accounting_policy",
                {"currency": "usd"},
            ),
        ),
        environment_fingerprint=ExecutionEnvironmentFingerprint(
            dependency_versions=(("dr-code", "0.1.0"),),
            code_revision="deadbeef",
            runtime_identity="linux-x86_64",
        ),
        provenance_note="schema test",
        provenance_ordinal=1,
    )


def _provider_execution_policy_ref() -> IdentityRef:
    policy = ProviderExecutionPolicy(
        transport_policy=ProviderTransportPolicy(
            api_key_env="TEST_PROVIDER_API_KEY",
            base_url="https://provider.test/v1",
        ),
        max_attempts=2,
    )
    return IdentityRef(
        record_ref=typed_ref_for_record(
            PROVIDER_EXECUTION_POLICY_SCHEMA,
            policy.identity_payload(),
        ),
        identity_hash=policy.identity_hash,
    )


def _evaluation_intent(
    proposed: Candidate,
    *,
    run_id: str = "run-proposal",
    step_index: int = 0,
) -> EvaluationIntent:
    target = eval_config_reference(eval_config())
    binding = _evaluation_binding(config=target)
    return EvaluationIntent(
        intent_id=f"{run_id}-{step_index}-{proposed.candidate_id}",
        candidate=candidate_reference(proposed),
        target_eval_config=target,
        evaluation_binding=binding,
        purpose="proposal",
        run_id=run_id,
        step_index=step_index,
        expected_reward_policy_hash=_reward_policy().identity_hash(),
    )


def _reward_policy() -> RewardPolicy:
    return RewardPolicy(
        policy_name="proposal-score/v1",
        terms=(RewardTerm(name="score", weight=1.0),),
    )


def _proposal_request(
    *,
    run_id: str = "run-proposal",
    step_index: int = 0,
    prior_step_result_ref: TypedRef | None = None,
    prior_state_ref: TypedRef | None = None,
    prior_history_ref: TypedRef | None = None,
    budget: BudgetState | None = None,
    contract: OutputContract | None = None,
    template_render_contract: TemplateRenderContract | None = None,
) -> OptimizationStepRequest:
    optimizer_config = IdentityRef(
        record_ref=typed_ref_for_record(
            "whetstone.test.optimizer_config",
            {"algorithm": "proposal"},
        ),
        identity_hash=FULL_A,
    )
    run = OptimizationRun(
        run_id=run_id,
        optimizer_config=optimizer_config,
        adapter_key="proposal-test",
        mode=StepMode.PROPOSAL_ONLY,
        terminal_output_contract=OutputContract(returned_proposal_count=1),
        template_render_contract=(
            template_render_contract or python_format_contract()
        ),
        reward_policy=_reward_policy(),
    )
    return OptimizationStepRequest(
        run=optimization_run_reference(run),
        step_id=f"{run_id}-s{step_index}",
        kind=StepKind.PROPOSAL,
        step_index=step_index,
        prior_step_result_ref=prior_step_result_ref,
        prior_state_ref=prior_state_ref,
        prior_history_ref=prior_history_ref,
        candidates=(candidate(),),
        step_output_contract=contract
        or OutputContract(returned_proposal_count=1),
        budget=budget or BudgetState(remaining={"rollouts": 10}),
    )


def _step_result(
    request: OptimizationStepRequest,
    *,
    status: StepStatus = StepStatus.COMPLETE,
    state_ref: TypedRef | None = None,
    history_ref: TypedRef | None = None,
    terminal_failure: TerminalFailure | None = None,
) -> OptimizationStepResult:
    proposed = candidate_reference(
        candidate_from_draft(
            base=request.candidates[0],
            candidate_id="P1",
            draft=ProposalDraft(template="proposed {query}"),
            run=request.run,
        )
    )
    return OptimizationStepResult(
        request=step_request_reference(request),
        proposed_candidates=(proposed,),
        accepted_candidates=() if status is StepStatus.FAILED else (proposed,),
        state_ref=state_ref,
        history_ref=history_ref,
        budget=request.budget,
        status=status,
        terminal_failure=terminal_failure,
    )


def test_candidate_identity_contract_literals_are_pinned() -> None:
    record = candidate()

    assert CANDIDATE_IDENTITY_SCHEMA == "whetstone.optimization_candidate"
    assert CANDIDATE_IDENTITY_SCHEMA_VERSION == 1
    assert tuple(record.identity_payload()) == (
        "candidate_id",
        "base_ref",
        "payload",
    )
    assert (
        record.identity_hash()
        == "d13bd9c7dcb859cc7260591eed0f7ec4bbd6a296dccc934bbf7090ae3a9ebca3"
    )


def test_evaluation_binding_identity_contract_literals_are_pinned() -> None:
    binding = _evaluation_binding()
    assert EVALUATION_BINDING_SCHEMA == "whetstone.evaluation_binding"
    assert EVALUATION_BINDING_SCHEMA_VERSION == 2
    assert binding.schema_version == EVALUATION_BINDING_SCHEMA_VERSION
    assert binding.record_content()["schema_version"] == 2
    assert tuple(binding.record_content()) == tuple(binding.identity_payload())
    assert tuple(binding.identity_payload()) == (
        "schema_version",
        "eval_config",
        "role",
        "authority_principal",
        "campaign",
        "provider_execution_policy_ref",
        "retry_policy_ref",
        "operational_policy_refs",
        "environment_fingerprint",
        "provenance_note",
        "provenance_ordinal",
    )
    assert binding.identity_payload()["provider_execution_policy_ref"] == {
        "record_ref": {
            "schema_name": "whetstone.provider_execution_policy",
            "content_hash": (
                "ddb2115fb1631560c9b02b1aa16820482"
                "e37b28523d1f43ddd7dbecbed664909"
            ),
        },
        "identity_hash": (
            "e11d5ffb3acb35048f57ae08dbc34cc4b68332115707ecf8fd304e8c5d147ac2"
        ),
    }
    assert (
        binding.identity_hash()
        == "d77a3ea054252f78bbce949e66569a32b2f01e71c43785443597f44c731e4391"
    )


def test_evaluation_binding_rejects_wrong_provider_policy_schema() -> None:
    binding = _evaluation_binding()
    payload = binding.model_dump(mode="json")
    payload["provider_execution_policy_ref"]["record_ref"]["schema_name"] = (
        "whetstone.test.wrong_policy"
    )

    with pytest.raises(
        ValidationError,
        match="provider_execution_policy_ref must use schema",
    ):
        EvaluationBinding.model_validate(payload)


@pytest.mark.parametrize(
    "provider_ref_present",
    [True, False],
    ids=["provider-ref-present", "provider-ref-absent"],
)
def test_evaluation_binding_v1_wire_is_partitioned_and_rejected(
    provider_ref_present: bool,
) -> None:
    current_payload = _evaluation_binding().model_dump(mode="json")
    if not provider_ref_present:
        current_payload["provider_execution_policy_ref"] = None
    current_binding = EvaluationBinding.model_validate(current_payload)

    legacy_wire = current_binding.model_dump(mode="json")
    legacy_wire.pop("schema_version")
    if provider_ref_present:
        policy_ref = current_binding.provider_execution_policy_ref
        assert policy_ref is not None
        legacy_wire["provider_execution_policy_ref"] = (
            policy_ref.record_ref.model_dump(mode="json")
        )

    legacy_identity_hash = compute_identity_hash(
        schema=EVALUATION_BINDING_SCHEMA,
        schema_version=1,
        payload=legacy_wire,
    )
    assert (
        legacy_identity_hash
        == {
            True: (
                "f95ccb10ad8717c32924c1ca2355caf9"
                "f7679ddc5b95d40472f61e1f3dc75f97"
            ),
            False: (
                "f182528f43640e0342fe996172213e68d"
                "c5a7049fa75fe3d0196ac88b735309f"
            ),
        }[provider_ref_present]
    )
    assert legacy_identity_hash != current_binding.identity_hash()

    with pytest.raises(ValidationError, match="Field required"):
        EvaluationBinding.model_validate(legacy_wire)
    with pytest.raises(ValidationError, match="Input should be 2"):
        EvaluationBinding.model_validate({"schema_version": 1, **legacy_wire})


def test_intent_resolution_v2_wire_contract_is_exact() -> None:
    intent = _evaluation_intent(candidate("P1"))
    resolution = IntentResolution(
        schema_version=INTENT_RESOLUTION_SCHEMA_VERSION,
        intent=intent,
        outcome=IntentOutcome.REJECTED,
        detail=ResolutionDetail(
            classification=ResolutionClass.VALIDATION,
            message="rejected",
        ),
        resolved_eval_config=intent.target_eval_config,
    )
    record = resolution.model_dump(mode="json")

    assert INTENT_RESOLUTION_SCHEMA == (
        "whetstone.optimization_intent_resolution"
    )
    assert INTENT_RESOLUTION_SCHEMA_VERSION == 2
    assert EVALUATION_EVIDENCE_SCHEMA == "whetstone.evaluation_evidence"
    assert EVALUATION_FAILURE_SCHEMA == "whetstone.evaluation_failure"
    assert tuple(record) == (
        "schema_version",
        "intent",
        "outcome",
        "detail",
        "evaluation_result_ref",
        "reward_evidence_refs",
        "resolved_eval_config",
        "reward_ref",
        "terminal_failure",
    )
    assert record["schema_version"] == 2
    assert record["evaluation_result_ref"] is None
    assert record["reward_evidence_refs"] == []
    assert (
        typed_ref_for_record(INTENT_RESOLUTION_SCHEMA, record).content_hash
        == "4390a1d15b03a38c06119832292033eb665909d6f0deb55025c56e84dc02f3ea"
    )


def test_intent_resolution_rejects_v1_wire() -> None:
    intent = _evaluation_intent(candidate("P1"))
    resolution = IntentResolution(
        schema_version=INTENT_RESOLUTION_SCHEMA_VERSION,
        intent=intent,
        outcome=IntentOutcome.REJECTED,
        detail=ResolutionDetail(
            classification=ResolutionClass.VALIDATION,
            message="rejected",
        ),
        resolved_eval_config=intent.target_eval_config,
    )
    payload = resolution.model_dump(mode="json")
    payload["evaluation_evidence_refs"] = []

    with pytest.raises(
        ValidationError, match="Extra inputs are not permitted"
    ):
        IntentResolution.model_validate(payload)

    payload.pop("schema_version")
    with pytest.raises(ValidationError):
        IntentResolution.model_validate(payload)

    with pytest.raises(ValidationError, match="Input should be 2"):
        IntentResolution.model_validate(
            {**resolution.model_dump(mode="json"), "schema_version": 1}
        )


def test_evaluation_binding_enforces_official_authority() -> None:
    with pytest.raises(ValidationError, match="required for official"):
        _evaluation_binding(role=EvaluationRole.OFFICIAL)
    with pytest.raises(ValidationError, match="absent for internal"):
        _evaluation_binding(authority_principal="official-publisher")

    official = _evaluation_binding(
        role=EvaluationRole.OFFICIAL,
        authority_principal="official-publisher",
    )
    assert official.authority_principal == "official-publisher"


def test_intent_requires_its_binding_exact_eval_config() -> None:
    intent = _evaluation_intent(candidate("P1"))
    payload = intent.model_dump(mode="json")
    payload["evaluation_binding"] = _evaluation_binding(
        config=eval_config_reference(eval_config("e" * 64))
    ).model_dump(mode="json")

    with pytest.raises(ValidationError, match="must match its exact"):
        EvaluationIntent.model_validate(payload)


def test_intent_reward_expectation_follows_evaluation_role() -> None:
    internal = _evaluation_intent(candidate("P1"))
    payload = internal.model_dump(mode="json")
    payload["expected_reward_policy_hash"] = None
    with pytest.raises(ValidationError, match=r"internal.*requires"):
        EvaluationIntent.model_validate(payload)

    official_binding = _evaluation_binding(
        role=EvaluationRole.OFFICIAL,
        authority_principal="official-publisher",
    )
    official_payload = internal.model_dump(mode="json")
    official_payload["evaluation_binding"] = official_binding.model_dump(
        mode="json"
    )
    official_payload["target_eval_config"] = (
        official_binding.eval_config.model_dump(mode="json")
    )
    with pytest.raises(ValidationError, match=r"official.*must not"):
        EvaluationIntent.model_validate(official_payload)
    official_payload["expected_reward_policy_hash"] = None
    assert (
        EvaluationIntent.model_validate(
            official_payload
        ).expected_reward_policy_hash
        is None
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("binding_role", EvaluationRole.INTERNAL.value),
        (
            "binding_policy_ref",
            {
                "schema_name": "whetstone.test.policy",
                "content_hash": FULL_A,
            },
        ),
    ],
)
def test_intent_rejects_uncomposed_binding_fields(
    field: str, value: Any
) -> None:
    intent = _evaluation_intent(candidate("P1"))
    with pytest.raises(
        ValidationError, match="Extra inputs are not permitted"
    ):
        EvaluationIntent.model_validate(
            {
                **intent.model_dump(mode="json"),
                field: value,
            }
        )


def test_evaluation_binding_defensively_copies_nested_identity_data() -> None:
    source = _evaluation_binding().model_dump(mode="json")
    binding = EvaluationBinding.model_validate(source)
    before = binding.identity_hash()

    source["environment_fingerprint"]["dependency_versions"][0][1] = "9.9.9"
    source["operational_policy_refs"][0]["schema_name"] = "whetstone.wrong"

    assert binding.environment_fingerprint.dependency_versions == (
        ("dr-code", "0.1.0"),
    )
    assert (
        binding.operational_policy_refs[0].schema_name
        == "whetstone.test.accounting_policy"
    )
    assert binding.identity_hash() == before
    assert (
        EvaluationBinding.model_validate(
            binding.model_dump(mode="json")
        ).identity_hash()
        == before
    )
    with pytest.raises(ValidationError, match="frozen"):
        binding.environment_fingerprint.__setattr__("code_revision", "changed")


def test_evaluation_binding_identity_is_sensitive_to_exact_content() -> None:
    internal = _evaluation_binding()
    changed_campaign_payload = internal.model_dump(mode="json")
    changed_campaign_payload["campaign"] = "another-campaign"
    changed_environment_payload = internal.model_dump(mode="json")
    changed_environment_payload["environment_fingerprint"][
        "runtime_identity"
    ] = "darwin"
    official = _evaluation_binding(
        role=EvaluationRole.OFFICIAL,
        authority_principal="official-publisher",
    )

    assert (
        EvaluationBinding.model_validate(
            changed_campaign_payload
        ).identity_hash()
        != internal.identity_hash()
    )
    assert (
        EvaluationBinding.model_validate(
            changed_environment_payload
        ).identity_hash()
        != internal.identity_hash()
    )
    assert official.identity_hash() != internal.identity_hash()


def test_environment_dependencies_are_unique_and_canonical_by_package() -> (
    None
):
    fingerprint = ExecutionEnvironmentFingerprint(
        dependency_versions=(
            ("whetstone-envs", "2"),
            ("dr-code", "1"),
        )
    )
    assert fingerprint.dependency_versions == (
        ("dr-code", "1"),
        ("whetstone-envs", "2"),
    )
    with pytest.raises(ValidationError, match="package names must be unique"):
        ExecutionEnvironmentFingerprint(
            dependency_versions=(("dr-code", "1"), ("dr-code", "2"))
        )


@pytest.mark.parametrize("unordered", [set(), frozenset(), {}])
def test_ordered_fields_reject_unordered_python_containers(
    unordered: object,
) -> None:
    with pytest.raises(ValidationError, match="ordered tuple or JSON array"):
        OptimizationRun.model_validate(
            {
                **_proposal_request().run.record.record_content(),
                "tool_configs": unordered,
            }
        )
    with pytest.raises(ValidationError, match="ordered tuple or JSON array"):
        EvaluationBinding.model_validate(
            {
                **_evaluation_binding().model_dump(mode="json"),
                "operational_policy_refs": unordered,
            }
        )
    with pytest.raises(ValidationError, match="ordered tuple or JSON array"):
        OptimizationStepRequest.model_validate(
            {
                **_proposal_request().model_dump(mode="json"),
                "candidates": unordered,
            }
        )
    resolution = IntentResolution(
        schema_version=INTENT_RESOLUTION_SCHEMA_VERSION,
        intent=_evaluation_intent(candidate("P1")),
        outcome=IntentOutcome.REJECTED,
        detail=ResolutionDetail(
            classification=ResolutionClass.VALIDATION,
            message="rejected",
        ),
        resolved_eval_config=eval_config_reference(eval_config()),
    )
    with pytest.raises(ValidationError, match="ordered tuple or JSON array"):
        IntentResolution.model_validate(
            {
                **resolution.model_dump(mode="json"),
                "reward_evidence_refs": unordered,
            }
        )


def test_dependency_pairs_reject_unordered_containers() -> None:
    with pytest.raises(ValidationError, match="package/version pairs"):
        ExecutionEnvironmentFingerprint(dependency_versions=[{"dr-code", "1"}])


def test_only_pre_execution_rejection_may_omit_evaluation_result() -> None:
    intent = _evaluation_intent(candidate("P1"))
    rejected = IntentResolution(
        schema_version=INTENT_RESOLUTION_SCHEMA_VERSION,
        intent=intent,
        outcome=IntentOutcome.REJECTED,
        detail=ResolutionDetail(
            classification=ResolutionClass.VALIDATION,
            message="bad candidate",
        ),
        resolved_eval_config=intent.target_eval_config,
    )
    assert rejected.evaluation_result_ref is None
    assert rejected.reward_evidence_refs == ()
    with pytest.raises(ValidationError, match="requires an Evaluation Result"):
        IntentResolution(
            schema_version=INTENT_RESOLUTION_SCHEMA_VERSION,
            intent=intent,
            outcome=IntentOutcome.FAILED,
            detail=ResolutionDetail(
                classification=ResolutionClass.UNSCORABLE,
                message="could not score",
            ),
            resolved_eval_config=intent.target_eval_config,
        )


def test_resolution_rejects_a_different_eval_config() -> None:
    intent = _evaluation_intent(candidate("P1"))
    with pytest.raises(ValidationError, match="exact target"):
        IntentResolution(
            schema_version=INTENT_RESOLUTION_SCHEMA_VERSION,
            intent=intent,
            outcome=IntentOutcome.REJECTED,
            detail=ResolutionDetail(
                classification=ResolutionClass.VALIDATION,
                message="rejected",
            ),
            resolved_eval_config=eval_config_reference(eval_config("e" * 64)),
        )


def test_noninitial_request_requires_prior_ref() -> None:
    with pytest.raises(ValidationError, match="prior result"):
        _proposal_request(step_index=1)


@pytest.mark.parametrize(
    "field",
    [
        "prior_step_result_ref",
        "prior_state_ref",
        "prior_history_ref",
    ],
)
def test_initial_request_rejects_all_prior_refs(field: str) -> None:
    prior_ref = typed_ref_for_record(
        (
            STEP_RESULT_SCHEMA
            if field == "prior_step_result_ref"
            else f"whetstone.test.{field}"
        ),
        {"field": field},
    )
    with pytest.raises(ValidationError, match="initial Step Request"):
        OptimizationStepRequest.model_validate(
            {
                **_proposal_request().model_dump(mode="json"),
                field: prior_ref.model_dump(mode="json"),
            }
        )


def test_noninitial_request_requires_a_step_result_schema_ref() -> None:
    wrong_ref = typed_ref_for_record(
        "whetstone.not_a_step_result",
        {"step_id": "previous"},
    )
    with pytest.raises(ValidationError, match="typed Step Result ref"):
        _proposal_request(step_index=1, prior_step_result_ref=wrong_ref)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("run_id", "other-run"),
        ("optimizer_config_hash", "f" * 64),
        ("adapter_key", "other-adapter"),
        ("mode", StepMode.PURE.value),
        (
            "template_render_contract",
            {
                "kind": "literal_body/v1",
                "available_fields": [],
                "required_fields": [],
            },
        ),
        (
            "output_contract",
            {"returned_proposal_count": 99},
        ),
        ("tool_configs", []),
        ("reward_policy", None),
    ],
)
def test_step_request_rejects_removed_run_duplicates(
    field: str, value: Any
) -> None:
    request = _proposal_request()
    with pytest.raises(
        ValidationError, match="Extra inputs are not permitted"
    ):
        OptimizationStepRequest.model_validate(
            {
                **request.model_dump(mode="json"),
                field: value,
            }
        )


def test_step_request_serializes_the_exact_composed_run_reference() -> None:
    request = _proposal_request()
    dumped = request.model_dump(mode="json")

    assert dumped["run"] == request.run.model_dump(mode="json")
    assert dumped["run"]["record"] == request.run.record.record_content()
    assert dumped["run"]["identity_hash"] == request.run.identity_hash
    assert dumped["run"]["record_ref"] == request.run.record_ref.model_dump(
        mode="json"
    )
    assert {
        "run_id",
        "optimizer_config_hash",
        "adapter_key",
        "mode",
        "output_contract",
        "tool_configs",
        "reward_policy",
    }.isdisjoint(dumped)
    assert dumped["step_output_contract"] == {
        "returned_proposal_count": 1,
        "require_distinct_bases": False,
    }
    assert dumped["run"]["record"]["template_render_contract"] == {
        "kind": "python_format/v1",
        "available_fields": ["query"],
        "required_fields": [],
    }


def test_run_requires_and_identity_binds_render_contract() -> None:
    request = _proposal_request()
    payload = request.run.record.model_dump(mode="json")
    payload.pop("template_render_contract")
    with pytest.raises(ValidationError, match="Field required"):
        OptimizationRun.model_validate(payload)

    literal_request = _proposal_request(
        template_render_contract=TemplateRenderContract(
            kind=TemplateRenderKind.LITERAL_BODY_V1,
            available_fields=(),
        )
    )
    assert request.run.identity_hash != literal_request.run.identity_hash
    assert (
        step_request_reference(request).record_ref
        != step_request_reference(literal_request).record_ref
    )
    assert (
        OptimizationStepRequest.model_validate(
            literal_request.model_dump(mode="json")
        )
        == literal_request
    )


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (
            ("record_ref", "schema_name"),
            "whetstone.wrong_run",
            "record_ref must address the exact run",
        ),
        (
            ("identity_hash",),
            "f" * 64,
            "identity_hash must match the exact run",
        ),
    ],
)
def test_step_request_rejects_a_corrupted_run_reference(
    path: tuple[str, ...],
    value: str,
    message: str,
) -> None:
    payload = _proposal_request().model_dump(mode="json")
    target = payload["run"]
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value

    with pytest.raises(ValidationError, match=message):
        OptimizationStepRequest.model_validate(payload)


def test_step_request_properties_derive_from_bound_run() -> None:
    request = _proposal_request()
    run = request.run.record

    assert request.run_id == run.run_id
    assert request.adapter_key == run.adapter_key
    assert request.mode is run.mode
    assert request.tool_configs == run.tool_configs
    assert (
        request.run.record.template_render_contract
        == run.template_render_contract
    )
    assert not hasattr(request, "optimizer_config_hash")
    assert not hasattr(request, "output_contract")
    assert not hasattr(request, "template_render_contract")
    assert {
        "run_id",
        "optimizer_config_hash",
        "adapter_key",
        "mode",
        "output_contract",
        "template_render_contract",
        "tool_configs",
        "reward_policy",
    }.isdisjoint(OptimizationStepRequest.model_fields)


def test_step_request_contract_may_differ_from_terminal_run_contract() -> None:
    request = _proposal_request()
    payload = request.model_dump(mode="json")
    payload["step_output_contract"]["returned_proposal_count"] = 0

    intermediate = OptimizationStepRequest.model_validate(payload)
    assert intermediate.step_output_contract.returned_proposal_count == 0
    assert (
        intermediate.run.record.terminal_output_contract.returned_proposal_count
        == 1
    )


def test_step_request_mode_constraint_comes_from_the_bound_run() -> None:
    request = _proposal_request()
    pure_run = OptimizationRun(
        run_id=request.run_id,
        optimizer_config=request.run.record.optimizer_config,
        adapter_key="identity",
        mode=StepMode.PURE,
        terminal_output_contract=request.run.record.terminal_output_contract,
        template_render_contract=request.run.record.template_render_contract,
    )
    payload = request.model_dump(mode="json")
    payload["run"] = optimization_run_reference(pure_run).model_dump(
        mode="json"
    )

    with pytest.raises(ValidationError, match="pure step"):
        OptimizationStepRequest.model_validate(payload)


def test_optimization_run_owns_tool_mode_constraints() -> None:
    optimizer_config = IdentityRef(
        record_ref=typed_ref_for_record(
            "whetstone.test.optimizer_config",
            {"algorithm": "tool"},
        ),
        identity_hash=FULL_A,
    )
    with pytest.raises(ValidationError, match="requires a Tool Config"):
        OptimizationRun(
            run_id="run-tool",
            optimizer_config=optimizer_config,
            adapter_key="tool-test",
            mode=StepMode.TOOL_USING,
            terminal_output_contract=OutputContract(returned_proposal_count=1),
            template_render_contract=python_format_contract(),
        )
    with pytest.raises(ValidationError, match="only tool-using runs"):
        OptimizationRun(
            run_id="run-proposal",
            optimizer_config=optimizer_config,
            adapter_key="proposal-test",
            mode=StepMode.PROPOSAL_ONLY,
            terminal_output_contract=OutputContract(returned_proposal_count=1),
            template_render_contract=python_format_contract(),
            reward_policy=_reward_policy(),
            tool_configs=(
                tool_config_reference(make_tool_definition_config()),
            ),
        )


def test_optimization_run_owns_proposal_reward_policy() -> None:
    request = _proposal_request()
    payload = request.run.record.model_dump(mode="json")
    payload["reward_policy"] = None
    with pytest.raises(ValidationError, match=r"proposal-only.*requires"):
        OptimizationRun.model_validate(payload)

    pure_payload = payload
    pure_payload["mode"] = StepMode.PURE.value
    pure_payload["adapter_key"] = "identity"
    pure_payload["reward_policy"] = _reward_policy().model_dump(mode="json")
    with pytest.raises(ValidationError, match="only a proposal-only"):
        OptimizationRun.model_validate(pure_payload)


def test_json_fields_defensively_copy_and_deep_freeze() -> None:
    source: dict[str, Any] = {
        "nested": {"enabled": True},
        "items": [1, {"name": "first"}],
    }
    record = candidate()
    record = record.model_validate(
        {
            **record.model_dump(mode="json"),
            "payload": source,
        }
    )
    before_identity = record.identity_hash()
    before_ref = candidate_reference(record).record_ref

    source["nested"]["enabled"] = False
    source_items: Any = source["items"]
    source_items[1]["name"] = "changed"

    frozen_dump = record.model_dump(mode="json")["payload"]
    assert frozen_dump["nested"]["enabled"] is True
    assert frozen_dump["items"][1]["name"] == "first"
    assert record.identity_hash() == before_identity
    assert candidate_reference(record).record_ref == before_ref
    frozen_nested: Any = record.payload["nested"]
    frozen_items: Any = record.payload["items"]
    with pytest.raises(TypeError):
        frozen_nested["enabled"] = False
    with pytest.raises(TypeError):
        frozen_items[1]["name"] = "changed"
    with pytest.raises(AttributeError):
        record.payload._items = ()


def test_json_fields_round_trip_as_ordinary_strict_json() -> None:
    record = candidate()
    dumped = record.model_dump(mode="json")
    assert type(dumped["payload"]) is dict
    assert type(dumped["payload"]["fixed"]) is str
    assert record.model_validate(dumped) == record


def test_json_fields_survive_pickle_round_trips() -> None:
    original = ImmutableJsonObject(
        {
            "nested": {"enabled": True, "depth": {"count": 2}},
            "items": [1, 2.5, "three", None, {"name": "first"}],
            "flag": False,
        }
    )

    restored = pickle.loads(pickle.dumps(original))

    assert type(restored) is ImmutableJsonObject
    assert restored == original
    assert restored.to_json() == original.to_json()
    restored_nested: Any = restored["nested"]
    restored_items: Any = restored["items"]
    assert type(restored_nested) is ImmutableJsonObject
    assert type(restored_items) is tuple
    with pytest.raises(TypeError):
        restored_nested["enabled"] = False
    with pytest.raises(AttributeError):
        restored._items = ()


def test_records_carrying_json_fields_survive_pickle_round_trips() -> None:
    record = candidate()
    record = record.model_validate(
        {
            **record.model_dump(mode="json"),
            "payload": {"nested": {"enabled": True}, "items": [1, "two"]},
        }
    )

    restored = pickle.loads(pickle.dumps(record))

    assert restored == record
    assert restored.identity_hash() == record.identity_hash()
    assert restored.payload.to_json() == record.payload.to_json()


@pytest.mark.parametrize(
    "payload",
    [
        {"bad": object()},
        {1: "non-string key"},
        {"bad": float("nan")},
        {"bad": float("inf")},
    ],
)
def test_json_fields_reject_non_json_and_nonfinite_values(payload) -> None:
    with pytest.raises((TypeError, ValidationError, ValueError)):
        candidate().model_validate(
            {
                **candidate().model_dump(mode="json"),
                "payload": payload,
            }
        )


def test_canonical_json_comparison_preserves_json_types() -> None:
    assert not canonical_json_equal({"value": True}, {"value": 1})
    assert not canonical_json_equal({"value": 1}, {"value": 1.0})
    assert canonical_json_equal(
        {"nested": [{"value": 1}]},
        {"nested": [{"value": 1}]},
    )


def test_budget_validates_overlapping_maps_independently() -> None:
    with pytest.raises(ValidationError, match=r"consumed.*cannot be negative"):
        BudgetState(
            consumed={"rollouts": -1},
            remaining={"rollouts": 10},
        )
    with pytest.raises(ValidationError, match="strict integer"):
        BudgetState(remaining={"rollouts": True})


def test_optimization_run_composes_exact_contract_refs() -> None:
    optimizer_config = IdentityRef(
        record_ref=typed_ref_for_record(
            "whetstone.test.optimizer_config",
            {"algorithm": "identity"},
        ),
        identity_hash="a" * 64,
    )
    run = OptimizationRun(
        run_id="run-1",
        optimizer_config=optimizer_config,
        adapter_key="identity",
        mode=StepMode.PURE,
        terminal_output_contract=OutputContract(returned_proposal_count=1),
        template_render_contract=python_format_contract(),
    )
    run_ref = optimization_run_reference(run)
    assert run_ref.record == run
    assert run_ref.identity_hash == run.identity_hash()
    assert run_ref.record_ref.schema_name == "whetstone.optimization_run"
    assert OPTIMIZATION_RUN_SCHEMA == "whetstone.optimization_run"
    assert OPTIMIZATION_RUN_SCHEMA_VERSION == 1
    assert tuple(run.identity_payload()) == (
        "run_id",
        "optimizer_config",
        "adapter_key",
        "mode",
        "terminal_output_contract",
        "template_render_contract",
        "reward_policy",
        "tool_configs",
    )
    assert (
        run.identity_hash()
        == "a1da7e6360588016d7e5d00ab145c8077a2ffbce099f7db7501aeaf554f8d044"
    )


def test_generic_failed_status_requires_shared_terminal_failure() -> None:
    request = _proposal_request()
    with pytest.raises(ValidationError, match="shared terminal failure"):
        OptimizationStepResult(
            request=step_request_reference(request),
            status=StepStatus.FAILED,
        )
    failed = OptimizationStepResult(
        request=step_request_reference(request),
        status=StepStatus.FAILED,
        terminal_failure=TerminalFailure(code="provider", message="failed"),
        budget=request.budget,
    )
    assert failed.terminal_failure is not None
    assert failed.terminal_failure.code == "provider"


def test_step_request_ref_binds_exact_record_content() -> None:
    request = _proposal_request()
    exact = step_request_reference(request)

    assert exact.record == request
    assert exact.record_ref.schema_name != STEP_RESULT_SCHEMA
    with pytest.raises(ValidationError, match="exact request"):
        OptimizationStepRequestRef(
            record=request,
            record_ref=TypedRef(
                schema_name=exact.record_ref.schema_name,
                content_hash="f" * 64,
            ),
        )


def test_step_result_composes_request_and_derives_identity_fields() -> None:
    request = _proposal_request()
    result = _step_result(request)
    dumped = result.model_dump(mode="json")

    assert result.run_id == request.run_id
    assert result.step_id == request.step_id
    assert result.step_index == request.step_index
    assert result.request_ref == result.request.record_ref
    assert dumped["request"] == result.request.model_dump(mode="json")
    assert {
        "run_id",
        "step_id",
        "step_index",
        "request_ref",
    }.isdisjoint(dumped)
    with pytest.raises(
        ValidationError, match="Extra inputs are not permitted"
    ):
        OptimizationStepResult.model_validate(
            {
                **dumped,
                "run_id": "contradictory-run",
                "step_id": "contradictory-step",
                "step_index": 99,
                "request_ref": typed_ref_for_record(
                    "whetstone.optimization_step_request", {}
                ).model_dump(mode="json"),
            }
        )


@pytest.mark.parametrize("field", ("run_id", "step_id"))
def test_step_result_rejects_empty_composed_identifiers(field: str) -> None:
    payload = _step_result(_proposal_request()).model_dump(mode="json")
    request = payload["request"]["record"]
    if field == "run_id":
        request["run"]["record"]["run_id"] = ""
    else:
        request["step_id"] = ""

    with pytest.raises(ValidationError, match="must be non-empty"):
        OptimizationStepResult.model_validate(payload)


def test_step_result_ref_binds_exact_record_content() -> None:
    result = _step_result(_proposal_request())
    exact = step_result_reference(result)

    assert exact.record == result
    assert exact.record_ref.schema_name == STEP_RESULT_SCHEMA
    with pytest.raises(ValidationError, match="exact result"):
        OptimizationStepResultRef(
            record=result,
            record_ref=TypedRef(
                schema_name=STEP_RESULT_SCHEMA,
                content_hash="f" * 64,
            ),
        )


def test_terminal_result_composes_contiguous_exact_history() -> None:
    state_ref = typed_ref_for_record("whetstone.test.state", {"step": 0})
    history_ref = typed_ref_for_record("whetstone.test.history", {"step": 0})
    first_request = _proposal_request()
    first = _step_result(
        first_request,
        status=StepStatus.CONTINUE,
        state_ref=state_ref,
        history_ref=history_ref,
    )
    first_ref = step_result_reference(first)
    final_request = _proposal_request(
        step_index=1,
        prior_step_result_ref=first_ref.record_ref,
        prior_state_ref=state_ref,
        prior_history_ref=history_ref,
        budget=first.budget,
    )
    final = _step_result(final_request)
    final_ref = step_result_reference(final)
    terminal = OptimizationResult(
        run=first_request.run,
        proposals=tuple(
            OptimizationProposal(candidate=accepted)
            for accepted in final.accepted_candidates
        ),
        step_results=(first_ref, final_ref),
    )

    assert terminal.run_id == first_request.run_id
    assert terminal.status is StepStatus.COMPLETE
    assert terminal.step_result_refs == (
        first_ref.record_ref,
        final_ref.record_ref,
    )
    assert {
        "run_id",
        "status",
        "step_result_refs",
    }.isdisjoint(terminal.model_dump(mode="json"))


def test_terminal_result_rejects_hostile_contradictory_exact_records() -> None:
    first_request = _proposal_request()
    first = _step_result(first_request, status=StepStatus.CONTINUE)
    first_ref = step_result_reference(first)
    fake_prior = typed_ref_for_record(STEP_RESULT_SCHEMA, {"other": "result"})
    final_request = _proposal_request(
        step_index=1,
        prior_step_result_ref=fake_prior,
        budget=first.budget,
    )
    final = _step_result(final_request)
    final_ref = step_result_reference(final)
    proposals = tuple(
        OptimizationProposal(candidate=accepted)
        for accepted in final.accepted_candidates
    )

    with pytest.raises(ValidationError, match="prior exact result"):
        OptimizationResult(
            run=first_request.run,
            proposals=proposals,
            step_results=(first_ref, final_ref),
        )
    with pytest.raises(ValidationError, match="exact run"):
        OptimizationResult(
            run=_proposal_request(run_id="another-run").run,
            proposals=tuple(
                OptimizationProposal(candidate=accepted)
                for accepted in first.accepted_candidates
            ),
            step_results=(step_result_reference(_step_result(first_request)),),
        )


def test_terminal_result_rejects_noncontiguous_or_early_terminal_steps() -> (
    None
):
    first_request = _proposal_request()
    continuing = _step_result(first_request, status=StepStatus.CONTINUE)
    continuing_ref = step_result_reference(continuing)
    later_request = _proposal_request(
        step_index=2,
        prior_step_result_ref=continuing_ref.record_ref,
        budget=continuing.budget,
    )
    later = _step_result(later_request)
    later_ref = step_result_reference(later)

    with pytest.raises(ValidationError, match="contiguous from zero"):
        OptimizationResult(
            run=first_request.run,
            proposals=tuple(
                OptimizationProposal(candidate=accepted)
                for accepted in later.accepted_candidates
            ),
            step_results=(continuing_ref, later_ref),
        )

    early_terminal = _step_result(first_request)
    early_ref = step_result_reference(early_terminal)
    contiguous_request = _proposal_request(
        step_index=1,
        prior_step_result_ref=early_ref.record_ref,
        budget=early_terminal.budget,
    )
    contiguous = _step_result(contiguous_request)
    with pytest.raises(ValidationError, match="only the final"):
        OptimizationResult(
            run=first_request.run,
            proposals=tuple(
                OptimizationProposal(candidate=accepted)
                for accepted in contiguous.accepted_candidates
            ),
            step_results=(early_ref, step_result_reference(contiguous)),
        )


def test_terminal_result_derives_exact_proposals_and_shared_failure() -> None:
    request = _proposal_request()
    complete = _step_result(request)
    with pytest.raises(ValidationError, match="exactly derive"):
        OptimizationResult(
            run=request.run,
            proposals=(),
            step_results=(step_result_reference(complete),),
        )

    failure = TerminalFailure(code="provider", message="failed")
    failed = _step_result(
        request,
        status=StepStatus.FAILED,
        terminal_failure=failure,
    )
    terminal = OptimizationResult(
        run=request.run,
        proposals=(),
        step_results=(step_result_reference(failed),),
        terminal_failure=failure,
    )
    assert terminal.status is StepStatus.FAILED
    assert terminal.terminal_failure == failed.terminal_failure
    with pytest.raises(ValidationError, match="match the final"):
        OptimizationResult(
            run=request.run,
            proposals=(),
            step_results=(step_result_reference(failed),),
            terminal_failure=TerminalFailure(
                code="different",
                message="different",
            ),
        )


def test_tool_chain_is_exact_and_terminal_variants_are_exclusive() -> None:
    config = make_tool_definition_config()
    config_ref = tool_config_reference(config)
    capacity_binding = tool_capacity_binding(
        ToolCapacityScope.RUN,
        tool_run().record_ref,
    )
    call = ToolCall(
        call_id="call-1",
        tool_config=config_ref,
        capacity_binding=capacity_binding,
        args={"model_route": "r0", "template": "prompt"},
    )
    call_ref = tool_call_reference(call)
    success = ToolResult(
        call=call_ref,
        output={"rollout_refs": [], "accepted_ordinal": 1},
        provenance_ordinal=1,
    )
    assert tool_result_reference(success).record == success
    refused = ToolResult(
        call=call_ref,
        refusal=ToolRefusal(
            refusal_class="validation",
            reason="bad template",
        ),
    )
    assert refused.output is None
    failed = ToolResult(
        call=call_ref,
        terminal_failure=TerminalFailure(
            code="evaluation_failed",
            message="provider exhausted",
        ),
        provenance_ordinal=1,
    )
    assert failed.terminal_failure is not None
    with pytest.raises(ValidationError, match="exactly success"):
        ToolResult(call=call_ref)
    with pytest.raises(ValidationError, match="exactly success"):
        ToolResult(
            call=call_ref,
            output={"rollout_refs": [], "accepted_ordinal": 1},
            refusal={"refusal_class": "capacity", "reason": "full"},
        )


def test_tool_definition_config_call_and_result_cannot_diverge() -> None:
    config = make_tool_definition_config()
    capacity_binding = tool_capacity_binding(
        ToolCapacityScope.RUN,
        tool_run().record_ref,
    )
    with pytest.raises(ValidationError, match="identity_hash"):
        ToolDefinitionRef(
            record=config.definition.record,
            record_ref=config.definition.record_ref,
            identity_hash="a" * 64,
        )
    with pytest.raises(ValidationError, match="input_fields"):
        ToolCall(
            call_id="call-1",
            tool_config=tool_config_reference(config),
            capacity_binding=capacity_binding,
            args={"template": "missing model route"},
        )
    with pytest.raises(ValidationError, match="ID must be non-empty"):
        ToolCall(
            call_id="",
            tool_config=tool_config_reference(config),
            capacity_binding=capacity_binding,
            args={"model_route": "r0", "template": "prompt"},
        )
    with pytest.raises(ValidationError, match="extra"):
        ToolCall.model_validate(
            {
                "call_id": "call-1",
                "tool_config": tool_config_reference(config).model_dump(
                    mode="json"
                ),
                "capacity_binding": capacity_binding.model_dump(mode="json"),
                "capacity_scope_id": "run-1",
                "args": {"model_route": "r0", "template": "prompt"},
            }
        )
    call = ToolCall(
        call_id="call-1",
        tool_config=tool_config_reference(config),
        capacity_binding=capacity_binding,
        args={"model_route": "r0", "template": "prompt"},
    )
    with pytest.raises(ValidationError, match="output_fields"):
        ToolResult(
            call=tool_call_reference(call),
            output={"rollout_refs": []},
        )
