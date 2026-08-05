from __future__ import annotations

from copy import deepcopy
from typing import cast

import pytest

from tests.optimization.support import (
    FULL_A,
    FULL_B,
    FULL_C,
    RecordingEvaluationService,
    evaluation_binding,
    internal_reward_policy,
    make_harness,
    make_store,
)
from whetstone.evaluation_role import EvaluationRole
from whetstone.lm.boundary import PlainPromptAdapter
from whetstone.optimization import (
    EVALUATION_EVIDENCE_SCHEMA,
    INTENT_RESOLUTION_SCHEMA_VERSION,
    AdapterReplayPolicyMismatchError,
    BudgetState,
    Candidate,
    CoproAdapter,
    CoproInjectedDefaults,
    EvaluationIntent,
    FakeProposerTransport,
    IdentityRef,
    IntentOutcome,
    IntentResolution,
    MappingAdapterRegistry,
    OptimizationRun,
    OptimizationRunRef,
    OptimizationStepRequest,
    OutputContract,
    ProposerConfig,
    ReplayPolicy,
    ResolutionClass,
    ResolutionDetail,
    StepKind,
    StepMode,
    StepStatus,
    TemplateRenderContract,
    TemplateRenderKind,
    apply_reward_policy,
    candidate_reference,
    configure_copro,
    optimization_run_reference,
    reward_reference,
    typed_ref_for_record,
)
from whetstone.optimization.copro import (
    HISTORY_PROPOSAL,
    SEED_PROPOSAL,
    CoproAttempt,
    CoproConfig,
    CoproDriver,
    CoproState,
    rank_attempt_history,
)
from whetstone.optimization.proposer import (
    DurableProposalExecutor,
    ProposalExecutorDurabilityContract,
    _durable_proposal_executor,
)
from whetstone.optimization.schema import STEP_RESULT_SCHEMA


def _direct_executor(
    *, policy_identity_hash: str = FULL_C
) -> DurableProposalExecutor:
    """Mint the canonical capability over an in-process pass-through."""

    def execute(*, config, request, transport, count):
        return transport.draft(config, request, count)

    return _durable_proposal_executor(
        durability_contract=ProposalExecutorDurabilityContract(
            recovery_policy=ReplayPolicy.DURABLE_WORKFLOW,
            policy_identity_hash=policy_identity_hash,
        ),
        execute=execute,
    )


def _prompt_model(*, temperature: float = 1.4) -> ProposerConfig:
    return ProposerConfig(
        provider_call_config=IdentityRef(
            record_ref=typed_ref_for_record(
                "dr_providers.provider_call_config",
                {"route": "copro-proposer"},
            ),
            identity_hash=FULL_A,
        ),
        temperature=temperature,
    )


def _control(
    *,
    breadth: int = 3,
    depth: int = 1,
    track_stats: bool = False,
):
    policy = internal_reward_policy()
    return configure_copro(
        breadth=breadth,
        depth=depth,
        track_stats=track_stats,
        defaults=CoproInjectedDefaults(
            prompt_model=_prompt_model(),
            evaluation_binding=evaluation_binding(),
            expected_reward_policy_hash=policy.identity_hash(),
            provider_execution_policy_hash=FULL_A,
            prompt_adapter=PlainPromptAdapter(),
        ),
    )


def _adapter(
    script: dict[tuple[str, int], tuple[str, ...]],
    *,
    control=None,
):
    exact_control = control or _control()
    transport = FakeProposerTransport(
        script,
        execution_policy_hash=FULL_A,
        prompt_adapter_identity_hash=exact_control.prompt_adapter_identity_hash,
    )
    return (
        CoproAdapter(
            control=exact_control,
            transport=transport,
            proposal_executor=_direct_executor(),
        ),
        transport,
        exact_control,
    )


def _candidate(cid: str, text: str, *, parent: str = "root") -> Candidate:
    return Candidate(
        candidate_id=cid,
        base_ref=typed_ref_for_record(
            "test.copro_candidate_parent", {"id": parent}
        ),
        payload={"user_prompt_template": text, "fixed": "unchanged"},
    )


def _run(control) -> OptimizationRunRef:
    return optimization_run_reference(
        OptimizationRun(
            run_id="copro-run",
            optimizer_config=control.reference(),
            adapter_key="copro",
            mode=StepMode.PROPOSAL_ONLY,
            terminal_output_contract=OutputContract(returned_proposal_count=1),
            template_render_contract=TemplateRenderContract(
                kind=TemplateRenderKind.PYTHON_FORMAT_V1,
                available_fields=("input",),
                required_fields=("input",),
            ),
            reward_policy=internal_reward_policy(),
        )
    )


def _request(
    control,
    *,
    step_index: int = 0,
    candidates: tuple[Candidate, ...] | None = None,
    history: list[dict[str, object]] | None = None,
    proposal_budget: int | None = None,
) -> OptimizationStepRequest:
    accepted_count = (
        control.breadth - 1 if step_index == 0 else control.breadth
    )
    return OptimizationStepRequest(
        run=_run(control),
        step_id=f"copro-{step_index}",
        kind=StepKind.PROPOSAL,
        step_index=step_index,
        prior_step_result_ref=(
            None
            if step_index == 0
            else typed_ref_for_record(
                STEP_RESULT_SCHEMA, {"step": step_index - 1}
            )
        ),
        candidates=candidates or (_candidate("baseline", "base {input}"),),
        pools={"attempt_history": history or []},
        hyperparameters=control.step_hyperparameters(iteration=step_index),
        budget=BudgetState(
            remaining={
                "proposal_calls": (
                    proposal_budget
                    if proposal_budget is not None
                    else control.breadth
                )
            }
        ),
        step_output_contract=OutputContract(
            returned_proposal_count=accepted_count
        ),
    )


def _entry(
    control,
    occurrence_ordinal: int,
    cid: str,
    template: str,
    reward_value: float,
) -> dict[str, object]:
    evidence_ref = typed_ref_for_record(
        "test.copro_evidence",
        {"occurrence_ordinal": occurrence_ordinal},
    )
    evaluation_result_ref = typed_ref_for_record(
        EVALUATION_EVIDENCE_SCHEMA,
        {"occurrence_ordinal": occurrence_ordinal},
    )
    reward_ref = reward_reference(
        apply_reward_policy(
            internal_reward_policy(),
            aggregates={"score": reward_value},
            evidence_role=EvaluationRole.INTERNAL,
            evidence_refs=(evidence_ref,),
        )
    )
    return CoproAttempt(
        occurrence_ordinal=occurrence_ordinal,
        round_index=occurrence_ordinal // control.breadth,
        run_id="copro-run",
        step_index=occurrence_ordinal // control.breadth,
        intent_id=f"intent-{occurrence_ordinal}",
        candidate=candidate_reference(_candidate(cid, template)),
        evaluation_binding=control.evaluation_binding,
        reward=reward_value,
        expected_reward_policy_hash=control.expected_reward_policy_hash,
        evaluation_result_ref=evaluation_result_ref,
        reward_evidence_refs=(evidence_ref,),
        reward_ref=reward_ref,
    ).model_dump(mode="json")


def test_public_hyperparameter_defaults_match_dspy() -> None:
    config = CoproConfig()

    assert (config.breadth, config.depth, config.init_temperature) == (
        10,
        3,
        1.4,
    )
    assert config.track_stats is False
    with pytest.raises(ValueError, match="greater than 1"):
        CoproConfig(breadth=1)


def test_seed_round_proposes_breadth_minus_one_and_evaluates_exact_base() -> (
    None
):
    adapter, transport, control = _adapter(
        {(SEED_PROPOSAL, 0): ('"new {input}"', "other {input}")}
    )
    request = _request(control)

    output = adapter.invoke(request, ())

    assert output.proposed_status is StepStatus.CONTINUE
    assert [
        item.payload["user_prompt_template"]
        for item in output.proposed_candidates
    ] == [
        "new {input}",
        "other {input}",
    ]
    assert output.accepted_candidates == output.proposed_candidates
    assert len(output.evaluation_intents) == 3
    assert output.evaluation_intents[-1].candidate == candidate_reference(
        request.candidates[0]
    )
    exact_base_ref = candidate_reference(request.candidates[0]).record_ref
    assert all(
        candidate.base_ref == exact_base_ref
        for candidate in output.proposed_candidates
    )
    assert transport.calls[0][1].base_candidate == candidate_reference(
        request.candidates[0]
    )
    assert transport.calls[0][2] == 2


def test_adapter_requires_the_executor_durable_workflow_replay() -> None:
    adapter, _, _ = _adapter({})

    assert adapter.required_replay_policy is ReplayPolicy.DURABLE_WORKFLOW
    assert (
        adapter.required_replay_policy
        is adapter.proposal_executor.recovery_policy
    )
    assert adapter.proposal_executor.policy_identity_hash == FULL_C


def test_adapter_rejects_a_structural_proposal_executor() -> None:
    class _StructuralExecutor:
        policy_identity_hash = FULL_C
        recovery_policy = ReplayPolicy.DURABLE_WORKFLOW

        def execute(self, **_kwargs):
            raise AssertionError("test does not execute proposal effects")

    control = _control()
    transport = FakeProposerTransport(
        {},
        execution_policy_hash=FULL_A,
        prompt_adapter_identity_hash=control.prompt_adapter_identity_hash,
    )

    with pytest.raises(TypeError, match="DurableProposalExecutor"):
        CoproAdapter(
            control=control,
            transport=transport,
            proposal_executor=cast(
                DurableProposalExecutor, _StructuralExecutor()
            ),
        )


def test_idempotent_harness_rejects_before_copro_effects(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter, transport, control = _adapter(
        {(SEED_PROPOSAL, 0): ("new {input}", "other {input}")}
    )
    store = make_store(tmp_path)
    service = RecordingEvaluationService(
        store, reward_policy=internal_reward_policy()
    )
    harness = make_harness(
        store=store,
        adapter_registry=MappingAdapterRegistry({"copro": adapter}),
        run=_run(control),
        evaluation_service=service,
        adapter_replay_policy=ReplayPolicy.IDEMPOTENT,
    )
    puts = 0
    real_put = harness._put

    def record_put(*args, **kwargs):
        nonlocal puts
        puts += 1
        return real_put(*args, **kwargs)

    monkeypatch.setattr(harness, "_put", record_put)

    with pytest.raises(AdapterReplayPolicyMismatchError) as caught:
        harness.run_step(_request(control))

    assert caught.value.configured_policy is ReplayPolicy.IDEMPOTENT
    assert caught.value.required_policy is ReplayPolicy.DURABLE_WORKFLOW
    assert adapter.invocations == 0
    assert transport.calls == []
    assert service.calls == []
    assert puts == 0


def test_durable_workflow_harness_reaches_copro_and_replays_result(
    tmp_path,
) -> None:
    adapter, transport, control = _adapter(
        {(SEED_PROPOSAL, 0): ("new {input}", "other {input}")}
    )
    store = make_store(tmp_path)
    service = RecordingEvaluationService(
        store, reward_policy=internal_reward_policy()
    )
    harness = make_harness(
        store=store,
        adapter_registry=MappingAdapterRegistry({"copro": adapter}),
        run=_run(control),
        evaluation_service=service,
        adapter_replay_policy=ReplayPolicy.DURABLE_WORKFLOW,
    )
    request = _request(control)

    result, result_ref = harness.run_step(request)

    assert len(result.proposed_candidates) == 2
    assert len(result.accepted_candidates) == 2
    assert len(result.resolved_intents) == 3
    assert len(service.calls) == 3
    assert adapter.invocations == 1
    assert len(transport.calls) == 1

    assert harness.run_step(request) == (result, result_ref)
    assert adapter.invocations == 1
    assert len(transport.calls) == 1
    assert len(service.calls) == 3


def test_history_uses_top_unique_attempts_and_immutable_prompt_context() -> (
    None
):
    control = _control(depth=3)
    history = [
        _entry(control, 0, "a-old", "a {input}", 0.7),
        _entry(control, 1, "b", "b {input}", 0.9),
        _entry(control, 2, "c", "c {input}", 0.8),
        _entry(control, 3, "a-new", "a {input}", 0.9),
        _entry(control, 4, "d", "d {input}", 0.95),
        _entry(control, 5, "e", "e {input}", 0.1),
    ]
    original = deepcopy(history)
    adapter, transport, _ = _adapter(
        {(HISTORY_PROPOSAL, 2): ("x {input}", "y {input}", "z {input}")},
        control=control,
    )

    output = adapter.invoke(
        _request(control, step_index=2, history=history), ()
    )

    assert history == original
    request = transport.calls[0][1]
    assert [
        item["candidate_id"] for item in request.context["prompt_history"]
    ] == [
        "b",
        "a-new",
        "d",
    ]
    assert "Instruction #3: d {input}" in str(
        request.context["proposal_prompt"]
    )
    assert len(output.evaluation_intents) == 3


def test_duplicate_templates_are_evaluated_before_history_deduplication() -> (
    None
):
    adapter, _, control = _adapter(
        {(SEED_PROPOSAL, 0): ("duplicate {input}", "duplicate {input}")}
    )

    output = adapter.invoke(_request(control), ())

    assert len(output.evaluation_intents) == 3
    assert [
        item.payload["user_prompt_template"]
        for item in output.proposed_candidates
    ] == ["duplicate {input}", "duplicate {input}"]


def test_driver_owns_round_counts_ranking_and_statistics() -> None:
    control = _control(depth=2, track_stats=True)
    driver = CoproDriver(CoproConfig(breadth=3, depth=2, track_stats=True))
    initial = _candidate("baseline", "base {input}")
    first = tuple(
        CoproAttempt.model_validate(item)
        for item in (
            _entry(control, 0, "x", "x {input}", 0.2),
            _entry(control, 1, "y", "y {input}", 0.6),
            _entry(control, 2, "baseline", "base {input}", 0.4),
        )
    )
    second = tuple(
        CoproAttempt.model_validate(item)
        for item in (
            _entry(control, 3, "x2", "x {input}", 0.7),
            _entry(control, 4, "z", "z {input}", 0.9),
            _entry(control, 5, "w", "w {input}", 0.8),
        )
    )

    restored = driver.restore_state(initial_candidate=initial, attempts=first)
    completed = driver.fold_round(restored, second)
    final = driver.finalize(completed)

    assert [
        item.candidate_id for item in rank_attempt_history(first + second)
    ] == [
        "z",
        "w",
        "x2",
        "y",
        "baseline",
    ]
    assert final.total_calls == 6
    assert final.statistics is not None
    assert final.statistics.results_latest.average[0] == pytest.approx(0.4)
    forged = CoproState(
        initial_candidate=initial,
        completed_rounds=2,
        attempts=(),
        total_calls=0,
    )
    with pytest.raises(ValueError, match="occurrence history"):
        driver.finalize(forged)


def test_attempt_folds_exact_reward_ref_and_evaluation_binding() -> None:
    control = _control()
    evaluation_result_ref = typed_ref_for_record(
        EVALUATION_EVIDENCE_SCHEMA, {"candidate": "a"}
    )
    reward_evidence_refs = (
        typed_ref_for_record("test.aggregate", {"name": "first"}),
        typed_ref_for_record("test.aggregate", {"name": "second"}),
    )
    reward_ref = reward_reference(
        apply_reward_policy(
            internal_reward_policy(),
            aggregates={"score": 0.75},
            evidence_role=EvaluationRole.INTERNAL,
            evidence_refs=reward_evidence_refs,
        )
    )
    intent = EvaluationIntent(
        intent_id="copro-run:0:0",
        candidate=candidate_reference(_candidate("a", "a {input}")),
        target_eval_config=control.evaluation_binding.eval_config,
        evaluation_binding=control.evaluation_binding,
        purpose=SEED_PROPOSAL,
        run_id="copro-run",
        step_index=0,
        expected_reward_policy_hash=control.expected_reward_policy_hash,
    )
    resolution = IntentResolution(
        schema_version=INTENT_RESOLUTION_SCHEMA_VERSION,
        intent=intent,
        outcome=IntentOutcome.COMPLETED,
        detail=ResolutionDetail(
            classification=ResolutionClass.MEASURED,
            message="measured",
        ),
        evaluation_result_ref=evaluation_result_ref,
        reward_evidence_refs=reward_evidence_refs,
        resolved_eval_config=control.evaluation_binding.eval_config,
        reward_ref=reward_ref,
    )

    attempt = CoproAttempt.from_resolution(
        occurrence_ordinal=0,
        round_index=0,
        resolution=resolution,
        expected_run_id="copro-run",
        expected_evaluation_binding=control.evaluation_binding,
        expected_reward_policy_hash=control.expected_reward_policy_hash,
    )

    assert attempt.reward_ref == reward_ref
    assert attempt.evaluation_binding == control.evaluation_binding
    assert attempt.evaluation_result_ref == evaluation_result_ref
    assert attempt.reward_evidence_refs == reward_evidence_refs
    assert attempt.evaluation_result_ref not in attempt.reward_evidence_refs
    with pytest.raises(ValueError, match="expects an unexpected"):
        CoproAttempt.from_resolution(
            occurrence_ordinal=0,
            round_index=0,
            resolution=resolution,
            expected_run_id="copro-run",
            expected_evaluation_binding=control.evaluation_binding,
            expected_reward_policy_hash=FULL_B,
        )

    forged_result_ref = evaluation_result_ref.model_copy(
        update={"content_hash": "not-a-hash"}
    )
    forged_resolution = resolution.model_copy(
        update={"evaluation_result_ref": forged_result_ref}
    )
    with pytest.raises(ValueError, match="content hash must be a full"):
        CoproAttempt.from_resolution(
            occurrence_ordinal=0,
            round_index=0,
            resolution=forged_resolution,
            expected_run_id="copro-run",
            expected_evaluation_binding=control.evaluation_binding,
            expected_reward_policy_hash=control.expected_reward_policy_hash,
        )

    forged_reward_ref = reward_ref.model_copy(
        update={
            "record_ref": reward_ref.record_ref.model_copy(
                update={"content_hash": "not-a-hash"}
            )
        }
    )
    forged_resolution = resolution.model_copy(
        update={"reward_ref": forged_reward_ref}
    )
    with pytest.raises(ValueError, match="content hash must be a full"):
        CoproAttempt.from_resolution(
            occurrence_ordinal=0,
            round_index=0,
            resolution=forged_resolution,
            expected_run_id="copro-run",
            expected_evaluation_binding=control.evaluation_binding,
            expected_reward_policy_hash=control.expected_reward_policy_hash,
        )


def test_attempt_wire_pins_separate_result_and_ordered_reward_refs() -> None:
    control = _control()
    attempt = CoproAttempt.model_validate(
        _entry(control, 0, "a", "a {input}", 0.75)
    )
    record = attempt.model_dump(mode="json")

    assert tuple(record) == (
        "occurrence_ordinal",
        "round_index",
        "run_id",
        "step_index",
        "intent_id",
        "candidate",
        "evaluation_binding",
        "reward",
        "expected_reward_policy_hash",
        "evaluation_result_ref",
        "reward_evidence_refs",
        "reward_ref",
    )
    assert record["evaluation_result_ref"]["schema_name"] == (
        EVALUATION_EVIDENCE_SCHEMA
    )
    assert len(record["reward_evidence_refs"]) == 1
    assert typed_ref_for_record(
        "test.copro_attempt_wire", record
    ).content_hash == (
        "6d6ace1d345e79006ba8d9096b390eb4621e75b06e54967f75916bf9d13371da"
    )


def test_attempt_replay_rejects_missing_or_mismatched_result_ref() -> None:
    control = _control()
    record = _entry(control, 0, "a", "a {input}", 0.75)

    missing = dict(record)
    missing.pop("evaluation_result_ref")
    with pytest.raises(ValueError, match="Field required"):
        CoproAttempt.model_validate(missing)

    mismatched = dict(record)
    mismatched["evaluation_result_ref"] = typed_ref_for_record(
        "test.aggregate", {"name": "not-an-evaluation-result"}
    ).model_dump(mode="json")
    with pytest.raises(ValueError, match="evaluation_result_ref must use"):
        CoproAttempt.model_validate(mismatched)


def test_attempt_replay_preserves_reward_citation_order() -> None:
    control = _control()
    first = typed_ref_for_record("test.aggregate", {"name": "first"})
    second = typed_ref_for_record("test.aggregate", {"name": "second"})
    reward_ref = reward_reference(
        apply_reward_policy(
            internal_reward_policy(),
            aggregates={"score": 0.75},
            evidence_role=EvaluationRole.INTERNAL,
            evidence_refs=(first, second),
        )
    )
    record = _entry(control, 0, "a", "a {input}", 0.75)
    record["reward_ref"] = reward_ref.model_dump(mode="json")
    record["reward_evidence_refs"] = [
        second.model_dump(mode="json"),
        first.model_dump(mode="json"),
    ]

    with pytest.raises(ValueError, match="citations must match"):
        CoproAttempt.model_validate(record)

    record["reward_evidence_refs"] = {
        first,
        second,
    }
    with pytest.raises(ValueError, match="must be an ordered"):
        CoproAttempt.model_validate(record)


def test_completed_resolution_requires_exact_primary_result() -> None:
    control = _control()
    intent = EvaluationIntent(
        intent_id="copro-run:0:0",
        candidate=candidate_reference(_candidate("a", "a {input}")),
        target_eval_config=control.evaluation_binding.eval_config,
        evaluation_binding=control.evaluation_binding,
        purpose=SEED_PROPOSAL,
        run_id="copro-run",
        step_index=0,
        expected_reward_policy_hash=control.expected_reward_policy_hash,
    )
    reward_evidence_ref = typed_ref_for_record(
        "test.aggregate", {"name": "score"}
    )
    reward_ref = reward_reference(
        apply_reward_policy(
            internal_reward_policy(),
            aggregates={"score": 0.75},
            evidence_role=EvaluationRole.INTERNAL,
            evidence_refs=(reward_evidence_ref,),
        )
    )
    resolution = IntentResolution(
        schema_version=INTENT_RESOLUTION_SCHEMA_VERSION,
        intent=intent,
        outcome=IntentOutcome.COMPLETED,
        detail=ResolutionDetail(
            classification=ResolutionClass.MEASURED,
            message="measured",
        ),
        evaluation_result_ref=typed_ref_for_record(
            EVALUATION_EVIDENCE_SCHEMA, {"candidate": "a"}
        ),
        reward_evidence_refs=(reward_evidence_ref,),
        resolved_eval_config=control.evaluation_binding.eval_config,
        reward_ref=reward_ref,
    )
    payload = resolution.model_dump(mode="json")

    payload["evaluation_result_ref"] = None
    with pytest.raises(ValueError, match="requires an Evaluation Result"):
        IntentResolution.model_validate(payload)

    payload["evaluation_result_ref"] = reward_evidence_ref.model_dump(
        mode="json"
    )
    with pytest.raises(ValueError, match="evaluation_result_ref must use"):
        IntentResolution.model_validate(payload)


def test_exact_control_and_transport_are_verified_before_effects() -> None:
    adapter, transport, control = _adapter({})
    other = _control(track_stats=True)
    mismatched_run = _request(control).model_copy(update={"run": _run(other)})

    with pytest.raises(ValueError, match="exact control"):
        adapter.invoke(mismatched_run, ())

    assert transport.calls == []
    wrong_policy = FakeProposerTransport(
        {},
        execution_policy_hash=FULL_B,
        prompt_adapter_identity_hash=control.prompt_adapter_identity_hash,
    )
    wrong_adapter = CoproAdapter(
        control=control,
        transport=wrong_policy,
        proposal_executor=_direct_executor(),
    )
    with pytest.raises(ValueError, match="execution policy"):
        wrong_adapter.invoke(_request(control), ())
    assert wrong_policy.calls == []


def test_render_contract_rejection_and_underfill_are_terminal_failures() -> (
    None
):
    adapter, _, control = _adapter(
        {(SEED_PROPOSAL, 0): ("missing field", "valid {input}")}
    )

    output = adapter.invoke(_request(control), ())

    assert output.proposed_status is StepStatus.FAILED
    assert output.terminal_failure is not None
    assert output.terminal_failure.code == "copro_proposal_cardinality"
    assert output.accepted_candidates == ()
    assert output.evaluation_intents == ()
    evidence = output.state_delta["proposer_evidence"]
    assert evidence[0]["disposition"] == "rejected"
    assert "required field" in str(evidence[0]["reason"])


def test_budget_exhaustion_is_an_exact_terminal_failure() -> None:
    adapter, transport, control = _adapter({})

    output = adapter.invoke(_request(control, proposal_budget=1), ())

    assert output.proposed_status is StepStatus.FAILED
    assert output.terminal_failure is not None
    assert output.terminal_failure.code == "copro_proposal_budget_exhausted"
    assert transport.calls == []


def test_history_lineage_uses_current_exact_request_base() -> None:
    control = _control(depth=2)
    history = [
        _entry(control, 0, "a", "a {input}", 0.1),
        _entry(control, 1, "b", "b {input}", 0.2),
        _entry(control, 2, "c", "c {input}", 0.3),
    ]
    current = _candidate("c", "c {input}", parent="prior-round")
    adapter, _, _ = _adapter(
        {(HISTORY_PROPOSAL, 1): ("x {input}", "y {input}", "z {input}")},
        control=control,
    )

    output = adapter.invoke(
        _request(
            control,
            step_index=1,
            candidates=(current,),
            history=history,
        ),
        (),
    )

    exact_base = candidate_reference(current).record_ref
    assert all(
        item.base_ref == exact_base for item in output.proposed_candidates
    )


def test_registry_key_and_mode_conform() -> None:
    adapter, _, _ = _adapter({})

    assert adapter.key == "copro"
    assert adapter.mode is StepMode.PROPOSAL_ONLY
