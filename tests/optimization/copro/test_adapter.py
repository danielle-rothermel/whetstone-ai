from __future__ import annotations

from copy import deepcopy
from typing import cast

import pytest

from tests.optimization.copro.support import (
    configure_test_copro,
    copro_candidate,
    copro_run,
    copro_step_request,
    durable_copro_proposal_executor,
    make_test_copro_adapter,
)
from tests.optimization.support import (
    FULL_A,
    FULL_B,
    FULL_C,
    RecordingEvaluationService,
    internal_reward_policy,
    make_harness,
    make_store,
)
from whetstone.core.effects.models import ReplayPolicy
from whetstone.core.identity import typed_ref_for_record
from whetstone.core.roles import EvaluationRole
from whetstone.evaluation.schema_names import EVALUATION_EVIDENCE_SCHEMA
from whetstone.experiment.candidate import candidate_reference
from whetstone.experiment.reward import (
    apply_reward_policy,
    reward_reference,
)
from whetstone.optimization.adapters import (
    AdapterReplayPolicyMismatchError,
    MappingAdapterRegistry,
)
from whetstone.optimization.contracts import (
    INTENT_RESOLUTION_SCHEMA_VERSION,
    EvaluationIntent,
    IntentOutcome,
    IntentResolution,
    ResolutionClass,
    ResolutionDetail,
    StepMode,
    StepStatus,
)
from whetstone.optimization.copro.adapter import (
    HISTORY_PROPOSAL,
    SEED_PROPOSAL,
    CoproAdapter,
    CoproAttempt,
    CoproConfig,
    CoproDriver,
    CoproState,
    rank_attempt_history,
)
from whetstone.optimization.proposal.proposer import (
    DurableProposalExecutor,
    FakeProposerTransport,
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
        candidate=candidate_reference(copro_candidate(cid, template)),
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
    adapter, transport, control = make_test_copro_adapter(
        {(SEED_PROPOSAL, 0): ('"new {input}"', "other {input}")}
    )
    request = copro_step_request(control)

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
    adapter, _, _ = make_test_copro_adapter({})

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

    control = configure_test_copro()
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
    adapter, transport, control = make_test_copro_adapter(
        {(SEED_PROPOSAL, 0): ("new {input}", "other {input}")}
    )
    store = make_store(tmp_path)
    service = RecordingEvaluationService(
        store, reward_policy=internal_reward_policy()
    )
    harness = make_harness(
        store=store,
        adapter_registry=MappingAdapterRegistry({"copro": adapter}),
        run=copro_run(control),
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
        harness.run_step(copro_step_request(control))

    assert caught.value.configured_policy is ReplayPolicy.IDEMPOTENT
    assert caught.value.required_policy is ReplayPolicy.DURABLE_WORKFLOW
    assert adapter.invocations == 0
    assert transport.calls == []
    assert service.calls == []
    assert puts == 0


def test_durable_workflow_harness_reaches_copro_and_replays_result(
    tmp_path,
) -> None:
    adapter, transport, control = make_test_copro_adapter(
        {(SEED_PROPOSAL, 0): ("new {input}", "other {input}")}
    )
    store = make_store(tmp_path)
    service = RecordingEvaluationService(
        store, reward_policy=internal_reward_policy()
    )
    harness = make_harness(
        store=store,
        adapter_registry=MappingAdapterRegistry({"copro": adapter}),
        run=copro_run(control),
        evaluation_service=service,
        adapter_replay_policy=ReplayPolicy.DURABLE_WORKFLOW,
    )
    request = copro_step_request(control)

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
    control = configure_test_copro(depth=3)
    history = [
        _entry(control, 0, "a-old", "a {input}", 0.7),
        _entry(control, 1, "b", "b {input}", 0.9),
        _entry(control, 2, "c", "c {input}", 0.8),
        _entry(control, 3, "a-new", "a {input}", 0.9),
        _entry(control, 4, "d", "d {input}", 0.95),
        _entry(control, 5, "e", "e {input}", 0.1),
    ]
    original = deepcopy(history)
    adapter, transport, _ = make_test_copro_adapter(
        {(HISTORY_PROPOSAL, 2): ("x {input}", "y {input}", "z {input}")},
        control=control,
    )

    output = adapter.invoke(
        copro_step_request(control, step_index=2, history=history), ()
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
    adapter, _, control = make_test_copro_adapter(
        {(SEED_PROPOSAL, 0): ("duplicate {input}", "duplicate {input}")}
    )

    output = adapter.invoke(copro_step_request(control), ())

    assert len(output.evaluation_intents) == 3
    assert [
        item.payload["user_prompt_template"]
        for item in output.proposed_candidates
    ] == ["duplicate {input}", "duplicate {input}"]


def test_driver_owns_round_counts_ranking_and_statistics() -> None:
    control = configure_test_copro(depth=2, track_stats=True)
    driver = CoproDriver(CoproConfig(breadth=3, depth=2, track_stats=True))
    initial = copro_candidate("baseline", "base {input}")
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
    control = configure_test_copro()
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
        candidate=candidate_reference(copro_candidate("a", "a {input}")),
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
    control = configure_test_copro()
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
        "dc542e8b75a3d7f3febd33b53d8a8759755a0b933acef0f97a6bec0c1d4db88d"
    )


def test_attempt_replay_rejects_missing_or_mismatched_result_ref() -> None:
    control = configure_test_copro()
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
    control = configure_test_copro()
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
    control = configure_test_copro()
    intent = EvaluationIntent(
        intent_id="copro-run:0:0",
        candidate=candidate_reference(copro_candidate("a", "a {input}")),
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
    adapter, transport, control = make_test_copro_adapter({})
    other = configure_test_copro(track_stats=True)
    mismatched_run = copro_step_request(control).model_copy(
        update={"run": copro_run(other)}
    )

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
        proposal_executor=durable_copro_proposal_executor(),
    )
    with pytest.raises(ValueError, match="execution policy"):
        wrong_adapter.invoke(copro_step_request(control), ())
    assert wrong_policy.calls == []


def test_round_index_must_match_step_index_before_any_spend() -> None:
    control = configure_test_copro(depth=2)
    adapter, transport, _ = make_test_copro_adapter(
        {(SEED_PROPOSAL, 0): ("x {input}", "y {input}")},
        control=control,
    )
    mismatched = copro_step_request(control, step_index=1).model_copy(
        update={"hyperparameters": control.step_hyperparameters(iteration=0)}
    )

    with pytest.raises(ValueError, match="must match the durable step index"):
        adapter.invoke(mismatched, ())

    assert transport.calls == []


def test_render_contract_rejection_and_underfill_are_terminal_failures() -> (
    None
):
    adapter, _, control = make_test_copro_adapter(
        {(SEED_PROPOSAL, 0): ("missing field", "valid {input}")}
    )

    output = adapter.invoke(copro_step_request(control), ())

    assert output.proposed_status is StepStatus.FAILED
    assert output.terminal_failure is not None
    assert output.terminal_failure.code == "copro_proposal_cardinality"
    assert output.accepted_candidates == ()
    assert output.evaluation_intents == ()
    evidence = output.state_delta["proposer_evidence"]
    assert evidence[0]["disposition"] == "rejected"
    assert "required field" in str(evidence[0]["reason"])


def test_budget_exhaustion_is_an_exact_terminal_failure() -> None:
    adapter, transport, control = make_test_copro_adapter({})

    output = adapter.invoke(copro_step_request(control, proposal_budget=1), ())

    assert output.proposed_status is StepStatus.FAILED
    assert output.terminal_failure is not None
    assert output.terminal_failure.code == "copro_proposal_budget_exhausted"
    assert transport.calls == []


def test_history_lineage_uses_current_exact_request_base() -> None:
    control = configure_test_copro(depth=2)
    history = [
        _entry(control, 0, "a", "a {input}", 0.1),
        _entry(control, 1, "b", "b {input}", 0.2),
        _entry(control, 2, "c", "c {input}", 0.3),
    ]
    current = copro_candidate("c", "c {input}", parent="prior-round")
    adapter, _, _ = make_test_copro_adapter(
        {(HISTORY_PROPOSAL, 1): ("x {input}", "y {input}", "z {input}")},
        control=control,
    )

    output = adapter.invoke(
        copro_step_request(
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
    adapter, _, _ = make_test_copro_adapter({})

    assert adapter.key == "copro"
    assert adapter.mode is StepMode.PROPOSAL_ONLY
