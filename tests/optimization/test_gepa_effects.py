from __future__ import annotations

import warnings
from typing import Any

import pytest
from dr_store import ObjectStore, SqliteBackend
from gepa.core.adapter import EvaluationBatch

from whetstone.optimization.gepa_effects import (
    GepaCandidateComponent,
    GepaComponentTraceProjection,
    GepaDataInstance,
    GepaEffectConflictError,
    GepaEffectContext,
    GepaEffectRecorder,
    GepaEffectSlot,
    GepaEffectTranscript,
    GepaEvaluationAuthorityBinding,
    GepaEvaluationEffectRequest,
    GepaEvaluationEffectResult,
    GepaEvaluationRow,
    GepaProposalAuthorityBinding,
    GepaProposalEffectRequest,
    GepaProposalEffectResult,
    GepaTrajectoryProjection,
)
from whetstone.optimization.gepa_engine import GepaDetailedResult
from whetstone.optimization.gepa_prompts import (
    GepaComponentFormat,
    GepaPromptFormatDescriptor,
    GepaPromptServices,
    NativeGepaReflectionPromptBuilder,
    NativeGepaReflectionResponseParser,
)
from whetstone.optimization.gepa_result_artifact import (
    GepaResultArtifactStore,
    GepaRunResultArtifact,
)
from whetstone.optimization.gepa_source import GEPA_SOURCE_MANIFEST_HASH
from whetstone.optimization.gepa_upstream_adapter import (
    GEPA_UPSTREAM_ADAPTER_IDENTITY_HASH,
    WhetstoneGepaAdapter,
)
from whetstone.optimization.identity import typed_ref_for_record
from whetstone.optimization.proposer import ProposerConfig

_A = "a" * 64
_B = "b" * 64
_C = "c" * 64
_D = "d" * 64
_E = "e" * 64
_F = "f" * 64


def _prompt_services() -> GepaPromptServices:
    return GepaPromptServices(
        descriptor=GepaPromptFormatDescriptor(
            format_name="test",
            components=(
                GepaComponentFormat(
                    component_name="alpha",
                    component_schema_identity_hash=_A,
                ),
                GepaComponentFormat(
                    component_name="beta",
                    component_schema_identity_hash=_B,
                ),
            ),
        ),
        reflection_builder=NativeGepaReflectionPromptBuilder(),
        reflection_parser=NativeGepaReflectionResponseParser(),
    )


def _context() -> GepaEffectContext:
    return GepaEffectContext(
        run_id="gepa:test",
        control_identity_hash=_A,
        source_manifest_identity_hash=_B,
        adapter_identity_hash=GEPA_UPSTREAM_ADAPTER_IDENTITY_HASH,
    )


def _data(index: int) -> GepaDataInstance:
    return GepaDataInstance(
        upstream_position=index,
        data_id=f"{index + 1:064x}",
        data_ref=typed_ref_for_record(
            "test.gepa.data",
            {"index": index},
        ),
        loader_identity_hash=_C,
    )


def _evaluation_authority(
    *,
    failure_score: float = 0.0,
    add_format_failure_as_feedback: bool = False,
    warn_on_score_mismatch: bool = True,
    selection_seed: int = 0,
) -> GepaEvaluationAuthorityBinding:
    return GepaEvaluationAuthorityBinding(
        authority_identity_hash=_A,
        evaluation_config_identity_hash=_B,
        reward_policy_identity_hash=_C,
        provider_route_identity_hash=_D,
        execution_policy_identity_hash=_E,
        prompt_adapter_identity_hash=_F,
        response_parser_identity_hash=_A,
        data_registry_identity_hash=_B,
        failure_score=failure_score,
        add_format_failure_as_feedback=add_format_failure_as_feedback,
        warn_on_score_mismatch=warn_on_score_mismatch,
        selection_seed=selection_seed,
    )


def _proposal_authority(
    services: GepaPromptServices | None = None,
) -> GepaProposalAuthorityBinding:
    active = services or _prompt_services()
    return GepaProposalAuthorityBinding(
        authority_identity_hash=_B,
        proposer_transport_identity_hash=_C,
        prompt_binding_identity_hash=active.binding.identity_hash(),
        execution_policy_identity_hash=_D,
        prompt_adapter_identity_hash=_E,
        durability_policy_identity_hash=_F,
        proposer_config=ProposerConfig(
            provider_call_config_ref="provider://gepa-reflection",
            provider_call_config_hash=_D,
        ),
    )


def _evaluation_request() -> GepaEvaluationEffectRequest:
    return GepaEvaluationEffectRequest(
        slot=GepaEffectSlot(context=_context(), invocation_ordinal=0),
        candidate=(
            GepaCandidateComponent(name="alpha", text="alpha-0"),
            GepaCandidateComponent(name="beta", text="beta-0"),
        ),
        data=(_data(0), _data(1)),
        capture_traces=False,
        authority=_evaluation_authority(),
    )


def _evaluation_result(
    request: GepaEvaluationEffectRequest,
) -> GepaEvaluationEffectResult:
    return GepaEvaluationEffectResult(
        request_identity_hash=request.identity_hash(),
        rows=tuple(
            GepaEvaluationRow(
                data=item,
                output={"data_id": item.data_id},
                score=float(index),
                evidence_refs=(item.data_ref,),
            )
            for index, item in enumerate(request.data)
        ),
        logical_metric_calls=len(request.data),
    )


def test_effect_recorder_reuses_exact_result_and_rejects_slot_drift(
    tmp_path,
) -> None:
    database = tmp_path / "gepa-effects.sqlite"
    first = GepaEffectRecorder(
        ObjectStore(SqliteBackend(database)),
    )
    request = _evaluation_request()
    result = _evaluation_result(request)

    first.record_request(request)
    first.record_request(request)
    assert first.record_evaluation_result(request, result) == result

    fresh = GepaEffectRecorder(
        ObjectStore(SqliteBackend(database)),
    )
    fresh.record_request(request)
    assert fresh.load_evaluation_result(request) == result

    drifted = request.model_copy(update={"capture_traces": True})
    with pytest.raises(GepaEffectConflictError, match="ordinal 0"):
        fresh.record_request(drifted)


def test_evaluation_row_rejects_unauditable_success() -> None:
    with pytest.raises(ValueError, match="canonical evidence"):
        GepaEvaluationRow(
            data=_data(0),
            output={"answer": "unproven"},
            score=1.0,
        )


def test_recorder_builds_ordered_semantic_effect_transcript(tmp_path) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "transcript.sqlite"))
    recorder = GepaEffectRecorder(store)
    request = _evaluation_request()
    result = _evaluation_result(request)
    recorder.record_request(request)
    recorder.record_evaluation_result(request, result)

    transcript = recorder.build_transcript(
        context=request.slot.context,
        effect_count=1,
    )
    transcript_ref = recorder.persist_transcript(transcript)

    assert transcript.entries[0].invocation_ordinal == 0
    assert transcript.entries[0].upstream_candidate_index is None
    assert transcript.entries[0].data_ids == tuple(
        item.data_id for item in request.data
    )
    assert (
        GepaEffectTranscript.model_validate(
            store.get(transcript_ref.reference)
        )
        == transcript
    )


def test_result_artifact_pairs_detail_and_effect_transcript_idempotently(
    tmp_path,
) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "artifact.sqlite"))
    context = GepaEffectContext(
        run_id="gepa:artifact",
        control_identity_hash=_A,
        source_manifest_identity_hash=GEPA_SOURCE_MANIFEST_HASH,
        adapter_identity_hash=GEPA_UPSTREAM_ADAPTER_IDENTITY_HASH,
    )
    recorder = GepaEffectRecorder(store)
    transcript_ref = recorder.persist_transcript(
        GepaEffectTranscript(context=context, entries=())
    )
    detail = GepaDetailedResult(
        candidates=({"alpha": "alpha-0"},),
        parents=((),),
        val_aggregate_scores=(0.5,),
        val_subscores=({_data(0).data_id: 0.5},),
        per_val_instance_best_candidates={_data(0).data_id: (0,)},
        discovery_eval_counts=(1,),
        seed=0,
        best_idx=0,
        control_identity_hash=_A,
    )
    artifact_store = GepaResultArtifactStore(store)

    first = artifact_store.persist(
        context=context,
        detailed_result=detail,
        transcript_ref=transcript_ref,
    )
    replay = artifact_store.persist(
        context=context,
        detailed_result=detail,
        transcript_ref=transcript_ref,
    )

    assert replay == first
    artifact = GepaRunResultArtifact.model_validate(store.get(first.reference))
    assert artifact.effect_transcript_ref == transcript_ref
    conflicting = detail.model_copy(update={"val_aggregate_scores": (0.75,)})
    with pytest.raises(ValueError, match="different terminal"):
        artifact_store.persist(
            context=context,
            detailed_result=conflicting,
            transcript_ref=transcript_ref,
        )


class _FakeBroker:
    def __init__(self) -> None:
        self.evaluations: list[GepaEvaluationEffectRequest] = []
        self.proposals: list[GepaProposalEffectRequest] = []

    def evaluate(
        self,
        request: GepaEvaluationEffectRequest,
    ) -> GepaEvaluationEffectResult:
        self.evaluations.append(request)
        rows = []
        for index, item in enumerate(request.data):
            trajectory = (
                GepaTrajectoryProjection(
                    data_id=item.data_id,
                    inputs={"data_id": item.data_id},
                    generated_outputs={"answer": index},
                    feedback=f"score={index / 2}",
                    module_score=float(index / 2),
                    component_records={
                        name: (
                            GepaComponentTraceProjection(
                                inputs={"data_id": item.data_id},
                                generated_outputs={"answer": index},
                                feedback=f"score={index / 2}",
                                feedback_score=float(index / 2),
                                source_refs=(item.data_ref,),
                            ),
                        )
                        for name in ("alpha", "beta")
                    },
                    source_refs=(item.data_ref,),
                )
                if request.capture_traces
                else None
            )
            rows.append(
                GepaEvaluationRow(
                    data=item,
                    output={"answer": index},
                    score=float(index / 2),
                    objective_scores={"quality": float(index / 2)},
                    trajectory=trajectory,
                    evidence_refs=(item.data_ref,),
                )
            )
        return GepaEvaluationEffectResult(
            request_identity_hash=request.identity_hash(),
            rows=tuple(rows),
            logical_metric_calls=len(rows),
        )

    def propose(
        self,
        request: GepaProposalEffectRequest,
    ) -> GepaProposalEffectResult:
        self.proposals.append(request)
        replacement = f"{request.component_name}-improved"
        attempt_ref = typed_ref_for_record(
            "test.gepa.proposal_attempt",
            {"component": request.component_name},
        )
        return GepaProposalEffectResult(
            request_identity_hash=request.identity_hash(),
            raw_response=f"```\n{replacement}\n```",
            parsed_components=(
                GepaCandidateComponent(
                    name=request.component_name,
                    text=replacement,
                ),
            ),
            request_evidence={"prompt": request.rendered_prompt.text},
            response_evidence={"raw": replacement},
            provider_attempt_refs=(attempt_ref,),
            usage={"output_tokens": 1},
            cost=0.0,
        )


def _adapter(
    broker: Any,
    *,
    evaluation_authority: GepaEvaluationAuthorityBinding | None = None,
) -> WhetstoneGepaAdapter:
    services = _prompt_services()
    return WhetstoneGepaAdapter(
        context=_context(),
        broker=broker,
        evaluation_authority=evaluation_authority or _evaluation_authority(),
        proposal_authority=_proposal_authority(services),
        prompt_services=services,
    )


def test_upstream_adapter_preserves_order_evidence_and_effect_ordinals() -> (
    None
):
    broker = _FakeBroker()
    adapter = _adapter(broker)
    candidate = {"alpha": "alpha-0", "beta": "beta-0"}
    batch = [_data(1), _data(0)]

    evaluated = adapter.evaluate(batch, candidate, capture_traces=True)

    assert evaluated.outputs == [{"answer": 0}, {"answer": 1}]
    assert evaluated.scores == [0.0, 0.5]
    assert evaluated.objective_scores == [
        {"quality": 0.0},
        {"quality": 0.5},
    ]
    assert evaluated.trajectories is not None
    reflective = adapter.make_reflective_dataset(
        candidate,
        evaluated,
        ["alpha", "beta"],
    )
    assert tuple(reflective) == ("alpha", "beta")
    assert reflective["alpha"][0] == {
        "Inputs": {"data_id": batch[0].data_id},
        "Generated Outputs": {"answer": 0},
        "Feedback": "score=0.0",
    }

    proposed = adapter.propose_new_texts(
        candidate,
        reflective,
        ["alpha", "beta"],
    )

    assert proposed == {
        "alpha": "alpha-improved",
        "beta": "beta-improved",
    }
    assert [
        request.slot.invocation_ordinal
        for request in (
            *broker.evaluations,
            *broker.proposals,
        )
    ] == [0, 1, 2]
    assert [request.component_name for request in broker.proposals] == [
        "alpha",
        "beta",
    ]
    assert all(
        request.authority.prompt_binding_identity_hash
        == _prompt_services().binding.identity_hash()
        for request in broker.proposals
    )


class _ReorderingBroker(_FakeBroker):
    def evaluate(
        self,
        request: GepaEvaluationEffectRequest,
    ) -> GepaEvaluationEffectResult:
        result = super().evaluate(request)
        return result.model_copy(update={"rows": tuple(reversed(result.rows))})


def test_upstream_adapter_rejects_reordered_evaluation_rows() -> None:
    adapter = _adapter(_ReorderingBroker())
    with pytest.raises(ValueError, match="requested data order"):
        adapter.evaluate(
            [_data(0), _data(1)],
            {"alpha": "alpha-0", "beta": "beta-0"},
        )


class _LiteralFenceBroker(_FakeBroker):
    def propose(
        self,
        request: GepaProposalEffectRequest,
    ) -> GepaProposalEffectResult:
        self.proposals.append(request)
        replacement = "Explain with a literal example: ```x = 1```"
        attempt_ref = typed_ref_for_record(
            "test.gepa.proposal_attempt",
            {"component": request.component_name},
        )
        return GepaProposalEffectResult(
            request_identity_hash=request.identity_hash(),
            raw_response=f"```\n{replacement}\n```",
            parsed_components=(
                GepaCandidateComponent(
                    name=request.component_name,
                    text=replacement,
                ),
            ),
            request_evidence={"prompt": request.rendered_prompt.text},
            response_evidence={"raw": replacement},
            provider_attempt_refs=(attempt_ref,),
        )


def test_adapter_does_not_parse_an_already_parsed_literal_fence() -> None:
    adapter = _adapter(_LiteralFenceBroker())
    replacement = adapter.propose_new_texts(
        {"alpha": "alpha-0", "beta": "beta-0"},
        {
            "alpha": (
                {
                    "Inputs": {"x": 1},
                    "Generated Outputs": "1",
                    "Feedback": "ok",
                },
            )
        },
        ["alpha"],
    )

    assert replacement == {
        "alpha": "Explain with a literal example: ```x = 1```"
    }


class _FailureBroker(_FakeBroker):
    def evaluate(
        self,
        request: GepaEvaluationEffectRequest,
    ) -> GepaEvaluationEffectResult:
        failure_ref = typed_ref_for_record(
            "test.gepa.failure",
            {"data_id": request.data[0].data_id},
        )
        return GepaEvaluationEffectResult(
            request_identity_hash=request.identity_hash(),
            rows=(
                GepaEvaluationRow(
                    data=request.data[0],
                    output=None,
                    score=0.0,
                    failure_ref=failure_ref,
                ),
            ),
            logical_metric_calls=1,
        )


def test_adapter_maps_failed_rows_to_bound_failure_score() -> None:
    adapter = _adapter(
        _FailureBroker(),
        evaluation_authority=_evaluation_authority(failure_score=-1.25),
    )

    result = adapter.evaluate(
        [_data(0)],
        {"alpha": "alpha-0", "beta": "beta-0"},
    )

    assert result.scores == [-1.25]


def _trajectory_with_component_records() -> GepaTrajectoryProjection:
    return GepaTrajectoryProjection(
        data_id=_data(0).data_id,
        inputs={"x": "fallback"},
        generated_outputs={"answer": "fallback"},
        feedback="fallback",
        component_records={
            "alpha": (
                GepaComponentTraceProjection(
                    inputs={"x": "format-failure"},
                    generated_outputs={"answer": "invalid"},
                    feedback="format feedback",
                    format_failure=True,
                ),
            ),
            "beta": (
                GepaComponentTraceProjection(
                    inputs={"x": "valid"},
                    generated_outputs={"answer": "valid"},
                    feedback="valid feedback",
                ),
            ),
        },
    )


def _evaluation_batch(
    *trajectories: GepaTrajectoryProjection,
) -> EvaluationBatch[GepaTrajectoryProjection, Any]:
    return EvaluationBatch(
        outputs=[None] * len(trajectories),
        scores=[0.0] * len(trajectories),
        trajectories=list(trajectories),
    )


@pytest.mark.parametrize(
    ("include_format_failure", "expected_components"),
    [
        (False, ("beta",)),
        (True, ("alpha", "beta")),
    ],
)
def test_adapter_applies_bound_format_failure_feedback_policy(
    include_format_failure: bool,
    expected_components: tuple[str, ...],
) -> None:
    adapter = _adapter(
        _FakeBroker(),
        evaluation_authority=_evaluation_authority(
            add_format_failure_as_feedback=include_format_failure,
        ),
    )
    eval_batch = _evaluation_batch(_trajectory_with_component_records())

    reflective = adapter.make_reflective_dataset(
        {"alpha": "alpha-0", "beta": "beta-0"},
        eval_batch,
        ["alpha", "beta"],
    )

    assert tuple(reflective) == expected_components


@pytest.mark.parametrize("warn_on_mismatch", [False, True])
def test_adapter_captures_and_warns_once_for_score_mismatch(
    warn_on_mismatch: bool,
) -> None:
    trajectory = GepaTrajectoryProjection(
        data_id=_data(0).data_id,
        inputs={},
        generated_outputs={},
        feedback="fallback",
        module_score=0.75,
        component_records={
            name: (
                GepaComponentTraceProjection(
                    inputs={"component": name},
                    generated_outputs={"answer": name},
                    feedback="mismatch",
                    feedback_score=0.25,
                ),
            )
            for name in ("alpha", "beta")
        },
    )
    adapter = _adapter(
        _FakeBroker(),
        evaluation_authority=_evaluation_authority(
            warn_on_score_mismatch=warn_on_mismatch,
        ),
    )
    eval_batch = _evaluation_batch(trajectory)

    if warn_on_mismatch:
        with pytest.warns(RuntimeWarning, match="differs"):
            adapter.make_reflective_dataset(
                {"alpha": "alpha-0", "beta": "beta-0"},
                eval_batch,
                ["alpha", "beta"],
            )
        assert len(adapter.score_mismatch_evidence) == 1
        assert adapter.score_mismatch_evidence[0].component_name == "alpha"
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            adapter.make_reflective_dataset(
                {"alpha": "alpha-0", "beta": "beta-0"},
                eval_batch,
                ["alpha", "beta"],
            )
        assert adapter.score_mismatch_evidence == ()


def test_adapter_seeded_trace_choice_restarts_exactly() -> None:
    traces = tuple(
        GepaComponentTraceProjection(
            inputs={"choice": index},
            generated_outputs={"answer": index},
            feedback=f"choice-{index}",
        )
        for index in range(3)
    )
    trajectory = GepaTrajectoryProjection(
        data_id=_data(0).data_id,
        inputs={},
        generated_outputs={},
        feedback="fallback",
        component_records={"alpha": traces},
    )
    eval_batch = _evaluation_batch(trajectory)
    adapter = _adapter(
        _FakeBroker(),
        evaluation_authority=_evaluation_authority(selection_seed=7),
    )

    first = adapter.make_reflective_dataset(
        {"alpha": "alpha-0", "beta": "beta-0"},
        eval_batch,
        ["alpha"],
    )
    second = adapter.make_reflective_dataset(
        {"alpha": "alpha-0", "beta": "beta-0"},
        eval_batch,
        ["alpha"],
    )
    adapter.reset_effect_ordinal()
    replay_first = adapter.make_reflective_dataset(
        {"alpha": "alpha-0", "beta": "beta-0"},
        eval_batch,
        ["alpha"],
    )
    replay_second = adapter.make_reflective_dataset(
        {"alpha": "alpha-0", "beta": "beta-0"},
        eval_batch,
        ["alpha"],
    )

    assert (replay_first, replay_second) == (first, second)
    assert first["alpha"][0]["Feedback"] == "choice-1"
    assert second["alpha"][0]["Feedback"] == "choice-0"


def test_adapter_treats_tiny_score_delta_as_mismatch() -> None:
    trajectory = GepaTrajectoryProjection(
        data_id=_data(0).data_id,
        inputs={},
        generated_outputs={},
        feedback="fallback",
        module_score=0.5,
        component_records={
            "alpha": (
                GepaComponentTraceProjection(
                    inputs={},
                    generated_outputs={},
                    feedback="tiny mismatch",
                    feedback_score=0.5 + 1e-12,
                ),
            ),
        },
    )
    eval_batch = _evaluation_batch(trajectory)
    adapter = _adapter(_FakeBroker())

    with pytest.warns(RuntimeWarning):
        adapter.make_reflective_dataset(
            {"alpha": "alpha-0", "beta": "beta-0"},
            eval_batch,
            ["alpha"],
        )

    assert len(adapter.score_mismatch_evidence) == 1


def test_format_failure_precedes_failed_prediction_skip() -> None:
    format_failure = GepaComponentTraceProjection(
        inputs={"choice": "failed"},
        generated_outputs="raw invalid response",
        feedback="format feedback",
        format_failure=True,
    )
    valid = GepaComponentTraceProjection(
        inputs={"choice": "valid"},
        generated_outputs={"answer": "valid"},
        feedback="valid feedback",
    )
    failed_trajectory = GepaTrajectoryProjection(
        data_id=_data(0).data_id,
        inputs={},
        generated_outputs={},
        feedback="fallback",
        prediction_failed=True,
        component_records={"alpha": (valid,)},
    )
    with_format_failure = failed_trajectory.model_copy(
        update={"component_records": {"alpha": (valid, format_failure)}}
    )
    eval_batch = _evaluation_batch(with_format_failure)
    adapter = _adapter(
        _FakeBroker(),
        evaluation_authority=_evaluation_authority(
            add_format_failure_as_feedback=True,
        ),
    )

    selected = adapter.make_reflective_dataset(
        {"alpha": "alpha-0", "beta": "beta-0"},
        eval_batch,
        ["alpha"],
    )

    assert selected["alpha"][0]["Feedback"] == "format feedback"
    without_failure_batch = _evaluation_batch(failed_trajectory)
    with pytest.raises(ValueError, match="No valid predictions"):
        adapter.make_reflective_dataset(
            {"alpha": "alpha-0", "beta": "beta-0"},
            without_failure_batch,
            ["alpha"],
        )
