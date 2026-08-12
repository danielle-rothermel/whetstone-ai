from __future__ import annotations

import warnings
from typing import Any

import pytest
from gepa.core.adapter import EvaluationBatch

from tests.optimization.gepa.support import (
    data_instance,
    effect_context,
    evaluation_authority_binding,
    prompt_services,
    proposal_authority_binding,
)
from whetstone.core.identity import typed_ref_for_record
from whetstone.optimization.gepa.contracts import (
    GepaCandidateComponent,
    GepaComponentTraceProjection,
    GepaEvaluationAuthorityBinding,
    GepaEvaluationEffectRequest,
    GepaEvaluationEffectResult,
    GepaEvaluationRow,
    GepaProposalEffectRequest,
    GepaProposalEffectResult,
    GepaTrajectoryProjection,
)
from whetstone.optimization.gepa.upstream_adapter import WhetstoneGepaAdapter


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
            request_hash=request.identity_hash(),
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
            request_hash=request.identity_hash(),
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
    services = prompt_services()
    return WhetstoneGepaAdapter(
        context=effect_context(),
        broker=broker,
        evaluation_authority=(
            evaluation_authority or evaluation_authority_binding()
        ),
        proposal_authority=proposal_authority_binding(services),
        prompt_services=services,
    )


def test_upstream_adapter_preserves_order_evidence_and_effect_ordinals() -> (
    None
):
    broker = _FakeBroker()
    adapter = _adapter(broker)
    candidate = {"alpha": "alpha-0", "beta": "beta-0"}
    batch = [data_instance(1), data_instance(0)]

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
        == prompt_services().binding.identity_hash()
        for request in broker.proposals
    )


def test_upstream_adapter_skips_components_without_reflective_traces() -> None:

    broker = _FakeBroker()
    adapter = _adapter(broker)
    candidate = {"alpha": "alpha-0", "beta": "beta-0"}
    partial_dataset = {
        "alpha": (
            {
                "Inputs": {"x": 1},
                "Generated Outputs": "1",
                "Feedback": "ok",
            },
        ),
    }

    proposed = adapter.propose_new_texts(
        candidate,
        partial_dataset,
        ["alpha", "beta"],
    )

    assert proposed == {"alpha": "alpha-improved"}
    assert [request.component_name for request in broker.proposals] == [
        "alpha"
    ]

    empty_broker = _FakeBroker()
    empty_adapter = _adapter(empty_broker)
    assert empty_adapter.propose_new_texts(
        candidate,
        {**partial_dataset, "beta": ()},
        ["alpha", "beta"],
    ) == {"alpha": "alpha-improved"}
    assert [request.component_name for request in empty_broker.proposals] == [
        "alpha"
    ]


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
            [data_instance(0), data_instance(1)],
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
            request_hash=request.identity_hash(),
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
            request_hash=request.identity_hash(),
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
        evaluation_authority=evaluation_authority_binding(failure_score=-1.25),
    )

    result = adapter.evaluate(
        [data_instance(0)],
        {"alpha": "alpha-0", "beta": "beta-0"},
    )

    assert result.scores == [-1.25]


def _trajectory_with_component_records() -> GepaTrajectoryProjection:
    return GepaTrajectoryProjection(
        data_id=data_instance(0).data_id,
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
        evaluation_authority=evaluation_authority_binding(
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
        data_id=data_instance(0).data_id,
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
        evaluation_authority=evaluation_authority_binding(
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
        data_id=data_instance(0).data_id,
        inputs={},
        generated_outputs={},
        feedback="fallback",
        component_records={"alpha": traces},
    )
    eval_batch = _evaluation_batch(trajectory)
    adapter = _adapter(
        _FakeBroker(),
        evaluation_authority=evaluation_authority_binding(selection_seed=7),
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
        data_id=data_instance(0).data_id,
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
        data_id=data_instance(0).data_id,
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
        evaluation_authority=evaluation_authority_binding(
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


def test_reflective_dataset_falls_back_without_component_traces() -> None:
    trajectory = GepaTrajectoryProjection(
        data_id=data_instance(0).data_id,
        inputs={"question": "2+2?"},
        generated_outputs={"test_results": [{"case_id": "case_0"}]},
        feedback="score=0 with case detail",
        prediction_failed=True,
        component_records={},
    )
    adapter = _adapter(_FakeBroker())
    reflective = adapter.make_reflective_dataset(
        {"alpha": "alpha-0", "beta": "beta-0"},
        _evaluation_batch(trajectory),
        ["alpha"],
    )

    assert reflective["alpha"][0]["Feedback"] == "score=0 with case detail"
