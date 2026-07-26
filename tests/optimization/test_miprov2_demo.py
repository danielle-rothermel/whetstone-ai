from __future__ import annotations

import hashlib

import pytest
from pydantic import ValidationError

from whetstone.optimization.miprov2_demo import (
    BootstrapAcceptance,
    ComponentDemoSequence,
    ComponentDemoSet,
    LabeledTaskDemo,
    ObservedTraceStep,
    bootstrap_accepts,
    proposal_demo_context,
    study_demo_context,
)


def _identity(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


@pytest.mark.parametrize(
    (
        "metric_present",
        "score",
        "threshold",
        "accepted",
    ),
    [
        (False, None, None, True),
        (False, None, 10.0, True),
        (True, 0.0, None, False),
        (True, -1.0, None, True),
        (True, 0.0, 0.0, False),
        (True, -1.0, 0.0, True),
        (True, 0.5, 0.5, True),
        (True, 0.49, 0.5, False),
        (True, -2.0, -1.0, False),
        (True, -0.5, -1.0, True),
        (True, False, None, False),
        (True, True, None, True),
    ],
)
def test_bootstrap_acceptance_matches_dspy_threshold_truthiness(
    metric_present: bool,
    score: bool | float | None,
    threshold: float | None,
    accepted: bool,
) -> None:
    assert (
        bootstrap_accepts(
            metric_present=metric_present,
            score=score,
            metric_threshold=threshold,
        )
        is accepted
    )


def test_acceptance_identity_binds_all_source_evidence_and_decision() -> None:
    acceptance = BootstrapAcceptance(
        source_task_identity=_identity("task-1"),
        source_rollout_identity=_identity("rollout-1"),
        source_trace_identity=_identity("trace-1"),
        source_output_identity=_identity("output-1"),
        source_score_identity=_identity("score-1"),
        metric_present=True,
        score=0.75,
        metric_threshold=0.5,
        accepted=True,
    )

    assert len(acceptance.identity_hash()) == 64
    changed = acceptance.model_copy(
        update={"source_output_identity": _identity("output-2")}
    )
    assert changed.identity_hash() != acceptance.identity_hash()


def test_acceptance_rejects_a_decision_that_disagrees_with_rule() -> None:
    with pytest.raises(ValidationError, match="threshold rule"):
        BootstrapAcceptance(
            source_task_identity=_identity("task-1"),
            source_rollout_identity=_identity("rollout-1"),
            source_trace_identity=_identity("trace-1"),
            source_output_identity=_identity("output-1"),
            source_score_identity=_identity("score-1"),
            metric_present=True,
            score=0.1,
            metric_threshold=0.5,
            accepted=True,
        )


def test_observed_trace_step_is_strict_json_and_allows_unknown_component() -> (
    None
):
    step = ObservedTraceStep(
        trace_index=0,
        component_id=None,
        inputs={"question": "q"},
        outputs={"answer": "a"},
    )
    assert step.component_id is None

    with pytest.raises(ValidationError, match="strict finite JSON"):
        ObservedTraceStep(
            trace_index=0,
            component_id="answerer",
            inputs={"bad": object()},
            outputs={},
        )


def test_labeled_demo_adapts_to_component_without_fake_rollout_data() -> None:
    task = LabeledTaskDemo(
        source_task_identity=_identity("task-1"),
        inputs_by_component={
            "first": {"question": "q"},
            "second": {"draft": "d"},
        },
        outputs_by_component={
            "first": {"draft": "d"},
            "second": {"answer": "a"},
        },
    )

    demo = task.for_component("second")

    assert demo.component_id == "second"
    assert demo.augmented is False
    assert demo.source_trace_index is None
    assert demo.inputs == {"draft": "d"}
    assert demo.outputs == {"answer": "a"}
    assert demo.source_task_identity == _identity("task-1")
    assert demo.source_rollout_identity == demo.source_trace_identity
    assert demo.source_trace_identity == demo.source_output_identity
    assert len(demo.acceptance_identity_hash) == 64


def test_labeled_demo_requires_a_production_task_content_hash() -> None:
    with pytest.raises(ValidationError, match="source_task_identity"):
        LabeledTaskDemo(
            source_task_identity="task-label",
            inputs_by_component={"first": {"question": "q"}},
            outputs_by_component={"first": {"answer": "a"}},
        )


def test_component_demo_set_is_component_ordered_and_identity_bearing() -> (
    None
):
    task = LabeledTaskDemo(
        source_task_identity=_identity("task-1"),
        inputs_by_component={"first": {"question": "q"}},
        outputs_by_component={"first": {"answer": "a"}},
    )
    demo_set = ComponentDemoSet(
        candidate_seed=-2,
        components=(
            ComponentDemoSequence(
                component_id="first",
                demos=(task.for_component("first"),),
            ),
        ),
    )

    assert demo_set.demos_for("first")[0].source_task_identity == _identity(
        "task-1"
    )
    assert len(demo_set.identity_hash()) == 64
    with pytest.raises(KeyError):
        demo_set.demos_for("missing")


def test_zero_shot_demo_context_respects_phase_boundary() -> None:
    demo_set = ComponentDemoSet(
        candidate_seed=-3,
        components=(ComponentDemoSequence(component_id="first"),),
    )
    candidates = (demo_set,)

    assert proposal_demo_context(candidates, zeroshot_opt=True) is candidates
    assert proposal_demo_context(candidates, zeroshot_opt=False) is candidates
    assert study_demo_context(candidates, zeroshot_opt=True) is None
    assert study_demo_context(candidates, zeroshot_opt=False) is candidates
