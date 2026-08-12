from __future__ import annotations

import hashlib
import operator
from typing import Any, cast

import pytest
from pydantic import ValidationError

from whetstone.core.identity import ImmutableJsonObject
from whetstone.optimization.miprov2.demo import (
    BootstrapAcceptance,
    ComponentDemo,
    ComponentDemoSequence,
    ComponentDemoSet,
    DemoSourceKind,
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


def test_acceptance_hash_binds_all_source_evidence_and_decision() -> None:
    acceptance = BootstrapAcceptance(
        source_task_hash=_identity("task-1"),
        source_generation_hash=_identity("generation-1"),
        source_trace_hash=_identity("trace-1"),
        source_output_hash=_identity("output-1"),
        source_score_hash=_identity("score-1"),
        metric_present=True,
        score=0.75,
        metric_threshold=0.5,
        accepted=True,
    )

    assert acceptance.identity_payload() == {
        "source_task_hash": _identity("task-1"),
        "source_generation_hash": _identity("generation-1"),
        "source_trace_hash": _identity("trace-1"),
        "source_output_hash": _identity("output-1"),
        "source_score_hash": _identity("score-1"),
        "metric_present": True,
        "score": 0.75,
        "metric_threshold": 0.5,
        "accepted": True,
    }
    assert acceptance.identity_hash() == (
        "5a50b929157ada0750a19d833ac68d7ddcae8f0d854ae60ba66f6aca909b4807"
    )
    changed = acceptance.model_copy(
        update={"source_output_hash": _identity("output-2")}
    )
    assert changed.identity_hash() != acceptance.identity_hash()


def test_acceptance_rejects_a_decision_that_disagrees_with_rule() -> None:
    with pytest.raises(ValidationError, match="threshold rule"):
        BootstrapAcceptance(
            source_task_hash=_identity("task-1"),
            source_generation_hash=_identity("generation-1"),
            source_trace_hash=_identity("trace-1"),
            source_output_hash=_identity("output-1"),
            source_score_hash=_identity("score-1"),
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


def test_demo_json_is_deeply_immutable_and_isolated_from_callers() -> None:
    caller_inputs = {"nested": {"items": [1, 2]}}
    step = ObservedTraceStep(
        trace_index=0,
        component_id="answerer",
        inputs=caller_inputs,
        outputs={"answer": "a"},
    )
    caller_inputs["nested"]["items"].append(3)

    assert step.inputs.to_json() == {"nested": {"items": [1, 2]}}
    nested = step.inputs["nested"]
    assert not isinstance(nested, dict)
    with pytest.raises(TypeError):
        operator.setitem(cast("Any", step.inputs), "new", "value")
    with pytest.raises(AttributeError):
        cast("Any", nested)["items"].append(3)


def test_demo_copy_and_construct_refreeze_json_and_stabilize_identity() -> (
    None
):
    task = LabeledTaskDemo(
        source_task_hash=_identity("task-immutable"),
        inputs_by_component={"first": {"question": "q"}},
        outputs_by_component={"first": {"answer": "a"}},
    )
    demo = task.for_component("first")
    copied_inputs = {"nested": {"items": [1]}}
    copied = demo.model_copy(update={"inputs": copied_inputs}, deep=True)
    copied_hash = copied.identity_hash()
    copied_inputs["nested"]["items"].append(2)
    assert copied.identity_hash() == copied_hash

    constructed_inputs = {"nested": {"items": [3]}}
    constructed = ComponentDemo.model_construct(
        **{
            field: getattr(demo, field)
            for field in ComponentDemo.model_fields
            if field != "inputs"
        },
        inputs=constructed_inputs,
    )
    constructed_hash = constructed.identity_hash()
    constructed_inputs["nested"]["items"].append(4)
    assert constructed.identity_hash() == constructed_hash
    with pytest.raises(TypeError):
        operator.setitem(cast("Any", constructed.inputs), "nested", {})


def test_labeled_demo_adapts_to_component_without_fake_generation_data() -> (
    None
):
    task = LabeledTaskDemo(
        source_task_hash=_identity("task-1"),
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
    assert demo.source_task_hash == _identity("task-1")
    assert demo.source_generation_hash == demo.source_trace_hash
    assert demo.source_trace_hash == demo.source_output_hash
    assert len(demo.acceptance_identity_hash) == 64


def _bootstrapped_demo(**overrides: object):
    fields: dict[str, object] = {
        "component_id": "answerer",
        "source_kind": DemoSourceKind.BOOTSTRAPPED,
        "inputs": {"question": "q"},
        "outputs": {"answer": "a"},
        "augmented": True,
        "source_task_hash": _identity("task"),
        "source_generation_hash": _identity("generation"),
        "source_trace_hash": _identity("trace"),
        "source_output_hash": _identity("output"),
        "source_score_hash": _identity("score"),
        "source_trace_index": 0,
        "score": 1.0,
        "acceptance_identity_hash": _identity("acceptance"),
    }
    fields.update(overrides)
    return ComponentDemo(**fields)  # type: ignore[arg-type]


def _assert_item_assignment_refused(mapping: object, key: str) -> None:

    with pytest.raises(TypeError):
        cast("Any", mapping)[key] = "tampered"


def test_demo_field_mappings_cannot_drift_under_their_identity_hash() -> None:
    demo = _bootstrapped_demo()
    before = demo.identity_hash()

    assert isinstance(demo.inputs, ImmutableJsonObject)
    assert isinstance(demo.outputs, ImmutableJsonObject)
    _assert_item_assignment_refused(demo.inputs, "question")
    _assert_item_assignment_refused(demo.outputs, "answer")
    demo.inputs.to_json()["question"] = "tampered"
    demo.model_dump(mode="json")["inputs"]["question"] = "tampered"

    assert demo.inputs == {"question": "q"}
    assert demo.identity_hash() == before


def test_trace_step_field_mappings_are_deeply_immutable() -> None:
    step = ObservedTraceStep(
        trace_index=0,
        component_id="answerer",
        inputs={"context": {"nested": ["a", "b"]}},
        outputs={"answer": "a"},
    )

    assert isinstance(step.inputs, ImmutableJsonObject)
    nested = step.inputs["context"]
    assert isinstance(nested, ImmutableJsonObject)
    _assert_item_assignment_refused(nested, "nested")
    assert nested["nested"] == ("a", "b")


def test_labeled_task_component_mappings_cannot_drift() -> None:
    task = LabeledTaskDemo(
        source_task_hash=_identity("task-1"),
        inputs_by_component={"first": {"question": "q"}},
        outputs_by_component={"first": {"answer": "a"}},
    )
    before = task.for_component("first").identity_hash()

    task.inputs_for("first")["question"] = "tampered"
    _assert_item_assignment_refused(task.inputs_by_component, "first")

    assert task.for_component("first").identity_hash() == before


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_metric_values_are_rejected_at_construction(
    bad: float,
) -> None:
    with pytest.raises(ValidationError):
        _bootstrapped_demo(score=bad)
    with pytest.raises(ValidationError):
        BootstrapAcceptance(
            source_task_hash=_identity("task"),
            source_generation_hash=_identity("generation"),
            source_trace_hash=_identity("trace"),
            source_output_hash=_identity("output"),
            source_score_hash=_identity("score"),
            metric_present=True,
            score=bad,
            metric_threshold=None,
            accepted=True,
        )
    with pytest.raises(ValidationError):
        BootstrapAcceptance(
            source_task_hash=_identity("task"),
            source_generation_hash=_identity("generation"),
            source_trace_hash=_identity("trace"),
            source_output_hash=_identity("output"),
            source_score_hash=_identity("score"),
            metric_present=True,
            score=1.0,
            metric_threshold=bad,
            accepted=True,
        )


def test_demo_input_and_output_field_names_must_be_disjoint() -> None:
    with pytest.raises(ValidationError, match="fields overlap: answer"):
        _bootstrapped_demo(
            inputs={"answer": "input"},
            outputs={"answer": "output"},
        )
    with pytest.raises(ValidationError, match="fields overlap: answer"):
        ObservedTraceStep(
            trace_index=0,
            component_id="answerer",
            inputs={"answer": "input"},
            outputs={"answer": "output"},
        )


def test_labeled_demo_requires_a_production_task_content_hash() -> None:
    with pytest.raises(ValidationError, match="source_task_hash"):
        LabeledTaskDemo(
            source_task_hash="task-label",
            inputs_by_component={"first": {"question": "q"}},
            outputs_by_component={"first": {"answer": "a"}},
        )


def test_component_demo_set_is_component_ordered_and_identity_bearing() -> (
    None
):
    task = LabeledTaskDemo(
        source_task_hash=_identity("task-1"),
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

    assert demo_set.demos_for("first")[0].source_task_hash == _identity(
        "task-1"
    )
    assert demo_set.demos_for("first")[0].identity_hash() == (
        "e3066ada2a184098a2a8be82cf2811b7ca8b667cd34308f066f3a7ba4d45b51a"
    )
    assert demo_set.identity_hash() == (
        "affbc6d23291c84d1526147e087fcebf85e9c8e527fc6c6707028f3504a96808"
    )
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
