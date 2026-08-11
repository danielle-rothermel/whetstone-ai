from __future__ import annotations

import hashlib
import random
from typing import Any

import pytest

from whetstone.optimization.miprov2.bootstrap import (
    MIPROV2_BOOTSTRAP_ATTEMPT_SCHEMA,
    MIPROV2_BOOTSTRAP_PLAN_SCHEMA,
    MIPROV2_BOOTSTRAP_SCHEMA_VERSION,
    MIPROV2_TRACE_SELECTION_PROJECTION_VERSION,
    ZERO_SHOT_BOOTSTRAPPED_DEMOS_IN_PROPOSAL,
    BootstrapAttemptPlan,
    BootstrapErrorLimitReached,
    BootstrapGenerationResult,
    FewshotCandidatePlan,
    FewshotSeedKind,
    TeacherSource,
    _frozen_dspy_demo_tuple_pickle,
    create_fewshot_candidate_plans,
    fold_bootstrap_result,
    initial_compiler_state,
    materialize_bootstrap_demo_set,
    materialize_labels_only_demo_set,
    materialize_reset_demo_set,
    next_bootstrap_attempt,
    plan_labeled_selection,
)
from whetstone.optimization.miprov2.demo import (
    DemoSourceKind,
    LabeledTaskDemo,
    ObservedTraceStep,
)
from whetstone.optimization.miprov2.rng import (
    Miprov2DurableBindings,
    Miprov2RandomState,
    Miprov2RngCheckpoint,
)

COMPONENTS = ("first", "second")


def _identity(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


TASKS = tuple(_identity(f"t{index}") for index in range(5))


def _bindings() -> Miprov2DurableBindings:
    return Miprov2DurableBindings(
        control_identity_hash=_identity("control"),
        prompt_route_identity_hash=_identity("prompt-route"),
        task_route_identity_hash=_identity("task-route"),
        execution_policy_identity_hash=_identity("execution-policy"),
        prompt_adapter_identity_hash=_identity("prompt-adapter"),
        proposal_executor_policy_identity_hash=_identity("proposal-executor"),
        proposal_transport_durability_identity_hash=_identity(
            "proposal-transport"
        ),
        base_candidate_identity_hash=_identity("base"),
        teacher_candidate_identity_hash=_identity("teacher"),
    )


def _golden_bindings() -> Miprov2DurableBindings:
    return Miprov2DurableBindings(
        control_identity_hash="a" * 64,
        prompt_route_identity_hash="b" * 64,
        task_route_identity_hash="c" * 64,
        execution_policy_identity_hash="d" * 64,
        prompt_adapter_identity_hash="e" * 64,
        proposal_executor_policy_identity_hash="f" * 64,
        proposal_transport_durability_identity_hash="1" * 64,
        base_candidate_identity_hash="2" * 64,
        teacher_candidate_identity_hash="3" * 64,
    )


def _planning(**overrides: Any):
    values: dict[str, Any] = {
        "bindings": _bindings(),
        "component_ids": COMPONENTS,
        "trainset_task_hashes": TASKS,
        "num_candidate_sets": 6,
        "max_bootstrapped_demos": 4,
        "max_labeled_demos": 4,
        "max_errors": 3,
        "rng_checkpoint": Miprov2RngCheckpoint.seeded(0),
    }
    values.update(overrides)
    return create_fewshot_candidate_plans(**values)


def _plans(**overrides: Any):
    return _planning(**overrides).plans


def _labeled_trainset(
    task_ids: tuple[str, ...] = TASKS,
) -> tuple[LabeledTaskDemo, ...]:
    return tuple(
        LabeledTaskDemo(
            source_task_hash=task_id,
            inputs_by_component={
                "first": {"question": task_id},
                "second": {"draft": f"draft-{task_id}"},
            },
            outputs_by_component={
                "first": {"draft": f"draft-{task_id}"},
                "second": {"answer": f"answer-{task_id}"},
            },
        )
        for task_id in task_ids
    )


def _result(
    attempt,
    *,
    score: bool | float | None = 1.0,
    metric_present: bool = True,
    trace_steps: tuple[ObservedTraceStep, ...] = (),
    error: str | None = None,
) -> BootstrapGenerationResult:
    return BootstrapGenerationResult(
        attempt_identity_hash=attempt.identity_hash(),
        source_generation_identity=_identity(
            f"generation-{attempt.task_index}-{attempt.round_index}"
        ),
        source_trace_identity=_identity(
            f"trace-{attempt.task_index}-{attempt.round_index}"
        ),
        source_output_identity=_identity(
            f"output-{attempt.task_index}-{attempt.round_index}"
        ),
        source_score_identity=_identity(
            f"score-{attempt.task_index}-{attempt.round_index}"
        ),
        metric_present=metric_present,
        score=score,
        trace_steps=trace_steps,
        error=error,
    )


def test_fewshot_candidate_plan_identity_payload_and_digest_are_pinned() -> (
    None
):
    assert MIPROV2_BOOTSTRAP_PLAN_SCHEMA == "whetstone.miprov2_bootstrap_plan"
    assert MIPROV2_BOOTSTRAP_SCHEMA_VERSION == 1
    plan = FewshotCandidatePlan(
        candidate_ordinal=4,
        candidate_seed=-3,
        bindings=_golden_bindings(),
        kind=FewshotSeedKind.RESET,
        component_ids=("generate",),
        trainset_task_hashes=("4" * 64,),
        max_bootstrapped_demos=0,
        max_labeled_demos=0,
        max_rounds=1,
        max_errors=2,
        metric_threshold=None,
        teacher=None,
        labels_only_selection=None,
    )

    assert plan.identity_payload() == {
        "candidate_ordinal": 4,
        "candidate_seed": -3,
        "bindings": {
            "control_identity_hash": "a" * 64,
            "prompt_route_identity_hash": "b" * 64,
            "task_route_identity_hash": "c" * 64,
            "execution_policy_identity_hash": "d" * 64,
            "prompt_adapter_identity_hash": "e" * 64,
            "proposal_executor_policy_identity_hash": "f" * 64,
            "proposal_transport_durability_identity_hash": "1" * 64,
            "base_candidate_identity_hash": "2" * 64,
            "teacher_candidate_identity_hash": "3" * 64,
            "demo_bridge_version": "whetstone_component_demo_bridge/v1",
        },
        "kind": "reset",
        "component_ids": ["generate"],
        "trainset_task_hashes": ["4" * 64],
        "max_bootstrapped_demos": 0,
        "max_labeled_demos": 0,
        "max_rounds": 1,
        "max_errors": 2,
        "metric_threshold": None,
        "teacher": None,
        "labels_only_selection": None,
        "trace_selection_projection_version": (
            "dspy_example_pickle_protocol4_cpython/v1"
        ),
    }
    assert plan.identity_hash() == (
        "56e9541ffe2a41ac3d6a6e2601b89783dd14ec36ffea53c5b44bcdae17884882"
    )


def test_bootstrap_attempt_plan_identity_payload_and_digest_are_pinned() -> (
    None
):
    assert MIPROV2_BOOTSTRAP_ATTEMPT_SCHEMA == (
        "whetstone.miprov2_bootstrap_attempt"
    )
    assert MIPROV2_BOOTSTRAP_SCHEMA_VERSION == 1
    attempt = BootstrapAttemptPlan(
        bindings=_golden_bindings(),
        plan_identity_hash="5" * 64,
        task_index=2,
        task_hash="6" * 64,
        round_index=1,
        copy_task_model=True,
        generation_id=1,
        temperature=1.0,
    )

    assert attempt.identity_payload() == {
        "bindings": {
            "control_identity_hash": "a" * 64,
            "prompt_route_identity_hash": "b" * 64,
            "task_route_identity_hash": "c" * 64,
            "execution_policy_identity_hash": "d" * 64,
            "prompt_adapter_identity_hash": "e" * 64,
            "proposal_executor_policy_identity_hash": "f" * 64,
            "proposal_transport_durability_identity_hash": "1" * 64,
            "base_candidate_identity_hash": "2" * 64,
            "teacher_candidate_identity_hash": "3" * 64,
            "demo_bridge_version": "whetstone_component_demo_bridge/v1",
        },
        "plan_identity_hash": "5" * 64,
        "task_index": 2,
        "task_hash": "6" * 64,
        "round_index": 1,
        "exclude_equal_task_from_all_teacher_components": True,
        "restore_teacher_demos_after_effect": True,
        "copy_task_model": True,
        "generation_id": 1,
        "temperature": 1.0,
    }
    assert attempt.identity_hash() == (
        "12a1c4dce0a41eccebac246a477fc2bea9cb0f4aa2236f7989fd2021e8a95f5c"
    )


@pytest.mark.parametrize(
    ("count", "seeds"),
    [
        (1, (-3,)),
        (2, (-3, -2)),
        (3, (-3, -2, -1)),
        (6, (-3, -2, -1, 0, 1, 2)),
    ],
)
def test_candidate_count_uses_exact_special_seed_range(
    count: int, seeds: tuple[int, ...]
) -> None:
    assert (
        tuple(plan.candidate_seed for plan in _plans(num_candidate_sets=count))
        == seeds
    )


def test_candidate_plans_require_production_task_content_hashes() -> None:
    with pytest.raises(ValueError, match="trainset_task_hashes"):
        _plans(trainset_task_hashes=("task-label",))


def test_special_seed_kinds_and_unshuffled_minus_one() -> None:
    reset, labels, unshuffled, *_ = _plans()

    assert reset.kind is FewshotSeedKind.RESET
    assert labels.kind is FewshotSeedKind.LABELS_ONLY
    assert unshuffled.kind is FewshotSeedKind.BOOTSTRAP
    assert unshuffled.trainset_task_hashes == TASKS
    assert unshuffled.max_bootstrapped_demos == 4


def test_minus_two_falls_through_to_shared_shuffle_when_labels_disabled() -> (
    None
):
    reset, minus_two, minus_one, seed_zero = _plans(max_labeled_demos=0)[:4]
    rng = random.Random(0)
    expected_minus_two = list(TASKS)
    rng.shuffle(expected_minus_two)
    expected_minus_two_size = rng.randint(1, 4)
    expected_zero = list(TASKS)
    rng.shuffle(expected_zero)
    expected_zero_size = rng.randint(1, 4)

    assert reset.kind is FewshotSeedKind.RESET
    assert minus_two.kind is FewshotSeedKind.BOOTSTRAP
    assert minus_two.trainset_task_hashes == tuple(expected_minus_two)
    assert minus_two.max_bootstrapped_demos == expected_minus_two_size
    assert minus_one.trainset_task_hashes == TASKS
    assert seed_zero.trainset_task_hashes == tuple(expected_zero)
    assert seed_zero.max_bootstrapped_demos == expected_zero_size


def test_shared_rng_checkpoint_flows_from_dataset_through_bootstrap() -> None:
    oracle = random.Random(23)
    sampled = tuple(oracle.sample(range(7), 4))
    checkpoint = Miprov2RngCheckpoint.after_validation_sampling(
        seed=23,
        population_size=7,
        sample_indices=sampled,
    )
    prior = checkpoint.draws[0]

    planning = _planning(
        max_labeled_demos=0,
        rng_checkpoint=checkpoint,
    )

    for candidate_seed in (-2, 0, 1, 2):
        expected = list(TASKS)
        oracle.shuffle(expected)
        expected_size = oracle.randint(1, 4)
        plan = next(
            item
            for item in planning.plans
            if item.candidate_seed == candidate_seed
        )
        assert plan.trainset_task_hashes == tuple(expected)
        assert plan.max_bootstrapped_demos == expected_size
    assert planning.rng_checkpoint.state == Miprov2RandomState.from_random(
        oracle
    )
    assert planning.rng_checkpoint.draws[0] == prior
    assert [
        (draw.phase, draw.operation)
        for draw in planning.rng_checkpoint.draws[1:]
    ] == [("bootstrap", "shuffle"), ("bootstrap", "randint")] * 4


def test_validation_rng_checkpoint_rejects_drift() -> None:
    manual = Miprov2RngCheckpoint.after_validation_sampling(
        seed=23,
        population_size=7,
        sample_indices=None,
    )
    assert manual == Miprov2RngCheckpoint.seeded(23)

    with pytest.raises(ValueError, match="validation sample"):
        Miprov2RngCheckpoint.after_validation_sampling(
            seed=23,
            population_size=7,
            sample_indices=(0, 1, 2, 3),
        )


def test_zero_shot_uses_three_bootstrapped_proposal_demos_only() -> None:
    planning = _planning(
        max_bootstrapped_demos=0,
        max_labeled_demos=0,
        zeroshot_opt=True,
    )

    assert planning.proposal_max_bootstrapped_demos == (
        ZERO_SHOT_BOOTSTRAPPED_DEMOS_IN_PROPOSAL
    )
    assert planning.proposal_max_labeled_demos == 0
    assert planning.study_uses_demo_candidates is False
    minus_two = planning.plans[1]
    assert minus_two.kind is FewshotSeedKind.BOOTSTRAP
    assert minus_two.max_labeled_demos == 0
    assert 1 <= minus_two.max_bootstrapped_demos <= 3


def test_bootstrap_planning_checkpoint_roundtrip_and_bindings() -> None:
    planning = _planning(max_labeled_demos=0)
    restored = type(planning).model_validate_json(planning.model_dump_json())
    assert restored == planning

    changed = _bindings().model_copy(
        update={"task_route_identity_hash": _identity("another-task-route")}
    )
    changed_planning = _planning(
        bindings=changed,
        max_labeled_demos=0,
    )
    assert planning.plans[0].identity_hash() != (
        changed_planning.plans[0].identity_hash()
    )
    with pytest.raises(ValueError, match="another candidate"):
        next_bootstrap_attempt(
            changed_planning.plans[2],
            initial_compiler_state(planning.plans[2]),
        )


def test_labeled_selection_uses_one_local_random_zero_stream() -> None:
    selection = plan_labeled_selection(
        component_ids=COMPONENTS,
        trainset_size=5,
        k=3,
        sample=True,
    )
    rng = random.Random(0)
    population = list(range(5))

    assert selection.per_component_task_indices == (
        tuple(rng.sample(population, 3)),
        tuple(rng.sample(population, 3)),
    )


def test_repeated_trace_pickle_matches_frozen_dspy_golden_vector() -> None:
    steps = [
        ObservedTraceStep(
            trace_index=0,
            component_id="first",
            inputs={"question": "2+2?"},
            outputs={"answer": "4"},
        ),
        ObservedTraceStep(
            trace_index=1,
            component_id="first",
            inputs={
                "payload": {
                    "items": [1, 2.5, False, None, "é"],
                }
            },
            outputs={"result": ["ok", {"n": 2**80}]},
        ),
    ]

    encoded = _frozen_dspy_demo_tuple_pickle(steps)

    assert len(encoded) == 250
    assert hashlib.sha256(encoded).hexdigest() == (
        "2cefcb67c77b21358cfd7cd07e90a61ccd03563a6ddca8276e81cae644b70f75"
    )


def test_teacher_copy_and_reset_plan_matches_reference() -> None:
    bootstrap = _plans(explicit_teacher=True)[2]

    assert bootstrap.teacher is not None
    assert bootstrap.teacher.source is TeacherSource.EXPLICIT
    assert bootstrap.teacher.initial_copy == "deepcopy"
    assert bootstrap.teacher.reset_before_labeled_compile is True
    assert bootstrap.teacher.labeled_selection is not None

    already_compiled = _plans(teacher_compiled=True)[2]
    assert already_compiled.teacher is not None
    assert already_compiled.teacher.reset_before_labeled_compile is False
    assert already_compiled.teacher.labeled_selection is None


def test_attempt_plan_excludes_current_demo_and_only_copies_on_retry() -> None:
    plan = _plans(max_rounds=2)[2]
    state = initial_compiler_state(plan)
    first = next_bootstrap_attempt(plan, state)
    assert first is not None
    assert first.exclude_equal_task_from_all_teacher_components is True
    assert first.restore_teacher_demos_after_effect is True
    assert first.copy_task_model is False
    assert first.generation_id is None
    assert first.temperature is None

    failed = fold_bootstrap_result(
        plan=plan,
        state=state,
        attempt=first,
        result=_result(first, score=0.0),
        metric_threshold=None,
        component_ids=COMPONENTS,
    )
    retry = next_bootstrap_attempt(plan, failed)
    assert retry is not None
    assert retry.task_index == first.task_index
    assert retry.round_index == 1
    assert retry.copy_task_model is True
    assert retry.generation_id == 1
    assert retry.temperature == 1.0


def test_invalid_numeric_boundaries_fail_before_rng_or_effect_planning() -> (
    None
):
    for overrides, message in (
        ({"num_candidate_sets": 0}, "num_candidate_sets"),
        ({"max_bootstrapped_demos": -1}, "max_bootstrapped_demos"),
        ({"max_labeled_demos": -1}, "max_labeled_demos"),
        ({"max_errors": -1}, "max_errors"),
        ({"max_rounds": 0}, "max_rounds"),
        ({"min_num_samples": 0}, "min_num_samples"),
        ({"metric_threshold": float("nan")}, "metric_threshold"),
    ):
        with pytest.raises(ValueError, match=message):
            _plans(**overrides)


def test_zero_bootstrapped_demos_is_valid_when_no_generic_seed_is_used() -> (
    None
):
    (reset,) = _plans(
        num_candidate_sets=1,
        max_bootstrapped_demos=0,
    )

    assert reset.kind is FewshotSeedKind.RESET


def test_generic_seed_requires_bootstrap_max_at_least_minimum() -> None:
    with pytest.raises(ValueError, match="at least min_num_samples"):
        _plans(
            num_candidate_sets=4,
            max_bootstrapped_demos=0,
            min_num_samples=1,
        )


@pytest.mark.parametrize(
    ("score", "threshold", "accepted"),
    [
        (0.0, None, False),
        (-1.0, 0.0, True),
        (0.49, 0.5, False),
        (0.5, 0.5, True),
        (-2.0, -1.0, False),
    ],
)
def test_fold_uses_threshold_truthiness(
    score: float, threshold: float | None, accepted: bool
) -> None:
    plan = _plans(max_rounds=1, metric_threshold=threshold)[2]
    state = initial_compiler_state(plan)
    attempt = next_bootstrap_attempt(plan, state)
    assert attempt is not None
    folded = fold_bootstrap_result(
        plan=plan,
        state=state,
        attempt=attempt,
        result=_result(attempt, score=score),
        metric_threshold=threshold,
        component_ids=COMPONENTS,
    )

    assert bool(folded.bootstrapped_task_indices) is accepted
    assert folded.task_cursor == 1


def test_absent_metric_accepts_and_unknown_trace_predictors_are_skipped() -> (
    None
):
    plan = _plans(metric_threshold=0.9)[2]
    state = initial_compiler_state(plan)
    attempt = next_bootstrap_attempt(plan, state)
    assert attempt is not None
    folded = fold_bootstrap_result(
        plan=plan,
        state=state,
        attempt=attempt,
        result=_result(
            attempt,
            score=None,
            metric_present=False,
            trace_steps=(
                ObservedTraceStep(
                    trace_index=0,
                    component_id=None,
                    inputs={"x": 1},
                    outputs={"y": 2},
                ),
                ObservedTraceStep(
                    trace_index=1,
                    component_id="unknown",
                    inputs={"x": 1},
                    outputs={"y": 2},
                ),
                ObservedTraceStep(
                    trace_index=2,
                    component_id="first",
                    inputs={"question": "q"},
                    outputs={"draft": "d"},
                ),
            ),
        ),
        metric_threshold=0.9,
        component_ids=COMPONENTS,
    )

    assert folded.bootstrapped_task_indices == (0,)
    assert set(folded.augmented_demos) == set(COMPONENTS)
    assert folded.augmented_demos["second"] == ()
    demo = folded.augmented_demos["first"][0]
    assert demo.source_trace_index == 2
    assert demo.score is None
    assert demo.source_kind is DemoSourceKind.BOOTSTRAPPED


def test_repeated_trace_uses_hash_seeded_earlier_or_last_choice() -> None:
    assert (
        MIPROV2_TRACE_SELECTION_PROJECTION_VERSION
        == "dspy_example_pickle_protocol4_cpython/v1"
    )
    plan = _plans()[2]
    state = initial_compiler_state(plan)
    attempt = next_bootstrap_attempt(plan, state)
    assert attempt is not None
    steps = tuple(
        ObservedTraceStep(
            trace_index=index,
            component_id="first",
            inputs={"call": index},
            outputs={"value": index},
        )
        for index in range(3)
    )
    folded_a = fold_bootstrap_result(
        plan=plan,
        state=state,
        attempt=attempt,
        result=_result(attempt, trace_steps=steps),
        metric_threshold=None,
        component_ids=COMPONENTS,
    )
    folded_b = fold_bootstrap_result(
        plan=plan,
        state=state,
        attempt=attempt,
        result=_result(attempt, trace_steps=steps),
        metric_threshold=None,
        component_ids=COMPONENTS,
    )

    selected_a = folded_a.augmented_demos["first"][0].source_trace_index
    selected_b = folded_b.augmented_demos["first"][0].source_trace_index
    rng = random.Random(
        hashlib.sha256(_frozen_dspy_demo_tuple_pickle(list(steps))).hexdigest()
    )
    expected = (
        rng.choice(steps[:-1]).trace_index
        if rng.random() < 0.5
        else steps[-1].trace_index
    )
    assert selected_a == selected_b
    assert selected_a == expected


def test_error_count_is_local_to_each_candidate_compiler() -> None:
    first_plan = _plans(max_errors=2, max_rounds=2)[2]
    first_state = initial_compiler_state(first_plan)
    attempt = next_bootstrap_attempt(first_plan, first_state)
    assert attempt is not None
    after_one = fold_bootstrap_result(
        plan=first_plan,
        state=first_state,
        attempt=attempt,
        result=_result(
            attempt,
            score=None,
            metric_present=False,
            error="provider failed",
        ),
        metric_threshold=None,
        component_ids=COMPONENTS,
    )
    assert after_one.error_count == 1

    retry = next_bootstrap_attempt(first_plan, after_one)
    assert retry is not None
    terminal = fold_bootstrap_result(
        plan=first_plan,
        state=after_one,
        attempt=retry,
        result=_result(
            retry,
            score=None,
            metric_present=False,
            error="metric failed",
        ),
        metric_threshold=None,
        component_ids=COMPONENTS,
    )
    restored = type(terminal).model_validate_json(terminal.model_dump_json())
    with pytest.raises(BootstrapErrorLimitReached) as caught:
        next_bootstrap_attempt(first_plan, restored)
    assert caught.value.error_count == 2

    second_plan = _plans(max_errors=2)[3]
    assert initial_compiler_state(second_plan).error_count == 0


def test_success_restores_loop_to_next_task_and_stops_at_demo_limit() -> None:
    plan = _plans(max_bootstrapped_demos=1, max_rounds=3)[2]
    state = initial_compiler_state(plan)
    attempt = next_bootstrap_attempt(plan, state)
    assert attempt is not None
    state = fold_bootstrap_result(
        plan=plan,
        state=state,
        attempt=attempt,
        result=_result(attempt),
        metric_threshold=None,
        component_ids=COMPONENTS,
    )

    assert state.task_cursor == 1
    assert state.round_cursor == 0
    assert next_bootstrap_attempt(plan, state) is None


def test_labels_only_and_reset_materialization_preserve_component_order() -> (
    None
):
    reset, labels, *_ = _plans()
    trainset = _labeled_trainset()
    reset_set = materialize_reset_demo_set(
        plan=reset, component_ids=COMPONENTS
    )
    labels_set = materialize_labels_only_demo_set(
        plan=labels, labeled_trainset=trainset
    )

    assert (
        tuple(item.component_id for item in reset_set.components) == COMPONENTS
    )
    assert all(not item.demos for item in reset_set.components)
    assert (
        tuple(item.component_id for item in labels_set.components)
        == COMPONENTS
    )
    assert all(len(item.demos) == 4 for item in labels_set.components)
    assert all(
        demo.source_kind is DemoSourceKind.LABELED
        for component in labels_set.components
        for demo in component.demos
    )


def test_train_is_augmented_first_and_raw_pool_narrows_across_components() -> (
    None
):
    plan = _plans(max_bootstrapped_demos=2, max_labeled_demos=4)[2]
    state = initial_compiler_state(plan)
    attempt = next_bootstrap_attempt(plan, state)
    assert attempt is not None
    state = fold_bootstrap_result(
        plan=plan,
        state=state,
        attempt=attempt,
        result=_result(
            attempt,
            trace_steps=(
                ObservedTraceStep(
                    trace_index=0,
                    component_id="first",
                    inputs={"question": "t0"},
                    outputs={"draft": "boot-first"},
                ),
                ObservedTraceStep(
                    trace_index=1,
                    component_id="second",
                    inputs={"draft": "boot-first"},
                    outputs={"answer": "boot-second"},
                ),
            ),
        ),
        metric_threshold=None,
        component_ids=COMPONENTS,
    )
    materialized = materialize_bootstrap_demo_set(
        plan=plan,
        state=state,
        labeled_trainset=_labeled_trainset(),
        component_ids=COMPONENTS,
    )

    first = materialized.demos_for("first")
    second = materialized.demos_for("second")
    assert first[0].augmented is True
    assert second[0].augmented is True

    validation = list(_labeled_trainset()[1:])
    random.Random(0).shuffle(validation)
    rng = random.Random(0)
    first_raw = rng.sample(validation, 3)
    second_raw = rng.sample(first_raw, 3)
    assert [demo.source_task_hash for demo in first[1:]] == [
        task.source_task_hash for task in first_raw
    ]
    assert [demo.source_task_hash for demo in second[1:]] == [
        task.source_task_hash for task in second_raw
    ]


def test_materialization_rejects_rows_not_in_generic_plan_order() -> None:
    generic = _plans()[3]
    assert generic.candidate_seed == 0
    state = initial_compiler_state(generic)

    with pytest.raises(ValueError, match="candidate plan task order"):
        materialize_bootstrap_demo_set(
            plan=generic,
            state=state,
            labeled_trainset=_labeled_trainset(),
            component_ids=COMPONENTS,
        )

    ordered = {task.source_task_hash: task for task in _labeled_trainset()}
    materialized = materialize_bootstrap_demo_set(
        plan=generic,
        state=state,
        labeled_trainset=tuple(
            ordered[identity] for identity in generic.trainset_task_hashes
        ),
        component_ids=COMPONENTS,
    )
    assert tuple(item.component_id for item in materialized.components) == (
        COMPONENTS
    )


def test_result_and_state_are_bound_to_exact_attempt_and_candidate() -> None:
    first_plan, second_plan = _plans()[2:4]
    state = initial_compiler_state(first_plan)
    attempt = next_bootstrap_attempt(first_plan, state)
    assert attempt is not None
    wrong_result = _result(attempt).model_copy(
        update={"attempt_identity_hash": "wrong"}
    )

    with pytest.raises(ValueError, match="another attempt"):
        fold_bootstrap_result(
            plan=first_plan,
            state=state,
            attempt=attempt,
            result=wrong_result,
            metric_threshold=None,
            component_ids=COMPONENTS,
        )
    with pytest.raises(ValueError, match="another candidate"):
        next_bootstrap_attempt(second_plan, state)


def test_acceptance_threshold_is_bound_into_candidate_plan() -> None:
    plan = _plans(metric_threshold=0.5)[2]
    state = initial_compiler_state(plan)
    attempt = next_bootstrap_attempt(plan, state)
    assert attempt is not None

    with pytest.raises(ValueError, match="metric_threshold"):
        fold_bootstrap_result(
            plan=plan,
            state=state,
            attempt=attempt,
            result=_result(attempt, score=0.75),
            metric_threshold=None,
            component_ids=COMPONENTS,
        )


def test_rng_checkpoint_replays_seed_draws_results_and_terminal_state() -> (
    None
):
    checkpoint = Miprov2RngCheckpoint.after_validation_sampling(
        seed=23,
        population_size=7,
        sample_indices=(6, 2, 0),
    )
    payload = checkpoint.model_dump(mode="json")

    wrong_seed = {**payload, "seed": 24}
    with pytest.raises(ValueError, match="does not match replay"):
        Miprov2RngCheckpoint.model_validate(wrong_seed)

    wrong_result = {
        **payload,
        "draws": [
            {**payload["draws"][0], "result": [0, 2, 6]},
        ],
    }
    with pytest.raises(ValueError, match="result does not match replay"):
        Miprov2RngCheckpoint.model_validate(wrong_result)

    wrong_state = {
        **payload,
        "state": Miprov2RandomState.seeded(23).model_dump(mode="json"),
    }
    with pytest.raises(ValueError, match="state does not match"):
        Miprov2RngCheckpoint.model_validate(wrong_state)


def test_planning_result_reconstructs_exact_sequence_and_rng_cursor() -> None:
    planning = _planning()
    payload = planning.model_dump(mode="json")

    reordered = {
        **payload,
        "plans": list(reversed(payload["plans"])),
    }
    with pytest.raises(ValueError, match="count, order, seeds, kinds"):
        type(planning).model_validate(reordered)

    wrong_count = {
        **payload,
        "inputs": {
            **payload["inputs"],
            "num_candidate_sets": payload["inputs"]["num_candidate_sets"] + 1,
        },
    }
    with pytest.raises(ValueError, match="count, order, seeds, kinds"):
        type(planning).model_validate(wrong_count)

    wrong_cursor = {
        **payload,
        "rng_checkpoint": payload["initial_rng_checkpoint"],
    }
    with pytest.raises(ValueError, match="RNG cursor"):
        type(planning).model_validate(wrong_cursor)


def test_bootstrap_ledger_rejects_restart_forgery() -> None:
    plan = _plans(max_bootstrapped_demos=2)[2]
    state = initial_compiler_state(plan)
    attempt = next_bootstrap_attempt(plan, state)
    assert attempt is not None
    state = fold_bootstrap_result(
        plan=plan,
        state=state,
        attempt=attempt,
        result=_result(attempt),
        metric_threshold=None,
        component_ids=COMPONENTS,
    )
    restored = type(state).model_validate_json(state.model_dump_json())
    assert next_bootstrap_attempt(plan, restored) is not None

    forged_cursor = restored.model_copy(update={"task_cursor": 0})
    with pytest.raises(ValueError, match="cursors or demos"):
        next_bootstrap_attempt(plan, forged_cursor)

    event = restored.evidence[0]
    forged_attempt = event.attempt.model_copy(update={"task_index": 1})
    forged_event = event.model_copy(update={"attempt": forged_attempt})
    forged_ledger = restored.model_copy(update={"evidence": (forged_event,)})
    with pytest.raises(ValueError, match="skips or rewrites"):
        next_bootstrap_attempt(plan, forged_ledger)
