"""Source-derived differential tests against frozen DSPy MIPROv2."""

from __future__ import annotations

import hashlib
from typing import Any

import pytest

from whetstone.core.identity import compute_identity_hash
from whetstone.optimization.miprov2.bootstrap import (
    FewshotSeedKind,
    TeacherSource,
    create_fewshot_candidate_plans,
)
from whetstone.optimization.miprov2.control import MIPROV2_REFERENCE_COMMIT
from whetstone.optimization.miprov2.demo import (
    ComponentDemo,
    ComponentDemoSequence,
    ComponentDemoSet,
    DemoSourceKind,
)
from whetstone.optimization.miprov2.proposal import (
    Miprov2DatasetExample,
    Miprov2PromptComponent,
    Miprov2ProposalRequest,
    Miprov2ProposalResponse,
    fold_proposal_response,
    plan_next_proposal_request,
    proposal_candidates_from_demo_sets,
    start_miprov2_proposal,
)
from whetstone.optimization.miprov2.rng import (
    Miprov2DurableBindings,
    Miprov2RngCheckpoint,
)
from whetstone.optimization.miprov2.study import Miprov2ParameterSpace

ORACLE_COMMIT = "6f68dcdb3ef46d70bf0c12596699ebc44e82d6b0"
CONTROL_HASH = "c" * 64
PROMPT_ADAPTER_HASH = "e" * 64


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _bindings() -> Miprov2DurableBindings:
    return Miprov2DurableBindings(
        control_identity_hash=CONTROL_HASH,
        prompt_route_identity_hash=_hash("prompt-route"),
        task_route_identity_hash=_hash("task-route"),
        execution_policy_identity_hash=_hash("execution-policy"),
        prompt_adapter_identity_hash=PROMPT_ADAPTER_HASH,
        base_candidate_identity_hash=_hash("base"),
        teacher_candidate_identity_hash=_hash("teacher"),
    )


def _components(count: int) -> tuple[Miprov2PromptComponent, ...]:
    return tuple(
        Miprov2PromptComponent(
            component_id=f"component-{index}",
            template=f"Current component {index}: {{input}}",
            allowed_placeholders=("input",),
            rendering_rules="Substitute the named native input.",
            example_execution=f"Current component {index}: example",
        )
        for index in range(count)
    )


def _empty_demo_sets(
    component_ids: tuple[str, ...],
) -> tuple[ComponentDemoSet, ...]:
    return tuple(
        ComponentDemoSet(
            candidate_seed=seed,
            components=tuple(
                ComponentDemoSequence(component_id=component_id)
                for component_id in component_ids
            ),
        )
        for seed in (-3, -2, -1, 0)
    )


def _component_projection(
    demo_set: ComponentDemoSet,
    component_id: str,
) -> ComponentDemoSet:
    return ComponentDemoSet(
        candidate_seed=demo_set.candidate_seed,
        components=(
            ComponentDemoSequence(
                component_id=component_id,
                demos=demo_set.demos_for(component_id),
            ),
        ),
    )


def _proposal_trace(
    component_count: int,
    demo_sets: tuple[ComponentDemoSet, ...],
) -> tuple[
    tuple[tuple[int | None, int | None, str], ...],
    tuple[tuple[str, ...], ...],
]:
    components = _components(component_count)
    field_order: dict[str, tuple[str, ...]] = {
        component.component_id: ("input", "output") for component in components
    }
    bridged = proposal_candidates_from_demo_sets(
        demo_sets,
        components=components,
        component_field_order=field_order,
    )
    state = start_miprov2_proposal(
        bindings=_bindings(),
        components=components,
        trainset=(
            Miprov2DatasetExample(
                task_identity=_hash("proposal-task"),
                rendered_record="input=example; expected=answer",
            ),
        ),
        demo_candidates=bridged,
        num_candidates=2,
        view_data_batch_size=10,
        init_temperature=0.7,
        data_aware=False,
        program_aware=False,
        tip_aware=False,
        fewshot_aware=True,
        rng_checkpoint=Miprov2RngCheckpoint.seeded(11),
    )
    trace: list[tuple[int | None, int | None, str]] = []
    while True:
        planned = plan_next_proposal_request(state)
        state = planned.state
        if planned.request is None:
            return tuple(trace), state.instruction_pools
        request: Miprov2ProposalRequest = planned.request
        trace.append(
            (
                request.component_index,
                request.proposal_index,
                request.effect,
            )
        )
        state = fold_proposal_response(
            state,
            Miprov2ProposalResponse(
                request_identity_hash=request.identity_hash,
                text=(
                    "Instruction: replacement "
                    f"{request.component_index}-{request.proposal_index} "
                    "{input}"
                ),
                evidence={"oracle": "scripted"},
            ),
        )


@pytest.mark.parametrize("component_count", [1, 2])
def test_bootstrap_proposal_and_search_shapes_match_source_oracle(
    component_count: int,
) -> None:
    """Internal frozen machinery retains DSPy's predictor-major behavior."""

    assert MIPROV2_REFERENCE_COMMIT == ORACLE_COMMIT
    component_ids = tuple(
        f"component-{index}" for index in range(component_count)
    )
    tasks = tuple(_hash(f"task-{index}") for index in range(4))
    planning = create_fewshot_candidate_plans(
        bindings=_bindings(),
        component_ids=component_ids,
        trainset_task_identities=tasks,
        num_candidate_sets=4,
        max_bootstrapped_demos=3,
        max_labeled_demos=2,
        max_errors=2,
        rng_checkpoint=Miprov2RngCheckpoint.seeded(0),
        explicit_teacher=True,
    )

    assert tuple(plan.candidate_seed for plan in planning.plans) == (
        -3,
        -2,
        -1,
        0,
    )
    assert tuple(plan.kind for plan in planning.plans) == (
        FewshotSeedKind.RESET,
        FewshotSeedKind.LABELS_ONLY,
        FewshotSeedKind.BOOTSTRAP,
        FewshotSeedKind.BOOTSTRAP,
    )
    assert all(plan.component_ids == component_ids for plan in planning.plans)

    demo_sets = _empty_demo_sets(component_ids)
    proposal_trace, instruction_pools = _proposal_trace(
        component_count,
        demo_sets,
    )
    assert proposal_trace == tuple(
        (component_index, proposal_index, "instruction_proposal")
        for component_index in range(component_count)
        for proposal_index in range(2)
    )

    space = Miprov2ParameterSpace(
        instruction_pool_identity_hashes=tuple(
            tuple(
                compute_identity_hash(
                    schema="whetstone.miprov2_instruction",
                    schema_version=1,
                    payload={"instruction": instruction},
                )
                for instruction in pool
            )
            for pool in instruction_pools
        ),
        demo_pool_identity_hashes=tuple(
            tuple(
                _component_projection(demo_set, component_id).identity_hash()
                for demo_set in demo_sets
            )
            for component_id in component_ids
        ),
    )
    expected_parameters = tuple(
        name
        for component_index in range(component_count)
        for name in (
            f"{component_index}_predictor_instruction",
            f"{component_index}_predictor_demos",
        )
    )
    assert space.parameter_names == expected_parameters
    assert space.instruction_candidate_counts == (2,) * component_count
    assert space.demo_candidate_counts == (4,) * component_count


def test_demo_category_identity_is_predictor_specific() -> None:
    component_ids = ("component-0", "component-1")
    original = _empty_demo_sets(component_ids)[0]
    component_b_demo = ComponentDemo(
        component_id="component-1",
        source_kind=DemoSourceKind.LABELED,
        inputs={"input": "changed"},
        outputs={"output": "changed"},
        augmented=False,
        source_task_identity=_hash("task"),
        source_rollout_identity=_hash("rollout"),
        source_trace_identity=_hash("trace"),
        source_output_identity=_hash("output"),
        source_score_identity=_hash("score"),
        source_trace_index=None,
        score=None,
        acceptance_identity_hash=_hash("acceptance"),
    )
    changed_b = original.model_copy(
        update={
            "components": (
                original.components[0],
                ComponentDemoSequence(
                    component_id="component-1",
                    demos=(component_b_demo,),
                ),
            )
        }
    )

    assert (
        _component_projection(original, "component-0").identity_hash()
        == _component_projection(changed_b, "component-0").identity_hash()
    )
    assert (
        _component_projection(original, "component-1").identity_hash()
        != _component_projection(changed_b, "component-1").identity_hash()
    )


def test_explicit_compiled_and_uncompiled_teacher_semantics_match_oracle() -> (
    None
):
    tasks = tuple(_hash(f"teacher-task-{index}") for index in range(4))
    common: dict[str, Any] = {
        "bindings": _bindings(),
        "component_ids": ("component-0", "component-1"),
        "trainset_task_identities": tasks,
        "num_candidate_sets": 3,
        "max_bootstrapped_demos": 3,
        "max_labeled_demos": 2,
        "max_errors": 2,
        "rng_checkpoint": Miprov2RngCheckpoint.seeded(0),
        "explicit_teacher": True,
    }
    uncompiled = create_fewshot_candidate_plans(
        **common,
        teacher_compiled=False,
    ).plans[2]
    compiled = create_fewshot_candidate_plans(
        **common,
        teacher_compiled=True,
    ).plans[2]

    assert uncompiled.teacher is not None
    assert compiled.teacher is not None
    assert (
        uncompiled.teacher.source,
        uncompiled.teacher.initial_copy,
        uncompiled.teacher.reset_before_labeled_compile,
        uncompiled.teacher.labeled_selection is not None,
    ) == (TeacherSource.EXPLICIT, "deepcopy", True, True)
    assert (
        compiled.teacher.source,
        compiled.teacher.initial_copy,
        compiled.teacher.reset_before_labeled_compile,
        compiled.teacher.labeled_selection,
    ) == (TeacherSource.EXPLICIT, "deepcopy", False, None)
