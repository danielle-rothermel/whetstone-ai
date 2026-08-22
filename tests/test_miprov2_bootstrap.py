"""Bootstrap plan / fold / resume without the harness loop."""

from __future__ import annotations

from whetstone.core.identity import ImmutableJsonObject
from whetstone.optim.miprov2.bootstrap import (
    BootstrapGenerationResult,
    FewshotSeedKind,
    create_fewshot_candidate_plans,
    fold_bootstrap_result,
    initial_compiler_state,
    next_bootstrap_attempt,
)
from whetstone.optim.miprov2.demo import ObservedTraceStep
from whetstone.optim.miprov2.demo_mode import Miprov2DemoMode
from whetstone.optim.miprov2.rng import (
    Miprov2DurableBindings,
    Miprov2RngCheckpoint,
)

_HASH_A = "a" * 64
_HASH_B = "b" * 64
_HASH_C = "c" * 64
_COMPONENT = "generate"


def _bindings() -> Miprov2DurableBindings:
    return Miprov2DurableBindings(
        control_identity_hash=_HASH_A,
        prompt_route_identity_hash=_HASH_A,
        task_route_identity_hash=_HASH_A,
        execution_policy_identity_hash=_HASH_A,
        prompt_adapter_identity_hash=_HASH_A,
        proposal_executor_policy_identity_hash=_HASH_A,
        proposal_transport_durability_identity_hash=_HASH_A,
        base_candidate_identity_hash=_HASH_A,
        teacher_candidate_identity_hash=_HASH_B,
    )


def _planning(*, demo_mode: Miprov2DemoMode = Miprov2DemoMode.FEWSHOT):
    maxima = (0, 0) if demo_mode is Miprov2DemoMode.ZEROSHOT else (1, 1)
    return create_fewshot_candidate_plans(
        bindings=_bindings(),
        component_ids=(_COMPONENT,),
        trainset_task_hashes=(_HASH_C,),
        num_candidate_sets=4,
        max_bootstrapped_demos=maxima[0],
        max_labeled_demos=maxima[1],
        max_errors=4,
        rng_checkpoint=Miprov2RngCheckpoint.seeded(9),
        demo_mode=demo_mode,
    )


def _bootstrap_plan():
    planning = _planning()
    return next(
        plan
        for plan in planning.plans
        if plan.kind is FewshotSeedKind.BOOTSTRAP
    )


def test_fewshot_plans_bootstrap_candidates() -> None:
    planning = _planning()
    assert planning.demo_mode is Miprov2DemoMode.FEWSHOT
    assert planning.study_uses_demo_candidates is True
    assert any(
        plan.kind is FewshotSeedKind.BOOTSTRAP for plan in planning.plans
    )


def test_zeroshot_plans_no_fewshot_candidates() -> None:
    planning = _planning(demo_mode=Miprov2DemoMode.ZEROSHOT)
    assert planning.plans == ()
    assert planning.proposal_max_bootstrapped_demos == 0
    assert planning.proposal_max_labeled_demos == 0


def test_fold_and_resume_keep_the_next_attempt() -> None:
    plan = _bootstrap_plan()
    state = initial_compiler_state(plan)
    first = next_bootstrap_attempt(plan, state)
    assert first is not None

    restored = type(state).model_validate(state.model_dump(mode="json"))
    assert next_bootstrap_attempt(plan, restored) == first

    folded = fold_bootstrap_result(
        plan=plan,
        state=state,
        attempt=first,
        result=BootstrapGenerationResult(
            attempt_identity_hash=first.identity_hash(),
            source_generation_hash=_HASH_A,
            source_trace_hash=_HASH_A,
            source_output_hash=_HASH_A,
            source_score_hash=_HASH_A,
            metric_present=True,
            score=1.0,
            trace_steps=(
                ObservedTraceStep(
                    trace_index=0,
                    component_id=_COMPONENT,
                    inputs=ImmutableJsonObject({"prompt": "hi"}),
                    outputs=ImmutableJsonObject({"response": "hello"}),
                ),
            ),
        ),
        metric_threshold=None,
        component_ids=(_COMPONENT,),
    )
    resumed = type(folded).model_validate(folded.model_dump(mode="json"))
    assert next_bootstrap_attempt(plan, resumed) == next_bootstrap_attempt(
        plan, folded
    )
    assert folded.attempt_count == 1
    assert folded.bootstrapped_task_indices == (first.task_index,)
