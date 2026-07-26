"""Source-derived differential acceptance tests for durable MIPROv2.

The oracle is DSPy commit ``6f68dcdb3ef46d70bf0c12596699ebc44e82d6b0``.
These tests intentionally compare normalized control-flow decisions rather
than generated text.  The literal fixtures below come from:

* ``dspy/teleprompt/utils.py::create_n_fewshot_demo_sets`` for the special
  seed range, predictor-major demo shape, and evaluation-exception score;
* ``dspy/teleprompt/bootstrap.py::_prepare_student_and_teacher`` for explicit
  teacher copy/reset and compiled-teacher behavior;
* ``dspy/propose/grounded_proposer.py::propose_instructions_for_program`` for
  predictor-major proposal order;
* ``dspy/teleprompt/mipro_optimizer_v2.py::_select_and_insert_...``,
  ``_get_param_distributions``, ``objective``, and
  ``_perform_full_evaluation`` for parameter order, minibatch statistics,
  promotion-before-tell ordering, and strict full-evaluation winner updates.
"""

from __future__ import annotations

import hashlib
from typing import Any, Literal, cast

import pytest
from dr_code.eval import (
    DefinitionRef,
    EvalConfig,
    RepeatPlan,
    SamplingDefinition,
    TaskSet,
)
from dr_code.eval.identity import SCHEMA_EVAL_CONFIG, identity_hash_for
from dr_store import ObjectStore, SqliteBackend

from whetstone.graph.rollout import EvaluationRole
from whetstone.optimization.identity import TypedRef, compute_identity_hash
from whetstone.optimization.miprov2_bootstrap import (
    FewshotSeedKind,
    TeacherSource,
    create_fewshot_candidate_plans,
)
from whetstone.optimization.miprov2_control import (
    MIPROV2_CANDIDATE_RENDERER_VERSION,
    Miprov2ComponentSpec,
    Miprov2ProgramLayout,
)
from whetstone.optimization.miprov2_demo import (
    ComponentDemo,
    ComponentDemoSequence,
    ComponentDemoSet,
    DemoSourceKind,
)
from whetstone.optimization.miprov2_eval_config import (
    Miprov2EvalConfigBinding,
    Miprov2EvalConfigBindingRequest,
    Miprov2EvalPurpose,
    Miprov2EvaluationExecutionPolicy,
    derive_eval_config_reference,
)
from whetstone.optimization.miprov2_evidence import (
    MIPROV2_INTENT_CONTEXT_SCHEMA,
    Miprov2IntentContext,
    persist_miprov2_intent_context,
    resolve_miprov2_evaluation_failure,
)
from whetstone.optimization.miprov2_proposal import (
    Miprov2DatasetExample,
    Miprov2PromptComponent,
    Miprov2ProposalRequest,
    Miprov2ProposalResponse,
    fold_proposal_response,
    plan_next_proposal_request,
    proposal_candidates_from_demo_sets,
    start_miprov2_proposal,
)
from whetstone.optimization.miprov2_rng import (
    Miprov2DurableBindings,
    Miprov2RngCheckpoint,
)
from whetstone.optimization.miprov2_study import (
    EVALUATION_EVIDENCE_SCHEMA,
    EVALUATION_FAILURE_SCHEMA,
    MIPROV2_CANDIDATE_RENDERING_SCHEMA,
    MIPROV2_CANDIDATE_RENDERING_SCHEMA_VERSION,
    MIPROV2_REFERENCE_COMMIT,
    REWARD_SCHEMA,
    EvaluationBinding,
    Miprov2CandidateAssemblyBinding,
    Miprov2ParameterSpace,
    Miprov2Study,
    Miprov2StudySchedule,
    VerifiedEvaluationCitation,
)
from whetstone.optimization.prompt_program import (
    PROMPT_PROGRAM_PAYLOAD_FIELD,
    PromptProgram,
    PromptProgramComponent,
)
from whetstone.optimization.schema import (
    Candidate,
    EvaluationIntent,
    IntentOutcome,
    IntentResolution,
    ResolutionClass,
    ResolutionDetail,
    candidate_reference,
    eval_config_reference,
)

ORACLE_COMMIT = "6f68dcdb3ef46d70bf0c12596699ebc44e82d6b0"
RUN_ID = "miprov2-differential"
CONTROL_HASH = "c" * 64
REWARD_POLICY_HASH = "d" * 64
PROMPT_ADAPTER_HASH = "e" * 64
BASE_CANDIDATE = candidate_reference(
    Candidate(
        candidate_id="base",
        base_ref="root",
        payload={"user_prompt_template": "base"},
    )
)


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _bindings() -> Miprov2DurableBindings:
    return Miprov2DurableBindings(
        control_identity_hash=CONTROL_HASH,
        prompt_route_identity_hash=_hash("prompt-route"),
        task_route_identity_hash=_hash("task-route"),
        execution_policy_identity_hash=_hash("execution-policy"),
        prompt_adapter_identity_hash=PROMPT_ADAPTER_HASH,
        base_candidate_identity_hash=BASE_CANDIDATE.identity_hash,
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
    """Compare one/two-predictor shapes without comparing generated text."""

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

    # Frozen utils.py uses range(-3, N - 3), collecting every program's demos
    # by predictor after each candidate has been built.
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
    expected_proposal_trace = tuple(
        (component_index, proposal_index, "instruction_proposal")
        for component_index in range(component_count)
        for proposal_index in range(2)
    )
    assert proposal_trace == expected_proposal_trace

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
                _component_projection(
                    demo_set,
                    component_id,
                ).identity_hash()
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
    assert space.baseline_params == tuple(
        (name, 0) for name in expected_parameters
    )


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
    teacher_tasks = tuple(_hash(f"teacher-task-{index}") for index in range(4))
    uncompiled = create_fewshot_candidate_plans(
        bindings=_bindings(),
        component_ids=("component-0", "component-1"),
        trainset_task_identities=teacher_tasks,
        num_candidate_sets=3,
        max_bootstrapped_demos=3,
        max_labeled_demos=2,
        max_errors=2,
        rng_checkpoint=Miprov2RngCheckpoint.seeded(0),
        explicit_teacher=True,
        teacher_compiled=False,
    ).plans[2]
    compiled = create_fewshot_candidate_plans(
        bindings=_bindings(),
        component_ids=("component-0", "component-1"),
        trainset_task_identities=teacher_tasks,
        num_candidate_sets=3,
        max_bootstrapped_demos=3,
        max_labeled_demos=2,
        max_errors=2,
        rng_checkpoint=Miprov2RngCheckpoint.seeded(0),
        explicit_teacher=True,
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


def _eval_config(sampling_hash: str) -> EvalConfig:
    definition = DefinitionRef(
        definition_id="differential-eval",
        version="1",
        schema_name="dr_code.eval_definition",
        identity_hash="a" * 64,
    )
    identity = identity_hash_for(
        schema=SCHEMA_EVAL_CONFIG,
        payload={
            "definition_identity": definition.identity_hash,
            "sampling_config": sampling_hash,
            "evaluation_procedure_config": "b" * 64,
            "aggregation_config": "f" * 64,
        },
    )
    return EvalConfig(
        definition_ref=definition,
        sampling_config_hash=sampling_hash,
        evaluation_procedure_config_hash="b" * 64,
        aggregation_config_hash="f" * 64,
        config_identity_hash=identity,
    )


def _eval_binding_artifact(
    *,
    tasks: tuple[str, ...],
    nonce: int,
    purpose: Miprov2EvalPurpose,
    effect_identity_hash: str,
    source: Any,
) -> Miprov2EvalConfigBinding:
    request = Miprov2EvalConfigBindingRequest(
        control_identity_hash=CONTROL_HASH,
        source_eval_config=source,
        purpose=purpose,
        effect_identity_hash=effect_identity_hash,
        execution_policy=Miprov2EvaluationExecutionPolicy(
            num_threads=1,
            max_errors=1,
            provide_traceback=False,
            task_model_identity_hash=_hash("task-model"),
            provider_execution_policy_hash=_hash("provider-policy"),
        ),
        task_batch_identities=tasks,
    )
    task_set = TaskSet(
        manifest_id=f"differential-tasks-{nonce}",
        version="1",
        dataset_revision="oracle",
        task_identities=tasks,
    )
    repeats = RepeatPlan(
        plan_id=f"differential-repeats-{nonce}",
        version="1",
        task_identities=tasks,
        repeat_count=1,
    )
    sampling = SamplingDefinition(
        definition_id="differential-sampling",
        version="1",
    ).materialize(
        {
            "task_set_hash": task_set.identity_hash(),
            "repeat_plan_hash": repeats.identity_hash(),
        }
    )
    return Miprov2EvalConfigBinding(
        request=request,
        task_set=task_set,
        repeat_plan=repeats,
        sampling_config=sampling,
        eval_config=derive_eval_config_reference(source, sampling),
    )


def _evaluation_binding(
    *,
    tasks: tuple[str, ...],
    nonce: int,
    purpose: Literal[
        "miprov2_baseline",
        "miprov2_sample",
        "miprov2_promotion",
    ],
    candidate_identity_hash: str,
    score: float,
    source: Any,
) -> EvaluationBinding:
    effect_hash = f"{25_000 + nonce:064x}"
    artifact = _eval_binding_artifact(
        tasks=tasks,
        nonce=nonce,
        purpose=cast(
            "Miprov2EvalPurpose",
            purpose.removeprefix("miprov2_"),
        ),
        effect_identity_hash=effect_hash,
        source=source,
    )
    evidence_ref = TypedRef(
        schema_name=EVALUATION_EVIDENCE_SCHEMA,
        content_hash=f"{30_000 + nonce:064x}",
    )
    reward_ref = TypedRef(
        schema_name=REWARD_SCHEMA,
        content_hash=f"{35_000 + nonce:064x}",
    )
    citation = VerifiedEvaluationCitation(
        run_id=RUN_ID,
        intent_id=f"intent-{nonce}",
        effect_identity_hash=effect_hash,
        purpose=purpose,
        candidate_identity_hash=candidate_identity_hash,
        task_batch_identities=tasks,
        validation_eval_source_identity_hash=source.identity_hash,
        eval_config_identity_hash=artifact.eval_config.identity_hash,
        eval_config_binding_identity_hash=artifact.identity_hash(),
        reward_policy_hash=REWARD_POLICY_HASH,
        evidence_ref=evidence_ref,
        reward_ref=reward_ref,
        normalized_score=score,
    )
    return EvaluationBinding(
        run_id=RUN_ID,
        intent_id=f"intent-{nonce}",
        effect_identity_hash=effect_hash,
        purpose=purpose,
        candidate_identity_hash=candidate_identity_hash,
        task_batch_identities=tasks,
        eval_config=artifact.eval_config,
        eval_config_binding=artifact,
        reward_policy_hash=REWARD_POLICY_HASH,
        reward_ref=reward_ref,
        evidence_citations=(citation,),
        normalized_score=score,
    )


def _candidate_assembly(
    study: Miprov2Study,
    params: Any,
) -> Miprov2CandidateAssemblyBinding:
    normalized = study.space.normalize(params)
    values = dict(normalized)
    combination = study.space.combination_identity_hash(normalized)
    instruction_index = values["0_predictor_instruction"]
    instruction = f"instruction-{instruction_index}"
    instruction_hash = compute_identity_hash(
        schema="whetstone.miprov2_instruction",
        schema_version=1,
        payload={"instruction": instruction},
    )
    rendering = {
        "control_identity_hash": CONTROL_HASH,
        "base_candidate_identity_hash": BASE_CANDIDATE.identity_hash,
        "categorical_combination_identity_hash": combination,
        "renderer_version": MIPROV2_CANDIDATE_RENDERER_VERSION,
        "components": [
            {
                "component_id": "component-0",
                "candidate_field": "user_prompt_template",
                "instruction_index": instruction_index,
                "instruction": instruction,
                "instruction_identity_hash": instruction_hash,
                "demo_index": None,
                "demo_set": None,
                "demo_identity_hash": None,
            }
        ],
    }
    program_hash = compute_identity_hash(
        schema=MIPROV2_CANDIDATE_RENDERING_SCHEMA,
        schema_version=MIPROV2_CANDIDATE_RENDERING_SCHEMA_VERSION,
        payload=rendering,
    )
    candidate = candidate_reference(
        Candidate(
            candidate_id=f"miprov2-{program_hash[:24]}",
            base_ref=BASE_CANDIDATE.record.base_ref,
            payload={
                **BASE_CANDIDATE.record.payload,
                "user_prompt_template": instruction,
                "miprov2_candidate_rendering": rendering,
                PROMPT_PROGRAM_PAYLOAD_FIELD: PromptProgram(
                    components=(
                        PromptProgramComponent(
                            component_id="component-0",
                            candidate_field="user_prompt_template",
                        ),
                    )
                ).model_dump(mode="json"),
            },
        )
    )
    return Miprov2CandidateAssemblyBinding(
        params=normalized,
        categorical_combination_identity_hash=combination,
        candidate=candidate,
        program_identity_hash=program_hash,
        control_identity_hash=CONTROL_HASH,
        base_candidate=BASE_CANDIDATE,
        program_layout=study.program_layout,
        prompt_adapter_identity_hash=PROMPT_ADAPTER_HASH,
    )


def _equal_minibatch_study() -> tuple[Miprov2Study, Any]:
    instruction_hashes = tuple(
        compute_identity_hash(
            schema="whetstone.miprov2_instruction",
            schema_version=1,
            payload={"instruction": f"instruction-{index}"},
        )
        for index in range(2)
    )
    space = Miprov2ParameterSpace(
        instruction_pool_identity_hashes=(instruction_hashes,)
    )
    schedule = Miprov2StudySchedule(
        num_trials=1,
        minibatch=True,
        minibatch_size=3,
        valset_size=3,
        minibatch_full_eval_steps=5,
    )
    tasks = tuple(_hash(f"validation-{index}") for index in range(3))
    source = eval_config_reference(_eval_config(_hash("source-sampling")))
    study = Miprov2Study(
        seed=9,
        space=space,
        schedule=schedule,
        run_id=RUN_ID,
        validation_task_identities=tasks,
        validation_eval_source=source,
        reward_policy_hash=REWARD_POLICY_HASH,
        control_identity_hash=CONTROL_HASH,
        prompt_adapter_identity_hash=PROMPT_ADAPTER_HASH,
        expected_base_candidate=BASE_CANDIDATE,
        program_layout=Miprov2ProgramLayout(
            layout_id="differential-layout",
            component_specs=(
                Miprov2ComponentSpec(
                    component_id="component-0",
                    candidate_field="user_prompt_template",
                    prompt_format_identity_hash=PROMPT_ADAPTER_HASH,
                ),
            ),
        ),
    )
    transcript = study.initial_transcript(
        baseline_score=0.25,
        baseline_evaluation=_evaluation_binding(
            tasks=tasks,
            nonce=0,
            purpose="miprov2_baseline",
            candidate_identity_hash=BASE_CANDIDATE.identity_hash,
            score=0.25,
            source=source,
        ),
    )
    return study, transcript


def test_equal_size_minibatch_stats_and_promotion_flow_match_oracle() -> None:
    study, transcript = _equal_minibatch_study()
    tasks = study.validation_task_identities
    suggestion = study.suggest_next(transcript)
    assembly = _candidate_assembly(study, suggestion.params)
    sample = _evaluation_binding(
        tasks=tasks,
        nonce=1,
        purpose="miprov2_sample",
        candidate_identity_hash=assembly.candidate.identity_hash,
        score=1.0,
        source=study.validation_eval_source,
    )

    selected = study.promotion_candidate(
        transcript,
        suggestion,
        score=1.0,
        evaluation=sample,
        candidate_assembly=assembly,
    )
    assert selected is not None
    transcript = study.record_sample(
        transcript,
        suggestion,
        score=1.0,
        evaluation=sample,
        candidate_assembly=assembly,
        promotion_full_score=0.2,
        promotion_evaluation=_evaluation_binding(
            tasks=tasks,
            nonce=2,
            purpose="miprov2_promotion",
            candidate_identity_hash=(
                selected.evaluated_candidate_identity_hash
            ),
            score=0.2,
            source=study.validation_eval_source,
        ),
    )

    # DSPy records full_eval from batch_size >= len(valset), but only the
    # explicit promotion may update best_program while minibatch=True.
    observation = transcript.samples[0]
    assert observation.batch_full_evaluation is True
    assert observation.promotion is not None
    assert observation.promotion.trial_number == suggestion.trial_number + 1
    assert observation.promotion.source_sample_trial_number == 1
    assert observation.promotion.minibatch_mean == 1.0
    assert [
        (trial.number, trial.value)
        for trial in study.reconstruct_study(transcript).trials
    ] == [(0, 0.25), (1, 1.0), (2, 0.2)]
    winner = study.best_full_evaluation(transcript)
    assert (winner.source, winner.score) == ("baseline", 0.25)


def _typed_put(
    store: ObjectStore,
    schema: str,
    record: dict[str, Any],
) -> TypedRef:
    ref, _ = store.put(schema, record)
    return TypedRef(schema_name=ref.schema, content_hash=ref.content_hash)


def test_evaluation_failure_maps_to_literal_dspy_zero(
    tmp_path: Any,
) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "miprov2-oracle.sqlite"))
    tasks = (_hash("failed-task"),)
    source = eval_config_reference(_eval_config(_hash("failure-source")))
    effect_hash = _hash("failure-effect")
    artifact = _eval_binding_artifact(
        tasks=tasks,
        nonce=99,
        purpose="sample",
        effect_identity_hash=effect_hash,
        source=source,
    )
    context = Miprov2IntentContext(
        control_identity_hash=CONTROL_HASH,
        run_id=RUN_ID,
        effect_kind="sample",
        effect_identity_hash=effect_hash,
        intent_id="failed-intent",
        candidate=BASE_CANDIDATE,
        task_batch_identities=tasks,
        eval_config=artifact.eval_config,
        eval_config_binding=artifact,
        execution_policy=artifact.request.execution_policy,
        reward_policy_hash=REWARD_POLICY_HASH,
    )
    context_ref = persist_miprov2_intent_context(store, context)
    assert context_ref.schema_name == MIPROV2_INTENT_CONTEXT_SCHEMA
    intent = EvaluationIntent(
        intent_id=context.intent_id,
        candidate=context.candidate,
        target_eval_config=context.eval_config,
        context_role=EvaluationRole.INTERNAL,
        context_policy_ref=context_ref.content_hash,
        purpose="miprov2_sample",
        run_id=RUN_ID,
        step_index=4,
    )
    failure_ref = _typed_put(
        store,
        EVALUATION_FAILURE_SCHEMA,
        {
            "candidate": context.candidate.model_dump(mode="json"),
            "eval_config": context.eval_config.model_dump(mode="json"),
            "purpose": intent.purpose,
            "error": "scripted evaluator exception",
        },
    )
    resolution = IntentResolution(
        intent=intent,
        outcome=IntentOutcome.FAILED,
        detail=ResolutionDetail(
            classification=ResolutionClass.PROVIDER,
            message="scripted evaluator exception",
        ),
        evaluation_evidence_refs=(failure_ref,),
        resolved_eval_config=intent.target_eval_config,
    )

    resolved = resolve_miprov2_evaluation_failure(store, resolution)

    assert (
        resolved.reward_value,
        resolved.normalized_score,
        resolved.evaluation.normalized_score,
    ) == (0.0, 0.0, 0.0)
    assert resolved.row_accounting.model_dump() == {
        "planned": 1,
        "present": 0,
        "missing": 0,
        "failed": 1,
        "invalid": 0,
    }
