from __future__ import annotations

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
from pydantic import ValidationError

from whetstone.optimization.identity import TypedRef, compute_identity_hash
from whetstone.optimization.miprov2_control import (
    Miprov2ComponentSpec,
    Miprov2ProgramLayout,
)
from whetstone.optimization.miprov2_eval_config import (
    Miprov2EvalConfigBinding,
    Miprov2EvalConfigBindingRequest,
    Miprov2EvalPurpose,
    Miprov2EvaluationExecutionPolicy,
    derive_eval_config_reference,
)
from whetstone.optimization.miprov2_study import (
    EVALUATION_EVIDENCE_SCHEMA,
    MIPROV2_CANDIDATE_RENDERING_SCHEMA,
    MIPROV2_CANDIDATE_RENDERING_SCHEMA_VERSION,
    MIPROV2_REFERENCE_COMMIT,
    OPTUNA_VERSION,
    REWARD_SCHEMA,
    EvaluationBinding,
    Miprov2CandidateAssemblyBinding,
    Miprov2ParameterSpace,
    Miprov2Study,
    Miprov2StudySchedule,
    Promotion,
    SampleObservation,
    StudyTranscript,
    StudyTranscriptMismatch,
    VerifiedEvaluationCitation,
    select_promotion,
)
from whetstone.optimization.prompt_program import (
    PROMPT_PROGRAM_PAYLOAD_FIELD,
    PromptProgram,
    PromptProgramComponent,
)
from whetstone.optimization.schema import (
    Candidate,
    candidate_reference,
    eval_config_reference,
)

FULL_A = "a" * 64
FULL_B = "b" * 64
FULL_C = "c" * 64
FULL_D = "d" * 64
RUN_ID = "run-miprov2-study"
REWARD_POLICY_HASH = "e" * 64
CONTROL_IDENTITY_HASH = "f" * 64
PROMPT_ADAPTER_IDENTITY_HASH = "9" * 64
BASE_CANDIDATE = candidate_reference(
    Candidate(
        candidate_id="base",
        base_ref="root",
        payload={"user_prompt_template": "base"},
    )
)


def _hash(index: int) -> str:
    return f"{index:064x}"


def _execution_policy() -> Miprov2EvaluationExecutionPolicy:
    return Miprov2EvaluationExecutionPolicy(
        num_threads=1,
        max_errors=1,
        provide_traceback=False,
        task_model_identity_hash=_hash(50_001),
        provider_execution_policy_hash=_hash(50_002),
    )


def _instruction(predictor_index: int, candidate_index: int) -> str:
    return f"instruction-{predictor_index}-{candidate_index}"


def _instruction_identity(
    predictor_index: int,
    candidate_index: int,
) -> str:
    return compute_identity_hash(
        schema="whetstone.miprov2_instruction",
        schema_version=1,
        payload={
            "instruction": _instruction(predictor_index, candidate_index)
        },
    )


def _demo_set(predictor_index: int, candidate_index: int) -> dict[str, Any]:
    return {
        "candidate_seed": predictor_index * 100 + candidate_index,
        "components": [
            {
                "component_id": f"component-{predictor_index}",
                "demos": [],
            }
        ],
    }


def _demo_identity(predictor_index: int, candidate_index: int) -> str:
    return compute_identity_hash(
        schema="whetstone.miprov2_component_demo_set",
        schema_version=1,
        payload=_demo_set(predictor_index, candidate_index),
    )


def _eval_config(sampling_config_hash: str) -> EvalConfig:
    definition_ref = DefinitionRef(
        definition_id="eval",
        version="1",
        schema_name="dr_code.eval_definition",
        identity_hash=FULL_A,
    )
    config_identity_hash = identity_hash_for(
        schema=SCHEMA_EVAL_CONFIG,
        payload={
            "definition_identity": definition_ref.identity_hash,
            "sampling_config": sampling_config_hash,
            "evaluation_procedure_config": FULL_C,
            "aggregation_config": FULL_D,
        },
    )
    return EvalConfig(
        definition_ref=definition_ref,
        sampling_config_hash=sampling_config_hash,
        evaluation_procedure_config_hash=FULL_C,
        aggregation_config_hash=FULL_D,
        config_identity_hash=config_identity_hash,
    )


def _program_layout(component_count: int) -> Miprov2ProgramLayout:
    return Miprov2ProgramLayout(
        layout_id=f"test-layout-{component_count}",
        component_specs=tuple(
            Miprov2ComponentSpec(
                component_id=f"component-{index}",
                candidate_field=f"field-{index}",
                prompt_format_identity_hash=PROMPT_ADAPTER_IDENTITY_HASH,
            )
            for index in range(component_count)
        ),
    )


def _space(
    instruction_counts: tuple[int, ...] = (3,),
    demo_counts: tuple[int, ...] | None = None,
    *,
    offset: int = 1,
) -> Miprov2ParameterSpace:
    cursor = offset
    instructions: list[tuple[str, ...]] = []
    for predictor_index, count in enumerate(instruction_counts):
        if offset == 1:
            identities = tuple(
                _instruction_identity(predictor_index, i) for i in range(count)
            )
        else:
            identities = tuple(_hash(cursor + i) for i in range(count))
        instructions.append(identities)
        cursor += count
    demos: list[tuple[str, ...]] | None = None
    if demo_counts is not None:
        demos = []
        for predictor_index, count in enumerate(demo_counts):
            if offset == 1:
                identities = tuple(
                    _demo_identity(predictor_index, i) for i in range(count)
                )
            else:
                identities = tuple(_hash(cursor + i) for i in range(count))
            demos.append(identities)
            cursor += count
    return Miprov2ParameterSpace(
        instruction_pool_identity_hashes=tuple(instructions),
        demo_pool_identity_hashes=tuple(demos) if demos is not None else None,
    )


def _schedule(
    *,
    num_trials: int = 12,
    minibatch: bool = False,
    minibatch_size: int = 3,
    valset_size: int = 3,
    steps: int = 5,
) -> Miprov2StudySchedule:
    return Miprov2StudySchedule(
        num_trials=num_trials,
        minibatch=minibatch,
        minibatch_size=minibatch_size,
        valset_size=valset_size,
        minibatch_full_eval_steps=steps,
    )


def _binding(
    task_identities: tuple[str, ...],
    *,
    nonce: int,
    purpose: Literal[
        "miprov2_baseline",
        "miprov2_sample",
        "miprov2_promotion",
    ],
    candidate_identity_hash: str,
    eval_config_source: Any,
    control_identity_hash: str = CONTROL_IDENTITY_HASH,
    normalized_score: float,
) -> EvaluationBinding:
    evidence_ref = TypedRef(
        schema_name=EVALUATION_EVIDENCE_SCHEMA,
        content_hash=_hash(30_000 + nonce),
    )
    reward_ref = TypedRef(
        schema_name=REWARD_SCHEMA,
        content_hash=_hash(35_000 + nonce),
    )
    effect_identity_hash = _hash(25_000 + nonce)
    derivation_request = Miprov2EvalConfigBindingRequest(
        control_identity_hash=control_identity_hash,
        source_eval_config=eval_config_source,
        purpose=cast(
            "Miprov2EvalPurpose",
            purpose.removeprefix("miprov2_"),
        ),
        effect_identity_hash=effect_identity_hash,
        execution_policy=_execution_policy(),
        task_batch_identities=task_identities,
    )
    task_set = TaskSet(
        manifest_id=f"miprov2-test-tasks-{nonce}",
        version="1",
        dataset_revision="test",
        task_identities=task_identities,
    )
    repeat_plan = RepeatPlan(
        plan_id=f"miprov2-test-repeats-{nonce}",
        version="1",
        task_identities=task_identities,
        repeat_count=1,
    )
    sampling_config = SamplingDefinition(
        definition_id="miprov2-test-sampling",
        version="1",
    ).materialize(
        {
            "task_set_hash": task_set.identity_hash(),
            "repeat_plan_hash": repeat_plan.identity_hash(),
        }
    )
    derived_eval_config = derive_eval_config_reference(
        eval_config_source,
        sampling_config,
    )
    eval_config_binding = Miprov2EvalConfigBinding(
        request=derivation_request,
        task_set=task_set,
        repeat_plan=repeat_plan,
        sampling_config=sampling_config,
        eval_config=derived_eval_config,
    )
    citation = VerifiedEvaluationCitation(
        run_id=RUN_ID,
        intent_id=f"intent-{nonce}",
        effect_identity_hash=effect_identity_hash,
        purpose=purpose,
        candidate_identity_hash=candidate_identity_hash,
        task_batch_identities=task_identities,
        validation_eval_source_identity_hash=eval_config_source.identity_hash,
        eval_config_identity_hash=derived_eval_config.identity_hash,
        eval_config_binding_identity_hash=eval_config_binding.identity_hash(),
        reward_policy_hash=REWARD_POLICY_HASH,
        evidence_ref=evidence_ref,
        reward_ref=reward_ref,
        normalized_score=normalized_score,
    )
    return EvaluationBinding(
        run_id=RUN_ID,
        intent_id=f"intent-{nonce}",
        effect_identity_hash=effect_identity_hash,
        purpose=purpose,
        candidate_identity_hash=candidate_identity_hash,
        task_batch_identities=task_identities,
        eval_config=derived_eval_config,
        eval_config_binding=eval_config_binding,
        reward_policy_hash=REWARD_POLICY_HASH,
        reward_ref=reward_ref,
        evidence_citations=(citation,),
        normalized_score=normalized_score,
    )


def _assembly(
    seam: Miprov2Study,
    params: Any,
    *,
    component_suffix: str = "",
    instruction_suffix: str = "",
) -> Miprov2CandidateAssemblyBinding:
    normalized = seam.space.normalize(params)
    combination = seam.space.combination_identity_hash(normalized)
    values = dict(normalized)
    components: list[dict[str, Any]] = []
    payload = dict(BASE_CANDIDATE.record.payload)
    prompt_components: list[PromptProgramComponent] = []
    for index, spec in enumerate(seam.program_layout.component_specs):
        instruction_index = values[f"{index}_predictor_instruction"]
        demo_name = f"{index}_predictor_demos"
        instruction = (
            _instruction(index, instruction_index) + instruction_suffix
        )
        components.append(
            {
                "component_id": f"{spec.component_id}{component_suffix}",
                "candidate_field": f"{spec.candidate_field}{component_suffix}",
                "instruction_index": instruction_index,
                "instruction": instruction,
                "instruction_identity_hash": compute_identity_hash(
                    schema="whetstone.miprov2_instruction",
                    schema_version=1,
                    payload={"instruction": instruction},
                ),
                "demo_index": values.get(demo_name),
                "demo_set": (
                    _demo_set(index, values[demo_name])
                    if demo_name in values
                    else None
                ),
                "demo_identity_hash": (
                    _demo_identity(index, values[demo_name])
                    if demo_name in values
                    else None
                ),
            }
        )
        candidate_field = f"{spec.candidate_field}{component_suffix}"
        payload[candidate_field] = instruction
        prompt_components.append(
            PromptProgramComponent(
                component_id=f"{spec.component_id}{component_suffix}",
                candidate_field=candidate_field,
            )
        )
    rendering = {
        "control_identity_hash": CONTROL_IDENTITY_HASH,
        "base_candidate_identity_hash": BASE_CANDIDATE.identity_hash,
        "categorical_combination_identity_hash": combination,
        "renderer_version": "whetstone_native_prompt_components/v1",
        "components": components,
    }
    program_identity_hash = compute_identity_hash(
        schema=MIPROV2_CANDIDATE_RENDERING_SCHEMA,
        schema_version=MIPROV2_CANDIDATE_RENDERING_SCHEMA_VERSION,
        payload=rendering,
    )
    candidate = candidate_reference(
        Candidate(
            candidate_id=f"miprov2-{program_identity_hash[:24]}",
            base_ref=BASE_CANDIDATE.record.base_ref,
            payload={
                **payload,
                "miprov2_candidate_rendering": rendering,
                PROMPT_PROGRAM_PAYLOAD_FIELD: PromptProgram(
                    components=tuple(prompt_components)
                ).model_dump(mode="json"),
            },
        )
    )
    return Miprov2CandidateAssemblyBinding(
        params=normalized,
        categorical_combination_identity_hash=combination,
        candidate=candidate,
        program_identity_hash=program_identity_hash,
        control_identity_hash=CONTROL_IDENTITY_HASH,
        base_candidate=BASE_CANDIDATE,
        program_layout=seam.program_layout,
        prompt_adapter_identity_hash=PROMPT_ADAPTER_IDENTITY_HASH,
    )


def _seam(
    *,
    space: Miprov2ParameterSpace | None = None,
    schedule: Miprov2StudySchedule | None = None,
    seed: int = 9,
) -> tuple[Miprov2Study, StudyTranscript]:
    resolved_space = space or _space()
    resolved_schedule = schedule or _schedule()
    validation_task_identities = tuple(
        _hash(40_000 + index) for index in range(resolved_schedule.valset_size)
    )
    validation_eval_source = eval_config_reference(_eval_config(_hash(20_000)))
    seam = Miprov2Study(
        seed=seed,
        space=resolved_space,
        schedule=resolved_schedule,
        run_id=RUN_ID,
        validation_task_identities=validation_task_identities,
        validation_eval_source=validation_eval_source,
        reward_policy_hash=REWARD_POLICY_HASH,
        control_identity_hash=CONTROL_IDENTITY_HASH,
        prompt_adapter_identity_hash=PROMPT_ADAPTER_IDENTITY_HASH,
        expected_base_candidate=BASE_CANDIDATE,
        program_layout=_program_layout(
            len(resolved_space.instruction_candidate_counts)
        ),
    )
    transcript = seam.initial_transcript(
        baseline_score=0.25,
        baseline_evaluation=_binding(
            validation_task_identities,
            nonce=0,
            purpose="miprov2_baseline",
            candidate_identity_hash=BASE_CANDIDATE.identity_hash,
            eval_config_source=validation_eval_source,
            normalized_score=0.25,
        ),
    )
    return seam, transcript


def _record(
    seam: Miprov2Study,
    transcript: StudyTranscript,
    *,
    score: float,
    nonce: int,
) -> StudyTranscript:
    suggestion = seam.suggest_next(transcript)
    candidate_assembly = _assembly(seam, suggestion.params)
    candidate_identity_hash = candidate_assembly.candidate.identity_hash
    sample_is_full = (
        not seam.schedule.minibatch
        or seam.schedule.minibatch_size >= seam.schedule.valset_size
    )
    task_identities = (
        seam.validation_task_identities
        if sample_is_full
        else seam.validation_task_identities[: seam.schedule.minibatch_size]
    )
    evaluation = _binding(
        task_identities,
        nonce=nonce,
        purpose="miprov2_sample",
        candidate_identity_hash=candidate_identity_hash,
        eval_config_source=seam.validation_eval_source,
        normalized_score=score,
    )
    promoted = seam.promotion_candidate(
        transcript,
        suggestion,
        score=score,
        evaluation=evaluation,
        candidate_assembly=candidate_assembly,
    )
    kwargs: dict[str, Any] = {}
    if promoted is not None:
        kwargs = {
            "promotion_full_score": score + 0.01,
            "promotion_evaluation": _binding(
                seam.validation_task_identities,
                nonce=500 + nonce,
                purpose="miprov2_promotion",
                candidate_identity_hash=(
                    promoted.evaluated_candidate_identity_hash
                ),
                eval_config_source=seam.validation_eval_source,
                normalized_score=score + 0.01,
            ),
        }
    return seam.record_sample(
        transcript,
        suggestion,
        score=score,
        evaluation=evaluation,
        candidate_assembly=candidate_assembly,
        **kwargs,
    )


def test_parameter_order_is_predictor_major_instruction_then_demo() -> None:
    space = _space((3, 2), (2, 4))

    assert space.parameter_names == (
        "0_predictor_instruction",
        "0_predictor_demos",
        "1_predictor_instruction",
        "1_predictor_demos",
    )
    assert space.baseline_params == tuple(
        (name, 0) for name in space.parameter_names
    )


def test_space_and_transcript_bind_exact_ordered_pool_identities() -> None:
    first_space = _space((2,), offset=1)
    other_space = _space((2,), offset=100)
    assert first_space.instruction_candidate_counts == (
        other_space.instruction_candidate_counts
    )
    assert first_space.identity_hash() != other_space.identity_hash()
    first, transcript = _seam(space=first_space)
    other = Miprov2Study(
        seed=first.seed,
        space=other_space,
        schedule=first.schedule,
        run_id=first.run_id,
        validation_task_identities=first.validation_task_identities,
        validation_eval_source=first.validation_eval_source,
        reward_policy_hash=first.reward_policy_hash,
        control_identity_hash=first.control_identity_hash,
        prompt_adapter_identity_hash=first.prompt_adapter_identity_hash,
        expected_base_candidate=first.expected_base_candidate,
        program_layout=first.program_layout,
    )

    with pytest.raises(
        StudyTranscriptMismatch,
        match="bound MIPROv2 study contract",
    ):
        other.reconstruct_study(transcript)


def test_transcript_binds_versions_schedule_baseline_and_distributions() -> (
    None
):
    seam, transcript = _seam()
    dumped = transcript.model_dump(mode="json")

    assert dumped["reference_commit"] == MIPROV2_REFERENCE_COMMIT
    assert dumped["optuna_version"] == OPTUNA_VERSION
    assert dumped["seed"] == seam.seed
    assert (
        dumped["parameter_space_identity_hash"] == seam.space.identity_hash()
    )
    assert dumped["distribution_identity_hash"] == (
        seam.space.distribution_identity_hash()
    )
    assert transcript.baseline.evaluation.evidence_refs
    assert (
        transcript.baseline.categorical_combination_identity_hash
        != transcript.baseline.evaluated_base_candidate.identity_hash
    )
    assert len(transcript.identity_hash()) == 64

    with pytest.raises(ValidationError, match="extra"):
        StudyTranscript.model_validate({**dumped, "runtime_handle": "no"})


def test_study_rejects_cross_run_candidate_task_and_config_evidence() -> None:
    seam, transcript = _seam(
        schedule=_schedule(minibatch=True, minibatch_size=2)
    )
    binding = transcript.baseline.evaluation
    citation = binding.evidence_citations[0]

    def rejected(changed_binding: EvaluationBinding) -> None:
        changed = transcript.model_copy(
            update={
                "baseline": transcript.baseline.model_copy(
                    update={"evaluation": changed_binding}
                )
            }
        )
        with pytest.raises(
            StudyTranscriptMismatch,
            match="identity and evidence contract",
        ):
            seam.reconstruct_study(changed)

    other_run_citation = citation.model_copy(update={"run_id": "other-run"})
    rejected(
        binding.model_copy(
            update={
                "run_id": "other-run",
                "evidence_citations": (other_run_citation,),
            }
        )
    )

    other_candidate_citation = citation.model_copy(
        update={"candidate_identity_hash": FULL_B}
    )
    rejected(
        binding.model_copy(
            update={"evidence_citations": (other_candidate_citation,)}
        )
    )

    reversed_tasks = tuple(reversed(binding.task_batch_identities))
    reordered_citation = citation.model_copy(
        update={"task_batch_identities": reversed_tasks}
    )
    rejected(
        binding.model_copy(
            update={
                "task_batch_identities": reversed_tasks,
                "evidence_citations": (reordered_citation,),
            }
        )
    )

    other_config = eval_config_reference(_eval_config(_hash(99_999)))
    other_config_citation = citation.model_copy(
        update={"eval_config_identity_hash": other_config.identity_hash}
    )
    rejected(
        binding.model_copy(
            update={
                "eval_config": other_config,
                "evidence_citations": (other_config_citation,),
            }
        )
    )

    with pytest.raises(ValidationError, match="canonical evaluation evidence"):
        VerifiedEvaluationCitation.model_validate(
            {
                **citation.model_dump(mode="json"),
                "evidence_ref": {
                    "schema_name": "other.evidence",
                    "content_hash": FULL_A,
                },
            }
        )


@pytest.mark.parametrize(
    "tamper",
    ["observation", "binding", "citation", "coherent_evidence"],
)
def test_verified_score_is_bound_at_every_persistence_layer(
    tamper: str,
) -> None:
    seam, transcript = _seam()
    baseline = transcript.baseline
    binding = baseline.evaluation
    citation = binding.evidence_citations[0]
    if tamper == "observation":
        changed_baseline = baseline.model_copy(update={"score": 0.75})
    elif tamper == "binding":
        changed_baseline = baseline.model_copy(
            update={
                "evaluation": binding.model_copy(
                    update={"normalized_score": 0.75}
                )
            }
        )
    elif tamper == "citation":
        changed_baseline = baseline.model_copy(
            update={
                "evaluation": binding.model_copy(
                    update={
                        "evidence_citations": (
                            citation.model_copy(
                                update={"normalized_score": 0.75}
                            ),
                        )
                    }
                )
            }
        )
    else:
        changed_baseline = baseline.model_copy(
            update={
                "evaluation": binding.model_copy(
                    update={
                        "normalized_score": 0.75,
                        "evidence_citations": (
                            citation.model_copy(
                                update={"normalized_score": 0.75}
                            ),
                        ),
                    }
                )
            }
        )
    changed = transcript.model_copy(update={"baseline": changed_baseline})

    with pytest.raises(
        StudyTranscriptMismatch,
        match="identity and evidence contract",
    ):
        seam.reconstruct_study(changed)


@pytest.mark.parametrize("observation", ["sample", "promotion"])
def test_sample_and_promotion_scores_equal_verified_values(
    observation: str,
) -> None:
    seam, transcript = _seam(
        schedule=_schedule(
            num_trials=1,
            minibatch=True,
            minibatch_size=2,
            steps=1,
        )
    )
    transcript = _record(seam, transcript, score=0.5, nonce=1)
    sample = transcript.samples[0]
    assert sample.promotion is not None
    if observation == "sample":
        changed_sample = sample.model_copy(update={"score": 0.9})
    else:
        changed_sample = sample.model_copy(
            update={
                "promotion": sample.promotion.model_copy(
                    update={"full_score": 0.9}
                )
            }
        )
    changed = transcript.model_copy(update={"samples": (changed_sample,)})

    with pytest.raises(
        StudyTranscriptMismatch,
        match="identity and evidence contract",
    ):
        seam.reconstruct_study(changed)


def test_bound_study_rejects_a_coherently_substituted_base_candidate() -> None:
    seam, transcript = _seam()
    other_base = candidate_reference(
        Candidate(
            candidate_id="other-base",
            base_ref="other-root",
            payload={"user_prompt_template": "substituted"},
        )
    )
    binding = transcript.baseline.evaluation
    citation = binding.evidence_citations[0].model_copy(
        update={"candidate_identity_hash": other_base.identity_hash}
    )
    other_binding = binding.model_copy(
        update={
            "candidate_identity_hash": other_base.identity_hash,
            "evidence_citations": (citation,),
        }
    )
    changed = transcript.model_copy(
        update={
            "expected_base_candidate": other_base,
            "baseline": transcript.baseline.model_copy(
                update={
                    "evaluated_base_candidate": other_base,
                    "evaluation": other_binding,
                }
            ),
        }
    )

    with pytest.raises(
        StudyTranscriptMismatch,
        match="bound MIPROv2 study contract",
    ):
        seam.reconstruct_study(changed)


def test_sample_candidate_identity_cannot_diverge_from_its_assembly() -> None:
    seam, transcript = _seam(schedule=_schedule(num_trials=1))
    transcript = _record(seam, transcript, score=0.5, nonce=1)
    sample = transcript.samples[0]
    changed_sample = sample.model_copy(
        update={
            "evaluated_candidate_identity_hash": FULL_B,
            "evaluation": _binding(
                seam.validation_task_identities,
                nonce=90,
                purpose="miprov2_sample",
                candidate_identity_hash=FULL_B,
                eval_config_source=seam.validation_eval_source,
                normalized_score=sample.score,
            ),
        }
    )
    changed = transcript.model_copy(update={"samples": (changed_sample,)})

    with pytest.raises(
        StudyTranscriptMismatch,
        match="identity and evidence contract",
    ):
        seam.reconstruct_study(changed)


def test_suggestion_rejects_noncanonical_candidate_assembly() -> None:
    seam, transcript = _seam(space=_space((3,)))
    suggestion = seam.suggest_next(transcript)
    changed_value = (suggestion.params[0][1] + 1) % 3
    other_params = (("0_predictor_instruction", changed_value),)
    other_assembly = _assembly(seam, other_params)

    with pytest.raises(ValueError, match="parameters do not match"):
        seam.promotion_candidate(
            transcript,
            suggestion,
            score=0.5,
            evaluation=_binding(
                seam.validation_task_identities,
                nonce=10,
                purpose="miprov2_sample",
                candidate_identity_hash=other_assembly.candidate.identity_hash,
                eval_config_source=seam.validation_eval_source,
                normalized_score=0.5,
            ),
            candidate_assembly=other_assembly,
        )

    foreign_instruction = _assembly(
        seam,
        suggestion.params,
        instruction_suffix="-not-from-pool",
    )
    with pytest.raises(ValueError, match=r"instruction.*frozen pool"):
        seam.promotion_candidate(
            transcript,
            suggestion,
            score=0.5,
            evaluation=_binding(
                seam.validation_task_identities,
                nonce=11,
                purpose="miprov2_sample",
                candidate_identity_hash=(
                    foreign_instruction.candidate.identity_hash
                ),
                eval_config_source=seam.validation_eval_source,
                normalized_score=0.5,
            ),
            candidate_assembly=foreign_instruction,
        )


@pytest.mark.parametrize(
    "tamper",
    ["omit_base_payload", "change_prompt_program"],
)
def test_candidate_assembly_recomputes_the_exact_native_candidate(
    tamper: str,
) -> None:
    seam, transcript = _seam(space=_space((3,)))
    suggestion = seam.suggest_next(transcript)
    assembly = _assembly(seam, suggestion.params)
    payload = dict(assembly.candidate.record.payload)
    if tamper == "omit_base_payload":
        del payload["user_prompt_template"]
    else:
        payload[PROMPT_PROGRAM_PAYLOAD_FIELD] = {
            "renderer_version": "whetstone_plain_examples/v1",
            "components": [
                {
                    "component_id": "forged",
                    "candidate_field": "field-0",
                    "examples": [],
                }
            ],
        }
    changed_candidate = candidate_reference(
        assembly.candidate.record.model_copy(update={"payload": payload})
    )
    changed_assembly = assembly.model_copy(
        update={"candidate": changed_candidate}
    )

    with pytest.raises(ValueError, match="canonical native rendering"):
        seam.promotion_candidate(
            transcript,
            suggestion,
            score=0.5,
            evaluation=_binding(
                seam.validation_task_identities,
                nonce=12,
                purpose="miprov2_sample",
                candidate_identity_hash=changed_candidate.identity_hash,
                eval_config_source=seam.validation_eval_source,
                normalized_score=0.5,
            ),
            candidate_assembly=changed_assembly,
        )


def test_eval_config_derivation_rejects_identity_and_task_tampering() -> None:
    _, transcript = _seam()
    derivation = transcript.baseline.evaluation.eval_config_binding

    changed_tasks = derivation.task_set.model_copy(
        update={
            "task_identities": tuple(
                reversed(derivation.task_set.task_identities)
            )
        }
    )
    with pytest.raises(ValidationError, match="wrong ordered tasks"):
        Miprov2EvalConfigBinding.model_validate(
            {
                **derivation.model_dump(mode="json"),
                "task_set": changed_tasks.model_dump(mode="json"),
            }
        )

    changed_sampling = derivation.sampling_config.model_copy(
        update={"config_identity_hash": FULL_B}
    )
    with pytest.raises(ValidationError, match="identity is not canonical"):
        Miprov2EvalConfigBinding.model_validate(
            {
                **derivation.model_dump(mode="json"),
                "sampling_config": changed_sampling.model_dump(mode="json"),
            }
        )

    changed_eval = eval_config_reference(_eval_config(_hash(99_100)))
    with pytest.raises(
        ValidationError,
        match="canonical source derivation",
    ):
        Miprov2EvalConfigBinding.model_validate(
            {
                **derivation.model_dump(mode="json"),
                "eval_config": changed_eval.model_dump(mode="json"),
            }
        )


def test_provisional_rejects_foreign_eval_source_before_promotion() -> None:
    seam, transcript = _seam(
        schedule=_schedule(
            num_trials=1,
            minibatch=True,
            minibatch_size=2,
            steps=1,
        )
    )
    suggestion = seam.suggest_next(transcript)
    assembly = _assembly(seam, suggestion.params)
    foreign_source = eval_config_reference(_eval_config(_hash(99_200)))
    evaluation = _binding(
        seam.validation_task_identities[:2],
        nonce=13,
        purpose="miprov2_sample",
        candidate_identity_hash=assembly.candidate.identity_hash,
        eval_config_source=foreign_source,
        normalized_score=0.5,
    )

    with pytest.raises(ValueError, match="evaluation derivation"):
        seam.promotion_candidate(
            transcript,
            suggestion,
            score=0.5,
            evaluation=evaluation,
            candidate_assembly=assembly,
        )


def test_pydantic_records_are_frozen_and_reject_unknown_fields() -> None:
    _, transcript = _seam()
    binding = transcript.baseline.evaluation
    with pytest.raises(ValidationError, match="extra"):
        EvaluationBinding.model_validate(
            {**binding.model_dump(mode="json"), "loose_label": "bad"}
        )
    with pytest.raises(ValidationError, match="frozen"):
        binding.task_batch_identities = (FULL_A,)  # ty: ignore[invalid-assignment]


def test_promotion_is_stable_by_first_observed_mean() -> None:
    seam, transcript = _seam(
        space=_space((2,)),
        schedule=_schedule(minibatch=True, minibatch_size=2),
    )
    first_suggestion = seam.suggest_next(transcript)
    first_params = (("0_predictor_instruction", 0),)
    first_assembly = _assembly(seam, first_params)
    first = SampleObservation(
        trial_number=first_suggestion.trial_number,
        params=first_params,
        candidate_combination_identity_hash=(
            first_assembly.categorical_combination_identity_hash
        ),
        evaluated_candidate_identity_hash=first_assembly.candidate.identity_hash,
        candidate_assembly=first_assembly,
        score=0.5,
        evaluation=_binding(
            seam.validation_task_identities[:2],
            nonce=1,
            purpose="miprov2_sample",
            candidate_identity_hash=first_assembly.candidate.identity_hash,
            eval_config_source=seam.validation_eval_source,
            normalized_score=0.5,
        ),
        batch_full_evaluation=False,
    )
    second_params = (("0_predictor_instruction", 1),)
    second_assembly = _assembly(seam, second_params)
    second = SampleObservation(
        trial_number=2,
        params=second_params,
        candidate_combination_identity_hash=(
            second_assembly.categorical_combination_identity_hash
        ),
        evaluated_candidate_identity_hash=(
            second_assembly.candidate.identity_hash
        ),
        candidate_assembly=second_assembly,
        score=0.5,
        evaluation=_binding(
            seam.validation_task_identities[:2],
            nonce=2,
            purpose="miprov2_sample",
            candidate_identity_hash=second_assembly.candidate.identity_hash,
            eval_config_source=seam.validation_eval_source,
            normalized_score=0.5,
        ),
        batch_full_evaluation=False,
    )

    assert select_promotion((first, second)).params == first.params
    first_again = first.model_copy(update={"trial_number": 3, "score": 0.1})
    assert select_promotion((first, second, first_again)).params == (
        second.params
    )


def test_exact_promotion_exhaustion_message() -> None:
    seam, _ = _seam(
        space=_space((1,)),
        schedule=_schedule(
            num_trials=2,
            minibatch=True,
            minibatch_size=2,
            steps=1,
        ),
    )
    params = seam.space.baseline_params
    combination = seam.space.combination_identity_hash(params)
    assembly = _assembly(seam, params)
    first = SampleObservation(
        trial_number=1,
        params=params,
        candidate_combination_identity_hash=combination,
        evaluated_candidate_identity_hash=assembly.candidate.identity_hash,
        candidate_assembly=assembly,
        score=0.5,
        evaluation=_binding(
            seam.validation_task_identities[:2],
            nonce=1,
            purpose="miprov2_sample",
            candidate_identity_hash=assembly.candidate.identity_hash,
            eval_config_source=seam.validation_eval_source,
            normalized_score=0.5,
        ),
        batch_full_evaluation=False,
    )
    promoted = first.model_copy(
        update={
            "promotion": Promotion(
                trial_number=2,
                params=params,
                candidate_combination_identity_hash=combination,
                evaluated_candidate_identity_hash=(
                    assembly.candidate.identity_hash
                ),
                candidate_assembly=assembly,
                source_sample_trial_number=1,
                minibatch_mean=0.5,
                full_score=0.6,
                evaluation=_binding(
                    seam.validation_task_identities,
                    nonce=2,
                    purpose="miprov2_promotion",
                    candidate_identity_hash=assembly.candidate.identity_hash,
                    eval_config_source=seam.validation_eval_source,
                    normalized_score=0.6,
                ),
            )
        }
    )

    with pytest.raises(
        ValueError,
        match=r"^No valid program found in param_score_dict$",
    ):
        select_promotion((promoted,))


def test_missing_and_extra_promotions_are_rejected() -> None:
    seam, transcript = _seam(
        schedule=_schedule(
            num_trials=3,
            minibatch=True,
            minibatch_size=2,
            steps=1,
        )
    )
    suggestion = seam.suggest_next(transcript)
    assembly = _assembly(seam, suggestion.params)
    candidate_identity_hash = assembly.candidate.identity_hash
    evaluation = _binding(
        seam.validation_task_identities[:2],
        nonce=1,
        purpose="miprov2_sample",
        candidate_identity_hash=candidate_identity_hash,
        eval_config_source=seam.validation_eval_source,
        normalized_score=0.5,
    )
    assert seam.schedule.promotion_due(
        optuna_trial_number=suggestion.trial_number
    )
    with pytest.raises(ValueError, match="promotion evaluation is required"):
        seam.record_sample(
            transcript,
            suggestion,
            score=0.5,
            evaluation=evaluation,
            candidate_assembly=assembly,
        )

    non_minibatch, full_transcript = _seam(schedule=_schedule(minibatch=False))
    full_suggestion = non_minibatch.suggest_next(full_transcript)
    full_assembly = _assembly(non_minibatch, full_suggestion.params)
    with pytest.raises(ValueError, match="supplied off cadence"):
        non_minibatch.record_sample(
            full_transcript,
            full_suggestion,
            score=0.5,
            evaluation=_binding(
                non_minibatch.validation_task_identities,
                nonce=2,
                purpose="miprov2_sample",
                candidate_identity_hash=full_assembly.candidate.identity_hash,
                eval_config_source=non_minibatch.validation_eval_source,
                normalized_score=0.5,
            ),
            candidate_assembly=full_assembly,
            promotion_full_score=0.6,
            promotion_evaluation=_binding(
                non_minibatch.validation_task_identities,
                nonce=3,
                purpose="miprov2_promotion",
                candidate_identity_hash=full_assembly.candidate.identity_hash,
                eval_config_source=non_minibatch.validation_eval_source,
                normalized_score=0.6,
            ),
        )


@pytest.mark.parametrize(
    ("num_trials", "expected_sample_promotions"),
    [
        (10, (5, 11)),
        (11, (5, 11, 13)),
    ],
)
def test_divisible_and_nondivisible_final_promotion_cadence(
    num_trials: int,
    expected_sample_promotions: tuple[int, ...],
) -> None:
    seam, transcript = _seam(
        space=_space((20,)),
        schedule=_schedule(
            num_trials=num_trials,
            minibatch=True,
            minibatch_size=2,
            steps=5,
        ),
    )
    for index in range(num_trials):
        transcript = _record(
            seam,
            transcript,
            score=float(index),
            nonce=index + 1,
        )

    assert (
        tuple(
            sample.trial_number
            for sample in transcript.samples
            if sample.promotion is not None
        )
        == expected_sample_promotions
    )
    with pytest.raises(ValueError, match="schedule is exhausted"):
        seam.suggest_next(transcript)


def test_promotion_is_inserted_before_tell_with_exact_mean_and_source() -> (
    None
):
    seam, transcript = _seam(
        space=_space((1,)),
        schedule=_schedule(
            num_trials=1,
            minibatch=True,
            minibatch_size=2,
            steps=1,
        ),
    )
    transcript = _record(seam, transcript, score=0.6, nonce=1)
    sample = transcript.samples[0]
    assert sample.promotion is not None
    assert sample.trial_number == 1
    assert sample.promotion.trial_number == 2
    assert sample.promotion.source_sample_trial_number == 1
    assert sample.promotion.minibatch_mean == 0.6

    study = seam.reconstruct_study(transcript)
    assert [trial.number for trial in study.trials] == [0, 1, 2]
    assert study.trials[1].value == 0.6
    assert study.trials[2].value == 0.61


def test_equal_minibatch_is_full_but_not_winner_eligible() -> None:
    seam, transcript = _seam(
        space=_space((2,)),
        schedule=_schedule(
            num_trials=1,
            minibatch=True,
            minibatch_size=3,
            valset_size=3,
            steps=5,
        ),
    )
    suggestion = seam.suggest_next(transcript)
    assembly = _assembly(seam, suggestion.params)
    candidate_identity_hash = assembly.candidate.identity_hash
    evaluation = _binding(
        seam.validation_task_identities,
        nonce=1,
        purpose="miprov2_sample",
        candidate_identity_hash=candidate_identity_hash,
        eval_config_source=seam.validation_eval_source,
        normalized_score=1.0,
    )
    assert (
        seam.promotion_candidate(
            transcript,
            suggestion,
            score=1.0,
            evaluation=evaluation,
            candidate_assembly=assembly,
        )
        is not None
    )
    transcript = seam.record_sample(
        transcript,
        suggestion,
        score=1.0,
        evaluation=evaluation,
        candidate_assembly=assembly,
        promotion_full_score=0.2,
        promotion_evaluation=_binding(
            seam.validation_task_identities,
            nonce=2,
            purpose="miprov2_promotion",
            candidate_identity_hash=candidate_identity_hash,
            eval_config_source=seam.validation_eval_source,
            normalized_score=0.2,
        ),
    )

    assert transcript.samples[0].batch_full_evaluation is True
    assert transcript.samples[0].promotion is not None
    assert seam.best_full_evaluation(transcript).source == "baseline"


def test_non_minibatch_strict_best_update_keeps_first_tie() -> None:
    seam, transcript = _seam(
        space=_space((3,)),
        schedule=_schedule(num_trials=2, minibatch=False),
    )
    transcript = _record(seam, transcript, score=0.8, nonce=1)
    first_trial = transcript.samples[0].trial_number
    transcript = _record(seam, transcript, score=0.8, nonce=2)

    best = seam.best_full_evaluation(transcript)
    assert best.source == "sample"
    assert best.trial_number == first_trial
    assert best.score == 0.8


def test_reconstruction_rejects_trial_parameter_and_promotion_tampering() -> (
    None
):
    seam, transcript = _seam(
        space=_space((4,)),
        schedule=_schedule(
            num_trials=2,
            minibatch=True,
            minibatch_size=2,
            steps=1,
        ),
    )
    transcript = _record(seam, transcript, score=0.6, nonce=1)
    sample = transcript.samples[0]
    assert sample.promotion is not None
    changed_value = (sample.params[0][1] + 1) % 4
    changed_params = (("0_predictor_instruction", changed_value),)
    changed_combination = seam.space.combination_identity_hash(changed_params)
    changed_sample = sample.model_copy(
        update={
            "params": changed_params,
            "candidate_combination_identity_hash": changed_combination,
            "promotion": sample.promotion.model_copy(
                update={
                    "params": changed_params,
                    "candidate_combination_identity_hash": changed_combination,
                }
            ),
        }
    )
    tampered = transcript.model_copy(update={"samples": (changed_sample,)})

    with pytest.raises(
        StudyTranscriptMismatch,
        match="identity and evidence contract",
    ):
        seam.reconstruct_study(tampered)

    changed_promotion = sample.promotion.model_copy(
        update={"minibatch_mean": 0.7}
    )
    tampered_promotion = transcript.model_copy(
        update={
            "samples": (
                sample.model_copy(update={"promotion": changed_promotion}),
            )
        }
    )
    with pytest.raises(
        StudyTranscriptMismatch,
        match="stable mean ranking",
    ):
        seam.reconstruct_study(tampered_promotion)

    alternate_assembly = _assembly(
        seam,
        sample.params,
        component_suffix="-substituted",
    )
    changed_promotion_candidate = sample.promotion.model_copy(
        update={
            "evaluated_candidate_identity_hash": (
                alternate_assembly.candidate.identity_hash
            ),
            "candidate_assembly": alternate_assembly,
            "evaluation": _binding(
                seam.validation_task_identities,
                nonce=700,
                purpose="miprov2_promotion",
                candidate_identity_hash=(
                    alternate_assembly.candidate.identity_hash
                ),
                eval_config_source=seam.validation_eval_source,
                normalized_score=sample.promotion.full_score,
            ),
        }
    )
    changed_promotion_transcript = transcript.model_copy(
        update={
            "samples": (
                sample.model_copy(
                    update={"promotion": changed_promotion_candidate}
                ),
            )
        }
    )
    with pytest.raises(
        StudyTranscriptMismatch,
        match="identity and evidence contract",
    ):
        seam.reconstruct_study(changed_promotion_transcript)

    changed_promotion_number = sample.promotion.model_copy(
        update={"trial_number": 9}
    )
    bad_promotion_chronology = transcript.model_copy(
        update={
            "samples": (
                sample.model_copy(
                    update={"promotion": changed_promotion_number}
                ),
            )
        }
    )
    with pytest.raises(
        StudyTranscriptMismatch,
        match="promotion trial chronology",
    ):
        seam.reconstruct_study(bad_promotion_chronology)

    bad_sample_chronology = transcript.model_copy(
        update={"samples": (sample.model_copy(update={"trial_number": 7}),)}
    )
    with pytest.raises(
        StudyTranscriptMismatch,
        match="sample trial chronology",
    ):
        seam.reconstruct_study(bad_sample_chronology)


def test_restart_after_every_event_reproduces_next_suggestion() -> None:
    seam, transcript = _seam(
        space=_space((5, 4), (3, 2)),
        schedule=_schedule(
            num_trials=14,
            minibatch=True,
            minibatch_size=2,
            steps=5,
        ),
        seed=23,
    )
    for index in range(14):
        before = seam.suggest_next(transcript)
        restored = StudyTranscript.model_validate_json(
            transcript.model_dump_json()
        )
        assert seam.suggest_next(restored) == before
        transcript = _record(
            seam,
            restored,
            score=(index % 7) / 10,
            nonce=index + 1,
        )
        restored_after = StudyTranscript.model_validate_json(
            transcript.model_dump_json()
        )
        assert seam.reconstruct_study(restored_after)


def test_frozen_optuna_oracle_exceeds_ten_sampled_trials() -> None:
    seam, transcript = _seam(
        space=_space((4, 3), (2, 2)),
        schedule=_schedule(num_trials=12, minibatch=False),
        seed=23,
    )
    observed: list[tuple[int, tuple[int, ...]]] = []
    for index in range(12):
        suggestion = seam.suggest_next(transcript)
        observed.append(
            (
                suggestion.trial_number,
                tuple(value for _, value in suggestion.params),
            )
        )
        score = (index * 17 % 11) / 10
        assembly = _assembly(seam, suggestion.params)
        transcript = seam.record_sample(
            transcript,
            suggestion,
            score=score,
            evaluation=_binding(
                seam.validation_task_identities,
                nonce=index + 1,
                purpose="miprov2_sample",
                candidate_identity_hash=assembly.candidate.identity_hash,
                eval_config_source=seam.validation_eval_source,
                normalized_score=score,
            ),
            candidate_assembly=assembly,
        )

    assert observed == [
        # Frozen from optuna==4.8.0 TPESampler(seed=23, multivariate=True),
        # after the all-zero completed baseline trial.
        (1, (1, 1, 2, 0)),
        (2, (1, 0, 1, 0)),
        (3, (2, 0, 0, 0)),
        (4, (3, 0, 2, 0)),
        (5, (2, 1, 0, 0)),
        (6, (2, 1, 1, 1)),
        (7, (2, 1, 0, 0)),
        (8, (2, 0, 1, 0)),
        (9, (0, 1, 0, 1)),
        (10, (2, 0, 1, 0)),
        (11, (2, 0, 1, 0)),
        (12, (1, 0, 2, 1)),
    ]


def test_frozen_optuna_oracle_interleaves_promotions_across_tpe_startup() -> (
    None
):
    seam, transcript = _seam(
        space=_space((4, 3), (2, 2)),
        schedule=_schedule(
            num_trials=12,
            minibatch=True,
            minibatch_size=2,
            steps=3,
        ),
        seed=23,
    )
    observed: list[
        tuple[int, tuple[int, ...], int | None, tuple[int, ...] | None]
    ] = []
    for index in range(12):
        suggestion = seam.suggest_next(transcript)
        transcript = _record(
            seam,
            transcript,
            score=(index * 17 % 11) / 10,
            nonce=index + 1,
        )
        promotion = transcript.samples[-1].promotion
        observed.append(
            (
                suggestion.trial_number,
                tuple(value for _, value in suggestion.params),
                promotion.trial_number if promotion is not None else None,
                (
                    tuple(value for _, value in promotion.params)
                    if promotion is not None
                    else None
                ),
            )
        )

    assert observed == [
        # Frozen from the DSPy API order under optuna==4.8.0:
        # ask/suggest, optional promotion add_trial, then sample tell.
        (1, (1, 1, 2, 0), None, None),
        (2, (1, 0, 1, 0), None, None),
        (3, (2, 0, 0, 0), 4, (1, 0, 1, 0)),
        (5, (3, 0, 2, 0), None, None),
        (6, (2, 1, 0, 0), None, None),
        (7, (2, 1, 1, 1), 8, (2, 1, 1, 1)),
        (9, (2, 1, 0, 0), None, None),
        (10, (3, 1, 0, 1), None, None),
        (11, (3, 1, 2, 1), 12, (3, 1, 0, 1)),
        (13, (0, 1, 1, 1), None, None),
        (14, (0, 1, 1, 1), None, None),
        (15, (3, 0, 0, 1), 16, (0, 1, 1, 1)),
    ]
