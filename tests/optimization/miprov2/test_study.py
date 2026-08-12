from __future__ import annotations

from typing import Any

import pytest
from dr_store import ObjectStore, SqliteBackend
from pydantic import ValidationError

from tests.optimization.miprov2.support import (
    MIPROV2_EVIDENCE_TASK_IDENTITY,
    configure_test_miprov2,
    make_miprov2_evidence_fixture,
)
from tests.optimization.support import (
    candidate,
    internal_reward_policy,
    optimizer_config_ref,
)
from whetstone.core.identity import (
    TypedRef,
    compute_identity_hash,
)
from whetstone.core.roles import EvaluationRole
from whetstone.evaluation.schema_names import (
    EVALUATION_EVIDENCE_SCHEMA,
    EVALUATION_FAILURE_SCHEMA,
)
from whetstone.experiment.binding import EvaluationBinding
from whetstone.experiment.candidate import (
    Candidate,
    candidate_reference,
)
from whetstone.experiment.reward import apply_reward_policy, reward_reference
from whetstone.optimization.contracts import (
    OptimizationRun,
    OutputContract,
    StepMode,
    optimization_run_reference,
)
from whetstone.optimization.miprov2.eval_config import (
    Miprov2EvalConfigBinding,
)
from whetstone.optimization.miprov2.render import candidate_from_components
from whetstone.optimization.miprov2.study import (
    MIPROV2_CANDIDATE_ASSEMBLY_SCHEMA_VERSION,
    MIPROV2_CANDIDATE_PROGRAM_SCHEMA,
    MIPROV2_CANDIDATE_PROGRAM_SCHEMA_VERSION,
    MIPROV2_STUDY_SCHEMA,
    MIPROV2_STUDY_SCHEMA_VERSION,
    Miprov2CandidateAssemblyBinding,
    Miprov2CandidateRendering,
    Miprov2ComponentSelection,
    Miprov2EvaluationObservation,
    Miprov2ParameterSpace,
    Miprov2Study,
    Miprov2StudySchedule,
    StudyTranscript,
    StudyTranscriptMismatch,
    select_promotion,
)

FULL_A = "a" * 64
FULL_B = "b" * 64


def _configure():
    return configure_test_miprov2()


def _run(
    control,
    *,
    run_id: str = "study-run",
    optimizer_config=None,
    reward_policy=None,
):
    return optimization_run_reference(
        OptimizationRun(
            run_id=run_id,
            optimizer_config=(
                control.reference()
                if optimizer_config is None
                else optimizer_config
            ),
            adapter_key="miprov2",
            mode=StepMode.PROPOSAL_ONLY,
            terminal_output_contract=OutputContract(returned_proposal_count=1),
            template_render_contract=control.template_render_contract,
            reward_policy=(
                control.reward_policy
                if reward_policy is None
                else reward_policy
            ),
        )
    )


def _space() -> Miprov2ParameterSpace:
    return Miprov2ParameterSpace(
        instruction_pool_identity_hashes=(("1" * 64, "2" * 64, "3" * 64),),
        demo_pool_identity_hashes=(("4" * 64, "5" * 64),),
    )


def test_parameter_order_is_instruction_then_demo_for_one_component() -> None:
    assert MIPROV2_STUDY_SCHEMA_VERSION == 5
    assert MIPROV2_CANDIDATE_PROGRAM_SCHEMA == (
        "whetstone.miprov2_candidate_program"
    )
    assert MIPROV2_CANDIDATE_PROGRAM_SCHEMA_VERSION == 1
    assert MIPROV2_CANDIDATE_ASSEMBLY_SCHEMA_VERSION == 4
    space = _space()

    assert space.parameter_names == (
        "0_predictor_instruction",
        "0_predictor_demos",
    )
    assert space.baseline_params == (
        ("0_predictor_instruction", 0),
        ("0_predictor_demos", 0),
    )


def test_parameter_space_normalization_is_exact_and_ordered() -> None:
    space = _space()
    params = {
        "0_predictor_demos": 1,
        "0_predictor_instruction": 2,
    }

    normalized = space.normalize(params)

    assert normalized == (
        ("0_predictor_instruction", 2),
        ("0_predictor_demos", 1),
    )
    assert len(space.combination_identity_hash(normalized)) == 64
    with pytest.raises(ValueError, match="does not match"):
        space.normalize({"0_predictor_instruction": 0})


def test_parameter_space_rejects_out_of_range_categories() -> None:
    with pytest.raises(ValueError, match="outside"):
        _space().normalize(
            (
                ("0_predictor_instruction", 3),
                ("0_predictor_demos", 0),
            )
        )


@pytest.mark.parametrize(
    ("trial_number", "due"),
    ((0, False), (4, False), (5, True), (10, False), (11, True)),
)
def test_minibatch_promotion_cadence_is_frozen(
    trial_number: int,
    due: bool,
) -> None:
    schedule = Miprov2StudySchedule(
        num_trials=10,
        minibatch=True,
        minibatch_size=2,
        valset_size=5,
        minibatch_full_eval_steps=5,
    )

    assert schedule.adjusted_num_trials == 13
    assert schedule.promotion_due(optuna_trial_number=trial_number) is due


def test_failed_observation_is_zero_and_has_no_reward(tmp_path) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "study-observation.sqlite"))
    intent, context = make_miprov2_evidence_fixture(
        store,
        reward_policy_hash=FULL_A,
    )

    observation = Miprov2EvaluationObservation(
        run_id="run",
        intent_id="intent",
        effect_identity_hash=FULL_B,
        purpose="miprov2_sample",
        candidate=intent.candidate,
        task_batch_hashes=context.task_batch_hashes,
        eval_config=context.eval_config,
        eval_config_binding=context.eval_config_binding,
        evaluation_binding=intent.evaluation_binding,
        evaluation_result_ref=TypedRef(
            schema_name=EVALUATION_FAILURE_SCHEMA,
            content_hash=FULL_B,
        ),
        expected_reward_policy_hash=FULL_A,
        reward_ref=None,
        normalized_score=0.0,
    )

    assert observation.normalized_score == 0.0
    assert observation.reward_ref is None


def test_failed_observation_rejects_score_or_reward_model_copy_bypass(
    tmp_path,
) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "study-bypass.sqlite"))
    intent, context = make_miprov2_evidence_fixture(
        store,
        reward_policy_hash=FULL_A,
    )
    payload = {
        "run_id": "run",
        "intent_id": "intent",
        "effect_identity_hash": FULL_B,
        "purpose": "miprov2_sample",
        "candidate": intent.candidate.model_dump(mode="json"),
        "task_batch_hashes": list(context.task_batch_hashes),
        "eval_config": context.eval_config.model_dump(mode="json"),
        "eval_config_binding": context.eval_config_binding.model_dump(
            mode="json"
        ),
        "evaluation_binding": intent.evaluation_binding.model_dump(
            mode="json"
        ),
        "evaluation_result_ref": {
            "schema_name": EVALUATION_FAILURE_SCHEMA,
            "content_hash": FULL_B,
        },
        "expected_reward_policy_hash": FULL_A,
        "reward_ref": None,
        "normalized_score": 1.0,
    }

    with pytest.raises(ValidationError, match="zero score"):
        Miprov2EvaluationObservation.model_validate(payload)


def test_observation_candidate_is_an_exact_candidate_ref(tmp_path) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "study-candidate.sqlite"))
    intent, context = make_miprov2_evidence_fixture(
        store,
        reward_policy_hash=FULL_A,
    )
    other = candidate_reference(candidate("other", text="Other {query}."))

    with pytest.raises(ValidationError):
        Miprov2EvaluationObservation(
            run_id="run",
            intent_id="intent",
            effect_identity_hash=FULL_B,
            purpose="miprov2_sample",
            candidate=other.model_copy(
                update={"record_ref": intent.candidate.record_ref}
            ),
            task_batch_hashes=context.task_batch_hashes,
            eval_config=context.eval_config,
            eval_config_binding=context.eval_config_binding,
            evaluation_binding=intent.evaluation_binding,
            evaluation_result_ref=TypedRef(
                schema_name=EVALUATION_FAILURE_SCHEMA,
                content_hash=FULL_B,
            ),
            expected_reward_policy_hash=FULL_A,
            reward_ref=None,
            normalized_score=0.0,
        )


def test_candidate_assembly_recomputes_exact_native_candidate() -> None:
    control = configure_test_miprov2()
    instruction = "Improved {query}."
    instruction_hash = compute_identity_hash(
        schema="whetstone.miprov2_instruction",
        schema_version=1,
        payload={"instruction": instruction},
    )
    params = (("0_predictor_instruction", 0),)
    combination = Miprov2ParameterSpace(
        instruction_pool_identity_hashes=((instruction_hash,),)
    ).combination_identity_hash(params)
    rendering = Miprov2CandidateRendering(
        control_identity_hash=control.identity_hash(),
        base_candidate_identity_hash=control.base_candidate.identity_hash,
        categorical_combination_identity_hash=combination,
        components=(
            Miprov2ComponentSelection(
                component_id="generate",
                instruction_index=0,
                instruction=instruction,
                instruction_identity_hash=instruction_hash,
                demo_index=None,
                demo_set=None,
                demo_identity_hash=None,
            ),
        ),
    )
    exact = candidate_from_components(
        base=control.base_candidate,
        candidate_id=f"miprov2-{rendering.identity_hash()[:24]}",
        components=rendering.model_dump(mode="json")["components"],
        run=_run(control),
    )
    exact_ref = candidate_reference(exact)
    program_hash = compute_identity_hash(
        schema=MIPROV2_CANDIDATE_PROGRAM_SCHEMA,
        schema_version=MIPROV2_CANDIDATE_PROGRAM_SCHEMA_VERSION,
        payload={"candidate": exact_ref.model_dump(mode="json")},
    )
    assembly = Miprov2CandidateAssemblyBinding(
        params=params,
        categorical_combination_identity_hash=combination,
        candidate=exact_ref,
        program_identity_hash=program_hash,
        rendering=rendering,
        optimizer_config=control.reference(),
        base_candidate=control.base_candidate,
        program_layout=control.program_layout,
        prompt_adapter_identity_hash=control.prompt_adapter_identity_hash,
        run=_run(control),
    )

    assert assembly.candidate == exact_ref

    foreign = Candidate.model_validate(
        {
            **exact.model_dump(mode="json"),
            "payload": {
                **exact.payload.to_json(),
                "user_prompt_template": "Foreign {query}.",
            },
        }
    )
    foreign_ref = candidate_reference(foreign)
    foreign_program_hash = compute_identity_hash(
        schema=MIPROV2_CANDIDATE_PROGRAM_SCHEMA,
        schema_version=MIPROV2_CANDIDATE_PROGRAM_SCHEMA_VERSION,
        payload={"candidate": foreign_ref.model_dump(mode="json")},
    )
    with pytest.raises(ValidationError, match="deterministic rendering"):
        Miprov2CandidateAssemblyBinding.model_validate(
            {
                **assembly.model_dump(mode="json"),
                "candidate": foreign_ref.model_dump(mode="json"),
                "program_identity_hash": foreign_program_hash,
            }
        )


@pytest.mark.parametrize("forged_ref", (False, True))
def test_candidate_assembly_rejects_foreign_run_control_authority(
    forged_ref: bool,
) -> None:
    control = _configure()
    space = Miprov2ParameterSpace(
        instruction_pool_identity_hashes=(
            (
                compute_identity_hash(
                    schema="whetstone.miprov2_instruction",
                    schema_version=1,
                    payload={"instruction": "Baseline {query}."},
                ),
            ),
        )
    )
    assembly = _study_assembly(
        control,
        space,
        (("0_predictor_instruction", 0),),
        instructions=("Baseline {query}.",),
    )
    foreign_run = _run(
        control,
        optimizer_config=optimizer_config_ref("foreign-miprov2-control"),
    )
    if forged_ref:
        foreign_run = assembly.run.model_copy(
            update={"record": foreign_run.record}
        )
    payload = assembly.model_dump(mode="json")
    payload["run"] = foreign_run.model_dump(mode="json")

    message = "record_ref" if forged_ref else "exact MIPROv2 control"
    with pytest.raises(ValidationError, match=message):
        Miprov2CandidateAssemblyBinding.model_validate(payload)


def test_candidate_assembly_rejects_same_hash_foreign_control_address() -> (
    None
):
    control = _configure()
    space = Miprov2ParameterSpace(
        instruction_pool_identity_hashes=(
            (
                compute_identity_hash(
                    schema="whetstone.miprov2_instruction",
                    schema_version=1,
                    payload={"instruction": "Baseline {query}."},
                ),
            ),
        )
    )
    assembly = _study_assembly(
        control,
        space,
        (("0_predictor_instruction", 0),),
        instructions=("Baseline {query}.",),
    )
    foreign_optimizer_config = control.reference().model_copy(
        update={
            "record_ref": optimizer_config_ref(
                "foreign-miprov2-address"
            ).record_ref
        }
    )
    foreign_run = _run(control, optimizer_config=foreign_optimizer_config)
    payload = assembly.model_dump(mode="json")
    payload["run"] = foreign_run.model_dump(mode="json")

    with pytest.raises(ValidationError, match="exact MIPROv2 control"):
        Miprov2CandidateAssemblyBinding.model_validate(payload)


def _study_observation(
    *,
    context,
    candidate_ref,
    purpose,
    score: float,
    nonce: int,
):
    effect_hash = f"{10_000 + nonce:064x}"
    request = context.eval_config_binding.request.model_copy(
        update={
            "purpose": purpose.removeprefix("miprov2_"),
            "effect_identity_hash": effect_hash,
        }
    )
    binding = Miprov2EvalConfigBinding.model_validate(
        {
            **context.eval_config_binding.model_dump(mode="json"),
            "request": request.model_dump(mode="json"),
        }
    )
    evaluation_binding = EvaluationBinding(
        schema_version=3,
        eval_config=binding.eval_config,
        role=EvaluationRole.INTERNAL,
        campaign="miprov2-study-test",
    )
    policy = internal_reward_policy()
    reward = apply_reward_policy(
        policy,
        aggregates={"score": score / 100},
        evidence_role=EvaluationRole.INTERNAL,
        evidence_refs=(
            TypedRef(
                schema_name="whetstone.aggregate",
                content_hash=f"{20_000 + nonce:064x}",
            ),
        ),
    )
    return Miprov2EvaluationObservation(
        run_id="study-run",
        intent_id=f"study-intent-{nonce}",
        effect_identity_hash=effect_hash,
        purpose=purpose,
        candidate=candidate_ref,
        task_batch_hashes=(MIPROV2_EVIDENCE_TASK_IDENTITY,),
        eval_config=binding.eval_config,
        eval_config_binding=binding,
        evaluation_binding=evaluation_binding,
        evaluation_result_ref=TypedRef(
            schema_name=EVALUATION_EVIDENCE_SCHEMA,
            content_hash=f"{30_000 + nonce:064x}",
        ),
        expected_reward_policy_hash=policy.identity_hash(),
        reward_ref=reward_reference(reward),
        normalized_score=score,
    )


def _study_assembly(
    control,
    space,
    params,
    *,
    instructions=("Baseline {query}.", "Improved {query}."),
):
    values = dict(params)
    instruction_index = values["0_predictor_instruction"]
    instruction = instructions[instruction_index]
    instruction_hash = space.instruction_pool_identity_hashes[0][
        instruction_index
    ]
    combination = space.combination_identity_hash(params)
    rendering = Miprov2CandidateRendering(
        control_identity_hash=control.identity_hash(),
        base_candidate_identity_hash=control.base_candidate.identity_hash,
        categorical_combination_identity_hash=combination,
        components=(
            Miprov2ComponentSelection(
                component_id="generate",
                instruction_index=instruction_index,
                instruction=instruction,
                instruction_identity_hash=instruction_hash,
                demo_index=None,
                demo_set=None,
                demo_identity_hash=None,
            ),
        ),
    )
    assembled = candidate_from_components(
        base=control.base_candidate,
        candidate_id=f"miprov2-{rendering.identity_hash()[:24]}",
        components=rendering.model_dump(mode="json")["components"],
        run=_run(control),
    )
    assembled_ref = candidate_reference(assembled)
    return Miprov2CandidateAssemblyBinding(
        params=params,
        categorical_combination_identity_hash=combination,
        candidate=assembled_ref,
        program_identity_hash=compute_identity_hash(
            schema=MIPROV2_CANDIDATE_PROGRAM_SCHEMA,
            schema_version=1,
            payload={"candidate": assembled_ref.model_dump(mode="json")},
        ),
        rendering=rendering,
        optimizer_config=control.reference(),
        base_candidate=control.base_candidate,
        program_layout=control.program_layout,
        prompt_adapter_identity_hash=control.prompt_adapter_identity_hash,
        run=_run(control),
    )


def _study_case(
    tmp_path,
    *,
    num_trials: int,
    minibatch: bool,
    steps: int = 5,
    seed: int = 23,
):
    store = ObjectStore(SqliteBackend(tmp_path / "study-case.sqlite"))
    control = configure_test_miprov2()
    _, context = make_miprov2_evidence_fixture(
        store,
        reward_policy_hash=internal_reward_policy().identity_hash(),
        control_identity_hash=control.identity_hash(),
    )
    instructions = tuple(
        f"Instruction {index} {{query}}." for index in range(4)
    )
    instruction_hashes = tuple(
        compute_identity_hash(
            schema="whetstone.miprov2_instruction",
            schema_version=1,
            payload={"instruction": instruction},
        )
        for instruction in instructions
    )
    space = Miprov2ParameterSpace(
        instruction_pool_identity_hashes=(instruction_hashes,)
    )
    study = Miprov2Study(
        seed=seed,
        space=space,
        schedule=Miprov2StudySchedule(
            num_trials=num_trials,
            minibatch=minibatch,
            minibatch_size=1,
            valset_size=1,
            minibatch_full_eval_steps=steps,
        ),
        run_id="study-run",
        validation_task_hashes=(MIPROV2_EVIDENCE_TASK_IDENTITY,),
        validation_eval_source=(
            context.eval_config_binding.request.source_eval_config
        ),
        reward_policy_hash=internal_reward_policy().identity_hash(),
        optimizer_config=control.reference(),
        prompt_adapter_identity_hash=control.prompt_adapter_identity_hash,
        expected_base_candidate=control.base_candidate,
        program_layout=control.program_layout,
        run=_run(control),
    )
    transcript = study.initial_transcript(
        baseline_score=0.0,
        baseline_evaluation=_study_observation(
            context=context,
            candidate_ref=control.base_candidate,
            purpose="miprov2_baseline",
            score=0.0,
            nonce=0,
        ),
    )
    return context, control, instructions, space, study, transcript


def _record_study_sample(
    *,
    context,
    control,
    instructions,
    space,
    study,
    transcript,
    score: float,
    nonce: int,
):
    suggestion = study.suggest_next(transcript)
    assembly = _study_assembly(
        control,
        space,
        suggestion.params,
        instructions=instructions,
    )
    evaluation = _study_observation(
        context=context,
        candidate_ref=assembly.candidate,
        purpose="miprov2_sample",
        score=score,
        nonce=nonce,
    )
    promotion = study.promotion_candidate(
        transcript,
        suggestion,
        score=score,
        evaluation=evaluation,
        candidate_assembly=assembly,
    )
    promotion_score = round(score + 0.1, 1)
    kwargs: dict[str, Any] = {}
    if promotion is not None:
        kwargs = {
            "promotion_full_score": promotion_score,
            "promotion_evaluation": _study_observation(
                context=context,
                candidate_ref=promotion.candidate_assembly.candidate,
                purpose="miprov2_promotion",
                score=promotion_score,
                nonce=10_000 + nonce,
            ),
        }
    completed = study.record_sample(
        transcript,
        suggestion,
        score=score,
        evaluation=evaluation,
        candidate_assembly=assembly,
        **kwargs,
    )
    return suggestion, completed


def test_equal_size_minibatch_promotion_flow_matches_frozen_oracle(
    tmp_path,
) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "study-flow.sqlite"))
    control = configure_test_miprov2()
    _, context = make_miprov2_evidence_fixture(
        store,
        reward_policy_hash=internal_reward_policy().identity_hash(),
        control_identity_hash=control.identity_hash(),
    )
    instruction_hashes = tuple(
        compute_identity_hash(
            schema="whetstone.miprov2_instruction",
            schema_version=1,
            payload={"instruction": instruction},
        )
        for instruction in ("Baseline {query}.", "Improved {query}.")
    )
    space = Miprov2ParameterSpace(
        instruction_pool_identity_hashes=(instruction_hashes,)
    )
    study = Miprov2Study(
        seed=9,
        space=space,
        schedule=Miprov2StudySchedule(
            num_trials=1,
            minibatch=True,
            minibatch_size=1,
            valset_size=1,
            minibatch_full_eval_steps=5,
        ),
        run_id="study-run",
        validation_task_hashes=(MIPROV2_EVIDENCE_TASK_IDENTITY,),
        validation_eval_source=(
            context.eval_config_binding.request.source_eval_config
        ),
        reward_policy_hash=internal_reward_policy().identity_hash(),
        optimizer_config=control.reference(),
        prompt_adapter_identity_hash=control.prompt_adapter_identity_hash,
        expected_base_candidate=control.base_candidate,
        program_layout=control.program_layout,
        run=_run(control),
    )
    baseline = _study_observation(
        context=context,
        candidate_ref=control.base_candidate,
        purpose="miprov2_baseline",
        score=25.0,
        nonce=0,
    )
    transcript = study.initial_transcript(
        baseline_score=25.0,
        baseline_evaluation=baseline,
    )
    suggestion = study.suggest_next(transcript)
    assembly = _study_assembly(control, space, suggestion.params)
    sample = _study_observation(
        context=context,
        candidate_ref=assembly.candidate,
        purpose="miprov2_sample",
        score=100.0,
        nonce=1,
    )
    selected = study.promotion_candidate(
        transcript,
        suggestion,
        score=100.0,
        evaluation=sample,
        candidate_assembly=assembly,
    )
    assert selected is not None
    promotion = _study_observation(
        context=context,
        candidate_ref=selected.candidate_assembly.candidate,
        purpose="miprov2_promotion",
        score=20.0,
        nonce=2,
    )
    completed = study.record_sample(
        transcript,
        suggestion,
        score=100.0,
        evaluation=sample,
        candidate_assembly=assembly,
        promotion_full_score=20.0,
        promotion_evaluation=promotion,
    )

    observation = completed.samples[0]
    assert observation.batch_full_evaluation is True
    assert observation.promotion is not None
    assert observation.promotion.trial_number == suggestion.trial_number + 1
    assert observation.promotion.minibatch_mean == 100.0
    assert [
        (trial.number, trial.value)
        for trial in study.reconstruct_study(completed).trials
    ] == [(0, 25.0), (1, 100.0), (2, 20.0)]
    winner = study.best_full_evaluation(completed)
    assert (winner.source, winner.score) == ("baseline", 25.0)

    restarted = type(completed).model_validate_json(
        completed.model_dump_json()
    )
    assert study.reconstruct_study(restarted).trials[-1].value == 20.0

    foreign_instruction = "Foreign {query}."
    foreign_hash = compute_identity_hash(
        schema="whetstone.miprov2_instruction",
        schema_version=1,
        payload={"instruction": foreign_instruction},
    )
    foreign_rendering = Miprov2CandidateRendering(
        control_identity_hash=control.identity_hash(),
        base_candidate_identity_hash=control.base_candidate.identity_hash,
        categorical_combination_identity_hash=(
            suggestion.candidate_combination_identity_hash
        ),
        components=(
            Miprov2ComponentSelection(
                component_id="generate",
                instruction_index=dict(suggestion.params)[
                    "0_predictor_instruction"
                ],
                instruction=foreign_instruction,
                instruction_identity_hash=foreign_hash,
                demo_index=None,
                demo_set=None,
                demo_identity_hash=None,
            ),
        ),
    )
    foreign_candidate = candidate_reference(
        candidate_from_components(
            base=control.base_candidate,
            candidate_id=f"miprov2-{foreign_rendering.identity_hash()[:24]}",
            components=foreign_rendering.model_dump(mode="json")["components"],
            run=_run(control),
        )
    )
    foreign_assembly = Miprov2CandidateAssemblyBinding(
        params=suggestion.params,
        categorical_combination_identity_hash=(
            suggestion.candidate_combination_identity_hash
        ),
        candidate=foreign_candidate,
        program_identity_hash=compute_identity_hash(
            schema=MIPROV2_CANDIDATE_PROGRAM_SCHEMA,
            schema_version=1,
            payload={"candidate": foreign_candidate.model_dump(mode="json")},
        ),
        rendering=foreign_rendering,
        optimizer_config=control.reference(),
        base_candidate=control.base_candidate,
        program_layout=control.program_layout,
        prompt_adapter_identity_hash=control.prompt_adapter_identity_hash,
        run=_run(control),
    )
    tampered: Any = completed.model_dump(mode="json")
    tampered_sample = tampered["samples"][0]
    tampered_sample["candidate_assembly"] = foreign_assembly.model_dump(
        mode="json"
    )
    tampered_sample["evaluated_candidate_identity_hash"] = (
        foreign_candidate.identity_hash
    )
    tampered_sample["evaluation"]["candidate"] = foreign_candidate.model_dump(
        mode="json"
    )

    with pytest.raises(ValidationError, match="selected category"):
        StudyTranscript.model_validate(tampered)


def test_frozen_optuna_oracle_exceeds_ten_sampled_trials(tmp_path) -> None:
    (
        context,
        control,
        instructions,
        space,
        study,
        transcript,
    ) = _study_case(tmp_path, num_trials=12, minibatch=False)
    observed: list[tuple[int, tuple[int, ...]]] = []

    for index in range(12):
        restored = StudyTranscript.model_validate_json(
            transcript.model_dump_json()
        )
        assert study.suggest_next(restored) == study.suggest_next(transcript)
        suggestion, transcript = _record_study_sample(
            context=context,
            control=control,
            instructions=instructions,
            space=space,
            study=study,
            transcript=restored,
            score=(index * 10 % 11) / 10,
            nonce=index + 1,
        )
        observed.append(
            (
                suggestion.trial_number,
                tuple(value for _, value in suggestion.params),
            )
        )

    assert observed == [
        (1, (1,)),
        (2, (1,)),
        (3, (3,)),
        (4, (3,)),
        (5, (0,)),
        (6, (0,)),
        (7, (0,)),
        (8, (0,)),
        (9, (1,)),
        (10, (2,)),
        (11, (3,)),
        (12, (1,)),
    ]
    with pytest.raises(ValueError, match="schedule is exhausted"):
        study.suggest_next(transcript)


def test_frozen_optuna_oracle_interleaves_promotions(tmp_path) -> None:
    (
        context,
        control,
        instructions,
        space,
        study,
        transcript,
    ) = _study_case(
        tmp_path,
        num_trials=12,
        minibatch=True,
        steps=3,
    )
    observed: list[
        tuple[int, tuple[int, ...], int | None, tuple[int, ...] | None]
    ] = []

    for index in range(12):
        suggestion, transcript = _record_study_sample(
            context=context,
            control=control,
            instructions=instructions,
            space=space,
            study=study,
            transcript=transcript,
            score=(index * 10 % 11) / 10,
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
        (1, (1,), None, None),
        (2, (1,), None, None),
        (3, (3,), 4, (3,)),
        (5, (3,), None, None),
        (6, (0,), None, None),
        (7, (0,), 8, (0,)),
        (9, (0,), None, None),
        (10, (1,), None, None),
        (11, (2,), 12, (1,)),
        (13, (3,), None, None),
        (14, (2,), None, None),
        (15, (3,), 16, (2,)),
    ]


def test_study_rejects_missing_and_off_cadence_promotions(tmp_path) -> None:
    (
        context,
        control,
        instructions,
        space,
        study,
        transcript,
    ) = _study_case(tmp_path, num_trials=1, minibatch=True, steps=1)
    suggestion = study.suggest_next(transcript)
    assembly = _study_assembly(
        control,
        space,
        suggestion.params,
        instructions=instructions,
    )
    evaluation = _study_observation(
        context=context,
        candidate_ref=assembly.candidate,
        purpose="miprov2_sample",
        score=0.5,
        nonce=1,
    )
    with pytest.raises(ValueError, match="promotion evaluation is required"):
        study.record_sample(
            transcript,
            suggestion,
            score=0.5,
            evaluation=evaluation,
            candidate_assembly=assembly,
        )

    (
        full_context,
        full_control,
        full_instructions,
        full_space,
        full_study,
        full_transcript,
    ) = _study_case(
        tmp_path,
        num_trials=1,
        minibatch=False,
    )
    full_suggestion = full_study.suggest_next(full_transcript)
    full_assembly = _study_assembly(
        full_control,
        full_space,
        full_suggestion.params,
        instructions=full_instructions,
    )
    full_evaluation = _study_observation(
        context=full_context,
        candidate_ref=full_assembly.candidate,
        purpose="miprov2_sample",
        score=0.5,
        nonce=2,
    )
    with pytest.raises(ValueError, match="supplied off cadence"):
        full_study.record_sample(
            full_transcript,
            full_suggestion,
            score=0.5,
            evaluation=full_evaluation,
            candidate_assembly=full_assembly,
            promotion_full_score=0.6,
            promotion_evaluation=_study_observation(
                context=full_context,
                candidate_ref=full_assembly.candidate,
                purpose="miprov2_promotion",
                score=0.6,
                nonce=3,
            ),
        )


def test_strict_best_update_and_stable_promotion_ranking(tmp_path) -> None:
    (
        context,
        control,
        instructions,
        space,
        study,
        transcript,
    ) = _study_case(tmp_path, num_trials=2, minibatch=False)
    _, transcript = _record_study_sample(
        context=context,
        control=control,
        instructions=instructions,
        space=space,
        study=study,
        transcript=transcript,
        score=0.8,
        nonce=1,
    )
    first_trial = transcript.samples[0].trial_number
    _, transcript = _record_study_sample(
        context=context,
        control=control,
        instructions=instructions,
        space=space,
        study=study,
        transcript=transcript,
        score=0.8,
        nonce=2,
    )

    best = study.best_full_evaluation(transcript)
    assert (best.source, best.trial_number, best.score) == (
        "sample",
        first_trial,
        0.8,
    )
    assert select_promotion(transcript.samples).source_sample_trial_number == (
        first_trial
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("seed", 99, "bound MIPROv2 study contract"),
        ("run_id", "foreign", "identity and evidence contract"),
        ("reward_policy_hash", FULL_B, "identity and evidence contract"),
        (
            "distribution_identity_hash",
            FULL_B,
            "identity and evidence contract",
        ),
    ),
)
def test_transcript_binds_study_authorities(
    tmp_path, field: str, value: Any, message: str
) -> None:
    *_, study, transcript = _study_case(
        tmp_path,
        num_trials=1,
        minibatch=False,
    )
    changed = transcript.model_copy(update={field: value})

    with pytest.raises(StudyTranscriptMismatch, match=message):
        study.reconstruct_study(changed)


def test_study_records_are_frozen_and_reject_unknown_fields(tmp_path) -> None:
    *_, transcript = _study_case(
        tmp_path,
        num_trials=1,
        minibatch=False,
    )
    dumped = transcript.model_dump(mode="json")

    with pytest.raises(ValidationError, match="extra"):
        StudyTranscript.model_validate({**dumped, "runtime_handle": "no"})
    with pytest.raises(ValidationError, match="frozen"):
        transcript.run_id = "other"


def test_study_transcript_identity_envelope_is_v5_only(tmp_path) -> None:
    *_, transcript = _study_case(
        tmp_path,
        num_trials=1,
        minibatch=False,
    )
    assert MIPROV2_STUDY_SCHEMA == "whetstone.miprov2_study_transcript"
    assert MIPROV2_STUDY_SCHEMA_VERSION == 5
    assert transcript._identity_schema == "whetstone.miprov2_study_transcript"
    assert transcript._identity_schema_version == 5
    payload = transcript.model_dump(mode="json")

    for legacy_version in (1, 2, 3, 4):
        legacy_payload = {**payload, "schema_version": legacy_version}
        with pytest.raises(ValidationError, match="schema_version"):
            StudyTranscript.model_validate(legacy_payload)
        assert (
            compute_identity_hash(
                schema="whetstone.miprov2_study_transcript",
                schema_version=legacy_version,
                payload=payload,
            )
            != transcript.identity_hash()
        )


@pytest.mark.parametrize("authority", ("optimizer_config", "reward_policy"))
def test_transcript_rejects_foreign_run_authorities(
    tmp_path,
    authority: str,
) -> None:
    *_, control, _, _, study, transcript = _study_case(
        tmp_path,
        num_trials=1,
        minibatch=False,
    )
    if authority == "optimizer_config":
        foreign_run = _run(
            control,
            optimizer_config=optimizer_config_ref("foreign-study-control"),
        )
        message = "exact MIPROv2 control"
    else:
        foreign_run = _run(
            control,
            reward_policy=control.reward_policy.model_copy(
                update={"policy_name": "foreign-study-policy/v1"}
            ),
        )
        message = "exact MIPROv2 reward policy"
    payload = transcript.model_dump(mode="json")
    payload["run"] = foreign_run.model_dump(mode="json")

    with pytest.raises(ValidationError, match=message):
        StudyTranscript.model_validate(payload)

    bypassed = transcript.model_copy(update={"run": foreign_run})
    with pytest.raises(
        StudyTranscriptMismatch,
        match="identity and evidence contract",
    ):
        study.reconstruct_study(bypassed)


def test_transcript_rejects_same_hash_foreign_control_address(
    tmp_path,
) -> None:
    *_, control, _, _, study, transcript = _study_case(
        tmp_path,
        num_trials=1,
        minibatch=False,
    )
    foreign_optimizer_config = control.reference().model_copy(
        update={
            "record_ref": optimizer_config_ref(
                "foreign-study-address"
            ).record_ref
        }
    )
    foreign_run = _run(control, optimizer_config=foreign_optimizer_config)
    payload = transcript.model_dump(mode="json")
    payload["run"] = foreign_run.model_dump(mode="json")

    with pytest.raises(ValidationError, match="exact MIPROv2 control"):
        StudyTranscript.model_validate(payload)

    bypassed = transcript.model_copy(update={"run": foreign_run})
    with pytest.raises(
        StudyTranscriptMismatch,
        match="identity and evidence contract",
    ):
        study.reconstruct_study(bypassed)


def test_score_is_bound_at_observation_and_transcript_layers(tmp_path) -> None:
    (
        context,
        control,
        instructions,
        space,
        study,
        transcript,
    ) = _study_case(tmp_path, num_trials=1, minibatch=False)
    _, transcript = _record_study_sample(
        context=context,
        control=control,
        instructions=instructions,
        space=space,
        study=study,
        transcript=transcript,
        score=0.5,
        nonce=1,
    )
    sample = transcript.samples[0]
    changed = transcript.model_copy(
        update={"samples": (sample.model_copy(update={"score": 0.9}),)}
    )

    with pytest.raises(
        StudyTranscriptMismatch,
        match="identity and evidence contract",
    ):
        study.reconstruct_study(changed)
