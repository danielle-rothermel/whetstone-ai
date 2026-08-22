"""Optuna replay determinism from a persisted StudyTranscript (D10)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from dr_store.sync import open_sqlite

from whetstone.coordination.harness_run_controller import RunRequest
from whetstone.core.identity import (
    ImmutableJsonObject,
    compute_identity_hash,
    typed_ref_for_record,
)
from whetstone.core.roles import EvalRole
from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
from whetstone.eval.schema_names import EVAL_EVIDENCE_SCHEMA
from whetstone.experiment.candidate import candidate_reference
from whetstone.experiment.reward import Reward, RewardInputCitation, reward_reference
from whetstone.optim.contracts import (
    OptimRun,
    OptimStepResult,
    OutputContract,
    StepMode,
    optimization_run_reference,
)
from whetstone.optim.harness import OptimHarness
from whetstone.optim.miprov2.adapter import MIPROV2_ADAPTER_KEY, MIPROV2_STATE_KEY
from whetstone.optim.miprov2.control import Miprov2DemoMode
from whetstone.optim.miprov2.demo import (
    ComponentDemoSequence,
    ComponentDemoSet,
    LabeledTaskDemo,
)
from whetstone.optim.miprov2.engine_binding import EngineEvalBindingResolver
from whetstone.optim.miprov2.eval_config import (
    EvalBindingRequest,
    Miprov2EvaluationExecutionPolicy,
)
from whetstone.optim.miprov2.render import candidate_from_components
from whetstone.optim.miprov2.runtime import (
    Miprov2Driver,
    Miprov2State,
    _miprov2_candidate_rendering,
)
from whetstone.optim.miprov2.study import (
    MIPROV2_CANDIDATE_PROGRAM_SCHEMA,
    MIPROV2_CANDIDATE_PROGRAM_SCHEMA_VERSION,
    EvalObservation,
    Miprov2CandidateAssemblyBinding,
    Miprov2ParameterSpace,
    Miprov2Study,
    Miprov2StudySchedule,
    StudyTranscript,
    StudyTranscriptMismatch,
    TrialParams,
)
from whetstone.testing.runtime import (
    build_miprov2_adapter,
    build_toy_copro_control,
    prepare_toy_miprov2_run,
    register_toy_runtime,
)
from whetstone.testing.toy.experiment import ToyTask, build_toy_experiment
from whetstone.testing.toy.miprov2 import (
    TOY_MIPROV2_COMPONENT_ID,
    build_toy_miprov2_control,
)


def _drive(
    store,
    *,
    run_id: str,
    demo_mode: Miprov2DemoMode,
) -> tuple[Miprov2Driver, list[Miprov2State]]:
    engine = ReferenceEvalRuntimeConfig().build_engine(store)
    control = build_toy_miprov2_control(engine=engine, demo_mode=demo_mode)
    adapter = build_miprov2_adapter(
        store=store, control=control, engine=engine
    )
    runtime = register_toy_runtime(
        store=store,
        engine=engine,
        copro_control=build_toy_copro_control(engine=engine),
        extra_adapters={MIPROV2_ADAPTER_KEY: adapter},
    )
    prepare_toy_miprov2_run(
        runtime, run_id=run_id, control=control, engine=engine
    )
    runtime.controller.drive(
        RunRequest(
            controller_identity_hash=runtime.controller.runtime_hash,
            run_id=run_id,
            control_identity_hash=control.identity_hash(),
        )
    )
    states: list[Miprov2State] = []
    index = 0
    while True:
        key = OptimHarness._result_binding_key(run_id, index)  # noqa: SLF001
        bound = store.resolve(key)
        if bound is None:
            break
        result = OptimStepResult.model_validate(store.get(bound))
        assert result.state_ref is not None
        snapshot = store.get(result.state_ref.reference)
        states.append(
            Miprov2State.model_validate(snapshot[MIPROV2_STATE_KEY])
        )
        index += 1
    return Miprov2Driver(), states


def _final_transcript(states: list[Miprov2State]) -> StudyTranscript:
    for state in reversed(states):
        transcript = state.study_transcript
        if transcript is not None and transcript.samples:
            return transcript
    raise AssertionError("no study transcript with samples survived the run")


def test_suggest_next_matches_the_live_runs_next_sample(tmp_path) -> None:
    with open_sqlite(str(tmp_path / "d10.sqlite")) as store:
        driver, states = _drive(
            store, run_id="d10-replay", demo_mode=Miprov2DemoMode.FEWSHOT
        )
        transcript = _final_transcript(states)
        assert transcript.demo_pool_identity_hashes is not None
        assert len(transcript.samples) >= 2
        prefix = transcript.model_copy(update={"samples": transcript.samples[:1]})
        recorded = transcript.samples[1]
        assert any(
            name.endswith("_predictor_demos") for name, _value in recorded.params
        )

        study = driver._study(states[-1])  # noqa: SLF001
        replayed = study.suggest_next(prefix)
        assert replayed.trial_number == recorded.trial_number
        assert replayed.params == recorded.params
        assert (
            replayed.candidate_combination_identity_hash
            == recorded.candidate_combination_identity_hash
        )


def test_a_drifted_suggestion_is_a_transcript_mismatch(tmp_path) -> None:
    with open_sqlite(str(tmp_path / "d10-mismatch.sqlite")) as store:
        driver, states = _drive(
            store, run_id="d10-mismatch", demo_mode=Miprov2DemoMode.FEWSHOT
        )
        transcript = _final_transcript(states)
        prefix = transcript.model_copy(update={"samples": transcript.samples[:1]})
        study = driver._study(states[-1])  # noqa: SLF001
        suggestion = study.suggest_next(prefix)
        drifted = suggestion.model_copy(
            update={"trial_number": suggestion.trial_number + 1}
        )

        with pytest.raises(StudyTranscriptMismatch):
            study.record_sample(
                prefix,
                drifted,
                score=0.0,
                evaluation=transcript.baseline.evaluation,
                candidate_assembly=MagicMock(),
            )


def _label_hash(label: str) -> str:
    return compute_identity_hash(
        schema="whetstone.test.miprov2_study_label",
        schema_version=1,
        payload={"label": label},
    )


def _labeled_demo(task_hash: str) -> LabeledTaskDemo:
    return LabeledTaskDemo(
        source_task_hash=task_hash,
        inputs_by_component=ImmutableJsonObject(
            {TOY_MIPROV2_COMPONENT_ID: {"prompt": "hello"}}
        ),
        outputs_by_component=ImmutableJsonObject(
            {TOY_MIPROV2_COMPONENT_ID: {"response": ""}}
        ),
    )


def _demo_set(*, seed: int, task_hash: str) -> ComponentDemoSet:
    return ComponentDemoSet(
        candidate_seed=seed,
        components=(
            ComponentDemoSequence(
                component_id=TOY_MIPROV2_COMPONENT_ID,
                demos=(_labeled_demo(task_hash).for_component(
                    TOY_MIPROV2_COMPONENT_ID
                ),),
            ),
        ),
    )


def _assemble(
    *,
    control,
    run,
    instruction_pools: tuple[tuple[str, ...], ...],
    demo_candidates: tuple[ComponentDemoSet, ...],
    params: TrialParams,
    combination_hash: str,
) -> Miprov2CandidateAssemblyBinding:
    rendering = _miprov2_candidate_rendering(
        control=control,
        instruction_pools=instruction_pools,
        demo_candidates=demo_candidates,
        params=params,
        categorical_combination_identity_hash=combination_hash,
    )
    candidate = candidate_from_components(
        base=control.base_candidate,
        candidate_id=f"miprov2-{rendering.identity_hash()[:24]}",
        components=rendering.model_dump(mode="json")["components"],
        run=run,
    )
    candidate_ref = candidate_reference(candidate)
    return Miprov2CandidateAssemblyBinding(
        params=params,
        categorical_combination_identity_hash=combination_hash,
        candidate=candidate_ref,
        program_identity_hash=compute_identity_hash(
            schema=MIPROV2_CANDIDATE_PROGRAM_SCHEMA,
            schema_version=MIPROV2_CANDIDATE_PROGRAM_SCHEMA_VERSION,
            payload={"candidate": candidate_ref.model_dump(mode="json")},
        ),
        rendering=rendering,
        optimizer_config=control.reference(),
        base_candidate=control.base_candidate,
        program_layout=control.program_layout,
        prompt_adapter_identity_hash=control.prompt_adapter_identity_hash,
        run=run,
    )


def _observation(
    *,
    control,
    engine,
    run_id: str,
    intent_id: str,
    purpose: str,
    candidate,
    tasks: tuple[str, ...],
    score: float,
) -> EvalObservation:
    effect_hash = _label_hash(intent_id)
    request = EvalBindingRequest(
        control_identity_hash=control.identity_hash(),
        source_eval_config=control.validation_eval_source,
        purpose=purpose.removeprefix("miprov2_"),
        effect_identity_hash=effect_hash,
        execution_policy=Miprov2EvaluationExecutionPolicy(
            max_errors=control.max_errors,
            provide_traceback=control.provide_traceback,
            task_model_identity_hash=control.task_model_identity_hash,
            provider_execution_policy_hash=(
                control.provider_execution_policy_hash
            ),
        ),
        task_batch_hashes=tasks,
    )
    binding = EngineEvalBindingResolver(engine=engine).resolve(request)
    evidence_ref = typed_ref_for_record(
        EVAL_EVIDENCE_SCHEMA,
        {"intent_id": intent_id, "score": score},
    )
    reward_value = score / 100.0
    reward = Reward(
        reward_name=control.reward_policy.reward_name,
        value=reward_value,
        reward_policy=control.reward_policy,
        evidence_role=EvalRole.INTERNAL,
        input_citations=(
            RewardInputCitation(
                name="score",
                value=reward_value,
                contributed=reward_value,
            ),
        ),
        evidence_refs=(evidence_ref,),
    )
    return EvalObservation(
        run_id=run_id,
        intent_id=intent_id,
        effect_identity_hash=effect_hash,
        purpose=purpose,
        candidate=candidate,
        task_batch_hashes=tasks,
        eval_config=binding.eval_config,
        eval_binding=binding,
        eval_role=EvalRole.INTERNAL,
        eval_result_ref=evidence_ref,
        expected_reward_policy_hash=control.reward_policy_hash,
        reward_ref=reward_reference(reward),
        normalized_score=score,
    )


def test_replay_matches_live_optuna_after_tpe_starts_and_across_promotions(
    tmp_path,
) -> None:
    import optuna

    experiment = build_toy_experiment(
        internal_tasks=tuple(
            ToyTask(
                task_id=f"task-{index}",
                prompt_inputs={"prompt": f"hello {index}"},
                gold=str(index),
            )
            for index in range(4)
        )
    )
    with open_sqlite(str(tmp_path / "tpe-replay.sqlite")) as store:
        engine = ReferenceEvalRuntimeConfig().build_engine(
            store, experiment=experiment
        )
        control = build_toy_miprov2_control(
            engine=engine,
            experiment=experiment,
            num_trials=25,
            minibatch=True,
            minibatch_full_eval_steps=5,
        )
        run = optimization_run_reference(
            OptimRun(
                run_id="tpe-replay",
                optimizer_config=control.reference(),
                adapter_key=MIPROV2_ADAPTER_KEY,
                mode=StepMode.PROPOSAL_ONLY,
                terminal_output_contract=OutputContract(
                    returned_proposal_count=1
                ),
                template_render_contract=control.template_render_contract,
                initial_candidate_ref=control.base_candidate,
                mutation_field=control.mutation_field,
                reward_policy=experiment.reward_policy,
            )
        )

    train_hash = control.trainset_task_hashes[0]
    instruction_pools = (
        tuple(f"Reply briefly to: {{prompt}} [{index}]" for index in range(3)),
    )
    demo_candidates = (
        _demo_set(seed=-1, task_hash=train_hash),
        _demo_set(seed=0, task_hash=train_hash),
    )
    space = Miprov2ParameterSpace(
        instruction_pool_identity_hashes=tuple(
            tuple(
                compute_identity_hash(
                    schema="whetstone.miprov2_instruction",
                    schema_version=1,
                    payload={"instruction": item},
                )
                for item in pool
            )
            for pool in instruction_pools
        ),
        demo_pool_identity_hashes=tuple(
            tuple(item.identity_hash() for item in demo_candidates)
            for _ in instruction_pools
        ),
    )
    valset = control.valset_task_hashes
    schedule = Miprov2StudySchedule(
        num_trials=25,
        minibatch=True,
        minibatch_size=2,
        valset_size=len(valset),
        minibatch_full_eval_steps=5,
    )
    study = Miprov2Study(
        seed=control.seed,
        demo_mode=control.demo_mode,
        space=space,
        schedule=schedule,
        run_id=run.record.run_id,
        validation_task_hashes=valset,
        validation_eval_source=control.validation_eval_source,
        reward_policy_hash=control.reward_policy_hash,
        optimizer_config=control.reference(),
        prompt_adapter_identity_hash=control.prompt_adapter_identity_hash,
        expected_base_candidate=control.base_candidate,
        program_layout=control.program_layout,
        run=run,
    )
    sample_tasks = valset[: schedule.minibatch_size]
    baseline_score = 10.0
    transcript = study.initial_transcript(
        baseline_score=baseline_score,
        baseline_evaluation=_observation(
            control=control,
            engine=engine,
            run_id=run.record.run_id,
            intent_id="baseline",
            purpose="miprov2_baseline",
            candidate=control.base_candidate,
            tasks=valset,
            score=baseline_score,
        ),
    )

    live = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=control.seed, multivariate=True),
    )
    live.add_trial(
        optuna.trial.create_trial(
            params=space.as_dict(space.baseline_params),
            distributions=study._distributions(optuna),  # noqa: SLF001
            value=baseline_score,
        )
    )

    promotions = 0
    for index in range(schedule.num_trials):
        prefix = transcript
        replayed = study.suggest_next(prefix)
        reconstructed = study.reconstruct_study(prefix)
        reconstructed_trial = reconstructed.ask()
        live_trial = live.ask()
        live_params = study._suggest(live_trial)  # noqa: SLF001
        reconstructed_params = study._suggest(reconstructed_trial)  # noqa: SLF001
        assert replayed.trial_number == live_trial.number == reconstructed_trial.number
        assert replayed.params == live_params
        assert replayed.params == reconstructed_params
        assert any(
            name.endswith("_predictor_demos") for name, _value in replayed.params
        )

        score = float(20 + (index % 7))
        assembly = _assemble(
            control=control,
            run=run,
            instruction_pools=instruction_pools,
            demo_candidates=demo_candidates,
            params=replayed.params,
            combination_hash=replayed.candidate_combination_identity_hash,
        )
        evaluation = _observation(
            control=control,
            engine=engine,
            run_id=run.record.run_id,
            intent_id=f"sample-{index}",
            purpose="miprov2_sample",
            candidate=assembly.candidate,
            tasks=sample_tasks,
            score=score,
        )
        promotion_kwargs: dict[str, object] = {}
        if schedule.promotion_due(optuna_trial_number=replayed.trial_number):
            selected = study.promotion_candidate(
                prefix,
                replayed,
                score=score,
                evaluation=evaluation,
                candidate_assembly=assembly,
            )
            assert selected is not None
            promotions += 1
            promotion_score = float(40 + promotions)
            promotion_evaluation = _observation(
                control=control,
                engine=engine,
                run_id=run.record.run_id,
                intent_id=f"promotion-{promotions}",
                purpose="miprov2_promotion",
                candidate=selected.candidate_assembly.candidate,
                tasks=valset,
                score=promotion_score,
            )
            added = study._add_completed_trial(  # noqa: SLF001
                live,
                params=selected.params,
                value=promotion_score,
            )
            assert added.number == replayed.trial_number + 1
            promotion_kwargs = {
                "promotion_full_score": promotion_score,
                "promotion_evaluation": promotion_evaluation,
            }

        live.tell(live_trial, score)
        transcript = study.record_sample(
            prefix,
            replayed,
            score=score,
            evaluation=evaluation,
            candidate_assembly=assembly,
            **promotion_kwargs,
        )

    assert len(transcript.samples) == 25
    assert promotions == 5
    assert sum(1 for sample in transcript.samples if sample.promotion) == 5

