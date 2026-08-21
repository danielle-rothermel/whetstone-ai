from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock

import pytest

from whetstone.coordination.eval_service import EvalDispatchMode
from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
from whetstone.eval.runtime_engine import RuntimeEvalEngine
from whetstone.testing.toy.experiment import build_toy_experiment
from whetstone.platform.contracts import (
    OptimWorkInput,
    load_eval_row_input,
    persist_work_input,
)
from whetstone.platform.eval_fanin import (
    build_platform_row_executor,
    execute_eval_row_sync,
)
from whetstone.platform.step_executor import (
    STAGE_EVAL_ROW,
    OptimWorkState,
    _load_work_state,
    execute_optim_step_sync,
)
from whetstone.platform.submit import OptimRunMemberSpec, submit_optim_run
from dr_providers import RequestControl, openrouter_chat_config

from whetstone.provider.language_model import PlainPromptAdapter
from whetstone.provider.llm_call import (
    build_provider_request,
    derive_rng_seed,
)
from whetstone.testing.toy.experiment import (
    _reference_provider_call_config,
)


def test_for_task_ids_preserves_explicit_zero_rng_seed(sqlite_store) -> None:
    config = ReferenceEvalRuntimeConfig()
    base = config.build_engine(sqlite_store)
    experiment = build_toy_experiment(num_seeds=2)
    engine = RuntimeEvalEngine(
        store=sqlite_store,
        experiment=experiment,
        sampling=experiment.eval_configs.internal,
        execution_policy=config.execution_policy,
        driver=base._driver,  # noqa: SLF001
    )
    task_id = engine.sampling.tasks[0].task_id
    task_hash = engine.sampling.task_hashes[0]
    patched_seed_plan = engine._sampling.seed_plan.model_copy(  # noqa: SLF001
        update={"rng_seeds": {f"{task_hash}#0": 0}}
    )
    patched_sampling = replace(
        engine._sampling,  # noqa: SLF001
        seed_plan=patched_seed_plan,
    )
    engine._sampling = patched_sampling  # noqa: SLF001
    derived = RuntimeEvalEngine._derive_sampling(  # noqa: SLF001
        patched_sampling,
        (task_id,),
    )
    derived_rng = dict(derived.seed_plan.rng_seeds)
    assert derived_rng[f"{task_hash}#0"] == 0
    assert derived_rng[f"{task_hash}#1"] == derive_rng_seed(task_hash, 1)


def test_execute_optim_step_rejects_mismatched_controller_identity(
    copro_launch,
) -> None:
    runtime, launch = copro_launch
    control = launch.control
    assert control is not None
    runtime.controller.bind_launch(launch)
    work_input = OptimWorkInput(
        run_id=launch.run.run_id,
        controller_identity_hash="0" * 64,
        control_identity_hash=control.identity_hash(),
    )
    input_reference = persist_work_input(runtime.store, work_input)
    with pytest.raises(ValueError, match="controller identity"):
        execute_optim_step_sync(
            runtime,
            input_reference=input_reference,
            stage_index=0,
        )


def test_execute_eval_row_rejects_mismatched_controller_identity(
    copro_launch,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, launch = copro_launch
    control = launch.control
    assert control is not None
    runtime.controller.bind_launch(launch)
    work_input = OptimWorkInput(
        run_id=launch.run.run_id,
        controller_identity_hash=runtime.controller.runtime_hash,
        control_identity_hash=control.identity_hash(),
        dispatch_mode=EvalDispatchMode.PLATFORM,
    )
    input_reference = persist_work_input(runtime.store, work_input)
    step_completion = execute_optim_step_sync(
        runtime,
        input_reference=input_reference,
        stage_index=0,
    )
    row_successor = next(
        successor
        for successor in step_completion.successors
        if successor.stage_key.value == STAGE_EVAL_ROW
    )
    row_input = load_eval_row_input(runtime.store, row_successor.input_reference)
    original_load = _load_work_state

    def load_with_tampered(runtime_arg, ref: str):
        state = original_load(runtime_arg, ref)
        if ref != row_input.work_state_ref:
            return state
        return OptimWorkState(
            work_input=state.work_input.model_copy(
                update={"controller_identity_hash": "0" * 64}
            ),
            step_index=state.step_index,
            step_result_refs=state.step_result_refs,
            terminal=state.terminal,
            pending_step_result_ref=state.pending_step_result_ref,
            deferral_optim_step_stage_index=state.deferral_optim_step_stage_index,
            pending_deferred_intents=state.pending_deferred_intents,
        )

    monkeypatch.setattr(
        "whetstone.platform.eval_fanin._load_work_state",
        load_with_tampered,
    )
    with pytest.raises(ValueError, match="controller identity"):
        execute_eval_row_sync(
            runtime,
            input_reference=row_successor.input_reference,
            stage_index=row_successor.stage_index,
            row_executor=build_platform_row_executor(runtime),
        )


def test_submit_optim_run_rejects_mismatched_controller_identity(
    copro_launch,
) -> None:
    runtime, launch = copro_launch
    with pytest.raises(ValueError, match="controller identity"):
        submit_optim_run(
            runtime=runtime,
            registry=MagicMock(),
            engine=MagicMock(),
            campaign_key="campaign-1",
            run_key="run-1",
            members=(OptimRunMemberSpec(work_key="work-1", launch=launch),),
            controller_identity_hash="0" * 64,
            execution_config_reference="exec-config-ref",
        )


def test_build_provider_request_carries_derived_seed() -> None:
    """The eval-derived rng_seed reaches the provider config controls."""
    config = openrouter_chat_config(model="seed-test-model")
    assert config.definition.constraints.supports(RequestControl.SEED)
    rng_seed = derive_rng_seed("candidate-a", "task-a", 0)
    request = build_provider_request(
        provider_config=config,
        rng_seed=rng_seed,
        prompt="hello",
        prompt_adapter=PlainPromptAdapter(),
    )
    assert request.config.controls.seed == rng_seed
    assert request.config.controls.identity_payload()["seed"] == rng_seed


def test_provider_seed_participates_in_request_identity() -> None:
    """Distinct eval seeds yield distinct provider request identities."""
    config = openrouter_chat_config(model="seed-test-model")
    identities = {
        build_provider_request(
            provider_config=config,
            rng_seed=seed,
            prompt="hello",
            prompt_adapter=PlainPromptAdapter(),
        ).identity_hash
        for seed in (11, 12)
    }
    assert len(identities) == 2


def test_toy_provider_definition_transports_seed() -> None:
    """The toy eval path is seeded, so its definition must advertise SEED."""
    request = build_provider_request(
        provider_config=_reference_provider_call_config(),
        rng_seed=4242,
        prompt="hello",
        prompt_adapter=PlainPromptAdapter(),
    )
    assert request.config.controls.seed == 4242


def test_build_provider_request_rejects_caller_supplied_seed() -> None:
    """Seed stays single-sourced from eval derivation."""
    with pytest.raises(ValueError, match="must not include seed"):
        build_provider_request(
            provider_config=openrouter_chat_config(model="seed-test-model"),
            rng_seed=5,
            prompt="hello",
            parameters={"seed": 9},
            prompt_adapter=PlainPromptAdapter(),
        )
