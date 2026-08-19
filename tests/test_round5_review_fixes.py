from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from whetstone.coordination.eval_service import EvalDispatchMode, EvalEngineService
from whetstone.eval.protocol import EvalRequest
from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
from whetstone.eval.runtime_engine import RuntimeEvalEngine
from whetstone.platform.contracts import OptimWorkInput, persist_work_input
from whetstone.platform.eval_fanin import (
    build_platform_row_executor,
    execute_eval_fanin_sync,
    execute_eval_row_sync,
)
from whetstone.platform.step_executor import STAGE_EVAL_FANIN, STAGE_EVAL_ROW, execute_optim_step_sync
from whetstone.provider.llm_call import derive_rng_seed
from whetstone.testing.toy.experiment import build_toy_experiment


def _multi_seed_engine(sqlite_store) -> RuntimeEvalEngine:
    config = ReferenceEvalRuntimeConfig()
    base = config.build_engine(sqlite_store)
    experiment = build_toy_experiment(num_seeds=2)
    sampling = experiment.eval_configs.internal
    assert dict(sampling.seed_plan.rng_seeds) == {}
    return RuntimeEvalEngine(
        store=sqlite_store,
        experiment=experiment,
        sampling=sampling,
        execution_policy=config.execution_policy,
        driver=base._driver,  # noqa: SLF001
    )


def _run_platform_deferral_to_fanin(copro_launch):
    runtime, launch = copro_launch
    control = launch.control
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
    row_successors = [
        successor
        for successor in step_completion.successors
        if successor.stage_key.value == STAGE_EVAL_ROW
    ]
    fanin_successors = [
        successor
        for successor in step_completion.successors
        if successor.stage_key.value == STAGE_EVAL_FANIN
    ]
    for row_successor in row_successors:
        execute_eval_row_sync(
            runtime,
            input_reference=row_successor.input_reference,
            stage_index=row_successor.stage_index,
            row_executor=build_platform_row_executor(runtime),
        )
    return runtime, fanin_successors[0]


def test_for_task_seed_synthesizes_logical_seed_without_provenance(
    sqlite_store,
) -> None:
    engine = _multi_seed_engine(sqlite_store)
    task_id = engine.sampling.tasks[0].task_id
    task_hash = engine.sampling.task_hashes[0]
    request = EvalRequest(
        request_id="logical-seed-synthesis",
        candidate=engine.experiment.initial_candidate,
    )
    captured: list[int] = []

    def capture_run_node(*, llm_deps, eval_deps):
        captured.append(llm_deps.rng_seed)
        return MagicMock()

    with patch(
        "whetstone.eval.drivers.graph_rollout.build_run_node",
        side_effect=capture_run_node,
    ):
        engine.for_task_seed(task_id, 0).evaluate_row(request)
        engine.for_task_seed(task_id, 1).evaluate_row(request)

    assert len(captured) >= 2
    assert captured[0] != captured[1]
    assert captured[0] == derive_rng_seed(task_hash, 0)
    assert captured[1] == derive_rng_seed(task_hash, 1)


def test_for_task_ids_synthesizes_missing_provenance_entries(sqlite_store) -> None:
    engine = _multi_seed_engine(sqlite_store)
    task_id = engine.sampling.tasks[0].task_id
    task_hash = engine.sampling.task_hashes[0]
    scoped = engine.for_task_ids((task_id,))
    derived_rng = dict(scoped._sampling.seed_plan.rng_seeds)  # noqa: SLF001
    assert derived_rng[f"{task_hash}#0"] == derive_rng_seed(task_hash, 0)
    assert derived_rng[f"{task_hash}#1"] == derive_rng_seed(task_hash, 1)


def test_fanin_resumes_after_preempted_clear_before_bind(copro_launch) -> None:
    runtime, fanin_successor = _run_platform_deferral_to_fanin(copro_launch)
    service = runtime.eval_service
    assert isinstance(service, EvalEngineService)
    intent = runtime.harness.last_deferred_platform_intents[0]  # noqa: SLF001
    assert runtime.store.resolve(service._key(intent)) is None
    service._clear_platform_intent(intent)
    assert service.load_platform_intent(intent) is None

    completion = execute_eval_fanin_sync(
        runtime,
        input_reference=fanin_successor.input_reference,
        stage_index=fanin_successor.stage_index,
    )
    assert runtime.store.resolve(service._key(intent)) is not None
    assert completion.successors[0].stage_key.value == "optim_step"
