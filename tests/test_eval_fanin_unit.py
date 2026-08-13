from __future__ import annotations

from unittest.mock import MagicMock

from whetstone.coordination.eval_service import EvalDispatchMode, EvalEngineService
from whetstone.coordination.runtime_bootstrap import build_toy_copro_control
from whetstone.eval.protocol import EvalRequest
from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
from whetstone.optim.contracts import OptimEvalRequest
from whetstone.platform.contracts import (
    EvalFaninInput,
    EvalRowInput,
    OptimWorkInput,
    persist_eval_fanin_input,
    persist_eval_row_input,
)
from whetstone.platform.eval_fanin import (
    execute_eval_fanin_sync,
    execute_eval_row_sync,
    serialize_platform_eval_intent,
)


def _platform_runtime(toy_runtime):
    runtime, control = toy_runtime
    engine = runtime.eval_service._engine  # noqa: SLF001
    eval_service = EvalEngineService(
        store=runtime.store,
        engine=engine,
        dispatch_mode=EvalDispatchMode.PLATFORM,
    )
    object.__setattr__(runtime, "eval_service", eval_service)
    runtime.harness._evaluation_service = eval_service  # noqa: SLF001
    return runtime, control


def test_platform_intent_serialization(toy_runtime) -> None:
    _runtime, _control = _platform_runtime(toy_runtime)
    from whetstone.testing.toy.experiment import build_toy_experiment

    experiment = build_toy_experiment(num_seeds=1)
    intent = OptimEvalRequest(
        optim_run_id="run-platform",
        optim_step_index=0,
        eval_request=EvalRequest(
            request_id="eval-1",
            candidate=experiment.initial_candidate,
        ),
        expected_reward_policy_hash=experiment.reward_policy.identity_hash(),
    )
    payload = serialize_platform_eval_intent(intent)
    assert payload["pending"] is True
    assert payload["optim_eval_request"]["optim_run_id"] == "run-platform"


def test_eval_fanin_resolution_with_mock_row_executor(toy_runtime) -> None:
    runtime, _control = _platform_runtime(toy_runtime)
    from whetstone.testing.toy.experiment import build_toy_experiment

    experiment = build_toy_experiment(num_seeds=1)
    intent = OptimEvalRequest(
        optim_run_id="run-platform",
        optim_step_index=0,
        eval_request=EvalRequest(
            request_id="eval-1",
            candidate=experiment.initial_candidate,
        ),
        expected_reward_policy_hash=experiment.reward_policy.identity_hash(),
    )
    runtime.eval_service.persist_platform_intent(intent)
    row_reference = persist_eval_row_input(
        runtime.store,
        EvalRowInput(
            batch_id="batch-1",
            optim_eval_request=intent,
            task_id="task-0",
            seed_index=0,
        ),
    )
    row_calls: list[str] = []

    def row_executor(**kwargs) -> None:
        row_calls.append(kwargs["task_id"])

    row_completion = execute_eval_row_sync(
        runtime,
        input_reference=row_reference,
        row_executor=row_executor,
    )
    assert row_calls == ["task-0"]
    assert row_completion.output_reference

    fanin_reference = persist_eval_fanin_input(
        runtime.store,
        EvalFaninInput(
            batch_id="batch-1",
            optim_eval_request=intent,
        ),
    )
    from whetstone.platform.contracts import EvalBatch, persist_eval_batch
    from whetstone.platform.step_executor import OptimWorkState, _persist_work_state

    pending_state = OptimWorkState(
        work_input=OptimWorkInput(
            run_id="run-platform",
            controller_identity_hash="a" * 64,
            control_identity_hash="b" * 64,
            platform_stage_index=3,
        ),
        step_index=1,
        step_result_refs=(),
        terminal=False,
    )
    work_state_ref = _persist_work_state(runtime, pending_state)
    persist_eval_batch(
        runtime.store,
        EvalBatch(
            batch_id="batch-1",
            run_id="run-platform",
            step_index=0,
            optim_step_stage_index=0,
            row_input_refs=(row_reference,),
            fanin_input_ref=fanin_reference,
            work_state_ref=work_state_ref,
        ),
    )
    row_loader = MagicMock()
    fanin_completion = execute_eval_fanin_sync(
        runtime,
        input_reference=fanin_reference,
        row_loader=row_loader,
    )
    row_loader.assert_called_once()
    assert fanin_completion.output_reference
    assert fanin_completion.successors
    assert fanin_completion.successors[0].stage_key.value == "optim_step"
