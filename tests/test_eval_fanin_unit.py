from __future__ import annotations

from unittest.mock import MagicMock

from dr_store.content_addressing import format_object_reference

from whetstone.coordination.eval_service import EvalDispatchMode, EvalEngineService
from whetstone.eval.protocol import EvalRequest
from whetstone.optim.contracts import OptimEvalRequest
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
    row_payload = {
        "optim_eval_request": intent.model_dump(mode="json"),
        "task_id": "task-0",
        "seed_index": 0,
    }
    row_ref, _ = runtime.store.put("whetstone.platform_eval_row_input", row_payload)
    row_reference = format_object_reference(row_ref)
    row_calls: list[str] = []

    def row_executor(**kwargs) -> None:
        row_calls.append(kwargs["task_id"])

    execute_eval_row_sync(
        runtime,
        input_reference=row_reference,
        row_executor=row_executor,
    )
    assert row_calls == ["task-0"]

    fanin_payload = {
        "optim_eval_request": intent.model_dump(mode="json"),
    }
    fanin_ref, _ = runtime.store.put(
        "whetstone.platform_eval_fanin_input",
        fanin_payload,
    )
    fanin_reference = format_object_reference(fanin_ref)
    row_loader = MagicMock()
    output = execute_eval_fanin_sync(
        runtime,
        input_reference=fanin_reference,
        row_loader=row_loader,
    )
    row_loader.assert_called_once()
    assert "whetstone.platform_eval_fanin" in output
