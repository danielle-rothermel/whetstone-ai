from __future__ import annotations

from unittest.mock import MagicMock, patch

from dr_store.content_addressing import format_object_reference, parse_object_reference

from whetstone.coordination.eval_service import EvalDispatchMode
from whetstone.eval.protocol import EvalRequest
from whetstone.optim.contracts import OptimEvalRequest
from whetstone.platform.contracts import (
    EvalFaninInput,
    EvalRowInput,
    OptimWorkInput,
    STAGE_EVAL_FANIN,
    persist_eval_fanin_input,
    persist_eval_row_input,
    persist_work_input,
)
from whetstone.platform.eval_fanin import (
    execute_eval_fanin_sync,
    execute_eval_row_sync,
    serialize_platform_eval_intent,
)
from whetstone.platform.step_executor import (
    STAGE_EVAL_ROW,
    execute_optim_step_sync,
)


def test_platform_intent_serialization(toy_runtime) -> None:
    _runtime, _control = toy_runtime
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


def test_eval_fanin_resolution_with_mock_row_executor(copro_launch) -> None:
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
    assert row_successors
    assert len(fanin_successors) == 1
    assert fanin_successors[0].barrier is True

    row_calls: list[str] = []

    def row_executor(**kwargs) -> None:
        row_calls.append(kwargs["task_id"])

    for row_successor in row_successors:
        execute_eval_row_sync(
            runtime,
            input_reference=row_successor.input_reference,
            stage_index=row_successor.stage_index,
            row_executor=row_executor,
        )
    assert row_calls

    row_loader = MagicMock()
    fanin_completion = execute_eval_fanin_sync(
        runtime,
        input_reference=fanin_successors[0].input_reference,
        stage_index=fanin_successors[0].stage_index,
        row_loader=row_loader,
    )
    row_loader.assert_called_once()
    assert fanin_completion.output_reference
    assert fanin_completion.successors
    assert fanin_completion.successors[0].stage_key.value == "optim_step"


def test_eval_fanin_ledger_predecessor_mismatch_raises(copro_launch) -> None:
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
        )
    object.__setattr__(runtime, "ledger_engine", MagicMock())
    with patch(
        "dr_platform.inspection.work_items.list_predecessor_stage_outputs",
        return_value=(),
    ):
        try:
            execute_eval_fanin_sync(
                runtime,
                input_reference=fanin_successors[0].input_reference,
                stage_index=fanin_successors[0].stage_index,
                work_item_id=1,
            )
        except ValueError as error:
            assert "ledger predecessors do not match batch row count" in str(error)
        else:
            raise AssertionError("expected ledger predecessor mismatch")


def test_platform_stage_index_mismatch_raises(copro_launch) -> None:
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
    try:
        execute_optim_step_sync(
            runtime,
            input_reference=input_reference,
            stage_index=99,
        )
    except ValueError as error:
        assert "stage_index mismatch" in str(error)
    else:
        raise AssertionError("expected stage_index mismatch")


def test_run_manifest_roundtrip(sqlite_store) -> None:
    from whetstone.platform.contracts import (
        OptimRunManifest,
        OptimRunMemberEntry,
        load_run_manifest,
        persist_run_manifest,
    )

    manifest = OptimRunManifest(
        platform_run_key="run-1",
        membership_digest="digest-1",
        members=(OptimRunMemberEntry(work_key="work-1", run_id="harness-run-1"),),
    )
    reference = persist_run_manifest(sqlite_store, manifest)
    loaded = load_run_manifest(sqlite_store, reference)
    assert loaded == manifest
    assert format_object_reference(parse_object_reference(reference)) == reference
