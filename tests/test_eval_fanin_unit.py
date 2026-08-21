from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from dr_platform._core.identities import StageKey
from dr_store.content_addressing import format_object_reference, parse_object_reference

from whetstone.coordination.eval_service import EvalDispatchMode
from whetstone.eval.protocol import EvalRequest
from whetstone.optim.contracts import OptimEvalRequest
from whetstone.platform.contracts import (
    OptimWorkInput,
    STAGE_EVAL_FANIN,
    STAGE_EVAL_ROW,
    load_deferral_join_input,
    persist_work_input,
)
from whetstone.platform.eval_fanin import (
    build_platform_row_executor,
    execute_eval_fanin_sync,
    execute_eval_row_sync,
    serialize_platform_eval_intent,
)
from whetstone.platform.step_executor import execute_optim_step_sync


def _emit_platform_deferral(copro_launch):
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
    fanin_successor = next(
        successor
        for successor in step_completion.successors
        if successor.stage_key.value == STAGE_EVAL_FANIN
    )
    return runtime, row_successors, fanin_successor


def _complete_eval_rows(runtime, row_successors) -> None:
    platform_executor = build_platform_row_executor(runtime)
    for row_successor in row_successors:
        execute_eval_row_sync(
            runtime,
            input_reference=row_successor.input_reference,
            stage_index=row_successor.stage_index,
            row_executor=platform_executor,
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
    platform_executor = build_platform_row_executor(runtime)

    def row_executor(**kwargs):
        row_calls.append(kwargs["task_id"])
        return platform_executor(**kwargs)

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
    assert fanin_completion.successors[0].stage_index > fanin_successors[0].stage_index


def test_eval_fanin_ignores_other_episode_predecessors(copro_launch) -> None:
    from dr_platform.inspection.work_items import PredecessorStageOutput

    from whetstone.platform.eval_fanin import PLATFORM_EVAL_ROW_SCHEMA

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
    row_outputs: list[str] = []
    platform_executor = build_platform_row_executor(runtime)
    for row_successor in row_successors:
        row_completion = execute_eval_row_sync(
            runtime,
            input_reference=row_successor.input_reference,
            stage_index=row_successor.stage_index,
            row_executor=platform_executor,
        )
        row_outputs.append(row_completion.output_reference)
    stale_output_ref, _ = runtime.store.put(
        PLATFORM_EVAL_ROW_SCHEMA,
        {
            "schema_version": 1,
            "optim_eval_request": {},
            "task_id": "stale-task",
            "seed_index": 0,
            "deferral_origin_stage_index": 0,
            "row_ordinal": 99,
            "completed": True,
        },
    )
    stale_output = format_object_reference(stale_output_ref)
    object.__setattr__(runtime, "ledger_engine", MagicMock())
    episode_predecessors = tuple(
        PredecessorStageOutput(
            stage_index=index + 1,
            stage_key=StageKey(STAGE_EVAL_ROW),
            input_reference=f"row-in-{index}",
            output_reference=output_reference,
        )
        for index, output_reference in enumerate(row_outputs)
    )
    with patch(
        "whetstone.platform.eval_fanin.list_episode_predecessor_outputs",
        return_value=episode_predecessors,
    ):
        execute_eval_fanin_sync(
            runtime,
            input_reference=fanin_successors[0].input_reference,
            stage_index=fanin_successors[0].stage_index,
            work_item_id=1,
        )
    _ = stale_output


def test_eval_fanin_requires_ledger_when_work_item_id_present(copro_launch) -> None:
    runtime, row_successors, fanin_successor = _emit_platform_deferral(copro_launch)
    _complete_eval_rows(runtime, row_successors)
    try:
        execute_eval_fanin_sync(
            runtime,
            input_reference=fanin_successor.input_reference,
            stage_index=fanin_successor.stage_index,
            work_item_id=1,
        )
    except ValueError as error:
        assert "requires a ledger engine" in str(error)
    else:
        raise AssertionError("expected missing ledger engine")


def test_eval_fanin_uses_persisted_row_refs_without_reexpansion(copro_launch) -> None:
    runtime, row_successors, fanin_successor = _emit_platform_deferral(copro_launch)
    _complete_eval_rows(runtime, row_successors)
    with patch(
        "whetstone.platform.step_executor._expand_eval_rows",
        side_effect=AssertionError("fan-in must not re-expand eval rows"),
    ):
        completion = execute_eval_fanin_sync(
            runtime,
            input_reference=fanin_successor.input_reference,
            stage_index=fanin_successor.stage_index,
        )
    assert completion.output_reference
    assert completion.successors


def test_eval_fanin_store_path_raises_when_row_unbound(copro_launch) -> None:
    runtime, row_successors, fanin_successor = _emit_platform_deferral(copro_launch)
    _complete_eval_rows(runtime, row_successors[:-1])
    try:
        execute_eval_fanin_sync(
            runtime,
            input_reference=fanin_successor.input_reference,
            stage_index=fanin_successor.stage_index,
        )
    except ValueError as error:
        assert "not bound to a completion record" in str(error)
    else:
        raise AssertionError("expected unbound eval row input")


def test_eval_fanin_store_path_raises_when_row_count_mismatches(
    copro_launch,
) -> None:
    runtime, row_successors, fanin_successor = _emit_platform_deferral(copro_launch)
    _complete_eval_rows(runtime, row_successors)
    with patch(
        "whetstone.platform.eval_fanin._expected_episode_row_count",
        return_value=len(row_successors) + 1,
    ):
        try:
            execute_eval_fanin_sync(
                runtime,
                input_reference=fanin_successor.input_reference,
                stage_index=fanin_successor.stage_index,
            )
        except ValueError as error:
            assert "persisted row refs do not match episode row count" in str(error)
        else:
            raise AssertionError("expected persisted row count mismatch")


def test_deferral_join_input_includes_emitted_row_refs(copro_launch) -> None:
    runtime, row_successors, fanin_successor = _emit_platform_deferral(copro_launch)
    join_input = load_deferral_join_input(
        runtime.store,
        fanin_successor.input_reference,
    )
    assert join_input.row_input_refs == tuple(
        successor.input_reference for successor in row_successors
    )


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
        "whetstone.platform.eval_fanin.list_episode_predecessor_outputs",
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
            assert "ledger predecessors do not match episode row count" in str(error)
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


def test_platform_row_executor_scopes_evaluation_to_task_seed(
    copro_launch, monkeypatch
) -> None:
    from unittest.mock import MagicMock

    from whetstone.platform.eval_fanin import build_platform_row_executor

    from whetstone.eval.row_slice import RowEvalCompletion

    from whetstone.core.identity import TypedRef

    runtime, launch = copro_launch
    runtime.controller.bind_launch(launch)
    scoped_engine = MagicMock()
    scoped_engine.evaluate_row.return_value = RowEvalCompletion(
        evidence_ref=TypedRef(
            schema_name="whetstone.eval_evidence",
            content_hash="a" * 64,
        ),
    )
    original_for_task_seed = runtime.eval_service._engine.for_task_seed  # noqa: SLF001
    captured: list[tuple[str, int]] = []

    def for_task_seed(task_id, seed_index):
        captured.append((task_id, seed_index))
        return scoped_engine

    monkeypatch.setattr(
        runtime.eval_service._engine,
        "for_task_seed",
        for_task_seed,
    )
    executor = build_platform_row_executor(runtime)
    from whetstone.testing.toy.experiment import build_toy_experiment
    from whetstone.optim.contracts import OptimEvalRequest
    from whetstone.eval.protocol import EvalRequest

    experiment = build_toy_experiment(num_seeds=2)
    intent = OptimEvalRequest(
        optim_run_id=launch.run.run_id,
        optim_step_index=0,
        eval_request=EvalRequest(
            request_id="row-eval",
            candidate=experiment.initial_candidate,
        ),
        expected_reward_policy_hash=experiment.reward_policy.identity_hash(),
    )
    executor(intent=intent, task_id="task-a", seed_index=1)
    assert captured == [("task-a", 1)]
    scoped_engine.evaluate_row.assert_called_once()
    _ = original_for_task_seed


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


def test_run_manifest_rejects_duplicate_work_keys() -> None:
    from whetstone.platform.contracts import OptimRunManifest, OptimRunMemberEntry

    with pytest.raises(ValueError, match="work_key"):
        OptimRunManifest(
            platform_run_key="run-1",
            membership_digest="digest-1",
            members=(
                OptimRunMemberEntry(work_key="work-1", run_id="harness-run-1"),
                OptimRunMemberEntry(work_key="work-1", run_id="harness-run-2"),
            ),
        )


def test_run_manifest_rejects_duplicate_run_ids() -> None:
    from whetstone.platform.contracts import OptimRunManifest, OptimRunMemberEntry

    with pytest.raises(ValueError, match="run_id"):
        OptimRunManifest(
            platform_run_key="run-1",
            membership_digest="digest-1",
            members=(
                OptimRunMemberEntry(work_key="work-1", run_id="harness-run-1"),
                OptimRunMemberEntry(work_key="work-2", run_id="harness-run-1"),
            ),
        )


def test_run_result_roundtrip(sqlite_store) -> None:
    from whetstone.platform.contracts import (
        OptimPlatformRunResult,
        OptimRunMemberResult,
        load_run_result,
        persist_run_result,
    )

    run_result = OptimPlatformRunResult(
        platform_run_key="run-1",
        membership_digest="digest-1",
        member_results=(
            OptimRunMemberResult(
                work_key="work-1",
                run_id="harness-run-1",
                result_reference="whetstone.optim_result:aaaa",
            ),
            OptimRunMemberResult(
                work_key="work-2",
                run_id="harness-run-2",
                result_reference="whetstone.optim_result:bbbb",
            ),
        ),
    )
    reference = persist_run_result(sqlite_store, run_result)
    loaded = load_run_result(sqlite_store, reference)
    assert loaded == run_result
    assert format_object_reference(parse_object_reference(reference)) == reference


def test_eval_fanin_scopes_predecessor_read_to_the_deferral_episode(
    copro_launch,
) -> None:
    """Fan-in must read eval rows above the deferring step, not the whole item.

    The origin filter is what keeps a second deferral episode from seeing the
    first episode's rows; it is passed to the upstream episode reader.
    """
    from dr_platform.inspection.work_items import PredecessorStageOutput

    runtime, row_successors, fanin_successor = _emit_platform_deferral(copro_launch)
    platform_executor = build_platform_row_executor(runtime)
    row_outputs = []
    for row_successor in row_successors:
        row_completion = execute_eval_row_sync(
            runtime,
            input_reference=row_successor.input_reference,
            stage_index=row_successor.stage_index,
            row_executor=platform_executor,
        )
        row_outputs.append(row_completion.output_reference)
    object.__setattr__(runtime, "ledger_engine", MagicMock())
    episode_predecessors = tuple(
        PredecessorStageOutput(
            stage_index=index + 1,
            stage_key=StageKey(STAGE_EVAL_ROW),
            input_reference=f"row-in-{index}",
            output_reference=output_reference,
        )
        for index, output_reference in enumerate(row_outputs)
    )
    with patch(
        "whetstone.platform.eval_fanin.list_episode_predecessor_outputs",
        return_value=episode_predecessors,
    ) as list_predecessors:
        execute_eval_fanin_sync(
            runtime,
            input_reference=fanin_successor.input_reference,
            stage_index=fanin_successor.stage_index,
            work_item_id=1,
        )
    list_predecessors.assert_called_once_with(
        1,
        fanin_successor.stage_index,
        origin_stage_index=0,
        stage_key=STAGE_EVAL_ROW,
        engine=runtime.ledger_engine,
    )
