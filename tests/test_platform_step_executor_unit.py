from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import patch
from uuid import uuid4

import pytest
from dr_platform._core.ledger.states import StageExecutionState
from dr_platform.completion.execution import RunCompletionPayload, StateCount
from dr_store.content_addressing import format_object_reference

from whetstone.coordination.eval_service import EvalDispatchMode
from whetstone.coordination.runtime_bootstrap import (
    build_toy_copro_control,
    prepare_copro_run,
    register_runtime,
)
from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
from whetstone.platform.contracts import (
    OPTIM_PIPELINE_KEY,
    OPTIM_PIPELINE_VERSION,
    STAGE_EVAL_FANIN,
    STAGE_EVAL_ROW,
    OptimRunManifest,
    OptimRunMemberEntry,
    OptimWorkInput,
    load_run_result,
    persist_run_manifest,
    persist_work_input,
)
from whetstone.platform.deferred_intents import (
    evict_deferred_intents,
    load_persisted_deferred_intents,
    persist_deferred_intents,
)
from whetstone.platform.eval_fanin import execute_eval_fanin_sync, execute_eval_row_sync
from whetstone.platform.step_executor import (
    RUN_MEMBER_TERMINAL_BINDING_PREFIX,
    OptimWorkState,
    _evict_step_result_binding,
    _load_work_state,
    _persist_work_state,
    _platform_deferred_successors,
    execute_optim_step_sync,
    execute_run_completion_for_run_sync,
)
from whetstone.platform.work_state_head import resolve_work_state_head


def _complete_deferral_episode(runtime, completion):
    row_successors = [
        successor
        for successor in completion.successors
        if successor.stage_key.value == STAGE_EVAL_ROW
    ]
    fanin_successors = [
        successor
        for successor in completion.successors
        if successor.stage_key.value == STAGE_EVAL_FANIN
    ]
    for row_successor in row_successors:
        execute_eval_row_sync(
            runtime,
            input_reference=row_successor.input_reference,
            stage_index=row_successor.stage_index,
        )
    return execute_eval_fanin_sync(
        runtime,
        input_reference=fanin_successors[0].input_reference,
        stage_index=fanin_successors[0].stage_index,
    )


def _depth2_copro_launch(sqlite_store):
    runtime_config = ReferenceEvalRuntimeConfig()
    engine = runtime_config.build_engine(sqlite_store)
    control = build_toy_copro_control(breadth=2, depth=2, engine=engine)
    runtime = register_runtime(store=sqlite_store, copro_control=control)
    launch = prepare_copro_run(
        runtime,
        run_id=f"test-run-{uuid4().hex[:8]}",
        control=control,
        terminal_top_k=1,
    )
    return runtime, launch


def _row_and_fanin_successors(completion):
    row_successors = [
        successor
        for successor in completion.successors
        if successor.stage_key.value == STAGE_EVAL_ROW
    ]
    fanin_successors = [
        successor
        for successor in completion.successors
        if successor.stage_key.value == STAGE_EVAL_FANIN
    ]
    return row_successors, fanin_successors


def test_platform_deferral_recovers_when_cached_step_clears_deferred(
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

    first = execute_optim_step_sync(
        runtime,
        input_reference=input_reference,
        stage_index=0,
    )
    assert any(
        successor.stage_key.value == STAGE_EVAL_ROW for successor in first.successors
    )
    pending_state = _load_work_state(runtime, first.output_reference)
    saved_intents = pending_state.pending_deferred_intents
    assert saved_intents

    real_run_step = runtime.harness.run_step

    def cached_run_step(step_request, *, eval_context=None):
        result, result_ref = real_run_step(step_request, eval_context=eval_context)
        runtime.harness._last_deferred_platform_intents = ()  # noqa: SLF001
        return result, result_ref

    original_load = _load_work_state

    def load_with_saved_intents(runtime_arg, ref: str):
        state = original_load(runtime_arg, ref)
        if ref == input_reference and not state.pending_deferred_intents:
            return OptimWorkState(
                work_input=state.work_input,
                step_index=state.step_index,
                step_result_refs=state.step_result_refs,
                terminal=state.terminal,
                pending_step_result_ref=state.pending_step_result_ref,
                deferral_optim_step_stage_index=state.deferral_optim_step_stage_index,
                pending_deferred_intents=saved_intents,
            )
        return state

    monkeypatch.setattr(runtime.harness, "run_step", cached_run_step)
    monkeypatch.setattr(
        "whetstone.platform.step_executor._load_work_state",
        load_with_saved_intents,
    )

    recovered = execute_optim_step_sync(
        runtime,
        input_reference=input_reference,
        stage_index=0,
    )
    row_successors = [
        successor
        for successor in recovered.successors
        if successor.stage_key.value == STAGE_EVAL_ROW
    ]
    fanin_successors = [
        successor
        for successor in recovered.successors
        if successor.stage_key.value == STAGE_EVAL_FANIN
    ]
    assert row_successors
    assert len(fanin_successors) == 1


class _CrashBeforeDeferralEmit(RuntimeError):
    pass


class _CrashBeforeStepResultEvict(RuntimeError):
    pass


class _CrashBeforePersist(RuntimeError):
    pass


class _ReachedRunStep(RuntimeError):
    pass


def test_platform_continue_without_eval_not_treated_as_deferral(
    sqlite_store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, launch = _depth2_copro_launch(sqlite_store)
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
    real_run_step = runtime.harness.run_step
    original_persist = _persist_work_state
    crashed = {"done": False}

    def continue_without_eval(step_request, *, eval_context=None):
        if step_request.step_index == 0:
            result, result_ref = real_run_step(step_request, eval_context=eval_context)
            runtime.harness._last_deferred_platform_intents = ()  # noqa: SLF001
            evict_deferred_intents(
                runtime.store,
                run_id=step_request.run_id,
                step_index=step_request.step_index,
            )
            return result, result_ref
        raise _ReachedRunStep()

    def persist_with_crash(runtime_arg, state):
        if not crashed["done"]:
            crashed["done"] = True
            raise _CrashBeforePersist()
        return original_persist(runtime_arg, state)

    monkeypatch.setattr(runtime.harness, "run_step", continue_without_eval)
    monkeypatch.setattr(
        "whetstone.platform.step_executor._persist_work_state",
        persist_with_crash,
    )

    with pytest.raises(_CrashBeforePersist):
        execute_optim_step_sync(
            runtime,
            input_reference=input_reference,
            stage_index=0,
        )

    assert not load_persisted_deferred_intents(
        runtime.store,
        run_id=launch.run.run_id,
        step_index=0,
    )

    with pytest.raises(_ReachedRunStep):
        execute_optim_step_sync(
            runtime,
            input_reference=input_reference,
            stage_index=0,
        )


def test_platform_deferral_evicts_persisted_intents_after_stage_commit(
    copro_launch,
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

    completion = execute_optim_step_sync(
        runtime,
        input_reference=input_reference,
        stage_index=0,
    )
    row_successors, fanin_successors = _row_and_fanin_successors(completion)
    assert row_successors
    assert len(fanin_successors) == 1

    assert not load_persisted_deferred_intents(
        runtime.store,
        run_id=launch.run.run_id,
        step_index=0,
    )
    pending_state = _load_work_state(runtime, completion.output_reference)
    assert pending_state.pending_deferred_intents
    head_ref = resolve_work_state_head(
        runtime.store,
        run_id=launch.run.run_id,
        work_key=work_input.work_key,
    )
    assert head_ref == completion.output_reference


def test_platform_deferral_recovers_via_head_pointer_after_emit_crash(
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
    real_evict_step_result = _evict_step_result_binding

    def crash_before_step_result_evict(*args, **kwargs):
        raise _CrashBeforeStepResultEvict()

    monkeypatch.setattr(
        "whetstone.platform.step_executor._evict_step_result_binding",
        crash_before_step_result_evict,
    )

    with pytest.raises(_CrashBeforeStepResultEvict):
        execute_optim_step_sync(
            runtime,
            input_reference=input_reference,
            stage_index=0,
        )

    head_ref = resolve_work_state_head(
        runtime.store,
        run_id=launch.run.run_id,
        work_key=work_input.work_key,
    )
    assert head_ref is not None
    head_state = _load_work_state(runtime, head_ref)
    assert head_state.pending_deferred_intents

    monkeypatch.setattr(
        "whetstone.platform.step_executor._evict_step_result_binding",
        real_evict_step_result,
    )

    recovered = execute_optim_step_sync(
        runtime,
        input_reference=input_reference,
        stage_index=0,
    )
    row_successors, fanin_successors = _row_and_fanin_successors(recovered)
    assert row_successors
    assert len(fanin_successors) == 1


def test_platform_deferral_recovers_after_emit_crash_via_persisted_intents(
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

    def crash_before_step_result_evict(*args, **kwargs):
        raise _CrashBeforeStepResultEvict()

    monkeypatch.setattr(
        "whetstone.platform.step_executor._evict_step_result_binding",
        crash_before_step_result_evict,
    )

    with pytest.raises(_CrashBeforeStepResultEvict):
        execute_optim_step_sync(
            runtime,
            input_reference=input_reference,
            stage_index=0,
        )

    assert load_persisted_deferred_intents(
        runtime.store,
        run_id=launch.run.run_id,
        step_index=0,
    )

    monkeypatch.setattr(
        "whetstone.platform.step_executor._evict_step_result_binding",
        _evict_step_result_binding,
    )

    recovered = execute_optim_step_sync(
        runtime,
        input_reference=input_reference,
        stage_index=0,
    )
    row_successors, fanin_successors = _row_and_fanin_successors(recovered)
    assert row_successors
    assert len(fanin_successors) == 1


def test_platform_work_input_redrive_prefers_head_over_binding_reconstruction(
    copro_launch,
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

    completion = execute_optim_step_sync(
        runtime,
        input_reference=input_reference,
        stage_index=0,
    )
    head_state = _load_work_state(runtime, completion.output_reference)
    assert head_state.pending_deferred_intents

    loaded_via_input = _load_work_state(runtime, input_reference)
    assert (
        loaded_via_input.pending_deferred_intents == head_state.pending_deferred_intents
    )

    from whetstone.platform.work_state_head import evict_work_state_head

    evict_work_state_head(
        runtime.store,
        run_id=launch.run.run_id,
        work_key=work_input.work_key,
    )
    loaded_without_head = _load_work_state(runtime, input_reference)
    assert not loaded_without_head.pending_deferred_intents
    assert loaded_without_head.step_index == 0


def test_platform_work_input_redrive_rejects_stale_head(
    copro_launch,
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
        platform_stage_index=5,
    )
    input_reference = persist_work_input(runtime.store, work_input)
    stale_state = OptimWorkState(
        work_input=work_input.model_copy(update={"platform_stage_index": 0}),
        step_index=0,
        terminal=False,
    )
    stale_ref = _persist_work_state(runtime, stale_state)

    loaded = _load_work_state(runtime, input_reference)
    assert loaded.step_index == 0
    assert not loaded.pending_deferred_intents
    assert stale_ref


def test_platform_deferral_recovers_from_persisted_binding_after_crash_window(
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
    real_deferred_successors = _platform_deferred_successors

    def crash_before_emit(*args, **kwargs):
        raise _CrashBeforeDeferralEmit()

    monkeypatch.setattr(
        "whetstone.platform.step_executor._platform_deferred_successors",
        crash_before_emit,
    )

    with pytest.raises(_CrashBeforeDeferralEmit):
        execute_optim_step_sync(
            runtime,
            input_reference=input_reference,
            stage_index=0,
        )

    persisted = load_persisted_deferred_intents(
        runtime.store,
        run_id=launch.run.run_id,
        step_index=0,
    )
    assert persisted

    original_state = _load_work_state(runtime, input_reference)
    assert not original_state.pending_deferred_intents

    real_run_step = runtime.harness.run_step

    def cached_run_step(step_request, *, eval_context=None):
        result, result_ref = real_run_step(step_request, eval_context=eval_context)
        runtime.harness._last_deferred_platform_intents = ()  # noqa: SLF001
        return result, result_ref

    monkeypatch.setattr(
        "whetstone.platform.step_executor._platform_deferred_successors",
        real_deferred_successors,
    )
    monkeypatch.setattr(runtime.harness, "run_step", cached_run_step)

    recovered = execute_optim_step_sync(
        runtime,
        input_reference=input_reference,
        stage_index=0,
    )
    row_successors, fanin_successors = _row_and_fanin_successors(recovered)
    assert row_successors
    assert len(fanin_successors) == 1


def test_platform_deferral_recovers_from_work_state_after_crash_window(
    sqlite_store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, launch = _depth2_copro_launch(sqlite_store)
    control = launch.control
    assert control is not None
    work_input = OptimWorkInput(
        run_id=launch.run.run_id,
        controller_identity_hash=runtime.controller.runtime_hash,
        control_identity_hash=control.identity_hash(),
        dispatch_mode=EvalDispatchMode.PLATFORM,
    )
    input_reference = persist_work_input(runtime.store, work_input)
    step0 = execute_optim_step_sync(
        runtime,
        input_reference=input_reference,
        stage_index=0,
    )
    fanin = _complete_deferral_episode(runtime, step0)
    step1_input_reference = fanin.successors[0].input_reference
    step1_stage_index = fanin.successors[0].stage_index
    pre_step1_state = _load_work_state(runtime, step1_input_reference)
    assert pre_step1_state.step_index == 1

    real_deferred_successors = _platform_deferred_successors

    def crash_before_emit(*args, **kwargs):
        raise _CrashBeforeDeferralEmit()

    monkeypatch.setattr(
        "whetstone.platform.step_executor._platform_deferred_successors",
        crash_before_emit,
    )
    with pytest.raises(_CrashBeforeDeferralEmit):
        execute_optim_step_sync(
            runtime,
            input_reference=step1_input_reference,
            stage_index=step1_stage_index,
        )

    step1_persisted = load_persisted_deferred_intents(
        runtime.store,
        run_id=launch.run.run_id,
        step_index=1,
    )
    assert step1_persisted

    real_run_step = runtime.harness.run_step

    def cached_run_step(step_request, *, eval_context=None):
        result, result_ref = real_run_step(step_request, eval_context=eval_context)
        runtime.harness._last_deferred_platform_intents = ()  # noqa: SLF001
        return result, result_ref

    monkeypatch.setattr(
        "whetstone.platform.step_executor._platform_deferred_successors",
        real_deferred_successors,
    )
    monkeypatch.setattr(runtime.harness, "run_step", cached_run_step)

    recovered = execute_optim_step_sync(
        runtime,
        input_reference=step1_input_reference,
        stage_index=step1_stage_index,
    )
    row_successors, fanin_successors = _row_and_fanin_successors(recovered)
    assert row_successors
    assert len(fanin_successors) == 1
    recovered_state = _load_work_state(runtime, recovered.output_reference)
    assert recovered_state.pending_deferred_intents == step1_persisted


def test_platform_deferral_ignores_stale_prior_step_persisted_intents(
    sqlite_store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, launch = _depth2_copro_launch(sqlite_store)
    control = launch.control
    assert control is not None
    work_input = OptimWorkInput(
        run_id=launch.run.run_id,
        controller_identity_hash=runtime.controller.runtime_hash,
        control_identity_hash=control.identity_hash(),
        dispatch_mode=EvalDispatchMode.PLATFORM,
    )
    input_reference = persist_work_input(runtime.store, work_input)
    step0 = execute_optim_step_sync(
        runtime,
        input_reference=input_reference,
        stage_index=0,
    )
    stale_step0_intents = _load_work_state(
        runtime, step0.output_reference
    ).pending_deferred_intents
    assert stale_step0_intents
    fanin = _complete_deferral_episode(runtime, step0)
    persist_deferred_intents(
        runtime.store,
        run_id=launch.run.run_id,
        step_index=0,
        intents=stale_step0_intents,
    )
    step1_input_reference = fanin.successors[0].input_reference
    step1_stage_index = fanin.successors[0].stage_index

    real_deferred_successors = _platform_deferred_successors

    def crash_before_emit(*args, **kwargs):
        raise _CrashBeforeDeferralEmit()

    monkeypatch.setattr(
        "whetstone.platform.step_executor._platform_deferred_successors",
        crash_before_emit,
    )
    with pytest.raises(_CrashBeforeDeferralEmit):
        execute_optim_step_sync(
            runtime,
            input_reference=step1_input_reference,
            stage_index=step1_stage_index,
        )

    step1_persisted = load_persisted_deferred_intents(
        runtime.store,
        run_id=launch.run.run_id,
        step_index=1,
    )
    assert step1_persisted

    real_run_step = runtime.harness.run_step

    def cached_run_step(step_request, *, eval_context=None):
        result, result_ref = real_run_step(step_request, eval_context=eval_context)
        runtime.harness._last_deferred_platform_intents = ()  # noqa: SLF001
        return result, result_ref

    monkeypatch.setattr(
        "whetstone.platform.step_executor._platform_deferred_successors",
        real_deferred_successors,
    )
    monkeypatch.setattr(runtime.harness, "run_step", cached_run_step)

    recovered = execute_optim_step_sync(
        runtime,
        input_reference=step1_input_reference,
        stage_index=step1_stage_index,
    )
    row_successors, fanin_successors = _row_and_fanin_successors(recovered)
    assert row_successors
    assert len(fanin_successors) == 1
    recovered_state = _load_work_state(runtime, recovered.output_reference)
    assert recovered_state.pending_deferred_intents == step1_persisted
    assert recovered_state.pending_deferred_intents != stale_step0_intents


def _bind_terminal_placeholder(
    runtime, *, run_key: str, work_key: str, token: str
) -> str:
    reference, _ = runtime.store.put(
        "whetstone.test_terminal_placeholder",
        {"token": token},
    )
    runtime.store.bind(
        f"{RUN_MEMBER_TERMINAL_BINDING_PREFIX}{run_key}:{work_key}",
        reference,
    )
    return format_object_reference(reference)


def _run_completion_payload(
    *,
    run_key: str,
    manifest_reference: str,
    membership_digest: str,
    member_count: int,
) -> RunCompletionPayload:
    return RunCompletionPayload(
        campaign_key="campaign-1",
        run_key=run_key,
        pipeline_key=OPTIM_PIPELINE_KEY,
        pipeline_version=OPTIM_PIPELINE_VERSION,
        execution_config_reference="exec-config-ref",
        manifest_reference=manifest_reference,
        membership_digest=membership_digest,
        member_count=member_count,
        released_at=datetime(2026, 8, 20, tzinfo=UTC),
        release_terminal_state_counts=(
            StateCount(state=StageExecutionState.SUCCEEDED, count=member_count),
            StateCount(state=StageExecutionState.FAILED, count=0),
            StateCount(state=StageExecutionState.CANCELLED, count=0),
        ),
    )


def test_run_completion_for_run_persists_single_member_aggregate(toy_runtime) -> None:
    runtime, _control = toy_runtime
    members = (OptimRunMemberEntry(work_key="work-1", run_id="harness-1"),)
    manifest = OptimRunManifest(
        platform_run_key="run-1",
        membership_digest="digest-1",
        members=members,
    )
    manifest_reference = persist_run_manifest(runtime.store, manifest)
    _bind_terminal_placeholder(
        runtime,
        run_key="run-1",
        work_key="work-1",
        token="terminal-1",
    )
    payload = _run_completion_payload(
        run_key="run-1",
        manifest_reference=manifest_reference,
        membership_digest="digest-1",
        member_count=1,
    )
    with patch(
        "whetstone.platform.step_executor.execute_run_completion_sync",
        return_value="whetstone.optim_result:member-1",
    ):
        result_ref = execute_run_completion_for_run_sync(runtime, payload=payload)
    loaded = load_run_result(runtime.store, result_ref)
    assert loaded.platform_run_key == "run-1"
    assert loaded.membership_digest == "digest-1"
    assert len(loaded.member_results) == 1
    assert loaded.member_results[0].work_key == "work-1"
    assert loaded.member_results[0].run_id == "harness-1"
    assert (
        loaded.member_results[0].result_reference == "whetstone.optim_result:member-1"
    )


def test_run_completion_for_run_persists_multi_member_aggregate(toy_runtime) -> None:
    runtime, _control = toy_runtime
    members = (
        OptimRunMemberEntry(work_key="work-a", run_id="harness-a"),
        OptimRunMemberEntry(work_key="work-b", run_id="harness-b"),
    )
    manifest = OptimRunManifest(
        platform_run_key="run-1",
        membership_digest="digest-2",
        members=members,
    )
    manifest_reference = persist_run_manifest(runtime.store, manifest)
    _bind_terminal_placeholder(
        runtime,
        run_key="run-1",
        work_key="work-a",
        token="terminal-a",
    )
    _bind_terminal_placeholder(
        runtime,
        run_key="run-1",
        work_key="work-b",
        token="terminal-b",
    )
    payload = _run_completion_payload(
        run_key="run-1",
        manifest_reference=manifest_reference,
        membership_digest="digest-2",
        member_count=2,
    )
    with patch(
        "whetstone.platform.step_executor.execute_run_completion_sync",
        side_effect=[
            "whetstone.optim_result:member-a",
            "whetstone.optim_result:member-b",
        ],
    ):
        result_ref = execute_run_completion_for_run_sync(runtime, payload=payload)
    loaded = load_run_result(runtime.store, result_ref)
    assert loaded.platform_run_key == "run-1"
    assert loaded.membership_digest == "digest-2"
    assert tuple(item.work_key for item in loaded.member_results) == (
        "work-a",
        "work-b",
    )
    assert tuple(item.run_id for item in loaded.member_results) == (
        "harness-a",
        "harness-b",
    )
    assert tuple(item.result_reference for item in loaded.member_results) == (
        "whetstone.optim_result:member-a",
        "whetstone.optim_result:member-b",
    )
