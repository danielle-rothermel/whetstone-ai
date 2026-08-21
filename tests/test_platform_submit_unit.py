from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from dr_platform._core.identities import StageKey
from dr_platform.admission.runner import AdmissionPayload
from dr_platform.pipeline.definitions import PipelineDefinition
from dr_platform.submission.stream import compute_run_membership_digest

from whetstone.platform.contracts import (
    OPTIM_PIPELINE_KEY,
    OPTIM_PIPELINE_VERSION,
    STAGE_EVAL_FANIN,
    STAGE_EVAL_ROW,
    STAGE_OPTIM_STEP,
    STAGE_RUN_COMPLETION,
    OptimRunManifest,
    OptimWorkInput,
    load_run_manifest,
    load_work_input,
    persist_work_input,
)
from whetstone.coordination.runtime_bootstrap import prepare_copro_run
from whetstone.platform.pipeline import build_optim_pipeline, register_optim_pipeline
from whetstone.platform.stages.eval_fanin import eval_fanin_args_for
from whetstone.platform.stages.eval_row import eval_row_args_for
from whetstone.platform.stages.optim_step import optim_step_args_for
from whetstone.platform.submit import (
    OptimRunMemberSpec,
    build_work_input,
    submit_optim_run,
)


def test_pipeline_stage_keys(toy_runtime) -> None:
    runtime, _control = toy_runtime
    pipeline = build_optim_pipeline(runtime)
    assert isinstance(pipeline, PipelineDefinition)
    assert pipeline.key.value == OPTIM_PIPELINE_KEY
    assert pipeline.version == OPTIM_PIPELINE_VERSION
    stage_keys = tuple(stage.key.value for stage in pipeline.stages)
    assert stage_keys == (STAGE_OPTIM_STEP, STAGE_EVAL_ROW, STAGE_EVAL_FANIN)
    assert pipeline.run_completion is not None
    assert pipeline.run_completion.key.value == STAGE_RUN_COMPLETION


def test_register_optim_pipeline_requires_ledger_engine(toy_runtime) -> None:
    runtime, _control = toy_runtime
    try:
        register_optim_pipeline(MagicMock(), runtime, max_recovery_attempts=1)
    except ValueError as error:
        assert "ledger engine" in str(error)
    else:
        raise AssertionError("expected missing ledger engine")


def test_work_input_ref_roundtrip(sqlite_store) -> None:
    work_input = OptimWorkInput(
        run_id="run-1",
        controller_identity_hash="a" * 64,
        control_identity_hash="b" * 64,
    )
    reference = persist_work_input(sqlite_store, work_input)
    loaded = load_work_input(sqlite_store, reference)
    assert loaded == work_input


def test_submit_optim_run_builds_member(copro_launch) -> None:
    runtime, launch = copro_launch
    registry = MagicMock()
    engine = MagicMock()
    submit_mock = MagicMock(return_value=MagicMock())
    import whetstone.platform.submit as submit_module

    original = submit_module.submit
    submit_module.submit = submit_mock
    try:
        submit_optim_run(
            runtime=runtime,
            registry=registry,
            engine=engine,
            campaign_key="campaign-1",
            run_key="run-1",
            members=(OptimRunMemberSpec(work_key="work-1", launch=launch),),
            controller_identity_hash=runtime.controller.runtime_hash,
            execution_config_reference="exec-config-ref",
        )
    finally:
        submit_module.submit = original

    submit_mock.assert_called_once()
    kwargs = submit_mock.call_args.kwargs
    members = list(kwargs["members"])
    assert len(members) == 1
    member = members[0]
    assert member.ordinal == 0
    assert member.work.work_key.value == "work-1"
    assert member.work.labels["run_id"] == launch.run.run_id
    loaded = load_work_input(runtime.store, member.work.input_reference)
    assert loaded.run_id == launch.run.run_id
    assert loaded.control_identity_hash == launch.control.identity_hash()
    assert loaded.platform_run_key == "run-1"
    assert loaded.work_key == "work-1"
    declaration = kwargs["declaration"]
    assert declaration.expected_member_count == 1
    assert declaration.manifest_reference is not None
    assert declaration.membership_digest == compute_run_membership_digest(
        members,
        expected_member_count=1,
    )


def test_submit_optim_run_builds_two_members(toy_runtime) -> None:
    runtime, control = toy_runtime
    launch_a = prepare_copro_run(
        runtime,
        run_id="harness-run-a",
        control=control,
        terminal_top_k=1,
    )
    launch_b = prepare_copro_run(
        runtime,
        run_id="harness-run-b",
        control=control,
        terminal_top_k=1,
    )
    registry = MagicMock()
    engine = MagicMock()
    submit_mock = MagicMock(return_value=MagicMock())
    import whetstone.platform.submit as submit_module

    original = submit_module.submit
    submit_module.submit = submit_mock
    try:
        submit_optim_run(
            runtime=runtime,
            registry=registry,
            engine=engine,
            campaign_key="campaign-1",
            run_key="run-1",
            members=(
                OptimRunMemberSpec(work_key="work-a", launch=launch_a),
                OptimRunMemberSpec(
                    work_key="work-b",
                    launch=launch_b,
                    priority=3,
                ),
            ),
            controller_identity_hash=runtime.controller.runtime_hash,
            execution_config_reference="exec-config-ref",
        )
    finally:
        submit_module.submit = original

    submit_mock.assert_called_once()
    kwargs = submit_mock.call_args.kwargs
    members = list(kwargs["members"])
    assert [member.ordinal for member in members] == [0, 1]
    assert members[0].work.work_key.value == "work-a"
    assert members[1].work.work_key.value == "work-b"
    assert members[0].work.labels["run_id"] == launch_a.run.run_id
    assert members[1].work.labels["run_id"] == launch_b.run.run_id
    assert members[1].work.priority == 3
    loaded_a = load_work_input(runtime.store, members[0].work.input_reference)
    loaded_b = load_work_input(runtime.store, members[1].work.input_reference)
    assert loaded_a.run_id == launch_a.run.run_id
    assert loaded_b.run_id == launch_b.run.run_id
    assert loaded_a.work_key == "work-a"
    assert loaded_b.work_key == "work-b"
    declaration = kwargs["declaration"]
    assert declaration.expected_member_count == 2
    assert declaration.membership_digest == compute_run_membership_digest(
        members,
        expected_member_count=2,
    )
    manifest = load_run_manifest(runtime.store, declaration.manifest_reference)
    assert isinstance(manifest, OptimRunManifest)
    assert tuple(member.work_key for member in manifest.members) == (
        "work-a",
        "work-b",
    )
    assert tuple(member.run_id for member in manifest.members) == (
        launch_a.run.run_id,
        launch_b.run.run_id,
    )


def test_submit_optim_run_rejects_empty_members(copro_launch) -> None:
    runtime, _launch = copro_launch
    with pytest.raises(ValueError, match="members must be non-empty"):
        submit_optim_run(
            runtime=runtime,
            registry=MagicMock(),
            engine=MagicMock(),
            campaign_key="campaign-1",
            run_key="run-1",
            members=(),
            controller_identity_hash=runtime.controller.runtime_hash,
            execution_config_reference="exec-config-ref",
        )


def test_stage_args_for_with_admission_payload(toy_runtime) -> None:
    runtime, _control = toy_runtime
    payload = AdmissionPayload(
        campaign_key="campaign-1",
        work_key="work-1",
        origin_run_key="run-1",
        input_reference="input-ref",
        labels={"run_id": "run-1"},
        pipeline_key=OPTIM_PIPELINE_KEY,
        pipeline_version=OPTIM_PIPELINE_VERSION,
        stage_key=StageKey(STAGE_OPTIM_STEP),
        work_item_id=1,
        stage_index=0,
        attempt_number=1,
    )
    assert optim_step_args_for(runtime, payload) == (runtime, "input-ref", 0)
    payload = payload.model_copy(
        update={"stage_key": StageKey(STAGE_EVAL_ROW), "stage_index": 1}
    )
    assert eval_row_args_for(runtime, payload) == (runtime, "input-ref", 1)
    payload = payload.model_copy(
        update={"stage_key": StageKey(STAGE_EVAL_FANIN), "stage_index": 2}
    )
    assert eval_fanin_args_for(runtime, payload) == (runtime, "input-ref", 2, 1)


def test_build_work_input_uses_launch_control(copro_launch) -> None:
    runtime, launch = copro_launch
    work_input = build_work_input(
        launch=launch,
        controller_identity_hash=runtime.controller.runtime_hash,
        platform_run_key="run-1",
        work_key="work-1",
    )
    assert work_input.run_id == launch.run.run_id
    assert work_input.control_identity_hash == launch.control.identity_hash()
    assert work_input.platform_run_key == "run-1"
    assert work_input.work_key == "work-1"
