from __future__ import annotations

from typing import TYPE_CHECKING

from dr_platform._core.identities import CampaignKey, RunKey, WorkKey
from dr_platform.submission.stream import (
    RunMemberInput,
    RunRegistrationDeclaration,
    SubmissionReceipt,
    WorkInput,
    submit,
)

from whetstone.coordination.harness_run_controller import OptimRunLaunch
from whetstone.coordination.runtime_bootstrap import RegisteredRuntime
from whetstone.platform.contracts import OptimWorkInput, persist_work_input
from whetstone.platform.pipeline import OPTIM_PIPELINE_IDENTITY

if TYPE_CHECKING:
    from dr_platform.pipeline.registry import PipelineRegistry
    from sqlalchemy.engine import Engine

    from whetstone.coordination.eval_service import EvalDispatchMode


def build_work_input(
    *,
    launch: OptimRunLaunch,
    controller_identity_hash: str,
    dispatch_mode: EvalDispatchMode | None = None,
) -> OptimWorkInput:
    from whetstone.coordination.eval_service import EvalDispatchMode as Mode

    control = launch.control
    if control is not None:
        control_identity_hash = control.identity_hash()
    else:
        control_identity_hash = launch.run.optimizer_config.record_hash
    mode = dispatch_mode or Mode.INLINE
    return OptimWorkInput(
        run_id=launch.run.run_id,
        controller_identity_hash=controller_identity_hash,
        control_identity_hash=control_identity_hash,
        dispatch_mode=mode,
    )


def submit_optim_run(
    *,
    runtime: RegisteredRuntime,
    registry: PipelineRegistry,
    engine: Engine,
    campaign_key: str,
    run_key: str,
    work_key: str,
    launch: OptimRunLaunch,
    controller_identity_hash: str,
    execution_config_reference: str,
    dispatch_mode: EvalDispatchMode | None = None,
) -> SubmissionReceipt:
    runtime.controller.bind_launch(launch)
    work_input = build_work_input(
        launch=launch,
        controller_identity_hash=controller_identity_hash,
        dispatch_mode=dispatch_mode,
    )
    input_reference = persist_work_input(runtime.store, work_input)
    member = RunMemberInput(
        ordinal=0,
        work=WorkInput(
            work_key=WorkKey(work_key),
            input_reference=input_reference,
            labels={"run_id": launch.run.run_id},
        ),
    )
    return submit(
        campaign_key=CampaignKey(campaign_key),
        run_key=RunKey(run_key),
        pipeline=OPTIM_PIPELINE_IDENTITY,
        execution_config_reference=execution_config_reference,
        declaration=RunRegistrationDeclaration(expected_member_count=1),
        members=[member],
        registry=registry,
        engine=engine,
    )


__all__ = [
    "build_work_input",
    "submit_optim_run",
]
