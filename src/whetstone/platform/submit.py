from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from dr_platform._core.identities import CampaignKey, RunKey, WorkKey
from dr_platform.submission.stream import (
    RunMemberInput,
    RunRegistrationDeclaration,
    SubmissionReceipt,
    WorkInput,
    compute_run_membership_digest,
    submit,
)

from whetstone.coordination.harness_run_controller import OptimRunLaunch
from whetstone.coordination.runtime_bootstrap import RegisteredRuntime
from whetstone.platform.contracts import (
    OptimRunManifest,
    OptimRunMemberEntry,
    OptimWorkInput,
    persist_run_manifest,
    persist_work_input,
)
from whetstone.platform.pipeline import OPTIM_PIPELINE_IDENTITY

if TYPE_CHECKING:
    from dr_platform.pipeline.registry import PipelineRegistry
    from sqlalchemy.engine import Engine

    from whetstone.coordination.eval_service import EvalDispatchMode


@dataclass(frozen=True, slots=True)
class OptimRunMemberSpec:
    work_key: str
    launch: OptimRunLaunch
    priority: int = 0


def build_work_input(
    *,
    launch: OptimRunLaunch,
    controller_identity_hash: str,
    platform_run_key: str = "",
    work_key: str = "",
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
        platform_run_key=platform_run_key,
        work_key=work_key,
    )


def submit_optim_run(
    *,
    runtime: RegisteredRuntime,
    registry: PipelineRegistry,
    engine: Engine,
    campaign_key: str,
    run_key: str,
    members: tuple[OptimRunMemberSpec, ...],
    controller_identity_hash: str,
    execution_config_reference: str,
    dispatch_mode: EvalDispatchMode | None = None,
) -> SubmissionReceipt:
    if controller_identity_hash != runtime.controller.runtime_hash:
        raise ValueError("controller identity hash does not match bound runtime")
    if not members:
        raise ValueError("members must be non-empty")
    run_members: list[RunMemberInput] = []
    manifest_members: list[OptimRunMemberEntry] = []
    for ordinal, spec in enumerate(members):
        runtime.controller.bind_launch(spec.launch)
        work_input = build_work_input(
            launch=spec.launch,
            controller_identity_hash=controller_identity_hash,
            platform_run_key=run_key,
            work_key=spec.work_key,
            dispatch_mode=dispatch_mode,
        )
        input_reference = persist_work_input(runtime.store, work_input)
        run_members.append(
            RunMemberInput(
                ordinal=ordinal,
                work=WorkInput(
                    work_key=WorkKey(spec.work_key),
                    input_reference=input_reference,
                    labels={"run_id": spec.launch.run.run_id},
                    priority=spec.priority,
                ),
            )
        )
        manifest_members.append(
            OptimRunMemberEntry(
                work_key=spec.work_key,
                run_id=spec.launch.run.run_id,
            )
        )
    membership_digest = compute_run_membership_digest(
        run_members,
        expected_member_count=len(run_members),
    )
    manifest = OptimRunManifest(
        platform_run_key=run_key,
        membership_digest=membership_digest,
        members=tuple(manifest_members),
    )
    manifest_reference = persist_run_manifest(runtime.store, manifest)
    return submit(
        campaign_key=CampaignKey(campaign_key),
        run_key=RunKey(run_key),
        pipeline=OPTIM_PIPELINE_IDENTITY,
        execution_config_reference=execution_config_reference,
        declaration=RunRegistrationDeclaration(
            expected_member_count=len(run_members),
            manifest_reference=manifest_reference,
            membership_digest=membership_digest,
        ),
        members=run_members,
        registry=registry,
        engine=engine,
    )


__all__ = [
    "OptimRunMemberSpec",
    "build_work_input",
    "submit_optim_run",
]
