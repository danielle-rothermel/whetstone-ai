from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dr_platform.admission.runner import AdmissionPayload
    from whetstone.coordination.runtime_bootstrap import RegisteredRuntime


async def optim_step_workflow(
    runtime: RegisteredRuntime,
    input_reference: str,
    stage_index: int,
):
    from whetstone.platform.step_executor import execute_optim_step_sync

    return await asyncio.to_thread(
        execute_optim_step_sync,
        runtime,
        input_reference=input_reference,
        stage_index=stage_index,
    )


def optim_step_args_for(
    runtime: RegisteredRuntime,
    payload: AdmissionPayload,
) -> tuple[RegisteredRuntime, str, int]:
    return (runtime, payload.input_reference, payload.stage_index)


__all__ = ["optim_step_args_for", "optim_step_workflow"]
