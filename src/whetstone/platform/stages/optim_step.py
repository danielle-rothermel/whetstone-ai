from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dr_platform.admission.runner import AdmissionPayload
    from whetstone.coordination.runtime_bootstrap import RegisteredRuntime


async def optim_step_workflow(
    runtime: RegisteredRuntime,
    input_reference: str,
) -> str:
    from whetstone.platform.step_executor import execute_optim_step_sync

    return await asyncio.to_thread(
        execute_optim_step_sync,
        runtime,
        input_reference=input_reference,
    )


def optim_step_args_for(
    runtime: RegisteredRuntime,
    payload: AdmissionPayload,
) -> tuple[RegisteredRuntime, str]:
    return (runtime, payload.input_reference)


__all__ = ["optim_step_args_for", "optim_step_workflow"]
