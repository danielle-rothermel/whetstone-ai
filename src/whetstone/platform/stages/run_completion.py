from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dr_platform.admission.runner import AdmissionPayload
    from whetstone.coordination.runtime_bootstrap import RegisteredRuntime


async def run_completion_workflow(
    runtime: RegisteredRuntime,
    input_reference: str,
) -> str:
    from whetstone.platform.step_executor import execute_run_completion_sync

    return await asyncio.to_thread(
        execute_run_completion_sync,
        runtime,
        input_reference=input_reference,
    )


def run_completion_args_for(
    runtime: RegisteredRuntime,
    payload: AdmissionPayload,
) -> tuple[RegisteredRuntime, str]:
    return (runtime, payload.input_reference)


__all__ = ["run_completion_args_for", "run_completion_workflow"]
