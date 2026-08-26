from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dr_platform.completion.execution import RunCompletionPayload
    from whetstone.coordination.runtime_bootstrap import RegisteredRuntime


async def run_completion_workflow(
    runtime: RegisteredRuntime,
    payload: RunCompletionPayload,
) -> str:
    from whetstone.platform.step_executor import execute_run_completion_for_run_sync

    return await asyncio.to_thread(
        execute_run_completion_for_run_sync,
        runtime,
        payload=payload,
    )


def run_completion_args_for(
    runtime: RegisteredRuntime,
    payload: RunCompletionPayload,
) -> tuple[RegisteredRuntime, RunCompletionPayload]:
    return (runtime, payload)


__all__ = ["run_completion_args_for", "run_completion_workflow"]
