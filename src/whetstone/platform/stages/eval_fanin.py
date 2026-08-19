from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dr_platform.admission.runner import AdmissionPayload
    from whetstone.coordination.runtime_bootstrap import RegisteredRuntime


async def eval_fanin_workflow(
    runtime: RegisteredRuntime,
    input_reference: str,
    stage_index: int,
):
    from whetstone.platform.eval_fanin import execute_eval_fanin_sync

    return await asyncio.to_thread(
        execute_eval_fanin_sync,
        runtime,
        input_reference=input_reference,
        stage_index=stage_index,
    )


def eval_fanin_args_for(
    runtime: RegisteredRuntime,
    payload: AdmissionPayload,
) -> tuple[RegisteredRuntime, str, int]:
    return (runtime, payload.input_reference, payload.stage_index)


__all__ = ["eval_fanin_args_for", "eval_fanin_workflow"]
