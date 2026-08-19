from __future__ import annotations

from typing import TYPE_CHECKING

from dr_platform._core.identities import (
    PipelineKey,
    RunCompletionKey,
    StageKey,
)
from dr_platform.execution.handoff import wrap_pipeline_workflows
from dr_platform.pipeline.definitions import (
    PipelineDefinition,
    PipelineIdentity,
    RunCompletionDefinition,
    StageDefinition,
)

from whetstone.platform.contracts import (
    OPTIM_PIPELINE_KEY,
    OPTIM_PIPELINE_VERSION,
    STAGE_EVAL_FANIN,
    STAGE_EVAL_ROW,
    STAGE_OPTIM_STEP,
    STAGE_RUN_COMPLETION,
)
from whetstone.platform.stages.eval_fanin import eval_fanin_args_for, eval_fanin_workflow
from whetstone.platform.stages.eval_row import eval_row_args_for, eval_row_workflow
from whetstone.platform.stages.optim_step import optim_step_args_for, optim_step_workflow
from whetstone.platform.stages.run_completion import (
    run_completion_args_for,
    run_completion_workflow,
)

if TYPE_CHECKING:
    from dr_platform.pipeline.registry import PipelineRegistry
    from whetstone.coordination.runtime_bootstrap import RegisteredRuntime

OPTIM_PIPELINE_IDENTITY = PipelineIdentity(
    key=PipelineKey(OPTIM_PIPELINE_KEY),
    version=OPTIM_PIPELINE_VERSION,
)

EVAL_ROW_QUEUE_CONCURRENCY = 4


def build_optim_pipeline(
    runtime: RegisteredRuntime,
) -> PipelineDefinition:
    return PipelineDefinition(
        key=PipelineKey(OPTIM_PIPELINE_KEY),
        version=OPTIM_PIPELINE_VERSION,
        stages=(
            StageDefinition(
                key=StageKey(STAGE_OPTIM_STEP),
                queue_name=f"{OPTIM_PIPELINE_KEY}.{STAGE_OPTIM_STEP}",
                workflow=optim_step_workflow,
                args_for=lambda payload: optim_step_args_for(runtime, payload),
            ),
            StageDefinition(
                key=StageKey(STAGE_EVAL_ROW),
                queue_name=f"{OPTIM_PIPELINE_KEY}.{STAGE_EVAL_ROW}",
                workflow=eval_row_workflow,
                args_for=lambda payload: eval_row_args_for(runtime, payload),
            ),
            StageDefinition(
                key=StageKey(STAGE_EVAL_FANIN),
                queue_name=f"{OPTIM_PIPELINE_KEY}.{STAGE_EVAL_FANIN}",
                workflow=eval_fanin_workflow,
                args_for=lambda payload: eval_fanin_args_for(runtime, payload),
            ),
        ),
        run_completion=RunCompletionDefinition(
            key=RunCompletionKey(STAGE_RUN_COMPLETION),
            queue_name=f"{OPTIM_PIPELINE_KEY}.{STAGE_RUN_COMPLETION}",
            workflow=run_completion_workflow,
            args_for=lambda payload: run_completion_args_for(runtime, payload),
        ),
    )


def register_optim_pipeline(
    registry: PipelineRegistry,
    runtime: RegisteredRuntime,
    *,
    max_recovery_attempts: int,
) -> PipelineDefinition:
    pipeline = build_optim_pipeline(runtime)
    wrapped = wrap_pipeline_workflows(
        pipeline,
        max_recovery_attempts=max_recovery_attempts,
    )
    registry.register(wrapped)
    return wrapped


__all__ = [
    "EVAL_ROW_QUEUE_CONCURRENCY",
    "OPTIM_PIPELINE_IDENTITY",
    "build_optim_pipeline",
    "register_optim_pipeline",
]
