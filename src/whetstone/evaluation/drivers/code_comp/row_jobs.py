from __future__ import annotations

from whetstone.envs.code_comp.modes.encdec import (
    Ed1TaskModelKind,
    EncDecTaskModelConfig,
)
from whetstone.evaluation.drivers.code_comp.encdec import (
    Ed1RowJobFactory,
    Ed1RowRequest,
)
from whetstone.execution.fanout import ProcessJob

_DUMMY_ROW_ENTRYPOINT = (
    "whetstone.evaluation.drivers.code_comp.workers:drive_dummy_ed1_generation"
)
_PROVIDER_ROW_ENTRYPOINT = (
    "whetstone.evaluation.drivers.code_comp.workers:"
    "drive_provider_ed1_generation"
)


def dummy_ed1_row_job(request: Ed1RowRequest) -> ProcessJob:
    """Build a process job with deterministic encoder/decoder generations."""

    return ProcessJob(
        entrypoint=_DUMMY_ROW_ENTRYPOINT,
        payload=request.model_dump(mode="json"),
    )


def provider_ed1_row_job(request: Ed1RowRequest) -> ProcessJob:
    """Build a process job with real dr-providers encoder/decoder calls."""

    return ProcessJob(
        entrypoint=_PROVIDER_ROW_ENTRYPOINT,
        payload=request.model_dump(mode="json"),
    )


def ed1_task_model_row_job(config: EncDecTaskModelConfig) -> Ed1RowJobFactory:
    """Select the row-job boundary for one validated task-model mode."""

    if config.kind is Ed1TaskModelKind.DUMMY:
        return dummy_ed1_row_job
    return provider_ed1_row_job


__all__ = [
    "dummy_ed1_row_job",
    "ed1_task_model_row_job",
    "provider_ed1_row_job",
]
