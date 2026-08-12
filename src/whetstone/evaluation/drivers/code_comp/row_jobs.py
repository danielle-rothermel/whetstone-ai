from __future__ import annotations

from whetstone.envs.code_comp.modes.encdec import (
    EncDecTaskModelConfig,
    EncDecTaskModelKind,
)
from whetstone.evaluation.drivers.code_comp.encdec import (
    EncDecRowJobFactory,
    EncDecRowRequest,
)
from whetstone.execution.fanout import ProcessJob

_DUMMY_ROW_ENTRYPOINT = (
    "whetstone.evaluation.drivers.code_comp.workers:"
    "drive_dummy_encdec_generation"
)
_PROVIDER_ROW_ENTRYPOINT = (
    "whetstone.evaluation.drivers.code_comp.workers:drive_provider_encdec_call"
)


def dummy_encdec_row_job(request: EncDecRowRequest) -> ProcessJob:
    """Build a process job with deterministic encoder/decoder generations."""

    return ProcessJob(
        entrypoint=_DUMMY_ROW_ENTRYPOINT,
        payload=request.model_dump(mode="json"),
    )


def provider_encdec_row_job(request: EncDecRowRequest) -> ProcessJob:
    """Build a process job with real dr-providers encoder/decoder calls."""

    return ProcessJob(
        entrypoint=_PROVIDER_ROW_ENTRYPOINT,
        payload=request.model_dump(mode="json"),
    )


def encdec_task_model_row_job(
    config: EncDecTaskModelConfig,
) -> EncDecRowJobFactory:
    """Select the row-job boundary for one validated task-model mode."""

    if config.kind is EncDecTaskModelKind.DUMMY:
        return dummy_encdec_row_job
    return provider_encdec_row_job


__all__ = [
    "dummy_encdec_row_job",
    "encdec_task_model_row_job",
    "provider_encdec_row_job",
]
