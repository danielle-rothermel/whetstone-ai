from __future__ import annotations

from enum import UNIQUE, StrEnum, verify

from dr_providers import ProviderCallConfig
from pydantic import BaseModel, ConfigDict

from whetstone.evaluation.drivers.ed1 import Ed1RowJobFactory, Ed1RowRequest
from whetstone.execution.fanout import ProcessJob
from whetstone.provider.policy import ProviderExecutionPolicy

_DUMMY_ROW_ENTRYPOINT = (
    "whetstone.optimization.copro.ed1_scoring_preview_worker:"
    "drive_dummy_ed1_generation"
)
_PROVIDER_ROW_ENTRYPOINT = (
    "whetstone.optimization.copro.ed1_scoring_preview_worker:"
    "drive_provider_ed1_generation"
)


@verify(UNIQUE)
class Ed1TaskModelKind(StrEnum):
    """Execution route for ED1 encoder and decoder generations."""

    DUMMY = "dummy"
    PROVIDER = "provider"


class Ed1TaskModelConfig(BaseModel):
    """Exact task-model mode, provider request, and execution policy."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Ed1TaskModelKind
    provider_call_config: ProviderCallConfig
    execution_policy: ProviderExecutionPolicy

    @property
    def model(self) -> str:
        """The exact provider request's model slug for display."""
        return self.provider_call_config.definition.route.model


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


def ed1_task_model_row_job(config: Ed1TaskModelConfig) -> Ed1RowJobFactory:
    """Select the row-job boundary for one validated task-model mode."""

    if config.kind is Ed1TaskModelKind.DUMMY:
        return dummy_ed1_row_job
    return provider_ed1_row_job


__all__ = [
    "Ed1TaskModelConfig",
    "Ed1TaskModelKind",
    "dummy_ed1_row_job",
    "ed1_task_model_row_job",
    "provider_ed1_row_job",
]
