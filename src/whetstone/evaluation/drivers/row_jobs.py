"""Row execution boundary between graph interpretation and process workers.

``graph_execution`` interprets one dr-graph run into node outcomes and metadata.
Row drivers live outside whetstone: a caller-supplied ``RowJobFactory`` builds
``ProcessJob`` payloads for subprocess workers, and a matching decoder turns
worker JSON back into :class:`~whetstone.evaluation.drivers.row_common.GenerationRowOutput`.
Aggregated evaluation results use
:class:`~whetstone.evaluation.drivers.eval_result.InternalEvalResult`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

from whetstone.evaluation.drivers.eval_result import InternalEvalResult
from whetstone.evaluation.drivers.row_common import GenerationRowOutput
from whetstone.execution.fanout import ProcessJob

TRequest = TypeVar("TRequest", bound=BaseModel)


@runtime_checkable
class RowJobFactory(Protocol[TRequest]):
    """Build one subprocess job from a strict row request model."""

    def __call__(self, request: TRequest) -> ProcessJob: ...


@runtime_checkable
class RowJobDecoder(Protocol[TRequest]):
    """Decode one worker JSON payload into a generation row output."""

    def __call__(
        self,
        payload: Mapping[str, object],
        *,
        request: TRequest,
    ) -> GenerationRowOutput: ...


RowBatchScorer = Callable[
    [tuple[GenerationRowOutput, ...]],
    InternalEvalResult,
]


def row_job_from_entrypoint(
    entrypoint: str,
) -> RowJobFactory[BaseModel]:
    """Wrap a runtime-config entrypoint string into a generic row job factory."""

    def build(request: BaseModel) -> ProcessJob:
        return ProcessJob(
            entrypoint=entrypoint,
            payload=request.model_dump(mode="json"),
        )

    return build


__all__ = [
    "InternalEvalResult",
    "GenerationRowOutput",
    "ProcessJob",
    "RowBatchScorer",
    "RowJobDecoder",
    "RowJobFactory",
    "row_job_from_entrypoint",
]
