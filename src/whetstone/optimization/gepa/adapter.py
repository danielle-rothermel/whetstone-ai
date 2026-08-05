"""Public façade for canonical upstream GEPA optimization.

The optimizer in this module does not implement selection, sampling,
acceptance, Pareto tracking, merging, budgets, or final ranking. Those
decisions remain inside the frozen public ``gepa.optimize`` engine hosted by
``run_gepa_engine``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, ConfigDict, StrictStr, model_validator

from whetstone.core.identity import TypedRef, require_full_hash
from whetstone.optimization.gepa.control import GepaControl
from whetstone.optimization.gepa.engine import (
    GepaDetailedResult,
    GepaEngineAdapter,
    run_gepa_engine,
)

GEPA_ADAPTER_KEY = "gepa"


class GepaAdapterFactory(Protocol):
    """Construct a fresh, identity-bound adapter for one engine replay."""

    def create(self, *, control: GepaControl) -> GepaEngineAdapter:
        """Return an ordinal-zero adapter bound to control and source."""

        ...

    def persist_result(
        self,
        *,
        control: GepaControl,
        adapter: GepaEngineAdapter,
        detailed_result: GepaDetailedResult,
    ) -> TypedRef:
        """Idempotently persist detail plus its effect transcript."""

        ...


class GepaPersistedRun(BaseModel):
    """Full engine result paired with its durable canonical artifact."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    detailed_result: GepaDetailedResult
    artifact_ref: TypedRef


class GepaTerminalResult(BaseModel):
    """DSPy-compatible terminal exposure after durable detail persistence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    best_candidate: dict[StrictStr, StrictStr]
    control_identity_hash: StrictStr
    artifact_ref: TypedRef
    detailed_results: GepaDetailedResult | None = None

    @model_validator(mode="after")
    def _validate(self) -> GepaTerminalResult:
        require_full_hash(
            self.control_identity_hash,
            field="control_identity_hash",
        )
        if (
            self.detailed_results is not None
            and self.detailed_results.control_identity_hash
            != self.control_identity_hash
        ):
            raise ValueError(
                "terminal and detailed GEPA results bind different controls"
            )
        return self


def project_gepa_terminal(
    *,
    control: GepaControl,
    detailed_result: GepaDetailedResult,
    artifact_ref: TypedRef,
) -> GepaTerminalResult:
    """Expose stats only when the frozen DSPy control requests them."""

    if detailed_result.control_identity_hash != control.identity_hash():
        raise ValueError("GEPA detailed result conflicts with GepaControl")
    return GepaTerminalResult(
        best_candidate=dict(
            detailed_result.candidates[detailed_result.best_idx]
        ),
        control_identity_hash=control.identity_hash(),
        artifact_ref=artifact_ref,
        detailed_results=detailed_result if control.track_stats else None,
    )


@dataclass(frozen=True, slots=True)
class GepaOptimizer:
    """Bind canonical controls to a durable upstream-adapter factory."""

    control: GepaControl
    adapter_factory: GepaAdapterFactory

    def run_detailed[DataInst](
        self,
        *,
        seed_candidate: Mapping[str, str],
        trainset: Sequence[DataInst],
        valset: Sequence[DataInst] | None = None,
    ) -> GepaPersistedRun:
        """Return full detail for persistence before terminal projection."""

        adapter = self.adapter_factory.create(control=self.control)
        detailed_result = run_gepa_engine(
            control=self.control,
            seed_candidate=seed_candidate,
            trainset=trainset,
            valset=valset,
            adapter=adapter,
        )
        artifact_ref = self.adapter_factory.persist_result(
            control=self.control,
            adapter=adapter,
            detailed_result=detailed_result,
        )
        return GepaPersistedRun(
            detailed_result=detailed_result,
            artifact_ref=artifact_ref,
        )

    def run[DataInst](
        self,
        *,
        seed_candidate: Mapping[str, str],
        trainset: Sequence[DataInst],
        valset: Sequence[DataInst] | None = None,
    ) -> GepaTerminalResult:
        """Run and apply DSPy's ``track_stats`` exposure rule."""

        persisted = self.run_detailed(
            seed_candidate=seed_candidate,
            trainset=trainset,
            valset=valset,
        )
        return project_gepa_terminal(
            control=self.control,
            detailed_result=persisted.detailed_result,
            artifact_ref=persisted.artifact_ref,
        )


__all__ = [
    "GEPA_ADAPTER_KEY",
    "GepaAdapterFactory",
    "GepaOptimizer",
    "GepaPersistedRun",
    "GepaTerminalResult",
    "project_gepa_terminal",
]
