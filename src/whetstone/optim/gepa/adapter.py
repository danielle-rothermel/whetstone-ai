from __future__ import annotations

from pydantic import BaseModel, ConfigDict, StrictStr, model_validator

from whetstone.core.identity import TypedRef, require_full_hash
from whetstone.optim.gepa.control import GepaControl
from whetstone.optim.gepa.engine import GepaDetailedResult


class GepaTerminalResult(BaseModel):
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


__all__ = [
    "GepaTerminalResult",
    "project_gepa_terminal",
]
