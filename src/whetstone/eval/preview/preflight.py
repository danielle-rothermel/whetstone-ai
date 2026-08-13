from __future__ import annotations

from pydantic import BaseModel, ConfigDict, StrictStr

__all__ = ["PreviewMetadata", "ScoringPreflight"]


class ScoringPreflight(BaseModel):
    """Ground-truth or runtime check completed before candidate evaluation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: StrictStr
    passed: bool
    infrastructure_unknown: bool
    outcome: StrictStr


class PreviewMetadata(BaseModel):
    """Caller-specific preview fields persisted alongside generic results."""

    model_config = ConfigDict(extra="allow", frozen=True)
