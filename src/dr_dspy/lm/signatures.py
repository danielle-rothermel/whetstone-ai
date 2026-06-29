"""Legacy DSPy signature config models for v0 workflow scripts."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, StrictStr


class FieldSignature(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    name: str
    type: type
    role: Any
    description: str | None = None


class DspySignatureConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    name: StrictStr
    fields: tuple[FieldSignature, ...]
    instructions: StrictStr
