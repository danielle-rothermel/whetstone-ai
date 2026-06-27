from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class FieldSignature(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    name: str
    type: type
    role: Any
    description: str | None = None

