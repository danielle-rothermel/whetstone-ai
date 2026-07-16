"""Shared fixtures for whetstone's serialization boundary tests."""

from __future__ import annotations

import pydantic


class BadModel(pydantic.BaseModel):
    x: object
