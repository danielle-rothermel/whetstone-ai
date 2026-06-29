from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    reasoning: dict[str, Any] = Field(default_factory=dict)


class LmEventBuffer:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def put_event(self, event_type: str, **kwargs: Any) -> None:
        self.events.append({"event_type": event_type, **kwargs})

    def latest_response_metadata(self) -> dict[str, Any]:
        for event in reversed(self.events):
            if event["event_type"] == "lm.response":
                payload = event.get("payload")
                if isinstance(payload, dict):
                    response = payload.get("response")
                    if isinstance(response, dict):
                        return response
        return {}

    def has_latest_response(self) -> bool:
        return any(
            event["event_type"] == "lm.response" for event in self.events
        )

    def latest_response_text(self) -> str | None:
        for event in reversed(self.events):
            if event["event_type"] != "lm.response":
                continue
            payload = event.get("payload")
            if not isinstance(payload, dict):
                continue
            text = response_text(payload.get("response"))
            if text:
                return text
        return None


def stable_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def response_text(response: Any) -> str | None:
    if isinstance(response, str):
        return response
    if not isinstance(response, Mapping):
        return None
    choices = response.get("choices")
    if not isinstance(choices, Sequence) or isinstance(choices, str | bytes):
        return None
    for choice in choices:
        if not isinstance(choice, Mapping):
            continue
        message = choice.get("message")
        if isinstance(message, Mapping):
            content_text = content_to_text(message.get("content"))
            if content_text:
                return content_text
        text = choice.get("text")
        if isinstance(text, str) and text:
            return text
    return None


def content_to_text(content: Any) -> str | None:
    if isinstance(content, str):
        return content
    if not isinstance(content, Sequence) or isinstance(content, str | bytes):
        return None
    parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, Mapping):
            text = part.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts) or None


def usage_metadata_from_response(
    response_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    usage = response_metadata.get("usage")
    return dict(usage) if isinstance(usage, Mapping) else {}


def provider_cost_from_response(
    response_metadata: Mapping[str, Any],
) -> float | None:
    for key in ("cost", "total_cost"):
        value = response_metadata.get(key)
        if isinstance(value, int | float):
            return float(value)
    usage = response_metadata.get("usage")
    if isinstance(usage, Mapping):
        value = usage.get("cost")
        if isinstance(value, int | float):
            return float(value)
    return None
