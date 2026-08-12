from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from enum import UNIQUE, StrEnum, verify

from pydantic import (
    BaseModel,
    ConfigDict,
    JsonValue,
    PrivateAttr,
    model_validator,
)

from whetstone.core.identity import ImmutableJsonObject
from whetstone.experiment.graph.nodes import PROVIDER_GENERATION_OUTPUT_FIELD

__all__ = [
    "MAX_EXECUTED_COMPONENT_FIELDS",
    "MAX_EXECUTED_COMPONENT_JSON_BYTES",
    "MAX_EXECUTED_COMPONENT_JSON_DEPTH",
    "MAX_EXECUTED_COMPONENT_STEPS",
    "RENDER_FAILURE_CODE",
    "ExecutedComponentStep",
    "ExecutedComponentTracePayload",
    "ExecutedRowState",
    "validate_executed_component_trace",
]

RENDER_FAILURE_CODE = "render_key_error"

# Executed-component traces cross worker and partial-log JSON boundaries. The
# fixed limits keep this audit payload finite independently of provider limits.
MAX_EXECUTED_COMPONENT_STEPS = 16
MAX_EXECUTED_COMPONENT_FIELDS = 32
MAX_EXECUTED_COMPONENT_JSON_BYTES = 4 * 1024 * 1024
MAX_EXECUTED_COMPONENT_JSON_DEPTH = 32
_COMPONENT_PROMPT_FIELD = "prompt"


@verify(UNIQUE)
class ExecutedRowState(StrEnum):
    """The explicit execution state of one planned environment row."""

    SUCCESS = "success"
    FAILED = "failed"
    MISSING = "missing"


class _JsonByteCounter:
    """Exact bounded UTF-8 byte counter for compact strict JSON."""

    __slots__ = ("limit", "total")

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.total = 0

    def add(self, count: int) -> None:
        self.total += count
        if self.total > self.limit:
            raise ValueError("executed-component JSON exceeds its byte bound")


def _add_json_string_bytes(counter: _JsonByteCounter, value: str) -> None:
    """Count the compact ``ensure_ascii=False`` JSON spelling of a string."""
    counter.add(2)
    for character in value:
        codepoint = ord(character)
        if character in {'"', "\\"} or character in "\b\f\n\r\t":
            counter.add(2)
        elif codepoint < 0x20:
            counter.add(6)
        elif codepoint < 0x80:
            counter.add(1)
        elif codepoint < 0x800:
            counter.add(2)
        elif codepoint < 0x10000:
            if 0xD800 <= codepoint <= 0xDFFF:
                raise ValueError(
                    "executed-component JSON contains an invalid surrogate"
                )
            counter.add(3)
        else:
            counter.add(4)


def _bounded_canonical_json_size(value: object, *, max_bytes: int) -> int:
    """Count compact strict-JSON bytes incrementally and stop at the bound."""
    counter = _JsonByteCounter(max_bytes)

    def add_value(current: object, *, depth: int) -> None:
        if depth > MAX_EXECUTED_COMPONENT_JSON_DEPTH:
            raise ValueError("executed-component JSON exceeds its depth bound")
        if current is None:
            counter.add(4)
        elif type(current) is bool:
            counter.add(4 if current else 5)
        elif type(current) is int:
            counter.add(len(str(current)))
        elif type(current) is float:
            if not math.isfinite(current):
                raise ValueError(
                    "executed-component JSON must contain finite numbers"
                )
            counter.add(
                len(
                    json.dumps(
                        current,
                        allow_nan=False,
                        separators=(",", ":"),
                    )
                )
            )
        elif type(current) is str:
            _add_json_string_bytes(counter, current)
        elif isinstance(current, Mapping):
            counter.add(1)
            for index, (key, nested) in enumerate(current.items()):
                if type(key) is not str:
                    raise ValueError(
                        "executed-component JSON object keys must be strings"
                    )
                if index:
                    counter.add(1)
                _add_json_string_bytes(counter, key)
                counter.add(1)
                add_value(nested, depth=depth + 1)
            counter.add(1)
        elif isinstance(current, (list, tuple)):
            counter.add(1)
            for index, item in enumerate(current):
                if index:
                    counter.add(1)
                add_value(item, depth=depth + 1)
            counter.add(1)
        else:
            raise ValueError(
                "executed-component values must contain only strict JSON"
            )

    add_value(value, depth=0)
    return counter.total


def _bounded_trace_json_size(
    step_sizes: Iterable[int],
    *,
    max_bytes: int = MAX_EXECUTED_COMPONENT_JSON_BYTES,
) -> int:
    """Add cached sizes and abort when the trace array is too big."""
    counter = _JsonByteCounter(max_bytes)
    counter.add(2)
    for index, step_size in enumerate(step_sizes):
        if index:
            counter.add(1)
        counter.add(step_size)
    return counter.total


class ExecutedComponentStep(BaseModel):
    """One observed component execution crossing a strict JSON boundary."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    trace_index: int
    component_id: str
    input_field_names: tuple[str, ...]
    output_field_names: tuple[str, ...]
    inputs: ImmutableJsonObject
    outputs: ImmutableJsonObject

    _canonical_json_bytes: int = PrivateAttr(default=0)

    @model_validator(mode="after")
    def _valid_step(self) -> ExecutedComponentStep:
        if self.trace_index < 0:
            raise ValueError("trace_index must be nonnegative")
        if (
            not self.component_id
            or self.component_id != self.component_id.strip()
        ):
            raise ValueError(
                "component_id must be a nonempty stable identifier"
            )
        names = self.input_field_names + self.output_field_names
        if len(names) > MAX_EXECUTED_COMPONENT_FIELDS:
            raise ValueError(
                "executed-component field count exceeds its bound"
            )
        if any(not name or name != name.strip() for name in names):
            raise ValueError("executed-component field names must be nonempty")
        if len(set(names)) != len(names):
            raise ValueError(
                "executed-component input and output names must be unique "
                "and non-overlapping"
            )
        if frozenset(self.inputs) != frozenset(self.input_field_names):
            raise ValueError(
                "input field names must exactly match input object keys"
            )
        if frozenset(self.outputs) != frozenset(self.output_field_names):
            raise ValueError(
                "output field names must exactly match output object keys"
            )
        input_values = self.inputs.to_json()
        output_values = self.outputs.to_json()
        object.__setattr__(
            self,
            "inputs",
            ImmutableJsonObject(
                {name: input_values[name] for name in self.input_field_names}
            ),
        )
        object.__setattr__(
            self,
            "outputs",
            ImmutableJsonObject(
                {name: output_values[name] for name in self.output_field_names}
            ),
        )
        canonical_json_bytes = _bounded_canonical_json_size(
            {
                "trace_index": self.trace_index,
                "component_id": self.component_id,
                "input_field_names": self.input_field_names,
                "output_field_names": self.output_field_names,
                "inputs": self.inputs,
                "outputs": self.outputs,
            },
            max_bytes=MAX_EXECUTED_COMPONENT_JSON_BYTES - 2,
        )
        object.__setattr__(self, "_canonical_json_bytes", canonical_json_bytes)
        return self

    @property
    def canonical_json_bytes(self) -> int:
        """The exact cached compact-JSON size of this immutable step."""
        return self._canonical_json_bytes


def validate_executed_component_trace(
    steps: tuple[ExecutedComponentStep, ...],
) -> tuple[ExecutedComponentStep, ...]:
    """Validate one bounded, authoritatively ordered execution trace."""
    if len(steps) > MAX_EXECUTED_COMPONENT_STEPS:
        raise ValueError("executed-component step count exceeds its bound")
    for expected_index, step in enumerate(steps):
        if step.trace_index != expected_index:
            raise ValueError(
                "executed-component trace indexes must be contiguous from zero"
            )
    _bounded_trace_json_size(step.canonical_json_bytes for step in steps)
    return steps


class ExecutedComponentTracePayload(BaseModel):
    """Strict row-state and trace payload persisted beside generic fields."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    row_state: ExecutedRowState
    executed_component_steps: tuple[ExecutedComponentStep, ...]

    @model_validator(mode="after")
    def _valid_trace(self) -> ExecutedComponentTracePayload:
        validate_executed_component_trace(self.executed_component_steps)
        if (
            self.row_state is ExecutedRowState.MISSING
            and self.executed_component_steps
        ):
            raise ValueError(
                "a missing row cannot contain executed components"
            )
        return self

    @classmethod
    def from_json_value(
        cls, payload: JsonValue | None
    ) -> ExecutedComponentTracePayload:
        """Validate a decoded partial payload using strict JSON semantics."""
        return cls.model_validate_json(json.dumps(payload))


def _llm_component_step(
    *,
    trace_index: int,
    component_id: str,
    prompt: str,
    generation: str,
) -> ExecutedComponentStep:
    """Capture the exact semantic input and accepted output of one LLM node."""
    return ExecutedComponentStep(
        trace_index=trace_index,
        component_id=component_id,
        input_field_names=(_COMPONENT_PROMPT_FIELD,),
        output_field_names=(PROVIDER_GENERATION_OUTPUT_FIELD,),
        inputs=ImmutableJsonObject({_COMPONENT_PROMPT_FIELD: prompt}),
        outputs=ImmutableJsonObject(
            {PROVIDER_GENERATION_OUTPUT_FIELD: generation}
        ),
    )


def _llm_component_values(
    step: ExecutedComponentStep, *, component_id: str
) -> tuple[str, str]:
    """Validate and return one closed LLM node's prompt and generation."""
    if step.component_id != component_id:
        raise ValueError(
            f"executed component must be graph node {component_id!r}"
        )
    if step.input_field_names != (
        _COMPONENT_PROMPT_FIELD,
    ) or step.output_field_names != (PROVIDER_GENERATION_OUTPUT_FIELD,):
        raise ValueError("LLM trace fields must be prompt and generation")
    prompt = step.inputs[_COMPONENT_PROMPT_FIELD]
    generation = step.outputs[PROVIDER_GENERATION_OUTPUT_FIELD]
    if type(prompt) is not str or type(generation) is not str:
        raise ValueError("LLM trace prompt and generation must be strings")
    return prompt, generation
