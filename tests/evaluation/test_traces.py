from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from whetstone.core.identity import ImmutableJsonObject
from whetstone.evaluation.drivers.internal import InternalRowOutcome
from whetstone.evaluation.traces import (
    MAX_EXECUTED_COMPONENT_FIELDS,
    MAX_EXECUTED_COMPONENT_JSON_BYTES,
    MAX_EXECUTED_COMPONENT_STEPS,
    ExecutedComponentStep,
    ExecutedComponentTracePayload,
    ExecutedRowState,
    _bounded_trace_json_size,
    _llm_component_step,
    validate_executed_component_trace,
)
from whetstone.execution.partials import PartialCallRecord, PartialLog


def _semantic_trace_bytes(
    steps: tuple[ExecutedComponentStep, ...],
) -> bytes:
    return json.dumps(
        [step.model_dump(mode="json") for step in steps],
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()


def _attempt_mapping_mutation(value: Any, key: str, replacement: Any) -> None:
    value[key] = replacement


def test_executed_component_step_pins_wire_fields_and_order() -> None:
    step = _llm_component_step(
        trace_index=0,
        component_id="generate",
        prompt="exact prompt",
        generation="exact generation",
    )
    assert step.model_dump(mode="json") == {
        "trace_index": 0,
        "component_id": "generate",
        "input_field_names": ["prompt"],
        "output_field_names": ["generation"],
        "inputs": {"prompt": "exact prompt"},
        "outputs": {"generation": "exact generation"},
    }

    ordered = ExecutedComponentStep.model_validate(
        {
            "trace_index": 0,
            "component_id": "generate",
            "input_field_names": ("second", "first"),
            "output_field_names": ("generation",),
            "inputs": {"first": 1, "second": 2},
            "outputs": {"generation": "ok"},
        },
        strict=True,
    )
    assert tuple(ordered.inputs) == ("second", "first")
    assert ordered.inputs.to_json() == {"second": 2, "first": 1}

    with pytest.raises(ValueError, match="unique and non-overlapping"):
        ExecutedComponentStep.model_validate(
            {
                "trace_index": 0,
                "component_id": "generate",
                "input_field_names": ("value",),
                "output_field_names": ("value",),
                "inputs": {"value": 1},
                "outputs": {"value": 2},
            },
            strict=True,
        )


@pytest.mark.parametrize("malformed", [("python tuple",), float("nan")])
def test_executed_component_step_rejects_non_strict_json(malformed) -> None:
    with pytest.raises(ValueError, match=r"strict JSON|finite numbers"):
        ExecutedComponentStep.model_validate(
            {
                "trace_index": 0,
                "component_id": "generate",
                "input_field_names": ("prompt",),
                "output_field_names": ("generation",),
                "inputs": {"prompt": malformed},
                "outputs": {"generation": "ok"},
            },
            strict=True,
        )


def test_executed_component_step_is_deeply_mutation_isolated() -> None:
    source: dict[str, Any] = {
        "payload": {"messages": ["public"], "metadata": {"safe": True}}
    }
    step = ExecutedComponentStep.model_validate(
        {
            "trace_index": 0,
            "component_id": "generate",
            "input_field_names": ("payload",),
            "output_field_names": ("generation",),
            "inputs": source,
            "outputs": {"generation": "accepted"},
        },
        strict=True,
    )
    original_bytes = _semantic_trace_bytes((step,))

    _attempt_mapping_mutation(source["payload"], "api_key", "source-secret")
    source["payload"]["messages"].append("source-secret")
    assert _semantic_trace_bytes((step,)) == original_bytes

    with pytest.raises(TypeError):
        _attempt_mapping_mutation(
            step.inputs, "payload", {"api_key": "model-secret"}
        )
    nested = step.inputs["payload"]
    assert isinstance(nested, ImmutableJsonObject)
    with pytest.raises(TypeError):
        _attempt_mapping_mutation(nested, "api_key", "model-secret")

    dumped = step.model_dump(mode="json")
    dumped["inputs"]["payload"]["api_key"] = "dump-secret"
    dumped["inputs"]["payload"]["messages"].append("dump-secret")
    assert _semantic_trace_bytes((step,)) == original_bytes
    assert b"secret" not in _semantic_trace_bytes((step,))


def test_executed_component_trace_enforces_all_fixed_bounds() -> None:
    names = tuple(
        f"field_{index}" for index in range(MAX_EXECUTED_COMPONENT_FIELDS + 1)
    )
    with pytest.raises(ValueError, match="field count"):
        ExecutedComponentStep.model_validate(
            {
                "trace_index": 0,
                "component_id": "generate",
                "input_field_names": names,
                "output_field_names": (),
                "inputs": {name: None for name in names},
                "outputs": {},
            },
            strict=True,
        )
    with pytest.raises(ValueError, match="byte bound"):
        _llm_component_step(
            trace_index=0,
            component_id="generate",
            prompt="x" * MAX_EXECUTED_COMPONENT_JSON_BYTES,
            generation="ok",
        )

    repeated = tuple(
        _llm_component_step(
            trace_index=index,
            component_id="generate",
            prompt="prompt",
            generation="generation",
        )
        for index in range(MAX_EXECUTED_COMPONENT_STEPS)
    )
    assert validate_executed_component_trace(repeated) == repeated
    with pytest.raises(ValueError, match="step count"):
        validate_executed_component_trace(
            (
                *repeated,
                _llm_component_step(
                    trace_index=MAX_EXECUTED_COMPONENT_STEPS,
                    component_id="generate",
                    prompt="prompt",
                    generation="generation",
                ),
            )
        )
    with pytest.raises(ValueError, match="contiguous from zero"):
        validate_executed_component_trace(
            (repeated[0], repeated[1].model_copy(update={"trace_index": 2}))
        )


def test_executed_component_trace_aborts_aggregate_accounting_early() -> None:
    empty = _llm_component_step(
        trace_index=0,
        component_id="generate",
        prompt="",
        generation="",
    )
    target_size = MAX_EXECUTED_COMPONENT_JSON_BYTES - 8
    large = _llm_component_step(
        trace_index=0,
        component_id="generate",
        prompt="x" * (target_size - empty.canonical_json_bytes),
        generation="",
    )
    small = _llm_component_step(
        trace_index=1,
        component_id="generate",
        prompt="small",
        generation="small",
    )
    assert large.canonical_json_bytes == target_size
    assert (
        large.canonical_json_bytes == len(_semantic_trace_bytes((large,))) - 2
    )
    assert validate_executed_component_trace((large,)) == (large,)
    with pytest.raises(ValueError, match="byte bound"):
        validate_executed_component_trace((large, small))

    consumed = 0

    def sizes():
        nonlocal consumed
        step_sizes = (
            large.canonical_json_bytes,
            small.canonical_json_bytes,
            *(1 for _ in range(MAX_EXECUTED_COMPONENT_STEPS - 2)),
        )
        for size in step_sizes:
            consumed += 1
            yield size

    with pytest.raises(ValueError, match="byte bound"):
        _bounded_trace_json_size(sizes())
    assert consumed == 2


def test_executed_component_trace_partial_round_trip_preserves_order(
    tmp_path: Path,
) -> None:
    step = ExecutedComponentStep.model_validate(
        {
            "trace_index": 0,
            "component_id": "nonlexical",
            "input_field_names": ("zeta", "alpha"),
            "output_field_names": ("omega", "beta"),
            "inputs": {"alpha": {"position": 2}, "zeta": [1, 2]},
            "outputs": {"beta": False, "omega": "first"},
        },
        strict=True,
    )
    payload = ExecutedComponentTracePayload(
        row_state=ExecutedRowState.SUCCESS,
        executed_component_steps=(step,),
    )
    before = _semantic_trace_bytes((step,))
    log = PartialLog(path=tmp_path / "ordered-trace.partial")
    log.append(
        PartialCallRecord(
            phase="internal",
            instance_id="instance",
            unit="unit",
            repeat_id=0,
            request_identity="0" * 64,
            redrive_pending=False,
            observation_payload=payload.model_dump(mode="json"),
        )
    )

    restored_payload = ExecutedComponentTracePayload.from_json_value(
        log.load()[0].observation_payload
    )
    restored = restored_payload.executed_component_steps[0]
    assert restored.input_field_names == ("zeta", "alpha")
    assert tuple(restored.inputs) == ("zeta", "alpha")
    assert restored.inputs["zeta"] == (1, 2)
    assert restored.inputs["alpha"] == {"position": 2}
    assert restored.output_field_names == ("omega", "beta")
    assert tuple(restored.outputs) == ("omega", "beta")
    assert restored.outputs["omega"] == "first"
    assert restored.outputs["beta"] is False
    assert _semantic_trace_bytes((restored,)) == before


def test_successful_row_cannot_omit_its_declared_trace() -> None:
    with pytest.raises(ValueError, match="requires its trace"):
        InternalRowOutcome(
            score=1.0,
            row_state=ExecutedRowState.SUCCESS,
            executed_component_steps=(),
            output_text=None,
        )
