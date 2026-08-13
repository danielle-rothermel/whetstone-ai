from __future__ import annotations

import json

from dr_store import ObjectStore

from whetstone.core.identity import TypedRef
from whetstone.eval import AggregationOutput
from whetstone.eval.schema import (
    EvalTraces,
    EvalEvidence,
    EvalOutputsRecord,
)

__all__ = [
    "load_aggregate_value",
    "load_component_traces",
    "load_evaluation_outputs",
]


def load_evaluation_outputs(
    store: ObjectStore,
    evidence: EvalEvidence,
) -> EvalOutputsRecord:
    raw = store.get(evidence.outputs_ref.reference)
    if raw is None:
        raise RuntimeError("persisted evaluation outputs are missing")
    return EvalOutputsRecord.model_validate(raw)


def load_component_traces(
    store: ObjectStore,
    evidence: EvalEvidence,
) -> EvalTraces:
    raw = store.get(evidence.traces_ref.reference)
    if raw is None:
        raise RuntimeError("persisted component traces are missing")
    return EvalTraces.model_validate_json(json.dumps(raw))


def load_aggregate_value(
    store: ObjectStore, reference: TypedRef
) -> float | None:
    raw = store.get(reference.reference)
    if not isinstance(raw, dict):
        raise RuntimeError("persisted aggregate is missing")
    output = AggregationOutput.model_validate(raw.get("aggregation_output"))
    return output.value
