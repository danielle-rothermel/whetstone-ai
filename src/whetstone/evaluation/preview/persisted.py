from __future__ import annotations

import json

from dr_store import ObjectStore

from whetstone.core.identity import TypedRef
from whetstone.evaluation import AggregationOutput
from whetstone.evaluation.schema import (
    EvaluationComponentTraces,
    EvaluationEvidence,
    EvaluationOutputsRecord,
)

__all__ = [
    "load_aggregate_value",
    "load_component_traces",
    "load_evaluation_outputs",
]


def load_evaluation_outputs(
    store: ObjectStore,
    evidence: EvaluationEvidence,
) -> EvaluationOutputsRecord:
    raw = store.get(evidence.outputs_ref.reference)
    if raw is None:
        raise RuntimeError("persisted evaluation outputs are missing")
    return EvaluationOutputsRecord.model_validate(raw)


def load_component_traces(
    store: ObjectStore,
    evidence: EvaluationEvidence,
) -> EvaluationComponentTraces:
    raw = store.get(evidence.component_traces_ref.reference)
    if raw is None:
        raise RuntimeError("persisted component traces are missing")
    return EvaluationComponentTraces.model_validate_json(json.dumps(raw))


def load_aggregate_value(
    store: ObjectStore, reference: TypedRef
) -> float | None:
    raw = store.get(reference.reference)
    if not isinstance(raw, dict):
        raise RuntimeError("persisted rollout aggregate is missing")
    output = AggregationOutput.model_validate(raw.get("aggregation_output"))
    return output.value
