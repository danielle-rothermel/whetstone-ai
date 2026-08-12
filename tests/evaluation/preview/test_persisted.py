from __future__ import annotations

from dr_store import MemoryBackend, ObjectStore

from tests.evaluation.support import _binding, _engine
from whetstone.core.identity import TypedRef
from whetstone.evaluation import AggregationOutput, AggregationStatus
from whetstone.evaluation.engine import EvaluationRequest
from whetstone.evaluation.preview.persisted import (
    load_aggregate_value,
    load_component_traces,
    load_evaluation_outputs,
)


def test_load_evaluation_outputs_and_component_traces(tmp_path) -> None:
    store = ObjectStore(MemoryBackend())
    engine = _engine(tmp_path, store=store)
    binding = _binding(engine)
    evaluated = engine.evaluate(
        EvaluationRequest(
            candidate=engine.experiment.initial_candidate,
            evaluation_binding=binding,
            purpose="persisted-loader-test",
        )
    )

    outputs = load_evaluation_outputs(store, evaluated.evidence)
    traces = load_component_traces(store, evaluated.evidence)

    assert outputs.purpose == "persisted-loader-test"
    assert traces.graph_hash == evaluated.evidence.graph_hash


def test_load_aggregate_value_reads_aggregate() -> None:
    store = ObjectStore(MemoryBackend())
    payload = {
        "aggregation_output": AggregationOutput(
            status=AggregationStatus.OK,
            value=0.75,
            count_total=1,
            count_applicable=1,
            count_present=1,
        ).model_dump(mode="json"),
    }
    reference, _ = store.put("whetstone.aggregate", payload)
    aggregate_ref = TypedRef(
        schema_name=reference.schema,
        content_hash=reference.content_hash,
    )

    assert load_aggregate_value(store, aggregate_ref) == 0.75
