from __future__ import annotations

import pytest
from dr_store import MemoryBackend, ObjectReference, ObjectStore, PutStatus

from tests.experiment.support import (
    SELECTION_QUALITY_AGGREGATE_NAME,
    aggregate_plan,
)
from whetstone.coordination.official import (
    EvaluationAuthority,
    PlannedKeyResult,
    SelectedRecordMapping,
    SelectedRecordMappingEntry,
    store_selection_evidence,
)
from whetstone.coordination.official.selection import (
    ObjectiveSpec,
    SelectionCandidate,
    SelectionEvidence,
    select_official,
)
from whetstone.core.identity import TypedRef
from whetstone.evaluation.aggregate import (
    Aggregate,
    RowValue,
    TaskRows,
    unweighted_task_mean,
)
from whetstone.experiment.objectives import Direction

from .support import eval_config_ref

GRAPH_A = "a" * 64
GRAPH_B = "b" * 64
BINDING_ID = "d" * 64

SPECS = (
    ObjectiveSpec(
        objective_name="quality",
        aggregate_name=SELECTION_QUALITY_AGGREGATE_NAME,
        direction=Direction.MAXIMIZE,
    ),
    ObjectiveSpec(
        objective_name="compression",
        aggregate_name="mean_compression_ratio",
        direction=Direction.MINIMIZE,
    ),
)


def _quality(graph_hash: str, value: float) -> Aggregate:
    return unweighted_task_mean(
        aggregate_name=SELECTION_QUALITY_AGGREGATE_NAME,
        graph_hash=graph_hash,
        evaluation_binding_hash=BINDING_ID,
        task_rows=(
            TaskRows(
                task_hash="task-0",
                rows=(RowValue(value=value),),
            ),
        ),
        plan=aggregate_plan(tasks=1, num_samples=1),
    )


def _compression(graph_hash: str, value: float) -> Aggregate:
    return unweighted_task_mean(
        aggregate_name="mean_compression_ratio",
        graph_hash=graph_hash,
        evaluation_binding_hash=BINDING_ID,
        task_rows=(
            TaskRows(
                task_hash="task-0",
                rows=(RowValue(value=value),),
            ),
        ),
        plan=aggregate_plan(tasks=1, num_samples=1),
    )


def _selection_evidence() -> SelectionEvidence:
    candidates = [
        SelectionCandidate(
            candidate_id="graph-a",
            graph_hash=GRAPH_A,
            aggregates={
                SELECTION_QUALITY_AGGREGATE_NAME: _quality(GRAPH_A, 1.0),
                "mean_compression_ratio": _compression(GRAPH_A, 2.0),
            },
        ),
        SelectionCandidate(
            candidate_id="graph-b",
            graph_hash=GRAPH_B,
            aggregates={
                SELECTION_QUALITY_AGGREGATE_NAME: _quality(GRAPH_B, 0.5),
                "mean_compression_ratio": _compression(GRAPH_B, 3.0),
            },
        ),
    ]
    return select_official(candidates, objective_specs=SPECS)


def test_selection_evidence_certified_and_persisted() -> None:
    evidence = _selection_evidence()
    assert evidence.selected_candidate_id == "graph-a"

    store = ObjectStore(MemoryBackend())
    evidence_ref = store_selection_evidence(store, evidence)
    assert evidence_ref == ObjectReference.for_record(
        "whetstone.selection_evidence", evidence.record_content()
    )
    assert store.get(evidence_ref) == evidence.record_content()

    authority = EvaluationAuthority(name="whetstone-official")
    binding = authority.issue_official_binding(
        eval_config=eval_config_ref(), campaign="camp-1"
    )

    mapping = SelectedRecordMapping(
        entries=(
            SelectedRecordMappingEntry(
                record_ref=TypedRef(
                    schema_name="whetstone.materialization_record",
                    content_hash="1" * 64,
                ),
                graph_hash=GRAPH_A,
                planned_key_set=("k0",),
                result_key_set=("k0",),
                aggregate_ref=TypedRef(
                    schema_name="whetstone.aggregate",
                    content_hash="9" * 64,
                ),
            ),
        )
    )

    selection_evidence_ref = TypedRef(
        schema_name=evidence_ref.schema,
        content_hash=evidence_ref.content_hash,
    )
    record = authority.certify(
        evaluation_binding=binding,
        planned_results=(
            PlannedKeyResult(
                planned_key="k0",
                result_ref=TypedRef(
                    schema_name="whetstone.generation_result",
                    content_hash="d" * 64,
                ),
            ),
        ),
        aggregate_refs=(
            TypedRef(
                schema_name="whetstone.aggregate",
                content_hash="9" * 64,
            ),
        ),
        selected_record_mapping=mapping,
        selection_evidence_ref=selection_evidence_ref,
    )
    assert record.completeness.certified
    assert record.selection_evidence_ref == selection_evidence_ref


def test_selection_store_rejects_a_misreported_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _selection_evidence()
    store = ObjectStore(MemoryBackend())

    def misreport_reference(
        schema: str, record: object
    ) -> tuple[ObjectReference, PutStatus]:
        del record
        return ObjectReference(schema=schema, content_hash="0" * 64), (
            PutStatus.STORED
        )

    monkeypatch.setattr(store, "put", misreport_reference)

    with pytest.raises(
        ValueError,
        match="reference does not match the record's own content-addressed",
    ):
        store_selection_evidence(store, evidence)
