"""End-to-end: official selection evidence certified by the authority.

Ties Workstream 8 (Objectives + selection) to Workstream 9 (certification):
official selection runs over complete certified aggregates, its evidence is
persisted, and an Official Evaluation Record references that evidence and
preserves the ordered selected-record mapping. No Reward is computed anywhere
on this path.
"""

from __future__ import annotations

from dr_store import MemoryBackend, ObjectStore

from tests.objectives.support import SELECTION_QUALITY_AGGREGATE_NAME
from whetstone.authority import (
    EvaluationAuthority,
    PlannedKeyResult,
    SelectedRecordMapping,
    SelectedRecordMappingEntry,
    TypedRef,
    store_selection_evidence,
)
from whetstone.code_eval.aggregate import (
    RolloutAggregate,
    RowValue,
    TaskRows,
    mean_compression_ratio,
    unweighted_task_mean,
)
from whetstone.objectives import (
    Direction,
    ObjectiveSpec,
    SelectionCandidate,
    select_official,
)

EVAL_HASH = "c" * 64
GRAPH_A = "a" * 64
GRAPH_B = "b" * 64
CONTEXT_ID = "ctx"

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


def _quality(graph_hash: str, value: float) -> RolloutAggregate:
    return unweighted_task_mean(
        aggregate_name=SELECTION_QUALITY_AGGREGATE_NAME,
        graph_hash=graph_hash,
        eval_config_hash=EVAL_HASH,
        evaluation_context_id=CONTEXT_ID,
        task_rows=(
            TaskRows(
                task_identity="t0",
                expected_repeats=1,
                rows=(RowValue(value=value),),
            ),
        ),
        repeat_count=1,
    )


def _compression(graph_hash: str, value: float) -> RolloutAggregate:
    return mean_compression_ratio(
        graph_hash=graph_hash,
        eval_config_hash=EVAL_HASH,
        evaluation_context_id=CONTEXT_ID,
        rows=(RowValue(value=value),),
        task_count=1,
        repeat_count=1,
    )


def test_selection_evidence_certified_and_persisted() -> None:
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
    evidence = select_official(candidates, objective_specs=SPECS)
    assert evidence.selected_candidate_id == "graph-a"

    # Persist the selection evidence immutably; reference it from the record.
    store = ObjectStore(MemoryBackend())
    evidence_ref = store_selection_evidence(store, evidence)

    authority = EvaluationAuthority(name="whetstone-official")
    context = authority.issue_official_context(
        eval_config_hash=EVAL_HASH, campaign="camp-1"
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
                    schema_name="whetstone.rollout_aggregate",
                    content_hash="9" * 64,
                ),
            ),
        )
    )

    record = authority.certify(
        context=context,
        planned_results=(
            PlannedKeyResult(
                planned_key="k0",
                result_ref=TypedRef(
                    schema_name="whetstone.rollout_result",
                    content_hash="d" * 64,
                ),
            ),
        ),
        aggregate_refs=(
            TypedRef(
                schema_name="whetstone.rollout_aggregate",
                content_hash="9" * 64,
            ),
        ),
        selected_record_mapping=mapping,
        selection_evidence_ref=TypedRef(
            schema_name=evidence_ref.schema,
            content_hash=evidence_ref.content_hash,
        ),
    )
    assert record.completeness.certified
    assert record.selection_evidence_ref is not None
    assert (
        record.selection_evidence_ref.content_hash == evidence_ref.content_hash
    )
