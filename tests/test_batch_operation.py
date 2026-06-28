from __future__ import annotations

from dr_dspy import batch_operation


def test_operation_key_is_stable_and_order_insensitive() -> None:
    left = {"experiment": "exp", "limit": 10, "nested": {"a": 1, "b": 2}}
    right = {"nested": {"b": 2, "a": 1}, "limit": 10, "experiment": "exp"}
    changed = {"experiment": "exp", "limit": 11, "nested": {"a": 1, "b": 2}}

    left_key = batch_operation.operation_key(left)

    assert left_key == batch_operation.operation_key(right)
    assert left_key != batch_operation.operation_key(changed)


def test_operation_workflow_id_includes_kind_key_and_attempt() -> None:
    workflow_id = batch_operation.operation_workflow_id(
        batch_operation.BatchOperationKind.REPAIR, "abc", 2
    )

    assert workflow_id == "repair:abc:2"


def test_merged_counters_adds_deltas_without_losing_existing_keys() -> None:
    assert batch_operation.merged_counters(
        {"generation_processed": 3, "marked": 1},
        {"generation_processed": 2, "scoring_processed": 5},
    ) == {
        "generation_processed": 5,
        "marked": 1,
        "scoring_processed": 5,
    }
