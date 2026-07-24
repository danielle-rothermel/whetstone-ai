"""The mandatory ordered selected-record -> graph -> keys -> aggregate mapping.

Proves the load-bearing property: two selected Materialization Records that
share one ``graph_hash`` (converged) share one planned/result-key set and one
aggregate reference, yet each keeps its own ordered entry so the two selected
records stay separately attributable.
"""

from __future__ import annotations

import pytest

from whetstone.authority import (
    SelectedRecordMapping,
    SelectedRecordMappingEntry,
)

from .support import GRAPH_A, GRAPH_B, aggregate_ref, mapping_entry, record_ref


def test_two_selected_records_sharing_a_graph_stay_attributable() -> None:
    # Records m1 and m2 converged on GRAPH_A: they share the planned/result key
    # set and aggregate reference, but each is its own ordered entry keyed by
    # its own record_ref.
    shared_keys = ("k0", "k1")
    entry_1 = mapping_entry(
        record_char="1",
        graph_hash=GRAPH_A,
        planned_keys=shared_keys,
        result_keys=shared_keys,
        aggregate_char="9",
    )
    entry_2 = mapping_entry(
        record_char="2",
        graph_hash=GRAPH_A,
        planned_keys=shared_keys,
        result_keys=shared_keys,
        aggregate_char="9",
    )
    mapping = SelectedRecordMapping(entries=(entry_1, entry_2))

    # Both entries are preserved, in order, separately attributable.
    assert len(mapping.entries) == 2
    assert mapping.entries[0].record_ref != mapping.entries[1].record_ref
    # They resolve to the same graph and aggregate (convergence).
    assert mapping.distinct_graph_hashes == (GRAPH_A,)
    for_graph = mapping.entries_for_graph(GRAPH_A)
    assert len(for_graph) == 2
    assert for_graph[0].aggregate_ref == for_graph[1].aggregate_ref
    # Each entry keeps its own selected Materialization Record reference.
    record_refs = {e.record_ref.content_hash for e in for_graph}
    assert len(record_refs) == 2


def test_distinct_graphs_are_ordered() -> None:
    mapping = SelectedRecordMapping(
        entries=(
            mapping_entry(
                record_char="1",
                graph_hash=GRAPH_A,
                planned_keys=("k0",),
                result_keys=("k0",),
                aggregate_char="8",
            ),
            mapping_entry(
                record_char="2",
                graph_hash=GRAPH_B,
                planned_keys=("k1",),
                result_keys=("k1",),
                aggregate_char="9",
            ),
        )
    )
    assert mapping.distinct_graph_hashes == (GRAPH_A, GRAPH_B)


def test_duplicate_selected_record_is_rejected() -> None:
    entry = mapping_entry(
        record_char="1",
        graph_hash=GRAPH_A,
        planned_keys=("k0",),
        result_keys=("k0",),
        aggregate_char="9",
    )
    with pytest.raises(ValueError, match="exactly once"):
        SelectedRecordMapping(entries=(entry, entry))


def test_converged_entries_must_agree_on_aggregate() -> None:
    # Same graph_hash but different aggregate refs is a contradiction: a shared
    # graph cannot produce two different aggregates.
    entry_1 = mapping_entry(
        record_char="1",
        graph_hash=GRAPH_A,
        planned_keys=("k0",),
        result_keys=("k0",),
        aggregate_char="8",
    )
    entry_2 = mapping_entry(
        record_char="2",
        graph_hash=GRAPH_A,
        planned_keys=("k0",),
        result_keys=("k0",),
        aggregate_char="9",
    )
    with pytest.raises(ValueError, match="disagree on aggregate_ref"):
        SelectedRecordMapping(entries=(entry_1, entry_2))


def test_converged_entries_must_agree_on_key_set() -> None:
    entry_1 = mapping_entry(
        record_char="1",
        graph_hash=GRAPH_A,
        planned_keys=("k0", "k1"),
        result_keys=("k0", "k1"),
        aggregate_char="9",
    )
    entry_2 = mapping_entry(
        record_char="2",
        graph_hash=GRAPH_A,
        planned_keys=("k0",),
        result_keys=("k0",),
        aggregate_char="9",
    )
    with pytest.raises(ValueError, match="disagree on planned_key_set"):
        SelectedRecordMapping(entries=(entry_1, entry_2))


def test_result_keys_must_be_subset_of_planned() -> None:
    with pytest.raises(ValueError, match="not in planned_key_set"):
        SelectedRecordMappingEntry(
            record_ref=record_ref("1"),
            graph_hash=GRAPH_A,
            planned_key_set=("k0",),
            result_key_set=("k0", "k_extra"),
            aggregate_ref=aggregate_ref("9"),
        )


def test_empty_mapping_is_rejected() -> None:
    with pytest.raises(ValueError, match=">=1 entry"):
        SelectedRecordMapping(entries=())
