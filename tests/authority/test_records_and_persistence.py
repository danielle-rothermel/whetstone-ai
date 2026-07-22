"""Official Plot Manifest publication and immutable persistence.

Proves the manifest preserves the ordered mapping (including two selected
records sharing a graph), the immutable content-addressed persistence path for
official records / manifests / selection evidence, and that the record's own
reference matches the stored reference.
"""

from __future__ import annotations

import pytest
from dr_store import MemoryBackend, ObjectStore

from whetstone.authority import (
    EvaluationAuthority,
    OfficialPlotManifest,
    PlannedKeyResult,
    SelectedRecordMapping,
    official_evaluation_record_reference,
    official_plot_manifest_reference,
    store_official_evaluation_record,
    store_official_plot_manifest,
)

from .support import (
    EVAL_HASH,
    GRAPH_A,
    aggregate_ref,
    mapping_entry,
    oer_ref,
    result_ref,
    single_entry_mapping,
)


def _authority() -> EvaluationAuthority:
    return EvaluationAuthority(name="whetstone-official")


def _shared_graph_mapping() -> SelectedRecordMapping:
    shared = ("k0", "k1")
    return SelectedRecordMapping(
        entries=(
            mapping_entry(
                record_char="1",
                graph_hash=GRAPH_A,
                planned_keys=shared,
                result_keys=shared,
                aggregate_char="9",
            ),
            mapping_entry(
                record_char="2",
                graph_hash=GRAPH_A,
                planned_keys=shared,
                result_keys=shared,
                aggregate_char="9",
            ),
        )
    )


def test_publish_plot_preserves_shared_graph_attribution() -> None:
    authority = _authority()
    mapping = _shared_graph_mapping()
    manifest = authority.publish_plot(
        record_refs=(oer_ref("5"),),
        aggregate_refs=(aggregate_ref("9"),),
        objective_selection_refs=(result_ref("6"),),
        selection_policy="select_all/v1",
        source_revisions=(("whetstone", "rev-1"),),
        dependency_lock=(("dr-code", "1.0.0"),),
        environment_identity="env-1",
        selected_record_mapping=mapping,
    )
    # Two selected records sharing a graph stay separately attributable in the
    # published manifest.
    assert len(manifest.selected_record_mapping.entries) == 2
    for_graph = manifest.selected_record_mapping.entries_for_graph(GRAPH_A)
    assert len(for_graph) == 2
    assert for_graph[0].record_ref != for_graph[1].record_ref


def test_manifest_is_immutable() -> None:
    manifest = _authority().publish_plot(
        record_refs=(oer_ref("5"),),
        aggregate_refs=(aggregate_ref("9"),),
        objective_selection_refs=(result_ref("6"),),
        selection_policy="select_all/v1",
        source_revisions=(),
        dependency_lock=(),
        environment_identity="env-1",
        selected_record_mapping=single_entry_mapping(planned_keys=("k0",)),
    )
    with pytest.raises((TypeError, ValueError)):
        manifest.selection_policy = "other"  # type: ignore


def test_manifest_aggregate_refs_must_cover_mapping() -> None:
    # The mapping attributes aggregate '9', but the manifest declares only '8'.
    with pytest.raises(ValueError, match="not named in the manifest"):
        OfficialPlotManifest(
            authority="whetstone-official",
            record_refs=(
                oer_ref("5"),
            ),
            aggregate_refs=(aggregate_ref("8"),),
            objective_selection_refs=(result_ref("6"),),
            selection_policy="select_all/v1",
            source_revisions=(),
            dependency_lock=(),
            environment_identity="env-1",
            selected_record_mapping=single_entry_mapping(
                planned_keys=("k0",)
            ),
        )


def test_official_record_persists_immutably() -> None:
    authority = _authority()
    context = authority.issue_official_context(
        eval_config_hash=EVAL_HASH, campaign="camp-1"
    )
    record = authority.certify(
        context=context,
        planned_results=(
            PlannedKeyResult(planned_key="k0", result_ref=result_ref("d")),
        ),
        aggregate_refs=(aggregate_ref("9"),),
        selected_record_mapping=single_entry_mapping(planned_keys=("k0",)),
        selection_evidence_ref=result_ref("f"),
    )
    store = ObjectStore(MemoryBackend())
    reference = store_official_evaluation_record(store, record)
    # The stored reference matches the record's own content-addressed ref.
    assert reference == official_evaluation_record_reference(record)
    # Round-trips: the exact immutable record comes back verified.
    fetched = store.get(reference)
    assert fetched == record.record_content()


def test_official_manifest_persists_immutably() -> None:
    manifest = _authority().publish_plot(
        record_refs=(oer_ref("5"),),
        aggregate_refs=(aggregate_ref("9"),),
        objective_selection_refs=(result_ref("6"),),
        selection_policy="select_all/v1",
        source_revisions=(),
        dependency_lock=(),
        environment_identity="env-1",
        selected_record_mapping=single_entry_mapping(planned_keys=("k0",)),
    )
    store = ObjectStore(MemoryBackend())
    reference = store_official_plot_manifest(store, manifest)
    assert reference == official_plot_manifest_reference(manifest)
    assert store.get(reference) == manifest.record_content()
