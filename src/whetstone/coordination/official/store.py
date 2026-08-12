from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dr_store import ObjectReference

from whetstone.coordination.official.records import (
    OFFICIAL_EVALUATION_RECORD_SCHEMA,
    OFFICIAL_PLOT_MANIFEST_SCHEMA,
    OfficialEvaluationRecord,
    OfficialPlotManifest,
)

if TYPE_CHECKING:
    from dr_store import ObjectStore

    from whetstone.coordination.official.selection import SelectionEvidence

__all__ = [
    "SELECTION_EVIDENCE_SCHEMA",
    "official_evaluation_record_reference",
    "official_plot_manifest_reference",
    "store_official_evaluation_record",
    "store_official_plot_manifest",
    "store_selection_evidence",
]


SELECTION_EVIDENCE_SCHEMA = "whetstone.selection_evidence"


def official_evaluation_record_reference(
    record: OfficialEvaluationRecord,
) -> ObjectReference:
    return ObjectReference.for_record(
        OFFICIAL_EVALUATION_RECORD_SCHEMA, record.record_content()
    )


def official_plot_manifest_reference(
    manifest: OfficialPlotManifest,
) -> ObjectReference:
    return ObjectReference.for_record(
        OFFICIAL_PLOT_MANIFEST_SCHEMA, manifest.record_content()
    )


def _put_checked(
    store: ObjectStore,
    schema: str,
    content: Any,
    expected: ObjectReference,
) -> ObjectReference:
    reference, _status = store.put(schema, content)
    if reference != expected:
        raise ValueError(
            f"stored {schema} reference does not match the record's own "
            "content-addressed reference"
        )
    return reference


def store_official_evaluation_record(
    store: ObjectStore,
    record: OfficialEvaluationRecord,
) -> ObjectReference:
    return _put_checked(
        store,
        OFFICIAL_EVALUATION_RECORD_SCHEMA,
        record.record_content(),
        official_evaluation_record_reference(record),
    )


def store_official_plot_manifest(
    store: ObjectStore,
    manifest: OfficialPlotManifest,
) -> ObjectReference:
    return _put_checked(
        store,
        OFFICIAL_PLOT_MANIFEST_SCHEMA,
        manifest.record_content(),
        official_plot_manifest_reference(manifest),
    )


def store_selection_evidence(
    store: ObjectStore,
    evidence: SelectionEvidence,
) -> ObjectReference:
    content = evidence.record_content()
    expected = ObjectReference.for_record(SELECTION_EVIDENCE_SCHEMA, content)
    return _put_checked(store, SELECTION_EVIDENCE_SCHEMA, content, expected)
