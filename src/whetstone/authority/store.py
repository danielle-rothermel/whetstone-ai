"""Persistence of immutable official records by typed Object Reference.

Official Evaluation Records, Official Plot Manifests, and official selection
evidence are immutable and stored by typed :class:`~dr_store.ObjectReference`
plus Content Hash — never an Identity Hash. These thin helpers put each record
through dr-store and return the typed reference, cross-checking that the stored
reference matches the record's own content-addressed reference so a caller can
never persist a record under a mismatched reference.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dr_store import ObjectReference

from whetstone.authority.records import (
    OFFICIAL_EVALUATION_RECORD_SCHEMA,
    OFFICIAL_PLOT_MANIFEST_SCHEMA,
    OfficialEvaluationRecord,
    OfficialPlotManifest,
)

if TYPE_CHECKING:
    from dr_store import ObjectStore

    from whetstone.objectives.selection import SelectionEvidence

__all__ = [
    "SELECTION_EVIDENCE_SCHEMA",
    "official_evaluation_record_reference",
    "official_plot_manifest_reference",
    "store_official_evaluation_record",
    "store_official_plot_manifest",
    "store_selection_evidence",
]

# dr-store record schema for persisted official selection evidence.
SELECTION_EVIDENCE_SCHEMA = "whetstone.selection_evidence"


def official_evaluation_record_reference(
    record: OfficialEvaluationRecord,
) -> ObjectReference:
    """The typed Object Reference an Official Evaluation Record resolves under.

    Addressed by Content Hash under the official-record schema; no Identity
    Hash is ever computed for an official record.
    """
    return ObjectReference.for_record(
        OFFICIAL_EVALUATION_RECORD_SCHEMA, record.record_content()
    )


def official_plot_manifest_reference(
    manifest: OfficialPlotManifest,
) -> ObjectReference:
    """The typed Object Reference an Official Plot Manifest resolves under."""
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
    """Immutably persist an Official Evaluation Record; return its ref."""
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
    """Immutably persist an Official Plot Manifest; return its ref."""
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
    """Immutably persist official selection evidence; return its ref."""
    content = evidence.record_content()
    expected = ObjectReference.for_record(SELECTION_EVIDENCE_SCHEMA, content)
    return _put_checked(store, SELECTION_EVIDENCE_SCHEMA, content, expected)
