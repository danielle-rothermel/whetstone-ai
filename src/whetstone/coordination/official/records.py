from __future__ import annotations

from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    StrictBool,
    StrictInt,
    StrictStr,
    model_validator,
)

from whetstone.coordination.official.mapping import SelectedRecordMapping
from whetstone.core.identity import TypedRef, require_full_hash
from whetstone.experiment.binding import EvalConfigRef

__all__ = [
    "OFFICIAL_EVALUATION_RECORD_SCHEMA",
    "OFFICIAL_PLOT_MANIFEST_SCHEMA",
    "CompletenessDecision",
    "OfficialEvaluationRecord",
    "OfficialPlotManifest",
    "PlannedKeyResult",
    "RecordRevision",
]

# dr-store record schemas (Content Hash addressing; no Identity Hash).
OFFICIAL_EVALUATION_RECORD_SCHEMA = "whetstone.official_evaluation_record"
OFFICIAL_PLOT_MANIFEST_SCHEMA = "whetstone.official_plot_manifest"


class PlannedKeyResult(BaseModel):
    """One planned Rollout Execution Key and its ordinary Result reference.

    ``planned_key`` is the canonical Rollout Execution Key string the official
    aggregation planned. ``result_ref`` is the ordinary Rollout Result Object
    Reference (plus Content Hash) that satisfied it, or ``None`` when the row
    is missing — a missing planned row is recorded here explicitly and stays
    visible; it is never dropped.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    planned_key: StrictStr
    result_ref: TypedRef | None = None

    @model_validator(mode="after")
    def _validate(self) -> PlannedKeyResult:
        if not self.planned_key:
            raise ValueError("planned_key must be non-empty")
        return self

    @property
    def is_present(self) -> bool:
        return self.result_ref is not None


class CompletenessDecision(BaseModel):
    """The explicit completeness + certification decision for one record.

    ``complete`` is true only when every planned key has a bound ordinary
    Rollout Result; ``certified`` is the authority's decision to certify. The
    counts make the accounting auditable — planned, present, and missing rows
    are all visible so no planned key is silently dropped.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    planned_count: StrictInt
    present_count: StrictInt
    missing_count: StrictInt
    complete: StrictBool
    certified: StrictBool
    #: Human/audit note on the certification decision.
    decision_note: StrictStr | None = None

    @model_validator(mode="after")
    def _validate(self) -> CompletenessDecision:
        if self.planned_count < 0:
            raise ValueError("planned_count cannot be negative")
        if self.present_count + self.missing_count != self.planned_count:
            raise ValueError(
                "present + missing must equal planned (every planned key is "
                "accounted for): "
                f"{self.present_count} + {self.missing_count} != "
                f"{self.planned_count}"
            )
        computed_complete = self.missing_count == 0
        if self.complete != computed_complete:
            raise ValueError("complete must be true iff missing_count == 0")
        if self.certified and not self.complete:
            raise ValueError("an incomplete evaluation cannot be certified")
        return self


class RecordRevision(BaseModel):
    """One immutable revision entry in an Official Evaluation Record.

    A record is immutable; a "revision" is a new record naming its predecessor.
    This entry captures the ordinal, the predecessor record reference, and the
    reason so the revision chain is auditable without mutating any record.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    ordinal: StrictInt
    supersedes_ref: TypedRef | None = None
    reason: StrictStr

    @model_validator(mode="after")
    def _validate(self) -> RecordRevision:
        if self.ordinal < 0:
            raise ValueError("revision ordinal cannot be negative")
        if not self.reason:
            raise ValueError("revision reason must be non-empty")
        return self


class OfficialEvaluationRecord(BaseModel):
    """Immutable authority-issued certification of ordinary Rollout Results.

    Names the official Evaluation Binding and exact Eval Config, the planned
    keys with their ordinary Result references + Content Hashes, the complete
    aggregate references, the completeness/certification decision, the
    Objectives + official selection evidence reference, revisions, and the
    MANDATORY ordered selected-record -> graph -> keys -> aggregate mapping.

    It introduces no distinct result role or type: the referenced ordinary
    Rollout Results become official by being certified here, not by relabeling.
    Record-local typed provenance fields only; no universal Provenance class.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # The named authority principal that issued this record.
    authority: StrictStr
    # Official Evaluation Binding id + the exact Eval Config it binds.
    evaluation_binding_id: StrictStr
    eval_config: EvalConfigRef

    # Planned Rollout Execution Keys with their ordinary Result references.
    planned_results: tuple[PlannedKeyResult, ...]

    # Complete aggregate references (one per admitted graph).
    aggregate_refs: tuple[TypedRef, ...]

    # Completeness + certification decision.
    completeness: CompletenessDecision

    # Objectives + official selection evidence (a stored SelectionEvidence
    # record, by typed ref + Content Hash). Absent only when uncertified.
    selection_evidence_ref: TypedRef | None = None

    # MANDATORY ordered mapping preserved by the record.
    selected_record_mapping: SelectedRecordMapping

    # Immutable revision chain (each entry names a predecessor record).
    revisions: tuple[RecordRevision, ...] = ()

    # Record-local typed provenance fields.
    source_revisions: tuple[tuple[str, str], ...] = ()
    dependency_lock: tuple[tuple[str, str], ...] = ()
    environment_identity: StrictStr | None = None
    provenance_note: StrictStr | None = None
    provenance_ordinal: StrictInt | None = None

    @model_validator(mode="after")
    def _validate(self) -> OfficialEvaluationRecord:
        if not self.authority:
            raise ValueError("an official record names its authority")
        # The evaluation_binding_id is the full canonical Evaluation Binding
        # Identity Hash, not merely a non-empty label.
        require_full_hash(
            self.evaluation_binding_id, field="evaluation_binding_id"
        )
        if not self.planned_results:
            raise ValueError("an official record has >=1 planned key")
        # Every planned key appears exactly once: a duplicated planned key
        # would inflate the planned/present counts and let a record certify as
        # complete while covering fewer keys than it claims.
        planned_keys = {p.planned_key for p in self.planned_results}
        if len(planned_keys) != len(self.planned_results):
            raise ValueError(
                "planned_results must have unique planned_key values"
            )

        # The completeness counts MUST match the planned/present accounting.
        present = sum(1 for p in self.planned_results if p.is_present)
        missing = len(self.planned_results) - present
        if self.completeness.planned_count != len(self.planned_results):
            raise ValueError(
                "completeness.planned_count must equal the planned key count"
            )
        if self.completeness.present_count != present:
            raise ValueError(
                "completeness.present_count must equal present result count"
            )
        if self.completeness.missing_count != missing:
            raise ValueError(
                "completeness.missing_count must equal missing result count"
            )
        # A certified record must carry its selection evidence.
        if self.completeness.certified and self.selection_evidence_ref is None:
            raise ValueError(
                "a certified Official Evaluation Record must reference its "
                "official selection evidence"
            )

        # The ordered mapping's planned keys MUST be a subset of the record's
        # planned keys, and its aggregate refs MUST all be declared aggregates.
        # The record's actual present/missing accounting: a planned key is
        # present iff its row carries a bound ordinary Result ref.
        present_keys = {
            p.planned_key for p in self.planned_results if p.is_present
        }
        declared_aggregates = set(self.aggregate_refs)
        for entry in self.selected_record_mapping.entries:
            unknown = set(entry.planned_key_set) - planned_keys
            if unknown:
                raise ValueError(
                    "ordered mapping references planned keys not in the "
                    f"record's planned keys: {sorted(unknown)}"
                )
            # Reconcile the mapping's result attribution with the record's
            # actual present accounting, in both directions: within the
            # entry's own planned set, the attributed result keys are exactly
            # the keys the record accounts as present. Over-attribution names
            # a key the record marks missing; under-attribution silently drops
            # a present result from the per-graph lineage. Either way the
            # load-bearing mapping would contradict the completeness
            # accounting, and plot publication stacks directly on it.
            expected_results = set(entry.planned_key_set) & present_keys
            attributed = set(entry.result_key_set)
            if attributed != expected_results:
                attributed_missing = sorted(attributed - expected_results)
                unattributed_present = sorted(expected_results - attributed)
                raise ValueError(
                    "ordered mapping result_key_set does not reconcile with "
                    "the record's present accounting; keys the record "
                    "accounts as missing (result_ref is None): "
                    f"{attributed_missing}; present keys the mapping omits: "
                    f"{unattributed_present}"
                )
            if entry.aggregate_ref not in declared_aggregates:
                raise ValueError(
                    "ordered mapping references an aggregate not declared in "
                    "the record's aggregate_refs"
                )
        return self

    def record_content(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class OfficialPlotManifest(BaseModel):
    """Immutable plot provenance with an ordered record-to-aggregate
    mapping.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    authority: StrictStr
    record_refs: tuple[TypedRef, ...]
    aggregate_refs: tuple[TypedRef, ...]
    objective_selection_refs: tuple[TypedRef, ...]
    selection_policy: StrictStr
    source_revisions: tuple[tuple[str, str], ...]
    dependency_lock: tuple[tuple[str, str], ...]
    environment_identity: StrictStr
    selected_record_mapping: SelectedRecordMapping
    provenance_note: StrictStr | None = None
    provenance_ordinal: StrictInt | None = None

    @model_validator(mode="after")
    def _validate(self) -> OfficialPlotManifest:
        if not self.authority:
            raise ValueError("an official manifest names its authority")
        if not self.record_refs:
            raise ValueError("a plot manifest names >=1 Official record")
        if not self.aggregate_refs:
            raise ValueError("a plot manifest names >=1 aggregate")
        if not self.objective_selection_refs:
            raise ValueError(
                "a plot manifest names >=1 objective-selection reference"
            )
        if not self.selection_policy:
            raise ValueError("selection_policy must be non-empty")
        if not self.environment_identity:
            raise ValueError("environment_identity must be non-empty")
        # The manifest's aggregate refs MUST cover every aggregate the ordered
        # mapping attributes, so a published plot cannot omit a curve slot.
        declared = set(self.aggregate_refs)
        for entry in self.selected_record_mapping.entries:
            if entry.aggregate_ref not in declared:
                raise ValueError(
                    "ordered mapping references an aggregate not named in the "
                    "manifest's aggregate_refs"
                )
        return self

    def record_content(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
