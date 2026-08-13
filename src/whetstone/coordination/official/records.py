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


OFFICIAL_EVALUATION_RECORD_SCHEMA = "whetstone.official_evaluation_record/v2"
OFFICIAL_PLOT_MANIFEST_SCHEMA = "whetstone.official_plot_manifest/v2"


class PlannedKeyResult(BaseModel):
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
    model_config = ConfigDict(frozen=True, extra="forbid")

    planned_count: StrictInt
    present_count: StrictInt
    missing_count: StrictInt
    complete: StrictBool
    certified: StrictBool

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
    model_config = ConfigDict(frozen=True, extra="forbid")

    authority: StrictStr

    eval_config_hash: StrictStr
    eval_config: EvalConfigRef

    planned_results: tuple[PlannedKeyResult, ...]

    aggregate_refs: tuple[TypedRef, ...]

    completeness: CompletenessDecision

    selection_evidence_ref: TypedRef | None = None

    selected_record_mapping: SelectedRecordMapping

    revisions: tuple[RecordRevision, ...] = ()

    source_revisions: tuple[tuple[str, str], ...] = ()
    dependency_lock: tuple[tuple[str, str], ...] = ()
    environment_label: StrictStr | None = None
    provenance_note: StrictStr | None = None
    provenance_ordinal: StrictInt | None = None

    @model_validator(mode="after")
    def _validate(self) -> OfficialEvaluationRecord:
        if not self.authority:
            raise ValueError("an official record names its authority")

        require_full_hash(self.eval_config_hash, field="eval_config_hash")
        if self.eval_config.config_hash != self.eval_config_hash:
            raise ValueError(
                "eval_config_hash must match the exact Eval Config record"
            )
        if not self.planned_results:
            raise ValueError("an official record has >=1 planned key")

        planned_keys = {p.planned_key for p in self.planned_results}
        if len(planned_keys) != len(self.planned_results):
            raise ValueError(
                "planned_results must have unique planned_key values"
            )

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

        if self.completeness.certified and self.selection_evidence_ref is None:
            raise ValueError(
                "a certified Official Evaluation Record must reference its "
                "official selection evidence"
            )

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
    model_config = ConfigDict(frozen=True, extra="forbid")

    authority: StrictStr
    record_refs: tuple[TypedRef, ...]
    aggregate_refs: tuple[TypedRef, ...]
    objective_selection_refs: tuple[TypedRef, ...]
    selection_policy: StrictStr
    source_revisions: tuple[tuple[str, str], ...]
    dependency_lock: tuple[tuple[str, str], ...]
    environment_label: StrictStr
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
        if not self.environment_label:
            raise ValueError("environment_label must be non-empty")

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
