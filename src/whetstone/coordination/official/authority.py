from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from whetstone.coordination.official.records import (
    CompletenessDecision,
    OfficialEvaluationRecord,
    OfficialPlotManifest,
    PlannedKeyResult,
    RecordRevision,
)
from whetstone.core.identity import IdentityRef, TypedRef
from whetstone.core.roles import EvalRole
from whetstone.experiment.binding import EvalConfigRef

if TYPE_CHECKING:
    from collections.abc import Sequence

    from whetstone.coordination.official.mapping import SelectedRecordMapping

__all__ = [
    "EvalAuthority",
    "RelabelingRefusedError",
    "UnauthorizedOfficialWriteError",
]


class RelabelingRefusedError(ValueError):
    pass


class UnauthorizedOfficialWriteError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class EvalAuthority:
    name: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("an Evaluation Authority must be named")

    def _require_official_context(
        self,
        *,
        eval_role: EvalRole,
        authority_principal: str | None,
    ) -> None:
        if eval_role is not EvalRole.OFFICIAL:
            raise RelabelingRefusedError(
                "internal evaluation evidence can never be certified or "
                "relabeled as official; matching config Identity Hashes "
                "permit comparison, never relabeling. Present official "
                "evaluation evidence."
            )
        if authority_principal != self.name:
            raise UnauthorizedOfficialWriteError(
                "official evidence names authority "
                f"{authority_principal!r}, not {self.name!r}; only "
                "the named authority may write it"
            )

    def certify(
        self,
        *,
        eval_config: EvalConfigRef,
        eval_role: EvalRole,
        authority_principal: str | None = None,
        planned_results: Sequence[PlannedKeyResult],
        aggregate_refs: Sequence[TypedRef],
        selected_record_mapping: SelectedRecordMapping,
        selection_evidence_ref: TypedRef | None = None,
        certify: bool = True,
        decision_note: str | None = None,
        revisions: Sequence[RecordRevision] = (),
        source_revisions: Sequence[tuple[str, str]] = (),
        dependency_lock: Sequence[tuple[str, str]] = (),
        environment_label: str | None = None,
        provenance_note: str | None = None,
        provenance_ordinal: int | None = None,
    ) -> OfficialEvaluationRecord:
        self._require_official_context(
            eval_role=eval_role,
            authority_principal=authority_principal or self.name,
        )

        planned = tuple(planned_results)
        present = sum(1 for p in planned if p.is_present)
        missing = len(planned) - present
        complete = missing == 0
        certified = bool(certify) and complete
        completeness = CompletenessDecision(
            planned_count=len(planned),
            present_count=present,
            missing_count=missing,
            complete=complete,
            certified=certified,
            decision_note=decision_note,
        )
        return OfficialEvaluationRecord(
            authority=self.name,
            eval_config_hash=eval_config.config_hash,
            eval_config=eval_config,
            planned_results=planned,
            aggregate_refs=tuple(aggregate_refs),
            completeness=completeness,
            selection_evidence_ref=selection_evidence_ref,
            selected_record_mapping=selected_record_mapping,
            revisions=tuple(revisions),
            source_revisions=tuple(source_revisions),
            dependency_lock=tuple(dependency_lock),
            environment_label=environment_label,
            provenance_note=provenance_note,
            provenance_ordinal=provenance_ordinal,
        )

    def publish_plot(
        self,
        *,
        record_refs: Sequence[TypedRef],
        aggregate_refs: Sequence[TypedRef],
        objective_selection_refs: Sequence[TypedRef],
        selection_policy: str,
        source_revisions: Sequence[tuple[str, str]],
        dependency_lock: Sequence[tuple[str, str]],
        environment_label: str,
        selected_record_mapping: SelectedRecordMapping,
        provenance_note: str | None = None,
        provenance_ordinal: int | None = None,
    ) -> OfficialPlotManifest:
        return OfficialPlotManifest(
            authority=self.name,
            record_refs=tuple(record_refs),
            aggregate_refs=tuple(aggregate_refs),
            objective_selection_refs=tuple(objective_selection_refs),
            selection_policy=selection_policy,
            source_revisions=tuple(source_revisions),
            dependency_lock=tuple(dependency_lock),
            environment_label=environment_label,
            selected_record_mapping=selected_record_mapping,
            provenance_note=provenance_note,
            provenance_ordinal=provenance_ordinal,
        )
