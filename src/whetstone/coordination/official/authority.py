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
from whetstone.core.roles import EvaluationRole
from whetstone.experiment.binding import (
    EVALUATION_BINDING_SCHEMA_VERSION,
    EvalConfigRef,
    EvaluationBinding,
    ExecutionEnvironmentFingerprint,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from whetstone.coordination.official.mapping import SelectedRecordMapping

__all__ = [
    "EvaluationAuthority",
    "RelabelingRefusedError",
    "UnauthorizedOfficialWriteError",
]


class RelabelingRefusedError(ValueError):
    pass


class UnauthorizedOfficialWriteError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class EvaluationAuthority:
    name: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("an Evaluation Authority must be named")

    def issue_official_binding(
        self,
        *,
        eval_config: EvalConfigRef,
        campaign: str,
        provider_execution_policy_ref: IdentityRef | None = None,
        retry_policy_ref: TypedRef | None = None,
        operational_policy_refs: Sequence[TypedRef] = (),
        environment_fingerprint: ExecutionEnvironmentFingerprint | None = None,
        provenance_note: str | None = None,
        provenance_ordinal: int | None = None,
    ) -> EvaluationBinding:
        return EvaluationBinding(
            schema_version=EVALUATION_BINDING_SCHEMA_VERSION,
            eval_config=eval_config,
            role=EvaluationRole.OFFICIAL,
            authority_principal=self.name,
            campaign=campaign,
            provider_execution_policy_ref=provider_execution_policy_ref,
            retry_policy_ref=retry_policy_ref,
            operational_policy_refs=tuple(operational_policy_refs),
            environment_fingerprint=(
                environment_fingerprint or ExecutionEnvironmentFingerprint()
            ),
            provenance_note=provenance_note,
            provenance_ordinal=provenance_ordinal,
        )

    def _require_official_binding(self, binding: EvaluationBinding) -> None:
        if binding.role is not EvaluationRole.OFFICIAL:
            raise RelabelingRefusedError(
                "internal evaluation evidence can never be certified or "
                "relabeled as official; matching config Identity Hashes "
                "permit comparison, never relabeling. Present an official "
                "Evaluation Binding issued by the authority."
            )
        if binding.authority_principal != self.name:
            raise UnauthorizedOfficialWriteError(
                "official Binding names authority "
                f"{binding.authority_principal!r}, not {self.name!r}; only "
                "the named authority may write it"
            )

    def certify(
        self,
        *,
        evaluation_binding: EvaluationBinding,
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
        self._require_official_binding(evaluation_binding)

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
            evaluation_binding_id=evaluation_binding.identity_hash(),
            eval_config=evaluation_binding.eval_config,
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
