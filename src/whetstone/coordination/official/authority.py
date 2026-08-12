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
    """Internal evidence was offered for official certification/relabeling.

    Raised whenever an internal-role Evaluation Binding (or evidence bearing
    one) is presented on the official write path. Byte-identical config
    Identity Hashes between an internal and an official run do not make the
    internal evidence official: the refusal is by role, not by identity.
    """


class UnauthorizedOfficialWriteError(ValueError):
    """An official artifact was requested with the wrong/absent authority.

    The official write path checks that the requesting authority principal
    matches the authority named on the official Evaluation Binding.
    """


@dataclass(frozen=True, slots=True)
class EvaluationAuthority:
    """A named trusted principal and the official write path.

    Instances are the only issuer of official Evaluation Bindings, Official
    Evaluation Records, and Official Plot Manifests. The ``name`` is the
    principal recorded on every artifact this authority issues.
    """

    name: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("an Evaluation Authority must be named")

    # ------------------------------------------------------------------
    # Official Evaluation Binding issuance
    # ------------------------------------------------------------------

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
        """Issue an official Evaluation Binding bound to this authority.

        The returned Binding has role ``official`` and ``authority_principal``
        set to
        this principal's name. Only an authority instance can mint an official
        Binding this way; the ordinary Eval Config it binds is unchanged (the
        role qualifies its use, it is not a new Config type or identity).
        """
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

    # ------------------------------------------------------------------
    # Relabeling refusal
    # ------------------------------------------------------------------

    def _require_official_binding(self, binding: EvaluationBinding) -> None:
        """Refuse any internal-role binding on the official write path.

        This is the seam that makes "internal can never be relabeled to
        official" true *on the authority write path*: an internal-role Binding
        is rejected here regardless of whether its Eval Config identity matches
        an official run byte for byte. This is an enforced funnel, not a
        cryptographic boundary — see the module docstring: direct model
        construction of an official-role Binding is not blocked here.
        """
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

    # ------------------------------------------------------------------
    # Official Evaluation Record certification
    # ------------------------------------------------------------------

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
        environment_identity: str | None = None,
        provenance_note: str | None = None,
        provenance_ordinal: int | None = None,
    ) -> OfficialEvaluationRecord:
        """Create an immutable Official Evaluation Record over results.

        Refuses internal-role evidence (relabeling refusal). Computes the
        completeness decision from the planned/present accounting so no planned
        key is silently dropped, and certifies only when the evaluation is
        complete. The referenced ordinary Generation Results become official by
        this certification; they are not copied or relabeled.
        """
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
            environment_identity=environment_identity,
            provenance_note=provenance_note,
            provenance_ordinal=provenance_ordinal,
        )

    # ------------------------------------------------------------------
    # Official Plot Manifest publication
    # ------------------------------------------------------------------

    def publish_plot(
        self,
        *,
        record_refs: Sequence[TypedRef],
        aggregate_refs: Sequence[TypedRef],
        objective_selection_refs: Sequence[TypedRef],
        selection_policy: str,
        source_revisions: Sequence[tuple[str, str]],
        dependency_lock: Sequence[tuple[str, str]],
        environment_identity: str,
        selected_record_mapping: SelectedRecordMapping,
        provenance_note: str | None = None,
        provenance_ordinal: int | None = None,
    ) -> OfficialPlotManifest:
        """Publish an immutable Official Plot Manifest naming Official records.

        The manifest preserves the same ordered mapping the certified records
        carry, so a published plot stays attributable to its selected records
        and curve slots even across graph convergence.
        """
        return OfficialPlotManifest(
            authority=self.name,
            record_refs=tuple(record_refs),
            aggregate_refs=tuple(aggregate_refs),
            objective_selection_refs=tuple(objective_selection_refs),
            selection_policy=selection_policy,
            source_revisions=tuple(source_revisions),
            dependency_lock=tuple(dependency_lock),
            environment_identity=environment_identity,
            selected_record_mapping=selected_record_mapping,
            provenance_note=provenance_note,
            provenance_ordinal=provenance_ordinal,
        )
