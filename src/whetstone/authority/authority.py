"""The Evaluation Authority: named principal and the official write path.

Per the vocabulary (*Evaluation Authority*, *Official Evaluation*, *Internal
Evaluation*) and Workstream 9. An :class:`EvaluationAuthority` is the *named
trusted principal and official write path* authorized to:

* issue **official** Evaluation Contexts (role ``official``, authority set);
* certify ordinary evaluation evidence and create immutable
  :class:`~whetstone.authority.records.OfficialEvaluationRecord` values;
* publish immutable
  :class:`~whetstone.authority.records.OfficialPlotManifest` values.

The official write path is the *authority-enforced* way official artifacts come
to exist. Officialness is a plain data field (``role``/``authority`` on the
Evaluation Context; ``authority``/``completeness.certified`` on the record),
not a capability or signature, so the ``EvaluationAuthority`` methods are an
enforced funnel rather than an unforgeable cryptographic boundary. The
authority-enforced guarantees are:

* the relabeling refusal (below) applies on every :meth:`certify` /
  :meth:`publish_plot` call — internal evidence is refused there;
* an :class:`~whetstone.authority.records.OfficialEvaluationRecord` will only
  accept an ``evaluation_context_id`` that is a full Evaluation Context
  Identity Hash, so a record cannot name a context id that no
  ``EvaluationContext`` could have produced (see ``records.py``).

Callers with direct model access CAN still construct an official-role
``EvaluationContext`` or an ``OfficialEvaluationRecord`` without going through
an authority instance; the relabeling refusal is real only on the
``certify()`` / ``issue_official_context()`` path. Treating officialness as an
unforgeable boundary would require a capability-held token or signing on
construction, which this schema deliberately does not add.

The load-bearing refusal — **internal evaluation can never be relabeled or
copied to official merely because identities match** — is enforced here:
:meth:`EvaluationAuthority.certify` refuses any evidence bearing an internal
Evaluation Context, even when its config Identity Hashes are byte-identical to
an official run. Equal identities permit *comparison*, never *relabeling*.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from whetstone.authority.records import (
    CompletenessDecision,
    OfficialEvaluationRecord,
    OfficialPlotManifest,
    PlannedKeyResult,
    RecordRevision,
)
from whetstone.graph.rollout import (
    EnvironmentAttestation,
    EvaluationContext,
    EvaluationRole,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from whetstone.authority.mapping import SelectedRecordMapping
    from whetstone.authority.reference import TypedContentRef

__all__ = [
    "EvaluationAuthority",
    "RelabelingRefusedError",
    "UnauthorizedOfficialWriteError",
]


class RelabelingRefusedError(ValueError):
    """Internal evidence was offered for official certification/relabeling.

    Raised whenever an internal-role Evaluation Context (or evidence bearing
    one) is presented on the official write path. Byte-identical config
    Identity Hashes between an internal and an official run do not make the
    internal evidence official: the refusal is by role, not by identity.
    """


class UnauthorizedOfficialWriteError(ValueError):
    """An official artifact was requested with the wrong/absent authority.

    The official write path checks that the requesting authority principal
    matches the authority named on the official Evaluation Context.
    """


class EvaluationAuthority:
    """A named trusted principal and the official write path.

    Instances are the only issuer of official Evaluation Contexts, Official
    Evaluation Records, and Official Plot Manifests. The ``name`` is the
    principal recorded on every artifact this authority issues.
    """

    def __init__(self, *, name: str) -> None:
        if not name:
            raise ValueError("an Evaluation Authority must be named")
        self._name = name

    @property
    def name(self) -> str:
        """The named principal recorded on issued official artifacts."""
        return self._name

    # ------------------------------------------------------------------
    # Official Evaluation Context issuance
    # ------------------------------------------------------------------

    def issue_official_context(
        self,
        *,
        eval_config_hash: str,
        campaign: str,
        provider_execution_policy_ref: str | None = None,
        retry_policy_ref: str | None = None,
        operational_policy_refs: Sequence[str] = (),
        environment: EnvironmentAttestation | None = None,
        provenance_note: str | None = None,
        provenance_ordinal: int | None = None,
    ) -> EvaluationContext:
        """Issue an official Evaluation Context bound to this authority.

        The returned Context has role ``official`` and ``authority`` set to
        this principal's name. Only an authority instance can mint an official
        Context this way; the ordinary Eval Config it binds is unchanged (the
        role qualifies its use, it is not a new Config type or identity).
        """
        return EvaluationContext(
            eval_config_hash=eval_config_hash,
            role=EvaluationRole.OFFICIAL,
            authority=self._name,
            campaign=campaign,
            provider_execution_policy_ref=provider_execution_policy_ref,
            retry_policy_ref=retry_policy_ref,
            operational_policy_refs=tuple(operational_policy_refs),
            environment=environment or EnvironmentAttestation(),
            provenance_note=provenance_note,
            provenance_ordinal=provenance_ordinal,
        )

    # ------------------------------------------------------------------
    # Relabeling refusal
    # ------------------------------------------------------------------

    def _require_official_context(
        self, context: EvaluationContext
    ) -> None:
        """Refuse any internal-role context on the official write path.

        This is the seam that makes "internal can never be relabeled to
        official" true *on the authority write path*: an internal-role Context
        is rejected here regardless of whether its ``eval_config_hash`` matches
        an official run byte for byte. This is an enforced funnel, not a
        cryptographic boundary — see the module docstring: direct model
        construction of an official-role Context is not blocked here.
        """
        if context.role is not EvaluationRole.OFFICIAL:
            raise RelabelingRefusedError(
                "internal evaluation evidence can never be certified or "
                "relabeled as official; matching config Identity Hashes "
                "permit comparison, never relabeling. Present an official "
                "Evaluation Context issued by the authority."
            )
        if context.authority != self._name:
            raise UnauthorizedOfficialWriteError(
                f"official Context names authority {context.authority!r}, not "
                f"{self._name!r}; only the named authority may write it"
            )

    # ------------------------------------------------------------------
    # Official Evaluation Record certification
    # ------------------------------------------------------------------

    def certify(
        self,
        *,
        context: EvaluationContext,
        planned_results: Sequence[PlannedKeyResult],
        aggregate_refs: Sequence[TypedContentRef],
        selected_record_mapping: SelectedRecordMapping,
        selection_evidence_ref: TypedContentRef | None = None,
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
        complete. The referenced ordinary Rollout Results become official by
        this certification; they are not copied or relabeled.
        """
        self._require_official_context(context)

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
            authority=self._name,
            evaluation_context_id=context.evaluation_context_id(),
            eval_config_hash=context.eval_config_hash,
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
        record_refs: Sequence[TypedContentRef],
        aggregate_refs: Sequence[TypedContentRef],
        objective_selection_refs: Sequence[TypedContentRef],
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
            authority=self._name,
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
