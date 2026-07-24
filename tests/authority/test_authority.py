"""Evaluation Authority: the named principal and official write path.

Proves that only an authority issues official Contexts / records / manifests,
that internal evaluation can never be relabeled or copied to official even when
identity hashes are byte-identical, and that certification computes the
completeness decision from the planned/present accounting.
"""

from __future__ import annotations

import pytest

from whetstone.authority import (
    CompletenessDecision,
    EvaluationAuthority,
    OfficialEvaluationRecord,
    PlannedKeyResult,
    RecordRevision,
    RelabelingRefusedError,
    SelectedRecordMapping,
    SelectedRecordMappingEntry,
    UnauthorizedOfficialWriteError,
)
from whetstone.graph.rollout import (
    EvaluationContext,
    EvaluationRole,
)

from .support import (
    EVAL_HASH,
    GRAPH_A,
    aggregate_ref,
    record_ref,
    result_ref,
    single_entry_mapping,
)


def _authority(name: str = "whetstone-official") -> EvaluationAuthority:
    return EvaluationAuthority(name=name)


def _official_context(
    authority: EvaluationAuthority,
) -> EvaluationContext:
    return authority.issue_official_context(
        eval_config_hash=EVAL_HASH,
        campaign="camp-1",
    )


def _internal_context() -> EvaluationContext:
    # Same ordinary Eval Config hash as the official run: identical config
    # identity, internal role.
    return EvaluationContext(
        eval_config_hash=EVAL_HASH,
        role=EvaluationRole.INTERNAL,
        campaign="camp-1",
    )


# ---------------------------------------------------------------------------
# Only the authority issues official Contexts
# ---------------------------------------------------------------------------


def test_authority_issues_official_context() -> None:
    authority = _authority()
    context = _official_context(authority)
    assert context.role is EvaluationRole.OFFICIAL
    assert context.authority == "whetstone-official"


def test_authority_must_be_named() -> None:
    with pytest.raises(ValueError, match="must be named"):
        EvaluationAuthority(name="")


# ---------------------------------------------------------------------------
# Relabeling refusal
# ---------------------------------------------------------------------------


def test_internal_evidence_is_refused_for_certification() -> None:
    authority = _authority()
    internal = _internal_context()
    planned = (PlannedKeyResult(planned_key="k0", result_ref=result_ref("d")),)
    with pytest.raises(RelabelingRefusedError):
        authority.certify(
            context=internal,
            planned_results=planned,
            aggregate_refs=(aggregate_ref("9"),),
            selected_record_mapping=single_entry_mapping(planned_keys=("k0",)),
        )


def test_identical_identity_hashes_still_refuse_relabeling() -> None:
    # The internal and official Contexts share the SAME ordinary Eval Config
    # hash; only the role and authority differ. The internal context's config
    # identity is byte-identical to the official one, but relabeling refused.
    authority = _authority()
    official = _official_context(authority)
    internal = _internal_context()
    assert internal.eval_config_hash == official.eval_config_hash
    # A shared measurement cell produces the SAME Rollout Key encoding under
    # each context except for the Evaluation Context id, proving identity
    # comparison is possible while relabeling is refused.
    assert internal.role is EvaluationRole.INTERNAL

    with pytest.raises(RelabelingRefusedError, match="never relabeling"):
        authority.certify(
            context=internal,
            planned_results=(
                PlannedKeyResult(planned_key="k0", result_ref=result_ref("d")),
            ),
            aggregate_refs=(aggregate_ref("9"),),
            selected_record_mapping=single_entry_mapping(planned_keys=("k0",)),
        )


def test_wrong_authority_is_refused() -> None:
    minter = _authority("authority-A")
    other = _authority("authority-B")
    context = _official_context(minter)  # names authority-A
    with pytest.raises(UnauthorizedOfficialWriteError):
        other.certify(
            context=context,
            planned_results=(
                PlannedKeyResult(planned_key="k0", result_ref=result_ref("d")),
            ),
            aggregate_refs=(aggregate_ref("9"),),
            selected_record_mapping=single_entry_mapping(planned_keys=("k0",)),
        )


# ---------------------------------------------------------------------------
# Certification + completeness accounting
# ---------------------------------------------------------------------------


def test_certify_complete_evidence() -> None:
    authority = _authority()
    context = _official_context(authority)
    planned = (
        PlannedKeyResult(planned_key="k0", result_ref=result_ref("d")),
        PlannedKeyResult(planned_key="k1", result_ref=result_ref("e")),
    )
    record = authority.certify(
        context=context,
        planned_results=planned,
        aggregate_refs=(aggregate_ref("9"),),
        selected_record_mapping=single_entry_mapping(
            planned_keys=("k0", "k1")
        ),
        selection_evidence_ref=result_ref("f"),
    )
    assert record.completeness.complete
    assert record.completeness.certified
    assert record.completeness.present_count == 2
    assert record.completeness.missing_count == 0
    assert record.authority == "whetstone-official"


def test_certify_refuses_when_incomplete() -> None:
    authority = _authority()
    context = _official_context(authority)
    planned = (
        PlannedKeyResult(planned_key="k0", result_ref=result_ref("d")),
        PlannedKeyResult(planned_key="k1", result_ref=None),  # missing
    )
    record = authority.certify(
        context=context,
        planned_results=planned,
        aggregate_refs=(aggregate_ref("9"),),
        selected_record_mapping=single_entry_mapping(
            planned_keys=("k0", "k1"),
            result_keys=("k0",),  # k1 missing
        ),
    )
    # An incomplete evaluation is uncertified; the missing row stays visible.
    assert not record.completeness.complete
    assert not record.completeness.certified
    assert record.completeness.missing_count == 1


def test_official_record_is_immutable() -> None:
    authority = _authority()
    context = _official_context(authority)
    record = authority.certify(
        context=context,
        planned_results=(
            PlannedKeyResult(planned_key="k0", result_ref=result_ref("d")),
        ),
        aggregate_refs=(aggregate_ref("9"),),
        selected_record_mapping=single_entry_mapping(planned_keys=("k0",)),
        selection_evidence_ref=result_ref("f"),
    )
    with pytest.raises((TypeError, ValueError)):
        record.authority = "someone-else"  # type: ignore


def test_official_record_certifies_ordinary_results_no_new_role() -> None:
    # The record references ORDINARY Rollout Result refs (the whetstone
    # rollout_result schema); certification introduces no distinct result type.
    authority = _authority()
    context = _official_context(authority)
    record = authority.certify(
        context=context,
        planned_results=(
            PlannedKeyResult(planned_key="k0", result_ref=result_ref("d")),
        ),
        aggregate_refs=(aggregate_ref("9"),),
        selected_record_mapping=single_entry_mapping(planned_keys=("k0",)),
        selection_evidence_ref=result_ref("f"),
    )
    assert isinstance(record, OfficialEvaluationRecord)
    ref = record.planned_results[0].result_ref
    assert ref is not None
    assert ref.schema_name == "whetstone.rollout_result"


def test_official_record_carries_immutable_revision_chain() -> None:
    # A record is immutable; a "revision" is a new record naming its
    # predecessor, captured as an auditable revision entry.
    authority = _authority()
    context = _official_context(authority)
    record = authority.certify(
        context=context,
        planned_results=(
            PlannedKeyResult(planned_key="k0", result_ref=result_ref("d")),
        ),
        aggregate_refs=(aggregate_ref("9"),),
        selected_record_mapping=single_entry_mapping(planned_keys=("k0",)),
        selection_evidence_ref=result_ref("f"),
        revisions=(
            RecordRevision(
                ordinal=1,
                supersedes_ref=result_ref("a"),
                reason="re-certified after dependency lock update",
            ),
        ),
    )
    assert len(record.revisions) == 1
    assert record.revisions[0].ordinal == 1
    assert record.revisions[0].reason.startswith("re-certified")


# ---------------------------------------------------------------------------
# The ordered mapping is reconciled against the actual present/missing
# accounting (result_key_set must be a subset of the present planned keys).
# ---------------------------------------------------------------------------


def _record(
    *,
    planned: tuple[PlannedKeyResult, ...],
    mapping: SelectedRecordMapping,
    evaluation_context_id: str = "e" * 64,
    present_count: int | None = None,
    certified: bool = False,
) -> OfficialEvaluationRecord:
    present = (
        sum(1 for p in planned if p.is_present)
        if present_count is None
        else present_count
    )
    missing = len(planned) - present
    return OfficialEvaluationRecord(
        authority="whetstone-official",
        evaluation_context_id=evaluation_context_id,
        eval_config_hash=EVAL_HASH,
        planned_results=planned,
        aggregate_refs=(aggregate_ref("9"),),
        completeness=CompletenessDecision(
            planned_count=len(planned),
            present_count=present,
            missing_count=missing,
            complete=missing == 0,
            certified=certified,
        ),
        selection_evidence_ref=result_ref("f") if certified else None,
        selected_record_mapping=mapping,
    )


def test_mapping_cannot_attribute_result_to_a_missing_planned_key() -> None:
    # k1 is missing while the mapping attributes a result to it, contradicting
    # the record's present/missing accounting.
    planned = (
        PlannedKeyResult(planned_key="k0", result_ref=result_ref("d")),
        PlannedKeyResult(planned_key="k1", result_ref=None),  # missing
    )
    bad_mapping = SelectedRecordMapping(
        entries=(
            SelectedRecordMappingEntry(
                record_ref=record_ref("1"),
                graph_hash=GRAPH_A,
                planned_key_set=("k0", "k1"),
                result_key_set=("k0", "k1"),  # attributes result to missing k1
                aggregate_ref=aggregate_ref("9"),
            ),
        )
    )
    with pytest.raises(ValueError, match="accounts as missing"):
        _record(planned=planned, mapping=bad_mapping)


def test_mapping_result_keys_matching_present_set_is_accepted() -> None:
    # A result_key_set reflecting the actual present keys is accepted.
    planned = (
        PlannedKeyResult(planned_key="k0", result_ref=result_ref("d")),
        PlannedKeyResult(planned_key="k1", result_ref=None),
    )
    ok_mapping = SelectedRecordMapping(
        entries=(
            SelectedRecordMappingEntry(
                record_ref=record_ref("1"),
                graph_hash=GRAPH_A,
                planned_key_set=("k0", "k1"),
                result_key_set=("k0",),
                aggregate_ref=aggregate_ref("9"),
            ),
        )
    )
    record = _record(planned=planned, mapping=ok_mapping)
    assert record.completeness.missing_count == 1


def test_converged_entries_agree_with_record_present_set() -> None:
    # Two selected records converge on one graph_hash. Their shared
    # result_key_set must reconcile with the record-level present set: if it
    # names a missing key, both converged entries are refused together.
    planned = (
        PlannedKeyResult(planned_key="k0", result_ref=result_ref("d")),
        PlannedKeyResult(planned_key="k1", result_ref=None),  # missing
    )
    converged = SelectedRecordMapping(
        entries=(
            SelectedRecordMappingEntry(
                record_ref=record_ref("1"),
                graph_hash=GRAPH_A,
                planned_key_set=("k0", "k1"),
                result_key_set=("k0", "k1"),
                aggregate_ref=aggregate_ref("9"),
            ),
            SelectedRecordMappingEntry(
                record_ref=record_ref("2"),
                graph_hash=GRAPH_A,
                planned_key_set=("k0", "k1"),
                result_key_set=("k0", "k1"),
                aggregate_ref=aggregate_ref("9"),
            ),
        )
    )
    with pytest.raises(ValueError, match="accounts as missing"):
        _record(planned=planned, mapping=converged)


# ---------------------------------------------------------------------------
# The evaluation_context_id is validated as a full Identity Hash, so a forged
# non-hash context id cannot be stamped onto an Official Evaluation Record via
# the direct constructor path.
# ---------------------------------------------------------------------------


def test_forged_non_hash_context_id_is_refused() -> None:
    planned = (PlannedKeyResult(planned_key="k0", result_ref=result_ref("d")),)
    with pytest.raises(ValueError, match="context_id must be a full"):
        _record(
            planned=planned,
            mapping=single_entry_mapping(planned_keys=("k0",)),
            evaluation_context_id="forged-ctx-id",
        )


def test_authority_issued_context_id_is_a_full_hash() -> None:
    # The authority path produces a context id that is a full 64-char identity
    # hash, so the record validator accepts it (the funnel and the validator
    # agree).
    authority = _authority()
    context = _official_context(authority)
    ctx_id = context.evaluation_context_id()
    assert len(ctx_id) == 64
    record = authority.certify(
        context=context,
        planned_results=(
            PlannedKeyResult(planned_key="k0", result_ref=result_ref("d")),
        ),
        aggregate_refs=(aggregate_ref("9"),),
        selected_record_mapping=single_entry_mapping(planned_keys=("k0",)),
        selection_evidence_ref=result_ref("f"),
    )
    assert record.evaluation_context_id == ctx_id
