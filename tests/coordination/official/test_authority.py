from __future__ import annotations

import pytest

from whetstone.coordination.official import (
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
from whetstone.core.roles import EvaluationRole
from whetstone.experiment.binding import (
    EVALUATION_BINDING_SCHEMA_VERSION,
    EvaluationBinding,
)

from .support import (
    GRAPH_A,
    aggregate_ref,
    eval_config_ref,
    record_ref,
    result_ref,
    single_entry_mapping,
)


def _authority(name: str = "whetstone-official") -> EvaluationAuthority:
    return EvaluationAuthority(name=name)


def _official_binding(
    authority: EvaluationAuthority,
) -> EvaluationBinding:
    return authority.issue_official_binding(
        eval_config=eval_config_ref(),
        campaign="camp-1",
    )


def _internal_binding() -> EvaluationBinding:
    return EvaluationBinding(
        schema_version=EVALUATION_BINDING_SCHEMA_VERSION,
        eval_config=eval_config_ref(),
        role=EvaluationRole.INTERNAL,
        campaign="camp-1",
    )


def test_authority_issues_official_binding() -> None:
    authority = _authority()
    binding = _official_binding(authority)
    assert binding.role is EvaluationRole.OFFICIAL
    assert binding.authority_principal == "whetstone-official"


def test_authority_must_be_named() -> None:
    with pytest.raises(ValueError, match="must be named"):
        EvaluationAuthority(name="")


def test_internal_evidence_is_refused_for_certification() -> None:
    authority = _authority()
    internal = _internal_binding()
    planned = (PlannedKeyResult(planned_key="k0", result_ref=result_ref("d")),)
    with pytest.raises(RelabelingRefusedError):
        authority.certify(
            evaluation_binding=internal,
            planned_results=planned,
            aggregate_refs=(aggregate_ref("9"),),
            selected_record_mapping=single_entry_mapping(planned_keys=("k0",)),
        )


def test_identical_identity_hashes_still_refuse_relabeling() -> None:
    authority = _authority()
    official = _official_binding(authority)
    internal = _internal_binding()
    assert internal.eval_config == official.eval_config
    assert internal.role is EvaluationRole.INTERNAL

    with pytest.raises(RelabelingRefusedError, match="never relabeling"):
        authority.certify(
            evaluation_binding=internal,
            planned_results=(
                PlannedKeyResult(planned_key="k0", result_ref=result_ref("d")),
            ),
            aggregate_refs=(aggregate_ref("9"),),
            selected_record_mapping=single_entry_mapping(planned_keys=("k0",)),
        )


def test_wrong_authority_is_refused() -> None:
    minter = _authority("authority-A")
    other = _authority("authority-B")
    binding = _official_binding(minter)
    with pytest.raises(UnauthorizedOfficialWriteError):
        other.certify(
            evaluation_binding=binding,
            planned_results=(
                PlannedKeyResult(planned_key="k0", result_ref=result_ref("d")),
            ),
            aggregate_refs=(aggregate_ref("9"),),
            selected_record_mapping=single_entry_mapping(planned_keys=("k0",)),
        )


def test_certify_complete_evidence() -> None:
    authority = _authority()
    binding = _official_binding(authority)
    planned = (
        PlannedKeyResult(planned_key="k0", result_ref=result_ref("d")),
        PlannedKeyResult(planned_key="k1", result_ref=result_ref("e")),
    )
    record = authority.certify(
        evaluation_binding=binding,
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
    binding = _official_binding(authority)
    planned = (
        PlannedKeyResult(planned_key="k0", result_ref=result_ref("d")),
        PlannedKeyResult(planned_key="k1", result_ref=None),
    )
    record = authority.certify(
        evaluation_binding=binding,
        planned_results=planned,
        aggregate_refs=(aggregate_ref("9"),),
        selected_record_mapping=single_entry_mapping(
            planned_keys=("k0", "k1"),
            result_keys=("k0",),
        ),
    )
    assert not record.completeness.complete
    assert not record.completeness.certified
    assert record.completeness.missing_count == 1


def test_official_record_is_immutable() -> None:
    authority = _authority()
    binding = _official_binding(authority)
    record = authority.certify(
        evaluation_binding=binding,
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
    authority = _authority()
    binding = _official_binding(authority)
    record = authority.certify(
        evaluation_binding=binding,
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
    assert ref.schema_name == "whetstone.generation_result"


def test_official_record_carries_immutable_revision_chain() -> None:
    authority = _authority()
    binding = _official_binding(authority)
    record = authority.certify(
        evaluation_binding=binding,
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


def _record(
    *,
    planned: tuple[PlannedKeyResult, ...],
    mapping: SelectedRecordMapping,
    evaluation_binding_id: str = "e" * 64,
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
        evaluation_binding_id=evaluation_binding_id,
        eval_config=eval_config_ref(),
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
    planned = (
        PlannedKeyResult(planned_key="k0", result_ref=result_ref("d")),
        PlannedKeyResult(planned_key="k1", result_ref=None),
    )
    bad_mapping = SelectedRecordMapping(
        entries=(
            SelectedRecordMappingEntry(
                record_ref=record_ref("1"),
                graph_hash=GRAPH_A,
                planned_key_set=("k0", "k1"),
                result_key_set=("k0", "k1"),
                aggregate_ref=aggregate_ref("9"),
            ),
        )
    )
    with pytest.raises(ValueError, match="accounts as missing"):
        _record(planned=planned, mapping=bad_mapping)


def test_mapping_result_keys_matching_present_set_is_accepted() -> None:
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


def test_mapping_cannot_omit_a_present_planned_key() -> None:
    planned = (
        PlannedKeyResult(planned_key="k0", result_ref=result_ref("d")),
        PlannedKeyResult(planned_key="k1", result_ref=result_ref("e")),
    )
    under_attributing = SelectedRecordMapping(
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
    with pytest.raises(ValueError, match="present keys the mapping omits"):
        _record(planned=planned, mapping=under_attributing)


def test_mapping_entry_need_not_cover_planned_keys_outside_its_set() -> None:
    planned = (
        PlannedKeyResult(planned_key="k0", result_ref=result_ref("d")),
        PlannedKeyResult(planned_key="k1", result_ref=result_ref("e")),
    )
    scoped = SelectedRecordMapping(
        entries=(
            SelectedRecordMappingEntry(
                record_ref=record_ref("1"),
                graph_hash=GRAPH_A,
                planned_key_set=("k0",),
                result_key_set=("k0",),
                aggregate_ref=aggregate_ref("9"),
            ),
        )
    )
    record = _record(planned=planned, mapping=scoped)
    assert record.completeness.present_count == 2


def test_converged_entries_agree_with_record_present_set() -> None:
    planned = (
        PlannedKeyResult(planned_key="k0", result_ref=result_ref("d")),
        PlannedKeyResult(planned_key="k1", result_ref=None),
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


def test_forged_non_hash_binding_id_is_refused() -> None:
    planned = (PlannedKeyResult(planned_key="k0", result_ref=result_ref("d")),)
    with pytest.raises(ValueError, match="binding_id must be a full"):
        _record(
            planned=planned,
            mapping=single_entry_mapping(planned_keys=("k0",)),
            evaluation_binding_id="forged-binding-id",
        )


def test_authority_issued_binding_id_is_a_full_hash() -> None:
    authority = _authority()
    binding = _official_binding(authority)
    binding_id = binding.identity_hash()
    assert len(binding_id) == 64
    record = authority.certify(
        evaluation_binding=binding,
        planned_results=(
            PlannedKeyResult(planned_key="k0", result_ref=result_ref("d")),
        ),
        aggregate_refs=(aggregate_ref("9"),),
        selected_record_mapping=single_entry_mapping(planned_keys=("k0",)),
        selection_evidence_ref=result_ref("f"),
    )
    assert record.evaluation_binding_id == binding_id


def test_duplicate_planned_keys_are_refused() -> None:
    duplicated = (
        PlannedKeyResult(planned_key="k0", result_ref=result_ref("d")),
        PlannedKeyResult(planned_key="k0", result_ref=result_ref("d")),
    )
    with pytest.raises(ValueError, match="unique planned_key"):
        _record(
            planned=duplicated,
            mapping=single_entry_mapping(planned_keys=("k0",)),
            certified=True,
        )
