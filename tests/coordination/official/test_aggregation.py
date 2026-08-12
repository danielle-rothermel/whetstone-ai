from __future__ import annotations

import pytest
from dr_store import ObjectReference

from whetstone.coordination.official import (
    MissingPlannedKeysError,
    OfficialFailurePolicy,
    account_planned_keys,
)

from .support import full_hash

GENERATION_RESULT_SCHEMA = "whetstone.generation_result"


def _resolver(bound: dict[str, str]):

    def resolve(key: str) -> ObjectReference | None:
        char = bound.get(key)
        if char is None:
            return None
        return ObjectReference(
            schema=GENERATION_RESULT_SCHEMA, content_hash=full_hash(char)
        )

    return resolve


def test_every_planned_key_is_accounted_for() -> None:
    planned = ["k0", "k1", "k2"]
    account = account_planned_keys(
        planned_keys=planned,
        resolve=_resolver({"k0": "1", "k1": "2", "k2": "3"}),
    )
    assert account.planned_count == 3
    assert account.present_count == 3
    assert account.missing_count == 0
    assert account.complete
    assert [p.planned_key for p in account.planned_results] == planned


def test_missing_rows_are_visible_never_dropped() -> None:
    planned = ["k0", "k1", "k2"]
    account = account_planned_keys(
        planned_keys=planned,
        resolve=_resolver({"k0": "1", "k2": "3"}),
        policy=OfficialFailurePolicy.RECORD_MISSING,
    )
    assert account.planned_count == 3
    assert [p.planned_key for p in account.planned_results] == planned
    assert account.missing_keys == ("k1",)
    assert account.missing_count == 1
    assert not account.complete
    missing_rows = [p for p in account.planned_results if not p.is_present]
    assert [p.planned_key for p in missing_rows] == ["k1"]
    assert missing_rows[0].result_ref is None


def test_strict_raising_policy_surfaces_full_missing_set() -> None:
    planned = ["k0", "k1", "k2"]
    with pytest.raises(MissingPlannedKeysError) as exc:
        account_planned_keys(
            planned_keys=planned,
            resolve=_resolver({"k0": "1"}),
            policy=OfficialFailurePolicy.STRICT,
            raise_on_missing=True,
        )
    assert set(exc.value.missing) == {"k1", "k2"}


def test_duplicate_planned_keys_rejected() -> None:
    with pytest.raises(ValueError, match="unique"):
        account_planned_keys(
            planned_keys=["k0", "k0"],
            resolve=_resolver({"k0": "1"}),
        )


def test_account_requires_at_least_one_key() -> None:
    with pytest.raises(ValueError, match=">=1 planned key"):
        account_planned_keys(planned_keys=[], resolve=_resolver({}))


def test_account_feeds_certification_with_missing_visible() -> None:
    from whetstone.coordination.official import EvaluationAuthority

    from .support import aggregate_ref, eval_config_ref, single_entry_mapping

    planned = ["k0", "k1"]
    account = account_planned_keys(
        planned_keys=planned,
        resolve=_resolver({"k0": "1"}),
        policy=OfficialFailurePolicy.RECORD_MISSING,
    )
    authority = EvaluationAuthority(name="whetstone-official")
    binding = authority.issue_official_binding(
        eval_config=eval_config_ref(), campaign="camp-1"
    )
    record = authority.certify(
        evaluation_binding=binding,
        planned_results=account.planned_results,
        aggregate_refs=(aggregate_ref("9"),),
        selected_record_mapping=single_entry_mapping(
            planned_keys=("k0", "k1"),
            result_keys=("k0",),
        ),
    )
    assert record.completeness.planned_count == 2
    assert record.completeness.missing_count == 1
    assert not record.completeness.certified
    missing = [p for p in record.planned_results if not p.is_present]
    assert [p.planned_key for p in missing] == ["k1"]
