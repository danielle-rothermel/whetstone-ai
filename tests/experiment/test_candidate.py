"""Candidate identity, exact-reference, and JSON boundary contracts."""

from __future__ import annotations

import pickle
from typing import Any

import pytest
from pydantic import ValidationError

from whetstone.core.identity import typed_ref_for_record
from whetstone.experiment.candidate import (
    CANDIDATE_IDENTITY_SCHEMA,
    CANDIDATE_IDENTITY_SCHEMA_VERSION,
    Candidate,
    CandidateRef,
    candidate_reference,
)

FULL_A = "a" * 64


def _candidate() -> Candidate:
    return Candidate(
        candidate_id="A",
        base_ref=typed_ref_for_record(
            "whetstone.test.candidate_base", {"label": "base"}
        ),
        payload={"user_prompt_template": "t", "fixed": "same"},
    )


def test_candidate_identity_contract_literals_are_pinned() -> None:
    record = _candidate()

    assert CANDIDATE_IDENTITY_SCHEMA == "whetstone.optimization_candidate"
    assert CANDIDATE_IDENTITY_SCHEMA_VERSION == 1
    assert record.identity_payload() == {
        "candidate_id": "A",
        "base_ref": {
            "schema_name": "whetstone.test.candidate_base",
            "content_hash": (
                "f276f3774e46a956d8edc3ac6845086a"
                "87e913ca1fbc099edf2ec65ce848b7df"
            ),
        },
        "payload": {"user_prompt_template": "t", "fixed": "same"},
    }
    assert (
        record.identity_hash()
        == "d13bd9c7dcb859cc7260591eed0f7ec4bbd6a296dccc934bbf7090ae3a9ebca3"
    )


def test_candidate_ref_round_trips_exact_json() -> None:
    ref = candidate_reference(_candidate())
    dumped = ref.model_dump(mode="json")

    assert CandidateRef.model_validate(dumped) == ref
    assert CandidateRef.model_validate_json(ref.model_dump_json()) == ref


def test_candidate_ref_rejects_tampered_record_ref() -> None:
    ref = candidate_reference(_candidate())
    payload = ref.model_dump(mode="json")
    payload["record_ref"]["schema_name"] = "whetstone.test.wrong"

    with pytest.raises(ValidationError, match=r"record_ref.*exact candidate"):
        CandidateRef.model_validate(payload)


def test_candidate_ref_rejects_tampered_identity_hash() -> None:
    ref = candidate_reference(_candidate())
    payload = ref.model_dump(mode="json")
    payload["identity_hash"] = FULL_A

    with pytest.raises(
        ValidationError, match=r"identity_hash.*exact candidate"
    ):
        CandidateRef.model_validate(payload)


def test_candidate_defensively_copies_and_deep_freezes_json() -> None:
    source: dict[str, Any] = {
        "nested": {"enabled": True},
        "items": [1, {"name": "first"}],
    }
    record = Candidate.model_validate(
        {
            **_candidate().model_dump(mode="json"),
            "payload": source,
        }
    )
    before_identity = record.identity_hash()
    before_ref = candidate_reference(record).record_ref

    source["nested"]["enabled"] = False
    source_items: Any = source["items"]
    source_items[1]["name"] = "changed"

    frozen_dump = record.model_dump(mode="json")["payload"]
    assert frozen_dump["nested"]["enabled"] is True
    assert frozen_dump["items"][1]["name"] == "first"
    assert record.identity_hash() == before_identity
    assert candidate_reference(record).record_ref == before_ref
    frozen_nested: Any = record.payload["nested"]
    frozen_items: Any = record.payload["items"]
    with pytest.raises(TypeError):
        frozen_nested["enabled"] = False
    with pytest.raises(TypeError):
        frozen_items[1]["name"] = "changed"
    with pytest.raises(AttributeError):
        record.payload._items = ()


def test_candidate_json_round_trip_is_exact() -> None:
    record = _candidate()
    dumped = record.model_dump(mode="json")

    assert dumped == {
        "candidate_id": "A",
        "base_ref": {
            "schema_name": "whetstone.test.candidate_base",
            "content_hash": (
                "f276f3774e46a956d8edc3ac6845086a"
                "87e913ca1fbc099edf2ec65ce848b7df"
            ),
        },
        "payload": {"user_prompt_template": "t", "fixed": "same"},
    }
    assert Candidate.model_validate(dumped) == record
    assert Candidate.model_validate_json(record.model_dump_json()) == record


def test_records_carrying_json_fields_survive_pickle_round_trips() -> None:
    record = _candidate()
    record = record.model_validate(
        {
            **record.model_dump(mode="json"),
            "payload": {"nested": {"enabled": True}, "items": [1, "two"]},
        }
    )

    restored = pickle.loads(pickle.dumps(record))

    assert restored == record
    assert restored.identity_hash() == record.identity_hash()
    assert restored.payload.to_json() == record.payload.to_json()


@pytest.mark.parametrize(
    "payload",
    [
        {"bad": object()},
        {1: "non-string key"},
        {"bad": float("nan")},
        {"bad": float("inf")},
    ],
)
def test_json_fields_reject_non_json_and_nonfinite_values(payload) -> None:
    with pytest.raises((TypeError, ValidationError, ValueError)):
        _candidate().model_validate(
            {
                **_candidate().model_dump(mode="json"),
                "payload": payload,
            }
        )
