from __future__ import annotations

import json
from typing import cast

import pytest

from tests.experiment.task_selection.support import manifest_payload
from whetstone.experiment.task_selection import (
    TASK_SELECTION_SCHEMA,
    TaskRoleSelectionMethod,
    TaskSplitManifestError,
    TaskSplitRole,
    parse_task_split_manifest,
)


def test_parse_accepts_json_bytes_and_hashes_canonical_content() -> None:
    payload = manifest_payload()
    direct = parse_task_split_manifest(payload)
    encoded = parse_task_split_manifest(json.dumps(payload).encode())
    assert direct.content_hash == encoded.content_hash
    changed = parse_task_split_manifest(
        manifest_payload(encdec_test=("Synthetic/3",))
    )
    assert changed.content_hash != direct.content_hash


@pytest.mark.parametrize(
    "payload",
    [
        b'{"schema":"first","schema":"second"}',
        b'{"value":NaN}',
        b'{"value":"\xff"}',
    ],
)
def test_parse_rejects_non_strict_json_bytes(payload: bytes) -> None:
    with pytest.raises(TaskSplitManifestError, match="not valid JSON"):
        parse_task_split_manifest(payload)


@pytest.mark.parametrize(
    "payload, match",
    [
        (b"{bad", "valid JSON"),
        ({"schema": "other", "pools": {}}, "schema"),
        ({"schema": TASK_SELECTION_SCHEMA}, "pools"),
    ],
)
def test_parse_rejects_invalid_boundaries(
    payload: object,
    match: str,
) -> None:
    with pytest.raises(TaskSplitManifestError, match=match):
        parse_task_split_manifest(payload)


def test_pool_roles_are_train_then_val_and_test_exactly() -> None:
    manifest = parse_task_split_manifest(manifest_payload())
    encdec = manifest.pool_roles("encdec")
    direct = manifest.pool_roles("direct")
    assert encdec.internal_ids == (
        "Synthetic/0",
        "Synthetic/1",
        "Synthetic/2",
    )
    assert encdec.official_ids == ("Synthetic/3", "Synthetic/4")
    assert direct.internal_ids == ("Synthetic/0", "Synthetic/1")
    assert direct.official_ids == ("Synthetic/2", "Synthetic/3")


@pytest.mark.parametrize(
    ("role", "expected"),
    [
        (TaskSplitRole.TRAIN, ("Synthetic/0", "Synthetic/1")),
        (TaskSplitRole.VALIDATION, ("Synthetic/2",)),
        (TaskSplitRole.TEST, ("Synthetic/3", "Synthetic/4")),
    ],
)
def test_select_role_preserves_role_and_manifest_order(
    role: TaskSplitRole, expected: tuple[str, ...]
) -> None:
    manifest = parse_task_split_manifest(manifest_payload())

    selected = manifest.select_role(pool_key="encdec", role=role)

    assert selected.manifest_content_hash == manifest.content_hash
    assert selected.pool_key == "encdec"
    assert selected.role is role
    assert selected.task_ids == expected
    assert selected.selection_method is TaskRoleSelectionMethod.FULL_ROLE
    assert selected.eligible_pool_count == len(expected)


def test_pool_roles_rejects_unknown_pool_key() -> None:
    manifest = parse_task_split_manifest(manifest_payload())
    with pytest.raises(TaskSplitManifestError, match="no pool 'missing'"):
        manifest.pool_roles("missing")


def test_duplicate_role_ids_are_rejected() -> None:
    payload = manifest_payload()
    pools = cast(dict[str, object], payload["pools"])
    ed1 = cast(dict[str, object], pools["encdec"])
    ed1["train"] = ["Synthetic/0", "Synthetic/0"]
    with pytest.raises(TaskSplitManifestError, match="duplicate"):
        parse_task_split_manifest(payload)


@pytest.mark.parametrize(
    ("role_a", "role_b"),
    [("train", "test"), ("val", "test"), ("train", "val")],
)
def test_cross_role_duplicate_ids_are_rejected(
    role_a: str, role_b: str
) -> None:
    payload = manifest_payload()
    pools = cast(dict[str, object], payload["pools"])
    ed1 = cast(dict[str, object], pools["encdec"])
    shared = cast(list[str], ed1[role_a])[0]
    ed1[role_b] = [*cast(list[str], ed1[role_b]), shared]
    with pytest.raises(TaskSplitManifestError, match="disjoint"):
        parse_task_split_manifest(payload)


def test_select_lowest_historical_pass_rate_uses_manifest_metadata() -> None:
    payload = {
        **manifest_payload(),
        "selection": {
            "historical_pass_rates": {
                "Synthetic/0": 0.9,
                "Synthetic/1": 0.2,
                "Synthetic/2": 0.5,
                "Synthetic/3": 0.8,
                "Synthetic/4": 0.7,
            }
        },
    }
    manifest = parse_task_split_manifest(payload)

    selected = manifest.select_lowest_historical_pass_rate(
        pool_key="encdec",
        role=TaskSplitRole.TRAIN,
        count=1,
    )

    assert selected.task_ids == ("Synthetic/1",)
    assert selected.historical_pass_rates == (0.2,)
    assert selected.selection_method is (
        TaskRoleSelectionMethod.LOWEST_HISTORICAL_PASS_RATE
    )
