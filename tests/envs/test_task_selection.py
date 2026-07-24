"""Task-selection parsing, role resolution, and identity tests."""

from __future__ import annotations

import json
from typing import cast

import pytest

from tests.envs.support import synthetic_ed1_tasks
from whetstone.envs.d1 import build_d1_experiment
from whetstone.envs.ed1 import build_ed1_experiment
from whetstone.envs.task_selection import (
    TASK_SELECTION_SCHEMA,
    TaskSplitManifestError,
    parse_task_split_manifest,
    resolve_manifest_split,
)


def _manifest(
    *,
    ed1_test: tuple[str, ...] = ("Synthetic/3", "Synthetic/4"),
) -> dict[str, object]:
    return {
        "schema": TASK_SELECTION_SCHEMA,
        "seed": 7,
        "pools": {
            "ed1": {
                "arm": "encdec_naive",
                "train": ["Synthetic/0", "Synthetic/1"],
                "val": ["Synthetic/2"],
                "test": list(ed1_test),
            },
            "d1": {
                "arm": "direct_original",
                "train": ["Synthetic/0"],
                "val": ["Synthetic/1"],
                "test": ["Synthetic/2", "Synthetic/3"],
            },
        },
    }


def _ids(split) -> tuple[str, ...]:
    return tuple(str(instance.id) for instance in split.instances)


def test_parse_accepts_json_bytes_and_hashes_canonical_content() -> None:
    payload = _manifest()
    direct = parse_task_split_manifest(payload)
    encoded = parse_task_split_manifest(json.dumps(payload).encode())
    assert direct.content_hash == encoded.content_hash
    changed = parse_task_split_manifest(_manifest(ed1_test=("Synthetic/3",)))
    assert changed.content_hash != direct.content_hash


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


def test_roles_are_train_then_val_and_test_exactly() -> None:
    manifest = parse_task_split_manifest(_manifest())
    ed1 = manifest.for_env("ed1")
    d1 = manifest.for_env("d1")
    assert ed1.internal_ids == (
        "Synthetic/0",
        "Synthetic/1",
        "Synthetic/2",
    )
    assert ed1.official_ids == ("Synthetic/3", "Synthetic/4")
    assert d1.internal_ids == ("Synthetic/0", "Synthetic/1")
    assert d1.official_ids == ("Synthetic/2", "Synthetic/3")


@pytest.mark.parametrize("env", ["ed1m", "c18"])
def test_roles_refuse_inapplicable_environment(env: str) -> None:
    manifest = parse_task_split_manifest(_manifest())
    with pytest.raises(TaskSplitManifestError):
        manifest.for_env(env)


def test_resolution_preserves_manifest_order_and_caps_test_membership() -> (
    None
):
    tasks = synthetic_ed1_tasks(5)
    roles = parse_task_split_manifest(_manifest()).for_env("ed1")
    resolved = resolve_manifest_split(
        roles=roles,
        items=tasks,
        id_of=lambda task: str(task.instance.id),
        official_n=1,
    )
    assert tuple(task.instance.id for task in resolved.internal) == (
        "Synthetic/0",
        "Synthetic/1",
        "Synthetic/2",
    )
    assert tuple(task.instance.id for task in resolved.official) == (
        "Synthetic/3",
    )
    assert resolved.official_capped is not None
    assert resolved.manifest_tag.startswith("tsm:")


def test_resolution_refuses_unknown_ids() -> None:
    roles = parse_task_split_manifest(
        _manifest(ed1_test=("Synthetic/999",))
    ).for_env("ed1")
    with pytest.raises(TaskSplitManifestError, match="Synthetic/999"):
        resolve_manifest_split(
            roles=roles,
            items=synthetic_ed1_tasks(4),
            id_of=lambda task: str(task.instance.id),
        )


@pytest.mark.parametrize("family", ["ed1", "d1"])
def test_family_builders_apply_role_membership_and_manifest_identity(
    family: str,
) -> None:
    tasks = synthetic_ed1_tasks(5)
    roles = parse_task_split_manifest(_manifest()).for_env(family)
    builder = build_ed1_experiment if family == "ed1" else build_d1_experiment
    selected = builder(tasks=tasks, split_manifest=roles)
    plain = builder(
        tasks=tasks,
        internal_n=len(roles.internal_ids),
        official_n=len(roles.official_ids),
    )
    assert _ids(selected.eval_configs.internal) == roles.internal_ids
    assert _ids(selected.eval_configs.official) == roles.official_ids
    assert (
        selected.eval_configs.internal.eval_config.config_identity_hash
        != plain.eval_configs.internal.eval_config.config_identity_hash
    )


def test_duplicate_role_ids_are_rejected() -> None:
    payload = _manifest()
    pools = cast(dict[str, object], payload["pools"])
    ed1 = cast(dict[str, object], pools["ed1"])
    ed1["train"] = ["Synthetic/0", "Synthetic/0"]
    with pytest.raises(TaskSplitManifestError, match="duplicate"):
        parse_task_split_manifest(payload)
