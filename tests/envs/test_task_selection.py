from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from tests.envs.support import synthetic_ed1_tasks
from whetstone.envs.d1 import build_d1_experiment
from whetstone.envs.ed1 import build_ed1_experiment
from whetstone.envs.task_selection import (
    TASK_SELECTION_SCHEMA,
    TaskRoleSelectionMethod,
    TaskSplitManifestError,
    TaskSplitRole,
    parse_task_split_manifest,
    resolve_manifest_split,
)

_COPRO_MANIFEST = (
    Path(__file__).parents[2]
    / "src/whetstone/optimization/copro/humaneval_copro_challenge_v1.json"
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
    manifest = parse_task_split_manifest(_manifest())

    selected = manifest.select_role(env="ed1", role=role)

    assert selected.manifest_content_hash == manifest.content_hash
    assert selected.pool_key == "ed1"
    assert selected.role is role
    assert selected.task_ids == expected
    assert selected.selection_method is TaskRoleSelectionMethod.FULL_ROLE
    assert selected.eligible_pool_count == len(expected)


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


@pytest.mark.parametrize(
    ("role_a", "role_b"),
    [("train", "test"), ("val", "test"), ("train", "val")],
)
def test_cross_role_duplicate_ids_are_rejected(
    role_a: str, role_b: str
) -> None:
    payload = _manifest()
    pools = cast(dict[str, object], payload["pools"])
    ed1 = cast(dict[str, object], pools["ed1"])
    shared = cast(list[str], ed1[role_a])[0]
    ed1[role_b] = [*cast(list[str], ed1[role_b]), shared]
    with pytest.raises(TaskSplitManifestError, match="disjoint"):
        parse_task_split_manifest(payload)


def test_train_val_test_roles_stay_disjoint_end_to_end() -> None:
    roles = parse_task_split_manifest(_manifest()).for_env("ed1")
    assert not set(roles.internal_ids) & set(roles.official_ids)


def test_frozen_copro_challenge_manifest_matches_reference_metadata() -> None:
    payload = json.loads(_COPRO_MANIFEST.read_text())
    manifest = parse_task_split_manifest(payload)
    roles = manifest.for_env("ed1")
    rates = payload["selection"]["historical_pass_rates"]

    assert tuple(
        map(len, (roles.train_ids, roles.val_ids, roles.test_ids))
    ) == (
        46,
        15,
        15,
    )
    assert set(rates) == roles.all_role_ids()
    assert all(0.0 < value < 1.0 for value in rates.values())
    assert sum(value < 0.5 for value in rates.values()) == 4
    assert sum(0.5 <= value < 0.75 for value in rates.values()) == 20
    assert sum(0.75 <= value < 0.9 for value in rates.values()) == 31
    assert sum(0.9 <= value < 1.0 for value in rates.values()) == 21


def test_copro_probe_selects_five_worst_eligible_train_tasks() -> None:
    manifest = parse_task_split_manifest(_COPRO_MANIFEST.read_bytes())
    excluded = (
        "HumanEval/39",
        "HumanEval/113",
        "HumanEval/116",
        "HumanEval/149",
        "HumanEval/162",
    )

    selection = manifest.select_lowest_historical_pass_rate(
        env="ed1",
        role=TaskSplitRole.TRAIN,
        count=5,
        excluded_task_ids=excluded,
    )

    assert selection.selection_method is (
        TaskRoleSelectionMethod.LOWEST_HISTORICAL_PASS_RATE
    )
    assert selection.task_ids == (
        "HumanEval/32",
        "HumanEval/163",
        "HumanEval/160",
        "HumanEval/124",
        "HumanEval/132",
    )
    assert selection.historical_pass_rates == pytest.approx(
        (0.3636363636, 0.4, 0.5, 0.5384615385, 0.5555555556)
    )
    assert selection.source_role_count == 46
    assert selection.eligible_pool_count == 43
    assert selection.excluded_task_ids == excluded
