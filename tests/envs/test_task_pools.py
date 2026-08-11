from __future__ import annotations

import pytest

from tests.envs.support import synthetic_ed1_tasks
from tests.experiment.task_selection.support import manifest_payload
from whetstone.envs.d1 import build_d1_experiment
from whetstone.envs.ed1 import build_ed1_experiment
from whetstone.envs.task_pools import roles_for_env, select_role_for_env
from whetstone.experiment.task_selection import (
    TaskSplitManifestError,
    TaskSplitRole,
    parse_task_split_manifest,
    resolve_manifest_split,
)


def _ids(split) -> tuple[str, ...]:
    return tuple(str(instance.id) for instance in split.instances)


def test_roles_for_env_maps_ed1_and_d1_pools() -> None:
    manifest = parse_task_split_manifest(manifest_payload())
    ed1 = roles_for_env(manifest, "ed1")
    d1 = roles_for_env(manifest, "d1")
    assert ed1.pool_key == "ed1"
    assert d1.pool_key == "d1"
    assert ed1.internal_ids == (
        "Synthetic/0",
        "Synthetic/1",
        "Synthetic/2",
    )
    assert d1.official_ids == ("Synthetic/2", "Synthetic/3")


@pytest.mark.parametrize("env", ["ed1m", "c18"])
def test_roles_for_env_refuses_inapplicable_environment(env: str) -> None:
    manifest = parse_task_split_manifest(manifest_payload())
    with pytest.raises(TaskSplitManifestError):
        roles_for_env(manifest, env)


def test_select_role_for_env_preserves_manifest_order() -> None:
    manifest = parse_task_split_manifest(manifest_payload())
    selected = select_role_for_env(
        manifest,
        env="ed1",
        role=TaskSplitRole.TRAIN,
    )
    assert selected.task_ids == ("Synthetic/0", "Synthetic/1")
    assert selected.pool_key == "ed1"


@pytest.mark.parametrize("family", ["ed1", "d1"])
def test_family_builders_apply_role_membership_and_manifest_identity(
    family: str,
) -> None:
    tasks = synthetic_ed1_tasks(5)
    manifest = parse_task_split_manifest(manifest_payload())
    roles = roles_for_env(manifest, family)
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


def test_train_val_test_roles_stay_disjoint_end_to_end() -> None:
    roles = roles_for_env(parse_task_split_manifest(manifest_payload()), "ed1")
    assert not set(roles.internal_ids) & set(roles.official_ids)


def test_env_resolution_preserves_manifest_order_and_caps_test() -> None:
    tasks = synthetic_ed1_tasks(5)
    roles = roles_for_env(parse_task_split_manifest(manifest_payload()), "ed1")
    resolved = resolve_manifest_split(
        roles=roles,
        items=tasks,
        id_of=lambda task: str(task.instance.id),
        official_n=1,
    )
    assert tuple(task.instance.id for task in resolved.official) == (
        "Synthetic/3",
    )
