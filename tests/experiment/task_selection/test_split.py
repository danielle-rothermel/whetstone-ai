from __future__ import annotations

import pytest

from tests.envs.support import synthetic_code_comp_tasks
from tests.experiment.task_selection.support import manifest_payload
from whetstone.experiment.task_selection import (
    TaskSplitManifestError,
    parse_task_split_manifest,
    resolve_manifest_split,
)


def test_resolution_preserves_manifest_order_and_caps_test_membership() -> (
    None
):
    tasks = synthetic_code_comp_tasks(5)
    roles = parse_task_split_manifest(manifest_payload()).pool_roles("encdec")
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
        manifest_payload(encdec_test=("Synthetic/999",))
    ).pool_roles("encdec")
    with pytest.raises(TaskSplitManifestError, match="Synthetic/999"):
        resolve_manifest_split(
            roles=roles,
            items=synthetic_code_comp_tasks(4),
            id_of=lambda task: str(task.instance.id),
        )
