from __future__ import annotations

import json

from whetstone.envs.task_selection import TASK_SELECTION_SCHEMA
from whetstone.runner.task_split_manifest import load_task_split_manifest


def test_filesystem_adapter_loads_pr6_typed_manifest(tmp_path) -> None:
    path = tmp_path / "tasks.json"
    path.write_text(
        json.dumps(
            {
                "schema": TASK_SELECTION_SCHEMA,
                "pools": {
                    "ed1": {
                        "train": ["a"],
                        "val": ["b"],
                        "test": ["c"],
                    }
                },
            }
        )
    )

    manifest = load_task_split_manifest(path)
    roles = manifest.for_env("ed1")

    assert roles.internal_ids == ("a", "b")
    assert roles.official_ids == ("c",)
    assert len(roles.content_hash) == 64
