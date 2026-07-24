"""Identity-stability pins for the per-call provenance work (task 26).

Task 26 adds RECORDING-only provenance fields to the ledger/rollout/partial/
screen/spend artifacts. It must NOT perturb config-identity computation --
``graph_hash`` / ``eval_config_hash`` must stay byte-identical to the base
commit for any existing configuration. These pins were computed at the base
commit (``fcad0ca``) BEFORE any recording change; if a provenance change ever
touches identity, one of these assertions fails loudly (and the change must be
reworked or skipped, per the task-26 hard constraint).

Two representative configurations are pinned: the d1 direct-optimization env
(a frozen input arm folded into both hashes) and the ed1 enc-dec env (a budget
ratio folded into the graph hash). Between them they exercise the
graph-identity and eval-config-identity computation an ordinary cell relies on.

Task 29 adds ``--task-split-manifest`` (role-true train/val/test splits). The
NO-manifest pins above MUST stay byte-identical (the manifest fold is
conditional). A SECOND pair of pins covers a WITH-manifest config: the graph
hash is UNCHANGED (the manifest is a split/eval-config concern, not a rollout
concern) while the internal + official eval hashes are DISTINCT (the
manifest's content hash + pool + role sets fold into each split's Task Set
identity). These manifest pins are computed against an in-test fixture so they
are reproducible without the run artifact.
"""

from __future__ import annotations

import json
from pathlib import Path

from whetstone.envs.d1 import build_d1_experiment
from whetstone.envs.ed1 import build_ed1_experiment, load_ed1_tasks
from whetstone.runner.task_split_manifest import load_task_split_manifest

# Pins computed at base commit fcad0ca (pre-task-26). See module docstring.
_D1_GRAPH_HASH = (
    "5be2d083ac1c056082ca06be19dce5755d77742f1e64523c4bae8135cf29cb01"
)
_D1_EVAL_HASH = (
    "fce39eeea8c39c44169cdd00f0c759a9fdd8db25aadf50f9ac4cbc0d8f515cba"
)
_ED1_GRAPH_HASH = (
    "f85b10faff7f008f7b393aad804257b14f80d6bf9298ee2db2436650be27d97f"
)
_ED1_EVAL_HASH = (
    "3cac7ea18825b51fd5763ffdf8d6c7e091db6e0970e95bf331b76c54c741d1c3"
)

# Task-29 with-manifest pins (fixture manifest below). The graph hashes MATCH
# the no-manifest pins; the eval hashes are DISTINCT (manifest folds in).
_TSM_FIXTURE = {
    "schema": "whetstone.run.task_selection/v1",
    "pools": {
        "ed1": {
            "train": ["HumanEval/0", "HumanEval/1"],
            "val": ["HumanEval/2"],
            "test": ["HumanEval/3", "HumanEval/4"],
        },
        "d1": {
            "train": ["HumanEval/0"],
            "val": ["HumanEval/1"],
            "test": ["HumanEval/2", "HumanEval/3"],
        },
    },
}
_ED1_TSM_INTERNAL_HASH = (
    "9834a10f8dfff122ac0ba5a63cdf89a5d5926df3bbff849704cbdcd7ab1a769f"
)
_ED1_TSM_OFFICIAL_HASH = (
    "75bc2c05efffd15ea4c87633adfde71ecc7aeff517cf824abe98d3c3901a4840"
)
_D1_TSM_INTERNAL_HASH = (
    "d0eb947ada5e744753d9428ed590610c64ac1241b228875c85e3ec1363efd46b"
)
_D1_TSM_OFFICIAL_HASH = (
    "8931aa87bcb97dff4d71e0e2104fbb2737b718b452b4aed791dcaf0a95f23e51"
)


def test_d1_config_identity_is_byte_stable() -> None:
    tasks = load_ed1_tasks(prefer_snapshot=True, limit=4)
    exp = build_d1_experiment(input_arm="original", tasks=tasks)
    assert exp.rollout_definition.graph_hash == _D1_GRAPH_HASH
    official = exp.eval_configs.official.eval_config
    assert official.config_identity_hash == _D1_EVAL_HASH


def test_ed1_config_identity_is_byte_stable() -> None:
    tasks = load_ed1_tasks(prefer_snapshot=True, limit=6)
    exp = build_ed1_experiment(tasks=tasks, budget_ratio=0.1)
    assert exp.rollout_definition.graph_hash == _ED1_GRAPH_HASH
    official = exp.eval_configs.official.eval_config
    assert official.config_identity_hash == _ED1_EVAL_HASH


def _fixture_manifest(tmp_path: Path):
    p = tmp_path / "manifest-v1.json"
    p.write_text(json.dumps(_TSM_FIXTURE))
    return load_task_split_manifest(p)


def test_ed1_manifest_config_identity_is_byte_stable(tmp_path: Path) -> None:
    manifest = _fixture_manifest(tmp_path)
    tasks = load_ed1_tasks(prefer_snapshot=True, limit=6)
    exp = build_ed1_experiment(
        tasks=tasks, budget_ratio=0.1, split_manifest=manifest.for_env("ed1")
    )
    # The manifest does NOT touch the rollout graph (split-only concern).
    assert exp.rollout_definition.graph_hash == _ED1_GRAPH_HASH
    internal = exp.eval_configs.internal.eval_config
    official = exp.eval_configs.official.eval_config
    assert internal.config_identity_hash == _ED1_TSM_INTERNAL_HASH
    assert official.config_identity_hash == _ED1_TSM_OFFICIAL_HASH
    # The manifest official hash is DISTINCT from the no-manifest ed1 pin.
    assert official.config_identity_hash != _ED1_EVAL_HASH


def test_d1_manifest_config_identity_is_byte_stable(tmp_path: Path) -> None:
    manifest = _fixture_manifest(tmp_path)
    tasks = load_ed1_tasks(prefer_snapshot=True, limit=6)
    exp = build_d1_experiment(
        input_arm="original", tasks=tasks,
        split_manifest=manifest.for_env("d1"),
    )
    assert exp.rollout_definition.graph_hash == _D1_GRAPH_HASH
    internal = exp.eval_configs.internal.eval_config
    official = exp.eval_configs.official.eval_config
    assert internal.config_identity_hash == _D1_TSM_INTERNAL_HASH
    assert official.config_identity_hash == _D1_TSM_OFFICIAL_HASH
    assert official.config_identity_hash != _D1_EVAL_HASH
