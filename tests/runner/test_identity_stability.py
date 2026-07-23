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
"""

from __future__ import annotations

from whetstone.envs.d1 import build_d1_experiment
from whetstone.envs.ed1 import build_ed1_experiment, load_ed1_tasks

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
