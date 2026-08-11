from __future__ import annotations

import json
from pathlib import Path

import pytest

from whetstone.envs.task_pools import (
    roles_for_env,
    select_lowest_historical_pass_rate_for_env,
)
from whetstone.experiment.task_selection import (
    TaskRoleSelectionMethod,
    TaskSplitRole,
    parse_task_split_manifest,
)

_COPRO_MANIFEST = (
    Path(__file__).parents[3]
    / "src/whetstone/optimization/copro/humaneval_copro_challenge_v1.json"
)


def test_frozen_copro_challenge_manifest_matches_reference_metadata() -> None:
    payload = json.loads(_COPRO_MANIFEST.read_text())
    manifest = parse_task_split_manifest(payload)
    roles = roles_for_env(manifest, "ed1")
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

    selection = select_lowest_historical_pass_rate_for_env(
        manifest,
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
