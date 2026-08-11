from __future__ import annotations

import pytest

from whetstone.envs.factory import build_env_experiment
from whetstone.envs.registry import (
    ENV_NAMES,
    ESTIMATE_INHERITED_PENDING,
    env_spec,
)

_MODEL = "deepseek/deepseek-v4-flash"
_HARD_IDS = frozenset(
    {
        "length_constraints:number_words",
        "keywords:letter_frequency",
        "keywords:forbidden_words",
    }
)


def test_c22h_is_a_bound_env_that_resolves() -> None:
    assert "c22h" in ENV_NAMES
    env = env_spec("c22h")
    assert env.name == "c22h"
    assert env.generate.__name__ == "whetstone_envs.c22.generate"
    assert env.oracle.__name__ == "whetstone_envs.c22.oracle"
    assert env.oracle_qualname == "whetstone_envs.c22.oracle.score_gold"
    assert env.gold_first is True
    assert env.stratified_split is True
    assert env.token_estimate.estimate_source == ESTIMATE_INHERITED_PENDING


def test_c22h_pool_is_the_hard_preset() -> None:
    env = env_spec("c22h")
    pool = env.generate_pool()
    assert len(pool) == 60
    assert pool.stratum_counts() == {
        "n3_hard": 20,
        "n6_hard": 20,
        "n8_hard": 20,
    }
    from whetstone_envs.c22.spec import ConstraintSpec

    for inst in pool.instances:
        ids = frozenset(
            ConstraintSpec.from_gold(inst.gold).instruction_id_list
        )
        assert _HARD_IDS <= ids


def test_c22h_dataset_revision_is_the_hard_generator_version() -> None:
    assert env_spec("c22h").generator_version == "c22-generate-1+hard"
    assert env_spec("c22").generator_version == "c22-generate-1"


def test_c22h_split_is_six_eighteen_thirtysix() -> None:
    env = env_spec("c22h")
    pool = env.generate_pool()
    assert env.default_split_sizes(pool) == (6, 18, 36)


def test_c22h_splits_are_disjoint_and_holdout_untouched() -> None:
    exp = build_env_experiment("c22h", model=_MODEL)
    internal = set(exp.eval_configs.internal.task_set.task_hashes)
    official = set(exp.eval_configs.official.task_set.task_hashes)
    held_out = set(exp.eval_configs.held_out_task_hashes)
    assert len(internal) == 6
    assert len(official) == 18
    assert len(held_out) == 36
    assert internal.isdisjoint(official)
    assert internal.isdisjoint(held_out)
    assert official.isdisjoint(held_out)


def test_c22h_eval_config_hash_differs_from_c22() -> None:
    c22 = build_env_experiment("c22", model=_MODEL)
    c22h = build_env_experiment("c22h", model=_MODEL)
    for role in ("internal", "official"):
        c22_hash = getattr(c22.eval_configs, role).eval_config.config_hash
        c22h_hash = getattr(c22h.eval_configs, role).eval_config.config_hash
        assert c22_hash != c22h_hash, f"{role} eval_config_hash collides"


def test_c22h_task_hashes_are_disjoint_from_c22() -> None:
    c22 = build_env_experiment("c22", model=_MODEL)
    c22h = build_env_experiment("c22h", model=_MODEL)
    c22_ids = set(c22.eval_configs.official.task_set.task_hashes) | set(
        c22.eval_configs.internal.task_set.task_hashes
    )
    c22h_ids = set(c22h.eval_configs.official.task_set.task_hashes) | set(
        c22h.eval_configs.internal.task_set.task_hashes
    )
    assert c22_ids.isdisjoint(c22h_ids)


def test_c22h_procedure_identity_partition_holds() -> None:
    exp = build_env_experiment("c22h", model=_MODEL)
    assert (
        exp.rollout_definition.procedure_config_hash
        == exp.eval_configs.procedure_config_hash
    )


@pytest.mark.parametrize("model", ["openai/gpt-5-nano", _MODEL])
def test_c22h_builds_under_both_task_models(model: str) -> None:
    exp = build_env_experiment("c22h", model=model)
    assert exp.env_name == "c22h"
