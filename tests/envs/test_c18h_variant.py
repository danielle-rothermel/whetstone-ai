from __future__ import annotations

import pytest

from whetstone.envs.factory import EnvExperiment, build_env_experiment
from whetstone.envs.registry import (
    ENV_NAMES,
    ESTIMATE_INHERITED_PENDING,
    env_spec,
)

_MODEL = "deepseek/deepseek-v4-flash"

# Pool generation starts a subprocess per depth; use tiny pools except for
# the full-shape contract.
_TINY_N = 2
_TINY_SPLIT = (1, 2, 3)


@pytest.fixture(scope="session")
def c18h_tiny() -> EnvExperiment:
    return build_env_experiment(
        "c18h",
        model=_MODEL,
        pool_n_per_stratum=_TINY_N,
        split_sizes=_TINY_SPLIT,
    )


@pytest.fixture(scope="session")
def c18_tiny() -> EnvExperiment:
    return build_env_experiment(
        "c18",
        model=_MODEL,
        pool_n_per_stratum=_TINY_N,
        split_sizes=_TINY_SPLIT,
    )


def test_c18h_is_a_bound_env_that_resolves() -> None:
    assert "c18h" in ENV_NAMES
    env = env_spec("c18h")
    assert env.name == "c18h"
    assert env.generate.__name__ == "whetstone_envs.c18.generate"
    assert env.oracle.__name__ == "whetstone_envs.c18.oracle"
    assert env.oracle_qualname == "whetstone_envs.c18.oracle.score_gold"
    assert env.gold_first is False
    assert env.stratified_split is False
    assert env.token_estimate.estimate_source == ESTIMATE_INHERITED_PENDING
    assert env.token_estimate.naive == 1959
    assert env.token_estimate.ceiling == 3672


def test_c18h_pool_is_the_deep_hard_preset() -> None:
    env = env_spec("c18h")
    pool = env.generate_pool(n_per_stratum=_TINY_N)
    assert pool.stratum_counts() == {
        "D5": _TINY_N,
        "D8": _TINY_N,
        "D10": _TINY_N,
    }
    for inst in pool.instances:
        assert inst.gold in {"True", "False"}


def test_c18h_full_pool_is_sixty_across_three_deep_strata() -> None:
    env = env_spec("c18h")
    pool = env.generate_pool()
    assert len(pool) == 60
    assert pool.stratum_counts() == {"D5": 20, "D8": 20, "D10": 20}
    for inst in pool.instances:
        assert inst.gold in {"True", "False"}


def test_c18h_dataset_revision_is_the_hard_generator_version() -> None:
    assert env_spec("c18h").generator_version == "c18-generate-1+hard"
    assert env_spec("c18").generator_version == "c18-generate-1"


def test_c18h_split_is_six_eighteen_thirtysix() -> None:
    env = env_spec("c18h")
    pool = env.generate_pool(n_per_stratum=_TINY_N)
    assert env.default_split_sizes(pool) == (6, 18, 36)


def test_c18h_splits_are_disjoint_and_holdout_untouched(
    c18h_tiny: EnvExperiment,
) -> None:
    exp = c18h_tiny
    internal = set(exp.eval_configs.internal.task_set.task_hashes)
    official = set(exp.eval_configs.official.task_set.task_hashes)
    held_out = set(exp.eval_configs.held_out_task_hashes)
    assert len(internal) == _TINY_SPLIT[0]
    assert len(official) == _TINY_SPLIT[1]
    assert len(held_out) == _TINY_SPLIT[2]
    assert internal.isdisjoint(official)
    assert internal.isdisjoint(held_out)
    assert official.isdisjoint(held_out)


def test_c18h_eval_config_hash_differs_from_c18(
    c18_tiny: EnvExperiment,
    c18h_tiny: EnvExperiment,
) -> None:
    for role in ("internal", "official"):
        c18_hash = getattr(c18_tiny.eval_configs, role).eval_config.config_hash
        c18h_hash = getattr(
            c18h_tiny.eval_configs, role
        ).eval_config.config_hash
        assert c18_hash != c18h_hash, f"{role} eval_config_hash collides"


def test_c18h_task_hashes_are_disjoint_from_c18(
    c18_tiny: EnvExperiment,
    c18h_tiny: EnvExperiment,
) -> None:
    c18_ids = set(c18_tiny.eval_configs.official.task_set.task_hashes) | set(
        c18_tiny.eval_configs.internal.task_set.task_hashes
    )
    c18h_ids = set(c18h_tiny.eval_configs.official.task_set.task_hashes) | set(
        c18h_tiny.eval_configs.internal.task_set.task_hashes
    )
    assert c18_ids.isdisjoint(c18h_ids)


def test_c18h_procedure_identity_partition_holds(
    c18h_tiny: EnvExperiment,
) -> None:
    assert (
        c18h_tiny.generation_graph.procedure_config_hash
        == c18h_tiny.eval_configs.procedure_config_hash
    )


@pytest.mark.parametrize("model", ["openai/gpt-5-nano", _MODEL])
def test_c18h_builds_under_both_task_models(model: str) -> None:
    exp = build_env_experiment(
        "c18h",
        model=model,
        pool_n_per_stratum=_TINY_N,
        split_sizes=_TINY_SPLIT,
    )
    assert exp.env_name == "c18h"
