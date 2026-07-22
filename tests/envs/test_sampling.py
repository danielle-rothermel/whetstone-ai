"""Sampling Configs, Task Sets, and the composite Eval Configs.

Proves: internal/official Task Sets are ordered + disjoint; held_out is never
referenced; the two Eval Configs share one Evaluation Procedure Config
identity (graph_hash unchanged, eval_config_hash differs); Aggregation is mean
with an explicit completeness policy.
"""

from __future__ import annotations

import pytest

from whetstone.envs.procedure import env_procedure_config
from whetstone.envs.registry import ENV_NAMES, env_spec
from whetstone.envs.rollout_definition import build_rollout_definition
from whetstone.envs.sampling import (
    INTERNAL_EVAL,
    OFFICIAL,
    Completeness,
    build_eval_configs,
)
from whetstone.graph.eval_config import validate_eval_identity_partition

_MODEL = "openai/gpt-5-nano"
# Tiny pools: c22 has 6 strata, the others 4-26; keep splits inside each pool.
_SPLIT = (1, 1, 1)


def _eval_configs(env_name: str, *, completeness=Completeness.PROPAGATE):
    env = env_spec(env_name)
    # One instance per stratum keeps the split trivially satisfiable; the
    # split sizes below are per whole-pool, so pick 1 internal / 1 official /
    # 1 held-out globally by generating enough instances.
    pool = env.generate_pool(n_per_stratum=1)
    procedure = env_procedure_config(env)
    n = len(pool)
    # split (a, b, c) must sum <= n; use a balanced tiny split.
    a = 1
    b = 1
    c = 1
    if a + b + c > n:  # pragma: no cover - all envs have >= 3 strata
        a = b = c = n // 3
    return env, build_eval_configs(
        env,
        pool=pool,
        procedure=procedure,
        completeness=completeness,
        split_sizes=(a, b, c),
    )


@pytest.mark.parametrize("env_name", ENV_NAMES)
def test_internal_and_official_are_ordered_and_disjoint(
    env_name: str,
) -> None:
    _, configs = _eval_configs(env_name)
    internal_ids = configs.internal.task_set.task_identities
    official_ids = configs.official.task_set.task_identities
    # Ordering is identity-bearing (a tuple, not a set).
    assert isinstance(internal_ids, tuple)
    assert isinstance(official_ids, tuple)
    assert set(internal_ids).isdisjoint(official_ids)


@pytest.mark.parametrize("env_name", ENV_NAMES)
def test_held_out_never_referenced_by_any_config(env_name: str) -> None:
    _, configs = _eval_configs(env_name)
    sampled = set(configs.internal.task_set.task_identities) | set(
        configs.official.task_set.task_identities
    )
    held_out = set(configs.held_out_task_identities)
    assert held_out
    assert sampled.isdisjoint(held_out)


@pytest.mark.parametrize("env_name", ENV_NAMES)
def test_both_eval_configs_share_one_procedure_identity(
    env_name: str,
) -> None:
    _, configs = _eval_configs(env_name)
    internal_ec = configs.internal.eval_config
    official_ec = configs.official.eval_config
    # Same Evaluation Procedure Config identity in both.
    assert (
        internal_ec.evaluation_procedure_config_hash
        == official_ec.evaluation_procedure_config_hash
        == configs.procedure_config_hash
    )
    # Distinct Sampling Configs => distinct composite eval_config_hash.
    assert internal_ec.config_identity_hash != official_ec.config_identity_hash


@pytest.mark.parametrize("env_name", ENV_NAMES)
def test_eval_config_hash_differs_graph_hash_unchanged(
    env_name: str,
) -> None:
    env, configs = _eval_configs(env_name)
    rd = build_rollout_definition(env, model=_MODEL)
    # The graph's Eval Node procedure hash matches BOTH composite Eval Configs,
    # so the same graph_hash validates against internal and official alike.
    validate_eval_identity_partition(
        rd.graph_config, configs.internal.eval_config
    )
    validate_eval_identity_partition(
        rd.graph_config, configs.official.eval_config
    )
    # graph_hash unchanged across the two eval configs; eval_config_hash not.
    assert (
        configs.internal.eval_config.config_identity_hash
        != configs.official.eval_config.config_identity_hash
    )


@pytest.mark.parametrize("env_name", ENV_NAMES)
def test_aggregation_is_mean_with_completeness_policy(
    env_name: str,
) -> None:
    env = env_spec(env_name)
    from whetstone.envs.sampling import build_aggregation_config

    propagate = build_aggregation_config(
        env, completeness=Completeness.PROPAGATE
    )
    skip = build_aggregation_config(env, completeness=Completeness.SKIP)
    assert dict(propagate.assignment)["reduction"] == "mean"
    assert dict(propagate.assignment)["missing_data"] == "propagate"
    assert dict(skip.assignment)["missing_data"] == "skip"
    # zero_denominator is explicit (not silently coerced).
    assert dict(propagate.assignment)["zero_denominator"] == "not_applicable"
    # The completeness policy is identity-bearing.
    assert propagate.config_identity_hash != skip.config_identity_hash


def test_eval_config_for_dispatch() -> None:
    _, configs = _eval_configs("c18")
    assert (
        configs.eval_config_for(INTERNAL_EVAL)
        is configs.internal.eval_config
    )
    assert (
        configs.eval_config_for(OFFICIAL) is configs.official.eval_config
    )
    with pytest.raises(KeyError):
        configs.eval_config_for("held_out")
