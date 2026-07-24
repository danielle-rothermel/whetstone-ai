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
    # A balanced tiny (1, 1, 1) split. For a contiguous-split env one instance
    # per stratum is enough; for a stratified-split env (c22, blocked pool)
    # each stratum must independently hold its per-stratum quota, so size the
    # pool to the largest single-stratum draw.
    a = b = c = 1
    if env.stratified_split:
        # ceil(part / n_strata) summed over the split parts is the max any one
        # stratum must supply; grow n_per_stratum to clear it.
        probe = env.generate_pool(n_per_stratum=1)
        n_strata = len(probe.strata)
        per_stratum = sum(-(-part // n_strata) for part in (a, b, c))
        pool = env.generate_pool(n_per_stratum=per_stratum)
    else:
        pool = env.generate_pool(n_per_stratum=1)
    procedure = env_procedure_config(env)
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


def _stratum_counts(instances) -> dict[str, int]:
    from collections import Counter

    counts: Counter[str] = Counter()
    for inst in instances:
        for label in inst.strata:
            counts[label] += 1
    return dict(counts)


def test_c22_split_is_stratum_balanced_on_the_real_pool() -> None:
    # c22's real pool is BLOCKED (all n3_easy first, then n3_mixed, ...).
    # TaskPool.split's contiguous slicing would put the whole internal_eval
    # slice in the single easiest stratum and drop the hardest strata into the
    # unused remainder tail (build-report judgment call #2's balance claim).
    # The adapter's stratified split must instead sample every stratum evenly.
    env = env_spec("c22")
    pool = env.generate_pool()  # the real 120-instance default pool
    procedure = env_procedure_config(env)
    configs = build_eval_configs(env, pool=pool, procedure=procedure)

    internal = configs.internal.instances
    official = configs.official.instances
    n_strata = len(pool.strata)

    internal_counts = _stratum_counts(internal)
    official_counts = _stratum_counts(official)

    # Every stratum is represented in BOTH internal and official (no stratum
    # missing), and per-stratum counts are balanced (max-min <= 1) rather than
    # concentrated in the leading strata.
    assert set(internal_counts) == set(pool.strata)
    assert set(official_counts) == set(pool.strata)
    assert max(internal_counts.values()) - min(internal_counts.values()) <= 1
    assert max(official_counts.values()) - min(official_counts.values()) <= 1
    # The default (12, 36, 36) totals over 6 strata => exactly 2 / 6 per
    # stratum: the small-internal / balanced-official shape the build report
    # claims (and which contiguous slicing does NOT yield for c22).
    assert internal_counts == dict.fromkeys(pool.strata, 12 // n_strata)
    assert official_counts == dict.fromkeys(pool.strata, 36 // n_strata)


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
