from __future__ import annotations

import pytest

from whetstone.core.roles import EvaluationRole
from whetstone.envs.generation_graph import build_generation_graph
from whetstone.envs.procedure import env_procedure_config
from whetstone.envs.registry import ENV_NAMES, env_spec
from whetstone.envs.sampling import (
    INTERNAL_EVAL,
    OFFICIAL,
    Completeness,
    build_aggregation_config,
    build_eval_configs,
    derive_split_sampling,
    validate_evaluation_role_for_split,
)
from whetstone.experiment.graph.eval_identity import (
    validate_eval_identity_partition,
)

_MODEL = "openai/gpt-5-nano"
_SPLIT = (1, 1, 1)


@pytest.mark.parametrize(
    ("split_role", "evaluation_role", "valid"),
    [
        (INTERNAL_EVAL, EvaluationRole.INTERNAL, True),
        (INTERNAL_EVAL, EvaluationRole.OFFICIAL, False),
        (OFFICIAL, EvaluationRole.INTERNAL, False),
        (OFFICIAL, EvaluationRole.OFFICIAL, True),
    ],
)
def test_evaluation_role_mapping_is_exact(
    split_role: str,
    evaluation_role: EvaluationRole,
    valid: bool,
) -> None:
    if valid:
        validate_evaluation_role_for_split(
            split_role=split_role,
            evaluation_role=evaluation_role,
        )
        return
    with pytest.raises(ValueError, match="does not match split role"):
        validate_evaluation_role_for_split(
            split_role=split_role,
            evaluation_role=evaluation_role,
        )


def _eval_configs(
    env_name: str,
    *,
    completeness=Completeness.PROPAGATE,
    max_skip_fraction: float = 0.0,
):
    env = env_spec(env_name)
    a = b = c = 1
    if env.stratified_split:
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
        max_skip_fraction=max_skip_fraction,
        split_sizes=(a, b, c),
    )


@pytest.mark.parametrize("env_name", ENV_NAMES)
def test_internal_and_official_are_ordered_and_disjoint(
    env_name: str,
) -> None:
    _, configs = _eval_configs(env_name)
    internal_ids = configs.internal.task_set.task_hashes
    official_ids = configs.official.task_set.task_hashes
    assert isinstance(internal_ids, tuple)
    assert isinstance(official_ids, tuple)
    assert set(internal_ids).isdisjoint(official_ids)


@pytest.mark.parametrize("env_name", ENV_NAMES)
def test_held_out_never_referenced_by_any_config(env_name: str) -> None:
    _, configs = _eval_configs(env_name)
    sampled = set(configs.internal.task_set.task_hashes) | set(
        configs.official.task_set.task_hashes
    )
    held_out = set(configs.held_out_task_hashes)
    assert held_out
    assert sampled.isdisjoint(held_out)


@pytest.mark.parametrize("env_name", ENV_NAMES)
def test_both_eval_configs_share_one_procedure_identity(
    env_name: str,
) -> None:
    _, configs = _eval_configs(env_name)
    internal_ec = configs.internal.eval_config
    official_ec = configs.official.eval_config
    assert (
        internal_ec.evaluation_procedure_config_hash
        == official_ec.evaluation_procedure_config_hash
        == configs.procedure_config_hash
    )
    assert internal_ec.config_hash != official_ec.config_hash


@pytest.mark.parametrize("env_name", ENV_NAMES)
def test_eval_config_hash_differs_graph_hash_unchanged(
    env_name: str,
) -> None:
    env, configs = _eval_configs(env_name)
    rd = build_generation_graph(env, model=_MODEL)
    validate_eval_identity_partition(
        rd.graph_config, configs.internal.eval_config
    )
    validate_eval_identity_partition(
        rd.graph_config, configs.official.eval_config
    )
    assert (
        configs.internal.eval_config.config_hash
        != configs.official.eval_config.config_hash
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
    assert dict(propagate.assignment)["zero_denominator"] == "not_applicable"
    assert propagate.config_hash != skip.config_hash


def test_skip_tolerance_is_identity_bearing() -> None:
    from whetstone.envs.sampling import build_aggregation_config

    env = env_spec("c18")
    skip_0 = build_aggregation_config(
        env, completeness=Completeness.SKIP, max_skip_fraction=0.0
    )
    skip_2 = build_aggregation_config(
        env, completeness=Completeness.SKIP, max_skip_fraction=0.02
    )
    skip_5 = build_aggregation_config(
        env, completeness=Completeness.SKIP, max_skip_fraction=0.05
    )
    assert dict(skip_2.assignment)["max_skip_fraction"] == "0.02"
    hashes = {
        skip_0.config_hash,
        skip_2.config_hash,
        skip_5.config_hash,
    }
    assert len(hashes) == 3


def test_c18_tolerant_official_eval_config_hash_differs_from_strict() -> None:
    _, strict = _eval_configs("c18", completeness=Completeness.PROPAGATE)
    _, tolerant = _eval_configs(
        "c18", completeness=Completeness.SKIP, max_skip_fraction=0.02
    )
    assert (
        strict.official.eval_config.config_hash
        != tolerant.official.eval_config.config_hash
    )


def _stratum_counts(tasks) -> dict[str, int]:
    from collections import Counter

    counts: Counter[str] = Counter()
    for inst in tasks:
        for label in inst.strata:
            counts[label] += 1
    return dict(counts)


def test_c22_split_is_stratum_balanced_on_the_real_pool() -> None:
    # c22 pools are contiguous by stratum, so contiguous slicing would omit
    # harder strata; sample each stratum independently.
    env = env_spec("c22")
    pool = env.generate_pool()
    procedure = env_procedure_config(env)
    configs = build_eval_configs(env, pool=pool, procedure=procedure)

    internal = configs.internal.tasks
    official = configs.official.tasks
    n_strata = len(pool.strata)

    internal_counts = _stratum_counts(internal)
    official_counts = _stratum_counts(official)

    assert set(internal_counts) == set(pool.strata)
    assert set(official_counts) == set(pool.strata)
    assert max(internal_counts.values()) - min(internal_counts.values()) <= 1
    assert max(official_counts.values()) - min(official_counts.values()) <= 1
    assert internal_counts == dict.fromkeys(pool.strata, 12 // n_strata)
    assert official_counts == dict.fromkeys(pool.strata, 36 // n_strata)


def test_eval_config_for_dispatch() -> None:
    _, configs = _eval_configs("c18")
    assert (
        configs.eval_config_for(INTERNAL_EVAL) is configs.internal.eval_config
    )
    assert configs.eval_config_for(OFFICIAL) is configs.official.eval_config
    with pytest.raises(KeyError):
        configs.eval_config_for("held_out")


def _derive_c23(
    *,
    tasks,
    num_samples: int,
    split_role: str = OFFICIAL,
):
    env = env_spec("c23")
    procedure = env_procedure_config(env)
    aggregation = build_aggregation_config(env)
    return derive_split_sampling(
        namespace="whetstone.env.c23.power",
        dataset_revision=env.generator_version,
        split_role=split_role,
        tasks=tuple(tasks),
        task_hash_of=lambda instance: str(instance.id),
        procedure=procedure,
        aggregation=aggregation,
        num_samples=num_samples,
    )


def test_exact_ordered_instances_change_eval_config_identity() -> None:
    env = env_spec("c23")
    instances = env.generate_pool(n_per_stratum=2).instances[:3]
    forward = _derive_c23(tasks=instances, num_samples=2)
    reversed_order = _derive_c23(
        tasks=tuple(reversed(instances)),
        num_samples=2,
    )
    subset = _derive_c23(tasks=instances[:2], num_samples=2)
    assert forward.task_set.task_hashes != (
        reversed_order.task_set.task_hashes
    )
    assert (
        len(
            {
                forward.eval_config.config_hash,
                reversed_order.eval_config.config_hash,
                subset.eval_config.config_hash,
            }
        )
        == 3
    )


def test_exact_repeats_and_role_change_eval_config_identity() -> None:
    env = env_spec("c23")
    instances = env.generate_pool(n_per_stratum=2).instances[:3]
    repeat_two = _derive_c23(tasks=instances, num_samples=2)
    repeat_three = _derive_c23(tasks=instances, num_samples=3)
    internal = _derive_c23(
        tasks=instances,
        num_samples=2,
        split_role=INTERNAL_EVAL,
    )
    assert (
        len(
            {
                repeat_two.eval_config.config_hash,
                repeat_three.eval_config.config_hash,
                internal.eval_config.config_hash,
            }
        )
        == 3
    )


@pytest.mark.parametrize("bad", [0, -1])
def test_exact_derivation_rejects_invalid_num_samples(bad: int) -> None:
    env = env_spec("c23")
    instances = env.generate_pool(n_per_stratum=1).instances[:1]
    with pytest.raises(ValueError, match="num_samples must be at least 1"):
        _derive_c23(tasks=instances, num_samples=bad)
