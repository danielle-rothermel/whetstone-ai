from __future__ import annotations

from tests.envs.support import code_comp_direct_experiment
from whetstone.envs.sampling import Completeness


def test_direct_experiment_internal_split_binds_samples() -> None:
    experiment = code_comp_direct_experiment(num_samples=2, internal_n=2)
    split = experiment.eval_configs.internal
    assert split.sample_plan.num_samples == 2
    assert len(split.task_set.task_hashes) == 2
    assert len(split.eval_config.config_hash) == 64


def test_completeness_skip_folds_tolerance_into_policy() -> None:
    policy = Completeness.SKIP.to_policy(max_skip_fraction=0.15)
    assert policy.row_policy.value == "skip"
    assert policy.max_skip_fraction == 0.15


def test_completeness_propagate_defaults_to_strict() -> None:
    policy = Completeness.PROPAGATE.to_policy()
    assert policy.row_policy.value == "propagate"
    assert policy.max_skip_fraction == 0.0
