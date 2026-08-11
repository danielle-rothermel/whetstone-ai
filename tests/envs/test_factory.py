from __future__ import annotations

import pytest

from tests.envs.support import TEST_MODEL, tiny_experiment
from whetstone.envs.factory import build_env_experiment
from whetstone.envs.registry import ENV_NAMES


@pytest.mark.parametrize("env_name", ENV_NAMES)
def test_build_env_experiment_returns_all_five_deliverables(
    env_name: str,
) -> None:
    exp = tiny_experiment(env_name)
    d = exp.as_dict()
    assert set(d) == {
        "generation_graph",
        "initial_candidate",
        "ceiling_candidate",
        "eval_configs",
        "reward_policy",
    }
    assert (
        exp.generation_graph.procedure_config_hash
        == exp.eval_configs.procedure_config_hash
    )


def test_unknown_env_rejected() -> None:
    from whetstone.envs.registry import UnknownEnvError

    with pytest.raises(UnknownEnvError):
        build_env_experiment("c99", model=TEST_MODEL)
