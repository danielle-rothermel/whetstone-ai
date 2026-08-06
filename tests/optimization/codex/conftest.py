import pytest

from tests.optimization.codex.support import experiment
from whetstone.envs.factory import EnvExperiment


@pytest.fixture(scope="session")
def codex_experiment() -> EnvExperiment:
    return experiment()
