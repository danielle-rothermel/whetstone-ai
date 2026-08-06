import pytest
from dr_store import ObjectStore, SqliteBackend

from tests.optimization.codex.support import engine, runtime_config
from whetstone.envs.factory import EnvExperiment


def test_runtime_rejects_reconstructed_eval_config_mismatch(
    tmp_path, codex_experiment: EnvExperiment
) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "runtime.sqlite"))
    evaluation_engine = engine(store, codex_experiment)
    runtime = runtime_config(evaluation_engine).model_copy(
        update={"expected_eval_config_hash": "0" * 64}
    )

    with pytest.raises(ValueError):
        runtime.build_engine(store)
