import pytest
from dr_store import ObjectStore, SqliteBackend
from pydantic import ValidationError

from tests.optimization.codex.support import engine, runtime_config
from whetstone.envs.factory import EnvExperiment
from whetstone.envs.sampling import INTERNAL_EVAL


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


def test_runtime_is_internal_only(
    tmp_path, codex_experiment: EnvExperiment
) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "runtime-internal.sqlite"))
    evaluation_engine = engine(store, codex_experiment)
    runtime = runtime_config(evaluation_engine)

    reconstructed = runtime.build_engine(store)

    assert reconstructed.sampling.split_role == INTERNAL_EVAL
    with pytest.raises(ValidationError):
        type(runtime).model_validate(
            {**runtime.model_dump(), "sampling_role": "official"}
        )
