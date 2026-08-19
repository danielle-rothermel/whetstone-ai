from __future__ import annotations

from uuid import uuid4

import pytest

from whetstone.core.blocking_store import open_blocking_sqlite_store
from whetstone.coordination.runtime_bootstrap import (
    build_toy_copro_control,
    copro_run_request,
    prepare_copro_run,
    register_runtime,
)
from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
from whetstone.optim.contracts import OPTIM_RESULT_SCHEMA


@pytest.fixture
def sqlite_store(tmp_path):
    path = tmp_path / "store.sqlite"
    with open_blocking_sqlite_store(str(path)) as store:
        yield store


@pytest.fixture
def toy_runtime(sqlite_store):
    runtime = register_runtime(store=sqlite_store)
    runtime_config = ReferenceEvalRuntimeConfig()
    engine = runtime_config.build_engine(sqlite_store)
    control = build_toy_copro_control(breadth=2, depth=1, engine=engine)
    return runtime, control


@pytest.fixture
def copro_launch(toy_runtime):
    runtime, control = toy_runtime
    run_id = f"test-run-{uuid4().hex[:8]}"
    launch = prepare_copro_run(
        runtime,
        run_id=run_id,
        control=control,
        terminal_top_k=1,
    )
    return runtime, launch


__all__ = ["copro_launch", "sqlite_store", "toy_runtime"]
