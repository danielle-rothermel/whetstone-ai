from __future__ import annotations

from uuid import uuid4

import pytest

from dr_store.testing import temp_sqlite_store
from whetstone.testing.runtime import (
    build_toy_copro_control,
    prepare_toy_copro_run,
    register_toy_runtime,
)
from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig


@pytest.fixture
def sqlite_store():
    with temp_sqlite_store() as store:
        yield store


@pytest.fixture
def toy_runtime(sqlite_store):
    runtime_config = ReferenceEvalRuntimeConfig()
    engine = runtime_config.build_engine(sqlite_store)
    control = build_toy_copro_control(breadth=2, depth=1, engine=engine)
    runtime = register_toy_runtime(
        store=sqlite_store,
        engine=engine,
        copro_control=control,
    )
    return runtime, control


@pytest.fixture
def copro_launch(toy_runtime):
    runtime, control = toy_runtime
    run_id = f"test-run-{uuid4().hex[:8]}"
    launch = prepare_toy_copro_run(
        runtime,
        run_id=run_id,
        control=control,
        terminal_top_k=1,
    )
    return runtime, launch


__all__ = ["copro_launch", "sqlite_store", "toy_runtime"]


@pytest.fixture
def codex_engine(sqlite_store):
    """The reference eval engine every Codex test binds its control to."""
    return ReferenceEvalRuntimeConfig().build_engine(sqlite_store)


@pytest.fixture
def codex_tool_config(codex_engine):
    """The one Tool Config a Codex run grants, plus its RUN subject ref."""
    from tests.codex_support import toy_codex_control, toy_codex_run
    from whetstone.optim.contracts import optimization_run_reference

    control = toy_codex_control(engine=codex_engine)
    run, config, _candidate = toy_codex_run(
        control=control, engine=codex_engine
    )
    return config, optimization_run_reference(run).record_ref
