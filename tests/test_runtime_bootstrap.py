from __future__ import annotations

from whetstone.coordination.runtime_bootstrap import register_runtime
from dr_store.sync import BlockingObjectStore


def test_register_runtime_is_idempotent_for_same_store(sqlite_store) -> None:
    first = register_runtime(store=sqlite_store)
    second = register_runtime(store=sqlite_store)
    assert first.harness is not second.harness
    assert first.controller.runtime_hash != second.controller.runtime_hash


def test_register_runtime_opens_store_when_omitted(tmp_path) -> None:
    runtime = register_runtime(sqlite_path=str(tmp_path / "runtime.sqlite"))
    assert isinstance(runtime.store, BlockingObjectStore)
    assert runtime.store.resolve("absent") is None


def test_register_runtime_accepts_caller_engine(sqlite_store) -> None:
    from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
    from whetstone.testing.toy.experiment import build_toy_experiment

    experiment = build_toy_experiment(num_seeds=1)
    engine = ReferenceEvalRuntimeConfig().build_engine(
        sqlite_store,
        experiment=experiment,
    )
    runtime = register_runtime(store=sqlite_store, engine=engine)
    assert runtime.eval_service is not None


def test_prepare_copro_run_uses_caller_experiment(sqlite_store) -> None:
    from whetstone.coordination.runtime_bootstrap import (
        build_toy_copro_control,
        prepare_copro_run,
    )
    from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
    from whetstone.testing.toy.experiment import (
        TOY_MUTATION_FIELD,
        build_toy_experiment,
        toy_template_render_contract,
    )

    experiment = build_toy_experiment(num_seeds=1)
    engine = ReferenceEvalRuntimeConfig().build_engine(
        sqlite_store,
        experiment=experiment,
    )
    runtime = register_runtime(store=sqlite_store, engine=engine)
    control = build_toy_copro_control(breadth=2, depth=1, engine=engine)
    launch = prepare_copro_run(
        runtime,
        run_id="caller-experiment",
        control=control,
        experiment=experiment,
        render_contract=toy_template_render_contract(),
        mutation_field=TOY_MUTATION_FIELD,
    )
    assert launch.run.reward_policy == experiment.reward_policy
    assert launch.initial_candidate == experiment.initial_candidate
