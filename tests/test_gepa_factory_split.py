"""The factory must hand upstream GEPA the control's train/val split.

One eval engine serves both splits, so the GEPA data registry holds their
ordered union. Passing that union as the trainset would reflect on validation
instances and let Pareto selection see training instances, contradicting the
split ``run_gepa_engine`` enforces.
"""

from __future__ import annotations

import pytest
from dr_store.sync import open_sqlite

from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
from whetstone.optim.gepa.step_engine import GepaStepCheckpoint
from whetstone.testing.runtime import (
    build_toy_gepa_adapter,
    build_toy_gepa_control,
)
from whetstone.testing.toy.experiment import build_toy_experiment


def _split_engine(store):
    experiment = build_toy_experiment(num_seeds=1)
    engine = ReferenceEvalRuntimeConfig().build_engine(
        store,
        experiment=experiment,
    )
    task_hashes = engine.sampling.task_hashes
    assert len(task_hashes) >= 2
    return experiment, engine, task_hashes[:1], task_hashes[1:]


def _adapter_for(store, *, control, experiment, engine, run_id):
    return build_toy_gepa_adapter(
        store=store,
        engine=engine,
        control=control,
        run_id=run_id,
        initial_candidate=experiment.initial_candidate,
    )


def _hashes(instances) -> tuple[str, ...]:
    return tuple(item.task_hash for item in instances)


def test_split_control_partitions_the_registry(tmp_path) -> None:
    with open_sqlite(str(tmp_path / "split.sqlite")) as store:
        experiment, engine, train, val = _split_engine(store)
        control = build_toy_gepa_control(
            engine=engine,
            trainset_task_hashes=train,
            valset_task_hashes=val,
        )
        adapter = _adapter_for(
            store,
            control=control,
            experiment=experiment,
            engine=engine,
            run_id="gepa-split-run",
        )

        assert _hashes(adapter._trainset) == control.trainset_task_hashes
        assert adapter._valset is not None
        assert _hashes(adapter._valset) == control.valset_task_hashes
        # The two splits are disjoint and together cover the registry.
        assert not set(_hashes(adapter._trainset)) & set(
            _hashes(adapter._valset)
        )
        assert set(_hashes(adapter._trainset)) | set(
            _hashes(adapter._valset)
        ) == set(engine.sampling.task_hashes)


def test_unsplit_control_keeps_upstream_default(tmp_path) -> None:
    with open_sqlite(str(tmp_path / "unsplit.sqlite")) as store:
        experiment, engine, _train, _val = _split_engine(store)
        control = build_toy_gepa_control(engine=engine)
        adapter = _adapter_for(
            store,
            control=control,
            experiment=experiment,
            engine=engine,
            run_id="gepa-unsplit-run",
        )

        assert control.source_valset_task_hashes is None
        assert _hashes(adapter._trainset) == control.trainset_task_hashes
        assert adapter._valset is None


def test_split_run_selects_only_on_validation_tasks(tmp_path) -> None:
    """Drive one in-process GEPA step and read the selection evidence.

    Upstream GEPA scores candidates over the valset, so the per-instance
    Pareto front it reports must be keyed by validation tasks only.
    """
    with open_sqlite(str(tmp_path / "step.sqlite")) as store:
        experiment, engine, train, val = _split_engine(store)
        control = build_toy_gepa_control(
            engine=engine,
            max_metric_calls=2,
            trainset_task_hashes=train,
            valset_task_hashes=val,
        )
        adapter = _adapter_for(
            store,
            control=control,
            experiment=experiment,
            engine=engine,
            run_id="gepa-step-run",
        )

        from whetstone.optim.gepa.step_engine import run_one_gepa_iteration

        adapter._adapter_factory.begin_step(step_index=0)
        engine_adapter = adapter._adapter_factory.create(control=control)
        detailed, checkpoint = run_one_gepa_iteration(
            control=control,
            seed_candidate=adapter.seed_candidate,
            trainset=adapter._trainset,
            valset=adapter._valset,
            adapter=engine_adapter,
            checkpoint=GepaStepCheckpoint(),
        )

        assert isinstance(checkpoint, GepaStepCheckpoint)
        val_ids = set(control.valset_task_hashes)
        train_only = set(control.trainset_task_hashes) - val_ids
        assert train_only, "the toy split must have a train-only task"

        pareto_keys = set(detailed.per_val_instance_best_candidates)
        assert pareto_keys
        assert pareto_keys <= val_ids
        assert not pareto_keys & train_only
        for subscores in detailed.val_subscores:
            assert set(subscores) <= val_ids


def test_registry_conflicting_with_the_control_is_loud(tmp_path) -> None:
    with open_sqlite(str(tmp_path / "conflict.sqlite")) as store:
        experiment, engine, train, val = _split_engine(store)
        control = build_toy_gepa_control(
            engine=engine,
            trainset_task_hashes=train,
            valset_task_hashes=val,
        )
        # An engine narrowed to the trainset cannot serve the valset.
        narrowed = engine.for_task_ids((engine.sampling.tasks[0].task_id,))
        with pytest.raises(ValueError):
            _adapter_for(
                store,
                control=control,
                experiment=experiment,
                engine=narrowed,
                run_id="gepa-conflict-run",
            )
