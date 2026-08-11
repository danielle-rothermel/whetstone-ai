from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

from tests.optimization.gepa.support import (
    gepa_control,
    make_gepa_detailed_result,
)
from whetstone.core.identity import ContentHash, typed_ref_for_record
from whetstone.optimization.gepa.contracts import GepaDataInstance


class _ReplayDbos:
    workflow_ids: ClassVar[list[str]] = []

    @classmethod
    def workflow(cls):
        def decorate(function):
            return function

        return decorate


class _SetWorkflowID:
    def __init__(self, workflow_id: str) -> None:
        self._workflow_id = workflow_id

    def __enter__(self):
        _ReplayDbos.workflow_ids.append(self._workflow_id)
        return self

    def __exit__(self, *_args):
        return False


def _load_runner():
    # Import the real DBOS API before replacing only the runner's decorator
    # seam.
    import whetstone.optimization.gepa.factory  # noqa: F401

    fake_dbos = types.ModuleType("dbos")
    fake_dbos.__dict__["DBOS"] = _ReplayDbos
    fake_dbos.__dict__["SetWorkflowID"] = _SetWorkflowID
    prior = sys.modules.get("dbos")
    sys.modules["dbos"] = fake_dbos
    module_name = "_gepa_runner_replay_test"
    try:
        path = Path("src/whetstone/optimization/gepa/runner.py")
        spec = importlib.util.spec_from_file_location(module_name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop(module_name, None)
        if prior is None:
            del sys.modules["dbos"]
        else:
            sys.modules["dbos"] = prior


class _Factory:
    def __init__(self, control, *, identity_salt: str = "unit") -> None:
        self.control = control
        self.runtime_hash = typed_ref_for_record(
            "test.gepa.factory",
            {
                "control": control.identity_hash(),
                "identity_salt": identity_salt,
            },
        ).content_hash
        self.create_calls = 0
        self.persist_calls = 0

    def create(self, *, control):
        assert control == self.control
        self.create_calls += 1
        return SimpleNamespace(effect_ordinal=0)

    def persist_result(self, *, control, adapter, detailed_result):
        assert control == self.control
        assert adapter.effect_ordinal == 7
        assert detailed_result == make_gepa_detailed_result(control)
        self.persist_calls += 1
        return typed_ref_for_record(
            "whetstone.gepa.run_result_artifact",
            {"control": control.identity_hash()},
        )


def test_parent_replay_recreates_adapter_at_ordinal_zero(
    monkeypatch,
) -> None:
    module = _load_runner()
    control = gepa_control()
    factory = _Factory(control)
    module.register_gepa_adapter_factory(factory)
    data_ref = typed_ref_for_record("test.gepa.data", {"id": "train"})
    trainset = (
        GepaDataInstance(
            upstream_position=0,
            data_id=control.trainset_task_hashes[0],
            data_ref=data_ref,
            loader_identity_hash="f" * 64,
        ),
    )
    request = module.GepaParentRunRequest(
        factory_identity_hash=factory.runtime_hash,
        control=control,
        seed_candidate={"prompt": "seed"},
        trainset=trainset,
    )
    observed_ordinals: list[int] = []

    def fake_run_gepa_engine(**kwargs):
        adapter = kwargs["adapter"]
        observed_ordinals.append(adapter.effect_ordinal)
        adapter.effect_ordinal = 7
        return make_gepa_detailed_result(control)

    monkeypatch.setattr(
        "whetstone.optimization.gepa.adapter.run_gepa_engine",
        fake_run_gepa_engine,
    )
    _ReplayDbos.workflow_ids.clear()
    runner = module.DbosGepaRunner()

    first = runner.run(request)
    replay = runner.run(request)

    assert replay == first
    assert observed_ordinals == [0, 0]
    assert factory.create_calls == factory.persist_calls == 2
    assert len(set(_ReplayDbos.workflow_ids)) == 1
    assert _ReplayDbos.workflow_ids[0] == (
        f"whetstone-gepa-run-{request.identity_hash()}"
    )


def test_parent_refuses_registered_factory_identity_drift() -> None:
    module = _load_runner()
    control = gepa_control()
    factory = _Factory(control)
    module.register_gepa_adapter_factory(factory)
    request = module.GepaParentRunRequest(
        factory_identity_hash=factory.runtime_hash,
        control=control,
        seed_candidate={"prompt": "seed"},
        trainset=(
            GepaDataInstance(
                upstream_position=0,
                data_id=control.trainset_task_hashes[0],
                data_ref=typed_ref_for_record(
                    "test.gepa.data",
                    {"id": "train"},
                ),
                loader_identity_hash="f" * 64,
            ),
        ),
    )
    factory.runtime_hash = ContentHash("9" * 64)

    with pytest.raises(RuntimeError, match="factory identity drifted"):
        module.DbosGepaRunner().run(request)


def _train_instance(control):
    return GepaDataInstance(
        upstream_position=0,
        data_id=control.trainset_task_hashes[0],
        data_ref=typed_ref_for_record("test.gepa.data", {"id": "train"}),
        loader_identity_hash="f" * 64,
    )


def test_parent_request_rejects_a_valset_the_control_never_bound() -> None:

    module = _load_runner()
    control = gepa_control()
    assert control.source_valset_task_hashes is None

    with pytest.raises(ValueError, match="supplied an unbound valset"):
        module.GepaParentRunRequest(
            factory_identity_hash="f" * 64,
            control=control,
            seed_candidate={"prompt": "seed"},
            trainset=(_train_instance(control),),
            valset=(
                GepaDataInstance(
                    upstream_position=0,
                    data_id=control.trainset_task_hashes[0],
                    data_ref=typed_ref_for_record(
                        "test.gepa.data",
                        {"id": "val"},
                    ),
                    loader_identity_hash="f" * 64,
                ),
            ),
        )


def test_parent_request_rejects_bound_valset_identity_drift() -> None:

    module = _load_runner()
    control = gepa_control(valset_task_hashes=("c" * 64,))
    assert control.source_valset_task_hashes is not None

    with pytest.raises(ValueError, match="valset identity drift"):
        module.GepaParentRunRequest(
            factory_identity_hash="f" * 64,
            control=control,
            seed_candidate={"prompt": "seed"},
            trainset=(_train_instance(control),),
            valset=(
                GepaDataInstance(
                    upstream_position=0,
                    data_id="d" * 64,
                    data_ref=typed_ref_for_record(
                        "test.gepa.data",
                        {"id": "val"},
                    ),
                    loader_identity_hash="f" * 64,
                ),
            ),
        )

    accepted = module.GepaParentRunRequest(
        factory_identity_hash="f" * 64,
        control=control,
        seed_candidate={"prompt": "seed"},
        trainset=(_train_instance(control),),
        valset=(
            GepaDataInstance(
                upstream_position=0,
                data_id=control.valset_task_hashes[0],
                data_ref=typed_ref_for_record(
                    "test.gepa.data",
                    {"id": "val"},
                ),
                loader_identity_hash="f" * 64,
            ),
        ),
    )
    assert accepted.identity_hash()


def test_parent_request_rejects_same_count_seed_reordering() -> None:
    module = _load_runner()
    control = gepa_control().model_copy(
        update={
            "component_names": ("alpha", "beta"),
            "num_predictors": 2,
        }
    )

    with pytest.raises(
        ValueError,
        match="seed component order drift",
    ):
        module.GepaParentRunRequest(
            factory_identity_hash="f" * 64,
            control=control,
            seed_candidate={"beta": "beta-0", "alpha": "alpha-0"},
            trainset=(
                GepaDataInstance(
                    upstream_position=0,
                    data_id=control.trainset_task_hashes[0],
                    data_ref=typed_ref_for_record(
                        "test.gepa.data",
                        {"id": "train"},
                    ),
                    loader_identity_hash="f" * 64,
                ),
            ),
        )
