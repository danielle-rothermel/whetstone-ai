"""Stable-parent replay tests for the canonical GEPA runner."""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, cast
from uuid import uuid4

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
    # Load the factory and package with the real DBOS API before replacing
    # only the runner's decorator/SetWorkflowID seam.
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
        self.runtime_identity_hash = typed_ref_for_record(
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
            data_id=control.trainset_task_identities[0],
            data_ref=data_ref,
            loader_identity_hash="f" * 64,
        ),
    )
    request = module.GepaParentRunRequest(
        factory_identity_hash=factory.runtime_identity_hash,
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
        factory_identity_hash=factory.runtime_identity_hash,
        control=control,
        seed_candidate={"prompt": "seed"},
        trainset=(
            GepaDataInstance(
                upstream_position=0,
                data_id=control.trainset_task_identities[0],
                data_ref=typed_ref_for_record(
                    "test.gepa.data",
                    {"id": "train"},
                ),
                loader_identity_hash="f" * 64,
            ),
        ),
    )
    factory.runtime_identity_hash = ContentHash("9" * 64)

    with pytest.raises(RuntimeError, match="factory identity drifted"):
        module.DbosGepaRunner().run(request)


def _train_instance(control):
    return GepaDataInstance(
        upstream_position=0,
        data_id=control.trainset_task_identities[0],
        data_ref=typed_ref_for_record("test.gepa.data", {"id": "train"}),
        loader_identity_hash="f" * 64,
    )


def test_parent_request_rejects_a_valset_the_control_never_bound() -> None:
    """Validation is symmetric with the engine's valset binding check.

    The engine refuses a supplied valset when the control binds valset
    omission, so a request carrying one is unrunnable and must not be able
    to validate and hash into a persistable workflow ID.
    """

    module = _load_runner()
    control = gepa_control()
    assert control.source_valset_task_identities is None

    with pytest.raises(ValueError, match="supplied an unbound valset"):
        module.GepaParentRunRequest(
            factory_identity_hash="f" * 64,
            control=control,
            seed_candidate={"prompt": "seed"},
            trainset=(_train_instance(control),),
            valset=(
                GepaDataInstance(
                    upstream_position=0,
                    data_id=control.trainset_task_identities[0],
                    data_ref=typed_ref_for_record(
                        "test.gepa.data",
                        {"id": "val"},
                    ),
                    loader_identity_hash="f" * 64,
                ),
            ),
        )


def test_parent_request_rejects_bound_valset_identity_drift() -> None:
    """A bound valset must still match the control's exact data identities."""

    module = _load_runner()
    control = gepa_control(valset_task_identities=("c" * 64,))
    assert control.source_valset_task_identities is not None

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

    # The matching valset validates and hashes.
    accepted = module.GepaParentRunRequest(
        factory_identity_hash="f" * 64,
        control=control,
        seed_candidate={"prompt": "seed"},
        trainset=(_train_instance(control),),
        valset=(
            GepaDataInstance(
                upstream_position=0,
                data_id=control.valset_task_identities[0],
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
                    data_id=control.trainset_task_identities[0],
                    data_ref=typed_ref_for_record(
                        "test.gepa.data",
                        {"id": "train"},
                    ),
                    loader_identity_hash="f" * 64,
                ),
            ),
        )


@pytest.mark.skipif(
    "WHETSTONE_TEST_POSTGRES_DSN" not in os.environ,
    reason="WHETSTONE_TEST_POSTGRES_DSN is required for real DBOS replay",
)
@pytest.mark.postgres_integration
def test_real_dbos_parent_same_id_returns_checkpointed_result(
    monkeypatch,
) -> None:
    from dbos import DBOS, DBOSConfig

    from whetstone.optimization.gepa.runner import (
        DbosGepaRunner,
        GepaParentRunRequest,
        register_gepa_adapter_factory,
    )

    suffix = uuid4().hex[:10]
    database_url = os.environ["WHETSTONE_TEST_POSTGRES_DSN"]
    config: DBOSConfig = {
        "name": f"gepa-parent-{suffix}",
        "system_database_url": database_url,
        "application_database_url": database_url,
        "application_version": f"gepa-parent-{suffix}",
        "run_admin_server": False,
        "use_listen_notify": False,
    }
    DBOS(config=config)
    control = gepa_control()
    factory = _Factory(control, identity_salt=suffix)
    register_gepa_adapter_factory(cast(Any, factory))
    request = GepaParentRunRequest(
        factory_identity_hash=factory.runtime_identity_hash,
        control=control,
        seed_candidate={"prompt": "seed"},
        trainset=(
            GepaDataInstance(
                upstream_position=0,
                data_id=control.trainset_task_identities[0],
                data_ref=typed_ref_for_record(
                    "test.gepa.data",
                    {"id": "train"},
                ),
                loader_identity_hash="f" * 64,
            ),
        ),
    )
    engine_calls = 0

    def fake_run_gepa_engine(**kwargs):
        nonlocal engine_calls
        engine_calls += 1
        kwargs["adapter"].effect_ordinal = 7
        return make_gepa_detailed_result(control)

    monkeypatch.setattr(
        "whetstone.optimization.gepa.adapter.run_gepa_engine",
        fake_run_gepa_engine,
    )
    try:
        DBOS.launch()
        runner = DbosGepaRunner()
        first = runner.run(request)
        replay = runner.run(request)
        assert replay == first
        assert engine_calls == 1
        assert factory.create_calls == factory.persist_calls == 1
    finally:
        DBOS.destroy()


@pytest.mark.skipif(
    "WHETSTONE_TEST_POSTGRES_DSN" not in os.environ,
    reason="WHETSTONE_TEST_POSTGRES_DSN is required for real DBOS replay",
)
@pytest.mark.postgres_integration
def test_real_dbos_parent_recovery_keeps_child_and_later_step_aligned(
    monkeypatch,
    tmp_path,
) -> None:
    from dbos import DBOS, DBOSClient, DBOSConfig
    from dbos._error import (
        DBOSAwaitedWorkflowCancelledError,
        DBOSWorkflowCancelledError,
    )
    from dr_store import ObjectStore, SqliteBackend

    from tests.optimization.gepa.support import (
        evaluation_authority_binding,
        evaluation_result,
        prompt_services,
        proposal_authority_binding,
    )
    from whetstone.core.identity import TypedRef
    from whetstone.optimization.gepa.contracts import GepaEffectContext
    from whetstone.optimization.gepa.effect_runtime import (
        DbosGepaEffectBroker,
        register_gepa_evaluation_authority,
    )
    from whetstone.optimization.gepa.engine import GepaDetailedResult
    from whetstone.optimization.gepa.runner import (
        DbosGepaRunner,
        GepaParentRunRequest,
        register_gepa_adapter_factory,
    )
    from whetstone.optimization.gepa.source import (
        GEPA_SOURCE_MANIFEST_HASH,
    )
    from whetstone.optimization.gepa.upstream_adapter import (
        GEPA_UPSTREAM_ADAPTER_IDENTITY_HASH,
        WhetstoneGepaAdapter,
    )

    suffix = uuid4().hex[:10]
    database_url = os.environ["WHETSTONE_TEST_POSTGRES_DSN"]
    config: DBOSConfig = {
        "name": f"gepa-recovery-{suffix}",
        "system_database_url": database_url,
        "application_database_url": database_url,
        "application_version": f"gepa-parent-recovery-{suffix}",
        "run_admin_server": False,
        "use_listen_notify": False,
    }
    DBOS(config=config)
    terminal_step_calls = 0

    @DBOS.step(retries_allowed=False)
    def terminal_step(control_hash: str) -> TypedRef:
        nonlocal terminal_step_calls
        terminal_step_calls += 1
        return typed_ref_for_record(
            "whetstone.gepa.run_result_artifact",
            {"control": control_hash, "suffix": suffix},
        )

    services = prompt_services()
    control = gepa_control().model_copy(
        update={
            "component_names": ("alpha", "beta"),
            "num_predictors": 2,
            "prompt_format_identity_hash": (
                services.descriptor.identity_hash()
            ),
            "prompt_binding_identity_hash": (services.binding.identity_hash()),
        }
    )
    data = GepaDataInstance(
        upstream_position=0,
        data_id=control.trainset_task_identities[0],
        data_ref=typed_ref_for_record(
            "test.gepa.data",
            {"id": "train"},
        ),
        loader_identity_hash="f" * 64,
    )
    evaluation_binding = evaluation_authority_binding()

    class EvaluationAuthority:
        runtime_identity_hash = evaluation_binding.authority_identity_hash
        calls = 0

        def evaluate(self, request):
            self.calls += 1
            return evaluation_result(request)

    authority = EvaluationAuthority()
    store = ObjectStore(
        SqliteBackend(tmp_path / "parent-recovery-effects.sqlite")
    )

    class RecoveryFactory:
        runtime_identity_hash = typed_ref_for_record(
            "test.gepa.recovery_factory",
            {"suffix": suffix},
        ).content_hash

        def __init__(self) -> None:
            self.crash_after_terminal_once = True
            self.create_calls = 0

        def create(self, *, control):
            self.create_calls += 1
            register_gepa_evaluation_authority(
                authority.runtime_identity_hash,
                authority,
            )
            return WhetstoneGepaAdapter(
                context=GepaEffectContext(
                    run_id=f"gepa:parent-recovery:{suffix}",
                    control_identity_hash=control.identity_hash(),
                    source_manifest_identity_hash=(GEPA_SOURCE_MANIFEST_HASH),
                    adapter_identity_hash=(
                        GEPA_UPSTREAM_ADAPTER_IDENTITY_HASH
                    ),
                ),
                broker=DbosGepaEffectBroker(store),
                evaluation_authority=evaluation_binding,
                proposal_authority=proposal_authority_binding(services),
                prompt_services=services,
            )

        def persist_result(self, *, control, adapter, detailed_result):
            del adapter, detailed_result
            artifact_ref = terminal_step(control.identity_hash())
            if self.crash_after_terminal_once:
                self.crash_after_terminal_once = False
                raise DBOSWorkflowCancelledError(
                    "injected interruption after terminal step"
                )
            return artifact_ref

    factory = RecoveryFactory()
    register_gepa_adapter_factory(cast(Any, factory))
    request = GepaParentRunRequest(
        factory_identity_hash=factory.runtime_identity_hash,
        control=control,
        seed_candidate={"alpha": "unchanged", "beta": "unchanged"},
        trainset=(data,),
    )

    def fake_run_gepa_engine(**kwargs):
        adapter = kwargs["adapter"]
        adapter.evaluate(
            list(kwargs["trainset"]),
            dict(kwargs["seed_candidate"]),
        )
        return GepaDetailedResult(
            candidates=({"alpha": "unchanged", "beta": "unchanged"},),
            parents=((),),
            val_aggregate_scores=(0.0,),
            val_subscores=({control.valset_task_identities[0]: 0.0},),
            per_val_instance_best_candidates={
                control.valset_task_identities[0]: (0,)
            },
            discovery_eval_counts=(0,),
            seed=control.seed,
            best_idx=0,
            control_identity_hash=control.identity_hash(),
        )

    monkeypatch.setattr(
        "whetstone.optimization.gepa.adapter.run_gepa_engine",
        fake_run_gepa_engine,
    )
    client = None
    try:
        DBOS.launch()
        runner = DbosGepaRunner()
        with pytest.raises(
            DBOSAwaitedWorkflowCancelledError,
        ):
            runner.run(request)
        assert authority.calls == terminal_step_calls == 1

        client = DBOSClient(system_database_url=database_url)
        workflow_id = f"whetstone-gepa-run-{request.identity_hash()}"
        recovered = client.resume_workflow(workflow_id).get_result()

        assert recovered.artifact_ref.schema_name == (
            "whetstone.gepa.run_result_artifact"
        )
        assert authority.calls == 1
        assert terminal_step_calls == 1
        assert factory.create_calls == 2
    finally:
        if client is not None:
            client.destroy()
        DBOS.destroy()
