"""GEPA DBOS replay pathway tests."""

from __future__ import annotations

import os
from typing import Any, Literal, cast
from uuid import uuid4

import pytest
from dr_store import ObjectStore, SqliteBackend

from tests.optimization.gepa.support import (
    evaluation_authority_binding,
    evaluation_result,
    gepa_control,
    make_gepa_detailed_result,
    prompt_services,
    proposal_authority_binding,
)
from tests.pathways.gepa.support import (
    DurableAuthority,
    GepaAdapterFactory,
    evaluation_request,
    proposal_request,
    proposal_result,
)
from whetstone.core.identity import TypedRef, typed_ref_for_record
from whetstone.optimization.gepa.contracts import (
    GepaDataInstance,
    GepaEvaluationEffectRequest,
    GepaProposalEffectRequest,
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
    factory = GepaAdapterFactory(control, identity_salt=suffix)
    register_gepa_adapter_factory(cast(Any, factory))
    request = GepaParentRunRequest(
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
        data_id=control.trainset_task_hashes[0],
        data_ref=typed_ref_for_record(
            "test.gepa.data",
            {"id": "train"},
        ),
        loader_identity_hash="f" * 64,
    )
    evaluation_binding = evaluation_authority_binding()

    class EvaluationAuthority:
        runtime_hash = evaluation_binding.authority_identity_hash
        calls = 0

        def evaluate(self, request):
            self.calls += 1
            return evaluation_result(request)

    authority = EvaluationAuthority()
    store = ObjectStore(
        SqliteBackend(tmp_path / "parent-recovery-effects.sqlite")
    )

    class RecoveryFactory:
        runtime_hash = typed_ref_for_record(
            "test.gepa.recovery_factory",
            {"suffix": suffix},
        ).content_hash

        def __init__(self) -> None:
            self.crash_after_terminal_once = True
            self.create_calls = 0

        def create(self, *, control):
            self.create_calls += 1
            register_gepa_evaluation_authority(
                authority.runtime_hash,
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
        factory_identity_hash=factory.runtime_hash,
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
            val_subscores=({control.valset_task_hashes[0]: 0.0},),
            per_val_instance_best_candidates={
                control.valset_task_hashes[0]: (0,)
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


@pytest.mark.skipif(
    "WHETSTONE_TEST_POSTGRES_DSN" not in os.environ,
    reason="WHETSTONE_TEST_POSTGRES_DSN is required for real DBOS replay",
)
@pytest.mark.parametrize("effect_kind", ["evaluate", "propose"])
@pytest.mark.postgres_integration
def test_real_dbos_child_checkpoint_survives_outer_bind_crash(
    tmp_path,
    effect_kind: Literal["evaluate", "propose"],
) -> None:
    from dbos import DBOS, DBOSConfig

    from whetstone.optimization.gepa.effect_runtime import (
        DbosGepaEffectBroker,
        register_gepa_evaluation_authority,
        register_gepa_proposal_authority,
    )

    suffix = uuid4().hex[:10]
    database_url = os.environ["WHETSTONE_TEST_POSTGRES_DSN"]
    config: DBOSConfig = {
        "name": f"gepa-effect-{suffix}",
        "system_database_url": database_url,
        "application_database_url": database_url,
        "application_version": f"gepa-effect-{suffix}",
        "run_admin_server": False,
        "use_listen_notify": False,
    }
    DBOS(config=config)
    if effect_kind == "evaluate":
        request = evaluation_request()
    else:
        request = proposal_request()
    context = request.slot.context.model_copy(
        update={"run_id": f"gepa:real-dbos:{suffix}"}
    )
    authority_hash = typed_ref_for_record(
        "test.gepa.real_authority",
        {"suffix": suffix},
    ).content_hash
    authority_binding = request.authority.model_copy(
        update={"authority_identity_hash": authority_hash}
    )
    request = request.model_copy(
        update={
            "slot": request.slot.model_copy(update={"context": context}),
            "authority": authority_binding,
        }
    )
    if effect_kind == "evaluate":
        result = evaluation_result(
            GepaEvaluationEffectRequest.model_validate(request)
        )
    else:
        result = proposal_result(
            GepaProposalEffectRequest.model_validate(request)
        )
    authority = DurableAuthority(authority_hash, result)
    if effect_kind == "evaluate":
        register_gepa_evaluation_authority(authority_hash, authority)
    else:
        register_gepa_proposal_authority(authority_hash, authority)
    database = tmp_path / f"real-dbos-{effect_kind}-effect.sqlite"
    store = ObjectStore(SqliteBackend(database))
    first = DbosGepaEffectBroker(store)
    recorder_method = (
        "record_evaluation_result"
        if effect_kind == "evaluate"
        else "record_proposal_result"
    )
    original_record = getattr(first._recorder, recorder_method)

    def crash_before_bind(*_args, **_kwargs):
        raise RuntimeError("injected crash before outer result bind")

    setattr(first._recorder, recorder_method, crash_before_bind)
    try:
        DBOS.launch()
        with pytest.raises(RuntimeError, match="before outer result bind"):
            getattr(first, effect_kind)(request)
        assert authority.calls == 1
        setattr(first._recorder, recorder_method, original_record)
        replay = getattr(
            DbosGepaEffectBroker(ObjectStore(SqliteBackend(database))),
            effect_kind,
        )(request)
        assert replay == result
        assert authority.calls == 1
    finally:
        DBOS.destroy()
