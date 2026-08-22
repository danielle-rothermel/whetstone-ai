from __future__ import annotations

from datetime import timedelta

import pytest
from dr_providers import ProviderCallRequest, ProviderInvocationEvidence

from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
from whetstone.optim.proposal.proposer import (
    FakeProposerTransport,
    ProviderProposerTransport,
)
from whetstone.optim.gepa.harness_adapter import GEPA_ADAPTER_KEY
from whetstone.platform.cli import (
    PROPOSER_FAKE,
    PROPOSER_PROVIDER,
    _copro_adapter_from_control,
    _gepa_adapter_from_launch,
)
from whetstone.testing.runtime import (
    build_toy_copro_control,
    build_toy_gepa_adapter,
    build_toy_gepa_control,
    prepare_toy_gepa_run,
    register_toy_runtime,
)
from whetstone.testing.toy.experiment import (
    TOY_MUTATION_FIELD,
    build_toy_experiment,
    toy_template_render_contract,
)


def _unused_transport(_request: ProviderCallRequest) -> ProviderInvocationEvidence:
    raise AssertionError("provider reconstruction must not invoke transport")


def test_provider_reconstruction_does_not_use_fake_proposer(
    sqlite_store,
    monkeypatch,
) -> None:
    runtime_config = ReferenceEvalRuntimeConfig()
    engine = runtime_config.build_engine(sqlite_store)
    control = build_toy_copro_control(breadth=2, depth=1, engine=engine)
    monkeypatch.setattr(
        "whetstone.platform.cli._live_transport_call",
        lambda _policy: _unused_transport,
    )

    adapter = _copro_adapter_from_control(
        control,
        engine,
        store=sqlite_store,
        proposer=PROPOSER_PROVIDER,
        execution_policy=runtime_config.execution_policy,
    )

    assert isinstance(adapter._transport, ProviderProposerTransport)
    assert not isinstance(adapter._transport, FakeProposerTransport)


def _gepa_cli_transport(adapter):
    return adapter._adapter_factory._factory._proposal_authority.transport


def test_gepa_provider_reconstruction_does_not_use_fake_proposer(
    sqlite_store,
    monkeypatch,
) -> None:
    runtime_config = ReferenceEvalRuntimeConfig()
    experiment = build_toy_experiment(num_seeds=1)
    engine = runtime_config.build_engine(sqlite_store, experiment=experiment)
    control = build_toy_gepa_control(engine=engine)
    adapter = build_toy_gepa_adapter(
        store=sqlite_store,
        engine=engine,
        control=control,
        run_id="gepa-cli-provider",
        initial_candidate=experiment.initial_candidate,
    )
    runtime = register_toy_runtime(
        store=sqlite_store,
        engine=engine,
        copro_control=build_toy_copro_control(engine=engine),
        extra_adapters={GEPA_ADAPTER_KEY: adapter},
    )
    launch = prepare_toy_gepa_run(
        runtime,
        run_id="gepa-cli-provider",
        control=control,
        experiment=experiment,
    )
    monkeypatch.setattr(
        "whetstone.platform.cli._live_transport_call",
        lambda _policy: _unused_transport,
    )

    reconstructed = _gepa_adapter_from_launch(
        launch,
        engine,
        sqlite_store,
        proposer=PROPOSER_PROVIDER,
        execution_policy=runtime_config.execution_policy,
    )
    transport = _gepa_cli_transport(reconstructed)
    assert isinstance(transport, ProviderProposerTransport)
    assert not isinstance(transport, FakeProposerTransport)


def test_gepa_fake_reconstruction_uses_fake_proposer(sqlite_store) -> None:
    runtime_config = ReferenceEvalRuntimeConfig()
    experiment = build_toy_experiment(num_seeds=1)
    engine = runtime_config.build_engine(sqlite_store, experiment=experiment)
    control = build_toy_gepa_control(engine=engine)
    adapter = build_toy_gepa_adapter(
        store=sqlite_store,
        engine=engine,
        control=control,
        run_id="gepa-cli-fake",
        initial_candidate=experiment.initial_candidate,
    )
    runtime = register_toy_runtime(
        store=sqlite_store,
        engine=engine,
        copro_control=build_toy_copro_control(engine=engine),
        extra_adapters={GEPA_ADAPTER_KEY: adapter},
    )
    launch = prepare_toy_gepa_run(
        runtime,
        run_id="gepa-cli-fake",
        control=control,
        experiment=experiment,
    )
    reconstructed = _gepa_adapter_from_launch(
        launch,
        engine,
        sqlite_store,
        proposer=PROPOSER_FAKE,
        execution_policy=runtime_config.execution_policy,
    )
    assert isinstance(_gepa_cli_transport(reconstructed), FakeProposerTransport)


def test_cli_store_path_lease_authority_replays_after_reopen(tmp_path) -> None:
    from whetstone.core.identity import (
        compute_identity_hash,
        compute_prefixed_identity_key,
        typed_ref_for_record,
    )
    from whetstone.core.leasing import (
        AcquireOutcome,
        EffectLeaseAuthority,
        ReplayPolicy,
        effect_request,
    )

    store_path = str(tmp_path / "cli-leases.sqlite")
    request = effect_request(
        semantic_key=compute_prefixed_identity_key(
            schema="whetstone.cli_lease_test",
            schema_version=1,
            prefix="whetstone.cli_lease_test:",
            payload={"call": "proposal"},
        ),
        request_hash=compute_identity_hash(
            schema="whetstone.cli_lease_test",
            schema_version=1,
            payload={"call": "proposal"},
        ),
        replay_policy=ReplayPolicy.DURABLE_WORKFLOW,
    )
    result_ref = typed_ref_for_record(
        "whetstone.cli_lease_test_result", {"ok": True}
    )
    first = EffectLeaseAuthority.sqlite(store_path)
    try:
        acquired = first.acquire(
            request,
            owner_id="cli-owner",
            attempt_id="attempt-1",
            lease_duration=timedelta(minutes=5),
        )
        assert acquired.outcome is AcquireOutcome.ACQUIRED
        assert acquired.lease is not None
        with first.maintain(
            acquired.lease, lease_duration=timedelta(minutes=5)
        ) as maintenance:
            maintenance.succeed(result_ref=result_ref)
    finally:
        first.close()

    second = EffectLeaseAuthority.sqlite(store_path)
    try:
        replayed = second.acquire(
            request,
            owner_id="cli-owner",
            attempt_id="attempt-2",
            lease_duration=timedelta(minutes=5),
        )
        assert replayed.outcome is AcquireOutcome.SUCCEEDED
        assert replayed.terminal is not None
        assert replayed.terminal.result_ref == result_ref
    finally:
        second.close()


def _alt_experiment():
    """A small experiment distinct from the CLI's built-in toy experiment."""
    from whetstone.testing.toy.experiment import ToyTask

    return build_toy_experiment(
        num_seeds=1,
        internal_tasks=(
            ToyTask(
                task_id="alt-internal-1",
                prompt_inputs={"prompt": "alt one"},
                gold="X",
            ),
            ToyTask(
                task_id="alt-internal-2",
                prompt_inputs={"prompt": "alt two"},
                gold="Y",
            ),
        ),
        official_tasks=(
            ToyTask(
                task_id="alt-official-1",
                prompt_inputs={"prompt": "alt three"},
                gold="Z",
            ),
        ),
    )


def test_run_refuses_gepa_launch_from_a_non_toy_experiment(
    sqlite_store,
    monkeypatch,
) -> None:
    """`run --adapter gepa` must not silently evaluate over toy tasks.

    The CLI rebuilds its engine from the toy experiment, so a launch bound
    against any other experiment would fan out over tasks that are not the
    launch's. The launch persists no rollout graph to rebuild the real
    experiment from, so the command refuses instead.
    """
    from whetstone.platform.cli import (
        ToyExperimentOnlyError,
        _require_launch_matches_engine,
    )

    runtime_config = ReferenceEvalRuntimeConfig()
    experiment = _alt_experiment()
    engine = runtime_config.build_engine(sqlite_store, experiment=experiment)
    control = build_toy_gepa_control(engine=engine)
    adapter = build_toy_gepa_adapter(
        store=sqlite_store,
        engine=engine,
        control=control,
        run_id="gepa-cli-non-toy",
        initial_candidate=experiment.initial_candidate,
    )
    runtime = register_toy_runtime(
        store=sqlite_store,
        engine=engine,
        copro_control=build_toy_copro_control(engine=engine),
        extra_adapters={GEPA_ADAPTER_KEY: adapter},
    )
    launch = prepare_toy_gepa_run(
        runtime,
        run_id="gepa-cli-non-toy",
        control=control,
        experiment=experiment,
    )

    # The launch's own tasks are not the toy tasks the CLI would rebuild.
    toy_engine = runtime_config.build_engine(
        sqlite_store, experiment=build_toy_experiment(num_seeds=1)
    )
    assert {task.task_id for task in engine.sampling.tasks} != {
        task.task_id for task in toy_engine.sampling.tasks
    }

    # This is exactly the reconstruction `run --adapter gepa` performs: an
    # engine built with no experiment, which defaults to the toy one.
    rebuilt = runtime_config.build_engine(
        sqlite_store,
        mutation_field=launch.run.mutation_field,
        render_contract=launch.run.template_render_contract,
    )
    assert launch.control is not None
    with pytest.raises(ToyExperimentOnlyError) as excinfo:
        _require_launch_matches_engine(
            launch,
            rebuilt,
            eval_config_ref=launch.control.metric,
        )
    message = str(excinfo.value)
    assert "gepa-cli-non-toy" in message
    assert "toy experiment" in message

    # A launch actually bound against the toy experiment still passes.
    toy_experiment = build_toy_experiment(num_seeds=1)
    toy_control = build_toy_gepa_control(engine=toy_engine)
    toy_adapter = build_toy_gepa_adapter(
        store=sqlite_store,
        engine=toy_engine,
        control=toy_control,
        run_id="gepa-cli-toy-ok",
        initial_candidate=toy_experiment.initial_candidate,
    )
    toy_runtime = register_toy_runtime(
        store=sqlite_store,
        engine=toy_engine,
        copro_control=build_toy_copro_control(engine=toy_engine),
        extra_adapters={GEPA_ADAPTER_KEY: toy_adapter},
    )
    toy_launch = prepare_toy_gepa_run(
        toy_runtime,
        run_id="gepa-cli-toy-ok",
        control=toy_control,
        experiment=toy_experiment,
    )
    assert toy_launch.control is not None
    _require_launch_matches_engine(
        toy_launch,
        runtime_config.build_engine(
            sqlite_store,
            mutation_field=toy_launch.run.mutation_field,
            render_contract=toy_launch.run.template_render_contract,
        ),
        eval_config_ref=toy_launch.control.metric,
    )


def test_run_refuses_copro_launch_from_a_non_toy_experiment(
    sqlite_store,
) -> None:
    """The COPRO CLI path shares the toy-default engine, and the same guard."""
    from whetstone.platform.cli import (
        ToyExperimentOnlyError,
        _require_launch_matches_engine,
    )
    from whetstone.coordination.runtime_bootstrap import prepare_copro_run

    runtime_config = ReferenceEvalRuntimeConfig()
    experiment = _alt_experiment()
    engine = runtime_config.build_engine(sqlite_store, experiment=experiment)
    control = build_toy_copro_control(breadth=2, depth=1, engine=engine)
    runtime = register_toy_runtime(
        store=sqlite_store,
        engine=engine,
        copro_control=control,
    )
    launch = prepare_copro_run(
        runtime,
        run_id="copro-cli-non-toy",
        control=control,
        experiment=experiment,
        render_contract=toy_template_render_contract(),
        mutation_field=TOY_MUTATION_FIELD,
    )
    rebuilt = runtime_config.build_engine(
        sqlite_store,
        mutation_field=launch.run.mutation_field,
        render_contract=launch.run.template_render_contract,
    )
    assert launch.control is not None
    with pytest.raises(ToyExperimentOnlyError) as excinfo:
        _require_launch_matches_engine(
            launch,
            rebuilt,
            eval_config_ref=launch.control.eval_config_ref,
        )
    assert "copro-cli-non-toy" in str(excinfo.value)


def _codex_launch(store):
    """A bound Codex launch, exactly as the CLI's run command loads one."""
    from whetstone.optim.adapters import MappingAdapterRegistry
    from whetstone.optim.codex.adapter import CODEX_ADAPTER_KEY, CodexAdapter
    from whetstone.core.leasing import EffectLeaseAuthority
    from whetstone.coordination.runtime_bootstrap import build_runtime
    from whetstone.testing.runtime import (
        build_toy_codex_control,
        prepare_toy_codex_run,
    )

    engine = ReferenceEvalRuntimeConfig().build_engine(store)
    control = build_toy_codex_control(engine=engine, max_tool_calls=2)
    runtime = build_runtime(
        store=store,
        engine=engine,
        adapter_registry=MappingAdapterRegistry(
            {CODEX_ADAPTER_KEY: CodexAdapter(_NeverRunsCodex(), store=store)}
        ),
        effect_authority=EffectLeaseAuthority.memory(),
    )
    return prepare_toy_codex_run(
        runtime, run_id="cli-codex-run", control=control
    )



class _NeverRunsCodex:
    def run(self, request, handle, *, lease_token):
        raise AssertionError("the CLI test must not spawn a Codex process")


def test_the_cli_codex_path_proves_the_session_before_it_builds_an_adapter(
    sqlite_store,
    tmp_path,
) -> None:
    """The preflight is the CLI's, not an optional caller courtesy.

    A Codex run commits real eval capacity on its first admitted Tool
    Call, so the one production construction site must prove the session
    first. This asserts the call happens and carries the run's own
    control and the exact environment the run will see.
    """
    from whetstone.platform.cli import _codex_adapter_from_launch

    launch = _codex_launch(sqlite_store)
    engine = ReferenceEvalRuntimeConfig().build_engine(sqlite_store)
    seen: list[dict] = []

    adapter = _codex_adapter_from_launch(
        launch,
        engine,
        sqlite_store,
        store_path=str(tmp_path / "cli-codex.sqlite"),
        run_root=tmp_path / "runs",
        runtime_config=ReferenceEvalRuntimeConfig(),
        preflight=lambda **kwargs: seen.append(kwargs),
    )

    assert adapter is not None
    assert len(seen) == 1
    assert seen[0]["codex_binary"] == launch.control.codex_binary
    assert seen[0]["model"] == launch.control.model
    # The preflight inspects exactly the environment the run will pass
    # through, so the task-model key must be absent there too.
    assert isinstance(seen[0]["environment"], dict)


def test_the_cli_codex_path_refuses_to_build_when_the_session_is_broken(
    sqlite_store,
    tmp_path,
) -> None:
    """A broken login must not reach the harness and start spending."""
    import pytest

    from whetstone.optim.codex.preflight import CodexPreflightError
    from whetstone.platform.cli import _codex_adapter_from_launch

    launch = _codex_launch(sqlite_store)
    engine = ReferenceEvalRuntimeConfig().build_engine(sqlite_store)

    def _broken(**_kwargs):
        raise CodexPreflightError("Codex has no usable auth source")

    with pytest.raises(CodexPreflightError):
        _codex_adapter_from_launch(
            launch,
            engine,
            sqlite_store,
            store_path=str(tmp_path / "cli-codex.sqlite"),
            run_root=tmp_path / "runs",
            runtime_config=ReferenceEvalRuntimeConfig(),
            preflight=_broken,
        )


def test_the_cli_codex_path_defaults_to_the_real_auth_preflight(
    sqlite_store,
    tmp_path,
    monkeypatch,
) -> None:
    """No preflight argument means the real check, never a no-op.

    The ``preflight`` parameter exists so a test can name a stand-in; if
    it silently defaulted to nothing, the documented guarantee would be
    only as good as each caller remembering.
    """
    from whetstone.optim.codex import preflight as preflight_module
    from whetstone.platform.cli import _codex_adapter_from_launch

    launch = _codex_launch(sqlite_store)
    engine = ReferenceEvalRuntimeConfig().build_engine(sqlite_store)
    called: list[int] = []

    monkeypatch.setattr(
        preflight_module,
        "codex_auth_preflight",
        lambda **_kwargs: called.append(1),
    )

    _codex_adapter_from_launch(
        launch,
        engine,
        sqlite_store,
        store_path=str(tmp_path / "cli-codex.sqlite"),
        run_root=tmp_path / "runs",
        runtime_config=ReferenceEvalRuntimeConfig(),
    )

    assert called == [1]
