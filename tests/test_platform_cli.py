from __future__ import annotations

from datetime import timedelta

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
from whetstone.testing.toy.experiment import build_toy_experiment


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
