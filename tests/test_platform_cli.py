from __future__ import annotations

from datetime import timedelta

from dr_providers import ProviderCallRequest, ProviderInvocationEvidence

from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
from whetstone.optim.proposal.proposer import (
    FakeProposerTransport,
    ProviderProposerTransport,
)
from whetstone.platform.cli import (
    PROPOSER_PROVIDER,
    _copro_adapter_from_control,
)
from whetstone.testing.runtime import build_toy_copro_control


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
