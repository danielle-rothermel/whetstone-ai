from __future__ import annotations

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
