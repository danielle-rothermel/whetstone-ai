"""The live-cell wiring test: the FULL live path constructs for every optimizer
kind with network calls stubbed at the HTTP boundary, and the fixture-only
proposer seam is UNREACHABLE from the live entrypoint for any proposal-using
optimizer.

This is the test that would have caught the live-run blocker: a COPRO/MIPROv2/
GEPA cell that reaches ``_LiveProposerUnavailable`` (which raises
``RuntimeError: live proposer transport is not wired ...``). It drives
:func:`_build_cell_config` -- the exact Config the live ``cell`` subcommand
builds -- for every optimizer kind, asserting (1) construction succeeds with no
network, (2) a real proposal-using optimizer wires the live
``_HttpProposerTransport`` (never the raising placeholder), and (3) driving a
proposer draft through the REAL dr-providers stack over an
``httpx.MockTransport`` HTTP-boundary stub round-trips with no fixture-only
seam reachable.

Pre-fix (proposal-using optimizers wired to ``_LiveProposerUnavailable``) the
construction assertions fail for copro/miprov2 and the draft raises the
fixture-only RuntimeError -- so this test fails against that code.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

import httpx
import pytest
from dr_providers import ProviderTransportPolicy
from dr_providers.transport import HttpProvider

from whetstone.optimization.proposer import ProposalRequest
from whetstone.runner.cli import (
    _build_cell_config,
    _HttpProposerTransport,
    _LiveProposerUnavailable,
)
from whetstone.runner.optimizers import OPTIMIZERS

#: Optimizer kinds that draft real proposals through the live proposer route.
#: (``gepa`` reflects through the same ProposerTransport seam; ``codex`` uses
#: its own MCP bridge, ``eval`` never drafts -- both keep the placeholder.)
_PROPOSAL_USING = ("copro", "miprov2", "gepa")
_NO_PROPOSER = ("eval", "codex")


def _cell_args(optimizer: str, **overrides: object) -> argparse.Namespace:
    base: dict[str, object] = dict(
        optimizer=optimizer,
        env="c11",
        lane="openrouter",
        attempt=0,
        task_model=None,
        proposer_model=None,
        non_canonical=False,
        execution_mode="in-process",
        concurrency=4,
        max_wall_seconds=3600.0,
        official_n=None,
        official_repeats=None,
        missing_data=None,
        max_skip_fraction=None,
        dry_run_fake=False,
        live=True,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _chat_completion_handler(
    text: str,
) -> Callable[[httpx.Request], httpx.Response]:
    """An httpx.MockTransport handler: a well-formed chat-completions body."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "resp-wiring",
                "model": "openai/gpt-5.4-nano",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 7,
                    "total_tokens": 18,
                },
            },
        )

    return handler


def _http_boundary_transport(
    policy: ProviderTransportPolicy, text: str
) -> HttpProvider:
    """A REAL dr-providers HttpProvider whose only stub is the httpx socket.

    Construction of the full transport/route/policy stack is exercised over the
    SAME transport policy the live cell would drive; only the wire call is
    intercepted by ``httpx.MockTransport``. ``api_key`` is supplied so the
    header build succeeds without reading the environment.
    """
    client = httpx.Client(
        transport=httpx.MockTransport(_chat_completion_handler(text))
    )
    return HttpProvider(policy=policy, client=client, api_key="test-key")


@pytest.mark.parametrize("optimizer", OPTIMIZERS)
def test_live_cell_path_constructs_for_every_optimizer(
    optimizer: str,
) -> None:
    # The full live Config builds with NO network for every optimizer kind:
    # task/proposer routes, execution policies, and both live transports.
    config, task_route = _build_cell_config(_cell_args(optimizer))
    assert task_route.lane == "openrouter"
    assert config.optimizer == optimizer
    # The rollout (task) transport is always the live dr-providers transport.
    assert callable(config.rollout_transport)


@pytest.mark.parametrize("optimizer", _PROPOSAL_USING)
def test_proposal_using_optimizer_never_reaches_fixture_seam(
    optimizer: str,
) -> None:
    # A proposal-using optimizer on the live path MUST wire the live proposer
    # transport, never the raising fixture-only placeholder.
    config, _ = _build_cell_config(_cell_args(optimizer))
    assert isinstance(config.proposer_transport, _HttpProposerTransport)
    assert not isinstance(config.proposer_transport, _LiveProposerUnavailable)


@pytest.mark.parametrize("optimizer", _NO_PROPOSER)
def test_non_proposal_optimizer_keeps_placeholder(optimizer: str) -> None:
    # eval never drafts; codex uses its own MCP bridge -- both keep the
    # placeholder, which is fine because run_optimize never calls draft() for
    # them (eval is identity; codex bridges elsewhere).
    config, _ = _build_cell_config(_cell_args(optimizer))
    assert isinstance(config.proposer_transport, _LiveProposerUnavailable)


@pytest.mark.parametrize("optimizer", _PROPOSAL_USING)
def test_live_proposer_drafts_over_http_boundary_stub(
    optimizer: str,
) -> None:
    # Drive the constructed proposer transport's draft() through the REAL
    # dr-providers stack, stubbing ONLY the httpx socket. This proves no
    # fixture-only seam is reachable when the proposer actually runs: a real
    # round-trip returns the completion text as the drafted template.
    config, _ = _build_cell_config(_cell_args(optimizer))
    transport = config.proposer_transport
    assert isinstance(transport, _HttpProposerTransport)
    # Re-point the transport at the HTTP-boundary stub using its OWN route
    # policy, so the whole payload/parse/attempt-loop path runs with no net.
    route = transport._route  # type: ignore[attr-defined]
    boundary = _http_boundary_transport(
        route.transport_policy, "Rewritten template: {input}"
    )
    live = _HttpProposerTransport(route, transport=boundary.invoke)
    request = ProposalRequest(
        proposal_mode="seed_proposal",
        request_ordinal=0,
        base_ref="wiring",
        base_template="Answer: {input}",
    )
    drafts = live.draft(config.proposer_config, request, count=2)
    assert len(drafts) == 2
    for draft in drafts:
        assert draft.template == "Rewritten template: {input}"
        assert draft.usage["proposer_calls"] == 1
    # Proposer token accounting is tallied for the cell heartbeat.
    assert live.proposer_calls == 2
    assert live.proposer_tokens == 36  # 18 tokens x 2 drafts


def test_placeholder_draft_raises_fixture_only_runtime_error() -> None:
    # The fixture-only seam still raises loudly if ever reached from a live
    # run -- the regression the wiring test guards against.
    request = ProposalRequest(
        proposal_mode="seed_proposal", request_ordinal=0, base_ref="x"
    )
    config, _ = _build_cell_config(_cell_args("copro"))
    with pytest.raises(RuntimeError, match="live proposer transport is not"):
        _LiveProposerUnavailable().draft(config.proposer_config, request, 1)
