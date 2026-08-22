"""Codex's registered step contract: one terminal, tool-using step."""

from __future__ import annotations

import pytest

from tests.codex_support import toy_codex_control, toy_codex_run
from whetstone.coordination.step_contracts import (
    resolve_step_contract_provider,
    step_contract_provider_keys,
)
from whetstone.optim.codex.adapter import CODEX_ADAPTER_KEY
from whetstone.optim.codex.step_contract import (
    CODEX_TOOL_CALLS_BUDGET_LABEL,
    CodexStepContractProvider,
)
from whetstone.optim.contracts import (
    StepKind,
    optimization_run_reference,
)


def test_codex_is_registered_under_its_own_adapter_key() -> None:
    assert CODEX_ADAPTER_KEY in step_contract_provider_keys()
    provider = resolve_step_contract_provider(CODEX_ADAPTER_KEY)
    assert isinstance(provider, CodexStepContractProvider)
    assert provider.adapter_key == CODEX_ADAPTER_KEY
    assert provider.requires_control() is True


def test_parse_control_round_trips_the_persisted_payload(
    codex_engine,
) -> None:
    control = toy_codex_control(engine=codex_engine)
    provider = resolve_step_contract_provider(CODEX_ADAPTER_KEY)

    parsed = provider.parse_control(control.model_dump(mode="json"))

    assert parsed == control
    assert parsed.identity_hash() == control.identity_hash()


def test_build_first_binds_the_tool_calls_budget(codex_engine) -> None:
    control = toy_codex_control(engine=codex_engine, max_tool_calls=5)
    run, _config, candidate = toy_codex_run(
        control=control, engine=codex_engine
    )
    provider = resolve_step_contract_provider(CODEX_ADAPTER_KEY)

    request = provider.build_first(
        store=None,
        run=optimization_run_reference(run),
        initial_candidate=candidate,
        control=control,
        extra_pools=None,
    )

    assert request.step_index == 0
    assert request.step_id == f"{run.run_id}:codex:0"
    assert request.kind is StepKind.TOOL
    assert request.budget.remaining[CODEX_TOOL_CALLS_BUDGET_LABEL] == 5
    contract = request.step_output_contract
    assert contract.returned_proposal_count == 0
    # A search-dependent terminal cardinality is what permits a terminal
    # step to keep the seed instead of accepting a candidate.
    assert contract.terminal_proposal_count is not None


def test_the_step_carries_the_run_tool_config(codex_engine) -> None:
    control = toy_codex_control(engine=codex_engine)
    run, config, candidate = toy_codex_run(
        control=control, engine=codex_engine
    )
    provider = resolve_step_contract_provider(CODEX_ADAPTER_KEY)

    request = provider.build_first(
        store=None,
        run=optimization_run_reference(run),
        initial_candidate=candidate,
        control=control,
        extra_pools=None,
    )

    assert len(request.tool_configs) == 1
    assert request.tool_configs[0].record == config
    # The Step's tool_calls budget and the Tool Config's per-run capacity
    # are the same number by construction; they cannot drift.
    assert (
        request.budget.remaining[CODEX_TOOL_CALLS_BUDGET_LABEL]
        == config.capacity.max_accepted_calls
    )


def test_build_first_requires_a_codex_control(codex_engine) -> None:
    control = toy_codex_control(engine=codex_engine)
    run, _config, candidate = toy_codex_run(
        control=control, engine=codex_engine
    )
    provider = resolve_step_contract_provider(CODEX_ADAPTER_KEY)
    run_ref = optimization_run_reference(run)

    with pytest.raises(ValueError, match="requires the exact control"):
        provider.build_first(
            store=None,
            run=run_ref,
            initial_candidate=candidate,
            control=None,
            extra_pools=None,
        )
    with pytest.raises(TypeError, match="requires CodexControl"):
        provider.build_first(
            store=None,
            run=run_ref,
            initial_candidate=candidate,
            control=object(),
            extra_pools=None,
        )


def test_build_next_refuses_a_second_step(codex_engine) -> None:
    control = toy_codex_control(engine=codex_engine)
    provider = resolve_step_contract_provider(CODEX_ADAPTER_KEY)

    class _Prior:
        step_index = 0

    with pytest.raises(ValueError, match="exactly one opaque step"):
        provider.build_next(
            store=None,
            prior=_Prior(),
            prior_ref=None,
            prior_results=(),
            control=control,
            mutation_field="user_prompt_template",
            extra_pools=None,
        )
