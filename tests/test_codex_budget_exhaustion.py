"""The Step's ``tool_calls`` budget stops dispatch before the tool runs.

Tool capacity refuses a call after admission; the Issued Tool Call ledger
refuses it *before* dispatch. Both are bound to the same number by
construction, so this test pins the ledger's own limit -- read from
``budget.remaining["tool_calls"]`` -- independently of capacity.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from dr_store.sync import open_sqlite

from tests.codex_support import (
    toy_capacity_binding,
    toy_codex_control,
    toy_codex_run,
    toy_codex_step_request,
    toy_tool_args,
)
from whetstone.core.identity import ImmutableJsonObject
from whetstone.core.leasing import EffectLeaseAuthority, ReplayPolicy
from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
from whetstone.optim.contracts import step_request_reference
from whetstone.optim.tools.contracts import ToolCall, tool_config_reference
from whetstone.optim.tools.evaluator import EngineToolEvaluator
from whetstone.optim.tools.execution import EvaluatingToolExecutor
from whetstone.optim.tools.facade import (
    ToolAdmissionAuthority,
    ToolCallStore,
)
from whetstone.optim.tools.issued import _IssuedToolCallLedger

LEDGER_BUDGET = 2


@pytest.fixture
def codex_ledger(tmp_path):
    """A ledger bound to a Step whose ``tool_calls`` budget is two."""
    with open_sqlite(str(tmp_path / "budget.sqlite")) as store:
        engine = ReferenceEvalRuntimeConfig().build_engine(store)
        # Capacity is deliberately wider than the Step budget so the
        # ledger's own limit is what this test observes.
        control = toy_codex_control(engine=engine, max_tool_calls=8)
        run, config, candidate = toy_codex_run(
            control=control, engine=engine
        )
        request = toy_codex_step_request(
            control=control,
            run=run,
            candidate=candidate,
            tool_calls=LEDGER_BUDGET,
        )
        effect_authority = EffectLeaseAuthority.memory()
        tool_store = ToolCallStore(
            store,
            ToolAdmissionAuthority.memory(),
            effect_authority,
        )
        handle = EvaluatingToolExecutor(
            EngineToolEvaluator(engine),
            engine.reward_policy,
            effect_authority,
            owner_id="codex-budget-test",
            replay_policy=ReplayPolicy.IDEMPOTENT,
            lease_duration=timedelta(minutes=5),
        ).runtime_handle(config, tool_store, toy_capacity_binding(run))
        ledger = _IssuedToolCallLedger(
            store=store,
            tool_store=tool_store,
            request=step_request_reference(request),
        )
        yield ledger, handle, config, run, candidate, engine


def _call(*, call_id, config, run, candidate, engine, template):
    return ToolCall(
        call_id=call_id,
        tool_config=tool_config_reference(config),
        capacity_binding=toy_capacity_binding(run),
        args=ImmutableJsonObject(
            toy_tool_args(
                candidate=candidate, engine=engine, template=template
            )
        ),
    )


def test_the_call_after_the_budget_is_refused_before_dispatch(
    codex_ledger,
) -> None:
    ledger, handle, config, run, candidate, engine = codex_ledger

    for ordinal in range(1, LEDGER_BUDGET + 1):
        ledger.issue(
            _call(
                call_id=f"c{ordinal}",
                config=config,
                run=run,
                candidate=candidate,
                engine=engine,
                template=f"Answer {{prompt}} in {ordinal} words.",
            ),
            handle,
        )

    with pytest.raises(
        ValueError, match="Tool Call budget exhausted before dispatch"
    ):
        ledger.issue(
            _call(
                call_id="c-over",
                config=config,
                run=run,
                candidate=candidate,
                engine=engine,
                template="Answer {prompt} in one word.",
            ),
            handle,
        )


def test_every_issued_call_has_a_durable_terminal(codex_ledger) -> None:
    ledger, handle, config, run, candidate, engine = codex_ledger

    for ordinal in range(1, LEDGER_BUDGET + 1):
        ledger.issue(
            _call(
                call_id=f"c{ordinal}",
                config=config,
                run=run,
                candidate=candidate,
                engine=engine,
                template=f"Answer {{prompt}} in {ordinal} words.",
            ),
            handle,
        )

    evidence = ledger.evidence()

    assert len(evidence) == LEDGER_BUDGET
    for entry in evidence:
        assert entry.store_entry is not None
        assert entry.result is not None


def test_the_ledger_budget_delta_counts_the_issued_calls(
    codex_ledger,
) -> None:
    from whetstone.optim.contracts import BudgetDelta

    ledger, handle, config, run, candidate, engine = codex_ledger
    ledger.issue(
        _call(
            call_id="c1",
            config=config,
            run=run,
            candidate=candidate,
            engine=engine,
            template="Answer {prompt} in one word.",
        ),
        handle,
    )

    delta = ledger.budget_delta(BudgetDelta(), issued_count=1)

    # The adapter cannot understate what the Step spent: the ledger
    # overwrites whatever tool_calls figure the adapter reported.
    assert delta.consumed["tool_calls"] == 1
    overstated = ledger.budget_delta(
        BudgetDelta(consumed=ImmutableJsonObject({"tool_calls": 99})),
        issued_count=1,
    )
    assert overstated.consumed["tool_calls"] == 1


def test_a_duplicate_call_id_within_one_step_is_rejected(
    codex_ledger,
) -> None:
    ledger, handle, config, run, candidate, engine = codex_ledger
    call = _call(
        call_id="c1",
        config=config,
        run=run,
        candidate=candidate,
        engine=engine,
        template="Answer {prompt} in one word.",
    )
    ledger.issue(call, handle)

    with pytest.raises(ValueError, match="Tool Call IDs must be unique"):
        ledger.issue(call, handle)
