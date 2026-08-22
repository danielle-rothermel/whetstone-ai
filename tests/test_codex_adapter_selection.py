"""What the adapter accepts as a scored, reported candidate.

These drive the real ``CodexAdapter`` against the real Tool Call Store,
admission authority, and ``EvaluatingToolExecutor`` with a stub runner
standing in for the Codex process, so they exercise the selection and
reconciliation contract on every platform rather than only under the
macOS sandbox.
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
from whetstone.core.identity import ImmutableJsonObject, TerminalFailure
from whetstone.core.leasing import EffectLeaseAuthority, ReplayPolicy
from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
from whetstone.optim.adapters import MappingAdapterRegistry
from whetstone.optim.codex.adapter import (
    CODEX_ADAPTER_KEY,
    CODEX_UNREPORTED_EVALUATION_CODE,
    CodexAdapter,
    CodexOutputArtifact,
    CodexRunResult,
    codex_lease_token_hash,
)
from whetstone.optim.contracts import StepStatus
from whetstone.optim.harness import OptimHarness
from whetstone.optim.tools.contracts import ToolCall, tool_config_reference
from whetstone.optim.tools.evaluator import (
    EngineToolEvaluator,
    ToolEvaluationError,
)
from whetstone.optim.tools.execution import EvaluatingToolExecutor
from whetstone.optim.tools.facade import (
    ToolAdmissionAuthority,
    ToolCallStore,
)

_TEMPLATE_A = "Answer {prompt} in one short sentence."
_TEMPLATE_B = "Answer {prompt} with a single friendly word."
_FIXED_LEASE_TOKEN = "e" * 64
_FORCED_FAILURE_CODE = "toy_forced_eval_failure"


class _FailingCallEvaluator(EngineToolEvaluator):
    """An evaluator that fails exactly the named calls.

    A terminally failed evaluation still reaches ``COMPLETED``: the
    executor persists a Tool Result carrying ``terminal_failure`` and
    completes the entry. That is the state this suite needs to build.
    """

    def __init__(self, engine, *, failing_call_ids: frozenset[str]) -> None:
        super().__init__(engine)
        self._failing_call_ids = failing_call_ids

    def evaluate(self, call, config):
        if str(call.call_id) in self._failing_call_ids:
            raise ToolEvaluationError(
                TerminalFailure(
                    code=_FORCED_FAILURE_CODE,
                    message="the toy evaluator was told to fail this call",
                    details={"call_id": str(call.call_id)},
                )
            )
        return super().evaluate(call, config)


class _ScriptedRunner:
    """Stands in for the Codex process without spawning one.

    It issues the scripted calls through the executor's own handle --
    exactly as the out-of-process MCP server does -- and then returns the
    artifact the agent would have written.
    """

    def __init__(self, world, *, calls, artifact_fields) -> None:
        self._world = world
        self._calls = calls
        self._artifact_fields = artifact_fields

    def run(self, request, handle, *, lease_token):
        del handle
        for call_id, template in self._calls:
            self._world.issue(call_id, template)
        fields = {
            "run_id": request.run_id,
            "evaluated_call_ids": tuple(
                call_id for call_id, _template in self._calls
            ),
            "lease_token_hash": codex_lease_token_hash(lease_token),
            **self._artifact_fields,
        }
        return CodexRunResult(artifact=CodexOutputArtifact(**fields))


class _SelectionWorld:
    def __init__(self, store, *, failing_call_ids: frozenset[str]) -> None:
        self.store = store
        self.engine = ReferenceEvalRuntimeConfig().build_engine(store)
        self.control = toy_codex_control(engine=self.engine, max_tool_calls=4)
        self.run, self.config, self.candidate = toy_codex_run(
            control=self.control, engine=self.engine
        )
        self.binding = toy_capacity_binding(self.run)
        self.effect_authority = EffectLeaseAuthority.memory()
        self.tool_store = ToolCallStore(
            store,
            ToolAdmissionAuthority.memory(),
            self.effect_authority,
        )
        self.tool_executor = EvaluatingToolExecutor(
            _FailingCallEvaluator(
                self.engine, failing_call_ids=failing_call_ids
            ),
            self.engine.reward_policy,
            self.effect_authority,
            owner_id="codex-selection-owner",
            replay_policy=ReplayPolicy.IDEMPOTENT,
            lease_duration=timedelta(minutes=5),
        )
        self._agent_handle = self.tool_executor.runtime_handle(
            self.config, self.tool_store, self.binding
        )

    def issue(self, call_id: str, template: str):
        """Make one real admitted, evaluated call, as the agent would."""
        return self._agent_handle(
            ToolCall(
                call_id=call_id,
                tool_config=tool_config_reference(self.config),
                capacity_binding=self.binding,
                args=ImmutableJsonObject(
                    toy_tool_args(
                        candidate=self.candidate,
                        engine=self.engine,
                        template=template,
                    )
                ),
            )
        )

    def run_step(self, *, calls, artifact_fields):
        adapter = CodexAdapter(
            _ScriptedRunner(
                self, calls=calls, artifact_fields=artifact_fields
            ),
            store=self.store,
            lease_token_factory=lambda: _FIXED_LEASE_TOKEN,
        )
        adapter.bind_tool_store(self.tool_store)
        harness = OptimHarness(
            store=self.store,
            adapter_registry=MappingAdapterRegistry(
                {CODEX_ADAPTER_KEY: adapter}
            ),
            tool_store=self.tool_store,
            effect_authority=self.effect_authority,
            owner_id="codex-selection-owner",
            adapter_replay_policy=ReplayPolicy.NO_REDRIVE,
            lease_duration=timedelta(minutes=5),
            tool_executor=self.tool_executor,
        )
        harness.bind_run(self.run)
        result, _ref = harness.run_step(
            toy_codex_step_request(
                control=self.control,
                run=self.run,
                candidate=self.candidate,
            )
        )
        return result


@pytest.fixture
def selection_world(tmp_path):
    with open_sqlite(str(tmp_path / "codex-selection.sqlite")) as store:

        def build(failing_call_ids: frozenset[str] = frozenset()):
            return _SelectionWorld(
                store, failing_call_ids=failing_call_ids
            )

        yield build


def test_a_terminally_failed_evaluation_cannot_be_the_selected_candidate(
    selection_world,
) -> None:
    """A COMPLETED call is not necessarily a scored call.

    ``EvaluatingToolExecutor`` completes an entry whose evaluation raised,
    persisting a Tool Result that carries ``terminal_failure`` and neither
    ``output`` nor ``reward``. Accepting that candidate would return a
    winner the agent never successfully measured.
    """
    world = selection_world(frozenset({"c2"}))

    result = world.run_step(
        calls=[("c1", _TEMPLATE_A), ("c2", _TEMPLATE_B)],
        artifact_fields={"selected_call_id": "c2"},
    )

    assert result.status is StepStatus.FAILED
    assert result.terminal_failure is not None
    # A Step Result carries one shared terminal failure, so a Step that
    # fails because its selected evaluation failed fails under that
    # evaluation's own failure rather than a second, adapter-owned code.
    assert result.terminal_failure.code == _FORCED_FAILURE_CODE
    assert result.terminal_failure.details["call_id"] == "c2"
    assert result.accepted_candidates == ()


def test_a_scored_sibling_of_a_failed_evaluation_is_still_selectable(
    selection_world,
) -> None:
    """Only the selected call must have scored; the rest may fail."""
    world = selection_world(frozenset({"c2"}))

    result = world.run_step(
        calls=[("c1", _TEMPLATE_A), ("c2", _TEMPLATE_B)],
        artifact_fields={"selected_call_id": "c1"},
    )

    assert result.terminal_failure is None, result.terminal_failure
    assert result.status is StepStatus.COMPLETE
    assert len(result.accepted_candidates) == 1
    accepted = world.store.get(
        result.accepted_candidates[0].record_ref.reference
    )
    assert accepted["payload"]["user_prompt_template"] == _TEMPLATE_A


def test_an_unreported_admitted_call_fails_the_step_without_a_sandbox(
    selection_world,
) -> None:
    """Ledger totality, asserted on every platform.

    The scripted runner makes two real admitted evaluations and names one,
    which is the under-reporting attack in its minimal form.
    """
    world = selection_world()

    result = world.run_step(
        calls=[("c1", _TEMPLATE_A), ("c2", _TEMPLATE_B)],
        artifact_fields={
            "selected_call_id": "c1",
            "evaluated_call_ids": ("c1",),
        },
    )

    assert result.status is StepStatus.FAILED
    assert result.terminal_failure is not None
    assert (
        result.terminal_failure.code == CODEX_UNREPORTED_EVALUATION_CODE
    )
    assert result.terminal_failure.details["admitted_call_count"] == 2
    assert result.terminal_failure.details["reported_call_count"] == 1
