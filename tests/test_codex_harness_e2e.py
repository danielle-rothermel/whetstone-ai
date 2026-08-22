"""End-to-end Codex steps driven by the scripted fake CLI.

The fake CLI is a real subprocess that speaks real MCP to the real
evaluation server, so these tests exercise the production admission,
lease, evaluation, and ledger path. Only the agent's decisions are
scripted.

Together they are the "no eval outside the tools" proof: the accepted
candidate is rebuilt from a recorded Tool Call's args, an artifact naming
an unissued call is a terminal failure, and an artifact carrying the
wrong lease token is refused.
"""

from __future__ import annotations

import os
import sys
from datetime import timedelta
from pathlib import Path

import pytest
from dr_store.sync import open_sqlite

from tests.codex_support import (
    toy_capacity_binding,
    toy_codex_control,
    toy_codex_run,
    toy_codex_step_request,
    toy_tool_args,
    transcript_json,
)
from whetstone.core.identity import TypedRef
from whetstone.core.leasing import EffectLeaseAuthority, ReplayPolicy
from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
from whetstone.optim.adapters import MappingAdapterRegistry
from whetstone.optim.codex.adapter import (
    CODEX_ADAPTER_KEY,
    CODEX_LEASE_TOKEN_MISMATCH_CODE,
    CODEX_SELECTION_UNEVALUATED_CODE,
    CODEX_UNREPORTED_EVALUATION_CODE,
    CODEX_WALL_BUDGET_EXCEEDED_CODE,
    CodexAdapter,
    codex_lease_token_hash,
)
from whetstone.optim.codex.executor import build_codex_executor
from whetstone.optim.cost_aggregation import aggregate_run_cost
from whetstone.optim.codex.runner import SubprocessCodexRunner
from whetstone.optim.contracts import (
    OptimRun,
    OutputContract,
    StepMode,
    StepStatus,
    optimization_run_reference,
)
from whetstone.optim.harness import OptimHarness
from whetstone.optim.tools.evaluator import EngineToolEvaluator
from whetstone.optim.tools.execution import EvaluatingToolExecutor
from whetstone.optim.tools.facade import (
    ToolAdmissionAuthority,
    ToolCallStore,
)
from whetstone.testing.fake_codex_cli import (
    FAKE_CODEX_TRANSCRIPT_ENV,
    install_fake_codex_binary,
)
from whetstone.testing.toy.experiment import (
    TOY_MUTATION_FIELD,
    toy_template_render_contract,
)

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin",
    reason="the Codex sandbox is macOS sandbox-exec only",
)

_TEMPLATE_A = "Answer {prompt} in one short sentence."
_TEMPLATE_B = "Answer {prompt} with a single friendly word."
_FIXED_LEASE_TOKEN = "f" * 64


class _CodexWorld:
    """Everything one Codex end-to-end step needs, built together."""

    def __init__(
        self,
        *,
        tmp_path: Path,
        store,
        sqlite_path: str,
        max_tool_calls: int,
    ) -> None:
        self.store = store
        self.tmp_path = tmp_path
        # The harness and the out-of-process MCP server read and write one
        # database: the server persists the Tool Results the harness's
        # ledger later resolves.
        self.sqlite_path = sqlite_path
        self.engine = ReferenceEvalRuntimeConfig().build_engine(store)
        self.control = toy_codex_control(
            engine=self.engine, max_tool_calls=max_tool_calls
        )
        self.run, self.config, self.candidate = toy_codex_run(
            control=self.control, engine=self.engine
        )
        self.binary_dir = tmp_path / "bin"
        install_fake_codex_binary(self.binary_dir)
        # Both the admission authority and the effect lease authority
        # must be durable: the MCP evaluation server runs in another
        # process and records its terminals against the same database.
        self.effect_authority = EffectLeaseAuthority.sqlite(self.sqlite_path)
        self.tool_store = ToolCallStore(
            store,
            ToolAdmissionAuthority.sqlite(self.sqlite_path),
            self.effect_authority,
        )
        self.tool_executor = EvaluatingToolExecutor(
            EngineToolEvaluator(self.engine),
            self.engine.reward_policy,
            self.effect_authority,
            owner_id="codex-e2e-owner",
            replay_policy=ReplayPolicy.IDEMPOTENT,
            lease_duration=timedelta(minutes=5),
        )

    def tool_step(self, template: str, call_id: str) -> dict:
        return {
            "tool": self.config.tool_name,
            "args": {
                "call_id": call_id,
                **toy_tool_args(
                    candidate=self.candidate,
                    engine=self.engine,
                    template=template,
                ),
            },
        }

    def harness(self, adapter) -> OptimHarness:
        harness = OptimHarness(
            store=self.store,
            adapter_registry=MappingAdapterRegistry(
                {CODEX_ADAPTER_KEY: adapter}
            ),
            tool_store=self.tool_store,
            effect_authority=self.effect_authority,
            owner_id="codex-e2e-owner",
            adapter_replay_policy=ReplayPolicy.NO_REDRIVE,
            lease_duration=timedelta(minutes=5),
            tool_executor=self.tool_executor,
        )
        harness.bind_run(self.run)
        return harness

    def adapter(
        self,
        transcript: list[dict],
        *,
        timeout_seconds: float = 180.0,
    ) -> CodexAdapter:
        transcript_document = transcript_json(transcript)
        runner = SubprocessCodexRunner(
            executor=build_codex_executor(run_root=self.tmp_path / "runs"),
            sqlite_path=self.sqlite_path,
            runtime_config=ReferenceEvalRuntimeConfig(
                mutation_field=TOY_MUTATION_FIELD,
                render_contract=toy_template_render_contract(),
            ),
            runtime_config_class=(
                "whetstone.eval.reference_runtime:ReferenceEvalRuntimeConfig"
            ),
            reward_policy=self.engine.reward_policy,
            codex_binary="codex",
            timeout_seconds=timeout_seconds,
            environment={
                "PATH": os.pathsep.join(
                    [str(self.binary_dir), os.environ.get("PATH", "")]
                ),
                FAKE_CODEX_TRANSCRIPT_ENV: transcript_document,
                CODEX_AUTH_PLACEHOLDER: "sk-fake",
                # The real Codex binary needs no whetstone import; the
                # scripted stand-in is a Python module, so this test
                # grants it a path explicitly rather than production
                # staging the package into every run's sandbox.
                "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
            },
            # The runner's environment allowlist is production behavior;
            # the fake CLI's needs are granted explicitly rather than by
            # widening it.
            extra_environment_keys=frozenset(
                {FAKE_CODEX_TRANSCRIPT_ENV, "PYTHONPATH"}
            ),
        )
        adapter = CodexAdapter(
            runner,
            store=self.store,
            lease_token_factory=lambda: _FIXED_LEASE_TOKEN,
        )
        adapter.bind_tool_store(self.tool_store)
        return adapter


CODEX_AUTH_PLACEHOLDER = "OPENAI_API_KEY"


@pytest.fixture
def codex_world(tmp_path):
    sqlite_path = str((tmp_path / "codex-e2e.sqlite").resolve())
    with open_sqlite(sqlite_path) as store:
        yield lambda max_tool_calls=3: _CodexWorld(
            tmp_path=tmp_path,
            store=store,
            sqlite_path=sqlite_path,
            max_tool_calls=max_tool_calls,
        )


def test_a_terminal_step_returns_the_selected_evaluated_candidate(
    codex_world,
) -> None:
    world = codex_world()
    adapter = world.adapter(
        [
            world.tool_step(_TEMPLATE_A, "c1"),
            world.tool_step(_TEMPLATE_B, "c2"),
            {"final": {"selected_call_id": "c2"}},
        ]
    )
    request = toy_codex_step_request(
        control=world.control, run=world.run, candidate=world.candidate
    )

    result, _ref = world.harness(adapter).run_step(request)

    assert result.terminal_failure is None, result.terminal_failure
    assert result.status is StepStatus.COMPLETE
    assert result.seed_retained is False
    assert len(result.accepted_candidates) == 1
    # The accepted candidate's template is the one recorded on the
    # selected Tool Call, not anything the artifact asserted.
    accepted = world.store.get(result.accepted_candidates[0].record_ref.reference)
    assert accepted["payload"][TOY_MUTATION_FIELD] == _TEMPLATE_B
    assert result.budget_delta.consumed["tool_calls"] == 2
    # A TOOL_USING Step Result carries Tool Evidence and never intent or
    # search evidence; the ledger produces one entry per admitted call,
    # so both paid evaluations stay reachable from the Step Result.
    assert result.resolved_intents == ()
    assert result.search_evidence == ()
    assert len(result.tool_evidence) == 2


def test_the_ledger_records_every_evaluation_codex_drove(
    codex_world,
) -> None:
    world = codex_world()
    adapter = world.adapter(
        [
            world.tool_step(_TEMPLATE_A, "c1"),
            world.tool_step(_TEMPLATE_B, "c2"),
            {"final": {"selected_call_id": "c1"}},
        ]
    )
    request = toy_codex_step_request(
        control=world.control, run=world.run, candidate=world.candidate
    )

    result, _ref = world.harness(adapter).run_step(request)

    recorded = {
        str(entry.store_entry.call_id) for entry in result.tool_evidence
    }
    assert recorded == {"c1", "c2"}
    for entry in result.tool_evidence:
        # Every issued call has a durable terminal; ledger.evidence()
        # raises otherwise, so reaching here is the assertion.
        assert entry.result is not None
    # The accepted candidate's call is one of the recorded calls.
    assert "c1" in recorded


def test_an_unissued_selection_is_a_terminal_failure(codex_world) -> None:
    world = codex_world()
    adapter = world.adapter(
        [
            world.tool_step(_TEMPLATE_A, "c1"),
            {
                "final": {
                    "selected_call_id": "never-issued",
                    "evaluated_call_ids": ["c1", "never-issued"],
                }
            },
        ]
    )
    request = toy_codex_step_request(
        control=world.control, run=world.run, candidate=world.candidate
    )

    result, _ref = world.harness(adapter).run_step(request)

    assert result.status is StepStatus.FAILED
    assert result.terminal_failure is not None
    assert (
        result.terminal_failure.code == CODEX_SELECTION_UNEVALUATED_CODE
    )
    assert result.accepted_candidates == ()


def test_a_wrong_lease_token_is_refused(codex_world) -> None:
    world = codex_world()
    adapter = world.adapter(
        [
            world.tool_step(_TEMPLATE_A, "c1"),
            {
                "final": {
                    "selected_call_id": "c1",
                    "lease_token_hash": codex_lease_token_hash("wrong-token"),
                }
            },
        ]
    )
    request = toy_codex_step_request(
        control=world.control, run=world.run, candidate=world.candidate
    )

    result, _ref = world.harness(adapter).run_step(request)

    assert result.status is StepStatus.FAILED
    assert result.terminal_failure is not None
    assert (
        result.terminal_failure.code == CODEX_LEASE_TOKEN_MISMATCH_CODE
    )


def test_a_null_selection_retains_the_seed(codex_world) -> None:
    world = codex_world()
    adapter = world.adapter(
        [
            world.tool_step(_TEMPLATE_A, "c1"),
            {"final": {"selected_call_id": None}},
        ]
    )
    request = toy_codex_step_request(
        control=world.control, run=world.run, candidate=world.candidate
    )

    result, _ref = world.harness(adapter).run_step(request)

    assert result.status is StepStatus.COMPLETE
    assert result.seed_retained is True
    assert result.accepted_candidates == ()
    assert result.retained_candidate_ref is not None
    assert (
        result.retained_candidate_ref
        == world.run.initial_candidate_ref
    )
    # The evaluation it did make is still reachable through the ledger.
    assert len(result.tool_evidence) == 1


def test_a_contract_without_terminal_cardinality_rejects_seed_retention(
    codex_world,
) -> None:
    """The harness guard, not the adapter, is what enforces this rule.

    Only a contract with a search-dependent terminal cardinality may admit
    a ``seed_retained`` step; a contract that binds its cardinality
    unconditionally must reject the same adapter output.
    """
    from whetstone.optim.adapters import AdapterOutput
    from whetstone.optim.contracts import OptimStepRequest

    world = codex_world()
    request = toy_codex_step_request(
        control=world.control, run=world.run, candidate=world.candidate
    )
    unconditional = OptimStepRequest.model_validate(
        {
            **request.model_dump(mode="json"),
            "step_output_contract": OutputContract(
                returned_proposal_count=1
            ).model_dump(mode="json"),
        }
    )
    output = AdapterOutput(
        proposed_status=StepStatus.COMPLETE,
        seed_retained=True,
        retained_candidate=world.candidate,
    )

    with pytest.raises(ValueError, match="terminal_proposal_count"):
        OptimHarness._validate_output(unconditional, output)


def test_the_run_is_tool_using_and_carries_exactly_one_tool_config(
    codex_world,
) -> None:
    world = codex_world()

    assert world.run.mode is StepMode.TOOL_USING
    assert len(world.run.tool_configs) == 1
    assert world.run.tool_configs[0].record == world.config
    # A TOOL_USING run carries no Reward Policy; the Tool Config pins the
    # policy hash instead.
    assert world.run.reward_policy is None
    assert isinstance(
        optimization_run_reference(world.run).record, OptimRun
    )
    assert (
        toy_capacity_binding(world.run).subject_ref
        == optimization_run_reference(world.run).record_ref
    )


def _accepted_call_count(world, result) -> int:
    """Read the durable accepted count the adapter recorded on the Step.

    The adapter's ``state_delta`` is persisted as the Step Result's
    ``state_ref`` snapshot, so the assertion reads it back from the store
    rather than from the in-memory adapter output.
    """
    snapshot = world.store.get(result.state_ref.reference)
    return int(snapshot["harness_store_accepted_call_count"])


def test_an_unreported_admitted_evaluation_fails_the_step(
    codex_world,
) -> None:
    """Under-reporting must not hide paid work from the ledger.

    The agent makes two real admitted evaluations and names only one. The
    ledger built from the artifact would see one entry and debit one
    ``tool_calls``, leaving a paid evaluation unreachable from the Step
    Result and the run's remaining budget overstated. The adapter
    reconciles against the durable accepted count and refuses to
    complete.
    """
    world = codex_world()
    adapter = world.adapter(
        [
            world.tool_step(_TEMPLATE_A, "c1"),
            world.tool_step(_TEMPLATE_B, "c2"),
            {
                "final": {
                    "selected_call_id": "c1",
                    "evaluated_call_ids": ["c1"],
                }
            },
        ]
    )
    request = toy_codex_step_request(
        control=world.control, run=world.run, candidate=world.candidate
    )

    result, _ref = world.harness(adapter).run_step(request)

    assert result.status is StepStatus.FAILED
    assert result.terminal_failure is not None
    assert (
        result.terminal_failure.code == CODEX_UNREPORTED_EVALUATION_CODE
    )
    assert result.terminal_failure.details["admitted_call_count"] == 2
    assert result.terminal_failure.details["reported_call_count"] == 1
    assert result.accepted_candidates == ()
    # The durable ground truth is on the Step Result either way.
    assert _accepted_call_count(world, result) == 2


def test_reporting_every_admitted_call_debits_the_full_budget(
    codex_world,
) -> None:
    """The reconciliation's success side: totality holds, budget matches."""
    world = codex_world()
    adapter = world.adapter(
        [
            world.tool_step(_TEMPLATE_A, "c1"),
            world.tool_step(_TEMPLATE_B, "c2"),
            {"final": {"selected_call_id": "c1"}},
        ]
    )
    request = toy_codex_step_request(
        control=world.control, run=world.run, candidate=world.candidate
    )

    result, _ref = world.harness(adapter).run_step(request)

    assert result.terminal_failure is None, result.terminal_failure
    accepted = _accepted_call_count(world, result)
    assert accepted == 2
    assert len(result.tool_evidence) == accepted
    assert result.budget_delta.consumed["tool_calls"] == accepted


def test_a_wall_budget_stop_terminalizes_the_real_spawned_step(
    codex_world,
) -> None:
    """The real process path, stopped by the real dr-exec wall budget.

    The transcript tells the fake CLI to hang; the runner's wall budget
    stops the process group, and the Step must terminalize rather than
    letting a subprocess exception unwind past the harness's lease
    maintenance. The retry is the state evidence that the lease was
    released -- it would raise ``EffectBusyError`` otherwise.
    """
    world = codex_world()
    adapter = world.adapter([{"hang": True}], timeout_seconds=2.0)
    request = toy_codex_step_request(
        control=world.control, run=world.run, candidate=world.candidate
    )

    result, _ref = world.harness(adapter).run_step(request)

    assert result.status is StepStatus.FAILED
    assert result.terminal_failure is not None
    assert (
        result.terminal_failure.code == CODEX_WALL_BUDGET_EXCEEDED_CODE
    )
    assert result.accepted_candidates == ()

    retry = world.adapter([{"hang": True}], timeout_seconds=2.0)
    retried, _retry_ref = world.harness(retry).run_step(request)
    assert retried.status is StepStatus.FAILED


def test_tool_mediated_evaluations_reach_task_model_spend(
    codex_world,
) -> None:
    """The Codex arm's paid evaluations must appear in run cost.

    The Codex arm has no proposer -- the agent proposes -- so its entire
    spend is task-model spend, and every bit of it is driven through the
    tools. Those evaluations are cited from ``tool_evidence`` rather than
    from ``resolved_intents``, so an aggregator reading only the intent
    path would report a Codex run as having cost nothing at all.

    The count is derived from the persisted rows rather than asserted as a
    literal, so it tracks the evidence the run actually wrote.
    """
    world = codex_world()
    adapter = world.adapter(
        [
            world.tool_step(_TEMPLATE_A, "c1"),
            world.tool_step(_TEMPLATE_B, "c2"),
            {"final": {"selected_call_id": "c2"}},
        ]
    )
    request = toy_codex_step_request(
        control=world.control, run=world.run, candidate=world.candidate
    )

    result, _ref = world.harness(adapter).run_step(request)

    assert result.terminal_failure is None, result.terminal_failure
    assert result.resolved_intents == ()
    assert len(result.tool_evidence) == 2

    expected_rows = _admitted_evaluation_row_count(world, result)
    assert expected_rows > 0

    report = aggregate_run_cost(store=world.store, step_results=(result,))

    assert report.task_model.calls == expected_rows


def _admitted_evaluation_row_count(world, result) -> int:
    """Output rows across every evaluation the admitted tool calls drove."""
    total = 0
    for evidence in result.tool_evidence:
        for ref in evidence.result.record.evaluation_evidence_refs:
            record = world.store.get(ref.reference)
            outputs_ref = record.get("outputs_ref")
            if not isinstance(outputs_ref, dict):
                continue
            outputs = world.store.get(
                TypedRef.model_validate(outputs_ref).reference
            )
            total += len(outputs["outputs"])
    return total
