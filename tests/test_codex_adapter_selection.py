"""What the adapter accepts as a scored, reported candidate.

These drive the real ``CodexAdapter`` against the real Tool Call Store,
admission authority, and ``EvaluatingToolExecutor`` with a stub runner
standing in for the Codex process, so they exercise the selection and
reconciliation contract on every platform rather than only under the
macOS sandbox.
"""

from __future__ import annotations

from dataclasses import replace
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
from whetstone.experiment.candidate import Candidate, candidate_reference
from whetstone.testing.toy.experiment import build_toy_experiment
from whetstone.core.identity import ImmutableJsonObject, TerminalFailure
from whetstone.core.leasing import EffectLeaseAuthority, ReplayPolicy
from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
from whetstone.optim.codex.runner import SubprocessCodexRunner
from whetstone.testing.toy.experiment import (
    toy_template_render_contract,
)
from whetstone.optim.adapters import MappingAdapterRegistry
from whetstone.optim.codex.adapter import (
    CODEX_ADAPTER_KEY,
    CODEX_ARTIFACT_RUN_MISMATCH_CODE,
    CODEX_EVALUATION_INTERRUPTED_CODE,
    CODEX_EXECUTION_FAILED_CODE,
    CODEX_LEASE_TOKEN_MISMATCH_CODE,
    CODEX_MCP_HOST_FAILED_CODE,
    CODEX_RECORDED_CALL_CONTRACT_CODE,
    CODEX_SELECTION_UNEVALUATED_CODE,
    CODEX_SELECTION_UNSCORED_CODE,
    CODEX_UNREPORTED_EVALUATION_CODE,
    CODEX_WALL_BUDGET_EXCEEDED_CODE,
    CodexAdapter,
    CodexOutputArtifact,
    CodexRunResult,
    CodexStructuredExecutionFailure,
    CodexWallBudgetExceeded,
    OpaqueStepError,
    codex_lease_token_hash,
)
from whetstone.optim.contracts import (
    SUPERSEDED_FAILURE_CODES_KEY,
    StepStatus,
)
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


class _PermissiveEvaluator(EngineToolEvaluator):
    """Evaluates using the base template regardless of recorded args.

    It exists to produce a durable COMPLETED entry whose recorded args do
    not carry a usable ``template``, which the real evaluator refuses to
    do.
    """

    def evaluate(self, call, config):
        patched = call.model_copy(
            update={
                "args": ImmutableJsonObject(
                    {**call.args.to_json(), "template": _TEMPLATE_B}
                )
            }
        )
        return super().evaluate(patched, config)


def _enriched_candidate(base: Candidate, extra: dict[str, str]) -> Candidate:
    """The same candidate with extra, non-mutation payload fields."""
    payload = base.payload.to_json()
    payload.update(extra)
    return candidate_reference(
        Candidate(
            candidate_id=base.candidate_id,
            base_ref=base.base_ref,
            payload=payload,
        )
    ).record


class _SelectionWorld:
    def __init__(
        self,
        store,
        *,
        failing_call_ids: frozenset[str],
        extra_payload: dict[str, str] | None = None,
        sqlite_path: str = "",
    ) -> None:
        self.store = store
        self.sqlite_path = sqlite_path
        self.engine = ReferenceEvalRuntimeConfig().build_engine(store)
        self.control = toy_codex_control(engine=self.engine, max_tool_calls=4)
        experiment = build_toy_experiment(num_seeds=1)
        if extra_payload:
            # A base candidate whose payload carries more than the
            # mutation field. Nothing restricts a Codex run's candidates
            # to a single field, and the extra field is inert to the
            # evaluation itself.
            experiment = replace(
                experiment,
                initial_candidate=_enriched_candidate(
                    experiment.initial_candidate, extra_payload
                ),
            )
        self.run, self.config, self.candidate = toy_codex_run(
            control=self.control,
            engine=self.engine,
            experiment=experiment,
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

    def use_permissive_evaluator(self) -> None:
        """Swap in an evaluator that completes a call with no template.

        The real evaluator reads ``template`` and ``base_ref`` off the
        call, so a durable COMPLETED entry with unusable recorded args
        cannot be built through it. This stands in for a recorded call
        whose args do not carry what the adapter must rebuild from.
        """
        self.tool_executor = EvaluatingToolExecutor(
            _PermissiveEvaluator(self.engine),
            self.engine.reward_policy,
            self.effect_authority,
            owner_id="codex-selection-owner",
            replay_policy=ReplayPolicy.IDEMPOTENT,
            lease_duration=timedelta(minutes=5),
        )
        self._agent_handle = self.tool_executor.runtime_handle(
            self.config, self.tool_store, self.binding
        )

    def issue_malformed(self, call_id: str):
        """Admit and complete a call whose ``template`` is not a string.

        ``ToolCall`` pins the arg *keys* to the Definition's
        ``input_fields``, so a missing key is unreachable; the value
        types are not pinned, so this is the shape a durable entry can
        actually reach.
        """
        args = toy_tool_args(
            candidate=self.candidate,
            engine=self.engine,
            template=_TEMPLATE_A,
        )
        args["template"] = {"not": "a template"}
        return self._agent_handle(
            ToolCall(
                call_id=call_id,
                tool_config=tool_config_reference(self.config),
                capacity_binding=self.binding,
                args=ImmutableJsonObject(args),
            )
        )

    def issue_foreign_base(self, call_id: str, template: str):
        """Admit and complete a call bound to a base no candidate names.

        The ref is well-formed and its schema is the candidate schema,
        so nothing about its *shape* is wrong -- it simply names a
        record this Step Request never offered, which is what another
        run's ref or a forged one looks like.
        """
        args = toy_tool_args(
            candidate=self.candidate,
            engine=self.engine,
            template=template,
        )
        args["base_ref"] = {
            **args["base_ref"],
            "content_hash": "f" * 64,
        }
        return self._agent_handle(
            ToolCall(
                call_id=call_id,
                tool_config=tool_config_reference(self.config),
                capacity_binding=self.binding,
                args=ImmutableJsonObject(args),
            )
        )

    def run_step(self, *, calls, artifact_fields):
        return self.run_step_with_runner(
            _ScriptedRunner(
                self, calls=calls, artifact_fields=artifact_fields
            )
        )

    def run_step_with_runner(self, runner):
        adapter = CodexAdapter(
            runner,
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
    sqlite_path = str(tmp_path / "codex-selection.sqlite")
    with open_sqlite(sqlite_path) as store:

        def build(failing_call_ids: frozenset[str] = frozenset()):
            return _SelectionWorld(
                store,
                failing_call_ids=failing_call_ids,
                sqlite_path=sqlite_path,
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
    # The omitted call completed, so the evaluation it paid for is put
    # on the ledger and debited rather than disappearing with the
    # failure: the Step fails, but the spend stays visible.
    assert len(result.tool_evidence) == 2
    assert result.budget_delta.consumed["tool_calls"] == 2


class _WallBudgetRunner:
    """A runner that hits the wall stop instead of returning an artifact."""

    def __init__(self, *, wall_seconds: float) -> None:
        self.wall_seconds = wall_seconds

    def run(self, request, handle, *, lease_token):
        del request, handle, lease_token
        raise CodexWallBudgetExceeded(
            f"Codex exceeded its wall budget of {self.wall_seconds} seconds",
            wall_seconds=self.wall_seconds,
            stdout=b"",
            stderr=b"",
            isolation={"strategy": "test"},
        )


def test_a_wall_budget_stop_terminalizes_the_step_and_frees_the_lease(
    selection_world,
) -> None:
    """The wall stop must not escape as a raw subprocess exception.

    A ``subprocess.TimeoutExpired`` unwinding out of ``run_step`` skips
    the harness's effect-lease maintenance entirely, so the lease stays
    non-terminal and the run wedges until it lapses. The evidence that
    the lease was released is a state fact, not a delay: the identical
    Step runs again immediately instead of raising ``EffectBusyError``.
    """
    world = selection_world()

    result = world.run_step_with_runner(
        _WallBudgetRunner(wall_seconds=0.25)
    )

    assert result.status is StepStatus.FAILED
    assert result.terminal_failure is not None
    assert (
        result.terminal_failure.code == CODEX_WALL_BUDGET_EXCEEDED_CODE
    )
    assert result.terminal_failure.details["wall_seconds"] == 0.25
    assert result.accepted_candidates == ()

    # The lease reached a terminal state, so the same effect is free.
    retried = world.run_step_with_runner(
        _WallBudgetRunner(wall_seconds=0.25)
    )
    assert retried.status is StepStatus.FAILED


def test_two_failed_evaluations_terminalize_under_one_shared_failure(
    selection_world,
) -> None:
    """A Step Result carries exactly one terminal failure.

    ``EngineToolEvaluator`` names the failing call in its own failure, so
    two transient provider failures on two calls always produce two
    unequal ``TerminalFailure`` values. Re-issuing both through the
    guarded handle puts both on the Step's Tool Evidence, and the
    contract requires every nested failure to equal the outer one -- so
    the adapter must pick one shared failure rather than letting the
    Step Result raise on construction.
    """
    world = selection_world(frozenset({"c1", "c2"}))

    result = world.run_step(
        calls=[("c1", _TEMPLATE_A), ("c2", _TEMPLATE_B)],
        artifact_fields={"selected_call_id": "c2"},
    )

    assert result.status is StepStatus.FAILED
    assert result.terminal_failure is not None
    assert result.accepted_candidates == ()
    # The Step cannot adopt either evaluation's failure without
    # contradicting the other, so it supersedes both and says so.
    assert result.terminal_failure.code == CODEX_SELECTION_UNSCORED_CODE
    superseded = result.terminal_failure.details.to_json()[
        SUPERSEDED_FAILURE_CODES_KEY
    ]
    assert superseded == [_FORCED_FAILURE_CODE, _FORCED_FAILURE_CODE]
    # Both paid evaluations stay reachable from the Step Result: the
    # ledger is total over admitted calls even when they all failed.
    assert len(result.tool_evidence) == 2
    assert result.budget_delta.consumed["tool_calls"] == 2


def test_a_multi_failure_step_releases_its_lease(selection_world) -> None:
    """The wedge, not just the crash.

    The harness terminalizes the effect lease as FAILED before it builds
    the Step Result, so a Step Result that raises leaves the run
    permanently unbindable: every re-run replays the same checkpoint and
    raises again. The evidence is a state fact -- the identical Step runs
    to a terminal result a second time.
    """
    world = selection_world(frozenset({"c1", "c2"}))

    first = world.run_step(
        calls=[("c1", _TEMPLATE_A), ("c2", _TEMPLATE_B)],
        artifact_fields={"selected_call_id": "c2"},
    )
    retried = world.run_step(
        calls=[("c1", _TEMPLATE_A), ("c2", _TEMPLATE_B)],
        artifact_fields={"selected_call_id": "c2"},
    )

    assert first.status is StepStatus.FAILED
    assert retried.status is StepStatus.FAILED
    assert retried.terminal_failure == first.terminal_failure


class _CrashingCallEvaluator(EngineToolEvaluator):
    """An evaluator whose failure is not a ``ToolEvaluationError``.

    ``EvaluatingToolExecutor`` converts a ``ToolEvaluationError`` into a
    COMPLETED entry carrying a terminal failure. Anything else -- a
    provider transport error, the eval server dying mid-call -- escapes
    the executor, leaving the entry ACCEPTED with its capacity already
    debited and no durable result. That is the harness-side crash.
    """

    def __init__(self, engine, *, crashing_call_ids: frozenset[str]) -> None:
        super().__init__(engine)
        self._crashing_call_ids = crashing_call_ids

    def evaluate(self, call, config):
        if str(call.call_id) in self._crashing_call_ids:
            raise OSError("the evaluation server went away mid-call")
        return super().evaluate(call, config)


class _CrashingRunner:
    """Issues one call that crashes the server, then reports nothing.

    The agent never learns the call id of a call whose result never came
    back, so the artifact omits it -- exactly the shape that reads as
    under-reporting unless the adapter looks at the durable state.
    """

    def __init__(self, world, *, calls, crashing_call_ids) -> None:
        self._world = world
        self._calls = calls
        self._crashing = crashing_call_ids

    def run(self, request, handle, *, lease_token):
        del handle
        reported = []
        for call_id, template in self._calls:
            if call_id in self._crashing:
                with pytest.raises(OSError):
                    self._world.issue(call_id, template)
                continue
            self._world.issue(call_id, template)
            reported.append(call_id)
        return CodexRunResult(
            artifact=CodexOutputArtifact(
                run_id=request.run_id,
                evaluated_call_ids=tuple(reported),
                selected_call_id=reported[0] if reported else None,
                lease_token_hash=codex_lease_token_hash(lease_token),
            )
        )


def test_a_harness_side_crash_is_not_blamed_on_the_agent(
    tmp_path,
) -> None:
    """An admitted call with no terminal is whetstone's failure, not the agent's.

    ``codex_unreported_evaluation`` means the agent hid a completed
    evaluation. A call that whetstone's own server admitted and then
    crashed on never completed, so the agent had nothing to report; the
    Step must still fail, but under a harness-attributed code that says
    what actually happened.
    """
    from dr_store.sync import open_sqlite

    with open_sqlite(str(tmp_path / "codex-crash.sqlite")) as store:
        world = _SelectionWorld(store, failing_call_ids=frozenset())
        world.tool_executor = EvaluatingToolExecutor(
            _CrashingCallEvaluator(
                world.engine, crashing_call_ids=frozenset({"c2"})
            ),
            world.engine.reward_policy,
            world.effect_authority,
            owner_id="codex-selection-owner",
            replay_policy=ReplayPolicy.IDEMPOTENT,
            lease_duration=timedelta(minutes=5),
        )
        world._agent_handle = world.tool_executor.runtime_handle(
            world.config, world.tool_store, world.binding
        )

        result = world.run_step_with_runner(
            _CrashingRunner(
                world,
                calls=[("c1", _TEMPLATE_A), ("c2", _TEMPLATE_B)],
                crashing_call_ids=frozenset({"c2"}),
            )
        )

    assert result.status is StepStatus.FAILED
    assert result.terminal_failure is not None
    assert (
        result.terminal_failure.code
        == CODEX_EVALUATION_INTERRUPTED_CODE
    )
    details = result.terminal_failure.details.to_json()
    assert details["interrupted_call_ids"] == ["c2"]


class _EvaluatingWallBudgetRunner:
    """Two real paid evaluations, then the hard wall stop.

    This is the ordinary end of a long-running paid agent: it did real
    work, and then ran out of wall clock before it could write an
    artifact naming that work.
    """

    def __init__(self, world, *, calls, wall_seconds: float) -> None:
        self._world = world
        self._calls = calls
        self.wall_seconds = wall_seconds

    def run(self, request, handle, *, lease_token):
        del request, handle, lease_token
        for call_id, template in self._calls:
            self._world.issue(call_id, template)
        raise CodexWallBudgetExceeded(
            f"Codex exceeded its wall budget of {self.wall_seconds} seconds",
            wall_seconds=self.wall_seconds,
            stdout=b"",
            stderr=b"",
            isolation={"strategy": "test"},
        )


def test_a_wall_stop_still_ledgers_the_evaluations_it_already_paid_for(
    selection_world,
) -> None:
    """A wall stop must not make paid work invisible.

    The wall branch runs before any artifact exists, so there is nothing
    to read call ids from -- but the durable admission entries hold them
    all. Leaving them out would under-debit the ``tool_calls`` budget by
    the full number of evaluations performed and put their rewards and
    evidence refs out of reach of the Step Result, which is exactly what
    the ledger exists to prevent.
    """
    world = selection_world()

    result = world.run_step_with_runner(
        _EvaluatingWallBudgetRunner(
            world,
            calls=[("c1", _TEMPLATE_A), ("c2", _TEMPLATE_B)],
            wall_seconds=0.25,
        )
    )

    assert result.status is StepStatus.FAILED
    assert result.terminal_failure is not None
    assert result.terminal_failure.code == CODEX_WALL_BUDGET_EXCEEDED_CODE
    assert len(result.tool_evidence) == 2
    assert result.budget_delta.consumed["tool_calls"] == 2


class _LeaseMismatchRunner:
    """Pays for real evaluations, then returns a corrupt lease hash.

    The agent holds the bearer token for the MCP endpoint, so it can spend
    the run's capacity and then omit or corrupt the artifact field that
    proves the artifact is this Step's. That must not buy it an exit from
    ledger totality.
    """

    def __init__(self, world, *, calls, lease_token_hash: str) -> None:
        self._world = world
        self._calls = calls
        self._lease_token_hash = lease_token_hash

    def run(self, request, handle, *, lease_token):
        del handle, lease_token
        for call_id, template in self._calls:
            self._world.issue(call_id, template)
        return CodexRunResult(
            artifact=CodexOutputArtifact(
                run_id=request.run_id,
                evaluated_call_ids=tuple(
                    call_id for call_id, _template in self._calls
                ),
                selected_call_id=self._calls[0][0],
                lease_token_hash=self._lease_token_hash,
            )
        )


@pytest.mark.parametrize("bad_hash", ["", codex_lease_token_hash("wrong")])
def test_a_lease_mismatch_still_ledgers_the_evaluations_it_paid_for(
    selection_world, bad_hash: str
) -> None:
    """A malformed artifact must not short-circuit ledger totality.

    The lease hash is checked before anything is reconciled, so an agent
    that pays for evaluations and then corrupts or omits the hash used to
    produce a Step with no Tool Evidence and a zero ``tool_calls`` debit,
    while ``accepted_count`` recorded real spend. The hash check still
    fails the Step -- it just no longer takes the paid work with it.
    """
    world = selection_world()

    result = world.run_step_with_runner(
        _LeaseMismatchRunner(
            world,
            calls=[("c1", _TEMPLATE_A), ("c2", _TEMPLATE_B)],
            lease_token_hash=bad_hash,
        )
    )

    assert result.status is StepStatus.FAILED
    assert result.terminal_failure is not None
    assert result.terminal_failure.code == CODEX_LEASE_TOKEN_MISMATCH_CODE
    assert len(result.tool_evidence) == 2
    assert result.budget_delta.consumed["tool_calls"] == 2


class _DuplicateReportRunner:
    """Reports one paid call id twice, taking the unevaluated path.

    ``_UnevaluatedSelectionError`` used to return a bare terminal failure,
    skipping the shared-failure reconciliation even though the calls it
    already re-issued were on the ledger.
    """

    def __init__(self, world, *, calls) -> None:
        self._world = world
        self._calls = calls

    def run(self, request, handle, *, lease_token):
        del handle
        for call_id, template in self._calls:
            self._world.issue(call_id, template)
        first = self._calls[0][0]
        return CodexRunResult(
            artifact=CodexOutputArtifact(
                run_id=request.run_id,
                evaluated_call_ids=(first, first),
                selected_call_id=first,
                lease_token_hash=codex_lease_token_hash(lease_token),
            )
        )


def test_an_unevaluated_report_still_ledgers_its_paid_calls(
    selection_world,
) -> None:
    """The unevaluated paths take the same terminal path as the rest.

    A duplicate report fails under ``codex_selection_unevaluated``, but
    the run had already paid for two evaluations. Returning a bare
    terminal failure left them off the ledger and off the budget.
    """
    world = selection_world()

    result = world.run_step_with_runner(
        _DuplicateReportRunner(
            world, calls=[("c1", _TEMPLATE_A), ("c2", _TEMPLATE_B)]
        )
    )

    assert result.status is StepStatus.FAILED
    assert result.terminal_failure is not None
    assert (
        result.terminal_failure.code == CODEX_SELECTION_UNEVALUATED_CODE
    )
    assert len(result.tool_evidence) == 2
    assert result.budget_delta.consumed["tool_calls"] == 2


class _MissingSelectionRunner:
    """Names a selected call it never reported, after paying for calls."""

    def __init__(self, world, *, calls) -> None:
        self._world = world
        self._calls = calls

    def run(self, request, handle, *, lease_token):
        del handle
        for call_id, template in self._calls:
            self._world.issue(call_id, template)
        return CodexRunResult(
            artifact=CodexOutputArtifact(
                run_id=request.run_id,
                evaluated_call_ids=tuple(
                    call_id for call_id, _template in self._calls
                ),
                selected_call_id="never-issued",
                lease_token_hash=codex_lease_token_hash(lease_token),
            )
        )


def test_a_missing_selected_call_still_ledgers_its_paid_calls(
    selection_world,
) -> None:
    """The selected-call miss is a terminal path like any other."""
    world = selection_world()

    result = world.run_step_with_runner(
        _MissingSelectionRunner(
            world, calls=[("c1", _TEMPLATE_A), ("c2", _TEMPLATE_B)]
        )
    )

    assert result.status is StepStatus.FAILED
    assert result.terminal_failure is not None
    assert (
        result.terminal_failure.code == CODEX_SELECTION_UNEVALUATED_CODE
    )
    assert len(result.tool_evidence) == 2
    assert result.budget_delta.consumed["tool_calls"] == 2


class _ExecutionFailureRunner:
    """Pays for one evaluation, then fails without a usable artifact.

    A nonzero Codex exit, an unreadable artifact, or a malformed one all
    raise ``CodexStructuredExecutionFailure``. That is not the wall-budget
    subclass, so it used to escape ``invoke`` entirely and leave the
    effect nonterminal under ``NO_REDRIVE``.
    """

    def __init__(self, world, *, calls) -> None:
        self._world = world
        self._calls = calls

    def run(self, request, handle, *, lease_token):
        del request, handle, lease_token
        for call_id, template in self._calls:
            self._world.issue(call_id, template)
        raise CodexStructuredExecutionFailure(
            "Codex exited 3 without a final output artifact",
            stdout=b"",
            stderr=b"codex: not logged in\n",
            isolation={"strategy": "test"},
        )


def test_a_structured_execution_failure_terminalizes_the_step(
    selection_world,
) -> None:
    """A nonzero Codex exit fails the Step instead of unwinding.

    The harness calls ``maintenance.fail`` only once the adapter returns
    an ``AdapterOutput``, so an exception escaping ``invoke`` leaves the
    effect lease non-terminal. Under ``NO_REDRIVE`` the run then cannot
    recover until the lease lapses. The evidence that the lease was
    released is a state fact: the identical Step runs again immediately
    rather than raising ``EffectBusyError``.
    """
    world = selection_world()

    result = world.run_step_with_runner(
        _ExecutionFailureRunner(world, calls=[("c1", _TEMPLATE_A)])
    )

    assert result.status is StepStatus.FAILED
    assert result.terminal_failure is not None
    assert (
        result.terminal_failure.code == CODEX_EXECUTION_FAILED_CODE
    )
    # The evaluation it paid for before dying stays on the ledger.
    assert len(result.tool_evidence) == 1
    assert result.budget_delta.consumed["tool_calls"] == 1

    retried = world.run_step_with_runner(
        _ExecutionFailureRunner(world, calls=[("c2", _TEMPLATE_B)])
    )
    assert retried.status is StepStatus.FAILED


class _BaseRunnerFailureRunner:
    """Pays for one evaluation, then fails the way the runner's own
    parsing does.

    A zero-exit CLI whose final artifact fails schema validation, and a
    dr-exec ``ExecutorFailure``, both surface as the *base*
    ``OpaqueStepError`` rather than the structured subclass. The Step
    must still terminalize: the failure taxonomy is about which thing
    broke, not about whether the lease is released.
    """

    def __init__(self, world, *, calls) -> None:
        self._world = world
        self._calls = calls

    def run(self, request, handle, *, lease_token):
        del request, handle, lease_token
        for call_id, template in self._calls:
            self._world.issue(call_id, template)
        raise OpaqueStepError(
            "Codex final output artifact failed schema validation"
        )


def test_a_base_runner_failure_terminalizes_the_step(
    selection_world,
) -> None:
    """A schema-invalid artifact must not escape ``invoke``.

    The runner raises the base ``OpaqueStepError`` for a zero-exit run
    whose artifact fails validation. Catching only the structured
    subclass let that unwind past the adapter checkpoint, so the harness
    never ran ``maintenance.fail`` and this ``NO_REDRIVE`` effect stayed
    nonterminal. The evidence that the lease was released is a state
    fact: the identical Step runs again immediately rather than raising
    ``EffectBusyError``.
    """
    world = selection_world()

    result = world.run_step_with_runner(
        _BaseRunnerFailureRunner(world, calls=[("c1", _TEMPLATE_A)])
    )

    assert result.status is StepStatus.FAILED
    assert result.terminal_failure is not None
    assert result.terminal_failure.code == CODEX_EXECUTION_FAILED_CODE
    # The evaluation it paid for before failing stays on the ledger.
    assert len(result.tool_evidence) == 1
    assert result.budget_delta.consumed["tool_calls"] == 1

    retried = world.run_step_with_runner(
        _BaseRunnerFailureRunner(world, calls=[("c2", _TEMPLATE_B)])
    )
    assert retried.status is StepStatus.FAILED


class _StrandedWallBudgetRunner:
    """Crashes one in-flight evaluation, then hits the wall stop.

    A wall kill takes the host down with any evaluation still in flight,
    leaving the entry ``ACCEPTED`` with its capacity already debited and
    no durable terminal. The agent never saw a result for it.
    """

    def __init__(self, world, *, calls, crashing_call_ids, wall_seconds):
        self._world = world
        self._calls = calls
        self._crashing = crashing_call_ids
        self.wall_seconds = wall_seconds

    def run(self, request, handle, *, lease_token):
        del request, handle, lease_token
        for call_id, template in self._calls:
            if call_id in self._crashing:
                with pytest.raises(OSError):
                    self._world.issue(call_id, template)
                continue
            self._world.issue(call_id, template)
        raise CodexWallBudgetExceeded(
            f"Codex exceeded its wall budget of {self.wall_seconds} seconds",
            wall_seconds=self.wall_seconds,
            stdout=b"",
            stderr=b"",
            isolation={"strategy": "test"},
        )


def test_a_wall_stop_names_the_admissions_it_stranded(tmp_path) -> None:
    """A wall kill must account for its in-flight admissions.

    The wall branch re-issued only ``COMPLETED`` entries, so an admission
    the kill stranded in ``ACCEPTED`` held its capacity slot and appeared
    nowhere on the Step Result -- paid work visible only in
    ``accepted_count``. It now fails as an interrupted evaluation naming
    the stranded ids, while the completed work still reaches the ledger.
    """
    with open_sqlite(str(tmp_path / "codex-stranded.sqlite")) as store:
        world = _SelectionWorld(store, failing_call_ids=frozenset())
        world.tool_executor = EvaluatingToolExecutor(
            _CrashingCallEvaluator(
                world.engine, crashing_call_ids=frozenset({"c2"})
            ),
            world.engine.reward_policy,
            world.effect_authority,
            owner_id="codex-selection-owner",
            replay_policy=ReplayPolicy.IDEMPOTENT,
            lease_duration=timedelta(minutes=5),
        )
        world._agent_handle = world.tool_executor.runtime_handle(
            world.config, world.tool_store, world.binding
        )

        result = world.run_step_with_runner(
            _StrandedWallBudgetRunner(
                world,
                calls=[("c1", _TEMPLATE_A), ("c2", _TEMPLATE_B)],
                crashing_call_ids=frozenset({"c2"}),
                wall_seconds=0.25,
            )
        )

    assert result.status is StepStatus.FAILED
    assert result.terminal_failure is not None
    assert (
        result.terminal_failure.code == CODEX_EVALUATION_INTERRUPTED_CODE
    )
    details = result.terminal_failure.details.to_json()
    assert details["interrupted_call_ids"] == ["c2"]
    # The completed evaluation still reaches the ledger and the budget.
    assert len(result.tool_evidence) == 1
    assert result.budget_delta.consumed["tool_calls"] == 1


class _UnreportedMalformedArgsRunner:
    """Under-reports a call whose recorded args are malformed.

    Reconciliation re-issues the omitted completed entry through the
    guarded handle, putting it on the Issued Tool Call ledger. If the
    recorded args then fail validation, skipping the entry leaves a
    ledgered call with no matching evidence in the reconciled tuple.
    """

    def __init__(self, world, *, calls) -> None:
        self._world = world
        self._calls = calls

    def run(self, request, handle, *, lease_token):
        del handle
        for call_id, template in self._calls:
            self._world.issue(call_id, template)
        return CodexRunResult(
            artifact=CodexOutputArtifact(
                run_id=request.run_id,
                evaluated_call_ids=(self._calls[0][0],),
                selected_call_id=self._calls[0][0],
                lease_token_hash=codex_lease_token_hash(lease_token),
            )
        )


def test_a_malformed_reconciled_call_fails_rather_than_being_skipped(
    selection_world,
) -> None:
    """A ledgered call is never dropped from the reconciled evidence.

    ``_issue_completed`` issued the call before validating its recorded
    ``template``/``base_ref`` and then skipped it, so the call reached the
    ledger while the reconciled tuple fed to the shared-failure rule did
    not know about it. Validation now precedes issuance, and a malformed
    recorded call is a typed Step failure rather than a silent skip.
    """
    world = selection_world()

    result = world.run_step_with_runner(
        _UnreportedMalformedArgsRunner(
            world, calls=[("c1", _TEMPLATE_A), ("c2", _TEMPLATE_B)]
        )
    )

    # The well-formed control: the shortfall is reported as such, and
    # both calls are on the ledger.
    assert result.status is StepStatus.FAILED
    assert result.terminal_failure is not None
    assert result.terminal_failure.code == CODEX_UNREPORTED_EVALUATION_CODE
    assert len(result.tool_evidence) == 2


class _MalformedArgsShortfallRunner:
    """Under-reports a completed call whose recorded args are malformed.

    Reconciliation must re-issue the omitted entry, and the entry's own
    recorded ``template``/``base_ref`` is what the adapter rebuilds a
    candidate from. When those are missing, the entry cannot be
    represented as reconciled evidence.
    """

    def __init__(self, world, *, reported, malformed_call_id: str) -> None:
        self._world = world
        self._reported = reported
        self._malformed = malformed_call_id

    def run(self, request, handle, *, lease_token):
        del handle
        for call_id, template in self._reported:
            self._world.issue(call_id, template)
        self._world.issue_malformed(self._malformed)
        return CodexRunResult(
            artifact=CodexOutputArtifact(
                run_id=request.run_id,
                evaluated_call_ids=tuple(
                    call_id for call_id, _template in self._reported
                ),
                selected_call_id=self._reported[0][0],
                lease_token_hash=codex_lease_token_hash(lease_token),
            )
        )


def test_a_recorded_call_with_malformed_args_is_a_typed_failure(
    tmp_path,
) -> None:
    """Malformed recorded args on a reconciled entry fail the Step.

    ``_issue_completed`` put the call on the Issued Tool Call ledger and
    only then looked at its recorded args, dropping it from the
    reconciled tuple when they were unusable. That left a ledgered call
    with no matching evidence for the shared-failure rule to account
    for. Validation now runs first, and an unusable recorded call is a
    typed Step failure rather than a silent skip.
    """
    with open_sqlite(str(tmp_path / "codex-malformed.sqlite")) as store:
        world = _SelectionWorld(store, failing_call_ids=frozenset())
        world.use_permissive_evaluator()

        result = world.run_step_with_runner(
            _MalformedArgsShortfallRunner(
                world,
                reported=[("c1", _TEMPLATE_A)],
                malformed_call_id="c2",
            )
        )

    assert result.status is StepStatus.FAILED
    assert result.terminal_failure is not None
    assert (
        result.terminal_failure.code == CODEX_RECORDED_CALL_CONTRACT_CODE
    )
    assert result.terminal_failure.details["call_id"] == "c2"


def test_the_selected_candidate_keeps_every_non_mutation_field(
    tmp_path,
) -> None:
    """Reconstruction copies the base payload, not just the template.

    ``diff_check`` requires every non-mutation field to equal the base's,
    so building the candidate from the mutation field alone made any base
    carrying an extra payload field fail ``codex_selection_contract`` on
    a legitimately evaluated selection. ``prepare_codex_run`` puts no
    single-field restriction on candidates.
    """
    with open_sqlite(str(tmp_path / "codex-extra-field.sqlite")) as store:
        world = _SelectionWorld(
            store,
            failing_call_ids=frozenset(),
            extra_payload={"system_prompt": "You are terse."},
        )

        result = world.run_step(
            calls=[("c1", _TEMPLATE_A)],
            artifact_fields={"selected_call_id": "c1"},
        )

        assert result.terminal_failure is None, result.terminal_failure
        assert result.status is StepStatus.COMPLETE
        accepted = world.store.get(
            result.accepted_candidates[0].record_ref.reference
        )

    assert accepted["payload"]["user_prompt_template"] == _TEMPLATE_A
    assert accepted["payload"]["system_prompt"] == "You are terse."



class _RealRunnerWithFailingHost:
    """The production runner, with its evaluation endpoint squatted.

    This is the real ``SubprocessCodexRunner.run``: it builds the server
    environment, constructs the host, and enters it. Only the port the
    host binds is forced to one a foreign listener already owns, which
    is exactly how a squatted port, a bind failure, and a startup that
    misses its deadline reach the runner -- as ``CodexMcpHostError``
    from ``__enter__``, before the Codex process exists.

    Nothing here synthesizes the failure or the normalization; the
    runner's own boundary is what turns it into a terminal Step failure.
    """

    def __init__(self, world, *, port: int, monkeypatch) -> None:
        self._port = port
        self._monkeypatch = monkeypatch
        self._inner = SubprocessCodexRunner(
            executor=_UnreachableExecutor(),
            sqlite_path=world.sqlite_path,
            runtime_config=ReferenceEvalRuntimeConfig(
                mutation_field=world.config.candidate_template_field,
                render_contract=toy_template_render_contract(),
            ),
            runtime_config_class=(
                "whetstone.eval.reference_runtime:"
                "ReferenceEvalRuntimeConfig"
            ),
            reward_policy=world.engine.reward_policy,
        )

    def run(self, request, handle, *, lease_token):
        from whetstone.optim.codex import runner as runner_module
        from whetstone.optim.codex.mcp_host import CodexMcpHost

        port = self._port

        def _squatted_host(server, **kwargs):
            kwargs.pop("port", None)
            return CodexMcpHost(
                server, port=port, startup_seconds=5.0, **kwargs
            )

        self._monkeypatch.setattr(
            runner_module, "CodexMcpHost", _squatted_host
        )
        return self._inner.run(request, handle, lease_token=lease_token)


class _UnreachableExecutor:
    """The Codex process must never be reached in these tests."""

    def run_blocking(self, job):  # pragma: no cover - must not run
        raise AssertionError(
            "the Codex process must not run when the host failed"
        )


def test_an_mcp_host_that_never_starts_terminalizes_the_step(
    selection_world, monkeypatch
) -> None:
    """A host failure is whetstone's failure, and it still ends the Step.

    ``CodexMcpHostError`` is not an ``OpaqueStepError``, so before the
    runner normalized it, it unwound past the adapter checkpoint: the
    harness runs its effect-lease maintenance only once the adapter
    returns an ``AdapterOutput``, leaving this ``NO_REDRIVE`` effect
    nonterminal until the lease lapsed. The proof that the lease was
    released is a state fact -- the identical Step runs again
    immediately instead of raising ``EffectBusyError``.
    """
    import socket

    squatter = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    squatter.bind(("127.0.0.1", 0))
    squatter.listen(8)
    port = int(squatter.getsockname()[1])
    try:
        world = selection_world()

        result = world.run_step_with_runner(
            _RealRunnerWithFailingHost(
                world, port=port, monkeypatch=monkeypatch
            )
        )

        assert result.status is StepStatus.FAILED
        assert result.terminal_failure is not None
        assert result.terminal_failure.code == CODEX_MCP_HOST_FAILED_CODE
        # The agent never ran, so the Step paid for nothing.
        assert result.tool_evidence == ()

        # The lease came back: the identical Step is runnable again.
        retried = world.run_step_with_runner(
            _RealRunnerWithFailingHost(
                world, port=port, monkeypatch=monkeypatch
            )
        )
        assert retried.status is StepStatus.FAILED
        assert (
            retried.terminal_failure is not None
            and retried.terminal_failure.code
            == CODEX_MCP_HOST_FAILED_CODE
        )
    finally:
        squatter.close()


class _RunnerWithRaisingExecutor:
    """A real host that comes up, then an executor that dies mid-run.

    The host is genuine -- ``build_server_from_env`` and
    ``CodexMcpHost.__enter__`` both succeed and the endpoint is live --
    so every failure this produces belongs to the agent execution, not
    to whetstone's evaluation server.

    ``run_blocking`` raises a bare ``RuntimeError``: not an
    ``ExecutorFailure`` (which the runner already maps to
    ``OpaqueStepError``) and not an ``OpaqueStepError`` itself. It stands
    for any unforeseen defect inside the execution path.
    """

    def __init__(self, world) -> None:
        self._inner = SubprocessCodexRunner(
            executor=_RaisingExecutor(),
            sqlite_path=world.sqlite_path,
            runtime_config=ReferenceEvalRuntimeConfig(
                mutation_field=world.config.candidate_template_field,
                render_contract=toy_template_render_contract(),
            ),
            runtime_config_class=(
                "whetstone.eval.reference_runtime:"
                "ReferenceEvalRuntimeConfig"
            ),
            reward_policy=world.engine.reward_policy,
        )

    def run(self, request, handle, *, lease_token):
        return self._inner.run(request, handle, lease_token=lease_token)


class _RaisingExecutor:
    """Fails the agent execution with something outside the taxonomy."""

    def run_blocking(self, job):
        del job
        raise RuntimeError("executor defect during agent execution")


def test_an_executor_defect_is_not_reported_as_a_host_failure(
    selection_world,
) -> None:
    """A failure after the host is up belongs to the execution, not the host.

    ``codex_mcp_host_failed`` means one specific thing to the ledger:
    whetstone's own evaluation server never came up, so the agent never
    ran and the Step paid for nothing. Wrapping the agent execution in
    the same handler makes every unforeseen failure inside the Codex run
    -- and every teardown failure after a successful one -- claim that,
    which reads as "we never started" for a Step that may well have
    spent money.

    The host here is real and comes up; the executor then dies. That has
    to terminalize under the execution-failure code.
    """
    world = selection_world()

    result = world.run_step_with_runner(_RunnerWithRaisingExecutor(world))

    assert result.status is StepStatus.FAILED
    assert result.terminal_failure is not None
    assert result.terminal_failure.code == CODEX_EXECUTION_FAILED_CODE
    assert result.terminal_failure.code != CODEX_MCP_HOST_FAILED_CODE


class _SchemaInvalidArtifactRunner:
    """Pays for one evaluation, then returns an unparseable artifact.

    This drives the runner's real parse path: ``_parse_output_artifact_bytes``
    with a genuine isolation record, exactly as ``run`` calls it after a
    zero-exit CLI. The failure is the parser's own, not a synthesized one.
    """

    #: Stands in for the record ``_execute_structured`` builds. Only the
    #: fields the evidence has to survive with are asserted on.
    ISOLATION = {
        "strategy": "macos_sandbox_exec",
        "output_truncation": {
            "stdout_truncated": True,
            "stdout_dropped_bytes": 4096,
            "stderr_truncated": False,
            "stderr_dropped_bytes": 0,
        },
    }

    def __init__(self, world, *, calls) -> None:
        self._world = world
        self._calls = calls

    def run(self, request, handle, *, lease_token):
        del handle, lease_token
        from whetstone.optim.codex.runner import (
            _parse_output_artifact_bytes,
        )

        for call_id, template in self._calls:
            self._world.issue(call_id, template)
        # A zero-exit run whose final message is not a valid artifact.
        return _parse_output_artifact_bytes(
            b'{"not": "an artifact"}',
            stdout=b"",
            stderr="",
            run_id=request.run_id,
            isolation=self.ISOLATION,
            stdout_truncated=True,
        )


def test_a_parse_failure_keeps_its_isolation_evidence(
    selection_world,
) -> None:
    """A failed parse is still a run that happened under the sandbox.

    The schema failure was raised as a bare ``OpaqueStepError``, so the
    adapter's ``getattr(exc, "isolation", {})`` stored an empty
    ``codex_isolation``: the terminalized Step recorded no profile, no
    budgets, and -- the evidence that actually explains this failure --
    no truncation flags, leaving a reader unable to tell a malformed
    artifact from one the output budget cut in half.
    """
    world = selection_world()

    result = world.run_step_with_runner(
        _SchemaInvalidArtifactRunner(world, calls=[("c1", _TEMPLATE_A)])
    )

    assert result.status is StepStatus.FAILED
    assert result.terminal_failure is not None
    assert result.terminal_failure.code == CODEX_EXECUTION_FAILED_CODE

    stored = world.store.get(result.state_ref.reference)
    isolation = stored["codex_isolation"]
    assert isolation["strategy"] == "macos_sandbox_exec"
    # The truncation record is what tells a stitched transcript from a
    # contiguous one, so it must survive the failure that references it.
    assert isolation["output_truncation"] == {
        "stdout_truncated": True,
        "stdout_dropped_bytes": 4096,
        "stderr_truncated": False,
        "stderr_dropped_bytes": 0,
    }


class _ForeignRunIdRunner:
    """Pays for real evaluations, then returns another run's artifact.

    ``CodexRunner`` is a Protocol, so the adapter cannot assume the
    runner validated the artifact it hands back. A crossed output path,
    a resumed scratch directory, or a second runner implementation can
    all produce an artifact naming a run this Step is not.
    """

    def __init__(self, world, *, calls, artifact_run_id: str) -> None:
        self._world = world
        self._calls = calls
        self._artifact_run_id = artifact_run_id

    def run(self, request, handle, *, lease_token):
        del handle, request
        for call_id, template in self._calls:
            self._world.issue(call_id, template)
        return CodexRunResult(
            artifact=CodexOutputArtifact(
                run_id=self._artifact_run_id,
                evaluated_call_ids=tuple(
                    call_id for call_id, _template in self._calls
                ),
                selected_call_id=self._calls[0][0],
                lease_token_hash=codex_lease_token_hash(_FIXED_LEASE_TOKEN),
            )
        )


def test_a_foreign_artifact_run_id_terminalizes_and_frees_the_lease(
    selection_world,
) -> None:
    """A mismatched artifact run must fail the Step, not escape it.

    This check used to raise ``OpaqueStepError`` from outside the
    terminalizing block, so it took the one exit that skips the
    harness's effect-lease maintenance: the lease is released only once
    ``invoke`` returns an ``AdapterOutput``, and under ``NO_REDRIVE``
    the run then could not recover until the lease lapsed. It also
    skipped reconciliation, so the evaluations this Step really paid for
    before the artifact arrived stayed off the ledger and undebited.

    The evidence that the lease was released is a state fact, not a
    delay: the identical Step runs again immediately rather than raising
    ``EffectBusyError``.
    """
    world = selection_world()

    result = world.run_step_with_runner(
        _ForeignRunIdRunner(
            world,
            calls=[("c1", _TEMPLATE_A), ("c2", _TEMPLATE_B)],
            artifact_run_id="some-other-run",
        )
    )

    assert result.status is StepStatus.FAILED
    assert result.terminal_failure is not None
    assert (
        result.terminal_failure.code == CODEX_ARTIFACT_RUN_MISMATCH_CODE
    )
    assert result.terminal_failure.details["artifact_run_id"] == (
        "some-other-run"
    )
    assert result.accepted_candidates == ()
    # Reconciliation ran first: the paid evaluations stay reachable from
    # the Step Result and debited from the tool_calls budget.
    assert len(result.tool_evidence) == 2
    assert result.budget_delta.consumed["tool_calls"] == 2

    # The lease reached a terminal state, so the same effect is free.
    retried = world.run_step_with_runner(
        _ForeignRunIdRunner(
            world,
            calls=[("c3", _TEMPLATE_A)],
            artifact_run_id="some-other-run",
        )
    )
    assert retried.status is StepStatus.FAILED


class _ReportedMalformedArgsRunner:
    """Reports a well-formed call and then a malformed one.

    ``_admitted_calls`` walks the reported ids in order, so the first is
    already on the Issued Tool Call ledger when the second is rejected.
    """

    def __init__(self, world, *, good_call_id: str, bad_call_id: str) -> None:
        self._world = world
        self._good = good_call_id
        self._bad = bad_call_id

    def run(self, request, handle, *, lease_token):
        del handle
        self._world.issue(self._good, _TEMPLATE_A)
        self._world.issue_malformed(self._bad)
        return CodexRunResult(
            artifact=CodexOutputArtifact(
                run_id=request.run_id,
                evaluated_call_ids=(self._good, self._bad),
                selected_call_id=self._good,
                lease_token_hash=codex_lease_token_hash(lease_token),
            )
        )


def test_a_reported_malformed_call_is_validated_before_it_is_issued(
    tmp_path,
) -> None:
    """A reported call's recorded args are checked before the ledger sees it.

    ``_admitted_calls`` called the guarded handle first and only then
    read the recorded ``template``/``base_ref``, so an unusable entry
    reached the Issued Tool Call ledger and was then left out of the
    ``issued`` tuple carried to the terminal path. That is the
    ledger-versus-evidence split the reconciliation rule exists to
    prevent, and it attributed the failure to the agent's *reporting*
    rather than to the recorded call's contract.

    Validation now precedes issuance on this path too, and an unusable
    recorded call fails under ``codex_recorded_call_contract``.
    """
    path = str(tmp_path / "codex-reported-malformed.sqlite")
    with open_sqlite(path) as store:
        world = _SelectionWorld(store, failing_call_ids=frozenset())
        world.use_permissive_evaluator()

        result = world.run_step_with_runner(
            _ReportedMalformedArgsRunner(
                world, good_call_id="c1", bad_call_id="c2"
            )
        )

    assert result.status is StepStatus.FAILED
    assert result.terminal_failure is not None
    assert result.terminal_failure.code == CODEX_RECORDED_CALL_CONTRACT_CODE
    assert result.terminal_failure.details["call_id"] == "c2"
    # The malformed call never reaches the ledger. The good call issued
    # ahead of it does, and it is carried to the terminal path rather
    # than re-issued or dropped, so the evidence is exactly the calls
    # the ledger holds -- no ledgered call without matching evidence.
    assert len(result.tool_evidence) == 1
    assert (
        str(result.tool_evidence[0].result.record.call.record.call_id) == "c1"
    )


class _ForeignBaseRunner:
    """Evaluates a call bound to a base outside this Step Request.

    The agent then selects a *legitimate* call, so nothing about the
    selection itself is wrong -- the foreign call is only paid Tool
    Evidence, which is exactly the case the selected-call check misses.
    """

    def __init__(self, world, *, foreign_call_id: str, selected_call_id: str):
        self._world = world
        self._foreign = foreign_call_id
        self._selected = selected_call_id

    def run(self, request, handle, *, lease_token):
        del handle
        self._world.issue_foreign_base(self._foreign, _TEMPLATE_B)
        self._world.issue(self._selected, _TEMPLATE_A)
        return CodexRunResult(
            artifact=CodexOutputArtifact(
                run_id=request.run_id,
                evaluated_call_ids=(self._foreign, self._selected),
                selected_call_id=self._selected,
                lease_token_hash=codex_lease_token_hash(lease_token),
            )
        )


def test_every_reported_call_must_bind_a_request_candidate_as_its_base(
    tmp_path,
) -> None:
    """A base outside the Step Request is rejected, not just an unselected one.

    Only ``_candidate_from_call`` resolved a base against the request,
    and it runs on the *selected* call alone. An agent could therefore
    evaluate a candidate built on another run's ref -- or a forged one
    -- and then select a legitimate call, and the Step would complete
    with paid Tool Evidence for a candidate outside the run's mutation
    ancestry.

    Every admitted call is evidence, so every admitted call's base must
    be one of this Step Request's candidates.
    """
    with open_sqlite(str(tmp_path / "codex-foreign-base.sqlite")) as store:
        world = _SelectionWorld(store, failing_call_ids=frozenset())

        result = world.run_step_with_runner(
            _ForeignBaseRunner(
                world, foreign_call_id="c1", selected_call_id="c2"
            )
        )

    assert result.status is StepStatus.FAILED
    assert result.terminal_failure is not None
    assert result.terminal_failure.code == CODEX_RECORDED_CALL_CONTRACT_CODE
    assert result.terminal_failure.details["call_id"] == "c1"
