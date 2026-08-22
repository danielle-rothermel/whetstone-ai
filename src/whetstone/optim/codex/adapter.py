"""The Codex-direct optimizer adapter.

Codex runs exactly one opaque Step. It is granted one Tool -- an external
MCP evaluation endpoint bound to the run's internal split -- and every
evaluation it performs is admitted, leased, persisted, and ledgered before
it sees a score.

A ``TOOL_USING`` Step Result carries Tool Evidence, never intent or
search evidence -- the two are mutually exclusive by contract. The
Issued Tool Call ledger produces one Tool Evidence entry per reported
call, and the adapter reconciles that count against the durable number
of calls this run actually admitted. A shortfall is a terminal failure,
so a Step never completes with paid evaluations invisible to the ledger
and undebited from the ``tool_calls`` budget.

Every failing exit leaves through one path, ``_terminalize``, which
reconciles first and fails second. Ledger totality therefore does not
depend on *which* thing went wrong: a shortfall, a corrupted or omitted
lease-token hash, a duplicated or unevaluated call id, a selection that
was never scored, a wall-budget stop with no artifact at all, or a
Codex process that exited nonzero without one. Each of those still
fails the Step under its own code -- it just does not take the run's
paid spend down with it. The durable admission entries are the record
of that spend, and the guarded handle reads their recorded terminals
rather than evaluating, so reconciliation surfaces work already paid
for and never buys more.

The durable entries also tell two kinds of shortfall apart. An omitted
entry that ``COMPLETED`` is the agent under-reporting. An entry still
``ACCEPTED`` is one whetstone's own evaluation server admitted and never
finished -- an in-flight evaluation killed with the host, say -- which
the agent could not have reported; that fails under a
harness-attributed code instead. Its capacity stays debited, because
the admission contract has no typed release and the run really did
commit that evaluation.

The final candidate is resolved *from the ledger*, not from the artifact.
The artifact names a ``selected_call_id``; the adapter reconstructs the
candidate from that call's recorded, content-addressed ``args``. A
candidate that was never evaluated through the Tool therefore cannot be
returned: there is no path from the artifact to a candidate body except
through a recorded, admitted Tool Call. The selected call must also
carry a real score -- a refusal or a terminally failed evaluation is
rejected, because the agent would be claiming a candidate it never
successfully measured.

Every reconciled call is checked against the Step Request, not just the
selected one, because every one of them is paid Tool Evidence on the
Step Result. Its recorded ``template`` and ``base_ref`` must be usable,
and its ``base_ref`` must be one of this Step Request's candidates by
exact ref equality: nothing resolves a ``base_ref`` during evaluation,
so a ref from another run or a forged one would otherwise be scored and
carried as evidence for a candidate outside the run's mutation
ancestry. Both checks run *before* the call is issued, so an unusable
one never reaches the ledger.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from secrets import token_hex
from typing import Any, Final, Protocol

from dr_store import ObjectStore
from pydantic import BaseModel, ConfigDict, Field, StrictStr

from whetstone.core.leasing import ReplayPolicy
from whetstone.core.identity import TerminalFailure, TypedRef
from whetstone.experiment.candidate import Candidate, candidate_reference
from whetstone.optim.adapters import AdapterOutput
from whetstone.optim.contracts import (
    SUPERSEDED_FAILURE_CODES_KEY,
    OptimStepRequest,
    StepMode,
    StepStatus,
)
from whetstone.optim.proposal.mutation import (
    DiffCheckError,
    candidate_from_draft,
)
from whetstone.optim.proposal.proposer import ProposalDraft
from whetstone.optim.tools.admission import ToolCallState
from whetstone.optim.tools.contracts import RuntimeToolHandle, ToolResult
from whetstone.optim.tools.facade import ToolCallStore

CODEX_ADAPTER_KEY = "codex"
CODEX_OUTPUT_ARTIFACT_SCHEMA = "whetstone.codex_output_artifact"

#: Terminal failure codes this adapter owns. They are persisted on the Step
#: Result, so they are named constants rather than call-site literals.
CODEX_SELECTION_UNEVALUATED_CODE = "codex_selection_unevaluated"
CODEX_LEASE_TOKEN_MISMATCH_CODE = "codex_lease_token_mismatch"
#: The artifact the runner returned names a different run than the Step
#: being invoked, so it cannot be this Step's evidence.
CODEX_ARTIFACT_RUN_MISMATCH_CODE = "codex_artifact_run_mismatch"
CODEX_SELECTION_CONTRACT_CODE = "codex_selection_contract"
#: The agent admitted more paid evaluations than it reported, so the
#: Issued Tool Call ledger would not be total over admitted calls.
CODEX_UNREPORTED_EVALUATION_CODE = "codex_unreported_evaluation"
#: The selected call reached a terminal state carrying no score.
CODEX_SELECTION_UNSCORED_CODE = "codex_selection_unscored"
#: whetstone's own evaluation server admitted a call and never reached a
#: terminal for it. The agent had no result to report, so this is a
#: harness failure rather than the agent hiding paid work.
CODEX_EVALUATION_INTERRUPTED_CODE = "codex_evaluation_interrupted"
#: The Codex process hit the run's hard wall stop.
CODEX_WALL_BUDGET_EXCEEDED_CODE = "codex_wall_budget_exceeded"
#: The Codex process failed without producing a usable artifact -- a
#: nonzero exit, an unspawnable process, an unreadable or malformed final
#: message. It is not the wall stop, and it terminalizes the Step so the
#: harness releases the effect lease rather than wedging the run.
CODEX_EXECUTION_FAILED_CODE = "codex_execution_failed"
#: A durable admission entry this Step must reconcile does not carry the
#: recorded ``template`` and ``base_ref`` the adapter rebuilds from, so
#: it cannot be represented as Tool Evidence.
CODEX_RECORDED_CALL_CONTRACT_CODE = "codex_recorded_call_contract"
#: whetstone's own evaluation server could not be built or brought up --
#: a mismatched runtime config, a squatted port, a bind or lifespan
#: failure, a startup that missed its deadline. The agent never ran, so
#: nothing was paid for; the Step still terminalizes so the harness
#: releases the effect lease.
CODEX_MCP_HOST_FAILED_CODE = "codex_mcp_host_failed"


class OpaqueStepError(RuntimeError):
    pass


class CodexStructuredExecutionFailure(OpaqueStepError):
    """The Codex process did not produce a usable structured artifact.

    It lives beside the adapter rather than beside the runner because the
    adapter is what turns these into Step terminal failures, and the
    runner already imports the adapter's contracts.
    """

    def __init__(
        self,
        message: str,
        *,
        stdout: bytes,
        stderr: bytes,
        artifact_bytes: bytes = b"",
        isolation: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.stdout = stdout
        self.stderr = stderr
        self.artifact_bytes = artifact_bytes
        self.isolation = isolation or {}


class CodexWallBudgetExceeded(CodexStructuredExecutionFailure):
    """The Codex process hit the run's hard wall stop.

    This is the expected end of a long-running paid agent, not a defect.
    The adapter turns it into a Step terminal failure so the harness's
    effect-lease maintenance sees a failure and releases the lease; a raw
    subprocess exception would unwind past that block and wedge the run
    until the lease lapsed.
    """

    def __init__(
        self,
        message: str,
        *,
        wall_seconds: float,
        stdout: bytes,
        stderr: bytes,
        isolation: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            stdout=stdout,
            stderr=stderr,
            isolation=isolation,
        )
        self.wall_seconds = wall_seconds


class CodexMcpHostFailure(OpaqueStepError):
    """whetstone's own evaluation server never came up, or never came down.

    This is a whetstone-side failure, not the agent's: the Codex process
    either never started or never got a usable endpoint, so nothing was
    paid for. It is an ``OpaqueStepError`` so it leaves through the
    adapter's single terminalizing path -- otherwise it would unwind past
    the adapter checkpoint and leave this ``NO_REDRIVE`` effect
    nonterminal until the lease lapsed.

    It carries no isolation evidence because there is no sandboxed
    process to have evidence about; ``cause`` preserves the underlying
    diagnostic for the terminal failure's message.
    """

    def __init__(self, message: str, *, cause: BaseException) -> None:
        super().__init__(f"{message}: {cause}")
        self.cause = cause


class CodexOutputArtifact(BaseModel):
    """What the Codex CLI writes as its final structured message.

    It carries no candidate body. ``selected_call_id`` names one Tool Call
    the run actually admitted; the adapter rebuilds the candidate from that
    call's durable ``args``. ``selected_call_id`` is absent exactly when
    Codex chose to keep the seed.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: StrictStr
    #: Every ``call_id`` Codex evaluated, in the order it issued them.
    evaluated_call_ids: tuple[StrictStr, ...] = ()
    selected_call_id: StrictStr | None = None
    #: Hash of the run-scoped lease token the adapter minted for this Step.
    lease_token_hash: StrictStr = ""
    conversation_evidence: dict[str, Any] = Field(default_factory=dict)
    control_cost: dict[str, Any] = Field(default_factory=dict)


#: The fields the *agent* is asked to produce. ``conversation_evidence`` and
#: ``control_cost`` are deliberately absent: the runner overwrites
#: ``conversation_evidence`` with whetstone's own process evidence after the
#: run, and nothing reads ``control_cost``. Asking the model for either would
#: invite it to invent evidence whetstone then discards.
CODEX_ARTIFACT_AGENT_FIELDS: Final = (
    "run_id",
    "evaluated_call_ids",
    "selected_call_id",
    "lease_token_hash",
)


def codex_output_schema(
    *, run_id: str, lease_token_hash: str
) -> dict[str, Any]:
    """The structured-output schema handed to the Codex CLI.

    This is written out explicitly rather than derived from
    :meth:`CodexOutputArtifact.model_json_schema`, because the two have
    different jobs and different validators. The model is whetstone's
    storage shape; this is a contract the OpenAI structured-output
    validator enforces, and it is stricter than JSON Schema:

    * every object -- nested ones included -- must set
      ``additionalProperties: false``;
    * every property must appear in ``required``, so an optional field is
      expressed as a nullable type rather than by omission.

    Pydantic emits ``additionalProperties: true`` for a ``dict[str, Any]``
    field, so the derived schema was rejected with
    ``invalid_json_schema`` before the agent produced a single token. The
    fake CLI never validated the schema, so only a real run surfaced it.

    ``run_id`` and ``lease_token_hash`` are pinned as constants: a
    non-conforming artifact then fails at the CLI boundary rather than as
    a Step terminal failure.
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(CODEX_ARTIFACT_AGENT_FIELDS),
        "properties": {
            "run_id": {"type": "string", "const": run_id},
            "evaluated_call_ids": {
                "type": "array",
                "items": {"type": "string"},
            },
            # Null is the seed-retaining selection, so it is a nullable
            # type rather than an omitted property.
            "selected_call_id": {"type": ["string", "null"]},
            "lease_token_hash": {
                "type": "string",
                "const": lease_token_hash,
            },
        },
    }


@dataclass(frozen=True, slots=True)
class CodexRunResult:
    artifact: CodexOutputArtifact


class CodexRunner(Protocol):
    def run(
        self,
        request: OptimStepRequest,
        handle: RuntimeToolHandle,
        *,
        lease_token: str,
    ) -> CodexRunResult: ...


@dataclass(frozen=True, slots=True)
class _AdmittedCall:
    """One Tool Call this Step admitted, with its durable terminal."""

    call_id: str
    result: ToolResult
    template: str
    base_ref: TypedRef


class CodexAdapter:
    def __init__(
        self,
        runner: CodexRunner,
        *,
        store: ObjectStore,
        tool_store: ToolCallStore | None = None,
        lease_token_factory: Callable[[], str] | None = None,
    ) -> None:
        self._runner = runner
        self._store = store
        self._tool_store = tool_store
        self._lease_token_factory = lease_token_factory or _mint_lease_token

    def bind_tool_store(self, tool_store: ToolCallStore) -> None:
        """Bind the exact Tool Call Store the harness admits through.

        ``build_runtime`` owns that store but needs the adapter registry to
        build it, so a Codex adapter constructed for the registry binds the
        store afterwards. Reading durable entries from a second store over
        the same database would not see an in-memory admission authority.
        """
        self._tool_store = tool_store

    @property
    def _bound_tool_store(self) -> ToolCallStore:
        if self._tool_store is None:
            raise OpaqueStepError(
                "the Codex adapter has no bound Tool Call Store; call "
                "bind_tool_store with the runtime's exact store"
            )
        return self._tool_store

    @property
    def key(self) -> str:
        return CODEX_ADAPTER_KEY

    @property
    def mode(self) -> StepMode:
        return StepMode.TOOL_USING

    @property
    def required_replay_policy(self) -> ReplayPolicy:
        return ReplayPolicy.NO_REDRIVE

    def invoke(
        self,
        request: OptimStepRequest,
        handles: tuple[RuntimeToolHandle, ...],
    ) -> AdapterOutput:
        if request.step_index != 0:
            raise OpaqueStepError("Codex runs exactly one opaque step")
        if len(handles) != 1:
            raise OpaqueStepError(
                "Codex requires exactly one Runtime Tool Handle for its "
                "external MCP evaluation Tool"
            )
        handle = handles[0]
        config = handle.config
        if not str(config.endpoint_key).startswith("mcp"):
            raise OpaqueStepError("Codex evaluation must be external MCP")

        lease_token = self._lease_token_factory()
        try:
            run = self._runner.run(request, handle, lease_token=lease_token)
        except CodexWallBudgetExceeded as exc:
            # The hard stop of a long-running paid agent, and the expected
            # end of one. It terminalizes the Step here so the harness's
            # effect-lease maintenance sees a failure and releases the
            # lease; letting this unwind would wedge the run until the
            # lease lapsed.
            return self._terminalize(
                handle,
                state_delta={
                    "tool_namespace": str(config.store_namespace_key),
                    "codex_isolation": exc.isolation,
                },
                fallback=TerminalFailure(
                    code=CODEX_WALL_BUDGET_EXCEEDED_CODE,
                    message=str(exc),
                    details={
                        "run_id": request.run_id,
                        "wall_seconds": exc.wall_seconds,
                    },
                ),
            )
        except CodexMcpHostFailure as exc:
            # whetstone's own evaluation server, not the agent: the Codex
            # process never got a usable endpoint, so this Step paid for
            # nothing. It still terminalizes, under its own code so the
            # ledger tells a host that never came up apart from an agent
            # that failed.
            return self._terminalize(
                handle,
                state_delta={
                    "tool_namespace": str(config.store_namespace_key),
                    "codex_isolation": {},
                },
                fallback=TerminalFailure(
                    code=CODEX_MCP_HOST_FAILED_CODE,
                    message=str(exc),
                    details={"run_id": request.run_id},
                ),
            )
        except OpaqueStepError as exc:
            # A nonzero exit, an unspawnable process, an unreadable or
            # malformed final message. It is not the wall stop, but it
            # ends the Step the same way: the harness only runs its
            # effect-lease maintenance once the adapter returns an
            # AdapterOutput, so an exception escaping here leaves the
            # effect non-terminal, and this adapter's NO_REDRIVE policy
            # then blocks the run from recovering until the lease lapses.
            #
            # This catches the base class, not just the structured
            # subclass: a zero-exit run whose artifact fails schema
            # validation, and a dr-exec ExecutorFailure, both raise the
            # base error from inside the runner. Terminalizing is about
            # releasing the lease, so every way the runner can fail has
            # to leave through here -- narrowing this to the subclass is
            # what let those two escape.
            isolation = getattr(exc, "isolation", {})
            return self._terminalize(
                handle,
                state_delta={
                    "tool_namespace": str(config.store_namespace_key),
                    "codex_isolation": isolation,
                },
                fallback=TerminalFailure(
                    code=CODEX_EXECUTION_FAILED_CODE,
                    message=str(exc),
                    details={"run_id": request.run_id},
                ),
            )
        artifact = run.artifact
        typed_artifact_ref = self._persist_artifact(artifact)
        # The durable, per-run count of calls this exact Tool Config and
        # capacity binding admitted. It is ground truth about what was
        # paid for, independent of what the agent chose to report.
        accepted_count = self._bound_tool_store.accepted_count(
            config, handle.binding
        )
        state_delta = {
            "codex_output_artifact_ref": typed_artifact_ref.model_dump(
                mode="json"
            ),
            "tool_namespace": str(config.store_namespace_key),
            "harness_store_accepted_call_count": accepted_count,
        }

        if artifact.run_id != request.run_id:
            # Same shape as the lease-token mismatch below, and handled
            # the same way. A runner that validated the artifact's run
            # would already have failed; this is the adapter's own
            # boundary check on the ``CodexRunner`` protocol, so it must
            # not assume the runner ran one. Raising here skipped the
            # harness's effect-lease maintenance entirely -- the lease is
            # released only once ``invoke`` returns an ``AdapterOutput``
            # -- so a NO_REDRIVE run wedged until the lease lapsed, with
            # any evaluations this Step already paid for left off the
            # ledger. Reconciling first keeps that spend visible; the
            # foreign run still fails the Step.
            return self._terminalize(
                handle,
                state_delta=state_delta,
                fallback=TerminalFailure(
                    code=CODEX_ARTIFACT_RUN_MISMATCH_CODE,
                    message=(
                        "Codex output artifact belongs to another run"
                    ),
                    details={
                        "run_id": request.run_id,
                        "artifact_run_id": artifact.run_id,
                    },
                ),
            )

        if artifact.lease_token_hash != codex_lease_token_hash(lease_token):
            # The agent holds the bearer token for the MCP endpoint, so it
            # can pay for evaluations and *then* omit or corrupt the hash
            # that proves this artifact is this Step's. Rejecting the hash
            # before reconciling would hand it an exit from ledger
            # totality: real spend recorded in accepted_count, no Tool
            # Evidence, and a zero tool_calls debit. The reconciliation
            # runs first; the hash still fails the Step.
            return self._terminalize(
                handle,
                state_delta=state_delta,
                fallback=TerminalFailure(
                    code=CODEX_LEASE_TOKEN_MISMATCH_CODE,
                    message=(
                        "Codex output artifact does not carry this Step's "
                        "exact run lease token"
                    ),
                    details={"run_id": request.run_id},
                ),
            )

        try:
            admitted = self._admitted_calls(request, artifact, handle)
        except (
            _UnevaluatedSelectionError,
            _RecordedCallContractError,
        ) as exc:
            # _admitted_calls re-issues as it goes, so calls may already
            # be on the ledger when it rejects a later one.
            return self._terminalize(
                handle,
                state_delta=state_delta,
                fallback=exc.failure,
                already_issued=exc.issued,
            )

        # Ledger totality. Every admitted call was a paid evaluation; the
        # ledger only sees the ones the agent reported, so a shortfall
        # means paid work is unreachable from the Step Result and the
        # tool_calls budget is under-debited. Completing here would leave
        # invisible spend, so the Step fails loudly instead.
        if len(admitted) < accepted_count:
            return self._terminalize(
                handle,
                state_delta=state_delta,
                fallback=TerminalFailure(
                    code=CODEX_UNREPORTED_EVALUATION_CODE,
                    message=(
                        "Codex admitted more paid evaluations than it "
                        "reported, so the Issued Tool Call ledger is not "
                        "total over this Step's admitted calls"
                    ),
                    details={
                        "admitted_call_count": accepted_count,
                        "reported_call_count": len(admitted),
                        "evaluated_call_ids": list(
                            artifact.evaluated_call_ids
                        ),
                    },
                ),
                already_issued=admitted,
            )

        if artifact.selected_call_id is None:
            return AdapterOutput(
                proposed_candidates=(),
                accepted_candidates=(),
                proposed_status=StepStatus.COMPLETE,
                seed_retained=True,
                retained_candidate=self._seed_candidate(request),
                state_delta=state_delta,
            )

        selected = next(
            (
                call
                for call in admitted
                if call.call_id == artifact.selected_call_id
            ),
            None,
        )
        if selected is None:
            return self._terminalize(
                handle,
                state_delta=state_delta,
                fallback=TerminalFailure(
                    code=CODEX_SELECTION_UNEVALUATED_CODE,
                    message=(
                        "Codex selected a candidate that was never evaluated "
                        "through an admitted Tool Call"
                    ),
                    details={
                        "selected_call_id": artifact.selected_call_id,
                        "evaluated_call_ids": list(
                            artifact.evaluated_call_ids
                        ),
                    },
                ),
                already_issued=admitted,
            )

        # ``COMPLETED`` is also the terminal state of an evaluation that
        # started and then failed: the executor persists a Tool Result
        # carrying ``terminal_failure``, with no ``output`` and no
        # ``reward``. Returning that candidate is the same violation as
        # returning an unevaluated one -- the agent would be claiming a
        # winner it never successfully measured -- so the Step fails.
        if selected.result.output is None or selected.result.reward is None:
            return self._terminalize(
                handle,
                state_delta=state_delta,
                # A Step Result carries one shared terminal failure. When
                # exactly one evaluation failed the Step adopts that
                # failure, so the Step says the same thing its evidence
                # does. When several failed, or the selected result
                # carried no failure at all, no single nested failure can
                # be adopted and the adapter's own code supersedes them.
                fallback=TerminalFailure(
                    code=CODEX_SELECTION_UNSCORED_CODE,
                    message=(
                        "Codex selected a Tool Call whose durable result "
                        "carries no score"
                    ),
                    details={
                        "selected_call_id": artifact.selected_call_id,
                    },
                ),
                already_issued=admitted,
                adoptable=True,
            )

        try:
            candidate = self._candidate_from_call(request, selected)
        except _SelectionContractError as exc:
            return self._terminalize(
                handle,
                state_delta=state_delta,
                fallback=exc.failure,
                already_issued=admitted,
            )

        return AdapterOutput(
            proposed_candidates=(candidate,),
            accepted_candidates=(candidate,),
            proposed_status=StepStatus.COMPLETE,
            state_delta=state_delta,
        )

    def _terminalize(
        self,
        handle: RuntimeToolHandle,
        *,
        state_delta: dict[str, Any],
        fallback: TerminalFailure,
        already_issued: tuple[_AdmittedCall, ...] = (),
        adoptable: bool = False,
    ) -> AdapterOutput:
        """The one path a failing Codex Step leaves by.

        Every terminalizing exit reconciles first and fails second, so
        ledger totality does not depend on *which* thing went wrong. The
        durable admission entries are the record of what this run paid
        for, independent of what the agent chose to report -- or whether
        it produced a usable artifact at all. Every COMPLETED entry not
        already on the ledger is re-issued through the guarded handle,
        which reads the recorded terminal rather than evaluating, so this
        surfaces work already paid for and never buys more.

        An entry still ACCEPTED is one whetstone's own evaluation server
        admitted and never reached a terminal for. The agent had no
        result to report, so that is a harness failure rather than the
        agent hiding paid work, and it supersedes the caller's code.

        ``already_issued`` names the calls the caller already put on the
        ledger, so they are not issued twice. ``adoptable`` lets a
        failure that really is a verdict on one evaluation adopt that
        evaluation's own failure; everything else is the adapter's own
        code and supersedes the nested ones.
        """
        issued = {call.call_id for call in already_issued}
        entries = self._bound_tool_store.admitted_entries(
            handle.config, handle.binding
        )
        state_delta = {
            **state_delta,
            "harness_store_accepted_call_count": (
                self._bound_tool_store.accepted_count(
                    handle.config, handle.binding
                )
            ),
        }
        outstanding = tuple(
            entry for entry in entries if str(entry.call_id) not in issued
        )
        try:
            reconciled = already_issued + self._issue_completed(
                tuple(
                    entry
                    for entry in outstanding
                    if entry.state is ToolCallState.COMPLETED
                ),
                handle,
            )
        except _RecordedCallContractError as exc:
            # The entry is already on the ledger by contract -- it is a
            # durable admitted call -- so it can never be dropped
            # silently. It simply cannot be represented as reconciled
            # evidence, which is itself the failure.
            return self._failed(state_delta, exc.failure)
        interrupted = tuple(
            str(entry.call_id)
            for entry in outstanding
            if entry.state is ToolCallState.ACCEPTED
        )
        if interrupted:
            # whetstone's own server admitted these and never reached a
            # terminal for them -- an in-flight evaluation killed with
            # the host, or a server that died mid-call. Their capacity is
            # already debited and the admission contract has no typed
            # release (ToolCallState is ACCEPTED, REFUSED, or COMPLETED
            # only), so the slot stays consumed: the run really did
            # commit that evaluation, and inventing a release here would
            # let a killed call be silently re-bought. The Step names
            # them instead, under a harness-attributed code.
            return self._failed(
                state_delta,
                _shared_failure(
                    reconciled,
                    fallback=TerminalFailure(
                        code=CODEX_EVALUATION_INTERRUPTED_CODE,
                        message=(
                            "whetstone's evaluation server admitted a paid "
                            "evaluation and never reached a terminal for it, "
                            "so this Step's ledger cannot be made total"
                        ),
                        details={
                            **fallback.details.to_json(),
                            "interrupted_call_ids": list(interrupted),
                        },
                    ),
                    always_supersede=True,
                ),
            )
        return self._failed(
            state_delta,
            _shared_failure(
                reconciled,
                fallback=fallback,
                always_supersede=not adoptable,
            ),
        )

    @staticmethod
    def _issue_completed(
        entries: tuple[Any, ...],
        handle: RuntimeToolHandle,
    ) -> tuple[_AdmittedCall, ...]:
        """Put durable completed calls on the ledger.

        The handle reads the durable terminal rather than evaluating, so
        this records work already paid for; it never buys more.

        The recorded ``template`` and ``base_ref`` are validated *before*
        the call is issued. Issuing first and skipping an unusable entry
        afterwards would leave it on the Issued Tool Call ledger while
        omitting it from the reconciled evidence the shared-failure rule
        reasons over, so the Step Result's outer code could contradict
        its own nested Tool Evidence. An unusable recorded call is a
        typed failure instead -- never a silent skip.
        """
        issued: list[_AdmittedCall] = []
        for entry in entries:
            call = entry.tool_call.record
            recorded_args = call.args.to_json()
            template = recorded_args.get("template")
            raw_base = recorded_args.get("base_ref")
            if not isinstance(template, str) or not isinstance(raw_base, dict):
                raise _RecordedCallContractError(
                    TerminalFailure(
                        code=CODEX_RECORDED_CALL_CONTRACT_CODE,
                        message=(
                            "a durable admitted Tool Call does not carry the "
                            "recorded template and base_ref this Step must "
                            "reconcile it from"
                        ),
                        details={"call_id": str(entry.call_id)},
                    )
                )
            issued.append(
                _AdmittedCall(
                    call_id=str(entry.call_id),
                    result=handle(call),
                    template=template,
                    base_ref=TypedRef.model_validate(raw_base),
                )
            )
        return tuple(issued)

    @staticmethod
    def _failed(
        state_delta: dict[str, Any],
        failure: TerminalFailure,
    ) -> AdapterOutput:
        return AdapterOutput(
            proposed_status=StepStatus.FAILED,
            terminal_failure=failure,
            state_delta=state_delta,
        )

    def _persist_artifact(self, artifact: CodexOutputArtifact) -> TypedRef:
        artifact_ref, _ = self._store.put(
            CODEX_OUTPUT_ARTIFACT_SCHEMA, artifact.model_dump(mode="json")
        )
        return TypedRef(
            schema_name=artifact_ref.schema,
            content_hash=artifact_ref.content_hash,
        )

    @staticmethod
    def _seed_candidate(request: OptimStepRequest) -> Candidate:
        seed_ref = request.run.record.initial_candidate_ref
        if seed_ref is None:
            raise OpaqueStepError(
                "a seed-retaining Codex Step requires the run to name its "
                "initial candidate"
            )
        for candidate in request.candidates:
            if candidate_reference(candidate) == seed_ref:
                return candidate
        raise OpaqueStepError(
            "the run seed candidate is not on the Codex Step Request"
        )

    def _admitted_calls(
        self,
        request: OptimStepRequest,
        artifact: CodexOutputArtifact,
        handle: RuntimeToolHandle,
    ) -> tuple[_AdmittedCall, ...]:
        config = handle.config
        namespace = str(config.store_namespace_key)
        request_bases = {
            candidate_reference(candidate).record_ref
            for candidate in request.candidates
        }
        seen: set[str] = set()
        admitted: list[_AdmittedCall] = []

        def reject(failure: TerminalFailure) -> _UnevaluatedSelectionError:
            return _UnevaluatedSelectionError(
                failure, issued=tuple(admitted)
            )

        for call_id in artifact.evaluated_call_ids:
            if call_id in seen:
                raise reject(
                    TerminalFailure(
                        code=CODEX_SELECTION_UNEVALUATED_CODE,
                        message=(
                            "Codex reported one Tool Call id more than once"
                        ),
                        details={"call_id": call_id},
                    )
                )
            seen.add(call_id)
            entry = self._bound_tool_store.find_entry(
                store_namespace_key=namespace,
                call_id=call_id,
            )
            if entry is None or entry.state is not ToolCallState.COMPLETED:
                raise reject(
                    TerminalFailure(
                        code=CODEX_SELECTION_UNEVALUATED_CODE,
                        message=(
                            "Codex reported a Tool Call with no durable "
                            "completed admission entry"
                        ),
                        details={"call_id": call_id},
                    )
                )
            call = entry.tool_call.record
            if (
                call.tool_config != handle.tool_config_ref
                or call.capacity_binding != handle.binding
            ):
                raise reject(
                    TerminalFailure(
                        code=CODEX_SELECTION_UNEVALUATED_CODE,
                        message=(
                            "Codex reported a Tool Call bound to another "
                            "exact Tool Config or capacity binding"
                        ),
                        details={"call_id": call_id},
                    )
                )
            # Validate the recorded args *before* issuing. Issuing first
            # would put the call on the Step's Issued Tool Call ledger
            # and then reject it, leaving a ledgered call that the
            # reconciled evidence does not know about -- the same
            # ledger-versus-evidence split ``_issue_completed`` avoids.
            recorded_args = call.args.to_json()
            template = recorded_args.get("template")
            raw_base = recorded_args.get("base_ref")
            if not isinstance(template, str) or not isinstance(raw_base, dict):
                raise _RecordedCallContractError(
                    TerminalFailure(
                        code=CODEX_RECORDED_CALL_CONTRACT_CODE,
                        message=(
                            "a recorded Codex Tool Call does not carry an "
                            "exact base_ref and template"
                        ),
                        details={"call_id": call_id},
                    ),
                    issued=tuple(admitted),
                )
            base_ref = TypedRef.model_validate(raw_base)
            # Every admitted call is paid Tool Evidence on this Step's
            # Result, so every one of them must sit inside the run's
            # mutation ancestry -- not only the call the agent selected.
            # A syntactically valid ref from another run, or a forged
            # one, is not a candidate this Step Request offered, and
            # scoring it would put a candidate outside the ancestry on
            # the Step's evidence.
            if base_ref not in request_bases:
                raise _RecordedCallContractError(
                    TerminalFailure(
                        code=CODEX_RECORDED_CALL_CONTRACT_CODE,
                        message=(
                            "a recorded Codex Tool Call binds a base that is "
                            "not a candidate on this Step Request"
                        ),
                        details={
                            "call_id": call_id,
                            "base_ref": raw_base,
                        },
                    ),
                    issued=tuple(admitted),
                )
            # Re-issue through the guarded handle so the Step's Issued Tool
            # Call ledger records every call Codex made out of process. The
            # ledger reads the durable terminal instead of executing, so
            # this is a durable read, not a second paid evaluation.
            admitted.append(
                _AdmittedCall(
                    call_id=call_id,
                    result=handle(call),
                    template=template,
                    base_ref=base_ref,
                )
            )
        return tuple(admitted)

    def _candidate_from_call(
        self,
        request: OptimStepRequest,
        selected: _AdmittedCall,
    ) -> Candidate:
        bases = {
            candidate_reference(candidate).record_ref: candidate
            for candidate in request.candidates
        }
        base = bases.get(selected.base_ref)
        if base is None:
            raise _SelectionContractError(
                TerminalFailure(
                    code=CODEX_SELECTION_CONTRACT_CODE,
                    message=(
                        "the selected Codex Tool Call does not bind an exact "
                        "Step Request candidate as its base"
                    ),
                    details={"call_id": selected.call_id},
                )
            )
        # Build from the base payload and replace only the mutation
        # field, exactly as every other proposal path does. Rebuilding
        # from the mutation field alone would drop every other payload
        # field the base carries, and diff_check requires those to equal
        # the base's -- so a legitimately evaluated selection on any
        # multi-field candidate would always fail the mutation diff.
        try:
            candidate = candidate_from_draft(
                base=base,
                candidate_id=selected.call_id,
                draft=ProposalDraft(template=selected.template),
                run=request.run,
            )
        except DiffCheckError as exc:
            raise _SelectionContractError(
                TerminalFailure(
                    code=CODEX_SELECTION_CONTRACT_CODE,
                    message=(
                        "the candidate rebuilt from the selected Codex Tool "
                        "Call violates the run mutation diff"
                    ),
                    details={
                        "call_id": selected.call_id,
                        "error": str(exc),
                    },
                )
            ) from exc
        return candidate


def _shared_failure(
    admitted: tuple[_AdmittedCall, ...],
    *,
    fallback: TerminalFailure,
    always_supersede: bool = False,
) -> TerminalFailure:
    """The one terminal failure a failing Step Result may carry.

    Every reported call is re-issued through the guarded handle, so a
    terminally failed evaluation puts its own failure on the Step's Tool
    Evidence. The Step Result contract requires the outer failure to
    account for those: adopt the single nested failure when there is
    exactly one, and otherwise supersede them under the adapter's own
    code, naming every nested code it stands for.

    ``always_supersede`` is for failures the adapter owns outright --
    under-reporting is the agent's contract violation, not a verdict on
    any one evaluation -- so they never masquerade as an evaluation
    failure even when only one evaluation failed.
    """
    nested = tuple(
        call.result.terminal_failure
        for call in admitted
        if call.result.terminal_failure is not None
    )
    if not nested:
        return fallback
    if not always_supersede and all(
        failure == nested[0] for failure in nested
    ):
        return nested[0]
    details = dict(fallback.details.to_json())
    details[SUPERSEDED_FAILURE_CODES_KEY] = sorted(
        str(failure.code) for failure in nested
    )
    return TerminalFailure(
        code=fallback.code,
        message=fallback.message,
        details=details,
    )


class _UnevaluatedSelectionError(Exception):
    """A reported call cannot be admitted as this Step's Tool Evidence.

    ``_admitted_calls`` re-issues each reported call as it validates it,
    so calls are already on the Issued Tool Call ledger when a later one
    is rejected. Those are carried here: the terminal path must not
    re-issue them (call ids are unique within a Step attempt) and must
    not omit them from the evidence it reconciles.
    """

    def __init__(
        self,
        failure: TerminalFailure,
        *,
        issued: tuple[_AdmittedCall, ...] = (),
    ) -> None:
        self.failure = failure
        self.issued = issued
        super().__init__(failure.message)


class _RecordedCallContractError(Exception):
    """A durable admitted call cannot be rebuilt into Tool Evidence.

    Raised from both reconciliation paths. ``_issue_completed``
    validates before issuing anything and carries no ``issued``;
    ``_admitted_calls`` walks the reported ids in order, so earlier
    ones are already on the ledger when a later one is rejected and
    must be carried here rather than re-issued or dropped.
    """

    def __init__(
        self,
        failure: TerminalFailure,
        *,
        issued: tuple[_AdmittedCall, ...] = (),
    ) -> None:
        self.failure = failure
        self.issued = issued
        super().__init__(failure.message)


class _SelectionContractError(Exception):
    def __init__(self, failure: TerminalFailure) -> None:
        self.failure = failure
        super().__init__(failure.message)


def _mint_lease_token() -> str:
    return token_hex(32)


def codex_lease_token_hash(token: str) -> str:
    """Hash a run lease token for recording in the output artifact."""
    return sha256(token.encode("utf-8")).hexdigest()


def codex_run_lease_binding(
    *,
    token: str,
    store_namespace_key: str,
    tool_config_hash: str,
    capacity_scope: str,
    capacity_subject: str,
) -> str:
    """Bind a run lease token to the exact run the server may serve.

    The token alone proves only that its holder received some token. The
    MCP server checks this digest against the one it recomputes from its
    own Tool Config and capacity binding, so a token minted for a
    different run -- or replayed at a server started for one -- does not
    verify. The inputs are joined under a length prefix so no two
    different tuples can produce the same payload.
    """
    parts = (
        token,
        store_namespace_key,
        tool_config_hash,
        capacity_scope,
        capacity_subject,
    )
    payload = "".join(f"{len(part)}:{part}" for part in parts)
    return sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "CODEX_ADAPTER_KEY",
    "CODEX_ARTIFACT_RUN_MISMATCH_CODE",
    "CODEX_EVALUATION_INTERRUPTED_CODE",
    "CODEX_EXECUTION_FAILED_CODE",
    "CODEX_LEASE_TOKEN_MISMATCH_CODE",
    "CODEX_RECORDED_CALL_CONTRACT_CODE",
    "CODEX_OUTPUT_ARTIFACT_SCHEMA",
    "CODEX_SELECTION_CONTRACT_CODE",
    "CODEX_SELECTION_UNEVALUATED_CODE",
    "CODEX_SELECTION_UNSCORED_CODE",
    "CODEX_UNREPORTED_EVALUATION_CODE",
    "CODEX_WALL_BUDGET_EXCEEDED_CODE",
    "CodexAdapter",
    "CodexOutputArtifact",
    "CodexRunResult",
    "CodexRunner",
    "CodexStructuredExecutionFailure",
    "CodexWallBudgetExceeded",
    "OpaqueStepError",
    "codex_lease_token_hash",
    "codex_run_lease_binding",
]
