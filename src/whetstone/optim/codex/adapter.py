"""The Codex-direct optimizer adapter.

Codex runs exactly one opaque Step. It is granted one Tool -- an external
MCP evaluation endpoint bound to the run's internal split -- and every
evaluation it performs is admitted, leased, persisted, and ledgered before
it sees a score.

A ``TOOL_USING`` Step Result carries Tool Evidence, never intent or
search evidence -- the two are mutually exclusive by contract. The
Issued Tool Call ledger produces one Tool Evidence entry per reported
call, and the adapter reconciles that count against the durable number
of calls this run actually admitted: a shortfall is a terminal failure,
so a Step never completes with paid evaluations invisible to the ledger
and undebited from the ``tool_calls`` budget.

The final candidate is resolved *from the ledger*, not from the artifact.
The artifact names a ``selected_call_id``; the adapter reconstructs the
candidate from that call's recorded, content-addressed ``args``. A
candidate that was never evaluated through the Tool therefore cannot be
returned: there is no path from the artifact to a candidate body except
through a recorded, admitted Tool Call. The selected call must also
carry a real score -- a refusal or a terminally failed evaluation is
rejected, because the agent would be claiming a candidate it never
successfully measured.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from secrets import token_hex
from typing import Any, Protocol

from dr_store import ObjectStore
from pydantic import BaseModel, ConfigDict, Field, StrictStr

from whetstone.core.leasing import ReplayPolicy
from whetstone.core.identity import TerminalFailure, TypedRef
from whetstone.experiment.candidate import Candidate, candidate_reference
from whetstone.optim.adapters import AdapterOutput
from whetstone.optim.contracts import (
    OptimStepRequest,
    StepMode,
    StepStatus,
)
from whetstone.optim.proposal.mutation import DiffCheckError, diff_check
from whetstone.optim.tools.admission import ToolCallState
from whetstone.optim.tools.contracts import RuntimeToolHandle, ToolResult
from whetstone.optim.tools.facade import ToolCallStore

CODEX_ADAPTER_KEY = "codex"
CODEX_OUTPUT_ARTIFACT_SCHEMA = "whetstone.codex_output_artifact"

#: Terminal failure codes this adapter owns. They are persisted on the Step
#: Result, so they are named constants rather than call-site literals.
CODEX_SELECTION_UNEVALUATED_CODE = "codex_selection_unevaluated"
CODEX_LEASE_TOKEN_MISMATCH_CODE = "codex_lease_token_mismatch"
CODEX_SELECTION_CONTRACT_CODE = "codex_selection_contract"
#: The agent admitted more paid evaluations than it reported, so the
#: Issued Tool Call ledger would not be total over admitted calls.
CODEX_UNREPORTED_EVALUATION_CODE = "codex_unreported_evaluation"
#: The selected call reached a terminal state carrying no score.
CODEX_SELECTION_UNSCORED_CODE = "codex_selection_unscored"
#: The Codex process hit the run's hard wall stop.
CODEX_WALL_BUDGET_EXCEEDED_CODE = "codex_wall_budget_exceeded"


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
            # The hard stop of a long-running paid agent. It terminalizes
            # the Step here so the harness's effect-lease maintenance sees
            # a failure and releases the lease; letting this unwind would
            # wedge the run until the lease lapsed.
            return self._failed(
                {
                    "tool_namespace": str(config.store_namespace_key),
                    "harness_store_accepted_call_count": (
                        self._bound_tool_store.accepted_count(
                            config, handle.binding
                        )
                    ),
                    "codex_isolation": exc.isolation,
                },
                TerminalFailure(
                    code=CODEX_WALL_BUDGET_EXCEEDED_CODE,
                    message=str(exc),
                    details={
                        "run_id": request.run_id,
                        "wall_seconds": exc.wall_seconds,
                    },
                ),
            )
        artifact = run.artifact
        if artifact.run_id != request.run_id:
            raise OpaqueStepError(
                "Codex output artifact belongs to another run"
            )
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

        if artifact.lease_token_hash != codex_lease_token_hash(lease_token):
            return self._failed(
                state_delta,
                TerminalFailure(
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
        except _UnevaluatedSelectionError as exc:
            return self._failed(state_delta, exc.failure)

        # Ledger totality. Every admitted call was a paid evaluation; the
        # ledger only sees the ones the agent reported, so a shortfall
        # means paid work is unreachable from the Step Result and the
        # tool_calls budget is under-debited. Completing here would leave
        # invisible spend, so the Step fails loudly instead.
        if len(admitted) < accepted_count:
            return self._failed(
                state_delta,
                TerminalFailure(
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
            return self._failed(
                state_delta,
                TerminalFailure(
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
            )

        # ``COMPLETED`` is also the terminal state of an evaluation that
        # started and then failed: the executor persists a Tool Result
        # carrying ``terminal_failure``, with no ``output`` and no
        # ``reward``. Returning that candidate is the same violation as
        # returning an unevaluated one -- the agent would be claiming a
        # winner it never successfully measured -- so the Step fails.
        if selected.result.output is None or selected.result.reward is None:
            return self._failed(
                state_delta,
                # A Step Result carries one shared terminal failure: when
                # any Tool Evidence entry failed, the Step must fail under
                # that exact failure. A failed evaluation always supplies
                # one; the adapter's own code names the residual case
                # where a scoreless result carried none.
                selected.result.terminal_failure
                or TerminalFailure(
                    code=CODEX_SELECTION_UNSCORED_CODE,
                    message=(
                        "Codex selected a Tool Call whose durable result "
                        "carries no score"
                    ),
                    details={
                        "selected_call_id": artifact.selected_call_id,
                    },
                ),
            )

        try:
            candidate = self._candidate_from_call(request, selected)
        except _SelectionContractError as exc:
            return self._failed(state_delta, exc.failure)

        return AdapterOutput(
            proposed_candidates=(candidate,),
            accepted_candidates=(candidate,),
            proposed_status=StepStatus.COMPLETE,
            state_delta=state_delta,
        )

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
        del request
        config = handle.config
        namespace = str(config.store_namespace_key)
        seen: set[str] = set()
        admitted: list[_AdmittedCall] = []
        for call_id in artifact.evaluated_call_ids:
            if call_id in seen:
                raise _UnevaluatedSelectionError(
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
                raise _UnevaluatedSelectionError(
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
                raise _UnevaluatedSelectionError(
                    TerminalFailure(
                        code=CODEX_SELECTION_UNEVALUATED_CODE,
                        message=(
                            "Codex reported a Tool Call bound to another "
                            "exact Tool Config or capacity binding"
                        ),
                        details={"call_id": call_id},
                    )
                )
            # Re-issue through the guarded handle so the Step's Issued Tool
            # Call ledger records every call Codex made out of process. The
            # ledger reads the durable terminal instead of executing, so
            # this is a durable read, not a second paid evaluation.
            result = handle(call)
            recorded_args = call.args.to_json()
            template = recorded_args.get("template")
            raw_base = recorded_args.get("base_ref")
            if not isinstance(template, str) or not isinstance(raw_base, dict):
                raise _UnevaluatedSelectionError(
                    TerminalFailure(
                        code=CODEX_SELECTION_UNEVALUATED_CODE,
                        message=(
                            "a recorded Codex Tool Call does not carry an "
                            "exact base_ref and template"
                        ),
                        details={"call_id": call_id},
                    )
                )
            admitted.append(
                _AdmittedCall(
                    call_id=call_id,
                    result=result,
                    template=template,
                    base_ref=TypedRef.model_validate(raw_base),
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
        candidate = Candidate(
            candidate_id=selected.call_id,
            base_ref=selected.base_ref,
            payload={
                request.run.record.mutation_field: selected.template,
            },
        )
        try:
            diff_check(base=base, proposed=candidate, run=request.run)
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


class _UnevaluatedSelectionError(Exception):
    def __init__(self, failure: TerminalFailure) -> None:
        self.failure = failure
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
    "CODEX_LEASE_TOKEN_MISMATCH_CODE",
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
