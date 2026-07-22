"""The algorithm-neutral durable Optimization Step harness engine.

One engine drives every run type. Per Step, in order:

1. **Validate** the immutable Step Request (identities, order, budgets,
   output contract, serialized-tools-only). Reject runtime handles.
2. **Restart/idempotency first.** Before invoking any adapter code or external
   effect, resolve whether this Step already has a persisted Step Result. If
   one exists for the same request, replay it idempotently; a *different*
   Result for the same Step identity conflicts and never replaces the winner.
3. **Dispatch by declared Step mode:**
   * *pure* — pass-through: the adapter returns unchanged candidates.
   * *proposal-only* — invoke the adapter, durably **checkpoint** its typed
     output, then resolve each Evaluation Intent **outside** the invocation
     under its exact target Eval Config. A crash after the checkpoint reuses it
     and never reruns a completed proposal invocation.
   * *tool-using* — construct Runtime Tool Handles **only** at the execution
     boundary, execute, and record every Tool Result + Store Entry.
4. **Finalize exactly once:** compute accepted candidates + state/history
   snapshot refs, account consumed/remaining budgets (carried forward from the
   prior Result, never process memory), assign the status, **persist exactly
   one Step Result**, and only then follow a back-edge or terminalize.

No opaque mutable optimizer pickle is authoritative: the restart position is
derivable exclusively from persisted Step Results, the checkpointed proposal
output, and referenced state/history/evidence — never from process memory.
"""

from __future__ import annotations

from typing import Protocol

from dr_store import BindingConflictError, BindStatus, ObjectStore

from whetstone.optimization.adapters import AdapterOutput, OptimizerAdapter
from whetstone.optimization.identity import TypedRef, typed_ref_for_record
from whetstone.optimization.schema import (
    STEP_REQUEST_SCHEMA,
    STEP_RESULT_SCHEMA,
    BudgetState,
    EvaluationIntent,
    IntentResolution,
    OptimizationProposal,
    OptimizationResult,
    OptimizationStepRequest,
    OptimizationStepResult,
    StepMode,
    StepStatus,
    ToolEvidence,
    step_request_reference,
)
from whetstone.optimization.tool_store import ToolCallStore
from whetstone.optimization.tools import (
    RuntimeToolHandle,
    ToolConfig,
    ToolResult,
    tool_result_reference,
)

__all__ = [
    "ADAPTER_CHECKPOINT_SCHEMA",
    "EvaluationService",
    "OptimizationHarness",
    "StepResultConflictError",
    "ToolExecutor",
]

# Stored record schema for the durably checkpointed typed adapter output.
ADAPTER_CHECKPOINT_SCHEMA = "whetstone.optimization_adapter_checkpoint"


class StepResultConflictError(Exception):
    """A different Step Result exists for the same Step identity.

    The durable winner is preserved and exposed; the losing candidate is
    described. There is no overwrite path — a Step Result is never updated in
    place.
    """

    def __init__(
        self,
        *,
        run_id: str,
        step_index: int,
        existing: TypedRef,
        requested: TypedRef,
    ) -> None:
        self.run_id = run_id
        self.step_index = step_index
        self.existing = existing
        self.requested = requested
        super().__init__(
            f"Step ({run_id}, index {step_index}) already has Step Result "
            f"{existing.content_hash}; refusing divergent Result "
            f"{requested.content_hash}"
        )


class EvaluationService(Protocol):
    """Whetstone's external evaluation path for Evaluation Intents.

    Resolves one Intent OUTSIDE the optimizer invocation: validate ->
    materialize -> bind the exact target Eval Config under the declared
    Evaluation Context role -> plan/execute Rollouts -> aggregate. Returns a
    typed :class:`IntentResolution` whose ``resolved_eval_config_hash`` MUST
    equal the Intent's target. It is the same path a tool's internal evaluation
    calls.
    """

    def resolve_evaluation_intent(
        self, intent: EvaluationIntent
    ) -> IntentResolution: ...


class ToolExecutor(Protocol):
    """Constructs a Runtime Tool Handle from a Tool Config at execution.

    The handle is non-serializable and exists only during the tool-using Step's
    execution boundary. The executor binds the Tool Config's internal-role Eval
    Config, capacity, and store namespace to a live callable that records every
    call through the authoritative Tool Call Store.
    """

    def runtime_handle(
        self, config: ToolConfig, store: ToolCallStore
    ) -> RuntimeToolHandle: ...


class OptimizationHarness:
    """The algorithm-neutral durable harness engine.

    Owns request validation, execution-mode dispatch, external evaluation,
    finalization, restart/idempotency, budgets, evidence, and terminal
    Optimization Result assembly. Every authoritative datum is durable in
    dr-store: records are content-addressed through ``put``/``get`` and the
    per-Step Step-Result binding + the per-Step proposal checkpoint are atomic
    ``bind``/``resolve`` entries. The harness carries no authoritative process
    state, so a *fresh* harness instance over the same durable backend
    resolves the exact restart position from the store alone — never rerunning
    a completed proposal invocation, re-debiting tool capacity, or re-executing
    a persisted Step.
    """

    def __init__(
        self,
        *,
        store: ObjectStore,
        evaluation_service: EvaluationService | None = None,
        tool_executor: ToolExecutor | None = None,
        tool_store: ToolCallStore | None = None,
    ) -> None:
        self._store = store
        self._evaluation_service = evaluation_service
        self._tool_executor = tool_executor
        self._tool_store = tool_store or ToolCallStore(store)

    @property
    def tool_store(self) -> ToolCallStore:
        return self._tool_store

    @staticmethod
    def _result_binding_key(run_id: str, step_index: int) -> str:
        """Opaque durable key binding a Step identity to its Step Result."""
        return f"whetstone.optimization_step_result:{run_id}#{step_index}"

    @staticmethod
    def _checkpoint_binding_key(run_id: str, step_index: int) -> str:
        """Opaque durable key binding a Step identity to its proposal ckpt."""
        return f"whetstone.optimization_step_checkpoint:{run_id}#{step_index}"

    def _resolve_result_binding(
        self, run_id: str, step_index: int
    ) -> TypedRef | None:
        reference = self._store.resolve(
            self._result_binding_key(run_id, step_index)
        )
        if reference is None:
            return None
        return TypedRef(
            schema_name=reference.schema, content_hash=reference.content_hash
        )

    def _resolve_checkpoint_binding(
        self, run_id: str, step_index: int
    ) -> TypedRef | None:
        reference = self._store.resolve(
            self._checkpoint_binding_key(run_id, step_index)
        )
        if reference is None:
            return None
        return TypedRef(
            schema_name=reference.schema, content_hash=reference.content_hash
        )

    # -- persistence helpers -------------------------------------------------

    def _put_request(self, request: OptimizationStepRequest) -> TypedRef:
        ref, _status = self._store.put(
            STEP_REQUEST_SCHEMA, request.record_content()
        )
        return TypedRef(schema_name=ref.schema, content_hash=ref.content_hash)

    def _put_result(self, result: OptimizationStepResult) -> TypedRef:
        ref, _status = self._store.put(
            STEP_RESULT_SCHEMA, result.record_content()
        )
        return TypedRef(schema_name=ref.schema, content_hash=ref.content_hash)

    def resolve_step_result(
        self, run_id: str, step_index: int
    ) -> TypedRef | None:
        """Return the persisted Step Result reference for a Step, or None.

        The restart position is read exclusively from dr-store's durable
        binding table — never reconstructed from process memory — so a fresh
        harness instance resolves the identical reference.
        """
        return self._resolve_result_binding(run_id, step_index)

    # -- the durable Step -----------------------------------------------------

    def run_step(
        self,
        request: OptimizationStepRequest,
        adapter: OptimizerAdapter,
    ) -> tuple[OptimizationStepResult, TypedRef]:
        """Run exactly one durable Optimization Step.

        Idempotent under replay: a persisted Step Result for this exact request
        replays without a second Result; a divergent Result conflicts. Returns
        the Step Result and its typed reference.
        """
        # 1. Validate + persist the immutable request (content-addressed).
        request_ref = self._put_request(request)
        expected_request_ref = step_request_reference(request)
        if request_ref != expected_request_ref:
            raise ValueError(
                "persisted Step Request reference does not match the "
                "record's content-addressed reference"
            )
        if adapter.mode is not request.mode:
            raise ValueError(
                f"adapter mode {adapter.mode.value!r} does not match request "
                f"mode {request.mode.value!r}"
            )

        # 2. Restart/idempotency FIRST: resolve any existing Step Result from
        #    dr-store's durable binding and reuse it before invoking adapter
        #    code or external effects.
        existing_ref = self._resolve_result_binding(
            request.run_id, request.step_index
        )
        if existing_ref is not None:
            existing_result = self._load_result(existing_ref)
            if existing_result.request_ref == request_ref:
                # Same request + same persisted Result: idempotent replay.
                return existing_result, existing_ref
            raise StepResultConflictError(
                run_id=request.run_id,
                step_index=request.step_index,
                existing=existing_ref,
                requested=request_ref,
            )

        # 3. Dispatch by declared Step mode.
        if request.mode is StepMode.PURE:
            output = self._run_pure(request, adapter)
            resolved_intents: tuple[IntentResolution, ...] = ()
            tool_evidence: tuple[ToolEvidence, ...] = ()
        elif request.mode is StepMode.PROPOSAL_ONLY:
            output = self._run_proposal(request, adapter)
            resolved_intents = self._resolve_intents(output)
            tool_evidence = ()
        elif request.mode is StepMode.TOOL_USING:
            output, tool_evidence = self._run_tool_using(request, adapter)
            resolved_intents = ()
        else:  # pragma: no cover - StepMode is closed
            raise ValueError(f"unknown Step mode {request.mode!r}")

        # 4. Finalize exactly once: compute state/history snapshot refs, carry
        #    budgets forward, assign status, persist ONE Result, then bind.
        state_ref = (
            typed_ref_for_record(
                f"{ADAPTER_CHECKPOINT_SCHEMA}.state", output.state_delta
            )
            if output.state_delta
            else None
        )
        history_ref = (
            typed_ref_for_record(
                f"{ADAPTER_CHECKPOINT_SCHEMA}.history", output.history_delta
            )
            if output.history_delta
            else None
        )

        result = OptimizationStepResult(
            run_id=request.run_id,
            step_id=request.step_id,
            step_index=request.step_index,
            request_ref=request_ref,
            proposed_candidates=output.proposed_candidates,
            accepted_candidates=output.accepted_candidates,
            resolved_intents=resolved_intents,
            tool_evidence=tool_evidence,
            state_ref=state_ref,
            history_ref=history_ref,
            budget=request.budget,
            status=output.proposed_status,
        )
        result_ref = self._put_result(result)

        # Bind the Step Result under the Step identity in dr-store's atomic
        # binding table (absent->bind; same->idempotent; divergent->conflict).
        # A back-edge is forbidden until this durable binding exists.
        try:
            status = self._store.bind(
                self._result_binding_key(
                    request.run_id, request.step_index
                ),
                result_ref.reference,
            )
        except BindingConflictError as conflict:
            existing = TypedRef(
                schema_name=conflict.existing.schema,
                content_hash=conflict.existing.content_hash,
            )
            raise StepResultConflictError(
                run_id=request.run_id,
                step_index=request.step_index,
                existing=existing,
                requested=result_ref,
            ) from conflict
        if status is BindStatus.IDEMPOTENT:
            return self._load_result(result_ref), result_ref
        return result, result_ref

    # -- mode handlers --------------------------------------------------------

    def _run_pure(
        self,
        request: OptimizationStepRequest,
        adapter: OptimizerAdapter,
    ) -> AdapterOutput:
        output = adapter.invoke(request, ())
        if output.evaluation_intents or output.tool_call_records:
            raise ValueError(
                "a pure Step emits no Evaluation Intents and no Tool Calls"
            )
        return output

    def _run_proposal(
        self,
        request: OptimizationStepRequest,
        adapter: OptimizerAdapter,
    ) -> AdapterOutput:
        # Crash-resume: if a durable proposal checkpoint exists for this Step,
        # reuse it and never rerun a completed proposal invocation. The
        # checkpoint binding is resolved from dr-store, so a fresh harness
        # instance reuses it exactly as an in-process replay would.
        existing_ckpt = self._resolve_checkpoint_binding(
            request.run_id, request.step_index
        )
        if existing_ckpt is not None:
            return self._load_checkpoint(existing_ckpt)

        output = adapter.invoke(request, ())
        if output.tool_call_records:
            raise ValueError(
                "a proposal-only Step issues no Tool Calls"
            )
        # Durably checkpoint the typed adapter output BEFORE resolving intents,
        # so a crash during resolution reuses the checkpoint. Persist the body
        # (content-addressed) and atomically bind it under the Step identity.
        ref, _status = self._store.put(
            ADAPTER_CHECKPOINT_SCHEMA, output.record_content()
        )
        self._store.bind(
            self._checkpoint_binding_key(request.run_id, request.step_index),
            ref,
        )
        return output

    def _resolve_intents(
        self, output: AdapterOutput
    ) -> tuple[IntentResolution, ...]:
        if not output.evaluation_intents:
            return ()
        if self._evaluation_service is None:
            raise ValueError(
                "a proposal-only Step with Evaluation Intents requires an "
                "EvaluationService to resolve them outside the invocation"
            )
        resolutions: list[IntentResolution] = []
        for intent in output.evaluation_intents:
            resolution = self._evaluation_service.resolve_evaluation_intent(
                intent
            )
            # The resolution MUST be for this exact Intent and target Eval
            # Config; IntentResolution's own validator enforces the latter.
            if resolution.intent != intent:
                raise ValueError(
                    "EvaluationService resolved a different Intent than the "
                    "one requested"
                )
            resolutions.append(resolution)
        return tuple(resolutions)

    def _run_tool_using(
        self,
        request: OptimizationStepRequest,
        adapter: OptimizerAdapter,
    ) -> tuple[AdapterOutput, tuple[ToolEvidence, ...]]:
        if self._tool_executor is None:
            raise ValueError(
                "a tool-using Step requires a ToolExecutor to construct "
                "Runtime Tool Handles at the execution boundary"
            )
        # Construct Runtime Tool Handles ONLY here, at the execution boundary.
        handles = tuple(
            self._tool_executor.runtime_handle(cfg, self._tool_store)
            for cfg in request.tool_configs
        )
        output = adapter.invoke(request, handles)
        if output.evaluation_intents:
            raise ValueError(
                "a tool-using Step carries measurement in Tool Calls, not "
                "Evaluation Intents"
            )
        # Record every Tool Result + terminal Tool Call Store Entry used. A
        # refused Tool Result is referenced as evidence against its refused
        # Store Entry; an accepted one is transitioned to completed.
        evidence: list[ToolEvidence] = []
        for record in output.tool_call_records:
            self._store_tool_result(record.result)
            result_ref = tool_result_reference(record.result)
            if record.result.refusal is not None:
                entry = self._tool_store.get(
                    record.result.tool_config_hash, record.result.call_id
                )
                if entry is None or entry.refusal is None:
                    raise ValueError(
                        "a refused Tool Result must correspond to a refused "
                        "Tool Call Store Entry"
                    )
            else:
                entry = self._tool_store.complete(
                    record.result.tool_config_hash, record.result
                )
            evidence.append(
                ToolEvidence(tool_result_ref=result_ref, store_entry=entry)
            )
        return output, tuple(evidence)

    def _store_tool_result(self, result: ToolResult) -> None:
        self._store.put(
            tool_result_reference(result).schema_name, result.record_content()
        )

    # -- loads ----------------------------------------------------------------

    def _load_result(self, ref: TypedRef) -> OptimizationStepResult:
        content = self._store.get(ref.reference)
        return OptimizationStepResult.model_validate(content)

    def _load_checkpoint(self, ref: TypedRef) -> AdapterOutput:
        content = self._store.get(ref.reference)
        return AdapterOutput.model_validate(content)

    # -- terminalization ------------------------------------------------------

    def terminalize(
        self,
        *,
        run_id: str,
        step_result_refs: tuple[TypedRef, ...],
        cost: dict[str, object] | None = None,
    ) -> OptimizationResult:
        """Assemble the terminal Optimization Result from persisted Results.

        A back-edge/terminalization is only valid after the prior Step Result
        is persisted; here every referenced Step Result is loaded from the
        store, its final proposals are drawn from the last (complete) Result's
        accepted candidates, and the ordered Step Result references are
        preserved. Makes no official-evaluation claim.
        """
        if not step_result_refs:
            raise ValueError(
                "terminalize requires at least one persisted Step Result"
            )
        results = [self._load_result(ref) for ref in step_result_refs]
        last = results[-1]

        if last.status is StepStatus.FAILED:
            # A failed terminal Step yields a failed run that blocks official
            # materialization; no proposals are claimed.
            return OptimizationResult(
                run_id=run_id,
                proposals=(),
                step_result_refs=step_result_refs,
                status=StepStatus.FAILED,
                cost=cost or {},
            )

        proposals = tuple(
            OptimizationProposal(
                candidate_id=candidate.candidate_id,
                base_ref=candidate.base_ref,
                payload=candidate.payload,
            )
            for candidate in last.accepted_candidates
        )
        return OptimizationResult(
            run_id=run_id,
            proposals=proposals,
            step_result_refs=step_result_refs,
            status=StepStatus.COMPLETE,
            cost=cost or {},
        )

    @staticmethod
    def carry_budget_forward(
        prior: OptimizationStepResult,
    ) -> BudgetState:
        """The next Step Request's budget is the prior Result's budget.

        Budget state advances only through immutable Step Results; it is never
        recomputed from process memory.
        """
        return prior.budget
