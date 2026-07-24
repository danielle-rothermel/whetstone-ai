"""Algorithm-neutral durable optimization harness."""

from __future__ import annotations

from typing import Any, Protocol

from dr_store import BindingConflictError, BindStatus, ObjectStore

from whetstone.optimization.adapters import (
    AdapterCheckpoint,
    AdapterOutput,
    AdapterRegistry,
    OptimizerAdapter,
)
from whetstone.optimization.identity import TypedRef
from whetstone.optimization.schema import (
    CANDIDATE_RECORD_SCHEMA,
    EVAL_CONFIG_RECORD_SCHEMA,
    OPTIMIZATION_RESULT_SCHEMA,
    STEP_REQUEST_SCHEMA,
    STEP_RESULT_SCHEMA,
    BudgetState,
    Candidate,
    CandidateRef,
    EvaluationIntent,
    IntentResolution,
    OptimizationProposal,
    OptimizationResult,
    OptimizationStepRequest,
    OptimizationStepResult,
    StepMode,
    StepStatus,
    ToolEvidence,
    candidate_reference,
    optimization_result_reference,
    step_request_reference,
)
from whetstone.optimization.tool_store import ToolCallState, ToolCallStore
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
    "OptimizationResultConflictError",
    "StepResultConflictError",
    "ToolExecutor",
]

ADAPTER_CHECKPOINT_SCHEMA = "whetstone.optimization_adapter_checkpoint"
STATE_SNAPSHOT_SCHEMA = "whetstone.optimization_state_snapshot"
HISTORY_SNAPSHOT_SCHEMA = "whetstone.optimization_history_snapshot"


class StepResultConflictError(Exception):
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
            f"Step ({run_id}, index {step_index}) already has result "
            f"{existing.content_hash}; refusing {requested.content_hash}"
        )


class OptimizationResultConflictError(Exception):
    def __init__(
        self, *, run_id: str, existing: TypedRef, requested: TypedRef
    ) -> None:
        self.run_id = run_id
        self.existing = existing
        self.requested = requested
        super().__init__(
            f"Optimization run {run_id!r} already has terminal result "
            f"{existing.content_hash}; refusing {requested.content_hash}"
        )


class EvaluationService(Protocol):
    def resolve_evaluation_intent(
        self, intent: EvaluationIntent
    ) -> IntentResolution: ...


class ToolExecutor(Protocol):
    def runtime_handle(
        self, config: ToolConfig, store: ToolCallStore
    ) -> RuntimeToolHandle: ...


class OptimizationHarness:
    """Durable coordinator with all algorithm behavior behind a registry."""

    def __init__(
        self,
        *,
        store: ObjectStore,
        adapter_registry: AdapterRegistry,
        evaluation_service: EvaluationService | None = None,
        tool_executor: ToolExecutor | None = None,
        tool_store: ToolCallStore | None = None,
    ) -> None:
        self._store = store
        self._adapter_registry = adapter_registry
        self._evaluation_service = evaluation_service
        self._tool_executor = tool_executor
        self._tool_store = tool_store or ToolCallStore(store)

    @property
    def tool_store(self) -> ToolCallStore:
        return self._tool_store

    @staticmethod
    def _result_binding_key(run_id: str, step_index: int) -> str:
        return f"whetstone.optimization_step_result:{run_id}#{step_index}"

    @staticmethod
    def _checkpoint_binding_key(run_id: str, step_index: int) -> str:
        return f"whetstone.optimization_step_checkpoint:{run_id}#{step_index}"

    @staticmethod
    def _terminal_binding_key(run_id: str) -> str:
        return f"whetstone.optimization_result:{run_id}"

    def _resolve_binding(self, key: str) -> TypedRef | None:
        reference = self._store.resolve(key)
        if reference is None:
            return None
        return TypedRef(
            schema_name=reference.schema, content_hash=reference.content_hash
        )

    def _resolve_result_binding(
        self, run_id: str, step_index: int
    ) -> TypedRef | None:
        return self._resolve_binding(
            self._result_binding_key(run_id, step_index)
        )

    def _put(self, schema: str, content: dict[str, Any]) -> TypedRef:
        reference, _status = self._store.put(schema, content)
        return TypedRef(
            schema_name=reference.schema,
            content_hash=reference.content_hash,
        )

    def _put_request(self, request: OptimizationStepRequest) -> TypedRef:
        return self._put(STEP_REQUEST_SCHEMA, request.record_content())

    def _put_result(self, result: OptimizationStepResult) -> TypedRef:
        return self._put(STEP_RESULT_SCHEMA, result.record_content())

    def _persist_candidate(self, candidate: Candidate) -> CandidateRef:
        expected = candidate_reference(candidate)
        persisted = self._put(
            CANDIDATE_RECORD_SCHEMA, candidate.record_content()
        )
        if persisted != expected.record_ref:
            raise ValueError(
                "persisted Candidate ref failed content validation"
            )
        return expected

    def _persist_intent_records(self, intent: EvaluationIntent) -> None:
        candidate = self._persist_candidate(intent.candidate.record)
        if candidate != intent.candidate:
            raise ValueError("Intent candidate ref is not its exact record")
        persisted_eval = self._put(
            EVAL_CONFIG_RECORD_SCHEMA,
            intent.target_eval_config.record.model_dump(mode="json"),
        )
        if persisted_eval != intent.target_eval_config.record_ref:
            raise ValueError("Intent Eval Config ref is not its exact record")

    def _snapshot(self, schema: str, delta: dict[str, Any]) -> TypedRef | None:
        return self._put(schema, delta) if delta else None

    def _load_result(self, ref: TypedRef) -> OptimizationStepResult:
        return OptimizationStepResult.model_validate(
            self._store.get(ref.reference)
        )

    def _load_checkpoint(self, ref: TypedRef) -> AdapterCheckpoint:
        return AdapterCheckpoint.model_validate(self._store.get(ref.reference))

    def _load_terminal(self, ref: TypedRef) -> OptimizationResult:
        return OptimizationResult.model_validate(
            self._store.get(ref.reference)
        )

    def resolve_step_result(
        self, run_id: str, step_index: int
    ) -> TypedRef | None:
        return self._resolve_result_binding(run_id, step_index)

    def resolve_optimization_result(self, run_id: str) -> TypedRef | None:
        return self._resolve_binding(self._terminal_binding_key(run_id))

    def resolve_adapter(self, adapter_key: str) -> OptimizerAdapter:
        """Resolve the exact configured adapter for controller validation."""
        return self._adapter_registry.resolve(adapter_key)

    def _validate_prior_binding(
        self, request: OptimizationStepRequest
    ) -> None:
        if request.step_index == 0:
            return
        actual = self._resolve_result_binding(
            request.run_id, request.step_index - 1
        )
        if actual is None:
            raise ValueError(
                "noninitial Step Request references no durably bound "
                "preceding Step Result"
            )
        if actual != request.prior_step_result_ref:
            raise ValueError(
                "prior_step_result_ref does not match the actual preceding "
                "Step Result binding"
            )
        preceding = self._load_result(actual)
        if preceding.run_id != request.run_id:
            raise ValueError("preceding Step Result belongs to another run")
        if preceding.step_index != request.step_index - 1:
            raise ValueError("preceding Step Result has the wrong step index")
        if preceding.status is not StepStatus.CONTINUE:
            raise ValueError(
                "a new Step may follow only a continuing Step Result"
            )
        if request.budget != preceding.budget:
            raise ValueError(
                "a new Step must carry forward the preceding durable budget"
            )

    def run_step(
        self, request: OptimizationStepRequest
    ) -> tuple[OptimizationStepResult, TypedRef]:
        request_ref = self._put_request(request)
        if request_ref != step_request_reference(request):
            raise ValueError("persisted request ref failed content validation")
        self._validate_prior_binding(request)
        for candidate in request.candidates:
            self._persist_candidate(candidate)

        existing_ref = self._resolve_result_binding(
            request.run_id, request.step_index
        )
        if existing_ref is not None:
            existing = self._load_result(existing_ref)
            if existing.request_ref == request_ref:
                return existing, existing_ref
            raise StepResultConflictError(
                run_id=request.run_id,
                step_index=request.step_index,
                existing=existing_ref,
                requested=request_ref,
            )

        adapter = self._adapter_registry.resolve(request.adapter_key)
        if adapter.key != request.adapter_key:
            raise ValueError(
                "registry returned an adapter under the wrong key"
            )
        if adapter.mode is not request.mode:
            raise ValueError(
                f"adapter mode {adapter.mode.value!r} does not match request "
                f"mode {request.mode.value!r}"
            )

        if request.mode is StepMode.PURE:
            output = self._invoke_pure(request, adapter)
        else:
            output = self._effectful_output(request, request_ref, adapter)

        self._validate_output(request, output)
        budget = request.budget.debit(output.budget_delta)
        proposed_refs = tuple(
            self._persist_candidate(candidate)
            for candidate in output.proposed_candidates
        )
        accepted_refs = tuple(
            self._persist_candidate(candidate)
            for candidate in output.accepted_candidates
        )

        if request.mode is StepMode.PROPOSAL_ONLY:
            resolutions = self._resolve_intents(
                request, output, proposed_refs, accepted_refs
            )
            tool_evidence: tuple[ToolEvidence, ...] = ()
        elif request.mode is StepMode.TOOL_USING:
            resolutions = ()
            tool_evidence = self._finalize_tool_records(output)
        else:
            resolutions = ()
            tool_evidence = ()

        result = OptimizationStepResult(
            run_id=request.run_id,
            step_id=request.step_id,
            step_index=request.step_index,
            request_ref=request_ref,
            proposed_candidates=proposed_refs,
            accepted_candidates=accepted_refs,
            resolved_intents=resolutions,
            tool_evidence=tool_evidence,
            state_ref=self._snapshot(
                STATE_SNAPSHOT_SCHEMA, output.state_delta
            ),
            history_ref=self._snapshot(
                HISTORY_SNAPSHOT_SCHEMA, output.history_delta
            ),
            budget_delta=output.budget_delta,
            budget=budget,
            status=output.proposed_status,
        )
        result_ref = self._put_result(result)
        key = self._result_binding_key(request.run_id, request.step_index)
        try:
            status = self._store.bind(key, result_ref.reference)
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

    def _invoke_pure(
        self,
        request: OptimizationStepRequest,
        adapter: OptimizerAdapter,
    ) -> AdapterOutput:
        output = adapter.invoke(request, ())
        if output.evaluation_intents or output.tool_call_records:
            raise ValueError("a pure Step emits no measurement requests")
        return output

    def _effectful_output(
        self,
        request: OptimizationStepRequest,
        request_ref: TypedRef,
        adapter: OptimizerAdapter,
    ) -> AdapterOutput:
        key = self._checkpoint_binding_key(request.run_id, request.step_index)
        existing_ref = self._resolve_binding(key)
        if existing_ref is not None:
            checkpoint = self._load_checkpoint(existing_ref)
            self._validate_checkpoint(
                checkpoint, request_ref, request.adapter_key
            )
            return checkpoint.output

        if request.mode is StepMode.PROPOSAL_ONLY:
            output = adapter.invoke(request, ())
            if output.tool_call_records:
                raise ValueError("proposal-only Steps issue no Tool Calls")
        elif request.mode is StepMode.TOOL_USING:
            if self._tool_executor is None:
                raise ValueError("tool-using Step requires a ToolExecutor")
            handles = tuple(
                self._tool_executor.runtime_handle(cfg, self._tool_store)
                for cfg in request.tool_configs
            )
            output = adapter.invoke(request, handles)
            if output.evaluation_intents:
                raise ValueError(
                    "tool-using Steps carry measurement in Tool Results"
                )
        else:  # pragma: no cover - closed enum
            raise ValueError(f"unsupported effectful mode {request.mode!r}")

        checkpoint = AdapterCheckpoint(
            request_ref=request_ref,
            adapter_key=request.adapter_key,
            output=output,
        )
        checkpoint_ref = self._put(
            ADAPTER_CHECKPOINT_SCHEMA, checkpoint.record_content()
        )
        try:
            self._store.bind(key, checkpoint_ref.reference)
        except BindingConflictError:
            winner_ref = self._resolve_binding(key)
            assert winner_ref is not None
            winner = self._load_checkpoint(winner_ref)
            self._validate_checkpoint(winner, request_ref, request.adapter_key)
            return winner.output
        return output

    @staticmethod
    def _validate_checkpoint(
        checkpoint: AdapterCheckpoint,
        request_ref: TypedRef,
        adapter_key: str,
    ) -> None:
        if checkpoint.request_ref != request_ref:
            raise ValueError(
                "durable adapter checkpoint belongs to another request"
            )
        if checkpoint.adapter_key != adapter_key:
            raise ValueError(
                "durable adapter checkpoint belongs to another adapter"
            )

    @staticmethod
    def _validate_output(
        request: OptimizationStepRequest, output: AdapterOutput
    ) -> None:
        contract = request.output_contract
        expected_count = (
            0
            if output.proposed_status is StepStatus.FAILED
            else contract.returned_proposal_count
        )
        if len(output.accepted_candidates) != expected_count:
            raise ValueError(
                "adapter violated returned proposal cardinality: expected "
                f"{expected_count}, got "
                f"{len(output.accepted_candidates)}"
            )
        if contract.require_distinct_bases:
            bases = [
                candidate.base_ref for candidate in output.accepted_candidates
            ]
            if len(bases) != len(set(bases)):
                raise ValueError(
                    "adapter violated the distinct-base output contract"
                )
        accepted = {
            candidate_reference(candidate).identity_hash
            for candidate in output.accepted_candidates
        }
        proposed = {
            candidate_reference(candidate).identity_hash
            for candidate in output.proposed_candidates
        }
        if not accepted <= proposed:
            raise ValueError("accepted candidates must have been proposed")

    def _resolve_intents(
        self,
        request: OptimizationStepRequest,
        output: AdapterOutput,
        proposed: tuple[CandidateRef, ...],
        accepted: tuple[CandidateRef, ...],
    ) -> tuple[IntentResolution, ...]:
        if not output.evaluation_intents:
            return ()
        if self._evaluation_service is None:
            raise ValueError(
                "proposal-only Step with Intents requires EvaluationService"
            )
        allowed = {
            candidate_reference(candidate).identity_hash
            for candidate in request.candidates
        }
        allowed.update(
            candidate.identity_hash for candidate in (*proposed, *accepted)
        )
        resolutions: list[IntentResolution] = []
        for intent in output.evaluation_intents:
            if intent.run_id != request.run_id:
                raise ValueError("Intent belongs to another optimization run")
            if intent.step_index != request.step_index:
                raise ValueError("Intent belongs to another optimization step")
            if intent.candidate.identity_hash not in allowed:
                raise ValueError(
                    "Intent candidate is not an exact Step output candidate"
                )
            self._persist_intent_records(intent)
            raw_resolution = (
                self._evaluation_service.resolve_evaluation_intent(intent)
            )
            resolution = IntentResolution.model_validate(
                raw_resolution.model_dump(mode="json")
            )
            if resolution.intent != intent:
                raise ValueError("EvaluationService resolved another Intent")
            resolutions.append(resolution)
        return tuple(resolutions)

    def _finalize_tool_records(
        self, output: AdapterOutput
    ) -> tuple[ToolEvidence, ...]:
        evidence: list[ToolEvidence] = []
        for record in output.tool_call_records:
            result_ref = self._store_tool_result(record.result)
            entry = self._tool_store.get(
                record.result.tool_config_hash, record.result.call_id
            )
            if entry is None:
                raise ValueError("Tool Result has no Tool Call Store entry")
            if record.result.refusal is not None:
                if (
                    entry.state is not ToolCallState.REFUSED
                    or entry.refusal != record.result.refusal
                ):
                    raise ValueError(
                        "refused Tool Result has no matching durable refusal"
                    )
            else:
                entry = self._tool_store.complete(
                    record.result.tool_config_hash, record.result
                )
            evidence.append(
                ToolEvidence(
                    tool_result_ref=result_ref,
                    store_entry=entry,
                )
            )
        return tuple(evidence)

    def _store_tool_result(self, result: ToolResult) -> TypedRef:
        expected = tool_result_reference(result)
        persisted = self._put(expected.schema_name, result.record_content())
        if persisted != expected:
            raise ValueError("persisted Tool Result ref failed validation")
        return persisted

    def terminalize(
        self,
        *,
        run_id: str,
        step_result_refs: tuple[TypedRef, ...],
        cost: dict[str, object] | None = None,
    ) -> tuple[OptimizationResult, TypedRef]:
        """Persist and bind the one terminal Optimization Result."""
        if not step_result_refs:
            raise ValueError("terminalize requires at least one Step Result")

        existing_ref = self.resolve_optimization_result(run_id)
        if existing_ref is not None:
            existing = self._load_terminal(existing_ref)
            if (
                existing.step_result_refs == step_result_refs
                and existing.cost == (cost or {})
            ):
                return existing, existing_ref
            requested = optimization_result_reference(
                self._assemble_terminal(
                    run_id=run_id,
                    step_result_refs=step_result_refs,
                    cost=cost or {},
                )
            )
            raise OptimizationResultConflictError(
                run_id=run_id,
                existing=existing_ref,
                requested=requested,
            )

        result = self._assemble_terminal(
            run_id=run_id,
            step_result_refs=step_result_refs,
            cost=cost or {},
        )
        result_ref = self._put(
            OPTIMIZATION_RESULT_SCHEMA, result.record_content()
        )
        if result_ref != optimization_result_reference(result):
            raise ValueError(
                "persisted Optimization Result ref failed validation"
            )
        try:
            status = self._store.bind(
                self._terminal_binding_key(run_id), result_ref.reference
            )
        except BindingConflictError as conflict:
            existing = TypedRef(
                schema_name=conflict.existing.schema,
                content_hash=conflict.existing.content_hash,
            )
            raise OptimizationResultConflictError(
                run_id=run_id,
                existing=existing,
                requested=result_ref,
            ) from conflict
        if status is BindStatus.IDEMPOTENT:
            return self._load_terminal(result_ref), result_ref
        return result, result_ref

    def _assemble_terminal(
        self,
        *,
        run_id: str,
        step_result_refs: tuple[TypedRef, ...],
        cost: dict[str, object],
    ) -> OptimizationResult:
        results: list[OptimizationStepResult] = []
        for index, ref in enumerate(step_result_refs):
            actual = self._resolve_result_binding(run_id, index)
            if actual != ref:
                raise ValueError(
                    "terminal Step Result refs must match ordered bindings"
                )
            result = self._load_result(ref)
            if result.run_id != run_id or result.step_index != index:
                raise ValueError(
                    "terminal Step Result belongs to another run or position"
                )
            results.append(result)
        last = results[-1]
        if last.status is StepStatus.CONTINUE:
            raise ValueError("cannot terminalize a continuing Step Result")
        proposals = (
            ()
            if last.status is StepStatus.FAILED
            else tuple(
                OptimizationProposal(candidate=candidate)
                for candidate in last.accepted_candidates
            )
        )
        return OptimizationResult(
            run_id=run_id,
            proposals=proposals,
            step_result_refs=step_result_refs,
            status=last.status,
            cost=cost,
        )

    @staticmethod
    def carry_budget_forward(
        prior: OptimizationStepResult,
    ) -> BudgetState:
        return prior.budget
