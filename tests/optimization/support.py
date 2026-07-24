"""Durable stores, builders, and generic adapter test doubles."""

from __future__ import annotations

from typing import Any

from dr_code.eval import DefinitionRef, EvalConfig
from dr_store import ObjectStore, SqliteBackend

from whetstone.graph.rollout import EvaluationRole
from whetstone.optimization import (
    AdapterOutput,
    BudgetDelta,
    BudgetState,
    Candidate,
    EvaluationIntent,
    IdentityOptimizerAdapter,
    IntentOutcome,
    IntentResolution,
    MappingAdapterRegistry,
    OptimizationStepRequest,
    OutputContract,
    ResolutionClass,
    ResolutionDetail,
    RuntimeToolHandle,
    StepKind,
    StepMode,
    StepStatus,
    ToolCall,
    ToolCallRecord,
    ToolCallStore,
    ToolCapacity,
    ToolConfig,
    ToolDefinition,
    ToolResult,
    candidate_reference,
    eval_config_reference,
    typed_ref_for_record,
)

FULL_A = "a" * 64
FULL_B = "b" * 64
FULL_C = "c" * 64
FULL_D = "d" * 64
EVIDENCE_SCHEMA = "whetstone.test.evaluation_evidence"


def make_store(tmp_path, name: str = "optimization.sqlite") -> ObjectStore:
    return ObjectStore(SqliteBackend(tmp_path / name))


def candidate(
    cid: str = "A", *, base: str = "base", text: str = "t"
) -> Candidate:
    return Candidate(
        candidate_id=cid,
        base_ref=base,
        payload={"user_prompt_template": text, "fixed": "same"},
    )


def eval_config(identity_hash: str = FULL_B) -> EvalConfig:
    return EvalConfig(
        definition_ref=DefinitionRef(
            definition_id="eval",
            version="1",
            schema_name="dr_code.eval_definition",
            identity_hash=FULL_A,
        ),
        sampling_config_hash=FULL_A,
        evaluation_procedure_config_hash=FULL_C,
        aggregation_config_hash=FULL_D,
        config_identity_hash=identity_hash,
    )


def output_contract(
    count: int = 1, *, distinct_bases: bool = False
) -> OutputContract:
    return OutputContract(
        returned_proposal_count=count,
        require_distinct_bases=distinct_bases,
    )


def pure_request(
    *,
    run_id: str = "run-pure",
    candidates: tuple[Candidate, ...] | None = None,
) -> OptimizationStepRequest:
    records = candidates if candidates is not None else (candidate(),)
    return OptimizationStepRequest(
        run_id=run_id,
        step_id=f"{run_id}-s0",
        optimizer_config_hash=FULL_A,
        adapter_key="identity",
        mode=StepMode.PURE,
        kind=StepKind.IDENTITY,
        step_index=0,
        candidates=records,
        output_contract=output_contract(len(records)),
        budget=BudgetState(remaining={"rollouts": 10}),
    )


def proposal_request(
    *,
    run_id: str = "run-proposal",
    step_index: int = 0,
    prior_step_result_ref=None,
    budget: BudgetState | None = None,
    contract: OutputContract | None = None,
) -> OptimizationStepRequest:
    return OptimizationStepRequest(
        run_id=run_id,
        step_id=f"{run_id}-s{step_index}",
        optimizer_config_hash=FULL_A,
        adapter_key="proposal-test",
        mode=StepMode.PROPOSAL_ONLY,
        kind=StepKind.PROPOSAL,
        step_index=step_index,
        prior_step_result_ref=prior_step_result_ref,
        candidates=(candidate(),),
        output_contract=contract or output_contract(),
        budget=budget or BudgetState(remaining={"rollouts": 10}),
    )


def make_intent(
    proposed: Candidate,
    *,
    run_id: str = "run-proposal",
    step_index: int = 0,
) -> EvaluationIntent:
    return EvaluationIntent(
        intent_id=f"{run_id}-{step_index}-{proposed.candidate_id}",
        candidate=candidate_reference(proposed),
        target_eval_config=eval_config_reference(eval_config()),
        context_role=EvaluationRole.INTERNAL,
        purpose="proposal",
        run_id=run_id,
        step_index=step_index,
    )


def make_tool_definition_config(
    *, capacity: int = 2, namespace: str = "ns-1"
) -> ToolConfig:
    definition = ToolDefinition(
        tool_name="evaluate_candidate",
        input_fields=("model_route", "template"),
        output_fields=("rollout_refs", "reward"),
    )
    return ToolConfig(
        tool_name="evaluate_candidate",
        tool_definition_ref="tooldef://evaluate_candidate",
        tool_definition_identity_hash=definition.identity_hash(),
        endpoint="tool://evaluate_candidate",
        eval_config_ref="evalcfg://internal",
        eval_config_identity_hash=FULL_B,
        reward_policy_ref=FULL_C,
        capacity=ToolCapacity(max_accepted_calls=capacity),
        store_namespace=namespace,
    )


def tool_request(
    *,
    run_id: str = "run-tool",
    config: ToolConfig | None = None,
) -> OptimizationStepRequest:
    cfg = config or make_tool_definition_config()
    return OptimizationStepRequest(
        run_id=run_id,
        step_id=f"{run_id}-s0",
        optimizer_config_hash=FULL_A,
        adapter_key="tool-test",
        mode=StepMode.TOOL_USING,
        kind=StepKind.TOOL,
        step_index=0,
        candidates=(candidate(),),
        output_contract=output_contract(),
        tool_configs=(cfg,),
        budget=BudgetState(remaining={"tool_calls": 10}),
    )


def registry(*adapters):
    values = [IdentityOptimizerAdapter(), *adapters]
    return MappingAdapterRegistry({adapter.key: adapter for adapter in values})


class RecordingEvaluationService:
    def __init__(
        self,
        store: ObjectStore,
        *,
        outcome: IntentOutcome = IntentOutcome.COMPLETED,
    ) -> None:
        self._store = store
        self._outcome = outcome
        self.resolved: list[EvaluationIntent] = []

    def resolve_evaluation_intent(
        self, intent: EvaluationIntent
    ) -> IntentResolution:
        self.resolved.append(intent)
        if self._outcome is IntentOutcome.REJECTED:
            return IntentResolution(
                intent=intent,
                outcome=self._outcome,
                detail=ResolutionDetail(
                    classification=ResolutionClass.VALIDATION,
                    message="candidate rejected before execution",
                ),
                resolved_eval_config=intent.target_eval_config,
            )
        evidence: dict[str, Any] = {
            "intent_id": intent.intent_id,
            "candidate_identity_hash": intent.candidate.identity_hash,
        }
        self._store.put(EVIDENCE_SCHEMA, evidence)
        classification = (
            ResolutionClass.MEASURED
            if self._outcome is IntentOutcome.COMPLETED
            else ResolutionClass.UNSCORABLE
        )
        return IntentResolution(
            intent=intent,
            outcome=self._outcome,
            detail=ResolutionDetail(
                classification=classification,
                message="evaluation completed"
                if self._outcome is IntentOutcome.COMPLETED
                else "candidate was unscorable",
            ),
            evaluation_evidence_refs=(
                typed_ref_for_record(EVIDENCE_SCHEMA, evidence),
            ),
            resolved_eval_config=intent.target_eval_config,
        )


class CountingProposalAdapter:
    def __init__(
        self,
        *,
        status: StepStatus = StepStatus.COMPLETE,
        candidates: tuple[Candidate, ...] | None = None,
        budget_delta: BudgetDelta | None = None,
    ) -> None:
        self.invocations = 0
        self._status = status
        self._candidates = candidates
        self._budget_delta = budget_delta or BudgetDelta(
            consumed={"rollouts": 1}
        )

    @property
    def key(self) -> str:
        return "proposal-test"

    @property
    def mode(self) -> StepMode:
        return StepMode.PROPOSAL_ONLY

    def invoke(
        self, request: OptimizationStepRequest, handles
    ) -> AdapterOutput:
        self.invocations += 1
        proposed = self._candidates or (
            candidate("P1", base="base", text="new"),
        )
        return AdapterOutput(
            proposed_candidates=proposed,
            accepted_candidates=proposed,
            evaluation_intents=tuple(
                make_intent(
                    record,
                    run_id=request.run_id,
                    step_index=request.step_index,
                )
                for record in proposed
            ),
            budget_delta=self._budget_delta,
            proposed_status=self._status,
        )


class RecordingToolExecutor:
    def __init__(self) -> None:
        self.handles_built = 0

    def runtime_handle(
        self, config: ToolConfig, store: ToolCallStore
    ) -> RuntimeToolHandle:
        self.handles_built += 1

        def execute(call: ToolCall) -> ToolResult:
            entry = store.accept_or_refuse(call, config)
            if entry.state.value == "refused":
                return ToolResult(
                    call_id=call.call_id,
                    tool_config_ref=config.tool_definition_ref,
                    tool_config_hash=config.identity_hash(),
                    store_namespace=config.store_namespace,
                    refusal=entry.refusal,
                )
            return ToolResult(
                call_id=call.call_id,
                tool_config_ref=config.tool_definition_ref,
                tool_config_hash=config.identity_hash(),
                store_namespace=config.store_namespace,
                output={
                    "rollout_refs": [],
                    "accepted_ordinal": entry.capacity_debit_ordinal,
                },
            )

        return RuntimeToolHandle(config, execute)


class ToolUsingAdapter:
    def __init__(self, *, call_ids: tuple[str, ...] = ("c1",)) -> None:
        self.call_ids = call_ids
        self.invocations = 0

    @property
    def key(self) -> str:
        return "tool-test"

    @property
    def mode(self) -> StepMode:
        return StepMode.TOOL_USING

    def invoke(
        self,
        request: OptimizationStepRequest,
        handles: tuple[RuntimeToolHandle, ...],
    ) -> AdapterOutput:
        self.invocations += 1
        handle = handles[0]
        records: list[ToolCallRecord] = []
        for call_id in self.call_ids:
            call = ToolCall(
                call_id=call_id,
                tool_config_hash=handle.tool_config_hash,
                store_namespace=handle.config.store_namespace,
                args={"model_route": "r0", "template": call_id},
            )
            records.append(ToolCallRecord(call=call, result=handle(call)))
        proposed = candidate("TP", text="tool")
        return AdapterOutput(
            proposed_candidates=(proposed,),
            accepted_candidates=(proposed,),
            tool_call_records=tuple(records),
            budget_delta=BudgetDelta(
                consumed={"tool_calls": len(self.call_ids)}
            ),
            proposed_status=StepStatus.COMPLETE,
        )
