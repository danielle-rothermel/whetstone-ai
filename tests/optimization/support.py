"""Shared builders and test doubles for the optimizer harness tests.

These build real, released-contract objects (dr-store ObjectStore over a
MemoryBackend, real pydantic schemas, real identity hashes) and two small,
deterministic test doubles:

* :class:`RecordingEvaluationService` — Whetstone's external evaluation path
  for Evaluation Intents. It resolves each Intent under its *exact* target Eval
  Config, storing a stand-in evaluation-evidence record and returning a typed
  :class:`IntentResolution`. It records every resolved Intent so a restart test
  can prove a completed proposal invocation is not rerun.

* :class:`RecordingToolExecutor` — constructs a Runtime Tool Handle only at the
  execution boundary, binding the Tool Config to a callable that accepts the
  call through the authoritative Tool Call Store and returns a Tool Result.
"""

from __future__ import annotations

from typing import Any

from dr_store import MemoryBackend, ObjectStore

from whetstone.graph.rollout import EvaluationRole
from whetstone.optimization import (
    AdapterOutput,
    Candidate,
    EvaluationIntent,
    IntentResolution,
    OptimizationStepRequest,
    OutputContract,
    RuntimeToolHandle,
    StepKind,
    StepMode,
    StepStatus,
    ToolCall,
    ToolCallStore,
    ToolCapacity,
    ToolConfig,
    ToolResult,
    typed_ref_for_record,
)
from whetstone.optimization.adapters import OptimizerAdapter, ToolCallRecord

FULL_A = "a" * 64
FULL_B = "b" * 64
FULL_C = "c" * 64
FULL_D = "d" * 64
EVIDENCE_SCHEMA = "whetstone.test.evaluation_evidence"


def make_store() -> ObjectStore:
    return ObjectStore(MemoryBackend())


def candidate(
    cid: str = "A", *, base: str = "base", text: str = "t"
) -> Candidate:
    return Candidate(
        candidate_id=cid, base_ref=base, payload={"template": text}
    )


def output_contract(n: int = 1) -> OutputContract:
    return OutputContract(returned_proposal_count=n)


def pure_request(
    *,
    run_id: str = "run-pure",
    step_index: int = 0,
    candidates: tuple[Candidate, ...] | None = None,
    prior_step_result_ref=None,
) -> OptimizationStepRequest:
    cands = candidates if candidates is not None else (candidate("A"),)
    return OptimizationStepRequest(
        run_id=run_id,
        step_id=f"{run_id}-s{step_index}",
        optimizer_config_hash=FULL_A,
        mode=StepMode.PURE,
        kind=StepKind.IDENTITY,
        step_index=step_index,
        candidates=cands,
        output_contract=output_contract(len(cands)),
        prior_step_result_ref=prior_step_result_ref,
    )


def proposal_request(
    *,
    run_id: str = "run-copro",
    step_index: int = 0,
    prior_step_result_ref=None,
    candidates: tuple[Candidate, ...] | None = None,
) -> OptimizationStepRequest:
    cands = candidates if candidates is not None else (candidate("A"),)
    return OptimizationStepRequest(
        run_id=run_id,
        step_id=f"{run_id}-s{step_index}",
        optimizer_config_hash=FULL_A,
        mode=StepMode.PROPOSAL_ONLY,
        kind=StepKind.PROPOSAL,
        step_index=step_index,
        candidates=cands,
        output_contract=output_contract(1),
        prior_step_result_ref=prior_step_result_ref,
    )


def make_intent(
    *,
    run_id: str = "run-copro",
    step_index: int = 0,
    candidate_id: str = "P1",
    target_hash: str = FULL_B,
    purpose: str = "seed_proposal",
) -> EvaluationIntent:
    return EvaluationIntent(
        intent_id=f"{run_id}-{step_index}-{candidate_id}",
        candidate_id=candidate_id,
        target_eval_config_ref="evalcfg://internal",
        target_eval_config_hash=target_hash,
        context_role=EvaluationRole.INTERNAL,
        purpose=purpose,
        run_id=run_id,
        step_index=step_index,
    )


def make_tool_definition_config(
    *,
    capacity: int = 2,
    namespace: str = "ns-1",
) -> ToolConfig:
    from whetstone.optimization import ToolDefinition

    definition = ToolDefinition(
        tool_name="evaluate_candidate",
        input_fields=("model_route", "template"),
        output_fields=("rollout_refs", "reward"),
    )
    return ToolConfig(
        tool_name="evaluate_candidate",
        tool_definition_ref="tooldef://evaluate_candidate",
        tool_definition_identity_hash=definition.identity_hash(),
        endpoint="mcp://bridge/evaluate_candidate",
        eval_config_ref="evalcfg://internal",
        eval_config_identity_hash=FULL_B,
        reward_policy_ref=FULL_C,
        capacity=ToolCapacity(max_accepted_calls=capacity),
        store_namespace=namespace,
    )


class RecordingEvaluationService:
    """Resolves Evaluation Intents under their exact target Eval Config."""

    def __init__(self, store: ObjectStore) -> None:
        self._store = store
        self.resolved: list[EvaluationIntent] = []

    def resolve_evaluation_intent(
        self, intent: EvaluationIntent
    ) -> IntentResolution:
        self.resolved.append(intent)
        # Store a stand-in evaluation-evidence record (a Rollout Aggregate ref
        # would be produced by the real path).
        evidence: dict[str, Any] = {
            "intent_id": intent.intent_id,
            "candidate_id": intent.candidate_id,
            "eval_config_hash": intent.target_eval_config_hash,
            "role": intent.context_role.value,
        }
        self._store.put(EVIDENCE_SCHEMA, evidence)
        evidence_ref = typed_ref_for_record(EVIDENCE_SCHEMA, evidence)
        return IntentResolution(
            intent=intent,
            evaluation_evidence_refs=(evidence_ref,),
            resolved_eval_config_hash=intent.target_eval_config_hash,
        )


class WrongTargetEvaluationService:
    """Resolves under a DIFFERENT Eval Config than the Intent's target."""

    def resolve_evaluation_intent(
        self, intent: EvaluationIntent
    ) -> IntentResolution:
        evidence: dict[str, Any] = {"intent_id": intent.intent_id}
        evidence_ref = typed_ref_for_record(EVIDENCE_SCHEMA, evidence)
        # Deliberately resolve under the wrong Eval Config identity.
        wrong = FULL_D if intent.target_eval_config_hash != FULL_D else FULL_C
        return IntentResolution(
            intent=intent,
            evaluation_evidence_refs=(evidence_ref,),
            resolved_eval_config_hash=wrong,
        )


class RecordingToolExecutor:
    """Constructs a Runtime Tool Handle bound to the Tool Call Store."""

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
                    tool_config_ref="toolcfg://evaluate_candidate",
                    tool_config_hash=config.identity_hash(),
                    store_namespace=config.store_namespace,
                    refusal=entry.refusal,
                )
            ordinal = entry.capacity_debit_ordinal
            return ToolResult(
                call_id=call.call_id,
                tool_config_ref="toolcfg://evaluate_candidate",
                tool_config_hash=config.identity_hash(),
                store_namespace=config.store_namespace,
                output={"rollout_refs": [], "accepted_ordinal": ordinal},
                reward={"reward_name": "reward", "value": 1.0},
            )

        return RuntimeToolHandle(config, execute)


class CountingProposalAdapter:
    """A proposal-only adapter that counts invocations.

    Returns one proposed candidate plus one Evaluation Intent. The invocation
    counter lets restart tests prove a completed proposal invocation is reused
    from the checkpoint rather than rerun.
    """

    def __init__(
        self,
        *,
        target_hash: str = FULL_B,
        status: StepStatus = StepStatus.CONTINUE,
    ) -> None:
        self.invocations = 0
        self._target_hash = target_hash
        self._status = status

    @property
    def mode(self) -> StepMode:
        return StepMode.PROPOSAL_ONLY

    def invoke(
        self, request: OptimizationStepRequest, handles
    ) -> AdapterOutput:
        self.invocations += 1
        cand = Candidate(
            candidate_id="P1", base_ref="base", payload={"template": "new"}
        )
        intent = make_intent(
            run_id=request.run_id,
            step_index=request.step_index,
            candidate_id="P1",
            target_hash=self._target_hash,
        )
        return AdapterOutput(
            proposed_candidates=(cand,),
            accepted_candidates=(cand,),
            evaluation_intents=(intent,),
            proposed_status=self._status,
        )


class ToolUsingAdapter:
    """A tool-using adapter that issues N Tool Calls through the handle."""

    def __init__(self, *, call_ids: tuple[str, ...] = ("c1",)) -> None:
        self._call_ids = call_ids
        self.invocations = 0

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
        records = []
        for call_id in self._call_ids:
            call = ToolCall(
                call_id=call_id,
                tool_config_hash=handle.tool_config_hash,
                store_namespace=handle.config.store_namespace,
                args={"model_route": "r0", "template": call_id},
            )
            result = handle(call)
            records.append(ToolCallRecord(call=call, result=result))
        cand = Candidate(
            candidate_id="TP", base_ref="base", payload={"template": "tool"}
        )
        return AdapterOutput(
            proposed_candidates=(cand,),
            accepted_candidates=(cand,),
            tool_call_records=tuple(records),
            proposed_status=StepStatus.COMPLETE,
        )


def assert_adapter(adapter: OptimizerAdapter) -> None:
    """Static-ish guard that a test double satisfies the adapter surface."""
    assert isinstance(adapter, OptimizerAdapter)


def tool_request(
    *,
    run_id: str = "run-tool",
    step_index: int = 0,
    config: ToolConfig | None = None,
    prior_step_result_ref=None,
) -> OptimizationStepRequest:
    cfg = config or make_tool_definition_config()
    return OptimizationStepRequest(
        run_id=run_id,
        step_id=f"{run_id}-s{step_index}",
        optimizer_config_hash=FULL_A,
        mode=StepMode.TOOL_USING,
        kind=StepKind.TOOL,
        step_index=step_index,
        candidates=(candidate("A"),),
        output_contract=output_contract(1),
        tool_configs=(cfg,),
        prior_step_result_ref=prior_step_result_ref,
    )
