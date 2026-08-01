"""Durable stores, builders, and generic adapter test doubles."""

from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from typing import Any

from dr_code.eval import DefinitionRef, EvalConfig
from dr_store import ObjectStore, SqliteBackend

from whetstone.evaluation_role import EvaluationRole
from whetstone.optimization import (
    EVALUATION_BINDING_SCHEMA_VERSION,
    EVALUATION_EVIDENCE_SCHEMA,
    EVALUATION_FAILURE_SCHEMA,
    INTENT_RESOLUTION_SCHEMA_VERSION,
    AdapterOutput,
    AdapterRegistry,
    BudgetDelta,
    BudgetState,
    Candidate,
    EvaluationIntent,
    IdentityOptimizerAdapter,
    IdentityRef,
    IntentOutcome,
    IntentResolution,
    MappingAdapterRegistry,
    MissingDataPolicy,
    OptimizationHarness,
    OptimizationRun,
    OptimizationRunRef,
    OptimizationStepRequest,
    OptimizerAdapter,
    OutputContract,
    ReplayPolicy,
    ResolutionClass,
    ResolutionDetail,
    RewardPolicy,
    RewardTerm,
    RuntimeToolHandle,
    StepKind,
    StepMode,
    StepStatus,
    TemplateRenderContract,
    TemplateRenderKind,
    TerminalFailure,
    ToolCall,
    ToolCapacity,
    ToolCapacityScope,
    ToolConfig,
    ToolDefinition,
    ToolResult,
    apply_reward_policy,
    candidate_reference,
    compute_identity_hash,
    eval_config_reference,
    optimization_run_reference,
    reward_reference,
    tool_call_reference,
    tool_config_reference,
    tool_definition_reference,
    typed_ref_for_record,
)
from whetstone.optimization.effect_authority import (
    AcquireOutcome,
    EffectAuthority,
)
from whetstone.optimization.identity import TypedRef
from whetstone.optimization.schema import (
    EVALUATION_BINDING_SCHEMA_VERSION,
    EvalConfigRef,
    EvaluationBinding,
)
from whetstone.optimization.tool_store import (
    ToolAdmissionAuthority,
    ToolCallState,
    ToolCallStore,
    tool_effect_request,
)
from whetstone.optimization.tools import ToolCapacityBinding

FULL_A = "a" * 64
FULL_B = "b" * 64
FULL_C = "c" * 64
FULL_D = "d" * 64
REWARD_EVIDENCE_SCHEMA = "whetstone.test.reward_evidence"
BASE_SCHEMA = "whetstone.test.candidate_base"
OPTIMIZER_CONFIG_SCHEMA = "whetstone.test.optimizer_config"
OPTIMIZER_CONFIG_SCHEMA_VERSION = 1
EVALUATION_CAMPAIGN = "optimization-support"

type RunInput = OptimizationRun | OptimizationRunRef


def internal_reward_policy() -> RewardPolicy:
    return RewardPolicy(
        policy_name="support-policy/v1",
        reward_name="support-score",
        terms=(RewardTerm(name="score", weight=1.0),),
        missing_data=MissingDataPolicy.FAIL,
    )


def base_ref(label: str = "base") -> TypedRef:
    return typed_ref_for_record(BASE_SCHEMA, {"label": label})


def make_store(
    tmp_path: Path, name: str = "optimization.sqlite"
) -> ObjectStore:
    return ObjectStore(SqliteBackend(tmp_path / name))


def memory_tool_call_store(
    store: ObjectStore, effect_authority: EffectAuthority
) -> ToolCallStore:
    """Build a test-only Tool store with explicitly ephemeral admission."""
    return ToolCallStore(
        store,
        ToolAdmissionAuthority.memory(),
        effect_authority,
    )


def make_harness(
    *,
    store: ObjectStore,
    adapter_registry: AdapterRegistry,
    run: OptimizationRunRef,
    effect_authority: EffectAuthority | None = None,
    tool_store: ToolCallStore | None = None,
    evaluation_service: Any | None = None,
    tool_executor: Any | None = None,
    owner_id: str = "optimization-test-owner",
    adapter_replay_policy: ReplayPolicy = ReplayPolicy.IDEMPOTENT,
    lease_duration: timedelta = timedelta(seconds=1),
) -> OptimizationHarness:
    authority = effect_authority or EffectAuthority.memory()
    exact_tool_store = tool_store or memory_tool_call_store(store, authority)
    harness = OptimizationHarness(
        store=store,
        adapter_registry=adapter_registry,
        tool_store=exact_tool_store,
        effect_authority=authority,
        owner_id=owner_id,
        adapter_replay_policy=adapter_replay_policy,
        lease_duration=lease_duration,
        evaluation_service=evaluation_service,
        tool_executor=tool_executor,
    )
    harness.bind_run(run)
    return harness


def candidate(
    cid: str = "A", *, base: str = "base", text: str = "t"
) -> Candidate:
    return Candidate(
        candidate_id=cid,
        base_ref=base_ref(base),
        payload={"user_prompt_template": text, "fixed": "same"},
    )


def proposed_candidate(
    base: Candidate,
    cid: str,
    *,
    text: str,
) -> Candidate:
    payload = base.payload.to_json()
    payload["user_prompt_template"] = text
    return Candidate(
        candidate_id=cid,
        base_ref=candidate_reference(base).record_ref,
        payload=payload,
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


def python_format_contract(
    *,
    available_fields: tuple[str, ...] = ("query",),
    required_fields: tuple[str, ...] = (),
) -> TemplateRenderContract:
    return TemplateRenderContract(
        kind=TemplateRenderKind.PYTHON_FORMAT_V1,
        available_fields=available_fields,
        required_fields=required_fields,
    )


def optimizer_config_ref(algorithm: str) -> IdentityRef:
    record = {"algorithm": algorithm}
    return IdentityRef(
        record_ref=typed_ref_for_record(OPTIMIZER_CONFIG_SCHEMA, record),
        identity_hash=compute_identity_hash(
            schema=OPTIMIZER_CONFIG_SCHEMA,
            schema_version=OPTIMIZER_CONFIG_SCHEMA_VERSION,
            payload=record,
        ),
    )


def pure_run(
    *,
    run_id: str = "run-pure",
    contract: OutputContract | None = None,
    template_render_contract: TemplateRenderContract | None = None,
) -> OptimizationRunRef:
    return optimization_run_reference(
        OptimizationRun(
            run_id=run_id,
            optimizer_config=optimizer_config_ref("identity"),
            adapter_key="identity",
            mode=StepMode.PURE,
            terminal_output_contract=contract or output_contract(),
            template_render_contract=(
                template_render_contract or python_format_contract()
            ),
        )
    )


def proposal_run(
    *,
    run_id: str = "run-proposal",
    contract: OutputContract | None = None,
    template_render_contract: TemplateRenderContract | None = None,
    reward_policy: RewardPolicy | None = None,
) -> OptimizationRunRef:
    return optimization_run_reference(
        OptimizationRun(
            run_id=run_id,
            optimizer_config=optimizer_config_ref("proposal"),
            adapter_key="proposal-test",
            mode=StepMode.PROPOSAL_ONLY,
            terminal_output_contract=contract or output_contract(),
            template_render_contract=(
                template_render_contract or python_format_contract()
            ),
            reward_policy=reward_policy or internal_reward_policy(),
        )
    )


def tool_run(
    *,
    run_id: str = "run-tool",
    config: ToolConfig | None = None,
    contract: OutputContract | None = None,
    template_render_contract: TemplateRenderContract | None = None,
) -> OptimizationRunRef:
    cfg = config or make_tool_definition_config()
    return optimization_run_reference(
        OptimizationRun(
            run_id=run_id,
            optimizer_config=optimizer_config_ref("tool"),
            adapter_key="tool-test",
            mode=StepMode.TOOL_USING,
            terminal_output_contract=contract or output_contract(),
            template_render_contract=(
                template_render_contract or python_format_contract()
            ),
            tool_configs=(tool_config_reference(cfg),),
        )
    )


def _exact_run_ref(
    run: RunInput | None,
    *,
    run_id: str | None,
    default_run_id: str,
    build: Callable[[str], OptimizationRunRef],
) -> OptimizationRunRef:
    if run is None:
        return build(run_id or default_run_id)
    exact = (
        run
        if isinstance(run, OptimizationRunRef)
        else optimization_run_reference(run)
    )
    if run_id is not None and exact.record.run_id != run_id:
        raise ValueError(
            f"run_id {run_id!r} does not match exact run "
            f"{exact.record.run_id!r}"
        )
    return exact


def pure_request(
    *,
    run: RunInput | None = None,
    run_id: str | None = None,
    candidates: tuple[Candidate, ...] | None = None,
    contract: OutputContract | None = None,
    template_render_contract: TemplateRenderContract | None = None,
) -> OptimizationStepRequest:
    records = candidates if candidates is not None else (candidate(),)
    step_contract = contract or output_contract(len(records))
    exact_run = _exact_run_ref(
        run,
        run_id=run_id,
        default_run_id="run-pure",
        build=lambda run_id: pure_run(
            run_id=run_id,
            contract=step_contract,
            template_render_contract=template_render_contract,
        ),
    )
    return OptimizationStepRequest(
        run=exact_run,
        step_id=f"{exact_run.record.run_id}-s0",
        kind=StepKind.IDENTITY,
        step_index=0,
        candidates=records,
        step_output_contract=step_contract,
        budget=BudgetState(remaining={"rollouts": 10}),
    )


def proposal_request(
    *,
    run: RunInput | None = None,
    run_id: str | None = None,
    step_index: int = 0,
    prior_step_result_ref: TypedRef | None = None,
    budget: BudgetState | None = None,
    contract: OutputContract | None = None,
    candidates: tuple[Candidate, ...] | None = None,
    template_render_contract: TemplateRenderContract | None = None,
) -> OptimizationStepRequest:
    step_contract = contract or output_contract()
    exact_run = _exact_run_ref(
        run,
        run_id=run_id,
        default_run_id="run-proposal",
        build=lambda value: proposal_run(
            run_id=value,
            contract=step_contract,
            template_render_contract=template_render_contract,
        ),
    )
    return OptimizationStepRequest(
        run=exact_run,
        step_id=f"{exact_run.record.run_id}-s{step_index}",
        kind=StepKind.PROPOSAL,
        step_index=step_index,
        prior_step_result_ref=prior_step_result_ref,
        candidates=candidates if candidates is not None else (candidate(),),
        step_output_contract=step_contract,
        budget=budget or BudgetState(remaining={"rollouts": 10}),
    )


def evaluation_binding(
    config: EvalConfigRef | None = None,
) -> EvaluationBinding:
    return EvaluationBinding(
        schema_version=EVALUATION_BINDING_SCHEMA_VERSION,
        eval_config=config or eval_config_reference(eval_config()),
        role=EvaluationRole.INTERNAL,
        authority_principal=None,
        campaign=EVALUATION_CAMPAIGN,
    )


def make_intent(
    proposed: Candidate,
    *,
    run_id: str = "run-proposal",
    step_index: int = 0,
    binding: EvaluationBinding | None = None,
    reward_policy: RewardPolicy | None = None,
) -> EvaluationIntent:
    exact_binding = binding or evaluation_binding()
    expected_policy = reward_policy or internal_reward_policy()
    return EvaluationIntent(
        intent_id=f"{run_id}-{step_index}-{proposed.candidate_id}",
        candidate=candidate_reference(proposed),
        target_eval_config=exact_binding.eval_config,
        evaluation_binding=exact_binding,
        purpose="proposal",
        run_id=run_id,
        step_index=step_index,
        expected_reward_policy_hash=(
            expected_policy.identity_hash()
            if exact_binding.role is EvaluationRole.INTERNAL
            else None
        ),
    )


def make_tool_definition_config(
    *, capacity: int = 2, namespace: str = "ns-1"
) -> ToolConfig:
    definition = ToolDefinition(
        tool_name="evaluate_candidate",
        input_fields=("model_route", "template"),
        output_fields=("rollout_refs", "accepted_ordinal"),
    )
    return ToolConfig(
        definition=tool_definition_reference(definition),
        endpoint_key="evaluate_candidate",
        eval_config=eval_config(),
        reward_policy_hash=FULL_C,
        capacity=ToolCapacity(
            max_accepted_calls=capacity,
            scope=ToolCapacityScope.RUN,
        ),
        store_namespace_key=namespace,
    )


def tool_request(
    *,
    run: RunInput | None = None,
    run_id: str | None = None,
    config: ToolConfig | None = None,
    contract: OutputContract | None = None,
    candidates: tuple[Candidate, ...] | None = None,
    budget: BudgetState | None = None,
    template_render_contract: TemplateRenderContract | None = None,
) -> OptimizationStepRequest:
    if run is not None and config is not None:
        raise ValueError("config must be composed into the supplied exact run")
    step_contract = contract or output_contract()
    exact_run = _exact_run_ref(
        run,
        run_id=run_id,
        default_run_id="run-tool",
        build=lambda value: tool_run(
            run_id=value,
            config=config,
            contract=step_contract,
            template_render_contract=template_render_contract,
        ),
    )
    return OptimizationStepRequest(
        run=exact_run,
        step_id=f"{exact_run.record.run_id}-s0",
        kind=StepKind.TOOL,
        step_index=0,
        candidates=candidates if candidates is not None else (candidate(),),
        step_output_contract=step_contract,
        budget=budget or BudgetState(remaining={"tool_calls": 10}),
    )


def registry(*adapters: OptimizerAdapter) -> MappingAdapterRegistry:
    values: list[OptimizerAdapter] = [
        IdentityOptimizerAdapter(),
        *adapters,
    ]
    return MappingAdapterRegistry({adapter.key: adapter for adapter in values})


class RecordingEvaluationService:
    def __init__(
        self,
        store: ObjectStore,
        *,
        outcome: IntentOutcome = IntentOutcome.COMPLETED,
        replay_policy: ReplayPolicy = ReplayPolicy.IDEMPOTENT,
        crash_on_call: int | None = None,
        reward_policy: RewardPolicy | None = None,
        persist_evaluation_result: bool = True,
        persist_reward_evidence: bool = True,
    ) -> None:
        if replay_policy not in {
            ReplayPolicy.IDEMPOTENT,
            ReplayPolicy.DURABLE_WORKFLOW,
        }:
            raise ValueError(
                "test EvaluationService replay policy must permit redrive"
            )
        if crash_on_call is not None and crash_on_call < 1:
            raise ValueError("crash_on_call must be one-based")
        self._store = store
        self._outcome = outcome
        self._replay_policy = replay_policy
        self._crash_on_call = crash_on_call
        self._reward_policy = reward_policy or internal_reward_policy()
        self._persist_evaluation_result = persist_evaluation_result
        self._persist_reward_evidence = persist_reward_evidence
        self.calls: list[EvaluationIntent] = []
        self.validation_calls: list[IntentResolution] = []

    @property
    def replay_policy(self) -> ReplayPolicy:
        return self._replay_policy

    def resolve_evaluation_intent(
        self, intent: EvaluationIntent
    ) -> IntentResolution:
        self.calls.append(intent)
        if self._crash_on_call == len(self.calls):
            raise RuntimeError("crash during evaluation resolution")
        if self._outcome is IntentOutcome.REJECTED:
            return IntentResolution(
                schema_version=INTENT_RESOLUTION_SCHEMA_VERSION,
                intent=intent,
                outcome=self._outcome,
                detail=ResolutionDetail(
                    classification=ResolutionClass.VALIDATION,
                    message="candidate rejected before execution",
                ),
                resolved_eval_config=intent.target_eval_config,
            )
        evaluation_result: dict[str, Any] = {
            "intent_id": intent.intent_id,
            "candidate_identity_hash": intent.candidate.identity_hash,
            "outcome": self._outcome.value,
        }
        evaluation_result_schema = (
            EVALUATION_EVIDENCE_SCHEMA
            if self._outcome is IntentOutcome.COMPLETED
            else EVALUATION_FAILURE_SCHEMA
        )
        if self._persist_evaluation_result:
            self._store.put(evaluation_result_schema, evaluation_result)
        classification = (
            ResolutionClass.MEASURED
            if self._outcome is IntentOutcome.COMPLETED
            else ResolutionClass.UNSCORABLE
        )
        evaluation_result_ref = typed_ref_for_record(
            evaluation_result_schema, evaluation_result
        )
        reward_evidence_refs: tuple[TypedRef, ...] = ()
        if self._outcome is IntentOutcome.COMPLETED:
            refs: list[TypedRef] = []
            for ordinal in range(2):
                evidence: dict[str, Any] = {
                    "intent_id": intent.intent_id,
                    "ordinal": ordinal,
                }
                if self._persist_reward_evidence:
                    self._store.put(REWARD_EVIDENCE_SCHEMA, evidence)
                refs.append(
                    typed_ref_for_record(REWARD_EVIDENCE_SCHEMA, evidence)
                )
            reward_evidence_refs = tuple(refs)
        reward = (
            reward_reference(
                apply_reward_policy(
                    self._reward_policy,
                    aggregates={"score": 1.0},
                    evidence_role=EvaluationRole.INTERNAL,
                    evidence_refs=reward_evidence_refs,
                )
            )
            if self._outcome is IntentOutcome.COMPLETED
            else None
        )
        return IntentResolution(
            schema_version=INTENT_RESOLUTION_SCHEMA_VERSION,
            intent=intent,
            outcome=self._outcome,
            detail=ResolutionDetail(
                classification=classification,
                message="evaluation completed"
                if self._outcome is IntentOutcome.COMPLETED
                else "candidate was unscorable",
            ),
            evaluation_result_ref=evaluation_result_ref,
            reward_evidence_refs=reward_evidence_refs,
            resolved_eval_config=intent.target_eval_config,
            reward_ref=reward,
            terminal_failure=(
                TerminalFailure(code="evaluation_failed", message="unscorable")
                if self._outcome is IntentOutcome.FAILED
                else None
            ),
        )

    def validate_resolution_graph(self, resolution: IntentResolution) -> None:
        self.validation_calls.append(resolution)


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

    @property
    def required_replay_policy(self) -> ReplayPolicy:
        return ReplayPolicy.IDEMPOTENT

    def invoke(
        self,
        request: OptimizationStepRequest,
        handles: tuple[RuntimeToolHandle, ...],
    ) -> AdapterOutput:
        self.invocations += 1
        proposed = (
            tuple(
                proposed_candidate(
                    request.candidates[0],
                    str(item.candidate_id),
                    text=(
                        str(item.payload["user_prompt_template"])
                        if item.payload["user_prompt_template"]
                        != request.candidates[0].payload[
                            "user_prompt_template"
                        ]
                        else f"{item.candidate_id}-proposal"
                    ),
                )
                for item in self._candidates
            )
            if self._candidates is not None
            else (proposed_candidate(request.candidates[0], "P1", text="new"),)
        )
        failed = self._status is StepStatus.FAILED
        return AdapterOutput(
            proposed_candidates=proposed,
            accepted_candidates=() if failed else proposed,
            evaluation_intents=(
                ()
                if failed
                else tuple(
                    make_intent(
                        record,
                        run_id=request.run_id,
                        step_index=request.step_index,
                        reward_policy=request.run.record.reward_policy,
                    )
                    for record in proposed
                )
            ),
            budget_delta=self._budget_delta,
            proposed_status=self._status,
            terminal_failure=(
                TerminalFailure(
                    code="proposal_adapter_failed",
                    message="proposal adapter failed before evaluation",
                )
                if failed
                else None
            ),
        )


class RecordingToolExecutor:
    def __init__(self, effect_authority: EffectAuthority) -> None:
        self.handles_built = 0
        self.calls: list[ToolCall] = []
        self._effect_authority = effect_authority

    def runtime_handle(
        self,
        config: ToolConfig,
        store: ToolCallStore,
        binding: ToolCapacityBinding,
    ) -> RuntimeToolHandle:
        self.handles_built += 1

        def execute(call: ToolCall) -> ToolResult:
            self.calls.append(call)
            entry = store.admit(call, config)
            if entry.state is not ToolCallState.ACCEPTED:
                return store.load_terminal_result(entry)
            assert entry.capacity_debit_ordinal is not None
            result = ToolResult(
                call=tool_call_reference(call),
                output={
                    "rollout_refs": [],
                    "accepted_ordinal": entry.capacity_debit_ordinal,
                },
                provenance_ordinal=entry.capacity_debit_ordinal,
            )
            result_ref = store.persist_result(result)
            acquisition = self._effect_authority.acquire(
                tool_effect_request(call),
                owner_id="recording-tool-executor",
                attempt_id=f"attempt-{call.call_id}",
                lease_duration=timedelta(seconds=1),
            )
            if (
                acquisition.outcome is not AcquireOutcome.ACQUIRED
                or acquisition.lease is None
            ):
                raise AssertionError(
                    "recording Tool execution did not acquire its exact effect"
                )
            terminal = self._effect_authority.succeed(
                acquisition.lease,
                result_ref=result_ref,
            )
            store.complete(
                result,
                terminal=terminal,
            )
            return result

        return RuntimeToolHandle(config, binding, execute)


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

    @property
    def required_replay_policy(self) -> ReplayPolicy:
        return ReplayPolicy.IDEMPOTENT

    def invoke(
        self,
        request: OptimizationStepRequest,
        handles: tuple[RuntimeToolHandle, ...],
    ) -> AdapterOutput:
        self.invocations += 1
        if tuple(handle.tool_config_ref for handle in handles) != (
            request.tool_configs
        ):
            raise ValueError("Runtime Tool Handles do not match the exact run")
        handle = handles[0]
        for call_id in self.call_ids:
            call = ToolCall(
                call_id=call_id,
                tool_config=handle.tool_config_ref,
                capacity_binding=handle.binding,
                args={"model_route": "r0", "template": call_id},
            )
            handle(call)
        proposed = proposed_candidate(request.candidates[0], "TP", text="tool")
        return AdapterOutput(
            proposed_candidates=(proposed,),
            accepted_candidates=(proposed,),
            proposed_status=StepStatus.COMPLETE,
        )
