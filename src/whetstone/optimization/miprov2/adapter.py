from __future__ import annotations

from typing import Any

from dr_store import ObjectStore

from whetstone.core.effects.authority import ReplayPolicy
from whetstone.core.identity import (
    TerminalFailure,
    TypedRef,
    require_full_hash,
)
from whetstone.experiment.binding import EvaluationBinding
from whetstone.experiment.candidate import candidate_reference
from whetstone.optimization.adapters import AdapterOutput
from whetstone.optimization.contracts import (
    STEP_RESULT_SCHEMA,
    BudgetDelta,
    BudgetState,
    EvaluationIntent,
    IntentOutcome,
    IntentResolution,
    OptimizationRunRef,
    OptimizationStepRequest,
    OptimizationStepResult,
    OutputContract,
    StepKind,
    StepMode,
    StepStatus,
    step_result_reference,
)
from whetstone.optimization.miprov2.eval_config import (
    Miprov2EvalConfigResolver,
)
from whetstone.optimization.miprov2.evidence import (
    Miprov2EvidenceResolver,
    Miprov2IntentContext,
    load_miprov2_intent_context,
    persist_miprov2_intent_context,
)
from whetstone.optimization.miprov2.proposal import (
    Miprov2ProposalRequest,
    Miprov2ProposalResponse,
)
from whetstone.optimization.miprov2.runtime import (
    Miprov2Driver,
    Miprov2DriverPlan,
    Miprov2State,
)
from whetstone.optimization.proposal.proposer import (
    DurableProposalExecutor,
    ProposalExecutorDurabilityContract,
    ProposalRequest,
    ProposerConfig,
    ProposerTransport,
)

MIPROV2_ADAPTER_KEY = "miprov2"
MIPROV2_STATE_KEY = "miprov2_state"
MIPROV2_BOOTSTRAP = "bootstrap_generation"
MIPROV2_PROPOSAL = "proposal_model"
MIPROV2_BASELINE = "baseline_evaluation"
MIPROV2_SAMPLE = "sample_evaluation"
MIPROV2_PROMOTION = "promotion_evaluation"
MIPROV2_COMPLETE = "complete"
MIPROV2_FAILED = "failed"
STATE_SNAPSHOT_SCHEMA = "whetstone.optimization_state_snapshot"

MIPROV2_PROPOSAL_FAILED_CODE = "miprov2_instruction_proposal_failed"
MIPROV2_INTENT_REJECTED_CODE = "miprov2_evaluation_intent_rejected"


def _rejection_detail(
    intent_id: str,
    resolution: IntentResolution,
) -> str:

    return (
        f"MIPROv2 Evaluation Intent {intent_id!r} was rejected before "
        f"execution ({resolution.detail.classification.value}): "
        f"{resolution.detail.message}"
    )


def _terminalized(state: Miprov2State, *, failure: str) -> Miprov2State:

    if not failure:
        raise ValueError("a terminal MIPROv2 state requires its exact cause")
    return Miprov2State.model_validate(
        state.model_copy(
            update={
                "phase": MIPROV2_FAILED,
                "failure": failure,
                "pending_bootstrap": None,
                "pending_bootstrap_candidate": None,
                "pending_proposal": None,
                "pending_evaluation_spec": None,
                "pending_eval_binding_request": None,
                "resolved_eval_binding": None,
                "pending_evaluation": None,
                "pending_sample": None,
            }
        ).model_dump(mode="json")
    )


class Miprov2Adapter:
    def __init__(
        self,
        *,
        store: ObjectStore,
        proposer_config: ProposerConfig,
        transport: ProposerTransport,
        eval_config_resolver: Miprov2EvalConfigResolver,
        proposal_executor: DurableProposalExecutor,
        driver: Miprov2Driver | None = None,
    ) -> None:
        if type(proposal_executor) is not DurableProposalExecutor:
            raise TypeError(
                "MIPROv2 requires the canonical DurableProposalExecutor "
                "capability for its paid proposal call"
            )
        self._proposer_config = proposer_config
        self._transport = transport
        self._store = store
        self._eval_config_resolver = eval_config_resolver
        self._proposal_executor = proposal_executor
        executor_contract = proposal_executor.durability_contract
        if (
            executor_contract.recovery_policy
            is not ReplayPolicy.DURABLE_WORKFLOW
        ):
            raise ValueError(
                "MIPROv2 proposal executor must provide durable-workflow "
                "recovery"
            )
        transport_durability_identity_hash = transport.durability_identity_hash
        require_full_hash(
            transport_durability_identity_hash,
            field="proposal_transport_durability_identity_hash",
        )
        self._proposal_executor_policy_identity_hash = (
            executor_contract.policy_identity_hash
        )
        self._proposal_executor_contract = executor_contract
        self._proposal_transport_durability_identity_hash = (
            transport_durability_identity_hash
        )
        self._driver = driver or Miprov2Driver()
        self._evidence = Miprov2EvidenceResolver(store)

    @property
    def key(self) -> str:
        return MIPROV2_ADAPTER_KEY

    @property
    def mode(self) -> StepMode:
        return StepMode.PROPOSAL_ONLY

    @property
    def required_replay_policy(self) -> ReplayPolicy:
        return ReplayPolicy.DURABLE_WORKFLOW

    @property
    def proposal_executor(self) -> DurableProposalExecutor:
        return self._proposal_executor

    @property
    def proposer_config(self) -> ProposerConfig:
        return self._proposer_config

    def build_step_request(
        self,
        *,
        run: OptimizationRunRef | None = None,
        step_index: int,
        initial_state: Miprov2State | None = None,
        initial_budget: BudgetState | None = None,
        prior_result: OptimizationStepResult | None = None,
        prior_result_ref: TypedRef | None = None,
    ) -> OptimizationStepRequest:

        if step_index == 0:
            if (
                initial_state is None
                or initial_budget is None
                or prior_result is not None
                or prior_result_ref is not None
            ):
                raise ValueError(
                    "initial request requires only state and budget"
                )
            if run is None:
                raise ValueError("initial request requires the exact run")
            state = initial_state
            budget = initial_budget
            pools = {MIPROV2_STATE_KEY: state.model_dump(mode="json")}
            prior_state_ref = None
            exact_run = run
        else:
            if (
                initial_state is not None
                or initial_budget is not None
                or prior_result is None
                or prior_result_ref is None
            ):
                raise ValueError(
                    "continuation requires only exact prior result and ref"
                )
            if (
                step_result_reference(prior_result).record_ref
                != prior_result_ref
            ):
                raise ValueError("prior result ref is not its exact record")
            state_ref = prior_result.state_ref
            if state_ref is None:
                raise ValueError("prior result has no state snapshot")
            snapshot = self._store.get(state_ref.reference)
            if not isinstance(snapshot, dict):
                raise ValueError("prior state snapshot must be an object")
            state = Miprov2State.model_validate(snapshot[MIPROV2_STATE_KEY])
            state = self._fold_prior_resolutions(state, prior_result)
            budget = prior_result.budget
            pools = {}
            prior_state_ref = state_ref
            exact_run = prior_result.request.record.run
        if exact_run.record.optimizer_config != state.control.reference():
            raise ValueError(
                "run optimizer config differs from MIPROv2 control"
            )
        if exact_run != state.run:
            raise ValueError("run differs from the MIPROv2 state authority")
        if (
            exact_run.record.template_render_contract
            != state.control.template_render_contract
        ):
            raise ValueError(
                "run render contract differs from MIPROv2 control"
            )
        if (
            exact_run.record.reward_policy is None
            or exact_run.record.reward_policy != state.control.reward_policy
        ):
            raise ValueError("run Reward Policy differs from MIPROv2 control")
        self._require_budget_agreement(budget, state)
        if state.phase == MIPROV2_FAILED:
            kind_label = MIPROV2_FAILED
            returned_count = 0
        else:
            preview = self._driver.plan(state)
            kind_label = preview.kind
            returned_count = 1 if preview.kind == MIPROV2_COMPLETE else 0
        return OptimizationStepRequest(
            run=exact_run,
            step_id=f"{state.run_id}:miprov2:{step_index}",
            kind=StepKind.PROPOSAL,
            kind_label=kind_label,
            step_index=step_index,
            prior_step_result_ref=prior_result_ref,
            prior_state_ref=prior_state_ref,
            pools=pools,
            candidates=(
                (state.control.base_candidate.record,)
                if state.control.teacher_candidate
                == state.control.base_candidate
                else (
                    state.control.base_candidate.record,
                    state.control.teacher_candidate.record,
                )
            ),
            budget=budget,
            step_output_contract=OutputContract(
                returned_proposal_count=returned_count
            ),
        )

    def invoke(
        self,
        request: OptimizationStepRequest,
        handles: tuple[Any, ...],
    ) -> AdapterOutput:
        if handles:
            raise ValueError("MIPROv2 receives no Runtime Tool Handles")
        if request.adapter_key != self.key or request.mode is not self.mode:
            raise ValueError("request is not bound to the MIPROv2 adapter")
        state, prior = self._load_request_state(request)
        if prior is not None and request.budget != prior.budget:
            raise ValueError(
                "MIPROv2 continuation must carry the prior exact budget"
            )
        self._require_transport_bindings(state)
        if request.run_id != state.run_id:
            raise ValueError("request run_id conflicts with MIPROv2 state")
        if request.run != state.run:
            raise ValueError(
                "request run conflicts with MIPROv2 state authority"
            )
        if request.run.record.optimizer_config != state.control.reference():
            raise ValueError("request run conflicts with MIPROv2 control")
        if request.run.record.reward_policy != state.control.reward_policy:
            raise ValueError(
                "request run conflicts with MIPROv2 Reward Policy"
            )
        if prior is not None:
            state = self._fold_prior_resolutions(state, prior)
        self._require_budget_agreement(request.budget, state)
        if state.phase == MIPROV2_FAILED:
            assert state.failure is not None
            return self._terminal_failure_output(
                state,
                failure=TerminalFailure(
                    code=MIPROV2_INTENT_REJECTED_CODE,
                    message=state.failure,
                ),
            )
        plan = self._driver.plan(state)
        if plan.kind == "eval_config_binding":
            assert plan.eval_config_binding is not None
            self._preflight_task_rows(
                request.budget,
                len(plan.eval_config_binding.task_batch_hashes),
            )
            binding = self._eval_config_resolver.resolve(
                plan.eval_config_binding
            )
            advanced = self._driver.fold_eval_config_binding(
                plan.state, binding
            )
            return AdapterOutput(
                state_delta={
                    MIPROV2_STATE_KEY: advanced.model_dump(mode="json")
                }
            )
        if plan.kind == MIPROV2_PROPOSAL:
            return self._proposal_output(plan)
        if plan.kind == MIPROV2_COMPLETE:
            assert plan.accepted_candidate is not None
            return AdapterOutput(
                proposed_candidates=(plan.accepted_candidate,),
                accepted_candidates=(plan.accepted_candidate,),
                proposed_status=StepStatus.COMPLETE,
                state_delta={
                    MIPROV2_STATE_KEY: plan.state.model_dump(mode="json")
                },
            )
        if plan.kind == MIPROV2_BOOTSTRAP:
            assert plan.bootstrap_generation is not None
            assert plan.state.resolved_eval_binding is not None
            assert plan.state.pending_bootstrap_candidate is not None
            attempt = plan.bootstrap_generation
            self._preflight_task_rows(request.budget, 1)
            teacher_candidate = plan.state.pending_bootstrap_candidate
            intent_id = (
                f"{request.run_id}:miprov2:bootstrap:{attempt.identity_hash()}"
            )
            component = state.control.component_specs[0]
            evaluation_binding = EvaluationBinding.model_validate(
                {
                    **state.control.evaluation_binding.model_dump(mode="json"),
                    "eval_config": (
                        plan.state.resolved_eval_binding.eval_config.model_dump(
                            mode="json"
                        )
                    ),
                }
            )
            context = Miprov2IntentContext(
                control_identity_hash=state.control.identity_hash(),
                run_id=state.run_id,
                effect_kind="bootstrap",
                effect_identity_hash=attempt.identity_hash(),
                intent_id=intent_id,
                candidate=teacher_candidate,
                task_batch_hashes=(attempt.task_hash,),
                eval_config=plan.state.resolved_eval_binding.eval_config,
                eval_config_binding=plan.state.resolved_eval_binding,
                evaluation_binding=evaluation_binding,
                execution_policy=(
                    plan.state.resolved_eval_binding.request.execution_policy
                ),
                reward_policy_hash=state.control.reward_policy_hash,
                bootstrap_attempt=attempt,
                optimizable_component_id=component.component_id,
                optimizable_trace_index=0,
            )
            persist_miprov2_intent_context(self._store, context)
            intent = EvaluationIntent(
                intent_id=intent_id,
                candidate=teacher_candidate,
                target_eval_config=context.eval_config,
                evaluation_binding=evaluation_binding,
                purpose="miprov2_bootstrap",
                run_id=request.run_id,
                step_index=request.step_index,
                expected_reward_policy_hash=(
                    request.run.record.reward_policy.identity_hash()
                    if request.run.record.reward_policy is not None
                    else None
                ),
            )
            return AdapterOutput(
                proposed_candidates=(teacher_candidate.record,),
                evaluation_intents=(intent,),
                budget_delta=BudgetDelta(
                    consumed={"bootstrap_generations": 1, "task_rows": 1}
                ),
                state_delta={
                    MIPROV2_STATE_KEY: plan.state.model_dump(mode="json"),
                    "miprov2_bootstrap_attempt": attempt.model_dump(
                        mode="json"
                    ),
                },
            )
        assert plan.evaluation is not None
        effect = plan.evaluation
        self._preflight_task_rows(
            request.budget,
            len(effect.task_batch_hashes),
        )
        if plan.state.resolved_eval_binding is None:
            raise ValueError("evaluation has no exact Eval Config binding")
        intent_id = (
            f"{request.run_id}:{effect.purpose}:{effect.identity_hash()}"
        )
        if effect.purpose == "miprov2_baseline":
            effect_kind = "baseline"
        elif effect.purpose == "miprov2_sample":
            effect_kind = "sample"
        else:
            effect_kind = "promotion"
        evaluation_binding = EvaluationBinding.model_validate(
            {
                **state.control.evaluation_binding.model_dump(mode="json"),
                "eval_config": effect.eval_config.model_dump(mode="json"),
            }
        )
        context = Miprov2IntentContext(
            control_identity_hash=state.control.identity_hash(),
            run_id=state.run_id,
            effect_kind=effect_kind,
            effect_identity_hash=effect.identity_hash(),
            intent_id=intent_id,
            candidate=candidate_reference(effect.candidate),
            task_batch_hashes=effect.task_batch_hashes,
            eval_config=effect.eval_config,
            eval_config_binding=plan.state.resolved_eval_binding,
            evaluation_binding=evaluation_binding,
            execution_policy=effect.execution_policy,
            reward_policy_hash=state.control.reward_policy_hash,
        )
        persist_miprov2_intent_context(self._store, context)
        intent = EvaluationIntent(
            intent_id=intent_id,
            candidate=candidate_reference(effect.candidate),
            target_eval_config=effect.eval_config,
            evaluation_binding=evaluation_binding,
            purpose=effect.purpose,
            run_id=request.run_id,
            step_index=request.step_index,
            expected_reward_policy_hash=(
                request.run.record.reward_policy.identity_hash()
                if request.run.record.reward_policy is not None
                else None
            ),
        )
        return AdapterOutput(
            proposed_candidates=(
                () if effect_kind == "baseline" else (effect.candidate,)
            ),
            evaluation_intents=(intent,),
            budget_delta=BudgetDelta(
                consumed={
                    "evaluations": 1,
                    "task_rows": len(effect.task_batch_hashes),
                }
            ),
            state_delta={
                MIPROV2_STATE_KEY: plan.state.model_dump(mode="json"),
                "miprov2_task_batch_hashes": list(effect.task_batch_hashes),
            },
        )

    @staticmethod
    def _preflight_task_rows(budget: BudgetState, row_count: int) -> None:

        budget.debit(BudgetDelta(consumed={"task_rows": row_count}))

    @staticmethod
    def _require_budget_agreement(
        request_budget: BudgetState,
        state: Miprov2State,
    ) -> None:

        for label in (
            "bootstrap_generations",
            "proposal_calls",
            "evaluations",
            "task_rows",
        ):
            consumed = state.effect_counts[label]
            ceiling = getattr(state.budget, label)
            expected_remaining = ceiling - consumed
            actual_consumed = request_budget.consumed.get(label, 0)
            actual_remaining = request_budget.remaining.get(label)
            if (
                actual_consumed != consumed
                or actual_remaining != expected_remaining
            ):
                raise ValueError(
                    "MIPROv2 request budget disagrees with durable state for "
                    f"{label!r}: expected consumed={consumed}, "
                    f"remaining={expected_remaining}; got "
                    f"consumed={actual_consumed}, "
                    f"remaining={actual_remaining}"
                )

    def _proposal_output(self, plan: Miprov2DriverPlan) -> AdapterOutput:
        native = plan.proposal_request
        assert native is not None
        generic = self._generic_request(plan.state, native)
        config = self._proposer_config.model_copy(
            update={"temperature": native.temperature}
        )
        drafts = self._proposal_executor.execute(
            config=config,
            request=generic,
            transport=self._transport,
            count=1,
        )
        if len(drafts) != 1:
            raise ValueError("MIPROv2 proposer must return exactly one draft")
        draft = drafts[0]
        evidence = {
            "proposal_request_hash": generic.identity_hash(),
            "request_evidence": draft.request_evidence.to_json(),
            "response_evidence": draft.response_evidence.to_json(),
            "usage": draft.usage.to_json(),
            "cost": draft.cost,
        }
        response = Miprov2ProposalResponse(
            request_hash=native.identity_hash,
            text=draft.template,
            failed=draft.failed,
            failure_detail=(
                draft.terminal_failure.message
                if draft.terminal_failure is not None
                else None
            ),
            evidence=evidence,
        )
        advanced = self._driver.fold_proposal(plan.state, response)
        budget_delta = BudgetDelta(consumed={"proposal_calls": 1})
        if advanced.proposal_state is None:
            raise ValueError("folded proposal state disappeared")
        if advanced.proposal_state.stage != MIPROV2_FAILED:
            return AdapterOutput(
                budget_delta=budget_delta,
                state_delta={
                    MIPROV2_STATE_KEY: advanced.model_dump(mode="json")
                },
            )
        detail = response.failure_detail or "instruction proposal failed"
        return self._terminal_failure_output(
            _terminalized(advanced, failure=detail),
            failure=TerminalFailure(
                code=MIPROV2_PROPOSAL_FAILED_CODE,
                message=detail,
                details={
                    "component_id": (
                        advanced.proposal_state.components[
                            advanced.proposal_state.component_index
                        ].component_id
                    ),
                    "proposal_request_hash": native.identity_hash,
                },
            ),
            budget_delta=budget_delta,
        )

    @staticmethod
    def _terminal_failure_output(
        state: Miprov2State,
        *,
        failure: TerminalFailure,
        budget_delta: BudgetDelta | None = None,
    ) -> AdapterOutput:

        if state.phase != MIPROV2_FAILED:
            raise ValueError("a terminal failure requires a failed state")
        return AdapterOutput(
            proposed_status=StepStatus.FAILED,
            terminal_failure=failure,
            budget_delta=budget_delta or BudgetDelta(),
            state_delta={MIPROV2_STATE_KEY: state.model_dump(mode="json")},
        )

    def fold_resolution(
        self,
        state: Miprov2State,
        resolution: IntentResolution,
    ) -> Miprov2State:

        context = load_miprov2_intent_context(self._store, resolution.intent)
        if context.control_identity_hash != state.control.identity_hash():
            raise ValueError("Intent Resolution belongs to another control")
        if resolution.outcome is IntentOutcome.REJECTED:
            return _terminalized(
                state, failure=_rejection_detail(context.intent_id, resolution)
            )
        if context.effect_kind == "bootstrap":
            if resolution.outcome is IntentOutcome.COMPLETED:
                result = self._evidence.resolve_bootstrap(resolution)
            else:
                result = self._evidence.resolve_bootstrap_failure(resolution)
            return self._driver.fold_bootstrap(state, result)
        if resolution.outcome is IntentOutcome.COMPLETED:
            resolved = self._evidence.resolve_evaluation(resolution)
        else:
            resolved = self._evidence.resolve_evaluation_failure(resolution)
        return self._driver.fold_evaluation(state, resolved)

    def _load_request_state(
        self,
        request: OptimizationStepRequest,
    ) -> tuple[Miprov2State, OptimizationStepResult | None]:
        ref = request.prior_step_result_ref
        if ref is None:
            if request.prior_state_ref is not None:
                raise ValueError(
                    "initial MIPROv2 request cannot cite prior state"
                )
            try:
                state = Miprov2State.model_validate(
                    request.pools[MIPROV2_STATE_KEY]
                )
            except KeyError:
                raise ValueError(
                    f"MIPROv2 requires {MIPROV2_STATE_KEY!r} in initial "
                    "request pools"
                ) from None
            return state, None
        if ref.schema_name != STEP_RESULT_SCHEMA:
            raise ValueError("prior result ref has the wrong schema")
        prior = OptimizationStepResult.model_validate(
            self._store.get(ref.reference)
        )
        if step_result_reference(prior).record_ref != ref:
            raise ValueError("prior result ref is not its exact record")
        if (
            prior.run_id != request.run_id
            or prior.step_index != request.step_index - 1
        ):
            raise ValueError(
                "prior result belongs to another run or step position"
            )
        state_ref = prior.state_ref
        if (
            state_ref is None
            or state_ref.schema_name != STATE_SNAPSHOT_SCHEMA
            or request.prior_state_ref != state_ref
        ):
            raise ValueError(
                "request does not cite the prior result's exact state snapshot"
            )
        snapshot = self._store.get(state_ref.reference)
        if not isinstance(snapshot, dict):
            raise ValueError("prior state snapshot must be an object")
        try:
            state = Miprov2State.model_validate(snapshot[MIPROV2_STATE_KEY])
        except KeyError:
            raise ValueError(
                "prior state snapshot has no MIPROv2 state"
            ) from None
        supplied = request.pools.get(MIPROV2_STATE_KEY)
        if (
            supplied is not None
            and Miprov2State.model_validate(supplied) != state
        ):
            raise ValueError(
                "request pools conflict with the prior state snapshot"
            )
        return state, prior

    def _fold_prior_resolutions(
        self,
        state: Miprov2State,
        prior: OptimizationStepResult,
    ) -> Miprov2State:
        for resolution in prior.resolved_intents:
            state = self.fold_resolution(state, resolution)
        return state

    def _require_transport_bindings(self, state: Miprov2State) -> None:
        current_executor_contract: ProposalExecutorDurabilityContract = (
            self._proposal_executor.durability_contract
        )
        if current_executor_contract != self._proposal_executor_contract:
            raise ValueError(
                "proposal executor policy changed after adapter construction"
            )
        if (
            current_executor_contract.recovery_policy
            is not ReplayPolicy.DURABLE_WORKFLOW
        ):
            raise ValueError(
                "proposal executor no longer provides durable-workflow "
                "recovery"
            )
        if (
            self._transport.durability_identity_hash
            != self._proposal_transport_durability_identity_hash
        ):
            raise ValueError(
                "proposal transport durability changed after adapter "
                "construction"
            )
        if (
            self._proposal_executor_policy_identity_hash
            != state.bindings.proposal_executor_policy_identity_hash
        ):
            raise ValueError(
                "proposal executor policy conflicts with MIPROv2 state"
            )
        if (
            self._proposal_transport_durability_identity_hash
            != state.bindings.proposal_transport_durability_identity_hash
        ):
            raise ValueError(
                "proposal transport durability conflicts with MIPROv2 state"
            )
        if (
            self._proposer_config.identity_hash()
            != state.bindings.prompt_route_identity_hash
        ):
            raise ValueError(
                "adapter proposer route conflicts with MIPROv2 state"
            )
        if (
            self._transport.execution_policy_hash
            != state.bindings.execution_policy_identity_hash
        ):
            raise ValueError(
                "proposer execution policy conflicts with MIPROv2 state"
            )
        if (
            self._transport.prompt_adapter_identity_hash
            != state.bindings.prompt_adapter_identity_hash
        ):
            raise ValueError(
                "proposer prompt adapter conflicts with MIPROv2 state"
            )

    @staticmethod
    def _generic_request(
        state: Miprov2State,
        native: Miprov2ProposalRequest,
    ) -> ProposalRequest:
        return ProposalRequest(
            proposal_mode=native.effect,
            request_ordinal=native.effect_ordinal,
            proposal_authority_identity_hash=(
                native.optimization_run_identity_hash
            ),
            mutation_field=state.control.mutation_field,
            base_candidate=state.control.base_candidate,
            context={
                "native_miprov2_request": native.model_dump(mode="json"),
                "proposal_prompt": native.prompt,
            },
        )


__all__ = [
    "MIPROV2_ADAPTER_KEY",
    "MIPROV2_BASELINE",
    "MIPROV2_BOOTSTRAP",
    "MIPROV2_COMPLETE",
    "MIPROV2_PROMOTION",
    "MIPROV2_PROPOSAL",
    "MIPROV2_SAMPLE",
    "MIPROV2_STATE_KEY",
    "Miprov2Adapter",
]
