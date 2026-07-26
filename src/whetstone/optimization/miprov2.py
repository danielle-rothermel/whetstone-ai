"""Canonical MIPROv2 adapter over the durable pure phase machine.

This file intentionally contains no optimizer algorithm of its own.  Exact
control, bootstrap, proposal, RNG, rendering, and Optuna behavior lives in the
typed MIPROv2 modules; the adapter only executes one already-planned proposal
effect or exposes one evaluation/bootstrap intent per harness step.
"""

from __future__ import annotations

from typing import Any, Protocol

from dr_store import BindingConflictError, ObjectStore
from pydantic import BaseModel, ConfigDict, StrictStr

from whetstone.graph.rollout import EvaluationRole
from whetstone.optimization.adapters import AdapterOutput
from whetstone.optimization.identity import (
    TypedRef,
    compute_identity_hash,
    require_full_hash,
)
from whetstone.optimization.miprov2_eval_config import (
    Miprov2EvalConfigResolver,
)
from whetstone.optimization.miprov2_evidence import (
    Miprov2BootstrapTraceProjection,
    Miprov2IntentContext,
    failure_bootstrap_result,
    load_miprov2_intent_context,
    persist_miprov2_intent_context,
    resolve_miprov2_bootstrap,
    resolve_miprov2_evaluation,
    resolve_miprov2_evaluation_failure,
)
from whetstone.optimization.miprov2_proposal import (
    Miprov2ProposalRequest,
    Miprov2ProposalResponse,
)
from whetstone.optimization.miprov2_runtime import (
    Miprov2Driver,
    Miprov2DriverPlan,
    Miprov2State,
)
from whetstone.optimization.proposer import (
    ProposalDraft,
    ProposalRequest,
    ProposerConfig,
    ProposerTransport,
)
from whetstone.optimization.schema import (
    STEP_RESULT_SCHEMA,
    BudgetDelta,
    BudgetState,
    EvaluationIntent,
    IntentOutcome,
    IntentResolution,
    OptimizationStepRequest,
    OptimizationStepResult,
    OutputContract,
    StepKind,
    StepMode,
    StepStatus,
    candidate_reference,
    step_result_reference,
)

MIPROV2_ADAPTER_KEY = "miprov2"
MIPROV2_STATE_KEY = "miprov2_state"
MIPROV2_BOOTSTRAP = "bootstrap_rollout"
MIPROV2_PROPOSAL = "proposal_model"
MIPROV2_BASELINE = "baseline_evaluation"
MIPROV2_SAMPLE = "sample_evaluation"
MIPROV2_PROMOTION = "promotion_evaluation"
MIPROV2_COMPLETE = "complete"
MIPROV2_PROPOSAL_EFFECT_CLAIM_SCHEMA = (
    "whetstone.miprov2_proposal_effect_claim"
)
MIPROV2_PROPOSAL_EFFECT_RESULT_SCHEMA = (
    "whetstone.miprov2_proposal_effect_result"
)
STATE_SNAPSHOT_SCHEMA = "whetstone.optimization_state_snapshot"


class Miprov2ProposalEffectClaim(BaseModel):
    """Durable accepted decision for one stable proposal request."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    native_request_identity_hash: StrictStr
    generic_request: ProposalRequest
    proposer_config_identity_hash: StrictStr
    execution_policy_identity_hash: StrictStr
    prompt_adapter_identity_hash: StrictStr
    transport_durability_identity_hash: StrictStr
    durability_scope_identity_hash: StrictStr

    def identity_hash(self) -> str:
        return compute_identity_hash(
            schema=MIPROV2_PROPOSAL_EFFECT_CLAIM_SCHEMA,
            schema_version=2,
            payload=self.model_dump(mode="json"),
        )

    def run_scope_identity_hash(self) -> str:
        """Stable logical request namespace shared by same-run retries."""

        if self.generic_request.run_id is None:
            raise ValueError(
                "MIPROv2 proposal effects require a run-scoped request"
            )
        return compute_identity_hash(
            schema="whetstone.miprov2_proposal_run_scope",
            schema_version=1,
            payload={
                "run_id": self.generic_request.run_id,
                "native_request_identity_hash": (
                    self.native_request_identity_hash
                ),
            },
        )


class Miprov2ProposalEffectResult(BaseModel):
    """Persisted provider response folded after an accepted effect claim."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    claim_identity_hash: StrictStr
    draft: ProposalDraft


class Miprov2ProposalRecoveryRequired(RuntimeError):
    """An accepted provider effect has no safely replayable result."""


class Miprov2ProposalEffectExecutor(Protocol):
    """Durable authority for one retry-disabled proposal provider effect."""

    @property
    def durability_scope_identity_hash(self) -> str: ...

    def execute(
        self,
        *,
        config: ProposerConfig,
        request: ProposalRequest,
        transport: ProposerTransport,
        count: int,
    ) -> tuple[ProposalDraft, ...]: ...


class Miprov2ProposalEffectStore:
    """Cross-restart ownership around a durable provider-effect executor."""

    def __init__(self, store: ObjectStore) -> None:
        self._store = store

    @staticmethod
    def _claim_key(run_scope_identity_hash: str) -> str:
        return f"whetstone.miprov2_proposal_claim:{run_scope_identity_hash}"

    @staticmethod
    def _result_key(identity_hash: str) -> str:
        return f"whetstone.miprov2_proposal_result:{identity_hash}"

    def execute(
        self,
        *,
        native_request_identity_hash: str,
        config: ProposerConfig,
        request: ProposalRequest,
        transport: ProposerTransport,
        executor: Miprov2ProposalEffectExecutor,
    ) -> ProposalDraft:
        claim = Miprov2ProposalEffectClaim(
            native_request_identity_hash=native_request_identity_hash,
            generic_request=request,
            proposer_config_identity_hash=config.identity_hash(),
            execution_policy_identity_hash=transport.execution_policy_hash,
            prompt_adapter_identity_hash=(
                transport.prompt_adapter_identity_hash
            ),
            transport_durability_identity_hash=(
                self._transport_durability_identity_hash(transport)
            ),
            durability_scope_identity_hash=(
                executor.durability_scope_identity_hash
            ),
        )
        identity = claim.identity_hash()
        require_full_hash(
            claim.durability_scope_identity_hash,
            field="durability_scope_identity_hash",
        )
        claim_key = self._claim_key(claim.run_scope_identity_hash())
        accepted_ref = self._store.resolve(claim_key)
        if accepted_ref is not None:
            accepted = Miprov2ProposalEffectClaim.model_validate(
                self._store.get(accepted_ref)
            )
            if accepted != claim:
                raise Miprov2ProposalRecoveryRequired(
                    "proposal effect belongs to another durability scope"
                )
        completed_ref = self._store.resolve(self._result_key(identity))
        completed_result = None
        if completed_ref is not None:
            completed_result = Miprov2ProposalEffectResult.model_validate(
                self._store.get(completed_ref)
            )
            if completed_result.claim_identity_hash != identity:
                raise ValueError("persisted proposal result has wrong claim")
        if accepted_ref is None:
            claim_ref, _ = self._store.put(
                MIPROV2_PROPOSAL_EFFECT_CLAIM_SCHEMA,
                claim.model_dump(mode="json"),
            )
            try:
                self._store.bind(claim_key, claim_ref)
            except BindingConflictError as exc:
                winner = self._store.resolve(claim_key)
                assert winner is not None
                accepted = Miprov2ProposalEffectClaim.model_validate(
                    self._store.get(winner)
                )
                if accepted != claim:
                    raise Miprov2ProposalRecoveryRequired(
                        "proposal effect claim conflicts with durable owner"
                    ) from exc
        drafts = executor.execute(
            config=config,
            request=request,
            transport=transport,
            count=1,
        )
        if len(drafts) != 1:
            raise ValueError(
                "MIPROv2 proposer transport must return exactly one draft"
            )
        result = Miprov2ProposalEffectResult(
            claim_identity_hash=identity,
            draft=drafts[0],
        )
        if completed_result is not None:
            if completed_result != result:
                raise ValueError(
                    "replayed proposal effect diverges from persisted result"
                )
            return completed_result.draft
        result_ref, _ = self._store.put(
            MIPROV2_PROPOSAL_EFFECT_RESULT_SCHEMA,
            result.model_dump(mode="json"),
        )
        try:
            self._store.bind(self._result_key(identity), result_ref)
        except BindingConflictError as exc:
            winner = self._store.resolve(self._result_key(identity))
            assert winner is not None
            persisted = Miprov2ProposalEffectResult.model_validate(
                self._store.get(winner)
            )
            if persisted != result:
                raise ValueError(
                    "divergent proposal result was rejected"
                ) from exc
        return result.draft

    @staticmethod
    def _transport_durability_identity_hash(
        transport: ProposerTransport,
    ) -> str:
        identity = getattr(transport, "durability_identity_hash", None)
        if identity is not None:
            require_full_hash(
                identity,
                field="transport_durability_identity_hash",
            )
            return identity
        return compute_identity_hash(
            schema="whetstone.miprov2_legacy_proposer_durability",
            schema_version=1,
            payload={
                "physical_attempt_boundary": "whole_draft_call",
                "crash_safety": "at_least_once",
                "execution_policy_identity_hash": (
                    transport.execution_policy_hash
                ),
                "prompt_adapter_identity_hash": (
                    transport.prompt_adapter_identity_hash
                ),
            },
        )


class Miprov2Adapter:
    """Execute at most one external effect selected by :class:`Miprov2Driver`.

    Proposal calls are the adapter's only direct side effect.  Bootstrap and
    task evaluation remain Evaluation Intents owned by the harness.  Their
    normalized typed results are folded with ``Miprov2Driver.fold_bootstrap``
    and ``Miprov2Driver.fold_evaluation`` before the next adapter request.
    """

    def __init__(
        self,
        *,
        store: ObjectStore,
        proposer_config: ProposerConfig,
        transport: ProposerTransport,
        eval_config_resolver: Miprov2EvalConfigResolver,
        proposal_effect_executor: Miprov2ProposalEffectExecutor,
        driver: Miprov2Driver | None = None,
    ) -> None:
        self._proposer_config = proposer_config
        self._transport = transport
        self._store = store
        self._effect_store = Miprov2ProposalEffectStore(store)
        self._eval_config_resolver = eval_config_resolver
        self._proposal_effect_executor = proposal_effect_executor
        self._driver = driver or Miprov2Driver()

    @property
    def key(self) -> str:
        return MIPROV2_ADAPTER_KEY

    @property
    def mode(self) -> StepMode:
        return StepMode.PROPOSAL_ONLY

    @property
    def proposer_config(self) -> ProposerConfig:
        return self._proposer_config

    def build_step_request(
        self,
        *,
        step_index: int,
        initial_state: Miprov2State | None = None,
        initial_budget: BudgetState | None = None,
        prior_result: OptimizationStepResult | None = None,
        prior_result_ref: TypedRef | None = None,
    ) -> OptimizationStepRequest:
        """Build the exact next harness request from durable prior state."""

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
            state = initial_state
            budget = initial_budget
            pools = {MIPROV2_STATE_KEY: state.model_dump(mode="json")}
            prior_state_ref = None
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
            if step_result_reference(prior_result) != prior_result_ref:
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
        preview = self._driver.plan(state)
        returned_count = 1 if preview.kind == MIPROV2_COMPLETE else 0
        return OptimizationStepRequest(
            run_id=state.run_id,
            step_id=f"{state.run_id}:miprov2:{step_index}",
            optimizer_config_hash=state.control.identity_hash(),
            adapter_key=self.key,
            mode=self.mode,
            kind=StepKind.PROPOSAL,
            step_index=step_index,
            prior_step_result_ref=prior_result_ref,
            prior_state_ref=prior_state_ref,
            pools=pools,
            budget=budget,
            output_contract=OutputContract(
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
        if request.run_id != state.run_id:
            raise ValueError("request run_id conflicts with MIPROv2 state")
        state.control.require_identity_hash(request.optimizer_config_hash)
        if prior is not None:
            state = self._fold_prior_resolutions(state, prior)
        self._require_budget_agreement(request.budget, state)
        plan = self._driver.plan(state)
        if plan.kind == "eval_config_binding":
            assert plan.eval_config_binding is not None
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
            assert plan.bootstrap_rollout is not None
            assert plan.state.resolved_eval_binding is not None
            assert plan.state.pending_bootstrap_candidate is not None
            attempt = plan.bootstrap_rollout
            teacher_candidate = plan.state.pending_bootstrap_candidate
            intent_id = (
                f"{request.run_id}:miprov2:bootstrap:{attempt.identity_hash()}"
            )
            try:
                labeled_task = next(
                    item
                    for item in state.labeled_trainset
                    if item.source_task_identity == attempt.task_identity
                )
            except (KeyError, StopIteration) as exc:
                raise ValueError(
                    "bootstrap task has no exact component projection"
                ) from exc
            trace_components: list[Miprov2BootstrapTraceProjection] = []
            for component_id in state.control.component_ids:
                try:
                    trace_inputs = labeled_task.inputs_by_component[
                        component_id
                    ]
                    labeled_outputs = labeled_task.outputs_by_component[
                        component_id
                    ]
                except KeyError as exc:
                    raise ValueError(
                        "bootstrap task has no exact component projection"
                    ) from exc
                if len(labeled_outputs) != 1:
                    raise ValueError(
                        "bootstrap components require exactly one generated "
                        "output field"
                    )
                trace_components.append(
                    Miprov2BootstrapTraceProjection(
                        component_id=component_id,
                        inputs=trace_inputs,
                        output_field=next(iter(labeled_outputs)),
                    )
                )
            context = Miprov2IntentContext(
                control_identity_hash=state.control.identity_hash(),
                run_id=state.run_id,
                effect_kind="bootstrap",
                effect_identity_hash=attempt.identity_hash(),
                intent_id=intent_id,
                candidate=teacher_candidate,
                task_batch_identities=(attempt.task_identity,),
                eval_config=plan.state.resolved_eval_binding.eval_config,
                eval_config_binding=plan.state.resolved_eval_binding,
                execution_policy=(
                    plan.state.resolved_eval_binding.request.execution_policy
                ),
                reward_policy_hash=state.control.reward_policy_hash,
                bootstrap_attempt=attempt,
                trace_components=tuple(trace_components),
            )
            context_ref = persist_miprov2_intent_context(self._store, context)
            intent = EvaluationIntent(
                intent_id=intent_id,
                candidate=teacher_candidate,
                target_eval_config=context.eval_config,
                context_role=EvaluationRole.INTERNAL,
                context_policy_ref=context_ref.content_hash,
                purpose="miprov2_bootstrap",
                run_id=request.run_id,
                step_index=request.step_index,
            )
            return AdapterOutput(
                proposed_candidates=(teacher_candidate.record,),
                evaluation_intents=(intent,),
                budget_delta=BudgetDelta(
                    consumed={"bootstrap_rollouts": 1, "task_rows": 1}
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
        context = Miprov2IntentContext(
            control_identity_hash=state.control.identity_hash(),
            run_id=state.run_id,
            effect_kind=effect_kind,
            effect_identity_hash=effect.identity_hash(),
            intent_id=intent_id,
            candidate=candidate_reference(effect.candidate),
            task_batch_identities=effect.task_batch_identities,
            eval_config=effect.eval_config,
            eval_config_binding=plan.state.resolved_eval_binding,
            execution_policy=effect.execution_policy,
            reward_policy_hash=state.control.reward_policy_hash,
        )
        context_ref = persist_miprov2_intent_context(self._store, context)
        intent = EvaluationIntent(
            intent_id=intent_id,
            candidate=candidate_reference(effect.candidate),
            target_eval_config=effect.eval_config,
            context_role=EvaluationRole.INTERNAL,
            context_policy_ref=context_ref.content_hash,
            purpose=effect.purpose,
            run_id=request.run_id,
            step_index=request.step_index,
        )
        return AdapterOutput(
            proposed_candidates=(effect.candidate,),
            evaluation_intents=(intent,),
            budget_delta=BudgetDelta(
                consumed={
                    "evaluations": 1,
                    "task_rows": len(effect.task_batch_identities),
                }
            ),
            state_delta={
                MIPROV2_STATE_KEY: plan.state.model_dump(mode="json"),
                "miprov2_task_batch_identities": list(
                    effect.task_batch_identities
                ),
            },
        )

    @staticmethod
    def _require_budget_agreement(
        request_budget: BudgetState,
        state: Miprov2State,
    ) -> None:
        """Require the harness budget to project the durable effect journal."""

        for label in (
            "bootstrap_rollouts",
            "proposal_calls",
            "evaluations",
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
        self._require_transport_bindings(plan.state)
        generic = self._generic_request(plan.state, native)
        config = self._proposer_config.model_copy(
            update={"temperature": native.temperature}
        )
        draft = self._effect_store.execute(
            native_request_identity_hash=native.identity_hash,
            config=config,
            request=generic,
            transport=self._transport,
            executor=self._proposal_effect_executor,
        )
        evidence = {
            "proposal_request_identity_hash": generic.identity_hash(),
            "request_evidence": draft.request_evidence,
            "response_evidence": draft.response_evidence,
            "usage": draft.usage,
            "cost": draft.cost,
        }
        response = Miprov2ProposalResponse(
            request_identity_hash=native.identity_hash,
            text=draft.template,
            failed=draft.failed,
            failure_detail=draft.failure_detail,
            evidence=evidence,
        )
        advanced = self._driver.fold_proposal(plan.state, response)
        return AdapterOutput(
            budget_delta=BudgetDelta(consumed={"proposal_calls": 1}),
            state_delta={MIPROV2_STATE_KEY: advanced.model_dump(mode="json")},
        )

    def fold_resolution(
        self,
        state: Miprov2State,
        resolution: IntentResolution,
    ) -> Miprov2State:
        """Production bridge from exact harness evidence to pure state."""

        context = load_miprov2_intent_context(self._store, resolution.intent)
        if context.control_identity_hash != state.control.identity_hash():
            raise ValueError("Intent Resolution belongs to another control")
        if context.effect_kind == "bootstrap":
            if resolution.outcome is IntentOutcome.COMPLETED:
                result = resolve_miprov2_bootstrap(self._store, resolution)
            else:
                result = failure_bootstrap_result(
                    context=context,
                    resolution=resolution,
                )
            return self._driver.fold_bootstrap(state, result)
        if resolution.outcome is IntentOutcome.COMPLETED:
            resolved = resolve_miprov2_evaluation(self._store, resolution)
        else:
            resolved = resolve_miprov2_evaluation_failure(
                self._store, resolution
            )
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
        if step_result_reference(prior) != ref:
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
        base_template = ""
        if native.component_index is not None:
            base_template = state.proposal_components[
                native.component_index
            ].template
        return ProposalRequest(
            proposal_mode=native.effect,
            request_ordinal=native.effect_ordinal,
            base_ref=state.control.base_candidate.record.base_ref,
            base_template=base_template,
            run_id=state.run_id,
            step_index=native.effect_ordinal,
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
    "Miprov2Driver",
    "Miprov2ProposalEffectExecutor",
]
