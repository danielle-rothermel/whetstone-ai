from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
from typing import Protocol
from uuid import uuid4

from dr_store import BindingConflictError, BindStatus, ObjectStore

from whetstone.coordination.eval_service import EvalExecutionContext
from whetstone.core.effects.authority import (
    AcquireOutcome,
    AcquireResult,
    EffectAuthority,
    EffectLease,
    EffectRequest,
    ReplayPolicy,
)
from whetstone.core.identity import (
    TerminalFailure,
    TypedRef,
    compute_identity_hash,
    compute_prefixed_identity_key,
    typed_ref_for_record,
)
from whetstone.core.roles import EvalRole
from whetstone.experiment.candidate import (
    CandidateRef,
    candidate_reference,
)
from whetstone.optim.adapters import (
    AdapterCheckpoint,
    AdapterOutput,
    AdapterRegistry,
    OptimizerAdapter,
)
from whetstone.optim.contracts import (
    INTENT_RESOLUTION_SCHEMA,
    OPTIM_RESULT_SCHEMA,
    BudgetState,
    OptimEvalRequest,
    IntentResolution,
    OptimProposal,
    OptimResult,
    OptimRunRef,
    OptimStepRequest,
    OptimStepResult,
    OptimStepResultRef,
    StepMode,
    StepStatus,
    optimization_result_reference,
    step_request_reference,
    step_result_reference,
)
from whetstone.optim.proposal.mutation import (
    diff_check,
    validate_candidate_template,
)
from whetstone.optim.run_store import (
    ADAPTER_CHECKPOINT_SCHEMA,
    HISTORY_SNAPSHOT_SCHEMA,
    STATE_SNAPSHOT_SCHEMA,
    OptimResultConflictError,
    OptimRunConflictError,
    OptimRunStore,
    StepResultConflictError,
    _ResolvedAdapter,
)
from whetstone.optim.tools.contracts import (
    RuntimeToolHandle,
    ToolCapacityBinding,
    ToolCapacityScope,
    ToolConfig,
    tool_capacity_binding,
)
from whetstone.optim.tools.facade import ToolCallStore
from whetstone.optim.tools.issued import (
    IssuedToolCallConflictError,
    _IssuedToolCallLedger,
)

__all__ = [
    "ADAPTER_CHECKPOINT_SCHEMA",
    "EffectBusyError",
    "EffectRecoveryRequiredError",
    "EffectRequestConflictError",
    "EvalService",
    "IssuedToolCallConflictError",
    "OptimHarness",
    "OptimResultConflictError",
    "OptimRunConflictError",
    "StepResultConflictError",
    "ToolExecutor",
]

ADAPTER_EFFECT_SCHEMA = "whetstone.optim_adapter_effect"
ADAPTER_EFFECT_SCHEMA_VERSION = 1
ADAPTER_EFFECT_KEY_SCHEMA = "whetstone.optim_adapter_effect_key"
ADAPTER_EFFECT_KEY_SCHEMA_VERSION = 1
ADAPTER_EFFECT_KEY_PREFIX = "whetstone.optim_adapter:"
INTENT_EFFECT_SCHEMA = "whetstone.optim_intent_effect"
INTENT_EFFECT_SCHEMA_VERSION = 1
INTENT_EFFECT_KEY_SCHEMA = "whetstone.optim_intent_effect_key"
INTENT_EFFECT_KEY_SCHEMA_VERSION = 2
INTENT_EFFECT_KEY_PREFIX = "whetstone.optim_intent:"


class EffectBusyError(RuntimeError):
    def __init__(
        self, *, semantic_key: str, busy_expires_at: datetime
    ) -> None:
        self.semantic_key = semantic_key
        self.busy_expires_at = busy_expires_at
        super().__init__(
            f"effect {semantic_key!r} is busy until "
            f"{busy_expires_at.isoformat(timespec='microseconds')}"
        )


class EffectRequestConflictError(RuntimeError):
    def __init__(self, *, semantic_key: str) -> None:
        self.semantic_key = semantic_key
        super().__init__(
            f"effect key {semantic_key!r} is bound to another exact request"
        )


class EffectRecoveryRequiredError(RuntimeError):
    def __init__(self, *, semantic_key: str, failure: TerminalFailure) -> None:
        self.semantic_key = semantic_key
        self.failure = failure
        super().__init__(failure.message)


class EvalService(Protocol):
    @property
    def replay_policy(self) -> ReplayPolicy: ...

    def resolve_optim_eval_request(
        self, optim_eval_request: OptimEvalRequest
    ) -> IntentResolution: ...

    def validate_resolution_graph(
        self, resolution: IntentResolution
    ) -> None: ...


class ToolExecutor(Protocol):
    def runtime_handle(
        self,
        config: ToolConfig,
        store: ToolCallStore,
        binding: ToolCapacityBinding,
    ) -> RuntimeToolHandle: ...


class OptimHarness(OptimRunStore):
    def __init__(
        self,
        *,
        store: ObjectStore,
        adapter_registry: AdapterRegistry,
        tool_store: ToolCallStore,
        effect_authority: EffectAuthority,
        owner_id: str,
        adapter_replay_policy: ReplayPolicy,
        lease_duration: timedelta,
        evaluation_service: EvalService | None = None,
        tool_executor: ToolExecutor | None = None,
    ) -> None:
        if type(adapter_replay_policy) is not ReplayPolicy:
            raise TypeError(
                "adapter_replay_policy must be an actual ReplayPolicy enum"
            )
        self._store = store
        self._adapter_registry = adapter_registry
        self._tool_store = tool_store
        self._effect_authority = effect_authority
        self._owner_id = owner_id
        self._adapter_replay_policy = adapter_replay_policy
        self._lease_duration = lease_duration
        self._evaluation_service = evaluation_service
        self._tool_executor = tool_executor
        self._bound_run: OptimRunRef | None = None
        if not owner_id:
            raise ValueError("owner_id must be non-empty")
        if lease_duration <= timedelta(0):
            raise ValueError("lease_duration must be positive")
        self._evaluation_replay_policy: ReplayPolicy | None = None
        if (
            evaluation_service is not None
            and evaluation_service.replay_policy
            not in {ReplayPolicy.IDEMPOTENT, ReplayPolicy.DURABLE_WORKFLOW}
        ):
            raise ValueError(
                "EvalService replay_policy must be idempotent or "
                "durable_workflow"
            )
        if evaluation_service is not None:
            self._evaluation_replay_policy = evaluation_service.replay_policy
        self._last_deferred_platform_intents: tuple[OptimEvalRequest, ...] = ()

    @property
    def last_deferred_platform_intents(self) -> tuple[OptimEvalRequest, ...]:
        return self._last_deferred_platform_intents

    def run_step(
        self,
        request: OptimStepRequest,
        *,
        eval_context: EvalExecutionContext | None = None,
    ) -> tuple[OptimStepResult, TypedRef]:
        validated_request = OptimStepRequest.model_validate(
            request.model_dump(mode="json")
        )
        self._validate_bound_run(validated_request)
        request = validated_request
        self._last_deferred_platform_intents = ()
        effective_eval_context = eval_context or EvalExecutionContext()
        for candidate in request.candidates:
            validate_candidate_template(candidate=candidate, run=request.run)
        exact_request = step_request_reference(request)
        self._validate_prior_binding(request)

        existing_ref = self._resolve_result_binding(
            request.run_id, request.step_index
        )
        if existing_ref is not None:
            existing = self._load_result(existing_ref)
            if (
                existing.run_id != request.run_id
                or existing.step_index != request.step_index
            ):
                raise ValueError(
                    "bound Step Result belongs to another run or position"
                )
            if existing.request == exact_request:
                return existing, existing_ref
            raise StepResultConflictError(
                run_id=request.run_id,
                step_index=request.step_index,
                existing=existing_ref,
                requested=exact_request.record_ref,
            )

        resolved_adapter = self._resolve_compatible_adapter(
            request.adapter_key,
            expected_mode=request.mode,
        )
        request_ref = self._put_request(request)
        if request_ref != exact_request.record_ref:
            raise ValueError("persisted request ref failed content validation")
        for candidate in request.candidates:
            self._persist_candidate(candidate)

        ledger = (
            _IssuedToolCallLedger(
                store=self._store,
                tool_store=self._tool_store,
                request=exact_request,
            )
            if request.mode is StepMode.TOOL_USING
            else None
        )
        guarded_handles = (
            self._prepare_tool_handles(
                request=request,
                request_ref=request_ref,
                ledger=ledger,
            )
            if request.mode is StepMode.TOOL_USING
            else ()
        )
        if request.mode is StepMode.PURE:
            output = self._invoke_pure(request, resolved_adapter.adapter)
        else:
            output = self._effectful_output(
                request,
                request_ref,
                resolved_adapter,
                ledger=ledger,
                guarded_handles=guarded_handles,
            )

        self._validate_output(request, output)
        self._validate_output_candidates(request, output)
        self._validate_output_intents(request, output)
        if ledger is not None:
            tool_evidence = ledger.evidence()
            expected_delta = ledger.budget_delta(
                output.budget_delta,
                issued_count=len(tool_evidence),
            )
            if output.budget_delta != expected_delta:
                raise ValueError(
                    "Adapter checkpoint tool_calls budget does not match the "
                    "durable issued-call ledger"
                )
        else:
            tool_evidence = ()
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
                request,
                output,
                proposed_refs,
                accepted_refs,
                eval_context=effective_eval_context,
            )
        else:
            resolutions = ()

        result = OptimStepResult(
            request=exact_request,
            proposed_candidates=proposed_refs,
            accepted_candidates=accepted_refs,
            resolved_intents=resolutions,
            search_evidence=output.search_evidence,
            tool_evidence=tool_evidence,
            state_ref=self._persist_snapshot(
                STATE_SNAPSHOT_SCHEMA, output.state_delta
            ),
            history_ref=self._persist_snapshot(
                HISTORY_SNAPSHOT_SCHEMA, output.history_delta
            ),
            budget_delta=output.budget_delta,
            budget=budget,
            status=output.proposed_status,
            terminal_failure=output.terminal_failure,
            seed_retained=output.seed_retained,
            retained_candidate_ref=(
                None
                if output.retained_candidate is None
                else self._persist_candidate(output.retained_candidate)
            ),
        )
        result_ref = self._put_result(result)
        if result_ref != step_result_reference(result).record_ref:
            raise ValueError("persisted Step Result ref failed validation")
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
        request: OptimStepRequest,
        adapter: OptimizerAdapter,
    ) -> AdapterOutput:
        raw_output = adapter.invoke(request, ())
        output = AdapterOutput.model_validate(
            raw_output.model_dump(mode="json")
        )
        if output.optim_eval_requests:
            raise ValueError("a pure Step emits no measurement requests")
        return output

    def _effectful_output(
        self,
        request: OptimStepRequest,
        request_ref: TypedRef,
        resolved_adapter: _ResolvedAdapter,
        ledger: _IssuedToolCallLedger | None,
        guarded_handles: tuple[RuntimeToolHandle, ...],
    ) -> AdapterOutput:
        effect_request = self._adapter_effect_request(
            request,
            request_ref,
            resolved_adapter.replay_policy,
        )
        acquisition = self._effect_authority.acquire(
            effect_request,
            owner_id=self._owner_id,
            attempt_id=uuid4().hex,
            lease_duration=self._lease_duration,
        )
        if acquisition.outcome in {
            AcquireOutcome.SUCCEEDED,
            AcquireOutcome.FAILED,
        }:
            terminal = acquisition.terminal
            if terminal is None or terminal.result_ref is None:
                raise RuntimeError(
                    "terminal Adapter effect has no exact checkpoint ref"
                )
            checkpoint = self._load_checkpoint(terminal.result_ref)
            self._validate_checkpoint(
                checkpoint,
                request=request,
                request_ref=request_ref,
                adapter_key=request.adapter_key,
            )
            if acquisition.outcome is AcquireOutcome.FAILED:
                if (
                    checkpoint.output.terminal_failure is None
                    or terminal.failure != checkpoint.output.terminal_failure
                ):
                    raise ValueError(
                        "failed Adapter effect does not match its exact "
                        "checkpoint failure"
                    )
            elif checkpoint.output.terminal_failure is not None:
                raise ValueError(
                    "successful Adapter effect references a failed checkpoint"
                )
            return checkpoint.output
        lease = self._acquired_lease(acquisition)
        with self._effect_authority.maintain(
            lease, lease_duration=self._lease_duration
        ) as maintenance:
            output, checkpoint_ref = self._invoke_and_persist_adapter(
                request=request,
                request_ref=request_ref,
                adapter=resolved_adapter.adapter,
                ledger=ledger,
                guarded_handles=guarded_handles,
            )
            if output.terminal_failure is None:
                maintenance.succeed(result_ref=checkpoint_ref)
            else:
                maintenance.fail(
                    result_ref=checkpoint_ref,
                    failure=output.terminal_failure,
                )
        return output

    def _invoke_and_persist_adapter(
        self,
        *,
        request: OptimStepRequest,
        request_ref: TypedRef,
        adapter: OptimizerAdapter,
        ledger: _IssuedToolCallLedger | None,
        guarded_handles: tuple[RuntimeToolHandle, ...],
    ) -> tuple[AdapterOutput, TypedRef]:
        if request.mode is StepMode.PROPOSAL_ONLY:
            output = adapter.invoke(request, ())
        elif request.mode is StepMode.TOOL_USING:
            if ledger is None:
                raise RuntimeError("tool-using Step has no issued-call ledger")
            output = adapter.invoke(
                request,
                guarded_handles,
            )
            ledger.validate_replay_complete()
            if output.optim_eval_requests:
                raise ValueError(
                    "tool-using Steps carry measurement in Tool Results"
                )
            evidence = ledger.evidence()
            output = output.model_copy(
                update={
                    "budget_delta": ledger.budget_delta(
                        output.budget_delta,
                        issued_count=len(evidence),
                    )
                }
            )
        else:
            raise ValueError(f"unsupported effectful mode {request.mode!r}")
        output = AdapterOutput.model_validate(output.model_dump(mode="json"))
        self._validate_output(request, output)
        self._validate_output_candidates(request, output)
        self._validate_output_intents(request, output)

        checkpoint = AdapterCheckpoint(
            request_ref=request_ref,
            adapter_key=request.adapter_key,
            output=output,
        )
        checkpoint_ref = self._put(
            ADAPTER_CHECKPOINT_SCHEMA, checkpoint.record_content()
        )
        expected = typed_ref_for_record(
            ADAPTER_CHECKPOINT_SCHEMA, checkpoint.record_content()
        )
        if checkpoint_ref != expected:
            raise ValueError(
                "persisted Adapter Checkpoint ref failed validation"
            )
        return output, checkpoint_ref

    def _prepare_tool_handles(
        self,
        *,
        request: OptimStepRequest,
        request_ref: TypedRef,
        ledger: _IssuedToolCallLedger | None,
    ) -> tuple[RuntimeToolHandle, ...]:
        if self._tool_executor is None:
            raise ValueError("tool-using Step requires a ToolExecutor")
        if ledger is None:
            raise RuntimeError("tool-using Step has no issued-call ledger")
        guarded: list[RuntimeToolHandle] = []
        for cfg in request.tool_configs:
            binding = self._tool_capacity_binding(
                request=request,
                request_ref=request_ref,
                config=cfg.record,
            )
            handle = self._tool_executor.runtime_handle(
                cfg.record,
                self._tool_store,
                binding,
            )
            if handle.config != cfg.record:
                raise ValueError(
                    "ToolExecutor returned a Runtime Tool Handle for another "
                    "exact Tool Config"
                )
            if handle.binding != binding:
                raise ValueError(
                    "ToolExecutor returned a Runtime Tool Handle with another "
                    "capacity binding"
                )
            guarded.append(
                RuntimeToolHandle(
                    cfg.record,
                    binding,
                    lambda call, handle=handle: ledger.issue(call, handle),
                )
            )
        return tuple(guarded)

    @staticmethod
    def _tool_capacity_binding(
        *,
        request: OptimStepRequest,
        request_ref: TypedRef,
        config: ToolConfig,
    ) -> ToolCapacityBinding:
        scope = config.capacity.scope
        if scope is ToolCapacityScope.GLOBAL:
            subject_ref = None
        elif scope is ToolCapacityScope.RUN:
            subject_ref = request.run.record_ref
        else:
            subject_ref = request_ref
        return tool_capacity_binding(scope, subject_ref)

    def _adapter_effect_request(
        self,
        request: OptimStepRequest,
        request_ref: TypedRef,
        replay_policy: ReplayPolicy,
    ) -> EffectRequest:

        payload = {
            "step_request_ref": request_ref.model_dump(mode="json"),
            "adapter_key": str(request.adapter_key),
        }
        return EffectRequest(
            semantic_key=compute_prefixed_identity_key(
                schema=ADAPTER_EFFECT_KEY_SCHEMA,
                schema_version=ADAPTER_EFFECT_KEY_SCHEMA_VERSION,
                prefix=ADAPTER_EFFECT_KEY_PREFIX,
                payload=payload,
            ),
            request_hash=compute_identity_hash(
                schema=ADAPTER_EFFECT_SCHEMA,
                schema_version=ADAPTER_EFFECT_SCHEMA_VERSION,
                payload=payload,
            ),
            replay_policy=replay_policy,
        )

    def _acquired_lease(self, acquisition: AcquireResult) -> EffectLease:
        request = acquisition.request
        semantic_key = str(request.semantic_key)
        if acquisition.outcome is AcquireOutcome.BUSY:
            if acquisition.busy_expires_at is None:
                raise RuntimeError("busy effect has no expiration")
            raise EffectBusyError(
                semantic_key=semantic_key,
                busy_expires_at=acquisition.busy_expires_at,
            )
        if acquisition.outcome is AcquireOutcome.REQUEST_CONFLICT:
            raise EffectRequestConflictError(semantic_key=semantic_key)
        if acquisition.outcome is AcquireOutcome.RECOVERY_REQUIRED:
            terminal = acquisition.terminal
            if terminal is None or terminal.failure is None:
                raise RuntimeError(
                    "recovery-required effect has no terminal failure"
                )
            raise EffectRecoveryRequiredError(
                semantic_key=semantic_key, failure=terminal.failure
            )
        if (
            acquisition.outcome is not AcquireOutcome.ACQUIRED
            or acquisition.lease is None
        ):
            raise RuntimeError("unrecognized Effect acquisition outcome")
        return acquisition.lease

    @classmethod
    def _validate_checkpoint(
        cls,
        checkpoint: AdapterCheckpoint,
        *,
        request: OptimStepRequest,
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
        cls._validate_output(request, checkpoint.output)
        cls._validate_output_candidates(request, checkpoint.output)
        cls._validate_output_intents(request, checkpoint.output)

    @staticmethod
    def _validate_output(
        request: OptimStepRequest, output: AdapterOutput
    ) -> None:
        request_ids = [
            str(optim_eval_request.eval_request.request_id)
            for optim_eval_request in output.optim_eval_requests
        ]
        if len(set(request_ids)) != len(request_ids):
            raise ValueError(
                "Optim Eval Request IDs must be unique within a Step"
            )
        contract = request.step_output_contract
        if output.seed_retained:
            if contract.terminal_proposal_count is None:
                raise ValueError(
                    "only a Step whose output contract sets "
                    "terminal_proposal_count -- a search-dependent terminal "
                    "cardinality -- may retain the seed; this contract binds "
                    "terminal cardinality unconditionally"
                )
            seed_ref = request.run.record.initial_candidate_ref
            if seed_ref is None:
                raise ValueError(
                    "a seed-retaining Step requires the run to name its "
                    "initial_candidate_ref"
                )
            retained = output.retained_candidate
            if retained is None or candidate_reference(retained) != seed_ref:
                raise ValueError(
                    "a seed-retaining Step must retain the exact run initial "
                    "candidate"
                )
        expected_count = (
            0
            if output.seed_retained
            else contract.accepted_count_for(output.proposed_status)
        )
        if len(output.accepted_candidates) != expected_count:
            raise ValueError(
                "adapter violated returned proposal cardinality: expected "
                f"{expected_count}, got "
                f"{len(output.accepted_candidates)}"
            )
        if contract.require_distinct_bases:
            bases = [
                candidate.base_ref for candidate in output.proposed_candidates
            ]
            if len(bases) != len(set(bases)):
                raise ValueError(
                    "adapter violated the distinct-base output contract"
                )
        accepted = Counter(
            candidate_reference(candidate).identity_hash
            for candidate in output.accepted_candidates
        )
        proposed = Counter(
            candidate_reference(candidate).identity_hash
            for candidate in output.proposed_candidates
        )
        missing = accepted - proposed
        if missing:
            raise ValueError(
                "accepted candidate multiset must be contained in proposed "
                "candidate multiset"
            )
        if (
            output.proposed_status is StepStatus.COMPLETE
            and not contract.honors_terminal(
                request.run.record.terminal_output_contract
            )
        ):
            raise ValueError(
                "a COMPLETE Step must honor the run terminal output contract"
            )

    @staticmethod
    def _validate_output_candidates(
        request: OptimStepRequest,
        output: AdapterOutput,
    ) -> None:
        bases = {
            candidate_reference(base).record_ref: base
            for base in request.candidates
        }
        for label, candidates in (
            ("proposed", output.proposed_candidates),
            ("accepted", output.accepted_candidates),
        ):
            for candidate in candidates:
                try:
                    validate_candidate_template(
                        candidate=candidate,
                        run=request.run,
                    )
                except ValueError as error:
                    raise ValueError(
                        f"every {label} candidate must satisfy the exact run "
                        f"template contract: {error}"
                    ) from error
                if request.mode is StepMode.PURE:
                    continue
                base = bases.get(candidate.base_ref)
                if base is None:
                    raise ValueError(
                        f"every {label} candidate must bind an exact request "
                        "candidate as its base"
                    )
                try:
                    diff_check(base=base, proposed=candidate, run=request.run)
                except ValueError as error:
                    raise ValueError(
                        f"every {label} candidate must satisfy the canonical "
                        f"run mutation diff: {error}"
                    ) from error

    @staticmethod
    def _validate_output_intents(
        request: OptimStepRequest,
        output: AdapterOutput,
    ) -> None:
        allowed = {
            str(candidate.identity_hash): candidate
            for candidate in (
                *(candidate_reference(item) for item in request.candidates),
                *(
                    candidate_reference(item)
                    for item in output.proposed_candidates
                ),
                *(
                    candidate_reference(item)
                    for item in output.accepted_candidates
                ),
            )
        }
        reward_policy = request.run.record.reward_policy
        for evidence in output.search_evidence:
            if evidence.optim_run_id != request.run_id:
                raise ValueError(
                    "search evidence belongs to another optimization run"
                )
            if evidence.optim_step_index != request.step_index:
                raise ValueError(
                    "search evidence belongs to another optimization step"
                )
        for optim_eval_request in output.optim_eval_requests:
            if optim_eval_request.optim_run_id != request.run_id:
                raise ValueError(
                    "Optim Eval Request belongs to another optimization run"
                )
            if optim_eval_request.optim_step_index != request.step_index:
                raise ValueError(
                    "Optim Eval Request belongs to another optimization step"
                )
            candidate_ref = candidate_reference(
                optim_eval_request.eval_request.candidate
            )
            exact_candidate = allowed.get(str(candidate_ref.identity_hash))
            if (
                exact_candidate is None
                or exact_candidate != candidate_ref
            ):
                raise ValueError(
                    "Optim Eval Request candidate is not an exact Step output "
                    "candidate"
                )
            if optim_eval_request.expected_reward_policy_hash is not None:
                if (
                    reward_policy is None
                    or optim_eval_request.expected_reward_policy_hash
                    != reward_policy.identity_hash()
                ):
                    raise ValueError(
                        "Optim Eval Request must expect the exact run Reward "
                        "Policy"
                    )

    def _resolve_intents(
        self,
        request: OptimStepRequest,
        output: AdapterOutput,
        proposed: tuple[CandidateRef, ...],
        accepted: tuple[CandidateRef, ...],
        *,
        eval_context: EvalExecutionContext,
    ) -> tuple[IntentResolution, ...]:
        if not output.optim_eval_requests:
            return ()
        self._validate_output_intents(request, output)
        if self._evaluation_service is None:
            raise ValueError(
                "proposal-only Step with Optim Eval Requests requires "
                "EvalService"
            )
        allowed = {
            str(candidate.identity_hash): candidate
            for candidate in (
                *(candidate_reference(item) for item in request.candidates),
                *proposed,
                *accepted,
            )
        }
        resolutions: list[IntentResolution] = []
        from whetstone.coordination.eval_service import (
            EvalDispatchMode,
            EvalEngineService,
            EvalPlatformDeferred,
        )

        platform_mode = (
            isinstance(self._evaluation_service, EvalEngineService)
            and eval_context.dispatch_mode is EvalDispatchMode.PLATFORM
        )
        deferred: list[OptimEvalRequest] = []
        for optim_eval_request in output.optim_eval_requests:
            if optim_eval_request.optim_run_id != request.run_id:
                raise ValueError(
                    "Optim Eval Request belongs to another optimization run"
                )
            if optim_eval_request.optim_step_index != request.step_index:
                raise ValueError(
                    "Optim Eval Request belongs to another optimization step"
                )
            candidate_ref = candidate_reference(
                optim_eval_request.eval_request.candidate
            )
            exact_candidate = allowed.get(str(candidate_ref.identity_hash))
            if (
                exact_candidate is None
                or exact_candidate != candidate_ref
            ):
                raise ValueError(
                    "Optim Eval Request candidate is not an exact Step output "
                    "candidate"
                )
            self._persist_intent_records(optim_eval_request)
            if platform_mode:
                assert isinstance(self._evaluation_service, EvalEngineService)
                self._evaluation_service.persist_platform_intent(
                    optim_eval_request,
                    context=eval_context,
                )
                deferred.append(optim_eval_request)
                continue
            try:
                resolutions.append(
                    self._resolve_one_intent(
                        request=request,
                        optim_eval_request=optim_eval_request,
                        eval_context=eval_context,
                    )
                )
            except EvalPlatformDeferred:
                continue
        if deferred:
            self._last_deferred_platform_intents = tuple(deferred)
            from whetstone.platform.deferred_intents import persist_deferred_intents

            persist_deferred_intents(
                self._store,
                run_id=request.run_id,
                step_index=request.step_index,
                intents=tuple(deferred),
            )
        return tuple(resolutions)

    def _resolve_one_intent(
        self,
        *,
        request: OptimStepRequest,
        optim_eval_request: OptimEvalRequest,
        eval_context: EvalExecutionContext,
    ) -> IntentResolution:
        if (
            self._evaluation_service is None
            or self._evaluation_replay_policy is None
        ):
            raise RuntimeError("EvalService is not configured")
        if (
            self._evaluation_service.replay_policy
            is not self._evaluation_replay_policy
        ):
            raise ValueError(
                "EvalService replay_policy changed after construction"
            )
        effect_request = self._intent_effect_request(request, optim_eval_request)
        acquisition = self._effect_authority.acquire(
            effect_request,
            owner_id=self._owner_id,
            attempt_id=uuid4().hex,
            lease_duration=self._lease_duration,
        )
        if acquisition.outcome in {
            AcquireOutcome.SUCCEEDED,
            AcquireOutcome.FAILED,
        }:
            terminal = acquisition.terminal
            if terminal is None or terminal.result_ref is None:
                raise RuntimeError(
                    "terminal Intent effect has no exact resolution ref"
                )
            resolution = self._load_intent_resolution(terminal.result_ref)
            self._validate_resolution(optim_eval_request, resolution)
            if acquisition.outcome is AcquireOutcome.FAILED:
                if (
                    resolution.terminal_failure is None
                    or terminal.failure != resolution.terminal_failure
                ):
                    raise ValueError(
                        "failed Intent effect does not match its exact "
                        "resolution failure"
                    )
            elif resolution.terminal_failure is not None:
                raise ValueError(
                    "successful Intent effect references a failed resolution"
                )
            self._evaluation_service.validate_resolution_graph(resolution)
            return resolution
        lease = self._acquired_lease(acquisition)
        with self._effect_authority.maintain(
            lease, lease_duration=self._lease_duration
        ) as maintenance:
            raw = self._evaluation_service.resolve_optim_eval_request(
                optim_eval_request,
                context=eval_context,
            )
            resolution = IntentResolution.model_validate(
                raw.model_dump(mode="json")
            )
            self._validate_resolution(optim_eval_request, resolution)
            self._evaluation_service.validate_resolution_graph(resolution)
            resolution_ref = self._put(
                INTENT_RESOLUTION_SCHEMA,
                resolution.model_dump(mode="json"),
            )
            expected = typed_ref_for_record(
                INTENT_RESOLUTION_SCHEMA,
                resolution.model_dump(mode="json"),
            )
            if resolution_ref != expected:
                raise ValueError(
                    "persisted Intent Resolution ref failed validation"
                )
            if resolution.terminal_failure is None:
                maintenance.succeed(result_ref=resolution_ref)
            else:
                maintenance.fail(
                    result_ref=resolution_ref,
                    failure=resolution.terminal_failure,
                )
        return resolution

    def _intent_effect_request(
        self,
        request: OptimStepRequest,
        optim_eval_request: OptimEvalRequest,
    ) -> EffectRequest:
        if self._evaluation_replay_policy is None:
            raise RuntimeError("EvalService is not configured")

        key_payload = {
            "step_request_ref": step_request_reference(
                request
            ).record_ref.model_dump(mode="json"),
            "eval_request_id": optim_eval_request.eval_request.request_id,
        }
        return EffectRequest(
            semantic_key=compute_prefixed_identity_key(
                schema=INTENT_EFFECT_KEY_SCHEMA,
                schema_version=INTENT_EFFECT_KEY_SCHEMA_VERSION,
                prefix=INTENT_EFFECT_KEY_PREFIX,
                payload=key_payload,
            ),
            request_hash=compute_identity_hash(
                schema=INTENT_EFFECT_SCHEMA,
                schema_version=INTENT_EFFECT_SCHEMA_VERSION,
                payload={
                    "optim_eval_request": optim_eval_request.model_dump(
                        mode="json"
                    ),
                },
            ),
            replay_policy=self._evaluation_replay_policy,
        )

    def _load_intent_resolution(self, ref: TypedRef) -> IntentResolution:
        if ref.schema_name != INTENT_RESOLUTION_SCHEMA:
            raise ValueError("Intent Resolution ref has the wrong schema")
        resolution = IntentResolution.model_validate(
            self._store.get(ref.reference)
        )
        if (
            typed_ref_for_record(
                INTENT_RESOLUTION_SCHEMA,
                resolution.model_dump(mode="json"),
            )
            != ref
        ):
            raise ValueError("persisted Intent Resolution ref is not exact")
        return resolution

    def _validate_resolution(
        self,
        optim_eval_request: OptimEvalRequest,
        resolution: IntentResolution,
    ) -> None:
        if resolution.optim_eval_request != optim_eval_request:
            raise ValueError(
                "EvalService resolved another exact Optim Eval Request"
            )
        if resolution.eval_result_ref is not None:
            self._validate_stored_ref(
                resolution.eval_result_ref,
                label="Evaluation Result",
            )
        for evidence_ref in resolution.reward_evidence_refs:
            self._validate_stored_ref(
                evidence_ref,
                label="Reward evidence",
            )

    def _validate_stored_ref(self, ref: TypedRef, *, label: str) -> None:
        record = self._store.get(ref.reference)
        if typed_ref_for_record(ref.schema_name, record) != ref:
            raise ValueError(f"{label} ref is not exact")

    def terminalize(
        self,
        *,
        run: OptimRunRef,
        step_results: tuple[OptimStepResultRef, ...],
        cost: dict[str, object] | None = None,
    ) -> tuple[OptimResult, TypedRef]:
        if self._bound_run is None:
            raise ValueError("bind_run must be called before terminalize")
        exact_run = OptimRunRef.model_validate(
            run.model_dump(mode="json")
        )
        if exact_run != self._bound_run:
            raise ValueError(
                "terminalize run differs from the bound exact run"
            )
        if not step_results:
            raise ValueError("terminalize requires at least one Step Result")
        exact_step_results = tuple(
            OptimStepResultRef.model_validate(
                result.model_dump(mode="json")
            )
            for result in step_results
        )
        run_id = str(exact_run.record.run_id)

        result = self._assemble_terminal(
            run=exact_run,
            step_results=exact_step_results,
            cost=cost or {},
        )
        requested_ref = optimization_result_reference(result)
        existing_ref = self.resolve_optimization_result(run_id)
        if existing_ref is not None:
            existing = self._load_terminal(existing_ref)
            if existing_ref == requested_ref and existing == result:
                return existing, existing_ref
            raise OptimResultConflictError(
                run_id=run_id,
                existing=existing_ref,
                requested=requested_ref,
            )

        result_ref = self._put(
            OPTIM_RESULT_SCHEMA, result.record_content()
        )
        if result_ref != requested_ref:
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
            raise OptimResultConflictError(
                run_id=run_id,
                existing=existing,
                requested=result_ref,
            ) from conflict
        if status is BindStatus.IDEMPOTENT:
            bound = self.resolve_optimization_result(run_id)
            if bound is None:
                raise RuntimeError(
                    "idempotent Optimization Result binding disappeared"
                )
            replay = self._load_terminal(bound)
            if bound != result_ref or replay != result:
                raise OptimResultConflictError(
                    run_id=run_id,
                    existing=bound,
                    requested=result_ref,
                )
            return replay, bound
        return result, result_ref

    def _assemble_terminal(
        self,
        *,
        run: OptimRunRef,
        step_results: tuple[OptimStepResultRef, ...],
        cost: dict[str, object],
    ) -> OptimResult:
        run_id = str(run.record.run_id)
        results: list[OptimStepResult] = []
        for index, exact_result in enumerate(step_results):
            ref = exact_result.record_ref
            actual = self._resolve_result_binding(run_id, index)
            if actual != ref:
                raise ValueError(
                    "terminal Step Result refs must match ordered bindings"
                )
            result = self._load_result(ref)
            if step_result_reference(result) != exact_result:
                raise ValueError(
                    "terminal Step Result is not the exact supplied result"
                )
            if result.run_id != run_id or result.step_index != index:
                raise ValueError(
                    "terminal Step Result belongs to another run or position"
                )
            if result.request.record.run != run:
                raise ValueError(
                    "terminal Step Result request belongs to another exact run"
                )
            if index < len(step_results) - 1 and (
                result.status is not StepStatus.CONTINUE
            ):
                raise ValueError(
                    "only the final terminal Step Result may stop the run"
                )
            results.append(result)
        if self._resolve_result_binding(run_id, len(step_results)) is not None:
            raise ValueError(
                "terminal Step Result refs omit a later bound Step Result"
            )
        last = results[-1]
        if last.status is StepStatus.CONTINUE:
            raise ValueError("cannot terminalize a continuing Step Result")
        proposals = (
            ()
            if last.status is StepStatus.FAILED
            else tuple(
                OptimProposal(candidate=candidate)
                for candidate in last.accepted_candidates
            )
        )
        return OptimResult(
            run=run,
            proposals=proposals,
            step_results=step_results,
            cost=cost,
            terminal_failure=last.terminal_failure,
            seed_retained=last.seed_retained,
        )

    @staticmethod
    def carry_budget_forward(
        prior: OptimStepResult,
    ) -> BudgetState:
        return prior.budget
