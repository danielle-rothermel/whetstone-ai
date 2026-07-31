"""Algorithm-neutral durable optimization harness."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from datetime import datetime, timedelta
from threading import RLock
from typing import Any, Protocol
from uuid import uuid4

from dr_store import BindingConflictError, BindStatus, ObjectStore
from pydantic import BaseModel, ConfigDict, model_validator

from whetstone.evaluation_role import EvaluationRole
from whetstone.optimization.adapters import (
    AdapterCheckpoint,
    AdapterOutput,
    AdapterRegistry,
    OptimizerAdapter,
)
from whetstone.optimization.effect_authority import (
    AcquireOutcome,
    AcquireResult,
    EffectAuthority,
    EffectLease,
    EffectRequest,
    ReplayPolicy,
)
from whetstone.optimization.identity import (
    NonNegativeInt,
    OpaqueKey,
    TerminalFailure,
    TypedRef,
    compute_identity_hash,
    typed_ref_for_record,
)
from whetstone.optimization.mutation import (
    diff_check,
    validate_candidate_template,
)
from whetstone.optimization.schema import (
    CANDIDATE_RECORD_SCHEMA,
    EVAL_CONFIG_RECORD_SCHEMA,
    EVALUATION_BINDING_SCHEMA,
    OPTIMIZATION_RESULT_SCHEMA,
    OPTIMIZATION_RUN_SCHEMA,
    STEP_REQUEST_SCHEMA,
    STEP_RESULT_SCHEMA,
    BudgetState,
    Candidate,
    CandidateRef,
    EvaluationIntent,
    IntentResolution,
    OptimizationProposal,
    OptimizationResult,
    OptimizationRun,
    OptimizationRunRef,
    OptimizationStepRequest,
    OptimizationStepRequestRef,
    OptimizationStepResult,
    OptimizationStepResultRef,
    StepMode,
    StepStatus,
    ToolEvidence,
    candidate_reference,
    optimization_result_reference,
    optimization_run_reference,
    step_request_reference,
    step_result_reference,
)
from whetstone.optimization.tool_store import ToolCallState, ToolCallStore
from whetstone.optimization.tools import (
    TOOL_CONFIG_SCHEMA,
    TOOL_DEFINITION_SCHEMA,
    RuntimeToolHandle,
    ToolCall,
    ToolCallRef,
    ToolCapacityBinding,
    ToolCapacityScope,
    ToolConfig,
    ToolResult,
    ToolResultRef,
    tool_call_reference,
    tool_capacity_binding,
    tool_result_reference,
)

__all__ = [
    "ADAPTER_CHECKPOINT_SCHEMA",
    "INTENT_RESOLUTION_SCHEMA",
    "EffectBusyError",
    "EffectRecoveryRequiredError",
    "EffectRequestConflictError",
    "EvaluationService",
    "IssuedToolCallConflictError",
    "OptimizationHarness",
    "OptimizationResultConflictError",
    "OptimizationRunConflictError",
    "StepResultConflictError",
    "ToolExecutor",
]

ADAPTER_CHECKPOINT_SCHEMA = "whetstone.optimization_adapter_checkpoint"
STATE_SNAPSHOT_SCHEMA = "whetstone.optimization_state_snapshot"
HISTORY_SNAPSHOT_SCHEMA = "whetstone.optimization_history_snapshot"
INTENT_RESOLUTION_SCHEMA = "whetstone.optimization_intent_resolution"
ADAPTER_EFFECT_SCHEMA = "whetstone.optimization_adapter_effect"
ADAPTER_EFFECT_SCHEMA_VERSION = 1
ADAPTER_EFFECT_KEY_SCHEMA = "whetstone.optimization_adapter_effect_key"
ADAPTER_EFFECT_KEY_SCHEMA_VERSION = 1
ADAPTER_EFFECT_KEY_PREFIX = "whetstone.optimization_adapter:"
INTENT_EFFECT_SCHEMA = "whetstone.optimization_intent_effect"
INTENT_EFFECT_SCHEMA_VERSION = 1
INTENT_EFFECT_KEY_SCHEMA = "whetstone.optimization_intent_effect_key"
INTENT_EFFECT_KEY_SCHEMA_VERSION = 1
INTENT_EFFECT_KEY_PREFIX = "whetstone.optimization_intent:"
ISSUED_TOOL_CALL_CLAIM_SCHEMA = "whetstone.optimization_issued_tool_call_claim"
ISSUED_TOOL_CALL_SLOT_SCHEMA = "whetstone.optimization_issued_tool_call_slot"
ISSUED_TOOL_CALL_TERMINAL_SCHEMA = (
    "whetstone.optimization_issued_tool_call_terminal"
)
ISSUED_TOOL_CALL_KEY_SCHEMA = "whetstone.optimization_issued_tool_call_key"
ISSUED_TOOL_CALL_KEY_SCHEMA_VERSION = 1
ISSUED_TOOL_CALL_KEY_PREFIX = "whetstone.optimization_issued_tool_call:"
ISSUED_TOOL_CALL_SLOT_KEY_SCHEMA = (
    "whetstone.optimization_issued_tool_call_slot_key"
)
ISSUED_TOOL_CALL_SLOT_KEY_SCHEMA_VERSION = 1
ISSUED_TOOL_CALL_SLOT_KEY_PREFIX = (
    "whetstone.optimization_issued_tool_call_slot:"
)
ISSUED_TOOL_CALL_TERMINAL_KEY_SCHEMA = (
    "whetstone.optimization_issued_tool_call_terminal_key"
)
ISSUED_TOOL_CALL_TERMINAL_KEY_SCHEMA_VERSION = 1
ISSUED_TOOL_CALL_TERMINAL_KEY_PREFIX = (
    "whetstone.optimization_issued_tool_call_terminal:"
)


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


class OptimizationRunConflictError(Exception):
    def __init__(
        self, *, run_id: str, existing: TypedRef, requested: TypedRef
    ) -> None:
        self.run_id = run_id
        self.existing = existing
        self.requested = requested
        super().__init__(
            f"Optimization run {run_id!r} is already bound to "
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


class IssuedToolCallConflictError(ValueError):
    """A step-local call ID was already claimed by another exact call."""

    def __init__(self, *, call_id: str) -> None:
        self.call_id = call_id
        super().__init__(
            f"Tool Call ID {call_id!r} is bound to another exact Tool Call"
        )


class _IssuedToolCallClaim(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    request: OptimizationStepRequestRef
    call: ToolCallRef


class _IssuedToolCallClaimRef(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    record: _IssuedToolCallClaim
    record_ref: TypedRef

    @model_validator(mode="after")
    def _validate(self) -> _IssuedToolCallClaimRef:
        expected = typed_ref_for_record(
            ISSUED_TOOL_CALL_CLAIM_SCHEMA,
            self.record.model_dump(mode="json"),
        )
        if self.record_ref != expected:
            raise ValueError(
                "Issued Tool Call claim ref must address the exact claim"
            )
        return self


class _IssuedToolCallSlot(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    request: OptimizationStepRequestRef
    ordinal: NonNegativeInt
    claim: _IssuedToolCallClaimRef

    @model_validator(mode="after")
    def _validate(self) -> _IssuedToolCallSlot:
        if self.claim.record.request != self.request:
            raise ValueError(
                "Issued Tool Call slot and claim must cite the exact request"
            )
        return self


class _IssuedToolCallTerminal(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    claim: _IssuedToolCallClaimRef
    result: ToolResultRef

    @model_validator(mode="after")
    def _validate(self) -> _IssuedToolCallTerminal:
        if self.result.record.call != self.claim.record.call:
            raise ValueError(
                "Issued Tool Call terminal belongs to another exact call"
            )
        return self


def _issued_tool_call_binding_key(
    request: OptimizationStepRequestRef,
    call_id: str,
) -> str:
    # Persisted-format contract: schema, version, prefix, and payload keys are
    # pinned by golden tests. Never derive these payload keys from model
    # fields.
    digest = compute_identity_hash(
        schema=ISSUED_TOOL_CALL_KEY_SCHEMA,
        schema_version=ISSUED_TOOL_CALL_KEY_SCHEMA_VERSION,
        payload={
            "step_request_ref": request.record_ref.model_dump(mode="json"),
            "call_id": call_id,
        },
    )
    return f"{ISSUED_TOOL_CALL_KEY_PREFIX}{digest}"


def _issued_tool_call_slot_binding_key(
    request: OptimizationStepRequestRef,
    ordinal: int,
) -> str:
    # Persisted-format contract; literals are pinned by golden tests.
    digest = compute_identity_hash(
        schema=ISSUED_TOOL_CALL_SLOT_KEY_SCHEMA,
        schema_version=ISSUED_TOOL_CALL_SLOT_KEY_SCHEMA_VERSION,
        payload={
            "step_request_ref": request.record_ref.model_dump(mode="json"),
            "ordinal": ordinal,
        },
    )
    return f"{ISSUED_TOOL_CALL_SLOT_KEY_PREFIX}{digest}"


def _issued_tool_call_terminal_binding_key(
    claim: _IssuedToolCallClaimRef,
) -> str:
    # Persisted-format contract; literals are pinned by golden tests.
    digest = compute_identity_hash(
        schema=ISSUED_TOOL_CALL_TERMINAL_KEY_SCHEMA,
        schema_version=ISSUED_TOOL_CALL_TERMINAL_KEY_SCHEMA_VERSION,
        payload={
            "issued_tool_call_claim_ref": claim.record_ref.model_dump(
                mode="json"
            ),
        },
    )
    return f"{ISSUED_TOOL_CALL_TERMINAL_KEY_PREFIX}{digest}"


class EvaluationService(Protocol):
    @property
    def replay_policy(self) -> ReplayPolicy: ...

    def resolve_evaluation_intent(
        self, intent: EvaluationIntent
    ) -> IntentResolution: ...


class ToolExecutor(Protocol):
    def runtime_handle(
        self,
        config: ToolConfig,
        store: ToolCallStore,
        binding: ToolCapacityBinding,
    ) -> RuntimeToolHandle: ...


class _IssuedToolCallLedger:
    """Harness-owned durable issuance, ordering, spend, and evidence."""

    def __init__(
        self,
        *,
        store: ObjectStore,
        tool_store: ToolCallStore,
        request: OptimizationStepRequestRef,
    ) -> None:
        self._store = store
        self._tool_store = tool_store
        self._request = request
        raw_limit = request.record.budget.remaining.get("tool_calls", 0)
        if type(raw_limit) is not int:
            raise TypeError("validated tool_calls budget is not an integer")
        self._limit = raw_limit
        self._lock = RLock()
        self._active_call_ids: set[str] = set()
        self._seen_call_ids: set[str] = set()
        self._replay_prefix = self._slots()
        self._replay_ordinal = 0

    @staticmethod
    def _typed_ref(reference: Any) -> TypedRef:
        return TypedRef(
            schema_name=reference.schema,
            content_hash=reference.content_hash,
        )

    def _put_exact(
        self,
        schema: str,
        content: dict[str, Any],
    ) -> TypedRef:
        expected = typed_ref_for_record(schema, content)
        reference, _status = self._store.put(schema, content)
        persisted = self._typed_ref(reference)
        if persisted != expected:
            raise ValueError(f"persisted {schema} ref failed validation")
        return persisted

    def _resolve(self, key: str) -> TypedRef | None:
        reference = self._store.resolve(key)
        return None if reference is None else self._typed_ref(reference)

    def _claim_reference(self, call: ToolCall) -> _IssuedToolCallClaimRef:
        record = _IssuedToolCallClaim(
            request=self._request,
            call=tool_call_reference(call),
        )
        expected = _IssuedToolCallClaimRef(
            record=record,
            record_ref=typed_ref_for_record(
                ISSUED_TOOL_CALL_CLAIM_SCHEMA,
                record.model_dump(mode="json"),
            ),
        )
        persisted = self._put_exact(
            ISSUED_TOOL_CALL_CLAIM_SCHEMA,
            record.model_dump(mode="json"),
        )
        if persisted != expected.record_ref:
            raise ValueError(
                "persisted Issued Tool Call claim ref failed validation"
            )
        return expected

    def _load_claim(self, ref: TypedRef) -> _IssuedToolCallClaimRef:
        if ref.schema_name != ISSUED_TOOL_CALL_CLAIM_SCHEMA:
            raise ValueError("Issued Tool Call claim ref has the wrong schema")
        record = _IssuedToolCallClaim.model_validate(
            self._store.get(ref.reference)
        )
        exact = _IssuedToolCallClaimRef(record=record, record_ref=ref)
        if exact.record.request != self._request:
            raise ValueError(
                "Issued Tool Call claim belongs to another exact request"
            )
        return exact

    def _load_slot(
        self,
        ref: TypedRef,
        *,
        ordinal: int,
    ) -> _IssuedToolCallSlot:
        if ref.schema_name != ISSUED_TOOL_CALL_SLOT_SCHEMA:
            raise ValueError("Issued Tool Call slot ref has the wrong schema")
        content = self._store.get(ref.reference)
        slot = _IssuedToolCallSlot.model_validate(content)
        if typed_ref_for_record(ISSUED_TOOL_CALL_SLOT_SCHEMA, content) != ref:
            raise ValueError(
                "persisted Issued Tool Call slot ref is not exact"
            )
        if slot.request != self._request or slot.ordinal != ordinal:
            raise ValueError(
                "Issued Tool Call slot belongs to another request or ordinal"
            )
        if self._load_claim(slot.claim.record_ref) != slot.claim:
            raise ValueError(
                "Issued Tool Call slot does not contain its exact durable "
                "claim"
            )
        return slot

    def _validate_or_reconcile_claim_binding(
        self,
        claim: _IssuedToolCallClaimRef,
    ) -> None:
        call_id = str(claim.record.call.record.call_id)
        key = _issued_tool_call_binding_key(self._request, call_id)
        bound = self._resolve(key)
        if bound is None:
            try:
                self._store.bind(key, claim.record_ref.reference)
            except BindingConflictError:
                bound = self._resolve(key)
            else:
                bound = self._resolve(key)
        if bound is None:
            raise RuntimeError("Issued Tool Call claim binding disappeared")
        if self._load_claim(bound) != claim:
            raise IssuedToolCallConflictError(call_id=call_id)

    def _slots(self) -> tuple[_IssuedToolCallSlot, ...]:
        # The supported writer can reserve only ordinals below _limit. Reading
        # the first impossible ordinal is therefore a complete bounded
        # overflow sentinel; arbitrary higher-key scans are neither required
        # nor supported by ObjectStore's binding interface.
        overflow = self._resolve(
            _issued_tool_call_slot_binding_key(self._request, self._limit)
        )
        if overflow is not None:
            raise ValueError(
                "Issued Tool Call slot ordinal is outside the bounded budget: "
                f"found ordinal {self._limit}, limit {self._limit}"
            )
        refs: list[TypedRef | None] = []
        for ordinal in range(self._limit):
            refs.append(
                self._resolve(
                    _issued_tool_call_slot_binding_key(self._request, ordinal)
                )
            )
        first_gap = next(
            (ordinal for ordinal, ref in enumerate(refs) if ref is None),
            len(refs),
        )
        if any(ref is not None for ref in refs[first_gap + 1 :]):
            raise ValueError(
                "Issued Tool Call slots must be contiguous from ordinal zero"
            )

        slots: list[_IssuedToolCallSlot] = []
        seen_claims: set[TypedRef] = set()
        for ordinal, ref in enumerate(refs[:first_gap]):
            if ref is None:  # pragma: no cover - established by first_gap
                raise AssertionError("contiguous slot prefix contains a gap")
            slot = self._load_slot(ref, ordinal=ordinal)
            if slot.claim.record_ref in seen_claims:
                raise ValueError(
                    "Issued Tool Call claim occupies multiple durable slots"
                )
            seen_claims.add(slot.claim.record_ref)
            slots.append(slot)
        for slot in slots:
            # The slot is the durable reservation. Publishing it first makes
            # the claim recoverable if the process dies before this binding.
            self._validate_or_reconcile_claim_binding(slot.claim)
        return tuple(slots)

    def _reserve(
        self,
        claim: _IssuedToolCallClaimRef,
    ) -> _IssuedToolCallSlot:
        slots = self._slots()
        matching = tuple(
            slot for slot in slots if slot.claim.record_ref == claim.record_ref
        )
        if len(matching) > 1:
            raise ValueError(
                "Issued Tool Call claim occupies multiple durable slots"
            )
        if matching:
            return matching[0]
        ordinal = len(slots)
        if ordinal >= self._limit:
            raise ValueError(
                "Tool Call budget exhausted before dispatch: attempted "
                f"call {claim.record.call.record.call_id!r}, but only "
                f"{self._limit} tool_calls remain"
            )
        slot = _IssuedToolCallSlot(
            request=self._request,
            ordinal=ordinal,
            claim=claim,
        )
        content = slot.model_dump(mode="json")
        slot_ref = self._put_exact(ISSUED_TOOL_CALL_SLOT_SCHEMA, content)
        key = _issued_tool_call_slot_binding_key(self._request, ordinal)
        try:
            self._store.bind(key, slot_ref.reference)
        except BindingConflictError:
            # Distinct calls may reserve concurrently. The durable binding
            # winner determines the ordinal; retry against the next slot.
            return self._reserve(claim)
        bound = self._resolve(key)
        if bound != slot_ref:
            raise RuntimeError("Issued Tool Call slot binding disappeared")
        return slot

    def _claim_and_reserve(self, call: ToolCall) -> _IssuedToolCallClaimRef:
        requested = self._claim_reference(call)
        key = _issued_tool_call_binding_key(self._request, str(call.call_id))
        existing_ref = self._resolve(key)
        if existing_ref is not None:
            existing = self._load_claim(existing_ref)
            if existing != requested:
                raise IssuedToolCallConflictError(call_id=str(call.call_id))
            self._reserve(existing)
            return existing
        self._reserve(requested)
        self._validate_or_reconcile_claim_binding(requested)
        return requested

    def _load_terminal(
        self,
        ref: TypedRef,
        *,
        claim: _IssuedToolCallClaimRef,
    ) -> _IssuedToolCallTerminal:
        if ref.schema_name != ISSUED_TOOL_CALL_TERMINAL_SCHEMA:
            raise ValueError(
                "Issued Tool Call terminal ref has the wrong schema"
            )
        content = self._store.get(ref.reference)
        terminal = _IssuedToolCallTerminal.model_validate(content)
        if (
            typed_ref_for_record(ISSUED_TOOL_CALL_TERMINAL_SCHEMA, content)
            != ref
        ):
            raise ValueError(
                "persisted Issued Tool Call terminal ref is not exact"
            )
        if terminal.claim != claim:
            raise ValueError(
                "Issued Tool Call terminal belongs to another exact claim"
            )
        return terminal

    def _authoritative_terminal(
        self,
        terminal: _IssuedToolCallTerminal,
    ) -> tuple[ToolResult, Any]:
        call = terminal.claim.record.call.record
        entry = self._tool_store.get(call)
        if entry is None:
            raise ValueError(
                "Issued Tool Call terminal has no Tool Call Store entry"
            )
        result = self._tool_store.load_terminal_result(entry)
        if tool_result_reference(result) != terminal.result:
            raise ValueError(
                "Issued Tool Call ledger and Tool Call Store terminal disagree"
            )
        return result, entry

    def _terminal_for_claim(
        self,
        claim: _IssuedToolCallClaimRef,
    ) -> _IssuedToolCallTerminal | None:
        ref = self._resolve(_issued_tool_call_terminal_binding_key(claim))
        return None if ref is None else self._load_terminal(ref, claim=claim)

    def _record_terminal(
        self,
        *,
        claim: _IssuedToolCallClaimRef,
        result: ToolResult,
    ) -> ToolResult:
        validated = ToolResult.model_validate(result.model_dump(mode="json"))
        if validated.call != claim.record.call:
            raise ValueError(
                "Tool executor returned a result for another exact Tool Call"
            )
        result_ref = tool_result_reference(validated)
        if self._tool_store.persist_result(validated) != result_ref.record_ref:
            raise ValueError("persisted Tool Result ref failed validation")
        entry = self._tool_store.get(claim.record.call.record)
        if entry is None:
            raise ValueError("Tool Result has no Tool Call Store entry")
        authoritative = self._tool_store.load_terminal_result(entry)
        if authoritative != validated:
            raise ValueError(
                "Tool executor result differs from its authoritative terminal"
            )
        terminal = _IssuedToolCallTerminal(
            claim=claim,
            result=result_ref,
        )
        content = terminal.model_dump(mode="json")
        terminal_ref = self._put_exact(
            ISSUED_TOOL_CALL_TERMINAL_SCHEMA,
            content,
        )
        key = _issued_tool_call_terminal_binding_key(claim)
        try:
            self._store.bind(key, terminal_ref.reference)
        except BindingConflictError as conflict:
            existing_ref = self._resolve(key)
            if existing_ref is None:
                raise RuntimeError(
                    "Issued Tool Call terminal binding disappeared"
                ) from conflict
            existing = self._load_terminal(existing_ref, claim=claim)
            if existing != terminal:
                raise ValueError(
                    "Issued Tool Call already has another exact terminal"
                ) from conflict
        bound = self._resolve(key)
        if bound is None:
            raise RuntimeError("Issued Tool Call terminal binding disappeared")
        persisted = self._load_terminal(bound, claim=claim)
        if persisted != terminal:
            raise ValueError(
                "Issued Tool Call already has another exact terminal"
            )
        return authoritative

    def issue(
        self,
        call: ToolCall,
        execute: RuntimeToolHandle,
    ) -> ToolResult:
        exact_call = ToolCall.model_validate(call.model_dump(mode="json"))
        call_id = str(exact_call.call_id)
        with self._lock:
            if (
                call_id in self._active_call_ids
                or call_id in self._seen_call_ids
            ):
                raise ValueError(
                    "Tool Call IDs must be unique within a Step attempt"
                )
            self._seen_call_ids.add(call_id)
            self._active_call_ids.add(call_id)
            try:
                if self._replay_ordinal < len(self._replay_prefix):
                    expected_slot = self._replay_prefix[self._replay_ordinal]
                    if expected_slot.claim.record.call != tool_call_reference(
                        exact_call
                    ):
                        expected_call = (
                            expected_slot.claim.record.call.record.call_id
                        )
                        if str(expected_call) == call_id:
                            raise IssuedToolCallConflictError(call_id=call_id)
                        raise ValueError(
                            "recovered adapter must replay the exact durable "
                            "Tool Call prefix in order before issuing a novel "
                            f"call: expected ordinal {self._replay_ordinal} "
                            f"call {expected_call!r}, got {call_id!r}"
                        )
                    claim = expected_slot.claim
                    self._validate_or_reconcile_claim_binding(claim)
                    self._replay_ordinal += 1
                else:
                    claim = self._claim_and_reserve(exact_call)
                existing = self._terminal_for_claim(claim)
            except BaseException:
                self._active_call_ids.remove(call_id)
                raise
        if existing is not None:
            try:
                result, _entry = self._authoritative_terminal(existing)
                return result
            finally:
                with self._lock:
                    self._active_call_ids.remove(call_id)
        try:
            entry = self._tool_store.get(exact_call)
            if entry is not None and entry.state in {
                ToolCallState.REFUSED,
                ToolCallState.COMPLETED,
            }:
                terminal_result = self._tool_store.load_terminal_result(entry)
                return self._record_terminal(
                    claim=claim,
                    result=terminal_result,
                )
            result = execute(exact_call)
            return self._record_terminal(claim=claim, result=result)
        finally:
            with self._lock:
                self._active_call_ids.remove(call_id)

    def validate_replay_complete(self) -> None:
        """Require an invoked recovery attempt to observe its whole prefix."""
        with self._lock:
            if self._replay_ordinal == len(self._replay_prefix):
                return
            expected = self._replay_prefix[self._replay_ordinal]
            raise ValueError(
                "recovered adapter skipped its durable Tool Call replay "
                f"prefix before checkpoint: expected ordinal "
                f"{self._replay_ordinal} call "
                f"{expected.claim.record.call.record.call_id!r}"
            )

    def evidence(self) -> tuple[ToolEvidence, ...]:
        evidence: list[ToolEvidence] = []
        seen_claims: set[TypedRef] = set()
        for slot in self._slots():
            claim = slot.claim
            if claim.record_ref in seen_claims:
                raise ValueError(
                    "Issued Tool Call claim occupies multiple durable slots"
                )
            seen_claims.add(claim.record_ref)
            terminal = self._terminal_for_claim(claim)
            if terminal is None:
                raise ValueError(
                    "Issued Tool Call has no durable terminal result; "
                    "adapter recovery is required"
                )
            _result, entry = self._authoritative_terminal(terminal)
            evidence.append(
                ToolEvidence(result=terminal.result, store_entry=entry)
            )
        return tuple(evidence)

    def budget_delta(
        self,
        adapter_delta: Any,
        *,
        issued_count: int,
    ) -> Any:
        consumed = dict(adapter_delta.consumed)
        consumed.pop("tool_calls", None)
        if "tool_calls" in self._request.record.budget.remaining:
            consumed["tool_calls"] = issued_count
        return type(adapter_delta)(consumed=consumed)


class OptimizationHarness:
    """Durable coordinator with all algorithm behavior behind a registry."""

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
        evaluation_service: EvaluationService | None = None,
        tool_executor: ToolExecutor | None = None,
    ) -> None:
        self._store = store
        self._adapter_registry = adapter_registry
        self._tool_store = tool_store
        self._effect_authority = effect_authority
        self._owner_id = owner_id
        self._adapter_replay_policy = adapter_replay_policy
        self._lease_duration = lease_duration
        self._evaluation_service = evaluation_service
        self._tool_executor = tool_executor
        self._bound_run: OptimizationRunRef | None = None
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
                "EvaluationService replay_policy must be idempotent or "
                "durable_workflow"
            )
        if evaluation_service is not None:
            self._evaluation_replay_policy = evaluation_service.replay_policy

    @staticmethod
    def _result_binding_key(run_id: str, step_index: int) -> str:
        return f"whetstone.optimization_step_result:{run_id}#{step_index}"

    @staticmethod
    def _terminal_binding_key(run_id: str) -> str:
        return f"whetstone.optimization_result:{run_id}"

    @staticmethod
    def _run_binding_key(run_id: str) -> str:
        return f"whetstone.optimization_run:{run_id}"

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

    def _load_run(self, ref: TypedRef) -> OptimizationRunRef:
        if ref.schema_name != OPTIMIZATION_RUN_SCHEMA:
            raise ValueError("bound run ref has the wrong schema")
        run = OptimizationRun.model_validate(self._store.get(ref.reference))
        exact = optimization_run_reference(run)
        if exact.record_ref != ref:
            raise ValueError("bound Optimization Run ref is not exact")
        return exact

    def bind_run(
        self, run: OptimizationRun | OptimizationRunRef
    ) -> OptimizationRunRef:
        if isinstance(run, OptimizationRunRef):
            exact = OptimizationRunRef.model_validate(
                run.model_dump(mode="json")
            )
        else:
            validated = OptimizationRun.model_validate(
                run.model_dump(mode="json")
            )
            exact = optimization_run_reference(validated)
        persisted = self._put(
            OPTIMIZATION_RUN_SCHEMA, exact.record.record_content()
        )
        if persisted != exact.record_ref:
            raise ValueError(
                "persisted Optimization Run ref failed content validation"
            )
        self._persist_tool_configs(exact)
        key = self._run_binding_key(str(exact.record.run_id))
        try:
            self._store.bind(key, exact.record_ref.reference)
        except BindingConflictError as conflict:
            existing = TypedRef(
                schema_name=conflict.existing.schema,
                content_hash=conflict.existing.content_hash,
            )
            if existing == exact.record_ref:
                self._bound_run = exact
                return exact
            raise OptimizationRunConflictError(
                run_id=str(exact.record.run_id),
                existing=existing,
                requested=exact.record_ref,
            ) from conflict
        existing = self._resolve_binding(key)
        if existing is None:
            raise RuntimeError("Optimization Run binding disappeared")
        if existing != exact.record_ref:
            raise OptimizationRunConflictError(
                run_id=str(exact.record.run_id),
                existing=existing,
                requested=exact.record_ref,
            )
        loaded = self._load_run(existing)
        if loaded != exact:
            raise ValueError("bound Optimization Run is not its exact record")
        self._bound_run = exact
        return exact

    def _persist_tool_configs(self, run: OptimizationRunRef) -> None:
        for config in run.record.tool_configs:
            definition = config.record.definition
            if (
                self._put(
                    TOOL_DEFINITION_SCHEMA,
                    definition.record.record_content(),
                )
                != definition.record_ref
            ):
                raise ValueError(
                    "persisted Tool Definition ref failed validation"
                )
            if (
                self._put(TOOL_CONFIG_SCHEMA, config.record.record_content())
                != config.record_ref
            ):
                raise ValueError("persisted Tool Config ref failed validation")

    def _validate_bound_run(self, request: OptimizationStepRequest) -> None:
        if self._bound_run is None:
            raise ValueError("bind_run must be called before run_step")
        if request.run != self._bound_run:
            raise ValueError(
                "Step Request belongs to a different exact Optimization Run"
            )
        actual = self._resolve_binding(
            self._run_binding_key(str(request.run_id))
        )
        if actual is None:
            raise ValueError("Optimization Run binding is absent")
        if actual != request.run.record_ref:
            raise ValueError(
                "Step Request run is not the durably bound exact run"
            )
        if self._load_run(actual) != request.run:
            raise ValueError("Step Request run ref is not exact")

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
        binding_content = intent.evaluation_binding.record_content()
        persisted_binding = self._put(
            EVALUATION_BINDING_SCHEMA, binding_content
        )
        if persisted_binding != typed_ref_for_record(
            EVALUATION_BINDING_SCHEMA, binding_content
        ):
            raise ValueError(
                "Intent Evaluation Binding ref failed content validation"
            )

    def _persist_snapshot(
        self, schema: str, delta: Mapping[str, Any]
    ) -> TypedRef | None:
        if not delta:
            return None
        content = dict(delta)
        expected = typed_ref_for_record(schema, content)
        persisted = self._put(schema, content)
        if persisted != expected:
            raise ValueError(f"persisted {schema} ref failed validation")
        return persisted

    def _load_result(self, ref: TypedRef) -> OptimizationStepResult:
        if ref.schema_name != STEP_RESULT_SCHEMA:
            raise ValueError("Step Result ref has the wrong schema")
        result = OptimizationStepResult.model_validate(
            self._store.get(ref.reference)
        )
        if step_result_reference(result).record_ref != ref:
            raise ValueError("persisted Step Result ref is not exact")
        return result

    def _load_checkpoint(self, ref: TypedRef) -> AdapterCheckpoint:
        if ref.schema_name != ADAPTER_CHECKPOINT_SCHEMA:
            raise ValueError("Adapter Checkpoint ref has the wrong schema")
        checkpoint = AdapterCheckpoint.model_validate(
            self._store.get(ref.reference)
        )
        if (
            typed_ref_for_record(
                ADAPTER_CHECKPOINT_SCHEMA, checkpoint.record_content()
            )
            != ref
        ):
            raise ValueError("persisted Adapter Checkpoint ref is not exact")
        return checkpoint

    def _load_terminal(self, ref: TypedRef) -> OptimizationResult:
        if ref.schema_name != OPTIMIZATION_RESULT_SCHEMA:
            raise ValueError("Optimization Result ref has the wrong schema")
        result = OptimizationResult.model_validate(
            self._store.get(ref.reference)
        )
        if optimization_result_reference(result) != ref:
            raise ValueError("persisted Optimization Result ref is not exact")
        return result

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
            if request.prior_step_result_ref is not None:
                raise ValueError(
                    "initial Step Request carries no prior Step Result"
                )
            if request.prior_state_ref is not None:
                raise ValueError("initial Step Request carries no prior state")
            if request.prior_history_ref is not None:
                raise ValueError(
                    "initial Step Request carries no prior history"
                )
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
        if preceding.request.record.run != request.run:
            raise ValueError(
                "preceding Step Result belongs to another exact Optimization "
                "Run"
            )
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
        if request.prior_state_ref != preceding.state_ref:
            raise ValueError(
                "a new Step must cite the preceding exact state, including "
                "its absence"
            )
        if request.prior_history_ref != preceding.history_ref:
            raise ValueError(
                "a new Step must cite the preceding exact history, including "
                "its absence"
            )

    def run_step(
        self, request: OptimizationStepRequest
    ) -> tuple[OptimizationStepResult, TypedRef]:
        validated_request = OptimizationStepRequest.model_validate(
            request.model_dump(mode="json")
        )
        self._validate_bound_run(validated_request)
        request = validated_request
        for candidate in request.candidates:
            validate_candidate_template(candidate=candidate, run=request.run)
        request_ref = self._put_request(request)
        exact_request = step_request_reference(request)
        if request_ref != exact_request.record_ref:
            raise ValueError("persisted request ref failed content validation")
        self._validate_prior_binding(request)
        for candidate in request.candidates:
            self._persist_candidate(candidate)

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
            output = self._invoke_pure(request, adapter)
        else:
            output = self._effectful_output(
                request,
                request_ref,
                adapter,
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
                request, output, proposed_refs, accepted_refs
            )
        else:
            resolutions = ()

        result = OptimizationStepResult(
            request=exact_request,
            proposed_candidates=proposed_refs,
            accepted_candidates=accepted_refs,
            resolved_intents=resolutions,
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
        request: OptimizationStepRequest,
        adapter: OptimizerAdapter,
    ) -> AdapterOutput:
        raw_output = adapter.invoke(request, ())
        output = AdapterOutput.model_validate(
            raw_output.model_dump(mode="json")
        )
        if output.evaluation_intents:
            raise ValueError("a pure Step emits no measurement requests")
        return output

    def _effectful_output(
        self,
        request: OptimizationStepRequest,
        request_ref: TypedRef,
        adapter: OptimizerAdapter,
        ledger: _IssuedToolCallLedger | None,
        guarded_handles: tuple[RuntimeToolHandle, ...],
    ) -> AdapterOutput:
        effect_request = self._adapter_effect_request(
            request, request_ref, adapter
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
                adapter=adapter,
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
        request: OptimizationStepRequest,
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
            if output.evaluation_intents:
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
        else:  # pragma: no cover - closed enum
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
        request: OptimizationStepRequest,
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
        request: OptimizationStepRequest,
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
        request: OptimizationStepRequest,
        request_ref: TypedRef,
        adapter: OptimizerAdapter,
    ) -> EffectRequest:
        # Persisted-format contract: key schema, version, prefix, and payload
        # literals are pinned by golden tests.
        payload = {
            "step_request_ref": request_ref.model_dump(mode="json"),
            "adapter_key": str(adapter.key),
        }
        semantic_key_hash = compute_identity_hash(
            schema=ADAPTER_EFFECT_KEY_SCHEMA,
            schema_version=ADAPTER_EFFECT_KEY_SCHEMA_VERSION,
            payload=payload,
        )
        return EffectRequest(
            semantic_key=OpaqueKey(
                f"{ADAPTER_EFFECT_KEY_PREFIX}{semantic_key_hash}"
            ),
            request_identity_hash=compute_identity_hash(
                schema=ADAPTER_EFFECT_SCHEMA,
                schema_version=ADAPTER_EFFECT_SCHEMA_VERSION,
                payload=payload,
            ),
            replay_policy=self._adapter_replay_policy,
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
        request: OptimizationStepRequest,
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
        request: OptimizationStepRequest, output: AdapterOutput
    ) -> None:
        intent_ids = [
            str(intent.intent_id) for intent in output.evaluation_intents
        ]
        if len(set(intent_ids)) != len(intent_ids):
            raise ValueError(
                "Evaluation Intent IDs must be unique within a Step"
            )
        contract = request.step_output_contract
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
            and contract != request.run.record.terminal_output_contract
        ):
            raise ValueError(
                "a COMPLETE Step must use the run terminal output contract"
            )

    @staticmethod
    def _validate_output_candidates(
        request: OptimizationStepRequest,
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
                    diff_check(base=base, proposed=candidate)
                except ValueError as error:
                    raise ValueError(
                        f"every {label} candidate must satisfy the canonical "
                        f"run mutation diff: {error}"
                    ) from error

    @staticmethod
    def _validate_output_intents(
        request: OptimizationStepRequest,
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
        for intent in output.evaluation_intents:
            if intent.run_id != request.run_id:
                raise ValueError("Intent belongs to another optimization run")
            if intent.step_index != request.step_index:
                raise ValueError("Intent belongs to another optimization step")
            exact_candidate = allowed.get(str(intent.candidate.identity_hash))
            if exact_candidate is None or exact_candidate != intent.candidate:
                raise ValueError(
                    "Intent candidate is not an exact Step output candidate"
                )
            if (
                intent.target_eval_config
                != intent.evaluation_binding.eval_config
            ):
                raise ValueError(
                    "Intent target Eval Config must match its exact "
                    "Evaluation Binding"
                )
            if intent.evaluation_binding.role is EvaluationRole.INTERNAL:
                if (
                    reward_policy is None
                    or intent.expected_reward_policy_hash
                    != reward_policy.identity_hash()
                ):
                    raise ValueError(
                        "Intent must expect the exact run Reward Policy"
                    )
            elif intent.expected_reward_policy_hash is not None:
                raise ValueError(
                    "official Intent must not expect a Reward Policy"
                )

    def _resolve_intents(
        self,
        request: OptimizationStepRequest,
        output: AdapterOutput,
        proposed: tuple[CandidateRef, ...],
        accepted: tuple[CandidateRef, ...],
    ) -> tuple[IntentResolution, ...]:
        if not output.evaluation_intents:
            return ()
        self._validate_output_intents(request, output)
        if self._evaluation_service is None:
            raise ValueError(
                "proposal-only Step with Intents requires EvaluationService"
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
        for intent in output.evaluation_intents:
            if intent.run_id != request.run_id:
                raise ValueError("Intent belongs to another optimization run")
            if intent.step_index != request.step_index:
                raise ValueError("Intent belongs to another optimization step")
            exact_candidate = allowed.get(str(intent.candidate.identity_hash))
            if exact_candidate is None or exact_candidate != intent.candidate:
                raise ValueError(
                    "Intent candidate is not an exact Step output candidate"
                )
            self._persist_intent_records(intent)
            resolutions.append(
                self._resolve_one_intent(
                    request=request,
                    intent=intent,
                )
            )
        return tuple(resolutions)

    def _resolve_one_intent(
        self,
        *,
        request: OptimizationStepRequest,
        intent: EvaluationIntent,
    ) -> IntentResolution:
        if (
            self._evaluation_service is None
            or self._evaluation_replay_policy is None
        ):
            raise RuntimeError("EvaluationService is not configured")
        if (
            self._evaluation_service.replay_policy
            is not self._evaluation_replay_policy
        ):
            raise ValueError(
                "EvaluationService replay_policy changed after construction"
            )
        effect_request = self._intent_effect_request(request, intent)
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
            self._validate_resolution(intent, resolution)
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
            return resolution
        lease = self._acquired_lease(acquisition)
        with self._effect_authority.maintain(
            lease, lease_duration=self._lease_duration
        ) as maintenance:
            raw = self._evaluation_service.resolve_evaluation_intent(intent)
            resolution = IntentResolution.model_validate(
                raw.model_dump(mode="json")
            )
            self._validate_resolution(intent, resolution)
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
        request: OptimizationStepRequest,
        intent: EvaluationIntent,
    ) -> EffectRequest:
        if self._evaluation_replay_policy is None:
            raise RuntimeError("EvaluationService is not configured")
        # Persisted-format contract: key schema, version, prefix, and payload
        # literals are pinned by golden tests.
        key_payload = {
            "step_request_ref": step_request_reference(
                request
            ).record_ref.model_dump(mode="json"),
            "intent_id": intent.intent_id,
        }
        semantic_key_hash = compute_identity_hash(
            schema=INTENT_EFFECT_KEY_SCHEMA,
            schema_version=INTENT_EFFECT_KEY_SCHEMA_VERSION,
            payload=key_payload,
        )
        return EffectRequest(
            semantic_key=OpaqueKey(
                f"{INTENT_EFFECT_KEY_PREFIX}{semantic_key_hash}"
            ),
            request_identity_hash=compute_identity_hash(
                schema=INTENT_EFFECT_SCHEMA,
                schema_version=INTENT_EFFECT_SCHEMA_VERSION,
                payload={
                    "intent": intent.model_dump(mode="json"),
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

    @staticmethod
    def _validate_resolution(
        intent: EvaluationIntent,
        resolution: IntentResolution,
    ) -> None:
        if resolution.intent != intent:
            raise ValueError("EvaluationService resolved another exact Intent")
        if resolution.resolved_eval_config != intent.target_eval_config:
            raise ValueError(
                "Intent Resolution used another exact Eval Config"
            )

    def terminalize(
        self,
        *,
        run: OptimizationRunRef,
        step_results: tuple[OptimizationStepResultRef, ...],
        cost: dict[str, object] | None = None,
    ) -> tuple[OptimizationResult, TypedRef]:
        """Persist and bind the one terminal Optimization Result."""
        if self._bound_run is None:
            raise ValueError("bind_run must be called before terminalize")
        exact_run = OptimizationRunRef.model_validate(
            run.model_dump(mode="json")
        )
        if exact_run != self._bound_run:
            raise ValueError(
                "terminalize run differs from the bound exact run"
            )
        if not step_results:
            raise ValueError("terminalize requires at least one Step Result")
        exact_step_results = tuple(
            OptimizationStepResultRef.model_validate(
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
            raise OptimizationResultConflictError(
                run_id=run_id,
                existing=existing_ref,
                requested=requested_ref,
            )

        result_ref = self._put(
            OPTIMIZATION_RESULT_SCHEMA, result.record_content()
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
            raise OptimizationResultConflictError(
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
                raise OptimizationResultConflictError(
                    run_id=run_id,
                    existing=bound,
                    requested=result_ref,
                )
            return replay, bound
        return result, result_ref

    def _assemble_terminal(
        self,
        *,
        run: OptimizationRunRef,
        step_results: tuple[OptimizationStepResultRef, ...],
        cost: dict[str, object],
    ) -> OptimizationResult:
        run_id = str(run.record.run_id)
        results: list[OptimizationStepResult] = []
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
                OptimizationProposal(candidate=candidate)
                for candidate in last.accepted_candidates
            )
        )
        return OptimizationResult(
            run=run,
            proposals=proposals,
            step_results=step_results,
            cost=cost,
            terminal_failure=last.terminal_failure,
        )

    @staticmethod
    def carry_budget_forward(
        prior: OptimizationStepResult,
    ) -> BudgetState:
        return prior.budget
