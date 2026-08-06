from __future__ import annotations

from threading import RLock
from typing import Any

from dr_store import BindingConflictError, ObjectStore
from pydantic import BaseModel, ConfigDict, model_validator

from whetstone.core.identity import (
    NonNegativeInt,
    TypedRef,
    compute_identity_hash,
    typed_ref_for_record,
)
from whetstone.optimization.contracts import (
    OptimizationStepRequestRef,
    ToolEvidence,
)
from whetstone.optimization.tools.admission import ToolCallState
from whetstone.optimization.tools.contracts import (
    RuntimeToolHandle,
    ToolCall,
    ToolCallRef,
    ToolResult,
    ToolResultRef,
    tool_call_reference,
    tool_result_reference,
)
from whetstone.optimization.tools.facade import ToolCallStore

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


__all__ = ["IssuedToolCallConflictError"]
