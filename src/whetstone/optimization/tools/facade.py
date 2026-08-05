"""Atomic admission and terminal persistence for exact Tool Calls."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dr_store import ObjectStore

from whetstone.core.effects.authority import (
    EffectAuthority,
    EffectTerminal,
    TerminalOutcome,
)
from whetstone.core.identity import (
    TypedRef,
    typed_ref_for_record,
)
from whetstone.experiment.reward import REWARD_SCHEMA
from whetstone.optimization.tools._memory import _MemoryAdmissionBackend
from whetstone.optimization.tools._postgres import (
    _Connect,
    _PostgreSQLAdmissionBackend,
)
from whetstone.optimization.tools._sqlite import _SQLiteAdmissionBackend
from whetstone.optimization.tools.admission import (
    ToolCallState,
    ToolCallStoreConflictError,
    ToolCallStoreEntry,
    _AdmissionBackend,
    _capacity_scope_key,
    _SQLiteTransactionObserver,
    _validate_admission_attempt,
    tool_effect_request,
)
from whetstone.optimization.tools.contracts import (
    TOOL_CALL_SCHEMA,
    TOOL_CONFIG_SCHEMA,
    TOOL_DEFINITION_SCHEMA,
    TOOL_RESULT_SCHEMA,
    RefusalClass,
    ToolCall,
    ToolCallRef,
    ToolCapacityBinding,
    ToolCapacityScope,
    ToolConfig,
    ToolRefusal,
    ToolResult,
    tool_call_reference,
    tool_config_reference,
    tool_result_reference,
)


class ToolAdmissionAuthority:
    """Atomic call-decision and scoped-capacity authority."""

    def __init__(self, backend: _AdmissionBackend) -> None:
        self._backend = backend
        self._backend.initialize()

    @classmethod
    def memory(cls) -> ToolAdmissionAuthority:
        return cls(_MemoryAdmissionBackend())

    @classmethod
    def sqlite(
        cls,
        path: str | Path,
        *,
        _transaction_observer: _SQLiteTransactionObserver | None = None,
    ) -> ToolAdmissionAuthority:
        return cls(
            _SQLiteAdmissionBackend(
                path,
                transaction_observer=_transaction_observer,
            )
        )

    @classmethod
    def postgresql(
        cls,
        dsn: str,
        *,
        _connect: _Connect | None = None,
    ) -> ToolAdmissionAuthority:
        return cls(_PostgreSQLAdmissionBackend(dsn, connect=_connect))

    def admit(
        self,
        *,
        accepted: ToolCallStoreEntry,
        refused: ToolCallStoreEntry,
        max_accepted_calls: int,
    ) -> ToolCallStoreEntry:
        _validate_admission_attempt(
            accepted=accepted,
            refused=refused,
            max_accepted_calls=max_accepted_calls,
        )
        return self._backend.admit(
            accepted=accepted,
            refused=refused,
            max_accepted_calls=max_accepted_calls,
        )

    def refuse(self, entry: ToolCallStoreEntry) -> ToolCallStoreEntry:
        if entry.state is not ToolCallState.REFUSED:
            raise ValueError("refusal candidate must be refused")
        if (
            entry.refusal is not None
            and entry.refusal.refusal_class is RefusalClass.CAPACITY
        ):
            raise ValueError("capacity refusal is owned by admission")
        return self._backend.refuse(entry)

    def get(
        self, store_namespace_key: str, call_id: str
    ) -> ToolCallStoreEntry | None:
        return self._backend.get(store_namespace_key, call_id)

    def complete(self, entry: ToolCallStoreEntry) -> ToolCallStoreEntry:
        if entry.state is not ToolCallState.COMPLETED:
            raise ValueError("completion candidate must be completed")
        return self._backend.complete(entry)

    def accepted_count(
        self,
        *,
        store_namespace_key: str,
        tool_config_hash: str,
        capacity_scope: ToolCapacityScope,
        capacity_scope_id: str,
    ) -> int:
        return self._backend.accepted_count(
            store_namespace_key=store_namespace_key,
            tool_config_hash=tool_config_hash,
            capacity_scope=capacity_scope,
            capacity_scope_id=capacity_scope_id,
        )

    def close(self) -> None:
        self._backend.close()


class ToolCallStore:
    """Persist exact Tool records and delegate mutable decisions atomically."""

    def __init__(
        self,
        store: ObjectStore,
        admission_authority: ToolAdmissionAuthority,
        effect_authority: EffectAuthority,
    ) -> None:
        self._store = store
        self._admission = admission_authority
        self._effect_authority = effect_authority

    def _put_exact(self, schema: str, content: dict[str, Any]) -> TypedRef:
        expected = typed_ref_for_record(schema, content)
        reference, _status = self._store.put(schema, content)
        persisted = TypedRef(
            schema_name=reference.schema,
            content_hash=reference.content_hash,
        )
        if persisted != expected:
            raise ValueError(f"persisted {schema} ref failed validation")
        return persisted

    def _persist_call_chain(
        self, call: ToolCall, config: ToolConfig
    ) -> ToolCallRef:
        validated_config = ToolConfig.model_validate(
            config.model_dump(mode="json")
        )
        validated_call = ToolCall.model_validate(call.model_dump(mode="json"))
        _capacity_scope_key(validated_call.capacity_binding)
        config_ref = tool_config_reference(validated_config)
        if validated_call.tool_config != config_ref:
            raise ValueError(
                "Tool Call must cite the exact supplied Tool Config"
            )
        definition = config_ref.record.definition
        if (
            self._put_exact(
                TOOL_DEFINITION_SCHEMA, definition.record.record_content()
            )
            != definition.record_ref
        ):
            raise ValueError("persisted Tool Definition ref failed validation")
        if (
            self._put_exact(
                TOOL_CONFIG_SCHEMA, validated_config.record_content()
            )
            != config_ref.record_ref
        ):
            raise ValueError("persisted Tool Config ref failed validation")
        call_ref = tool_call_reference(validated_call)
        if (
            self._put_exact(TOOL_CALL_SCHEMA, validated_call.record_content())
            != call_ref.record_ref
        ):
            raise ValueError("persisted Tool Call ref failed validation")
        return call_ref

    def get(self, call: ToolCall) -> ToolCallStoreEntry | None:
        validated = ToolCall.model_validate(call.model_dump(mode="json"))
        _capacity_scope_key(validated.capacity_binding)
        existing = self._admission.get(
            str(validated.store_namespace_key), str(validated.call_id)
        )
        if existing is not None and existing.tool_call != tool_call_reference(
            validated
        ):
            raise ToolCallStoreConflictError(
                existing=existing,
                attempted_state=ToolCallState.ACCEPTED,
                detail="the existing key cites a different exact Tool Call",
            )
        return existing

    def admit(self, call: ToolCall, config: ToolConfig) -> ToolCallStoreEntry:
        call_ref = self._persist_call_chain(call, config)
        config_ref = tool_config_reference(config)
        common: dict[str, Any] = {
            "tool_call": call_ref,
            "tool_config": config_ref,
            "store_namespace_key": call_ref.record.store_namespace_key,
            "capacity_scope": call_ref.record.capacity_scope,
            "capacity_scope_id": call_ref.record.capacity_scope_id,
        }
        accepted = ToolCallStoreEntry(
            **common,
            state=ToolCallState.ACCEPTED,
            capacity_debit_ordinal=1,
        )
        refusal = ToolRefusal(
            refusal_class=RefusalClass.CAPACITY,
            reason=(
                "Tool Capacity exhausted: "
                f"{config.capacity.max_accepted_calls}/"
                f"{config.capacity.max_accepted_calls} accepted calls consumed"
            ),
        )
        refused_result = ToolResult(call=call_ref, refusal=refusal)
        refused_ref = self.persist_result(refused_result)
        refused = ToolCallStoreEntry(
            **common,
            state=ToolCallState.REFUSED,
            refusal=refusal,
            tool_result_ref=refused_ref,
        )
        return self._admission.admit(
            accepted=accepted,
            refused=refused,
            max_accepted_calls=int(config.capacity.max_accepted_calls),
        )

    def refuse(
        self,
        call: ToolCall,
        config: ToolConfig,
        *,
        refusal: ToolRefusal,
    ) -> ToolCallStoreEntry:
        if refusal.refusal_class is RefusalClass.CAPACITY:
            raise ValueError("capacity refusal is owned by admission")
        call_ref = self._persist_call_chain(call, config)
        result = ToolResult(call=call_ref, refusal=refusal)
        result_ref = self.persist_result(result)
        entry = ToolCallStoreEntry(
            tool_call=call_ref,
            tool_config=call_ref.record.tool_config,
            store_namespace_key=call_ref.record.store_namespace_key,
            capacity_scope=call_ref.record.capacity_scope,
            capacity_scope_id=call_ref.record.capacity_scope_id,
            state=ToolCallState.REFUSED,
            refusal=refusal,
            tool_result_ref=result_ref,
        )
        return self._admission.refuse(entry)

    def persist_result(self, result: ToolResult) -> TypedRef:
        validated = ToolResult.model_validate(result.model_dump(mode="json"))
        if validated.reward is not None:
            persisted_reward = self._put_exact(
                REWARD_SCHEMA,
                validated.reward.record.record_content(),
            )
            if persisted_reward != validated.reward.record_ref:
                raise ValueError("persisted Reward ref failed validation")
        expected = tool_result_reference(validated)
        persisted = self._put_exact(
            TOOL_RESULT_SCHEMA, validated.record_content()
        )
        if persisted != expected.record_ref:
            raise ValueError("persisted Tool Result ref failed validation")
        return persisted

    def complete(
        self,
        result: ToolResult,
        *,
        terminal: EffectTerminal | None = None,
    ) -> ToolCallStoreEntry:
        validated = ToolResult.model_validate(result.model_dump(mode="json"))
        if validated.refusal is not None:
            raise ValueError("a refused Tool Result is terminal at admission")
        existing = self.get(validated.call.record)
        if existing is None:
            raise ValueError("the exact Tool Call was never admitted")
        self._validate_result_authority(validated, existing)
        result_ref = tool_result_reference(validated).record_ref
        if terminal is None:
            raise ValueError(
                "completion requires an exact authoritative "
                "EffectTerminal proof"
            )
        exact_terminal = EffectTerminal.model_validate_json(
            terminal.model_dump_json()
        )
        expected_request = tool_effect_request(validated.call.record)
        if exact_terminal.request != expected_request:
            raise ValueError(
                "effect terminal belongs to another exact Tool request"
            )
        if exact_terminal.result_ref != result_ref:
            raise ValueError(
                "effect terminal references another exact Tool Result"
            )
        if (
            exact_terminal.outcome is TerminalOutcome.SUCCEEDED
            and validated.terminal_failure is not None
        ):
            raise ValueError(
                "succeeded effect terminal references a failed Tool Result"
            )
        if exact_terminal.outcome is TerminalOutcome.FAILED:
            if exact_terminal.failure != validated.terminal_failure:
                raise ValueError(
                    "failed effect terminal and Tool Result disagree"
                )
        elif exact_terminal.outcome is not TerminalOutcome.SUCCEEDED:
            raise ValueError(
                "recovery-required effects have no completed Tool Result"
            )
        authoritative_terminal = self._effect_authority.verify_terminal(
            exact_terminal
        )
        self.persist_result(validated)
        completed = ToolCallStoreEntry(
            tool_call=validated.call,
            tool_config=validated.tool_config,
            store_namespace_key=validated.store_namespace_key,
            capacity_scope=validated.call.record.capacity_scope,
            capacity_scope_id=validated.call.record.capacity_scope_id,
            state=ToolCallState.COMPLETED,
            capacity_debit_ordinal=existing.capacity_debit_ordinal,
            tool_result_ref=result_ref,
            effect_terminal=authoritative_terminal,
        )
        return self._admission.complete(completed)

    def load_terminal_result(self, entry: ToolCallStoreEntry) -> ToolResult:
        if entry.tool_result_ref is None:
            raise ValueError("entry has no terminal Tool Result")
        durable_entry = self._admission.get(
            str(entry.store_namespace_key),
            str(entry.call_id),
        )
        if durable_entry != entry:
            raise ValueError(
                "terminal entry does not match the durable admission decision"
            )
        if entry.state is ToolCallState.COMPLETED:
            if entry.effect_terminal is None:
                raise ValueError(
                    "completed entry has no exact effect terminal"
                )
            self._effect_authority.verify_terminal(entry.effect_terminal)
        content = self._store.get(entry.tool_result_ref.reference)
        result = ToolResult.model_validate(content)
        expected = tool_result_reference(result)
        if expected.record_ref != entry.tool_result_ref:
            raise ValueError(
                "persisted Tool Result ref does not match the terminal entry"
            )
        if result.call != entry.tool_call:
            raise ValueError(
                "persisted Tool Result belongs to a different exact Tool Call"
            )
        if entry.state is ToolCallState.REFUSED:
            if result.refusal != entry.refusal:
                raise ValueError(
                    "persisted Tool Result and admission refusal disagree"
                )
        elif entry.state is ToolCallState.COMPLETED:
            terminal = entry.effect_terminal
            if terminal is None:
                raise AssertionError(
                    "completed entry validation lost its effect terminal"
                )
            expected_outcome = (
                TerminalOutcome.FAILED
                if result.terminal_failure is not None
                else TerminalOutcome.SUCCEEDED
            )
            if terminal.outcome is not expected_outcome:
                raise ValueError(
                    "persisted Tool Result and effect terminal outcome "
                    "disagree"
                )
            if terminal.failure != result.terminal_failure:
                raise ValueError(
                    "persisted Tool Result and effect terminal failure "
                    "disagree"
                )
            self._validate_result_authority(result, entry)
        return result

    def _validate_result_authority(
        self,
        result: ToolResult,
        entry: ToolCallStoreEntry,
    ) -> None:
        if result.refusal is not None:
            raise ValueError(
                "a refused Tool Result has no capacity provenance"
            )
        ordinal = entry.capacity_debit_ordinal
        if ordinal is None:
            raise ValueError(
                "a non-refused Tool Result requires an admitted capacity "
                "ordinal"
            )
        capacity_scope, capacity_scope_id = _capacity_scope_key(
            entry.tool_call.record.capacity_binding
        )
        accepted_count = self._admission.accepted_count(
            store_namespace_key=str(entry.store_namespace_key),
            tool_config_hash=str(entry.tool_config_hash),
            capacity_scope=capacity_scope,
            capacity_scope_id=capacity_scope_id,
        )
        if not 1 <= ordinal <= accepted_count:
            raise ValueError(
                "completed entry capacity ordinal is outside the durable "
                "admission projection"
            )
        result_ordinal = result.provenance_ordinal
        if result_ordinal is None or result_ordinal == 0:
            raise ValueError(
                "a non-refused Tool Result requires a positive provenance "
                "ordinal"
            )
        if result_ordinal != ordinal:
            raise ValueError(
                "Tool Result provenance ordinal disagrees with the durable "
                "admission capacity ordinal"
            )

    def load_result(
        self, result_ref: TypedRef, *, expected_call: ToolCall
    ) -> ToolResult:
        """Load one exact Tool Result and validate its complete call chain."""
        if result_ref.schema_name != TOOL_RESULT_SCHEMA:
            raise ValueError("effect terminal references a non-Tool Result")
        content = self._store.get(result_ref.reference)
        result = ToolResult.model_validate(content)
        if tool_result_reference(result).record_ref != result_ref:
            raise ValueError("effect terminal Tool Result ref is not exact")
        if result.call != tool_call_reference(expected_call):
            raise ValueError(
                "effect terminal Tool Result belongs to another Tool Call"
            )
        return result

    def accepted_count(
        self,
        config: ToolConfig,
        binding: ToolCapacityBinding,
    ) -> int:
        validated = ToolConfig.model_validate(config.model_dump(mode="json"))
        exact_binding = ToolCapacityBinding.model_validate(
            binding.model_dump(mode="json")
        )
        capacity_scope, capacity_scope_id = _capacity_scope_key(exact_binding)
        if capacity_scope is not validated.capacity.scope:
            raise ValueError(
                "Tool Capacity binding scope must match the exact Tool Config"
            )
        return self._admission.accepted_count(
            store_namespace_key=str(validated.store_namespace_key),
            tool_config_hash=str(validated.identity_hash()),
            capacity_scope=capacity_scope,
            capacity_scope_id=capacity_scope_id,
        )

    @property
    def effect_authority(self) -> EffectAuthority:
        """The sole effect authority allowed to terminalize this store."""
        return self._effect_authority

    def close(self) -> None:
        self._admission.close()


__all__ = ["ToolAdmissionAuthority", "ToolCallStore"]
