"""Validated, admitted, and fenced execution of exact Tool Calls."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from types import MappingProxyType
from typing import Protocol
from uuid import uuid4

from whetstone.evaluation_role import EvaluationRole
from whetstone.optimization.effect_authority import (
    AcquireOutcome,
    EffectAuthority,
    ReplayPolicy,
)
from whetstone.optimization.identity import (
    IdentityHash,
    ImmutableJsonObject,
    TerminalFailure,
    TypedRef,
)
from whetstone.optimization.reward import (
    RewardPolicy,
    apply_reward_policy,
    reward_reference,
)
from whetstone.optimization.tool_store import (
    ToolCallState,
    ToolCallStore,
    tool_effect_request,
)
from whetstone.optimization.tools import (
    RefusalClass,
    RuntimeToolHandle,
    ToolCall,
    ToolCapacityBinding,
    ToolConfig,
    ToolRefusal,
    ToolResult,
    tool_call_reference,
)

__all__ = [
    "EvaluatingToolExecutor",
    "ToolEvaluation",
    "ToolEvaluationError",
    "ToolEvaluator",
    "ToolExecutionBusyError",
    "ToolExecutionConflictError",
    "ToolExecutionRecoveryRequiredError",
    "ToolValidationError",
]


class ToolValidationError(ValueError):
    """Pure pre-admission validation refused one Tool Call."""


class ToolEvaluationError(RuntimeError):
    """An expected evaluator exhaustion with shared terminal evidence."""

    def __init__(self, failure: TerminalFailure) -> None:
        self.failure = failure
        super().__init__(failure.message)


class ToolExecutionBusyError(RuntimeError):
    """Another owner holds the unexpired exact execution lease."""

    def __init__(self, *, busy_expires_at: datetime) -> None:
        self.busy_expires_at = busy_expires_at
        super().__init__(
            "Tool execution is busy until "
            f"{busy_expires_at.isoformat(timespec='microseconds')}"
        )


class ToolExecutionConflictError(RuntimeError):
    """The semantic execution key is bound to another exact request."""


class ToolExecutionRecoveryRequiredError(RuntimeError):
    """A non-redrivable expired Tool execution needs operator recovery."""

    def __init__(self, failure: TerminalFailure) -> None:
        self.failure = failure
        super().__init__(failure.message)


@dataclass(frozen=True, slots=True)
class ToolEvaluation:
    """Evaluator output before Reward scalarization and ToolResult creation."""

    output: ImmutableJsonObject
    rollout_refs: tuple[TypedRef, ...]
    aggregates: Mapping[str, float | None]
    eval_config_hash: IdentityHash

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "aggregates", MappingProxyType(dict(self.aggregates))
        )


class ToolEvaluator(Protocol):
    """Pure validation followed by internal-role effectful evaluation."""

    def validate(self, call: ToolCall, config: ToolConfig) -> None:
        """Validate without performing evaluation or consuming capacity."""

    def evaluate(self, call: ToolCall, config: ToolConfig) -> ToolEvaluation:
        """Perform one accepted internal evaluation.

        Expected exhaustion raises :class:`ToolEvaluationError`. Unexpected
        exceptions represent process/worker failure and intentionally leave
        the acquired lease unterminated. Lease loss fences authoritative
        completion, but cannot guarantee cancellation of arbitrary external
        work already dispatched by an evaluator.
        """


class EvaluatingToolExecutor:
    """One paved path from pure validation to fenced terminal persistence."""

    def __init__(
        self,
        evaluator: ToolEvaluator,
        reward_policy: RewardPolicy,
        effect_authority: EffectAuthority,
        *,
        owner_id: str,
        replay_policy: ReplayPolicy,
        lease_duration: timedelta = timedelta(minutes=5),
    ) -> None:
        if not owner_id:
            raise ValueError("owner_id must be non-empty")
        if lease_duration <= timedelta(0):
            raise ValueError("lease_duration must be positive")
        self._evaluator = evaluator
        self._reward_policy = reward_policy
        self._effect_authority = effect_authority
        self._owner_id = owner_id
        self._replay_policy = replay_policy
        self._lease_duration = lease_duration

    def runtime_handle(
        self,
        config: ToolConfig,
        store: ToolCallStore,
        binding: ToolCapacityBinding,
    ) -> RuntimeToolHandle:
        validated_config = ToolConfig.model_validate(
            config.model_dump(mode="json")
        )
        validated_binding = ToolCapacityBinding.model_validate(
            binding.model_dump(mode="json")
        )
        effect_authority = store.effect_authority
        if effect_authority is not self._effect_authority:
            raise ValueError(
                "Tool executor and Tool Call Store must share one exact "
                "EffectAuthority instance"
            )
        if (
            validated_config.reward_policy_hash
            != self._reward_policy.identity_hash()
        ):
            raise ValueError(
                "Tool Config reward_policy_hash does not match the executor's "
                "exact Reward Policy Identity Hash"
            )
        configured_replay = (
            ReplayPolicy.IDEMPOTENT
            if validated_config.idempotent_replay
            else ReplayPolicy.NO_REDRIVE
        )
        if self._replay_policy is not configured_replay:
            raise ValueError(
                "explicit ReplayPolicy disagrees with the exact Tool Config"
            )

        def execute(raw_call: ToolCall) -> ToolResult:
            call = ToolCall.model_validate(raw_call.model_dump(mode="json"))
            if call.tool_config.record != validated_config:
                raise ValueError(
                    "Tool Call must cite the runtime handle's exact Config"
                )

            existing = store.get(call)
            if existing is not None:
                if existing.state is ToolCallState.REFUSED:
                    return store.load_terminal_result(existing)
                entry = existing
            else:
                try:
                    self._evaluator.validate(call, validated_config)
                except ToolValidationError as exc:
                    entry = store.refuse(
                        call,
                        validated_config,
                        refusal=ToolRefusal(
                            refusal_class=RefusalClass.VALIDATION,
                            reason=str(exc),
                        ),
                    )
                    return store.load_terminal_result(entry)
                entry = store.admit(call, validated_config)
                if entry.state is ToolCallState.REFUSED:
                    return store.load_terminal_result(entry)

            request = tool_effect_request(call)
            acquisition = effect_authority.acquire(
                request,
                owner_id=self._owner_id,
                attempt_id=uuid4().hex,
                lease_duration=self._lease_duration,
            )
            if acquisition.outcome is AcquireOutcome.SUCCEEDED:
                terminal = acquisition.terminal
                if terminal is None or terminal.result_ref is None:
                    raise RuntimeError(
                        "succeeded Tool effect has no exact Tool Result ref"
                    )
                result = store.load_result(
                    terminal.result_ref, expected_call=call
                )
                if result.terminal_failure is not None:
                    raise RuntimeError(
                        "succeeded Tool effect references a failed Tool Result"
                    )
                completed = store.complete(
                    result,
                    terminal=terminal,
                )
                return store.load_terminal_result(completed)
            if acquisition.outcome is AcquireOutcome.FAILED:
                terminal = acquisition.terminal
                if (
                    terminal is None
                    or terminal.result_ref is None
                    or terminal.failure is None
                ):
                    raise RuntimeError(
                        "failed Tool effect has incomplete terminal evidence"
                    )
                result = store.load_result(
                    terminal.result_ref, expected_call=call
                )
                if result.terminal_failure != terminal.failure:
                    raise RuntimeError(
                        "failed Tool effect and Tool Result disagree"
                    )
                completed = store.complete(
                    result,
                    terminal=terminal,
                )
                return store.load_terminal_result(completed)
            if acquisition.outcome is AcquireOutcome.BUSY:
                if acquisition.busy_expires_at is None:
                    raise RuntimeError("busy Tool effect has no expiration")
                raise ToolExecutionBusyError(
                    busy_expires_at=acquisition.busy_expires_at
                )
            if acquisition.outcome is AcquireOutcome.REQUEST_CONFLICT:
                raise ToolExecutionConflictError(
                    "Tool execution key is bound to another exact request"
                )
            if acquisition.outcome is AcquireOutcome.RECOVERY_REQUIRED:
                terminal = acquisition.terminal
                if terminal is None or terminal.failure is None:
                    raise RuntimeError(
                        "recovery-required Tool effect has no failure"
                    )
                raise ToolExecutionRecoveryRequiredError(terminal.failure)
            lease = acquisition.lease
            if (
                acquisition.outcome is not AcquireOutcome.ACQUIRED
                or lease is None
            ):
                raise RuntimeError("unrecognized Tool effect acquisition")
            entry_ordinal = entry.capacity_debit_ordinal
            if entry_ordinal is None:
                raise RuntimeError(
                    "accepted Tool Call entry has no capacity ordinal"
                )
            exact_ordinal = int(entry_ordinal)

            with effect_authority.maintain(
                lease, lease_duration=self._lease_duration
            ) as maintenance:
                try:
                    evaluation = self._evaluator.evaluate(
                        call, validated_config
                    )
                    result = self._successful_result(
                        call=call,
                        entry_ordinal=exact_ordinal,
                        evaluation=evaluation,
                        config=validated_config,
                    )
                except ToolEvaluationError as exc:
                    result = ToolResult(
                        call=tool_call_reference(call),
                        terminal_failure=exc.failure,
                        provenance_ordinal=exact_ordinal,
                    )
                    result_ref = store.persist_result(result)
                    terminal_failure = exc.failure
                else:
                    result_ref = store.persist_result(result)
                    terminal_failure = None
                if terminal_failure is None:
                    terminal = maintenance.succeed(result_ref=result_ref)
                else:
                    terminal = maintenance.fail(
                        result_ref=result_ref,
                        failure=terminal_failure,
                    )
            completed = store.complete(
                result,
                terminal=terminal,
            )
            return store.load_terminal_result(completed)

        return RuntimeToolHandle(validated_config, validated_binding, execute)

    def _successful_result(
        self,
        *,
        call: ToolCall,
        entry_ordinal: int,
        evaluation: ToolEvaluation,
        config: ToolConfig,
    ) -> ToolResult:
        if evaluation.eval_config_hash != config.eval_config_identity_hash:
            raise ToolEvaluationError(
                TerminalFailure(
                    code="tool_eval_config_mismatch",
                    message=(
                        "Tool evaluation bound a different exact Eval Config"
                    ),
                    details={
                        "expected": config.eval_config_identity_hash,
                        "actual": evaluation.eval_config_hash,
                    },
                )
            )
        try:
            reward = apply_reward_policy(
                self._reward_policy,
                aggregates=evaluation.aggregates,
                evidence_role=EvaluationRole.INTERNAL,
                evidence_refs=evaluation.rollout_refs,
                provenance_ordinal=entry_ordinal,
            )
            return ToolResult(
                call=tool_call_reference(call),
                output=evaluation.output,
                evaluation_evidence_refs=evaluation.rollout_refs,
                reward=reward_reference(reward),
                provenance_ordinal=entry_ordinal,
            )
        except ValueError as exc:
            raise ToolEvaluationError(
                TerminalFailure(
                    code="tool_evaluation_contract",
                    message=(
                        "Tool evaluation produced invalid deterministic "
                        "result evidence"
                    ),
                    details={"error": str(exc)},
                )
            ) from exc
