from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from threading import Event, Lock

import pytest
from dr_store import ObjectStore, SqliteBackend
from pydantic import ValidationError

from tests.optimization.support import eval_config
from whetstone.core.effects.authority import (
    EffectAuthority,
    EffectLease,
    EffectTerminal,
    ReplayPolicy,
    StaleLeaseError,
    TerminalConflictError,
    TerminalOutcome,
)
from whetstone.core.identity import (
    IdentityHash,
    ImmutableJsonObject,
    NonEmptyId,
    TerminalFailure,
    typed_ref_for_record,
)
from whetstone.experiment.reward import RewardPolicy, RewardTerm
from whetstone.optimization.contracts import ToolEvidence
from whetstone.optimization.tools.admission import (
    ToolCallState,
    ToolCallStoreEntry,
    tool_effect_request,
)
from whetstone.optimization.tools.contracts import (
    RUN_CAPACITY_SUBJECT_SCHEMA,
    RefusalClass,
    ToolCall,
    ToolCapacity,
    ToolCapacityBinding,
    ToolCapacityScope,
    ToolConfig,
    ToolDefinition,
    ToolResult,
    tool_call_reference,
    tool_config_reference,
    tool_definition_reference,
    tool_result_reference,
)
from whetstone.optimization.tools.execution import (
    EvaluatingToolExecutor,
    ToolEvaluation,
    ToolEvaluationError,
    ToolExecutionBusyError,
    ToolExecutionConflictError,
    ToolExecutionRecoveryRequiredError,
    ToolValidationError,
)
from whetstone.optimization.tools.facade import (
    ToolAdmissionAuthority,
    ToolCallStore,
)

FULL_A = IdentityHash("a" * 64)


def _policy() -> RewardPolicy:
    return RewardPolicy(
        policy_name="tool",
        terms=(RewardTerm(name="score", weight=1.0),),
    )


def _config(
    policy: RewardPolicy,
    *,
    capacity: int = 2,
    idempotent_replay: bool = True,
    store_namespace_key: str = "tool-ns",
) -> ToolConfig:
    definition = ToolDefinition(
        tool_name="evaluate_candidate",
        input_fields=("model_route", "template"),
        output_fields=("generation_refs", "accepted_ordinal"),
    )
    return ToolConfig(
        definition=tool_definition_reference(definition),
        endpoint_key="evaluate_candidate",
        eval_config=eval_config(FULL_A),
        reward_policy_hash=policy.identity_hash(),
        capacity=ToolCapacity(
            max_accepted_calls=capacity,
            scope=ToolCapacityScope.RUN,
        ),
        store_namespace_key=store_namespace_key,
        idempotent_replay=idempotent_replay,
    )


def _call(
    config: ToolConfig,
    call_id: str,
    *,
    template: str = "template",
) -> ToolCall:
    return ToolCall(
        call_id=call_id,
        tool_config=tool_config_reference(config),
        capacity_binding=_binding(),
        args={"model_route": "route", "template": template},
    )


def _binding() -> ToolCapacityBinding:
    return ToolCapacityBinding(
        scope=ToolCapacityScope.RUN,
        subject_ref=typed_ref_for_record(
            RUN_CAPACITY_SUBJECT_SCHEMA, {"run_id": "run-1"}
        ),
    )


def _store(
    database,
    *,
    effect_authority: EffectAuthority | None = None,
) -> ToolCallStore:
    return ToolCallStore(
        ObjectStore(SqliteBackend(database)),
        ToolAdmissionAuthority.sqlite(database),
        effect_authority or EffectAuthority.memory(),
    )


@dataclass
class MutableClock:
    current: datetime

    def __call__(self) -> datetime:
        return self.current


@dataclass
class CountingEvaluator:
    evaluations: int = 0
    validations: int = 0
    failure: TerminalFailure | None = None
    crash: bool = False

    def validate(self, call: ToolCall, config: ToolConfig) -> None:
        del config
        self.validations += 1
        if call.args["template"] == "":
            raise ToolValidationError("template must be non-empty")

    def evaluate(self, call: ToolCall, config: ToolConfig) -> ToolEvaluation:
        del call, config
        self.evaluations += 1
        if self.crash:
            raise RuntimeError("worker exited")
        if self.failure is not None:
            raise ToolEvaluationError(self.failure)
        evidence = typed_ref_for_record(
            "whetstone.test.tool_generation",
            {"ordinal": self.evaluations},
        )
        return ToolEvaluation(
            output=ImmutableJsonObject(
                {
                    "generation_refs": [evidence.model_dump(mode="json")],
                    "accepted_ordinal": self.evaluations,
                }
            ),
            generation_refs=(evidence,),
            aggregates={"score": 1.0},
            eval_config_hash=FULL_A,
        )


def _executor(
    evaluator,
    policy: RewardPolicy,
    authority: EffectAuthority,
    *,
    owner_id: str,
    replay_policy: ReplayPolicy = ReplayPolicy.IDEMPOTENT,
    lease_duration: timedelta = timedelta(seconds=10),
) -> EvaluatingToolExecutor:
    return EvaluatingToolExecutor(
        evaluator,
        policy,
        authority,
        owner_id=owner_id,
        replay_policy=replay_policy,
        lease_duration=lease_duration,
    )


def test_validation_refusal_precedes_admission_and_evaluation(
    tmp_path,
) -> None:
    database = tmp_path / "tool.sqlite"
    policy = _policy()
    config = _config(policy)
    evaluator = CountingEvaluator()
    authority = EffectAuthority.memory()
    store = _store(database, effect_authority=authority)
    handle = _executor(
        evaluator,
        policy,
        authority,
        owner_id="owner",
    ).runtime_handle(config, store, _binding())

    result = handle(_call(config, "bad", template=""))
    assert result.refusal is not None
    assert result.refusal.refusal_class is RefusalClass.VALIDATION
    assert evaluator.evaluations == 0
    assert store.accepted_count(config, _binding()) == 0
    entry = store.get(_call(config, "bad", template=""))
    assert entry is not None and entry.state is ToolCallState.REFUSED
    assert entry.tool_result_ref == tool_result_reference(result).record_ref


def test_success_replay_never_invokes_evaluator_again(tmp_path) -> None:
    database = tmp_path / "tool.sqlite"
    effect_database = tmp_path / "effects.sqlite"
    policy = _policy()
    config = _config(policy)
    call = _call(config, "success")
    evaluator = CountingEvaluator()

    first_authority = EffectAuthority.sqlite(effect_database)
    first = _executor(
        evaluator,
        policy,
        first_authority,
        owner_id="owner-1",
    ).runtime_handle(
        config,
        _store(database, effect_authority=first_authority),
        _binding(),
    )(call)
    assert evaluator.evaluations == 1

    replay_authority = EffectAuthority.sqlite(effect_database)
    replay = _executor(
        evaluator,
        policy,
        replay_authority,
        owner_id="owner-2",
    ).runtime_handle(
        config,
        _store(database, effect_authority=replay_authority),
        _binding(),
    )(call)
    assert replay == first
    assert replay.tool_config == tool_config_reference(config)
    assert evaluator.evaluations == 1


def test_terminal_replay_precedes_changed_validator_policy(tmp_path) -> None:
    database = tmp_path / "tool.sqlite"
    effect_database = tmp_path / "effects.sqlite"
    policy = _policy()
    config = _config(policy)
    call = _call(config, "validator-drift")
    first_evaluator = CountingEvaluator()
    first_authority = EffectAuthority.sqlite(effect_database)
    first = _executor(
        first_evaluator,
        policy,
        first_authority,
        owner_id="owner-1",
    ).runtime_handle(
        config,
        _store(database, effect_authority=first_authority),
        _binding(),
    )(call)

    class RejectingEvaluator(CountingEvaluator):
        def validate(self, call: ToolCall, config: ToolConfig) -> None:
            del call, config
            self.validations += 1
            raise ToolValidationError("current policy rejects this call")

    changed_evaluator = RejectingEvaluator()
    replay_authority = EffectAuthority.sqlite(effect_database)
    replay = _executor(
        changed_evaluator,
        policy,
        replay_authority,
        owner_id="owner-2",
    ).runtime_handle(
        config,
        _store(database, effect_authority=replay_authority),
        _binding(),
    )(call)

    assert replay == first
    assert changed_evaluator.validations == 0
    assert changed_evaluator.evaluations == 0


def test_failure_is_shared_terminal_and_replayed_without_evaluator(
    tmp_path,
) -> None:
    database = tmp_path / "tool.sqlite"
    effect_database = tmp_path / "effects.sqlite"
    policy = _policy()
    config = _config(policy)
    call = _call(config, "failure")
    failure = TerminalFailure(
        code="evaluator_exhausted",
        message="all evaluator attempts failed",
        details={"attempts": 3},
    )
    evaluator = CountingEvaluator(failure=failure)

    first_authority = EffectAuthority.sqlite(effect_database)
    first = _executor(
        evaluator,
        policy,
        first_authority,
        owner_id="owner-1",
    ).runtime_handle(
        config,
        _store(database, effect_authority=first_authority),
        _binding(),
    )(call)
    assert first.terminal_failure == failure
    assert evaluator.evaluations == 1
    assert _store(database).accepted_count(config, _binding()) == 1

    replay_authority = EffectAuthority.sqlite(effect_database)
    replay = _executor(
        evaluator,
        policy,
        replay_authority,
        owner_id="owner-2",
    ).runtime_handle(
        config,
        _store(database, effect_authority=replay_authority),
        _binding(),
    )(call)
    assert replay == first
    assert evaluator.evaluations == 1


def test_invalid_reward_is_terminal_failure_and_replays_exactly(
    tmp_path,
) -> None:
    database = tmp_path / "tool.sqlite"
    effect_database = tmp_path / "effects.sqlite"
    policy = _policy()
    config = _config(policy)
    call = _call(config, "invalid-reward")

    class MissingRewardEvaluator(CountingEvaluator):
        def evaluate(
            self, call: ToolCall, config: ToolConfig
        ) -> ToolEvaluation:
            evaluation = super().evaluate(call, config)
            return ToolEvaluation(
                output=evaluation.output,
                generation_refs=evaluation.generation_refs,
                aggregates={},
                eval_config_hash=evaluation.eval_config_hash,
            )

    evaluator = MissingRewardEvaluator()
    first_authority = EffectAuthority.sqlite(effect_database)
    first = _executor(
        evaluator,
        policy,
        first_authority,
        owner_id="owner-1",
    ).runtime_handle(
        config,
        _store(database, effect_authority=first_authority),
        _binding(),
    )(call)

    assert first.terminal_failure is not None
    assert first.terminal_failure.code == "tool_evaluation_contract"
    assert "missing or invalid" in str(first.terminal_failure.details["error"])
    assert evaluator.evaluations == 1
    entry = _store(database).get(call)
    assert entry is not None and entry.state is ToolCallState.COMPLETED

    evaluator.crash = True
    replay_authority = EffectAuthority.sqlite(effect_database)
    replay = _executor(
        evaluator,
        policy,
        replay_authority,
        owner_id="owner-2",
    ).runtime_handle(
        config,
        _store(database, effect_authority=replay_authority),
        _binding(),
    )(call)
    assert replay == first
    assert evaluator.evaluations == 1


def test_terminal_effect_reconciles_missing_store_completion(tmp_path) -> None:
    database = tmp_path / "tool.sqlite"
    policy = _policy()
    config = _config(policy)
    call = _call(config, "reconcile")
    clock = MutableClock(datetime(2026, 1, 1, tzinfo=UTC))
    authority = EffectAuthority.memory(clock=clock)
    store = _store(database, effect_authority=authority)
    entry = store.admit(call, config)
    assert entry.capacity_debit_ordinal == 1
    result = ToolResult(
        call=tool_call_reference(call),
        output={"generation_refs": [], "accepted_ordinal": 1},
        provenance_ordinal=1,
    )
    result_ref = store.persist_result(result)
    request = tool_effect_request(call)
    acquired = authority.acquire(
        request,
        owner_id="crashed-owner",
        attempt_id="crashed-attempt",
        lease_duration=timedelta(seconds=10),
    )
    assert acquired.lease is not None
    authority.succeed(acquired.lease, result_ref=result_ref)
    still_accepted = store.get(call)
    assert still_accepted is not None
    assert still_accepted.state is ToolCallState.ACCEPTED

    evaluator = CountingEvaluator()
    replay = _executor(
        evaluator,
        policy,
        authority,
        owner_id="replay-owner",
    ).runtime_handle(config, store, _binding())(call)
    assert replay == result
    assert evaluator.evaluations == 0
    reconciled = store.get(call)
    assert reconciled is not None
    assert reconciled.state is ToolCallState.COMPLETED


def test_terminal_replay_rejects_divergent_durable_capacity_authority(
    tmp_path,
) -> None:
    database = tmp_path / "tool.sqlite"
    effect_database = tmp_path / "effects.sqlite"
    policy = _policy()
    config = _config(policy, capacity=1)
    call = _call(config, "poisoned-ordinal")
    evaluator = CountingEvaluator()
    authority = EffectAuthority.sqlite(effect_database)
    first_store = _store(database, effect_authority=authority)

    result = _executor(
        evaluator,
        policy,
        authority,
        owner_id="owner-1",
    ).runtime_handle(config, first_store, _binding())(call)
    assert result.provenance_ordinal == 1
    first_store.close()
    authority.close()

    with sqlite3.connect(database) as connection:
        raw = connection.execute(
            """
            SELECT entry_json FROM whetstone_tool_admission_entry
            WHERE store_namespace_key = ? AND call_id = ?
            """,
            (str(call.store_namespace_key), str(call.call_id)),
        ).fetchone()
        assert raw is not None
        poisoned = json.loads(str(raw[0]))
        poisoned["capacity_debit_ordinal"] = 2
        connection.execute(
            """
            UPDATE whetstone_tool_admission_entry SET entry_json = ?
            WHERE store_namespace_key = ? AND call_id = ?
            """,
            (
                json.dumps(poisoned),
                str(call.store_namespace_key),
                str(call.call_id),
            ),
        )
        cursor = connection.execute(
            """
            UPDATE whetstone_tool_admission_capacity SET consumed = 0
            WHERE store_namespace_key = ?
            """,
            (str(call.store_namespace_key),),
        )
        assert cursor.rowcount == 1

    restarted_authority = EffectAuthority.sqlite(effect_database)
    restarted_store = _store(
        database,
        effect_authority=restarted_authority,
    )
    replay = _executor(
        evaluator,
        policy,
        restarted_authority,
        owner_id="owner-2",
    ).runtime_handle(config, restarted_store, _binding())

    with pytest.raises(ValueError, match="durable admission projection"):
        replay(call)
    assert evaluator.evaluations == 1


def test_effect_semantic_key_is_golden_and_delimiter_collision_safe() -> None:
    policy = _policy()
    first = tool_effect_request(
        _call(
            _config(policy, store_namespace_key="alpha:beta"),
            "gamma",
        ),
    )
    second = tool_effect_request(
        _call(
            _config(policy, store_namespace_key="alpha"),
            "beta:gamma",
        ),
    )

    assert first.semantic_key == (
        "whetstone.tool_execution:"
        "fae12aa1c3dcaf54471dbaff151ffa72a0e81348ff1ba038b883415165e0336e"
    )
    assert second.semantic_key == (
        "whetstone.tool_execution:"
        "419b68076b82038e2f0e8c1ac3c404c60e05ff540ba59b15329a42b43614732b"
    )
    assert first.semantic_key != second.semantic_key


def test_expired_lease_takeover_fences_stale_completion(tmp_path) -> None:
    database = tmp_path / "tool.sqlite"
    policy = _policy()
    config = _config(policy)
    call = _call(config, "takeover")
    clock = MutableClock(datetime(2026, 1, 1, tzinfo=UTC))
    authority = EffectAuthority.memory(clock=clock)
    store = _store(database, effect_authority=authority)
    store.admit(call, config)
    request = tool_effect_request(call)
    first = authority.acquire(
        request,
        owner_id="owner-1",
        attempt_id="attempt-1",
        lease_duration=timedelta(seconds=10),
    )
    assert first.lease is not None and first.lease.fence == 1

    clock.current += timedelta(seconds=11)
    evaluator = CountingEvaluator()
    result = _executor(
        evaluator,
        policy,
        authority,
        owner_id="owner-2",
    ).runtime_handle(config, store, _binding())(call)
    assert result.output is not None
    assert evaluator.evaluations == 1

    with pytest.raises((StaleLeaseError, TerminalConflictError)):
        authority.succeed(
            first.lease,
            result_ref=tool_result_reference(result).record_ref,
        )


def test_lease_loss_prevents_executor_completion_and_surfaces(
    tmp_path,
) -> None:
    database = tmp_path / "tool.sqlite"
    policy = _policy()
    config = _config(policy)
    call = _call(config, "lease-loss")
    clock = MutableClock(datetime(2026, 1, 1, tzinfo=UTC))

    class LeaseExpiringEvaluator(CountingEvaluator):
        def evaluate(
            self, call: ToolCall, config: ToolConfig
        ) -> ToolEvaluation:
            evaluation = super().evaluate(call, config)
            clock.current += timedelta(seconds=11)
            return evaluation

    evaluator = LeaseExpiringEvaluator()
    authority = EffectAuthority.memory(clock=clock)
    store = _store(database, effect_authority=authority)
    handle = _executor(
        evaluator,
        policy,
        authority,
        owner_id="owner",
        lease_duration=timedelta(seconds=10),
    ).runtime_handle(config, store, _binding())

    with pytest.raises(StaleLeaseError):
        handle(call)
    assert evaluator.evaluations == 1
    entry = _store(database).get(call)
    assert entry is not None
    assert entry.state is ToolCallState.ACCEPTED
    assert entry.tool_result_ref is None


def test_busy_execution_is_explicit_and_fabricates_no_result(tmp_path) -> None:
    database = tmp_path / "tool.sqlite"
    policy = _policy()
    config = _config(policy)
    call = _call(config, "busy")
    clock = MutableClock(datetime(2026, 1, 1, tzinfo=UTC))
    authority = EffectAuthority.memory(clock=clock)
    store = _store(database, effect_authority=authority)
    store.admit(call, config)
    authority.acquire(
        tool_effect_request(call),
        owner_id="owner-1",
        attempt_id="attempt-1",
        lease_duration=timedelta(seconds=10),
    )

    evaluator = CountingEvaluator()
    handle = _executor(
        evaluator,
        policy,
        authority,
        owner_id="owner-2",
    ).runtime_handle(config, store, _binding())
    with pytest.raises(ToolExecutionBusyError):
        handle(call)
    assert evaluator.evaluations == 0
    entry = store.get(call)
    assert entry is not None
    assert entry.state is ToolCallState.ACCEPTED
    assert entry.tool_result_ref is None


def test_execution_request_conflict_is_explicit(tmp_path) -> None:
    database = tmp_path / "tool.sqlite"
    policy = _policy()
    config = _config(policy)
    call = _call(config, "conflict")
    authority = EffectAuthority.memory()
    store = _store(database, effect_authority=authority)
    store.admit(call, config)
    request = tool_effect_request(call)
    authority.acquire(
        request.model_copy(update={"request_hash": IdentityHash("b" * 64)}),
        owner_id="other-owner",
        attempt_id="other-attempt",
        lease_duration=timedelta(seconds=10),
    )

    handle = _executor(
        CountingEvaluator(),
        policy,
        authority,
        owner_id="owner",
    ).runtime_handle(config, store, _binding())
    with pytest.raises(ToolExecutionConflictError):
        handle(call)


def test_no_redrive_expiration_requires_recovery(tmp_path) -> None:
    database = tmp_path / "tool.sqlite"
    policy = _policy()
    config = _config(policy, idempotent_replay=False)
    call = _call(config, "recovery")
    clock = MutableClock(datetime(2026, 1, 1, tzinfo=UTC))
    authority = EffectAuthority.memory(clock=clock)
    store = _store(database, effect_authority=authority)
    store.admit(call, config)
    authority.acquire(
        tool_effect_request(call),
        owner_id="crashed-owner",
        attempt_id="crashed-attempt",
        lease_duration=timedelta(seconds=10),
    )
    clock.current += timedelta(seconds=11)
    fabricated = ToolResult(
        call=tool_call_reference(call),
        output={"generation_refs": [], "accepted_ordinal": 1},
        provenance_ordinal=1,
    )
    with pytest.raises(
        ValueError,
        match="exact authoritative EffectTerminal proof",
    ):
        store.complete(fabricated)
    still_accepted = store.get(call)
    assert still_accepted is not None
    assert still_accepted.state is ToolCallState.ACCEPTED

    handle = _executor(
        evaluator := CountingEvaluator(),
        policy,
        authority,
        owner_id="replay-owner",
        replay_policy=ReplayPolicy.NO_REDRIVE,
    ).runtime_handle(config, store, _binding())
    with pytest.raises(ToolExecutionRecoveryRequiredError):
        handle(call)
    assert evaluator.evaluations == 0


def test_completed_admission_projection_cannot_override_recovery_required(
    tmp_path,
) -> None:
    policy = _policy()
    config = _config(policy, idempotent_replay=False)
    call = _call(config, "poisoned-projection")
    clock = MutableClock(datetime(2026, 1, 1, tzinfo=UTC))
    effect_authority = EffectAuthority.memory(clock=clock)
    admission_authority = ToolAdmissionAuthority.memory()
    store = ToolCallStore(
        ObjectStore(SqliteBackend(tmp_path / "objects.sqlite")),
        admission_authority,
        effect_authority,
    )
    accepted = store.admit(call, config)
    fabricated = ToolResult(
        call=tool_call_reference(call),
        output={"generation_refs": [], "accepted_ordinal": 1},
        provenance_ordinal=1,
    )
    fabricated_ref = store.persist_result(fabricated)
    poisoned_terminal = EffectTerminal(
        request=tool_effect_request(call),
        outcome=TerminalOutcome.SUCCEEDED,
        owner_id=NonEmptyId("poisoned-owner"),
        attempt_id=NonEmptyId("poisoned-attempt"),
        fence=1,
        result_ref=fabricated_ref,
    )
    poisoned_entry = ToolCallStoreEntry(
        **accepted.model_dump(
            mode="python",
            exclude={"state", "tool_result_ref", "effect_terminal"},
        ),
        state=ToolCallState.COMPLETED,
        tool_result_ref=fabricated_ref,
        effect_terminal=poisoned_terminal,
    )
    admission_authority.complete(poisoned_entry)

    acquisition = effect_authority.acquire(
        tool_effect_request(call),
        owner_id="crashed-owner",
        attempt_id="crashed-attempt",
        lease_duration=timedelta(seconds=10),
    )
    assert acquisition.lease is not None
    clock.current += timedelta(seconds=11)
    evaluator = CountingEvaluator()
    handle = _executor(
        evaluator,
        policy,
        effect_authority,
        owner_id="replay-owner",
        replay_policy=ReplayPolicy.NO_REDRIVE,
    ).runtime_handle(config, store, _binding())

    with pytest.raises(ToolExecutionRecoveryRequiredError):
        handle(call)
    assert evaluator.evaluations == 0


def test_refusal_evidence_rejects_unrelated_result(tmp_path) -> None:
    database = tmp_path / "tool.sqlite"
    policy = _policy()
    config = _config(policy)
    evaluator = CountingEvaluator()
    authority = EffectAuthority.memory()
    store = _store(database, effect_authority=authority)
    handle = _executor(
        evaluator,
        policy,
        authority,
        owner_id="owner",
    ).runtime_handle(config, store, _binding())
    refused = handle(_call(config, "refused", template=""))
    entry = store.get(_call(config, "refused", template=""))
    assert entry is not None
    ToolEvidence(result=tool_result_reference(refused), store_entry=entry)

    unrelated = ToolResult(
        call=tool_call_reference(_call(config, "unrelated")),
        refusal=refused.refusal,
    )
    with pytest.raises(ValidationError, match="exact Tool Call"):
        ToolEvidence(
            result=tool_result_reference(unrelated),
            store_entry=entry,
        )


def test_same_executor_concurrent_replay_has_one_active_evaluator(
    tmp_path,
) -> None:
    database = tmp_path / "tool.sqlite"
    effect_database = tmp_path / "effects.sqlite"
    policy = _policy()
    config = _config(policy)
    call = _call(config, "concurrent")
    entered = Event()
    release = Event()

    class BlockingEvaluator(CountingEvaluator):
        def evaluate(
            self, call: ToolCall, config: ToolConfig
        ) -> ToolEvaluation:
            entered.set()
            assert release.wait(timeout=10)
            return super().evaluate(call, config)

    evaluator = BlockingEvaluator()
    authority = EffectAuthority.sqlite(effect_database)
    handle = _executor(
        evaluator,
        policy,
        authority,
        owner_id="shared-owner",
    ).runtime_handle(
        config,
        _store(database, effect_authority=authority),
        _binding(),
    )

    with ThreadPoolExecutor(max_workers=2) as pool:
        future = pool.submit(handle, call)
        assert entered.wait(timeout=10)
        try:
            with pytest.raises(ToolExecutionBusyError):
                handle(call)
        finally:
            release.set()
        result = future.result(timeout=10)
    assert result.output is not None
    assert evaluator.evaluations == 1


@pytest.mark.sqlite_time_integration
def test_long_evaluation_renews_lease_and_prevents_takeover(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "tool.sqlite"
    effect_database = tmp_path / "effects.sqlite"
    policy = _policy()
    config = _config(policy)
    call = _call(config, "long-evaluation")
    entered = Event()
    release = Event()

    class BlockingEvaluator(CountingEvaluator):
        def evaluate(
            self, call: ToolCall, config: ToolConfig
        ) -> ToolEvaluation:
            entered.set()
            assert release.wait(timeout=10)
            return super().evaluate(call, config)

    evaluator = BlockingEvaluator()
    lease_duration = timedelta(seconds=1.2)
    renewal_observed_past_original_expiry = Event()
    renewal_lock = Lock()
    original_expiry: datetime | None = None
    successful_renewals = 0
    first_authority = EffectAuthority.sqlite(effect_database)
    real_renew = first_authority.renew

    def recording_renew(
        lease: EffectLease,
        *,
        lease_duration: timedelta,
    ) -> EffectLease:
        nonlocal original_expiry, successful_renewals
        with renewal_lock:
            if original_expiry is None:
                original_expiry = lease.expires_at
        renewed = real_renew(lease, lease_duration=lease_duration)
        renewal_authority_time = renewed.expires_at - lease_duration
        with renewal_lock:
            successful_renewals += 1
            if renewal_authority_time > original_expiry:
                renewal_observed_past_original_expiry.set()
        return renewed

    monkeypatch.setattr(first_authority, "renew", recording_renew)
    first_store = _store(database, effect_authority=first_authority)
    first = _executor(
        evaluator,
        policy,
        first_authority,
        owner_id="owner-1",
        lease_duration=lease_duration,
    ).runtime_handle(config, first_store, _binding())
    contender_authority = EffectAuthority.sqlite(effect_database)
    contender = _executor(
        evaluator,
        policy,
        contender_authority,
        owner_id="owner-2",
        lease_duration=lease_duration,
    ).runtime_handle(
        config,
        _store(database, effect_authority=contender_authority),
        _binding(),
    )

    with ThreadPoolExecutor(max_workers=2) as pool:
        future = pool.submit(first, call)
        assert entered.wait(timeout=10)
        try:
            assert renewal_observed_past_original_expiry.wait(timeout=10)
            with renewal_lock:
                assert successful_renewals >= 3
            with pytest.raises(ToolExecutionBusyError):
                contender(call)
        finally:
            release.set()
        result = future.result(timeout=10)

    assert result.output is not None
    assert evaluator.evaluations == 1
