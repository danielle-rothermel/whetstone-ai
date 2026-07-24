from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
from dr_store import ObjectStore, SqliteBackend

from tests.envs.support import (
    FakeTransport,
    constant_reply,
    execution_policy,
)
from whetstone.envs.factory import build_env_experiment
from whetstone.evaluation import (
    EngineEvaluationService,
    EngineToolEvaluator,
    EvaluationEngine,
    EvaluationRequest,
)
from whetstone.execution.partials import PartialLog
from whetstone.execution.prompt_cache import PromptResultCache
from whetstone.graph.rollout import EvaluationRole
from whetstone.optimization import (
    Candidate,
    EvaluationIntent,
    IntentOutcome,
    Reward,
    ToolCall,
    ToolCapacity,
    ToolConfig,
    ToolDefinition,
    ToolValidationError,
    TypedRef,
    candidate_reference,
)


def _experiment(*, repeats: int = 1):
    return build_env_experiment(
        "c18",
        model="openai/test",
        pool_n_per_stratum=2,
        split_sizes=(1, 1, 1),
        repeats=repeats,
    )


def _engine(
    tmp_path,
    *,
    store: ObjectStore,
    transport: FakeTransport,
    repeats: int = 1,
    partial: bool = False,
    cache: bool = False,
) -> EvaluationEngine:
    experiment = _experiment(repeats=repeats)
    return EvaluationEngine(
        store=store,
        experiment=experiment,
        sampling=experiment.eval_configs.internal,
        execution_policy=execution_policy(),
        transport=transport,
        partial_log=PartialLog(tmp_path / "partials.jsonl")
        if partial
        else None,
        prompt_cache=PromptResultCache(tmp_path / "cache") if cache else None,
    )


def test_engine_persists_exact_evidence_and_reward(tmp_path) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "engine.sqlite"))
    transport = FakeTransport(reply=constant_reply("wrong"))
    engine = _engine(tmp_path, store=store, transport=transport)

    result = engine.evaluate(
        EvaluationRequest(
            candidate=engine.experiment.initial_candidate,
            evaluation_role=EvaluationRole.INTERNAL,
            evaluation_context_id="ctx",
            purpose="test",
        )
    )

    evidence = result.evidence
    assert store.get(result.evidence_ref.reference) == (
        evidence.record_content()
    )
    assert store.get(evidence.candidate.record_ref.reference)
    assert store.get(evidence.eval_config.record_ref.reference)
    assert store.get(evidence.outputs_ref.reference)
    assert store.get(evidence.aggregate_ref.reference)
    assert evidence.reward_ref is not None
    reward = Reward.model_validate(store.get(evidence.reward_ref.reference))
    assert reward.evidence_ref_content_hash == (
        evidence.aggregate_ref.content_hash
    )
    assert evidence.row_accounting.planned == 1
    assert evidence.row_accounting.present == 1
    assert evidence.per_task_counts == (1,)
    assert evidence.eval_config == engine.eval_config_ref


def test_invalid_intent_rejects_without_provider_spend(tmp_path) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "reject.sqlite"))
    transport = FakeTransport(reply=constant_reply("unused"))
    engine = _engine(tmp_path, store=store, transport=transport)
    invalid = Candidate(
        candidate_id="invalid",
        base_ref=engine.experiment.initial_candidate.base_ref,
        payload={"user_prompt_template": "Use {private_gold}."},
    )
    intent = EvaluationIntent(
        intent_id="invalid-intent",
        candidate=candidate_reference(invalid),
        target_eval_config=engine.eval_config_ref,
        context_role=EvaluationRole.INTERNAL,
        purpose="preflight",
        run_id="run",
        step_index=0,
    )

    resolution = EngineEvaluationService(
        store=store, engine=engine
    ).resolve_evaluation_intent(intent)

    assert resolution.outcome is IntentOutcome.REJECTED
    assert resolution.evaluation_evidence_refs == ()
    assert transport.served == []


def test_resolution_and_prompt_results_replay_after_restart(tmp_path) -> None:
    database = tmp_path / "restart.sqlite"
    store = ObjectStore(SqliteBackend(database))
    transport = FakeTransport(reply=constant_reply("wrong"))
    engine = _engine(
        tmp_path,
        store=store,
        transport=transport,
        partial=True,
        cache=True,
    )
    candidate = engine.experiment.initial_candidate
    intent = EvaluationIntent(
        intent_id="restart-intent",
        candidate=candidate_reference(candidate),
        target_eval_config=engine.eval_config_ref,
        context_role=EvaluationRole.INTERNAL,
        purpose="restart",
        run_id="run",
        step_index=0,
    )
    first = EngineEvaluationService(
        store=store, engine=engine
    ).resolve_evaluation_intent(intent)
    assert len(transport.served) == 1

    fresh_store = ObjectStore(SqliteBackend(database))
    never = FakeTransport(
        reply=lambda _prompt: (_ for _ in ()).throw(
            AssertionError("durable resolution must replay")
        )
    )
    fresh_engine = _engine(
        tmp_path,
        store=fresh_store,
        transport=never,
        partial=True,
        cache=True,
    )
    replay = EngineEvaluationService(
        store=fresh_store, engine=fresh_engine
    ).resolve_evaluation_intent(intent)

    assert replay == first
    assert never.served == []


def test_two_resolvers_share_one_durable_evaluation(tmp_path) -> None:
    database = tmp_path / "concurrent.sqlite"
    transport_entered = Event()
    waiter_entered = Event()
    release = Event()

    def blocked_reply(_prompt: str) -> str:
        transport_entered.set()
        assert release.wait(timeout=2)
        return "wrong"

    transport = FakeTransport(reply=blocked_reply)
    first_store = ObjectStore(SqliteBackend(database))
    second_store = ObjectStore(SqliteBackend(database))
    first_engine = _engine(tmp_path, store=first_store, transport=transport)
    second_engine = _engine(tmp_path, store=second_store, transport=transport)
    intent = EvaluationIntent(
        intent_id="concurrent-intent",
        candidate=candidate_reference(
            first_engine.experiment.initial_candidate
        ),
        target_eval_config=first_engine.eval_config_ref,
        context_role=EvaluationRole.INTERNAL,
        purpose="concurrent",
        run_id="run",
        step_index=0,
    )

    def wait_for_winner(_seconds: float) -> None:
        waiter_entered.set()
        assert release.wait(timeout=2)

    first_service = EngineEvaluationService(
        store=first_store, engine=first_engine
    )
    second_service = EngineEvaluationService(
        store=second_store,
        engine=second_engine,
        sleep=wait_for_winner,
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(first_service.resolve_evaluation_intent, intent)
        assert transport_entered.wait(timeout=2)
        second = pool.submit(second_service.resolve_evaluation_intent, intent)
        assert waiter_entered.wait(timeout=2)
        assert len(transport.served) == 1
        release.set()
        assert second.result(timeout=2) == first.result(timeout=2)

    assert len(transport.served) == 1


def test_live_slow_evaluation_renews_its_durable_claim(tmp_path) -> None:
    database = tmp_path / "heartbeat.sqlite"
    transport_entered = Event()
    waiter_entered = Event()
    release = Event()

    def blocked_reply(_prompt: str) -> str:
        transport_entered.set()
        assert release.wait(timeout=10)
        return "wrong"

    def wait_for_winner(_seconds: float) -> None:
        waiter_entered.set()
        assert release.wait(timeout=10)

    transport = FakeTransport(reply=blocked_reply)
    first_store = ObjectStore(SqliteBackend(database))
    second_store = ObjectStore(SqliteBackend(database))
    first_engine = _engine(tmp_path, store=first_store, transport=transport)
    second_engine = _engine(tmp_path, store=second_store, transport=transport)
    intent = EvaluationIntent(
        intent_id="slow-live-intent",
        candidate=candidate_reference(
            first_engine.experiment.initial_candidate
        ),
        target_eval_config=first_engine.eval_config_ref,
        context_role=EvaluationRole.INTERNAL,
        purpose="heartbeat",
        run_id="run",
        step_index=0,
    )
    first_service = EngineEvaluationService(
        store=first_store,
        engine=first_engine,
        claim_lease_seconds=3.0,
    )
    second_service = EngineEvaluationService(
        store=second_store,
        engine=second_engine,
        claim_lease_seconds=3.0,
        sleep=wait_for_winner,
    )

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(first_service.resolve_evaluation_intent, intent)
        assert transport_entered.wait(timeout=10)
        initial = first_service._latest_claim(intent)
        assert initial is not None
        deadline = time.monotonic() + 10
        while True:
            renewed = first_service._latest_claim(intent)
            assert renewed is not None
            if (
                renewed.event_ordinal > initial.event_ordinal
                and renewed.expires_at > initial.expires_at
            ):
                break
            if time.monotonic() >= deadline:
                pytest.fail("durable heartbeat did not renew the live claim")
            time.sleep(0.01)
        second = pool.submit(second_service.resolve_evaluation_intent, intent)
        assert waiter_entered.wait(timeout=10)
        assert len(transport.served) == 1
        release.set()
        assert second.result(timeout=10) == first.result(timeout=10)

    assert len(transport.served) == 1


def test_renewal_wins_same_event_slot_as_stale_takeover(
    tmp_path, monkeypatch
) -> None:
    database = tmp_path / "claim-renewal-race.sqlite"
    now = [100.0]
    renewal_paused = Event()
    stale_takeover_ready = Event()
    renewal_bound = Event()
    transport_entered = Event()
    waiter_entered = Event()
    release = Event()

    def blocked_reply(_prompt: str) -> str:
        transport_entered.set()
        assert release.wait(timeout=2)
        return "wrong"

    def wait_for_winner(_seconds: float) -> None:
        waiter_entered.set()
        assert release.wait(timeout=2)

    transport = FakeTransport(reply=blocked_reply)
    first_store = ObjectStore(SqliteBackend(database))
    second_store = ObjectStore(SqliteBackend(database))
    first_engine = _engine(tmp_path, store=first_store, transport=transport)
    second_engine = _engine(tmp_path, store=second_store, transport=transport)
    intent = EvaluationIntent(
        intent_id="renewal-race-intent",
        candidate=candidate_reference(
            first_engine.experiment.initial_candidate
        ),
        target_eval_config=first_engine.eval_config_ref,
        context_role=EvaluationRole.INTERNAL,
        purpose="renewal-race",
        run_id="run",
        step_index=0,
    )
    first = EngineEvaluationService(
        store=first_store,
        engine=first_engine,
        claim_lease_seconds=1.0,
        clock=lambda: now[0],
    )
    second = EngineEvaluationService(
        store=second_store,
        engine=second_engine,
        claim_lease_seconds=1.0,
        clock=lambda: now[0],
        sleep=wait_for_winner,
    )
    renew_claim = first._renew_claim
    append_claim_event = second._append_claim_event

    def paused_renewal(intent, owned) -> None:
        renewal_paused.set()
        assert stale_takeover_ready.wait(timeout=2)
        renew_claim(intent, owned)
        renewal_bound.set()

    def delayed_takeover(**kwargs):
        prior = kwargs["prior"]
        if prior is not None and kwargs["generation"] == 1:
            stale_takeover_ready.set()
            assert renewal_bound.wait(timeout=2)
        return append_claim_event(**kwargs)

    monkeypatch.setattr(first, "_renew_claim", paused_renewal)
    monkeypatch.setattr(second, "_append_claim_event", delayed_takeover)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first_result = pool.submit(first.resolve_evaluation_intent, intent)
        assert renewal_paused.wait(timeout=2)
        now[0] = 102.0
        second_result = pool.submit(second.resolve_evaluation_intent, intent)
        assert stale_takeover_ready.wait(timeout=2)
        assert renewal_bound.wait(timeout=2)
        assert transport_entered.wait(timeout=2)
        assert waiter_entered.wait(timeout=2)
        assert len(transport.served) == 1
        release.set()
        assert second_result.result(timeout=2) == first_result.result(
            timeout=2
        )

    assert len(transport.served) == 1


def test_expired_claim_retries_after_resolver_crash(tmp_path) -> None:
    database = tmp_path / "claim-retry.sqlite"
    now = [100.0]
    transport = FakeTransport(reply=constant_reply("wrong"))

    def crash_once(_prompt: str) -> str:
        if len(transport.served) == 1:
            raise KeyboardInterrupt("simulated resolver crash")
        return "wrong"

    transport.reply = crash_once
    first_store = ObjectStore(SqliteBackend(database))
    first_engine = _engine(tmp_path, store=first_store, transport=transport)
    intent = EvaluationIntent(
        intent_id="crashed-intent",
        candidate=candidate_reference(
            first_engine.experiment.initial_candidate
        ),
        target_eval_config=first_engine.eval_config_ref,
        context_role=EvaluationRole.INTERNAL,
        purpose="crash-retry",
        run_id="run",
        step_index=0,
    )
    crashed = EngineEvaluationService(
        store=first_store,
        engine=first_engine,
        claim_lease_seconds=1.0,
        clock=lambda: now[0],
        sleep=lambda _seconds: None,
    )

    with pytest.raises(KeyboardInterrupt, match="simulated resolver crash"):
        crashed.resolve_evaluation_intent(intent)

    now[0] = 102.0
    retry_store = ObjectStore(SqliteBackend(database))
    retry_engine = _engine(tmp_path, store=retry_store, transport=transport)
    completed = EngineEvaluationService(
        store=retry_store,
        engine=retry_engine,
        claim_lease_seconds=1.0,
        clock=lambda: now[0],
        sleep=lambda _seconds: None,
    ).resolve_evaluation_intent(intent)

    assert completed.outcome is IntentOutcome.COMPLETED
    assert len(transport.served) == 2


def test_expired_owner_cannot_renew_after_new_generation_claims(
    tmp_path,
) -> None:
    database = tmp_path / "claim-fence.sqlite"
    now = [100.0]
    first_store = ObjectStore(SqliteBackend(database))
    second_store = ObjectStore(SqliteBackend(database))
    transport = FakeTransport(reply=constant_reply("wrong"))
    first_engine = _engine(tmp_path, store=first_store, transport=transport)
    second_engine = _engine(tmp_path, store=second_store, transport=transport)
    intent = EvaluationIntent(
        intent_id="fenced-intent",
        candidate=candidate_reference(
            first_engine.experiment.initial_candidate
        ),
        target_eval_config=first_engine.eval_config_ref,
        context_role=EvaluationRole.INTERNAL,
        purpose="fence",
        run_id="run",
        step_index=0,
    )
    first = EngineEvaluationService(
        store=first_store,
        engine=first_engine,
        claim_lease_seconds=1.0,
        clock=lambda: now[0],
        sleep=lambda _seconds: None,
    )
    second = EngineEvaluationService(
        store=second_store,
        engine=second_engine,
        claim_lease_seconds=1.0,
        clock=lambda: now[0],
        sleep=lambda _seconds: None,
    )
    first_claim = first._claim(intent)
    assert first_claim is not None
    now[0] = 102.0
    second_claim = second._claim(intent)
    assert second_claim is not None
    assert second_claim.generation == 1

    with pytest.raises(RuntimeError, match="not owned"):
        first._renew_claim(intent, first_claim)
    assert transport.served == []


def test_cache_provenance_avoids_transport_replay(tmp_path) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "cache.sqlite"))
    transport = FakeTransport(reply=constant_reply("wrong"))
    engine = _engine(
        tmp_path,
        store=store,
        transport=transport,
        partial=True,
        cache=True,
    )
    base = engine.experiment.initial_candidate
    engine.evaluate(
        EvaluationRequest(
            candidate=base,
            evaluation_role=EvaluationRole.INTERNAL,
            evaluation_context_id="first",
            purpose="cache",
        )
    )
    same_prompt = base.model_copy(update={"candidate_id": "same-prompt"})
    result = engine.evaluate(
        EvaluationRequest(
            candidate=same_prompt,
            evaluation_role=EvaluationRole.INTERNAL,
            evaluation_context_id="second",
            purpose="cache",
        )
    )

    assert len(transport.served) == 1
    assert result.evidence.cache.cache_hit_count == 1
    assert result.evidence.cache.source_call_ids


def test_sampling_repeat_change_changes_exact_eval_identity(tmp_path) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "identity.sqlite"))
    transport = FakeTransport(reply=constant_reply("wrong"))
    one = _engine(tmp_path, store=store, transport=transport, repeats=1)
    two = _engine(tmp_path, store=store, transport=transport, repeats=2)

    assert (
        one.eval_config_ref.identity_hash != two.eval_config_ref.identity_hash
    )


def test_tool_projection_uses_same_engine_evidence(tmp_path) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "tool.sqlite"))
    transport = FakeTransport(reply=constant_reply("wrong"))
    engine = _engine(tmp_path, store=store, transport=transport)
    definition = ToolDefinition(
        tool_name="evaluate_candidate",
        input_fields=("base_ref", "model_route", "template"),
        output_fields=("evaluation_evidence_ref", "output_artifact_ref"),
    )
    config = ToolConfig(
        tool_name=definition.tool_name,
        tool_definition_ref="tooldef://evaluate_candidate",
        tool_definition_identity_hash=definition.identity_hash(),
        endpoint="mcp://whetstone/evaluate_candidate",
        eval_config_ref=engine.eval_config_ref.record_ref.content_hash,
        eval_config_identity_hash=engine.eval_config_ref.identity_hash,
        reward_policy_ref=engine.experiment.reward_policy.identity_hash(),
        capacity=ToolCapacity(max_accepted_calls=1),
        store_namespace="tool-projection",
    )
    base = engine.experiment.initial_candidate
    call = ToolCall(
        call_id="tool-call",
        tool_config_hash=config.identity_hash(),
        store_namespace=config.store_namespace,
        args={
            "base_ref": base.base_ref,
            "model_route": base.base_ref,
            "template": base.payload["user_prompt_template"],
        },
    )

    projected = EngineToolEvaluator(engine).evaluate(call, config)

    assert projected.eval_config_hash == engine.eval_config_ref.identity_hash
    assert len(projected.rollout_refs) == 1
    assert projected.extra_output["evaluation_evidence_ref"] == (
        projected.rollout_refs[0].model_dump(mode="json")
    )
    artifact = TypedRef.model_validate(
        projected.extra_output["output_artifact_ref"]
    )
    assert store.get(artifact.reference)

    mismatched = call.model_copy(
        update={
            "call_id": "wrong-task",
            "args": {**call.args, "task_ids": ["not-the-bound-task"]},
        }
    )
    served = len(transport.served)
    with pytest.raises(ToolValidationError, match="unknown task IDs"):
        EngineToolEvaluator(engine).evaluate(mismatched, config)
    duplicate = call.model_copy(
        update={
            "call_id": "duplicate-task",
            "args": {
                **call.args,
                "task_ids": [
                    engine.sampling.task_set.task_identities[0],
                    engine.sampling.task_set.task_identities[0],
                ],
            },
        }
    )
    with pytest.raises(ToolValidationError, match="must be unique"):
        EngineToolEvaluator(engine).evaluate(duplicate, config)
    assert len(transport.served) == served
