"""Proposal evaluation is outside the checkpointed adapter invocation."""

import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pytest

from whetstone.optimization import (
    AdapterOutput,
    BudgetDelta,
    Candidate,
    IntentOutcome,
    ReplayPolicy,
    RuntimeToolHandle,
    StepMode,
    StepStatus,
    candidate_reference,
    step_result_reference,
)
from whetstone.optimization.effect_authority import EffectAuthority
from whetstone.optimization.harness import (
    ADAPTER_CHECKPOINT_SCHEMA,
    INTENT_EFFECT_KEY_PREFIX,
)
from whetstone.optimization.identity import ImmutableJsonObject
from whetstone.optimization.schema import eval_config_reference

from .support import (
    CountingProposalAdapter,
    RecordingEvaluationService,
    base_ref,
    candidate,
    eval_config,
    evaluation_binding,
    internal_reward_policy,
    make_harness,
    make_intent,
    make_store,
    output_contract,
    proposal_request,
    proposed_candidate,
    registry,
)


class CrashOnceEvaluationService:
    def __init__(self) -> None:
        self.calls = 0

    @property
    def replay_policy(self):
        return ReplayPolicy.IDEMPOTENT

    def resolve_evaluation_intent(self, intent):
        del intent
        self.calls += 1
        raise RuntimeError("crash during external evaluation")


@dataclass
class MutableClock:
    current: datetime

    def __call__(self) -> datetime:
        return self.current


class PoisonThenValidAdapter:
    def __init__(self, poison: str) -> None:
        self.poison = poison
        self.invocations = 0

    @property
    def key(self) -> str:
        return "proposal-test"

    @property
    def mode(self) -> StepMode:
        return StepMode.PROPOSAL_ONLY

    def invoke(
        self,
        request,
        handles: tuple[RuntimeToolHandle, ...],
    ) -> AdapterOutput:
        assert handles == ()
        self.invocations += 1
        proposed = proposed_candidate(
            request.candidates[0],
            "poison-retry",
            text="valid-retry",
        )
        intent = make_intent(
            proposed,
            run_id=request.run_id,
            step_index=request.step_index,
            reward_policy=request.run.record.reward_policy,
        )
        if self.invocations == 1:
            proposed, intent = self._poison(request, proposed, intent)
        return AdapterOutput(
            proposed_candidates=(proposed,),
            accepted_candidates=(proposed,),
            evaluation_intents=(intent,),
            budget_delta=BudgetDelta(consumed={"rollouts": 1}),
            proposed_status=StepStatus.COMPLETE,
        )

    def _poison(self, request, proposed, intent):
        if self.poison == "template":
            proposed = proposed_candidate(
                request.candidates[0],
                "poison-retry",
                text="{unavailable}",
            )
            intent = make_intent(
                proposed,
                run_id=request.run_id,
                step_index=request.step_index,
                reward_policy=request.run.record.reward_policy,
            )
        elif self.poison == "base":
            proposed = Candidate(
                candidate_id=proposed.candidate_id,
                base_ref=base_ref("foreign"),
                payload=proposed.payload,
            )
            intent = intent.model_copy(
                update={"candidate": candidate_reference(proposed)}
            )
        elif self.poison == "diff":
            payload = proposed.payload.to_json()
            payload["fixed"] = "poisoned"
            proposed = Candidate(
                candidate_id=proposed.candidate_id,
                base_ref=proposed.base_ref,
                payload=ImmutableJsonObject(payload),
            )
            intent = intent.model_copy(
                update={"candidate": candidate_reference(proposed)}
            )
        elif self.poison == "run":
            intent = intent.model_copy(update={"run_id": "foreign-run"})
        elif self.poison == "step":
            intent = intent.model_copy(
                update={"step_index": request.step_index + 1}
            )
        elif self.poison == "candidate":
            outsider = proposed_candidate(
                request.candidates[0],
                "outsider",
                text="outsider",
            )
            intent = intent.model_copy(
                update={"candidate": candidate_reference(outsider)}
            )
        elif self.poison == "binding":
            other_config = eval_config_reference(eval_config("e" * 64))
            intent = intent.model_copy(
                update={"evaluation_binding": evaluation_binding(other_config)}
            )
        elif self.poison == "policy":
            other_policy = internal_reward_policy().model_copy(
                update={"policy_name": "other-policy/v1"}
            )
            intent = make_intent(
                proposed,
                run_id=request.run_id,
                step_index=request.step_index,
                reward_policy=other_policy,
            )
        else:  # pragma: no cover - closed test parameter
            raise AssertionError(f"unknown poison {self.poison!r}")
        return proposed, intent


@pytest.mark.parametrize(
    ("poison", "error"),
    [
        ("template", "exact run template contract"),
        ("base", "bind an exact request candidate"),
        ("diff", "canonical run mutation diff"),
        ("run", "another optimization run"),
        ("step", "another optimization step"),
        ("candidate", "not an exact Step output candidate"),
        ("binding", "target Eval Config must match"),
        ("policy", "exact run Reward Policy"),
    ],
)
def test_invalid_adapter_output_is_not_checkpointed_and_can_retry(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    poison: str,
    error: str,
) -> None:
    clock = MutableClock(datetime(2026, 1, 1, tzinfo=UTC))
    authority = EffectAuthority.memory(clock=clock)
    acquired_keys: list[str] = []
    real_acquire = authority.acquire

    def record_acquire(effect_request, **kwargs):
        acquired_keys.append(str(effect_request.semantic_key))
        return real_acquire(effect_request, **kwargs)

    monkeypatch.setattr(authority, "acquire", record_acquire)
    store = make_store(tmp_path)
    service = RecordingEvaluationService(store)
    adapter = PoisonThenValidAdapter(poison)
    request = proposal_request()
    harness = make_harness(
        store=store,
        adapter_registry=registry(adapter),
        run=request.run,
        effect_authority=authority,
        evaluation_service=service,
        lease_duration=timedelta(seconds=1),
    )
    persisted_schemas: list[str] = []
    real_put = harness._put

    def record_put(schema, content):
        persisted_schemas.append(schema)
        return real_put(schema, content)

    monkeypatch.setattr(harness, "_put", record_put)

    with pytest.raises(ValueError, match=error):
        harness.run_step(request)

    assert ADAPTER_CHECKPOINT_SCHEMA not in persisted_schemas
    assert service.calls == []
    assert not any(
        key.startswith(INTENT_EFFECT_KEY_PREFIX) for key in acquired_keys
    )
    assert harness.resolve_step_result(request.run_id, 0) is None

    clock.current += timedelta(seconds=2)
    result, _ = harness.run_step(request)

    assert adapter.invocations == 2
    assert len(service.calls) == 1
    assert len(result.resolved_intents) == 1
    assert ADAPTER_CHECKPOINT_SCHEMA in persisted_schemas


def test_fresh_sqlite_restart_reuses_adapter_checkpoint(tmp_path) -> None:
    adapter = CountingProposalAdapter()
    request = proposal_request()
    crashed_store = make_store(tmp_path)
    crashed = make_harness(
        store=crashed_store,
        adapter_registry=registry(adapter),
        run=request.run,
        effect_authority=EffectAuthority.sqlite(tmp_path / "effects.sqlite"),
        evaluation_service=CrashOnceEvaluationService(),
        lease_duration=timedelta(milliseconds=20),
    )
    with pytest.raises(RuntimeError, match="crash during"):
        crashed.run_step(request)
    assert adapter.invocations == 1
    assert crashed.resolve_step_result(request.run_id, 0) is None

    fresh_store = make_store(tmp_path)
    time.sleep(0.05)
    fresh = make_harness(
        store=fresh_store,
        adapter_registry=registry(adapter),
        run=request.run,
        effect_authority=EffectAuthority.sqlite(tmp_path / "effects.sqlite"),
        evaluation_service=RecordingEvaluationService(fresh_store),
        lease_duration=timedelta(milliseconds=20),
    )
    result, result_ref = fresh.run_step(request)
    assert adapter.invocations == 1
    assert result.resolved_intents[0].outcome is IntentOutcome.COMPLETED
    assert fresh.resolve_step_result(request.run_id, 0) == result_ref


def test_restart_reuses_terminal_intent_prefix_after_later_crash(
    tmp_path,
) -> None:
    clock = MutableClock(datetime(2026, 1, 1, tzinfo=UTC))
    authority = EffectAuthority.memory(clock=clock)
    store = make_store(tmp_path)
    adapter = CountingProposalAdapter(
        candidates=(candidate("first"), candidate("second")),
        budget_delta=BudgetDelta(consumed={"rollouts": 2}),
    )
    request = proposal_request(contract=output_contract(2))
    service = RecordingEvaluationService(store, crash_on_call=2)
    harness = make_harness(
        store=store,
        adapter_registry=registry(adapter),
        run=request.run,
        effect_authority=authority,
        evaluation_service=service,
        lease_duration=timedelta(seconds=1),
    )

    with pytest.raises(RuntimeError, match="crash during evaluation"):
        harness.run_step(request)
    first_intent, crashed_intent = service.calls
    clock.current += timedelta(seconds=2)

    result, _ = harness.run_step(request)

    assert adapter.invocations == 1
    assert service.calls == [first_intent, crashed_intent, crashed_intent]
    assert tuple(
        resolution.intent for resolution in result.resolved_intents
    ) == (first_intent, crashed_intent)


def test_candidate_local_failure_does_not_erase_successful_steps(
    tmp_path,
) -> None:
    store = make_store(tmp_path)
    first_adapter = CountingProposalAdapter(status=StepStatus.CONTINUE)
    first_request = proposal_request()
    harness = make_harness(
        store=store,
        adapter_registry=registry(first_adapter),
        run=first_request.run,
        evaluation_service=RecordingEvaluationService(store),
    )
    first, first_ref = harness.run_step(first_request)

    failed_service = RecordingEvaluationService(
        store, outcome=IntentOutcome.FAILED
    )
    second_adapter = CountingProposalAdapter(status=StepStatus.COMPLETE)
    second = make_harness(
        store=store,
        adapter_registry=registry(second_adapter),
        run=first_request.run,
        evaluation_service=failed_service,
    )
    second_request = proposal_request(
        step_index=1,
        prior_step_result_ref=first_ref,
        budget=first.budget,
    )
    second_result, second_ref = second.run_step(second_request)
    assert second_result.status is StepStatus.COMPLETE
    assert second_result.resolved_intents[0].outcome is IntentOutcome.FAILED

    terminal, _ = second.terminalize(
        run=first_request.run,
        step_results=(
            step_result_reference(first),
            step_result_reference(second_result),
        ),
    )
    assert terminal.step_result_refs == (first_ref, second_ref)
    assert len(terminal.proposals) == 1


def test_pre_execution_rejection_is_recorded_without_evidence(
    tmp_path,
) -> None:
    store = make_store(tmp_path)
    adapter = CountingProposalAdapter()
    request = proposal_request()
    harness = make_harness(
        store=store,
        adapter_registry=registry(adapter),
        run=request.run,
        evaluation_service=RecordingEvaluationService(
            store, outcome=IntentOutcome.REJECTED
        ),
    )
    result, _ = harness.run_step(request)
    resolution = result.resolved_intents[0]
    assert resolution.outcome is IntentOutcome.REJECTED
    assert resolution.evaluation_evidence_refs == ()
