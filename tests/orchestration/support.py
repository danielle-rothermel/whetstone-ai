"""Shared builders for the orchestration tests.

Wires the real released dependency contracts: real Rollout Execution Keys,
real Provider Call Requests / Transport Policies / Provider Invocation
Evidence, a real in-memory dr-store ObjectStore behind the Whetstone Result
Store, and a real ``ExecutorContext``. A scripted transport peer returns real
evidence from a list of Provider Transport Outcomes so the durable loop's
success / retry / exhaustion paths are exercised without a network.
"""

from __future__ import annotations

from dataclasses import dataclass

from dr_providers import (
    FailureClass,
    MessageRole,
    PromptMessage,
    Protocol,
    ProviderCallRequest,
    ProviderInvocationEvidence,
    ProviderKind,
    ProviderQuotaIdentity,
    ProviderTransportFailure,
    ProviderTransportOutcome,
    ProviderTransportPolicy,
    ProviderTransportResponse,
    RawHttpRequest,
    Transcript,
    openrouter_chat_config,
    policy_for,
)
from dr_store import MemoryBackend, ObjectStore

from whetstone.graph.rollout import (
    EvaluationContext,
    EvaluationRole,
    RolloutExecutionKey,
    RolloutKey,
    rollout_execution_key,
)
from whetstone.orchestration import (
    ExecutorContext,
    ExpectedSchemaIdentities,
    RepeatData,
    RolloutWorkRequest,
    encode_work_request_ref,
    work_request_reference,
)
from whetstone.provider.attempt import PROVIDER_CALL_ATTEMPT_SCHEMA
from whetstone.provider.policy import BackoffSchedule, ProviderExecutionPolicy
from whetstone.result import ROLLOUT_RESULT_SCHEMA, ResultStore

API_KEY_ENV = "OPENROUTER_API_KEY"


def full_hash(char: str) -> str:
    return char * 64


GRAPH_HASH = full_hash("a")
EVAL_CONFIG_HASH = full_hash("b")


def quota(*, model: str = "test-model") -> ProviderQuotaIdentity:
    return ProviderQuotaIdentity(
        provider=ProviderKind.OPENROUTER,
        protocol=Protocol.CHAT_COMPLETIONS,
        model=model,
    )


def evaluation_context(
    *, campaign: str = "camp-1"
) -> EvaluationContext:
    return EvaluationContext(
        eval_config_hash=EVAL_CONFIG_HASH,
        role=EvaluationRole.INTERNAL,
        campaign=campaign,
    )


def execution_key(
    *,
    task_identity: str = "task-1",
    repeat_id: str = "r0",
    context: EvaluationContext | None = None,
) -> RolloutExecutionKey:
    ctx = context or evaluation_context()
    return rollout_execution_key(
        rollout_key=RolloutKey(
            graph_hash=GRAPH_HASH,
            eval_config_hash=ctx.eval_config_hash,
            task_identity=task_identity,
            repeat_id=repeat_id,
        ),
        context=ctx,
    )


def work_request(
    *,
    key: RolloutExecutionKey | None = None,
) -> RolloutWorkRequest:
    exec_key = key or execution_key()
    return RolloutWorkRequest(
        rollout_execution_key=exec_key,
        graph_config_ref="graphcfg://a",
        evaluation_context_ref="evalctx://b",
        task_inputs={"prompt": "task.prompt"},
        repeat_data=RepeatData(
            repeat_id=exec_key.rollout_key.repeat_id,
            repeat_index=0,
        ),
        expected_schema_identities=ExpectedSchemaIdentities(
            rollout_result_schema=ROLLOUT_RESULT_SCHEMA,
            provider_call_attempt_schema=PROVIDER_CALL_ATTEMPT_SCHEMA,
        ),
    )


def response_outcome(*, text: str = "the answer") -> ProviderTransportResponse:
    return ProviderTransportResponse(
        text=text,
        raw_body={"choices": [{"message": {"content": text}}]},
        response_id="resp-1",
        model="test-model",
        finish_reason="stop",
    )


def failure_outcome(
    *,
    failure_class: FailureClass = FailureClass.TRANSIENT,
    message: str = "transport failed",
) -> ProviderTransportFailure:
    return ProviderTransportFailure(
        failure_class=failure_class,
        message=message,
        retryable=failure_class
        in (FailureClass.TRANSIENT, FailureClass.RATE_LIMITED),
        raw_request={"model": "test-model"},
        raw_response_body={"error": message},
    )


def build_request(*, content: str = "hello") -> ProviderCallRequest:
    return ProviderCallRequest(
        config=openrouter_chat_config(model="test-model"),
        transcript=Transcript(
            messages=(PromptMessage(role=MessageRole.USER, content=content),)
        ),
    )


def transport_policy() -> ProviderTransportPolicy:
    return policy_for(
        api_key_env=API_KEY_ENV,
        base_url="https://example.test/v1",
        native_retry_count=0,
    )


def execution_policy(
    *,
    max_attempts: int = 3,
    base_delay: float = 0.01,
) -> ProviderExecutionPolicy:
    return ProviderExecutionPolicy(
        transport_policy=transport_policy(),
        max_attempts=max_attempts,
        backoff=BackoffSchedule(
            base_seconds=base_delay,
            multiplier=2.0,
            max_seconds=1.0,
        ),
    )


def build_evidence(
    *,
    request: ProviderCallRequest,
    policy: ProviderTransportPolicy,
    outcome: ProviderTransportOutcome,
) -> ProviderInvocationEvidence:
    raw_request = RawHttpRequest.build(
        url="https://example.test/v1/chat/completions",
        headers={"Authorization": "Bearer test-key", "content-type": "json"},
        body={"model": "test-model"},
    )
    return ProviderInvocationEvidence.build(
        request=request,
        policy=policy,
        raw_request=raw_request,
        outcome=outcome,
    )


@dataclass
class ScriptedTransport:
    """A transport callable replaying scripted outcomes as real evidence.

    Consumes ``outcomes`` in order (the last repeats). Counts invocations so
    tests can prove replay reuses checkpoints rather than re-calling.
    """

    policy: ProviderTransportPolicy
    outcomes: list[ProviderTransportOutcome]
    calls: int = 0

    def __call__(
        self, request: ProviderCallRequest
    ) -> ProviderInvocationEvidence:
        index = min(self.calls, len(self.outcomes) - 1)
        outcome = self.outcomes[index]
        self.calls += 1
        return build_evidence(
            request=request,
            policy=self.policy,
            outcome=outcome,
        )


@dataclass
class Harness:
    """A ready-to-run in-process executor over an in-memory Result Store."""

    result_store: ResultStore
    transport: ScriptedTransport
    context: ExecutorContext
    request: RolloutWorkRequest

    @property
    def input_ref(self) -> str:
        return encode_work_request_ref(self.request)


def in_memory_result_store() -> ResultStore:
    return ResultStore(ObjectStore(MemoryBackend()))


def build_harness(
    *,
    outcomes: list[ProviderTransportOutcome],
    key: RolloutExecutionKey | None = None,
    max_attempts: int = 3,
    result_store: ResultStore | None = None,
    use_durable_sleep: bool = False,
    request_content: str = "hello",
) -> Harness:
    """Assemble a Harness whose executor runs the pure (non-DBOS) loop.

    ``use_durable_sleep`` defaults False so ``ExecutorContext`` never calls
    ``DBOS.sleep`` off a runtime; the DBOS integration test builds its own
    context with durable sleep enabled.
    """
    store = result_store or in_memory_result_store()
    policy = transport_policy()
    request = work_request(key=key)
    scripted = ScriptedTransport(policy=policy, outcomes=outcomes)

    def resolve(input_ref: str) -> RolloutWorkRequest:
        # A minimal resolver: the reference round-trips to the known request.
        # (A production resolver reads the stored Work Request through
        # dr-store; here the single request is deterministic from its ref.)
        assert input_ref == encode_work_request_ref(request)
        return request

    context = ExecutorContext(
        result_store=store,
        transport=scripted,
        policy=execution_policy(max_attempts=max_attempts),
        resolve_work_request=resolve,
        build_request=lambda _request: build_request(content=request_content),
        use_durable_sleep=use_durable_sleep,
    )
    return Harness(
        result_store=store,
        transport=scripted,
        context=context,
        request=request,
    )


def persist_work_request_store() -> tuple[ObjectStore, str]:
    """Persist one Work Request through dr-store and return (store, input_ref).

    Exercises the real opaque-transport path: the Work Request is put into
    dr-store, its typed reference encoded as the opaque input-ref string.
    """
    store = ObjectStore(MemoryBackend())
    request = work_request()
    reference = work_request_reference(request)
    stored, _status = store.put(
        reference.schema, request.record_content()
    )
    assert stored == reference
    return store, encode_work_request_ref(request)
