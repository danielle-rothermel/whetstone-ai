"""Shared fixtures for the whetstone env-adapter tests.

Builds a scripted fake transport (no network, no DBOS, no live paid call)
that returns a per-request generation text so the internal-eval loop can be
driven deterministically, plus small helpers to construct env pools and
execution policies.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from dr_code.humaneval import HumanEvalTask
from dr_providers import (
    ProviderCallRequest,
    ProviderInvocationEvidence,
    ProviderKind,
    ProviderTransportPolicy,
    ProviderTransportResponse,
    RawHttpRequest,
    policy_for,
)
from pydantic import BaseModel, JsonValue
from whetstone_envs.core import Instance

from whetstone.envs.ed1 import Ed1Instance, ed1_instance_from_task
from whetstone.envs.sampling import EnvSplitSampling
from whetstone.evaluation_role import EvaluationRole
from whetstone.execution.fanout import ProcessJob
from whetstone.optimization.schema import (
    EVALUATION_BINDING_SCHEMA_VERSION,
    EvaluationBinding,
    eval_config_reference,
)
from whetstone.provider.policy import ProviderExecutionPolicy

API_KEY_ENV = "OPENROUTER_API_KEY"

#: A text-returning callable keyed on the request's user-message content
#: (the rendered prompt), so a fake can answer differently per task.
ReplyFn = Callable[[str], str]
RowPayloadFn = Callable[[Instance, int, int], BaseModel | JsonValue]


def row_job_factory(
    payload_for: RowPayloadFn,
) -> Callable[[BaseModel], ProcessJob]:
    """Build request-bound importable jobs for reducer-focused tests."""

    def build(request: BaseModel) -> ProcessJob:
        from whetstone.envs.d1_eval import (
            D1RowOutcome,
            D1RowRequest,
            D1RowResult,
        )
        from whetstone.envs.ed1_eval import (
            Ed1RowOutcome,
            Ed1RowRequest,
            Ed1RowResult,
        )
        from whetstone.envs.internal_eval import (
            InternalRowOutcome,
            InternalRowRequest,
            InternalRowResult,
        )

        if not isinstance(
            request, InternalRowRequest | D1RowRequest | Ed1RowRequest
        ):
            raise TypeError(f"unsupported row request {type(request)!r}")
        instance = request.instance.to_instance()
        repeat = request.repeat_index
        drive_ordinal = request.drive_ordinal
        outcome = payload_for(instance, repeat, drive_ordinal)
        if isinstance(request, InternalRowRequest):
            if not isinstance(outcome, InternalRowOutcome):
                raise TypeError("internal request requires InternalRowOutcome")
            envelope = InternalRowResult(
                request_identity=request.request_identity,
                outcome=outcome,
            )
        elif isinstance(request, D1RowRequest):
            if not isinstance(outcome, D1RowOutcome):
                raise TypeError("D1 request requires D1RowOutcome")
            envelope = D1RowResult(
                request_identity=request.request_identity,
                outcome=outcome,
            )
        else:
            if not isinstance(outcome, Ed1RowOutcome):
                raise TypeError("ED1 request requires Ed1RowOutcome")
            envelope = Ed1RowResult(
                request_identity=request.request_identity,
                outcome=outcome,
            )
        return ProcessJob(
            entrypoint="tests.envs.process_workers:return_payload",
            payload=envelope.model_dump(mode="json"),
        )

    return build


def process_row_job_factory(
    entrypoint: str,
) -> Callable[[BaseModel], ProcessJob]:
    """Send the complete typed row request to a real driver worker."""

    def build(request: BaseModel) -> ProcessJob:
        return ProcessJob(
            entrypoint=entrypoint,
            payload=request.model_dump(mode="json"),
        )

    return build


def evaluation_binding(
    sampling: EnvSplitSampling, *, official: bool = False
) -> EvaluationBinding:
    role = EvaluationRole.OFFICIAL if official else EvaluationRole.INTERNAL
    return EvaluationBinding(
        schema_version=EVALUATION_BINDING_SCHEMA_VERSION,
        eval_config=eval_config_reference(sampling.eval_config),
        role=role,
        authority_principal="test-authority" if official else None,
        campaign="env-test",
    )


def transport_policy() -> ProviderTransportPolicy:
    return policy_for(
        ProviderKind.OPENROUTER,
        api_key_env=API_KEY_ENV,
        base_url="https://example.test/v1",
        native_retry_count=0,
    )


def execution_policy(*, max_attempts: int = 1) -> ProviderExecutionPolicy:
    return ProviderExecutionPolicy(
        transport_policy=transport_policy(),
        max_attempts=max_attempts,
    )


def _response(text: str) -> ProviderTransportResponse:
    return ProviderTransportResponse(
        text=text,
        raw_body={"choices": [{"message": {"content": text}}]},
        response_id="resp-1",
        model="test-model",
        finish_reason="stop",
    )


def _prompt_of(request: ProviderCallRequest) -> str:
    messages = request.transcript.messages
    return messages[-1].content if messages else ""


@dataclass
class FakeTransport:
    """A scripted transport: maps a rendered prompt to a reply text.

    ``reply`` is a pure function of the rendered prompt (the user message),
    so the fake can return the correct answer for one task and a wrong answer
    for another. Records every request it served for determinism assertions.
    No network is touched.
    """

    reply: ReplyFn
    policy: ProviderTransportPolicy = field(default_factory=transport_policy)
    served: list[ProviderCallRequest] = field(default_factory=list)

    def __call__(
        self, request: ProviderCallRequest
    ) -> ProviderInvocationEvidence:
        self.served.append(request)
        text = self.reply(_prompt_of(request))
        raw_request = RawHttpRequest.build(
            url="https://example.test/v1/chat/completions",
            headers={"Authorization": "Bearer k", "content-type": "json"},
            body={"model": "test-model"},
        )
        return ProviderInvocationEvidence.build(
            request=request,
            policy=self.policy,
            raw_request=raw_request,
            outcome=_response(text),
        )


def constant_reply(text: str) -> ReplyFn:
    """A reply function returning ``text`` for every prompt."""

    def _reply(_prompt: str) -> str:
        return text

    return _reply


def synthetic_ed1_tasks(count: int = 3) -> tuple[Ed1Instance, ...]:
    """Build a deterministic, sandbox-runnable HumanEval fixture pool."""
    tasks: list[Ed1Instance] = []
    for index in range(count):
        entry_point = f"add_{index}"
        task = HumanEvalTask(
            task_id=f"Synthetic/{index}",
            prompt=(
                f"def {entry_point}(x):\n"
                f'    """Return x plus {index + 1}.\n\n'
                f"    >>> {entry_point}(1)\n"
                f"    {index + 2}\n"
                '    """\n'
            ),
            canonical_solution=f"    return x + {index + 1}\n",
            entry_point=entry_point,
            test=(
                "def check(candidate):\n"
                "    inputs = [(1,), (-1,)]\n"
                f"    results = [{index + 2}, {index}]\n"
                "    for inp, expected in zip(inputs, results):\n"
                "        assertion(candidate(*inp), expected)\n"
            ),
        )
        tasks.append(ed1_instance_from_task(task))
    return tuple(tasks)
