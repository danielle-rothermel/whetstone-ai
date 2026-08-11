from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from dr_code.humaneval import HumanEvalTask
from dr_providers import (
    ProviderCallRequest,
    ProviderHttpRequestEvidence,
    ProviderInvocationEvidence,
    ProviderKind,
    ProviderTransportPolicy,
    ProviderTransportResponse,
    policy_for,
)
from pydantic import BaseModel, JsonValue
from whetstone_envs.core import Instance

from whetstone.core.roles import EvaluationRole
from whetstone.envs.code_comp.dataset import (
    CodeCompTaskInstance,
    ed1_instance_from_task,
)
from whetstone.envs.factory import EnvExperiment, build_env_experiment
from whetstone.envs.registry import env_spec
from whetstone.envs.sampling import EnvSplitSampling
from whetstone.execution.fanout import ProcessJob
from whetstone.experiment.binding import (
    EVALUATION_BINDING_SCHEMA_VERSION,
    EvaluationBinding,
    eval_config_reference,
)
from whetstone.provider.policy import ProviderExecutionPolicy

API_KEY_ENV = "OPENROUTER_API_KEY"
TEST_MODEL = "openai/gpt-5-nano"
_TINY_SPLIT = (2, 2, 2)
TINY_SPLIT_FIT_CEILING = sum(_TINY_SPLIT)

ReplyFn = Callable[[str], str]
RowPayloadFn = Callable[[Instance, int, int], BaseModel | JsonValue]


def tiny_split_fits(env, n: int) -> bool:
    pool = env.generate_pool(n_per_stratum=n)
    if not env.stratified_split:
        return len(pool) >= sum(_TINY_SPLIT)
    n_strata = len(pool.strata)
    per_stratum_max = sum(-(-part // n_strata) for part in _TINY_SPLIT)
    return n >= per_stratum_max


def tiny_experiment(env_name: str) -> EnvExperiment:
    env = env_spec(env_name)
    attempted_sizes: list[int] = []
    for n in range(1, TINY_SPLIT_FIT_CEILING + 1):
        attempted_sizes.append(n)
        if tiny_split_fits(env, n):
            break
    else:
        raise AssertionError(
            f"{env_name} could not fit split {_TINY_SPLIT} by independently "
            f"derived n_per_stratum ceiling {TINY_SPLIT_FIT_CEILING}; "
            f"attempted_sizes={attempted_sizes}; "
            f"final_attempted_size={attempted_sizes[-1]}"
        )
    return build_env_experiment(
        env_name,
        model=TEST_MODEL,
        pool_n_per_stratum=n,
        split_sizes=_TINY_SPLIT,
        num_samples=2,
    )


def row_job_factory(
    payload_for: RowPayloadFn,
) -> Callable[[BaseModel], ProcessJob]:

    def build(request: BaseModel) -> ProcessJob:
        from whetstone.evaluation.drivers.code_comp.direct import (
            DirectGeneratedRowOutcome,
            DirectRowOutcome,
            DirectRowRequest,
            DirectRowResult,
        )
        from whetstone.evaluation.drivers.code_comp.encdec import (
            EncDecGeneratedRowOutcome,
            EncDecRowOutcome,
            EncDecRowRequest,
            EncDecRowResult,
        )
        from whetstone.evaluation.drivers.internal import (
            InternalRowOutcome,
            InternalRowRequest,
            InternalRowResult,
        )

        if not isinstance(
            request, InternalRowRequest | DirectRowRequest | EncDecRowRequest
        ):
            raise TypeError(f"unsupported row request {type(request)!r}")
        instance = request.instance.to_instance()
        sample_index = request.sample_index
        drive_ordinal = request.drive_ordinal
        outcome = payload_for(instance, sample_index, drive_ordinal)
        if isinstance(request, InternalRowRequest):
            if not isinstance(outcome, InternalRowOutcome):
                raise TypeError("internal request requires InternalRowOutcome")
            envelope = InternalRowResult(
                request_hash=request.request_hash,
                outcome=outcome,
            )
        elif isinstance(request, DirectRowRequest):
            if not isinstance(
                outcome, DirectRowOutcome | DirectGeneratedRowOutcome
            ):
                raise TypeError("D1 request requires a D1 row outcome")
            envelope = DirectRowResult(
                request_hash=request.request_hash,
                outcome=outcome,
            )
        else:
            if not isinstance(
                outcome, EncDecRowOutcome | EncDecGeneratedRowOutcome
            ):
                raise TypeError("ED1 request requires an ED1 row outcome")
            envelope = EncDecRowResult(
                request_hash=request.request_hash,
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

    def build(request: BaseModel) -> ProcessJob:
        return ProcessJob(
            entrypoint=entrypoint,
            payload=request.model_dump(mode="json"),
        )

    return build


_PROCESS_INTERNAL_ROW_JOB_FACTORY = process_row_job_factory(
    "tests.envs.process_workers:drive_internal_success"
)


def in_process_internal_row_job_factory(
    reply_for: ReplyFn | None = None,
) -> Callable[[BaseModel], ProcessJob]:
    """Drive the real internal row adapter in-process and return its payload.

    The row still travels the engine's exact ProcessJob seam, but the work
    happens here rather than in a spawned worker, so the test stays fast while
    the evidence it produces is real.
    """
    from whetstone.envs.procedure import env_procedure_config
    from whetstone.evaluation.drivers.internal import (
        InternalRowRequest,
        InternalRowResult,
        drive_internal_row,
    )

    def build(request: BaseModel) -> ProcessJob:
        if not isinstance(request, InternalRowRequest):
            raise TypeError(f"unsupported row request {type(request)!r}")
        instance = request.instance.to_instance()
        env = env_spec(request.env_name)
        if (
            env_procedure_config(env).config_hash
            != request.procedure_config_hash
        ):
            raise ValueError("row procedure identity is not canonical")
        answer = (
            reply_for(instance.gold)
            if reply_for is not None
            else instance.gold
        )
        outcome = drive_internal_row(
            env,
            candidate=request.candidate,
            instance=instance,
            provider_call_config=request.provider_call_config,
            execution_policy=request.execution_policy,
            transport=FakeTransport(constant_reply(answer)),
            procedure_config_hash=request.procedure_config_hash,
            logical_call_id=request.logical_call_id,
            sample_index=request.sample_index,
            drive_ordinal=request.drive_ordinal,
            cache=None,
            cache_phase=request.cache_phase,
            cache_unit=request.cache_unit,
            render_guard=request.render_guard,
        )
        return ProcessJob(
            entrypoint="tests.envs.process_workers:return_payload",
            payload=InternalRowResult(
                request_hash=request.request_hash,
                outcome=outcome,
            ).model_dump(mode="json"),
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
        response_body={"choices": [{"message": {"content": text}}]},
        response_id="resp-1",
        model="test-model",
        finish_reason="stop",
    )


def _prompt_of(request: ProviderCallRequest) -> str:
    messages = request.transcript.messages
    return messages[-1].content if messages else ""


@dataclass
class FakeTransport:
    reply: ReplyFn
    policy: ProviderTransportPolicy = field(default_factory=transport_policy)
    served: list[ProviderCallRequest] = field(default_factory=list)

    def __call__(
        self, request: ProviderCallRequest
    ) -> ProviderInvocationEvidence:
        self.served.append(request)
        text = self.reply(_prompt_of(request))
        http_request = ProviderHttpRequestEvidence.build(
            url="https://example.test/v1/chat/completions",
            headers={"Authorization": "Bearer k", "content-type": "json"},
            body={"model": "test-model"},
        )
        return ProviderInvocationEvidence.build(
            request=request,
            policy=self.policy,
            http_request=http_request,
            outcome=_response(text),
        )


def constant_reply(text: str) -> ReplyFn:

    def _reply(_prompt: str) -> str:
        return text

    return _reply


def synthetic_code_comp_tasks(
    count: int = 3,
) -> tuple[CodeCompTaskInstance, ...]:
    tasks: list[CodeCompTaskInstance] = []
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
