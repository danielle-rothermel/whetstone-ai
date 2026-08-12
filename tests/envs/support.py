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
from whetstone.envs.factory import EnvExperiment
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

ReplyFn = Callable[[str], str]
RowPayloadFn = Callable[[Instance, int, int], BaseModel | JsonValue]


def non_code_comp_experiment() -> EnvExperiment:
    """An EnvExperiment that is not a code_comp typed experiment."""
    base = code_comp_direct_experiment(
        task_count=2,
        internal_n=1,
        official_n=1,
    )
    return EnvExperiment(
        env_name="legacy",
        generation_graph=base.generation_graph,
        initial_candidate=base.initial_candidate,
        ceiling_candidate=base.ceiling_candidate,
        eval_configs=base.eval_configs,
        reward_policy=base.reward_policy,
        completeness_policy=base.completeness_policy,
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

        if not isinstance(request, DirectRowRequest | EncDecRowRequest):
            raise TypeError(f"unsupported row request {type(request)!r}")
        instance = request.instance.to_instance()
        sample_index = request.sample_index
        drive_ordinal = request.drive_ordinal
        outcome = payload_for(instance, sample_index, drive_ordinal)
        if isinstance(request, DirectRowRequest):
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


def code_comp_direct_experiment(
    *,
    num_samples: int = 1,
    task_count: int = 3,
    internal_n: int = 1,
    official_n: int = 1,
    model: str = TEST_MODEL,
) -> EnvExperiment:
    from whetstone.envs.code_comp import (
        CodeCompMode,
        build_code_comp_experiment,
    )

    return build_code_comp_experiment(
        CodeCompMode.DIRECT,
        tasks=synthetic_code_comp_tasks(task_count),
        internal_n=internal_n,
        official_n=official_n,
        num_samples=num_samples,
        model=model,
    )


def in_process_direct_row_job_factory(
    reply_for: ReplyFn | None = None,
) -> Callable[[BaseModel], ProcessJob]:
    """Drive the real D1 row adapter in-process and return its payload."""
    from functools import partial

    from tests.execution.fake_python import local_python_executor
    from whetstone.envs.code_comp.dataset import CodeCompTaskInstance
    from whetstone.envs.code_comp.modes.direct import build_direct_experiment
    from whetstone.envs.code_comp.scoring import score_code_comp_submission
    from whetstone.evaluation.drivers.code_comp.direct import (
        DirectRowRequest,
        DirectRowResult,
        drive_direct_row,
    )

    def build(request: BaseModel) -> ProcessJob:
        if not isinstance(request, DirectRowRequest):
            raise TypeError(f"unsupported row request {type(request)!r}")
        instance = request.instance.to_instance()
        task = request.humaneval_task.to_task()
        experiment = build_direct_experiment(
            input_arm=request.input_arm,
            rename_token=request.rename_token,
            tasks=(
                CodeCompTaskInstance(
                    instance=instance,
                    humaneval_task=task,
                ),
            ),
            internal_n=1,
            official_n=1,
            num_samples=1,
        )
        if (
            experiment.generation_graph.procedure_config_hash
            != request.procedure_config_hash
        ):
            raise ValueError("D1 row procedure identity is not canonical")
        answer = (
            reply_for(task.ground_truth_code)
            if reply_for is not None
            else task.ground_truth_code
        )
        outcome = drive_direct_row(
            experiment=experiment,
            candidate_body=request.candidate_body,
            instance=instance,
            provider_call_config=request.provider_call_config,
            execution_policy=request.execution_policy,
            transport=FakeTransport(constant_reply(answer)),
            scorer=partial(
                score_code_comp_submission,
                executor=local_python_executor(),
            ),
            logical_call_id=request.logical_call_id,
            sample_index=request.sample_index,
            drive_ordinal=request.drive_ordinal,
            cache=None,
            cache_phase=request.cache_phase,
            cache_unit=request.cache_unit,
        )
        return ProcessJob(
            entrypoint="tests.envs.process_workers:return_payload",
            payload=DirectRowResult(
                request_hash=request.request_hash,
                outcome=outcome,
            ).model_dump(mode="json"),
        )

    return build
