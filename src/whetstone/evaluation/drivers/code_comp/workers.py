from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

from dr_providers import (
    HttpProvider,
    ProviderCallRequest,
    ProviderHttpRequestEvidence,
    ProviderInvocationEvidence,
    ProviderTransportResponse,
)
from pydantic import JsonValue

from whetstone.envs.code_comp.constants import DECODER_TEMPLATE, ENCODER_BODY_A
from whetstone.envs.code_comp.dataset import (
    CodeCompTaskInstance,
    humaneval_task_from_instance,
)
from whetstone.envs.code_comp.modes.encdec import EncDecExperiment
from whetstone.envs.code_comp.registry import (
    CodeCompMode,
    build_code_comp_experiment,
)
from whetstone.evaluation.drivers.code_comp.encdec import (
    EncDecRowRequest,
    EncDecRowResult,
    drive_encdec_row,
)
from whetstone.execution.prompt_cache import PromptResultCache

DUMMY_PASSING_BODY = (
    "Describe the inputs, outputs, edge cases, and exact behavior"
)
DUMMY_FAILING_BODY = "Give only a tiny implementation hint"
DUMMY_ALTERNATE_PASSING_BODY = (
    "Explain the observable behavior for an exact Python reconstruction"
)

_BASELINE_DESCRIPTION = (
    "Reconstruct the function exactly while preserving every behavior and "
    "edge case described by the implementation."
)
_PASSING_DESCRIPTION = "Implement identical behavior for every input."
_ALTERNATE_PASSING_DESCRIPTION = (
    "Recreate all observable input and output behavior."
)
_FAILING_DESCRIPTION = "Return a placeholder value."


def _description_for(body: str) -> str:
    if body == DUMMY_PASSING_BODY:
        return _PASSING_DESCRIPTION
    if body == DUMMY_FAILING_BODY:
        return _FAILING_DESCRIPTION
    if body == DUMMY_ALTERNATE_PASSING_BODY:
        return _ALTERNATE_PASSING_DESCRIPTION
    if body == ENCODER_BODY_A:
        return _BASELINE_DESCRIPTION
    return f"Reconstruct the function using this guidance: {body}."


def _reconstruct_worker_experiment(
    request: EncDecRowRequest,
) -> tuple[EncDecExperiment, CodeCompTaskInstance]:
    if request.mutant_record is not None:
        raise ValueError("COPRO scoring preview supports ED1 only")
    instance = request.instance.to_instance()
    task = humaneval_task_from_instance(instance)
    fixture = CodeCompTaskInstance(instance=instance, humaneval_task=task)
    experiment = cast(
        EncDecExperiment,
        build_code_comp_experiment(
            CodeCompMode.ENCDEC,
            provider_call_config=request.provider_call_config,
            budget_ratio=request.budget_ratio,
            tasks=(fixture,),
            internal_n=1,
            official_n=1,
            num_samples=1,
        ),
    )
    generation_graph = experiment.encdec_generation_graph
    assert generation_graph is not None
    if generation_graph.graph_hash != request.graph_hash:
        raise ValueError("ED1 worker reconstructed a different graph")
    if generation_graph.procedure_config_hash != request.procedure_config_hash:
        raise ValueError("ED1 worker reconstructed a different procedure")
    if generation_graph.provider_call_config != request.provider_call_config:
        raise ValueError(
            "ED1 worker reconstructed a different provider config"
        )
    return experiment, fixture


def _prompt_cache(request: EncDecRowRequest) -> PromptResultCache | None:
    if request.cache_root is None:
        return None
    return PromptResultCache(Path(request.cache_root))


@dataclass(slots=True)
class _DummyGenerationTransport:
    request: EncDecRowRequest
    passing_source: str
    failing_source: str
    description: str
    call_index: int = 0

    def __call__(
        self, provider_request: ProviderCallRequest
    ) -> ProviderInvocationEvidence:
        messages = provider_request.transcript.messages
        prompt = messages[-1].content if messages else ""
        if self.call_index == 0:
            text = self.description
        elif self.call_index == 1:
            expected = DECODER_TEMPLATE.format(encoder_output=self.description)
            if prompt != expected:
                raise ValueError(
                    "dummy decoder received an unexpected rendered prompt"
                )
            text = (
                self.failing_source
                if self.request.candidate_template == DUMMY_FAILING_BODY
                else self.passing_source
            )
        else:
            raise ValueError("dummy ED1 row made more than two provider calls")
        self.call_index += 1
        http_request = ProviderHttpRequestEvidence.build(
            url="https://example.invalid/v1/chat/completions",
            headers={"content-type": "application/json"},
            body={
                "model": (provider_request.config.definition.route.model),
                "dummy": True,
            },
        )
        response = ProviderTransportResponse(
            text=text,
            response_body={"choices": [{"message": {"content": text}}]},
            response_id=f"dummy-ed1-{self.call_index}",
            model=provider_request.config.definition.route.model,
            finish_reason="stop",
        )
        return ProviderInvocationEvidence.build(
            request=provider_request,
            policy=self.request.execution_policy.transport_policy,
            http_request=http_request,
            outcome=response,
        )


def drive_dummy_encdec_generation(payload: JsonValue) -> JsonValue:
    """Drive one real ED1 encode/decode row with scripted model responses."""

    request = EncDecRowRequest.from_process_payload(payload)
    experiment, fixture = _reconstruct_worker_experiment(request)
    instance = fixture.instance
    task = fixture.humaneval_task
    description = _description_for(request.candidate_template)
    transport = _DummyGenerationTransport(
        request=request,
        passing_source=task.ground_truth_code,
        failing_source=(
            f"def {task.entry_point}(*args, **kwargs):\n    return None\n"
        ),
        description=description,
    )
    outcome = drive_encdec_row(
        experiment=experiment,
        candidate_template=request.candidate_template,
        instance=instance,
        provider_call_config=request.provider_call_config,
        execution_policy=request.execution_policy,
        transport=transport,
        scorer=None,
        logical_call_id=request.logical_call_id,
        sample_index=request.sample_index,
        drive_ordinal=request.drive_ordinal,
        cache=_prompt_cache(request),
        cache_phase=request.cache_phase,
        cache_unit=request.cache_unit,
    )
    if transport.call_index != 2:
        raise ValueError("dummy ED1 row did not complete both provider calls")
    return EncDecRowResult(
        request_hash=request.request_hash,
        outcome=outcome,
    ).model_dump(mode="json")


def drive_provider_encdec_call(payload: JsonValue) -> JsonValue:
    """Drive one ED1 row through real dr-providers encoder/decoder calls."""

    request = EncDecRowRequest.from_process_payload(payload)
    experiment, fixture = _reconstruct_worker_experiment(request)
    with HttpProvider(
        policy=request.execution_policy.transport_policy
    ) as provider:
        outcome = drive_encdec_row(
            experiment=experiment,
            candidate_template=request.candidate_template,
            instance=fixture.instance,
            provider_call_config=request.provider_call_config,
            execution_policy=request.execution_policy,
            transport=provider.invoke,
            scorer=None,
            logical_call_id=request.logical_call_id,
            sample_index=request.sample_index,
            drive_ordinal=request.drive_ordinal,
            cache=_prompt_cache(request),
            cache_phase=request.cache_phase,
            cache_unit=request.cache_unit,
        )
    return EncDecRowResult(
        request_hash=request.request_hash,
        outcome=outcome,
    ).model_dump(mode="json")


__all__ = [
    "DUMMY_ALTERNATE_PASSING_BODY",
    "DUMMY_FAILING_BODY",
    "DUMMY_PASSING_BODY",
    "drive_dummy_encdec_generation",
    "drive_provider_encdec_call",
]
