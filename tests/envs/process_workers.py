from __future__ import annotations

from functools import partial

from pydantic import JsonValue


def return_payload(payload: JsonValue) -> JsonValue:
    return payload


def _cache(root: str | None):
    if root is None:
        return None
    from pathlib import Path

    from whetstone.execution.prompt_cache import PromptResultCache

    return PromptResultCache(Path(root))


def drive_internal_success(payload: JsonValue) -> JsonValue:
    from tests.envs.support import FakeTransport, constant_reply
    from whetstone.envs.procedure import env_procedure_config
    from whetstone.envs.registry import env_spec
    from whetstone.evaluation.drivers.internal import (
        InternalRowRequest,
        InternalRowResult,
        drive_internal_row,
    )

    request = InternalRowRequest.from_process_payload(payload)
    instance = request.instance.to_instance()
    env = env_spec(request.env_name)
    if (
        env_procedure_config(env).config_identity_hash
        != request.procedure_config_hash
    ):
        raise ValueError("internal row procedure identity is not canonical")
    outcome = drive_internal_row(
        env,
        candidate=request.candidate,
        instance=instance,
        provider_call_config=request.provider_call_config,
        execution_policy=request.execution_policy,
        transport=FakeTransport(constant_reply(instance.gold)),
        procedure_config_hash=request.procedure_config_hash,
        logical_call_id=request.logical_call_id,
        repeat_index=request.repeat_index,
        drive_ordinal=request.drive_ordinal,
        cache=_cache(request.cache_root),
        cache_phase=request.cache_phase,
        cache_unit=request.cache_unit,
        render_guard=request.render_guard,
    )
    return InternalRowResult(
        request_identity=request.request_identity,
        outcome=outcome,
    ).model_dump(mode="json")


def drive_d1_success(payload: JsonValue) -> JsonValue:
    from tests.envs.support import FakeTransport, constant_reply
    from tests.execution.fake_python import local_python_executor
    from whetstone.envs.code_comp.dataset import CodeCompTaskInstance
    from whetstone.envs.code_comp.modes.direct import build_direct_experiment
    from whetstone.envs.code_comp.scoring import score_code_comp_submission
    from whetstone.evaluation.drivers.code_comp.direct import (
        DirectRowRequest,
        DirectRowResult,
        drive_direct_row,
    )

    request = DirectRowRequest.from_process_payload(payload)
    instance = request.instance.to_instance()
    task = request.humaneval_task.to_task()
    experiment = build_direct_experiment(
        input_arm=request.input_arm,
        rename_token=request.rename_token,
        tasks=(CodeCompTaskInstance(instance=instance, humaneval_task=task),),
        internal_n=1,
        official_n=1,
        repeats=1,
    )
    if (
        experiment.rollout_definition.procedure_config_hash
        != request.procedure_config_hash
    ):
        raise ValueError("D1 row procedure identity is not canonical")
    outcome = drive_direct_row(
        experiment=experiment,
        candidate_body=request.candidate_body,
        instance=instance,
        provider_call_config=request.provider_call_config,
        execution_policy=request.execution_policy,
        transport=FakeTransport(constant_reply(task.ground_truth_code)),
        scorer=partial(
            score_code_comp_submission,
            executor=local_python_executor(),
        ),
        logical_call_id=request.logical_call_id,
        repeat_index=request.repeat_index,
        drive_ordinal=request.drive_ordinal,
        cache=_cache(request.cache_root),
        cache_phase=request.cache_phase,
        cache_unit=request.cache_unit,
    )
    return DirectRowResult(
        request_identity=request.request_identity,
        outcome=outcome,
    ).model_dump(mode="json")


def drive_ed1_success(payload: JsonValue) -> JsonValue:
    return _drive_ed1(payload, transient_first=False)


def drive_ed1_transient_then_success(payload: JsonValue) -> JsonValue:
    return _drive_ed1(payload, transient_first=True)


def _drive_ed1(payload: JsonValue, *, transient_first: bool) -> JsonValue:
    from dataclasses import fields

    from dr_code.humaneval import HumanEvalTask
    from dr_providers import (
        FailureClass,
        ProviderHttpRequestEvidence,
        ProviderInvocationEvidence,
        ProviderTransportFailure,
    )

    from tests.envs.support import FakeTransport
    from tests.execution.fake_python import local_python_executor
    from whetstone.envs.code_comp.constants import DECODER_TEMPLATE
    from whetstone.envs.code_comp.dataset import (
        CodeCompTaskInstance,
        ed1_instance_from_task,
        humaneval_task_from_instance,
    )
    from whetstone.envs.code_comp.modes.encdec import (
        EncDecExperiment,
        build_encdec_experiment,
    )
    from whetstone.envs.code_comp.mutant.oracle import (
        score_mutant_reconstruction,
    )
    from whetstone.envs.code_comp.scoring import score_code_comp_submission
    from whetstone.evaluation.drivers.code_comp.encdec import (
        EncDecRowRequest,
        EncDecRowResult,
        drive_encdec_row,
    )

    request = EncDecRowRequest.from_process_payload(payload)
    instance = request.instance.to_instance()
    mutant = request.mutant_record
    if mutant is None:
        task = humaneval_task_from_instance(instance)
        task_fixture = CodeCompTaskInstance(
            instance=instance, humaneval_task=task
        )
        decoder_source = task.ground_truth_code
    else:
        canonical_solution = mutant.canonical_full_source
        if canonical_solution.startswith(mutant.prompt):
            canonical_solution = canonical_solution[len(mutant.prompt) :]
        task = HumanEvalTask(
            task_id=mutant.task_id,
            prompt=mutant.prompt,
            canonical_solution=canonical_solution,
            entry_point=mutant.entry_point,
            test=mutant.canonical_test,
        )
        task_fixture = ed1_instance_from_task(task)
        decoder_source = mutant.mutated_full_source
    experiment = build_encdec_experiment(
        provider_call_config=request.provider_call_config,
        budget_ratio=request.budget_ratio,
        tasks=(task_fixture,),
        internal_n=1,
        official_n=1,
        repeats=1,
    )
    if mutant is not None:
        from whetstone.envs.code_comp.modes.mutant import (
            MutantExperiment,
            build_mutant_procedure_config,
        )

        if (
            build_mutant_procedure_config().config_identity_hash
            != request.procedure_config_hash
        ):
            raise ValueError("ED1M row procedure identity is not canonical")

        experiment_fields = {
            field.name: getattr(experiment, field.name)
            for field in fields(EncDecExperiment)
        }
        experiment_fields["env_name"] = request.env_name
        experiment_fields["dataset_revision"] = request.dataset_revision
        experiment = MutantExperiment(
            **experiment_fields,
            mutants={mutant.content_identity: mutant},
        )
    elif (
        experiment.rollout_definition.procedure_config_hash
        != request.procedure_config_hash
    ):
        raise ValueError("ED1 row procedure identity is not canonical")

    def reply(prompt: str) -> str:
        decoder_prefix = DECODER_TEMPLATE.split("{encoder_output}", 1)[0]
        if prompt.startswith(decoder_prefix):
            return decoder_source
        return "A compact executable reconstruction description."

    transport = FakeTransport(reply)
    if transient_first and request.drive_ordinal == 0:

        def transient_transport(provider_request):
            http_request = ProviderHttpRequestEvidence.build(
                url="https://example.test/v1/chat/completions",
                headers={"content-type": "application/json"},
                body={"model": "test-model"},
            )
            return ProviderInvocationEvidence.build(
                request=provider_request,
                policy=request.execution_policy.transport_policy,
                http_request=http_request,
                outcome=ProviderTransportFailure(
                    failure_class=FailureClass.TRANSIENT,
                    code="transport_error",
                    message="scripted first-drive failure",
                    retryable=True,
                ),
            )

        transport = transient_transport

    scorer = partial(
        (
            score_mutant_reconstruction
            if mutant is not None
            else score_code_comp_submission
        ),
        executor=local_python_executor(),
    )
    outcome = drive_encdec_row(
        experiment=experiment,
        candidate_template=request.candidate_template,
        instance=instance,
        provider_call_config=request.provider_call_config,
        execution_policy=request.execution_policy,
        transport=transport,
        scorer=scorer,
        logical_call_id=request.logical_call_id,
        repeat_index=request.repeat_index,
        drive_ordinal=request.drive_ordinal,
        cache=_cache(request.cache_root),
        cache_phase=request.cache_phase,
        cache_unit=request.cache_unit,
    )
    return EncDecRowResult(
        request_identity=request.request_identity,
        outcome=outcome,
    ).model_dump(mode="json")
