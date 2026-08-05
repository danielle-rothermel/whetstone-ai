"""Importable process jobs for environment-driver integration tests."""

from __future__ import annotations

from pydantic import JsonValue


def return_payload(payload: JsonValue) -> JsonValue:
    """Return a prevalidated row payload from a real child process."""
    return payload


def _cache(root: str | None):
    if root is None:
        return None
    from pathlib import Path

    from whetstone.execution.prompt_cache import PromptResultCache

    return PromptResultCache(Path(root))


def drive_internal_success(payload: JsonValue) -> JsonValue:
    """Run the real internal-row adapter with a child-local fake transport."""
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
    """Run the real D1 row adapter with child-local transport and scorer."""
    from tests.envs.support import FakeTransport, constant_reply
    from whetstone.envs.d1 import build_d1_experiment
    from whetstone.envs.ed1 import Ed1Instance
    from whetstone.envs.ed1_scoring import score_ed1_submission
    from whetstone.evaluation.drivers.d1 import (
        D1RowRequest,
        D1RowResult,
        drive_d1_row,
    )

    request = D1RowRequest.from_process_payload(payload)
    instance = request.instance.to_instance()
    task = request.humaneval_task.to_task()
    experiment = build_d1_experiment(
        input_arm=request.input_arm,
        rename_token=request.rename_token,
        tasks=(Ed1Instance(instance=instance, humaneval_task=task),),
        internal_n=1,
        official_n=1,
        repeats=1,
    )
    if (
        experiment.rollout_definition.procedure_config_hash
        != request.procedure_config_hash
    ):
        raise ValueError("D1 row procedure identity is not canonical")
    outcome = drive_d1_row(
        experiment=experiment,
        candidate_body=request.candidate_body,
        instance=instance,
        provider_call_config=request.provider_call_config,
        execution_policy=request.execution_policy,
        transport=FakeTransport(constant_reply(task.ground_truth_code)),
        scorer=score_ed1_submission,
        logical_call_id=request.logical_call_id,
        repeat_index=request.repeat_index,
        drive_ordinal=request.drive_ordinal,
        cache=_cache(request.cache_root),
        cache_phase=request.cache_phase,
        cache_unit=request.cache_unit,
    )
    return D1RowResult(
        request_identity=request.request_identity,
        outcome=outcome,
    ).model_dump(mode="json")


def drive_ed1_success(payload: JsonValue) -> JsonValue:
    """Run the real ED1 row adapter with child-local transport and scorer."""
    return _drive_ed1(payload, transient_first=False)


def drive_ed1_transient_then_success(payload: JsonValue) -> JsonValue:
    """Fail drive zero in transport, then run drive one normally."""
    return _drive_ed1(payload, transient_first=True)


def _drive_ed1(payload: JsonValue, *, transient_first: bool) -> JsonValue:
    from dataclasses import fields

    from dr_code.humaneval import HumanEvalTask
    from dr_providers import (
        FailureClass,
        ProviderInvocationEvidence,
        ProviderTransportFailure,
        RawHttpRequest,
    )

    from tests.envs.support import FakeTransport
    from whetstone.envs.ed1 import (
        DECODER_TEMPLATE,
        Ed1Experiment,
        Ed1Instance,
        build_ed1_experiment,
        ed1_instance_from_task,
        humaneval_task_from_instance,
    )
    from whetstone.envs.ed1_scoring import score_ed1_submission
    from whetstone.evaluation.drivers.ed1 import (
        Ed1RowRequest,
        Ed1RowResult,
        drive_ed1_row,
    )

    request = Ed1RowRequest.from_process_payload(payload)
    instance = request.instance.to_instance()
    mutant = request.mutant_record
    if mutant is None:
        task = humaneval_task_from_instance(instance)
        task_fixture = Ed1Instance(instance=instance, humaneval_task=task)
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
    experiment = build_ed1_experiment(
        budget_ratio=request.budget_ratio,
        tasks=(task_fixture,),
        internal_n=1,
        official_n=1,
        repeats=1,
    )
    if mutant is not None:
        from whetstone.envs.ed1m import (
            Ed1mExperiment,
            build_ed1m_procedure_config,
        )

        if (
            build_ed1m_procedure_config().config_identity_hash
            != request.procedure_config_hash
        ):
            raise ValueError("ED1M row procedure identity is not canonical")

        experiment_fields = {
            field.name: getattr(experiment, field.name)
            for field in fields(Ed1Experiment)
        }
        experiment_fields["env_name"] = request.env_name
        experiment_fields["dataset_revision"] = request.dataset_revision
        experiment = Ed1mExperiment(
            **experiment_fields,
            mutants={mutant.content_identity: mutant},
        )
    elif (
        experiment.rollout_definition.procedure_config_hash
        != request.procedure_config_hash
    ):
        raise ValueError("ED1 row procedure identity is not canonical")

    def reply(prompt: str) -> str:
        if prompt.startswith(DECODER_TEMPLATE.split("{encoder_output}")[0]):
            return decoder_source
        return "A compact executable reconstruction description."

    transport = FakeTransport(reply)
    if transient_first and request.drive_ordinal == 0:

        def transient_transport(provider_request):
            raw_request = RawHttpRequest.build(
                url="https://example.test/v1/chat/completions",
                headers={"content-type": "application/json"},
                body={"model": "test-model"},
            )
            return ProviderInvocationEvidence.build(
                request=provider_request,
                policy=request.execution_policy.transport_policy,
                raw_request=raw_request,
                outcome=ProviderTransportFailure(
                    failure_class=FailureClass.TRANSIENT,
                    code="transport_error",
                    message="scripted first-drive failure",
                    retryable=True,
                ),
            )

        transport = transient_transport

    outcome = drive_ed1_row(
        experiment=experiment,
        candidate_template=request.candidate_template,
        instance=instance,
        provider_call_config=request.provider_call_config,
        execution_policy=request.execution_policy,
        transport=transport,
        scorer=score_ed1_submission,
        logical_call_id=request.logical_call_id,
        repeat_index=request.repeat_index,
        drive_ordinal=request.drive_ordinal,
        cache=_cache(request.cache_root),
        cache_phase=request.cache_phase,
        cache_unit=request.cache_unit,
    )
    return Ed1RowResult(
        request_identity=request.request_identity,
        outcome=outcome,
    ).model_dump(mode="json")
