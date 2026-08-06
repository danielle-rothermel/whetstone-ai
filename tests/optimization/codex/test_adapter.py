from __future__ import annotations

import pytest

from tests.optimization.codex.support import (
    binding,
    fake_runner,
    proposals,
    request,
    stack,
)
from tests.optimization.support import make_harness, registry
from whetstone.core.effects.authority import ReplayPolicy
from whetstone.core.identity import TypedRef
from whetstone.envs.factory import EnvExperiment
from whetstone.experiment.candidate import Candidate, candidate_reference
from whetstone.optimization.codex.adapter import (
    CODEX_OUTPUT_ARTIFACT_SCHEMA,
    CodexAdapter,
    OpaqueStepError,
)
from whetstone.optimization.contracts import StepStatus


def test_harness_persists_codex_result_and_replays_without_redrive(
    tmp_path, codex_experiment: EnvExperiment
) -> None:
    codex = stack(tmp_path, codex_experiment)
    adapter = CodexAdapter(
        codex.runner,
        store=codex.store,
        tool_store=codex.tool_store,
    )
    step_request = request(codex.base, codex.config)
    harness = make_harness(
        store=codex.store,
        adapter_registry=registry(adapter),
        run=step_request.run,
        tool_store=codex.tool_store,
        effect_authority=codex.tool_store.effect_authority,
        tool_executor=codex.executor,
        adapter_replay_policy=ReplayPolicy.NO_REDRIVE,
    )

    result, result_ref = harness.run_step(step_request)

    assert result.status is StepStatus.COMPLETE
    assert len(result.accepted_candidates) == 2
    assert {item.record.base_ref for item in result.accepted_candidates} == {
        candidate_reference(codex.base).record_ref
    }
    assert len(result.tool_evidence) == 1
    assert codex.runner.observed_payloads[0]["refused"] is False
    assert result.state_ref is not None
    state = codex.store.get(result.state_ref.reference)
    assert isinstance(state, dict)
    artifact_ref = TypedRef.model_validate(state["codex_output_artifact_ref"])
    assert artifact_ref.schema_name == CODEX_OUTPUT_ARTIFACT_SCHEMA
    artifact = codex.store.get(artifact_ref.reference)
    assert isinstance(artifact, dict)
    assert artifact["run_id"] == step_request.run_id
    assert state["harness_store_accepted_call_count"] == 1
    assert state["tool_namespace"] == str(codex.config.store_namespace_key)

    class ExplodingRegistry:
        def resolve(self, adapter_key: str):
            raise AssertionError(f"resolved {adapter_key}")

    replay, replay_ref = make_harness(
        store=codex.store,
        adapter_registry=ExplodingRegistry(),
        run=step_request.run,
        tool_store=codex.tool_store,
        effect_authority=codex.tool_store.effect_authority,
        tool_executor=codex.executor,
        adapter_replay_policy=ReplayPolicy.NO_REDRIVE,
    ).run_step(step_request)
    assert (replay, replay_ref) == (result, result_ref)


def test_distinct_bases_is_conditional(
    tmp_path, codex_experiment: EnvExperiment
) -> None:
    codex = stack(tmp_path, codex_experiment)

    allowed_request = request(codex.base, codex.config, distinct=False)
    allowed_handle = codex.executor.runtime_handle(
        codex.config,
        codex.tool_store,
        binding(allowed_request),
    )
    allowed = CodexAdapter(
        fake_runner(codex.base, call_id="agent-call-allowed"),
        store=codex.store,
        tool_store=codex.tool_store,
    ).invoke(allowed_request, (allowed_handle,))

    rejected_request = request(
        codex.base,
        codex.config,
        distinct=True,
        run_id="codex-run-distinct",
    )
    rejected_handle = codex.executor.runtime_handle(
        codex.config,
        codex.tool_store,
        binding(rejected_request),
    )
    rejected = CodexAdapter(
        fake_runner(codex.base, call_id="agent-call-rejected"),
        store=codex.store,
        tool_store=codex.tool_store,
    ).invoke(rejected_request, (rejected_handle,))

    assert len(allowed.accepted_candidates) == 2
    assert allowed.proposed_status is StepStatus.COMPLETE
    assert rejected.accepted_candidates == ()
    assert rejected.proposed_status is StepStatus.FAILED


def test_codex_requires_one_runtime_tool_handle(
    tmp_path, codex_experiment: EnvExperiment
) -> None:
    codex = stack(tmp_path, codex_experiment)
    adapter = CodexAdapter(
        codex.runner,
        store=codex.store,
        tool_store=codex.tool_store,
    )

    with pytest.raises(OpaqueStepError):
        adapter.invoke(request(codex.base, codex.config), ())


def test_codex_accepts_valid_proposals_without_mcp_calls(
    tmp_path, codex_experiment: EnvExperiment
) -> None:
    codex = stack(tmp_path, codex_experiment)
    step_request = request(codex.base, codex.config)
    handle = codex.executor.runtime_handle(
        codex.config,
        codex.tool_store,
        binding(step_request),
    )
    runner = fake_runner(codex.base, scripted_calls=())

    output = CodexAdapter(
        runner,
        store=codex.store,
        tool_store=codex.tool_store,
    ).invoke(step_request, (handle,))

    assert output.proposed_status is StepStatus.COMPLETE
    assert len(output.accepted_candidates) == 2
    assert runner.observed_payloads == []


def test_codex_rejects_an_artifact_from_another_run(
    tmp_path, codex_experiment: EnvExperiment
) -> None:
    codex = stack(tmp_path, codex_experiment)
    step_request = request(codex.base, codex.config)
    handle = codex.executor.runtime_handle(
        codex.config,
        codex.tool_store,
        binding(step_request),
    )
    adapter = CodexAdapter(
        fake_runner(codex.base, artifact_run_id="another-run"),
        store=codex.store,
        tool_store=codex.tool_store,
    )

    with pytest.raises(OpaqueStepError):
        adapter.invoke(step_request, (handle,))


def _invalid_proposals(case: str, base: Candidate) -> tuple[Candidate, ...]:
    valid = proposals(base)
    if case == "wrong_count":
        return valid[:1]
    if case == "unrequested_base":
        return (
            valid[0].model_copy(update={"base_ref": base.base_ref}),
            valid[1],
        )
    if case == "unchanged_mutation":
        return (
            valid[0].model_copy(update={"payload": base.payload}),
            valid[1],
        )
    raise AssertionError(case)


@pytest.mark.parametrize(
    "case",
    ["wrong_count", "unrequested_base", "unchanged_mutation"],
)
def test_codex_rejects_proposal_contract_violations(
    tmp_path, codex_experiment: EnvExperiment, case: str
) -> None:
    codex = stack(tmp_path, codex_experiment, namespace=f"codex-{case}")
    step_request = request(codex.base, codex.config, run_id=f"run-{case}")
    handle = codex.executor.runtime_handle(
        codex.config,
        codex.tool_store,
        binding(step_request),
    )
    adapter = CodexAdapter(
        fake_runner(
            codex.base,
            call_id=f"call-{case}",
            final_proposals=_invalid_proposals(case, codex.base),
        ),
        store=codex.store,
        tool_store=codex.tool_store,
    )

    output = adapter.invoke(step_request, (handle,))

    assert output.proposed_status is StepStatus.FAILED
    assert output.accepted_candidates == ()
    assert output.terminal_failure is not None
