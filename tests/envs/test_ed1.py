"""Focused ED1 environment-contract tests with no orchestration dependency."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest
from dr_providers import (
    FailureClass,
    ProviderCallRequest,
    ProviderInvocationEvidence,
    ProviderTransportFailure,
    ProviderTransportPolicy,
    RawHttpRequest,
)

from tests.envs.support import (
    FakeTransport,
    _prompt_of,
    _response,
    execution_policy,
    synthetic_ed1_tasks,
    transport_policy,
)
from whetstone.envs.ed1 import (
    ED1_CANONICAL_MODEL,
    ED1_DATASET_REVISION,
    ED1_ENV_NAME,
    ED1_INVALID_BODY,
    ENCODER_BODY_A,
    Ed1BodyError,
    build_ed1_experiment,
    ed1_body_rejection,
    ed1_initial_candidate,
    render_encoder_frame,
)
from whetstone.envs.ed1_eval import run_ed1_eval
from whetstone.envs.ed1_scoring import CodeScore, score_ed1_submission
from whetstone.envs.encdec_rollout import (
    DECODER_NODE_ID,
    ENCODER_NODE_ID,
    EVAL_NODE_ID,
    build_encdec_rollout_definition,
    encdec_graph_definition,
)
from whetstone.envs.sampling import Completeness
from whetstone.execution.partials import PartialLog
from whetstone.execution.prompt_cache import PromptResultCache
from whetstone.optimization.mutation import MUTATION_FIELD


def _tasks(limit: int = 3):
    return synthetic_ed1_tasks(limit)


def _passing_scorer(**_kwargs: object) -> CodeScore:
    return CodeScore(
        passed=True,
        infrastructure_unknown=False,
        outcome="passed",
    )


def _encdec_transport(tasks) -> FakeTransport:
    entries = tuple(task.humaneval_task.entry_point for task in tasks)

    def reply(prompt: str) -> str:
        if prompt.startswith(("Provide", "Compress")):
            for entry in entries:
                if f"def {entry}(" in prompt:
                    return f"REBUILD:{entry}"
            return "REBUILD:unknown"
        if prompt.startswith("Decode"):
            return "def rebuilt():\n    return 1\n"
        return ""

    return FakeTransport(reply=reply)


def _evaluate(
    *,
    tasks=None,
    transport=None,
    repeats: int = 1,
    completeness: Completeness = Completeness.PROPAGATE,
    max_skip_fraction: float = 0.0,
    scorer=_passing_scorer,
    cache: PromptResultCache | None = None,
    partial_log: PartialLog | None = None,
):
    selected = tasks or _tasks()
    experiment = build_ed1_experiment(
        tasks=selected,
        internal_n=len(selected),
        official_n=len(selected),
        repeats=repeats,
        completeness=completeness,
        max_skip_fraction=max_skip_fraction,
    )
    candidate = ed1_initial_candidate()
    result = run_ed1_eval(
        experiment,
        candidate_template=str(candidate.payload[MUTATION_FIELD]),
        candidate_id=candidate.candidate_id,
        sampling=experiment.eval_configs.internal,
        execution_policy=execution_policy(max_attempts=1),
        transport=transport or _encdec_transport(selected),
        scorer=scorer,
        apply_reward=False,
        cache=cache,
        partial_log=partial_log,
    )
    return experiment, result


def test_encdec_graph_and_output_affecting_identity() -> None:
    definition = encdec_graph_definition()
    assert [node.node_id for node in definition.nodes] == [
        ENCODER_NODE_ID,
        DECODER_NODE_ID,
        EVAL_NODE_ID,
    ]
    assert definition.terminal_node_id == EVAL_NODE_ID
    base = build_encdec_rollout_definition(
        ED1_ENV_NAME,
        model=ED1_CANONICAL_MODEL,
        procedure_config_hash="a" * 64,
        budget_ratio=0.5,
    )
    ratio = build_encdec_rollout_definition(
        ED1_ENV_NAME,
        model=ED1_CANONICAL_MODEL,
        procedure_config_hash="a" * 64,
        budget_ratio=0.75,
    )
    model = build_encdec_rollout_definition(
        ED1_ENV_NAME,
        model="openai/gpt-5-nano",
        procedure_config_hash="a" * 64,
        budget_ratio=0.5,
    )
    assert base.graph_hash != ratio.graph_hash != model.graph_hash
    assert base.provider_call_config.definition.route.model == (
        ED1_CANONICAL_MODEL
    )


def test_humaneval_scoring_canonical_passes_wrong_fails() -> None:
    task = _tasks(1)[0].humaneval_task
    good = score_ed1_submission(
        raw_submission=task.ground_truth_code,
        task=task,
        timeout_seconds=30.0,
    )
    bad = score_ed1_submission(
        raw_submission=(
            f"def {task.entry_point}(*args, **kwargs):\n    return None\n"
        ),
        task=task,
        timeout_seconds=30.0,
    )
    assert good.passed and not good.infrastructure_unknown
    assert not bad.passed and not bad.infrastructure_unknown


def test_body_validation_rejects_before_transport() -> None:
    assert ed1_body_rejection("Solve {input_code}") == ("{input_code}",)
    assert ed1_body_rejection("```python\npass\n```") == ("```",)
    assert ed1_body_rejection("Solve carefully.") == ()
    tasks = _tasks(1)
    experiment = build_ed1_experiment(tasks=tasks)
    transport = FakeTransport(reply=lambda _prompt: "unused")

    with pytest.raises(Ed1BodyError) as error:
        run_ed1_eval(
            experiment,
            candidate_template="Solve {input_code}",
            candidate_id="invalid-body",
            sampling=experiment.eval_configs.internal,
            execution_policy=execution_policy(max_attempts=1),
            transport=transport,
            scorer=_passing_scorer,
            apply_reward=False,
        )

    assert error.value.code == ED1_INVALID_BODY
    assert error.value.offending == ("{input_code}",)
    assert transport.served == []


def test_no_budget_frame_omits_budget_instruction() -> None:
    rendered = render_encoder_frame(
        ENCODER_BODY_A,
        input_code="def f(): pass",
        max_budget=None,
    )
    assert "Use at most" not in rendered
    assert "```python\ndef f(): pass\n```" in rendered


def test_end_to_end_records_exact_dual_scores_and_outputs() -> None:
    experiment, result = _evaluate(repeats=2)
    assert result.pass_aggregate.aggregation_output.value == pytest.approx(1)
    compression = result.compression_aggregate.aggregation_output.value
    assert compression is not None and compression > 0
    assert result.pass_aggregate.eval_config_hash == (
        experiment.eval_configs.internal.eval_config.config_identity_hash
    )
    assert result.pass_aggregate.repeat_count == 2
    assert len(result.outputs) == len(_tasks()) * 2
    assert all("ENCODER:" in (row.output_text or "") for row in result.outputs)
    assert experiment.dataset_revision == ED1_DATASET_REVISION


def test_budget_and_healthy_diagnostics_are_explicit() -> None:
    tasks = _tasks(1)
    long_description = "x" * (len(tasks[0].input_code) * 4)

    def reply(prompt: str) -> str:
        if prompt.startswith(("Provide", "Compress")):
            return long_description
        return "def rebuilt():\n    return 1\n"

    _, result = _evaluate(
        tasks=tasks,
        transport=FakeTransport(reply=reply),
    )
    row = result.row_diags[0]
    assert row.max_budget == round(0.5 * len(tasks[0].input_code))
    assert row.encoder_len == len(long_description)
    assert row.over_budget is True
    assert row.failed is False
    assert result.diagnostics.present_rows == 1
    assert result.diagnostics.failed_rows == 0
    assert result.diagnostics.none_reason is None


def test_all_failed_diagnostics_name_dominant_failure() -> None:
    def failed(**_kwargs: object) -> CodeScore:
        return CodeScore(
            passed=False,
            infrastructure_unknown=True,
            outcome="harness_failure",
        )

    _, result = _evaluate(scorer=failed)
    assert result.pass_aggregate.aggregation_output.value is None
    assert result.diagnostics.present_rows == 0
    assert result.diagnostics.failed_rows == 3
    assert result.diagnostics.none_reason is not None
    assert "code_eval_infrastructure_unknown" in (
        result.diagnostics.none_reason
    )


def test_bounded_skip_certifies_retained_scores_and_accounting() -> None:
    tasks = _tasks(4)
    calls = 0

    def scorer(**_kwargs: object) -> CodeScore:
        nonlocal calls
        calls += 1
        return CodeScore(
            passed=calls != 1,
            infrastructure_unknown=calls == 1,
            outcome="timed_out" if calls == 1 else "passed",
        )

    _, result = _evaluate(
        tasks=tasks,
        completeness=Completeness.SKIP,
        max_skip_fraction=0.30,
        scorer=scorer,
    )
    assert result.pass_aggregate.rows_failed == 1
    assert result.pass_aggregate.rows_present == 3
    assert result.pass_aggregate.aggregation_output.value == pytest.approx(1)
    assert result.per_task_counts == (1, 1, 1, 1)


def test_streaming_resume_restores_rows_without_transport(
    tmp_path: Path,
) -> None:
    tasks = _tasks(2)
    log = PartialLog(path=tmp_path / "ed1.partial.jsonl")
    experiment, first = _evaluate(tasks=tasks, partial_log=log)
    assert len(log.load()) == 2

    def boom(_prompt: str) -> str:
        raise AssertionError("recorded rows must not be called again")

    candidate = ed1_initial_candidate()
    resumed = run_ed1_eval(
        experiment,
        candidate_template=str(candidate.payload[MUTATION_FIELD]),
        candidate_id=candidate.candidate_id,
        sampling=experiment.eval_configs.internal,
        execution_policy=execution_policy(max_attempts=1),
        transport=FakeTransport(reply=boom),
        scorer=_passing_scorer,
        apply_reward=False,
        partial_log=log,
    )
    assert resumed.pass_aggregate == first.pass_aggregate
    assert resumed.compression_aggregate == first.compression_aggregate


def test_prompt_cache_reuses_both_calls_with_provenance(
    tmp_path: Path,
) -> None:
    tasks = _tasks(1)
    cache = PromptResultCache(root=tmp_path / "cache")
    experiment, first = _evaluate(tasks=tasks, cache=cache)

    def boom(_prompt: str) -> str:
        raise AssertionError("cache hit must not invoke transport")

    candidate = ed1_initial_candidate()
    second = run_ed1_eval(
        experiment,
        candidate_template=str(candidate.payload[MUTATION_FIELD]),
        candidate_id=candidate.candidate_id,
        sampling=experiment.eval_configs.internal,
        execution_policy=execution_policy(max_attempts=1),
        transport=FakeTransport(reply=boom),
        scorer=_passing_scorer,
        apply_reward=False,
        cache=cache,
    )
    assert second.pass_aggregate == first.pass_aggregate
    assert cache.counters()["hits"] == 2


@dataclass
class _TransientEncoderOnce:
    policy: ProviderTransportPolicy = field(default_factory=transport_policy)
    seen: set[str] = field(default_factory=set)
    failures: int = 0

    def __call__(
        self,
        request: ProviderCallRequest,
    ) -> ProviderInvocationEvidence:
        prompt = _prompt_of(request)
        raw = RawHttpRequest.build(
            url="https://example.test/v1/chat/completions",
            headers={"content-type": "json"},
            body={"model": "test-model"},
        )
        if (
            prompt.startswith(("Provide", "Compress"))
            and prompt not in self.seen
        ):
            self.seen.add(prompt)
            self.failures += 1
            return ProviderInvocationEvidence.build(
                request=request,
                policy=self.policy,
                raw_request=raw,
                outcome=ProviderTransportFailure(
                    failure_class=FailureClass.TRANSIENT,
                    code="transport_error",
                    message="connection reset",
                    retryable=True,
                ),
            )
        text = (
            "REBUILD:ok"
            if prompt.startswith(("Provide", "Compress"))
            else "def rebuilt():\n    return 1\n"
        )
        return ProviderInvocationEvidence.build(
            request=request,
            policy=self.policy,
            raw_request=raw,
            outcome=_response(text),
        )


def test_transient_encoder_failure_is_redriven_to_success() -> None:
    transport = _TransientEncoderOnce()
    _, result = _evaluate(transport=transport)
    assert transport.failures == 3
    assert result.pass_aggregate.rows_failed == 0
    assert result.pass_aggregate.aggregation_output.value == pytest.approx(1)
