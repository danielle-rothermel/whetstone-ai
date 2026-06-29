from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import pytest
from sqlalchemy.dialects import postgresql

from dr_dspy.db import io as db_io
from dr_dspy.graph import (
    BindingRef,
    FieldRole,
    FieldSpec,
    GraphSpec,
    NodeConfig,
    NodeSpec,
    graph_digest,
)
from dr_dspy.humaneval.scoring import GeneratedCodeOutcome
from dr_dspy.humaneval.task import HumanEvalTask
from dr_dspy.lm.boundary import EndpointKind, ProviderKind
from dr_dspy.platform import scoring_workflow
from dr_dspy.platform.code_parsing import (
    BEST_EFFORT_HUMANEVAL_PARSER_PROFILE,
    BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_ID,
    PARSER_PROFILE_VERSION,
    STRICT_FIELD_MARKER_PARSER_PROFILE,
    STRICT_FIELD_MARKER_PARSER_PROFILE_ID,
    ExtractionMethod,
    extract_best_effort_code,
    extract_strict_field_marker_code,
)
from dr_dspy.platform.metrics import (
    HUMANEVAL_METRICS_PROFILE_ID,
    ast_metrics,
    build_metrics_payload,
    python_leakage_metrics,
    text_metrics,
)
from dr_dspy.platform.persistence import idempotent_insert_score_attempt
from dr_dspy.platform.scoring import (
    HUMANEVAL_SCORING_PROFILE_ID,
    HUMANEVAL_SCORING_PROFILE_VERSION,
    score_generation_run,
)
from dr_dspy.records import (
    DimensionsPayload,
    GenerationRunRecord,
    GenerationRunStatus,
    GenerationRunSummaryPayload,
    GraphSnapshotPayload,
    NodeAttemptRecord,
    NodeAttemptStatus,
    NodeOutputPayload,
    PredictionSpecRecord,
    ProviderConfigRef,
    ScoreAttemptStatus,
    TaskInputsPayload,
    TaskSnapshotPayload,
    dimensions_digest,
    fair_order_key,
    stable_generation_run_id,
    stable_prediction_id,
    stable_score_attempt_id,
)

NOW = datetime(2026, 6, 29, 12, 0, tzinfo=UTC)
LATER = NOW + timedelta(seconds=1)


def _task() -> HumanEvalTask:
    return HumanEvalTask(
        task_id="HumanEval/fixture",
        prompt="def add_one(x):\n",
        canonical_solution="    return x + 1\n",
        entry_point="add_one",
        test=(
            "def check(candidate):\n"
            "    inputs = [(1,), (2,)]\n"
            "    results = [2, 3]\n"
            "    for inp, expected in zip(inputs, results):\n"
            "        assertion(candidate(*inp), expected)\n"
        ),
    )


def _node(
    node_id: str,
    *,
    bindings: dict[str, str] | None = None,
    output_field: str = "output",
) -> NodeSpec:
    input_bindings = {
        name: BindingRef.model_validate(ref)
        for name, ref in (bindings or {}).items()
    }
    fields = [
        FieldSpec(name=name, role=FieldRole.INPUT)
        for name in input_bindings
    ]
    fields.append(FieldSpec(name=output_field, role=FieldRole.OUTPUT))
    return NodeSpec(
        id=node_id,
        config=NodeConfig(
            fields=tuple(fields),
            input_bindings=input_bindings,
            output_field=output_field,
        ),
    )


def _graph(layout: str = "direct") -> GraphSpec:
    if layout == "encdec":
        return GraphSpec(
            nodes=(
                _node(
                    "encoder",
                    bindings={"prompt": "task.prompt"},
                    output_field="description",
                ),
                _node(
                    "decoder",
                    bindings={"description": "encoder.description"},
                    output_field="code",
                ),
            ),
            terminal_node_id="decoder",
        )
    return GraphSpec(
        nodes=(_node("direct", bindings={"prompt": "task.prompt"}),),
        terminal_node_id="direct",
    )


def _provider() -> ProviderConfigRef:
    return ProviderConfigRef(
        provider_kind=ProviderKind.OPENAI,
        endpoint_kind=EndpointKind.RESPONSES,
        model="gpt-test",
        throttle_key="openai:responses:gpt-test",
    )


def _spec(layout: str = "direct") -> PredictionSpecRecord:
    graph = _graph(layout)
    graph_id = graph_digest(graph)
    dimensions = DimensionsPayload(values={"temperature": 0.2})
    dimensions_id = dimensions_digest(dimensions)
    prediction_id = stable_prediction_id(
        experiment_name="exp",
        task_id="HumanEval/fixture",
        graph_digest=graph_id,
        dimensions_digest=dimensions_id,
        repetition_seed=0,
    )
    provider = _provider()
    return PredictionSpecRecord(
        prediction_id=prediction_id,
        experiment_name="exp",
        task_id="HumanEval/fixture",
        repetition_seed=0,
        graph=GraphSnapshotPayload(
            graph=graph,
            graph_digest=graph_id,
            layout=layout,
        ),
        dimensions=dimensions,
        dimensions_digest=dimensions_id,
        task=TaskSnapshotPayload(
            task_id="HumanEval/fixture",
            inputs=TaskInputsPayload(values={"prompt": "write add"}),
        ),
        provider_configs=(provider,),
        provider_axis=provider,
        fair_order_seed="seed",
        fair_order_key=fair_order_key(
            experiment_seed="seed",
            prediction_id=prediction_id,
            provider=provider.provider_kind.value,
            endpoint_kind=provider.endpoint_kind.value,
            model=provider.model,
            throttle_key=provider.throttle_key,
            graph_layout=layout,
            task_id="HumanEval/fixture",
            repetition_seed=0,
            config_axis=dimensions_id,
        ),
        created_at=NOW,
    )


def _generation_run(
    spec: PredictionSpecRecord,
    raw_generation: Any,
) -> GenerationRunRecord:
    return GenerationRunRecord(
        generation_run_id=stable_generation_run_id(
            prediction_id=spec.prediction_id,
            attempt_index=0,
        ),
        prediction_id=spec.prediction_id,
        attempt_index=0,
        status=GenerationRunStatus.SUCCESS,
        terminal_node_id=spec.graph.graph.terminal_node_id,
        terminal_output_node_id=spec.graph.graph.terminal_node_id,
        summary=GenerationRunSummaryPayload(
            execution_order=tuple(node.id for node in spec.graph.graph.nodes),
            terminal_node_id=spec.graph.graph.terminal_node_id,
            terminal_output=raw_generation,
        ),
        started_at=NOW,
        completed_at=LATER,
    )


def _node_attempt(
    spec: PredictionSpecRecord,
    *,
    node_id: str,
    values: Mapping[str, Any],
) -> NodeAttemptRecord:
    return NodeAttemptRecord(
        node_attempt_id=f"node-{node_id}",
        generation_run_id=stable_generation_run_id(
            prediction_id=spec.prediction_id,
            attempt_index=0,
        ),
        prediction_id=spec.prediction_id,
        node_id=node_id,
        attempt_index=0,
        status=NodeAttemptStatus.SUCCESS,
        output=NodeOutputPayload(values=dict(values)),
        started_at=NOW,
        completed_at=LATER,
    )


def test_best_effort_parser_unwraps_json_code_and_cleans_fence() -> None:
    result = extract_best_effort_code(
        '{"code": "```python\\ndef add_one(x):\\n    return x + 1\\n```"}'
    )

    assert result.extracted_code == "def add_one(x):\n    return x + 1"
    assert result.extraction_method is ExtractionMethod.JSON_CODE_FIELD
    assert result.selected_candidate_index == 0


def test_best_effort_parser_unwraps_code_like_object_without_repr() -> None:
    class CodeValue:
        code = "def add_one(x):\n    return x + 1\n"

        def __str__(self) -> str:
            return "Code(code='bad repr')"

    class Prediction:
        code = CodeValue()

    result = extract_best_effort_code(Prediction())

    assert result.extracted_code == "def add_one(x):\n    return x + 1"
    assert result.extraction_method is ExtractionMethod.DSPY_CODE_FIELD


@pytest.mark.parametrize("raw_generation", ["{'code': 'bad'}", "[1, 2, 3]"])
def test_best_effort_parser_rejects_plain_literals(
    raw_generation: str,
) -> None:
    result = extract_best_effort_code(raw_generation)

    assert result.succeeded is False
    assert result.extraction_error is not None


def test_strict_parser_only_accepts_field_marker_format() -> None:
    good = extract_strict_field_marker_code(
        "[[ ## code ## ]]\ndef add_one(x):\n    return x + 1\n",
    )
    json_result = extract_strict_field_marker_code(
        '{"code": "def add_one(x): return x + 1"}',
    )
    bare_result = extract_strict_field_marker_code(
        "def add_one(x):\n    return x + 1\n",
    )

    assert good.succeeded is True
    assert good.extraction_method is ExtractionMethod.FIELD_MARKER
    assert json_result.succeeded is False
    assert bare_result.succeeded is False


def test_metrics_payload_includes_full_stage_metrics() -> None:
    spec = _spec(layout="encdec")
    metrics = build_metrics_payload(
        raw_generation="```python\ndef add_one(x):\n    return x + 1\n```",
        extracted_code="def add_one(x):\n    return x + 1",
        task=_task(),
        node_attempts=(
            _node_attempt(
                spec,
                node_id="encoder",
                values={"description": "Use return and add_one carefully."},
            ),
        ),
    )

    assert metrics.profile_id == HUMANEVAL_METRICS_PROFILE_ID
    assert metrics.text is not None
    assert metrics.text.line_count == 4
    assert metrics.python_leakage is not None
    assert metrics.python_leakage.fenced_code_block_count == 1
    assert metrics.ast is not None
    assert metrics.ast.top_level_function_count == 1
    assert "raw" in metrics.compression
    assert [stage.stage_id for stage in metrics.stages] == [
        "terminal",
        "extracted_code",
        "node:encoder:description",
    ]


def test_metric_primitives_are_deterministic() -> None:
    text = text_metrics("def add_one(x):\n    return x + 1\n")
    leakage = python_leakage_metrics(
        "Describe add_one with def and return.",
        task_names=("add_one",),
    )
    ast_result = ast_metrics("def add_one(x):\n    return x + 1\n")
    ast_error = ast_metrics("def add_one(x)\n    return x")

    assert text.word_count == 6
    assert text.punctuation_count == 5
    assert leakage.code_marker_count == 2
    assert leakage.task_name_hit_count == 1
    assert ast_result.parse_ok is True
    assert ast_error.parse_ok is False


def test_score_generation_run_persists_passing_score_attempt() -> None:
    spec = _spec()
    run = _generation_run(spec, "def add_one(x):\n    return x + 1\n")

    score = score_generation_run(
        spec=spec,
        generation_run=run,
        node_attempts=(),
        task=_task(),
        parser_profile=BEST_EFFORT_HUMANEVAL_PARSER_PROFILE,
        started_at=NOW,
        completed_at=LATER,
    )

    assert score.status is ScoreAttemptStatus.SUCCESS
    assert score.score == 1.0
    assert score.generated_code_outcome is GeneratedCodeOutcome.PASSED
    assert score.extracted_code is not None
    assert score.extracted_code.extraction_method == "bare_python"
    assert [result.status for result in score.per_test_results] == [
        "passed",
        "passed",
    ]


def test_score_generation_run_persists_tests_failed_as_success() -> None:
    spec = _spec()
    run = _generation_run(spec, "def add_one(x):\n    return x\n")

    score = score_generation_run(
        spec=spec,
        generation_run=run,
        node_attempts=(),
        task=_task(),
        parser_profile=BEST_EFFORT_HUMANEVAL_PARSER_PROFILE,
        started_at=NOW,
        completed_at=LATER,
    )

    assert score.status is ScoreAttemptStatus.SUCCESS
    assert score.score == 0.0
    assert score.generated_code_outcome is GeneratedCodeOutcome.TESTS_FAILED
    assert score.per_test_results


@pytest.mark.parametrize(
    ("raw_generation", "outcome"),
    [
        ("   ", GeneratedCodeOutcome.EMPTY_GENERATION),
        (
            "def add_one(x)\n    return x",
            GeneratedCodeOutcome.EXTRACTION_FAILED,
        ),
    ],
)
def test_score_generation_run_persists_extraction_failures_as_success(
    raw_generation: str,
    outcome: GeneratedCodeOutcome,
) -> None:
    spec = _spec()
    run = _generation_run(spec, raw_generation)

    score = score_generation_run(
        spec=spec,
        generation_run=run,
        node_attempts=(),
        task=_task(),
        parser_profile=BEST_EFFORT_HUMANEVAL_PARSER_PROFILE,
        started_at=NOW,
        completed_at=LATER,
    )

    assert score.status is ScoreAttemptStatus.SUCCESS
    assert score.score == 0.0
    assert score.generated_code_outcome is outcome
    assert score.per_test_results == ()


def test_score_generation_run_persists_infrastructure_error() -> None:
    spec = _spec()
    run = _generation_run(spec, ["not", "scoreable"])

    score = score_generation_run(
        spec=spec,
        generation_run=run,
        node_attempts=(),
        task=_task(),
        parser_profile=BEST_EFFORT_HUMANEVAL_PARSER_PROFILE,
        started_at=NOW,
        completed_at=LATER,
    )

    assert score.status is ScoreAttemptStatus.ERROR
    assert score.score is None
    assert score.failure is not None
    assert score.failure.metadata["generation_run_id"] == run.generation_run_id


def test_score_generation_run_scores_encdec_terminal_output() -> None:
    spec = _spec(layout="encdec")
    run = _generation_run(
        spec,
        {"code": "def add_one(x):\n    return x + 1\n"},
    )

    score = score_generation_run(
        spec=spec,
        generation_run=run,
        node_attempts=(
            _node_attempt(
                spec,
                node_id="encoder",
                values={"description": "Plain description."},
            ),
            _node_attempt(
                spec,
                node_id="decoder",
                values={"code": "def add_one(x):\n    return x + 1\n"},
            ),
        ),
        task=_task(),
        parser_profile=BEST_EFFORT_HUMANEVAL_PARSER_PROFILE,
        started_at=NOW,
        completed_at=LATER,
    )

    assert score.status is ScoreAttemptStatus.SUCCESS
    assert score.score == 1.0
    assert score.extracted_code is not None
    assert score.extracted_code.extraction_method == "json_code_field"
    assert score.metrics is not None
    assert {stage.stage_id for stage in score.metrics.stages} >= {
        "node:encoder:description",
        "node:decoder:code",
    }


def test_score_attempt_id_and_insert_are_idempotent_by_profile() -> None:
    spec = _spec()
    run = _generation_run(spec, "def add_one(x):\n    return x + 1\n")
    score = score_generation_run(
        spec=spec,
        generation_run=run,
        node_attempts=(),
        task=_task(),
        parser_profile=BEST_EFFORT_HUMANEVAL_PARSER_PROFILE,
        started_at=NOW,
        completed_at=LATER,
    )

    assert score.score_attempt_id == stable_score_attempt_id(
        generation_run_id=run.generation_run_id,
        scoring_profile_id=HUMANEVAL_SCORING_PROFILE_ID,
        scoring_profile_version=HUMANEVAL_SCORING_PROFILE_VERSION,
        parser_profile_id=BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_ID,
        parser_version=PARSER_PROFILE_VERSION,
        attempt_index=0,
    )
    statement = idempotent_insert_score_attempt(score)
    compiled = str(statement.compile(dialect=postgresql.dialect()))

    assert "ON CONFLICT (score_attempt_id) DO NOTHING" in compiled
    row = db_io.score_attempt_row(score)
    assert row["score_attempt_id"] == score.score_attempt_id
    assert row["metrics"]["stages"]


def test_scoring_workflow_uses_dbos_step_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec()
    run = _generation_run(spec, "def add_one(x):\n    return x + 1\n")
    calls: list[tuple[str, Any]] = []

    def load_target(
        database_url: str,
        generation_run_id: str,
    ) -> dict[str, Any]:
        calls.append(("load", (database_url, generation_run_id)))
        return {
            "spec": spec.model_dump(mode="json"),
            "generation_run": run.model_dump(mode="json"),
            "node_attempts": [],
        }

    def load_task(
        dataset_name: str,
        dataset_split: str,
        task_id: str,
    ) -> dict[str, Any]:
        calls.append(("task", (dataset_name, dataset_split, task_id)))
        return _task().model_dump(mode="json")

    def started(score_attempt_id: str) -> str:
        calls.append(("started", score_attempt_id))
        return NOW.isoformat()

    def completed(score_attempt_id: str) -> str:
        calls.append(("completed", score_attempt_id))
        return LATER.isoformat()

    def score_step(*args: Any) -> dict[str, Any]:
        calls.append(("score", args[4:9]))
        return score_generation_run(
            spec=spec,
            generation_run=run,
            node_attempts=(),
            task=_task(),
            parser_profile=BEST_EFFORT_HUMANEVAL_PARSER_PROFILE,
            started_at=NOW,
            completed_at=LATER,
        ).model_dump(mode="json")

    def persist(database_url: str, payload: dict[str, Any]) -> None:
        calls.append(("persist", (database_url, payload["score_attempt_id"])))

    monkeypatch.setattr(
        scoring_workflow,
        "load_scoring_target_step",
        load_target,
    )
    monkeypatch.setattr(
        scoring_workflow,
        "load_humaneval_task_step",
        load_task,
    )
    monkeypatch.setattr(
        scoring_workflow,
        "scoring_started_at_step",
        started,
    )
    monkeypatch.setattr(
        scoring_workflow,
        "scoring_completed_at_step",
        completed,
    )
    monkeypatch.setattr(scoring_workflow, "score_generation_step", score_step)
    monkeypatch.setattr(
        scoring_workflow,
        "persist_score_attempt_step",
        persist,
    )

    workflow = cast(Any, scoring_workflow.run_score_generation_workflow)
    result = workflow.__wrapped__(
        "postgresql://example/db",
        run.generation_run_id,
    )

    expected_score_id = stable_score_attempt_id(
        generation_run_id=run.generation_run_id,
        scoring_profile_id=HUMANEVAL_SCORING_PROFILE_ID,
        scoring_profile_version=HUMANEVAL_SCORING_PROFILE_VERSION,
        parser_profile_id=BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_ID,
        parser_version=PARSER_PROFILE_VERSION,
        attempt_index=0,
    )
    assert result == expected_score_id
    assert calls == [
        ("load", ("postgresql://example/db", run.generation_run_id)),
        (
            "task",
            (
                scoring_workflow.DEFAULT_HUMANEVAL_DATASET_NAME,
                scoring_workflow.DEFAULT_HUMANEVAL_DATASET_SPLIT,
                spec.task_id,
            ),
        ),
        ("started", expected_score_id),
        ("completed", expected_score_id),
        (
            "score",
            (
                HUMANEVAL_SCORING_PROFILE_ID,
                HUMANEVAL_SCORING_PROFILE_VERSION,
                BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_ID,
                PARSER_PROFILE_VERSION,
                0,
            ),
        ),
        ("persist", ("postgresql://example/db", expected_score_id)),
    ]


def test_strict_profile_can_be_used_for_scoring() -> None:
    spec = _spec()
    run = _generation_run(
        spec,
        "[[ ## code ## ]]\ndef add_one(x):\n    return x + 1\n",
    )

    score = score_generation_run(
        spec=spec,
        generation_run=run,
        node_attempts=(),
        task=_task(),
        parser_profile=STRICT_FIELD_MARKER_PARSER_PROFILE,
        started_at=NOW,
        completed_at=LATER,
    )

    assert score.status is ScoreAttemptStatus.SUCCESS
    assert score.score == 1.0
    assert score.parser_profile_id == STRICT_FIELD_MARKER_PARSER_PROFILE_ID
