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
from dr_dspy.humaneval import scoring as humaneval_scoring
from dr_dspy.humaneval.code_parsing import (
    BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_ID,
    PARSER_PROFILE_VERSION,
    STRICT_FIELD_MARKER_PARSER_PROFILE,
    STRICT_FIELD_MARKER_PARSER_PROFILE_ID,
    ExtractionMethod,
    extract_best_effort_code,
    extract_strict_field_marker_code,
)
from dr_dspy.humaneval.metrics import (
    HUMANEVAL_METRICS_PROFILE_ID,
    NodeOutputMetricsSource,
    ast_metrics,
    build_metrics_payload,
    python_leakage_metrics,
    task_test_metrics,
    text_metrics,
)
from dr_dspy.humaneval.parsed_tests import HumanEvalTestCaseKind
from dr_dspy.humaneval.profiles import (
    HUMANEVAL_SCORING_PROFILE_ID,
    HUMANEVAL_SCORING_PROFILE_VERSION,
    HumanEvalScoringProfile,
)
from dr_dspy.humaneval.scoring import GeneratedCodeOutcome
from dr_dspy.humaneval.task import EvaluationTaskResult, HumanEvalTask
from dr_dspy.lm.boundary import EndpointKind, ProviderKind
from dr_dspy.platform import rescoring, scoring_workflow
from dr_dspy.platform.persistence import (
    ScoreAttemptInsertResult,
    ScoreAttemptInsertStatus,
    idempotent_insert_score_attempt,
    persist_score_attempt,
)
from dr_dspy.platform.scoring import (
    score_generation_run,
)
from dr_dspy.records import (
    DimensionsPayload,
    FailureMetadataPayload,
    GenerationRunRecord,
    GenerationRunStatus,
    GenerationRunSummaryPayload,
    GenerationTerminalErrorPayload,
    GraphSnapshotPayload,
    MetricsPayload,
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


class DummyConnection:
    pass


class DummyTransaction:
    def __init__(self, engine: DummyEngine) -> None:
        self.engine = engine

    def __enter__(self) -> DummyConnection:
        self.engine.begin_count += 1
        return self.engine.connection

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any,
    ) -> None:
        pass


class DummyEngine:
    def __init__(self) -> None:
        self.connection = DummyConnection()
        self.begin_count = 0

    def begin(self) -> DummyTransaction:
        return DummyTransaction(self)


def _task(*, test: str | None = None) -> HumanEvalTask:
    return HumanEvalTask(
        task_id="HumanEval/fixture",
        prompt="def add_one(x):\n",
        canonical_solution="    return x + 1\n",
        entry_point="add_one",
        test=test or (
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


def _failed_generation_run(spec: PredictionSpecRecord) -> GenerationRunRecord:
    terminal_node_id = spec.graph.graph.terminal_node_id
    return GenerationRunRecord(
        generation_run_id=stable_generation_run_id(
            prediction_id=spec.prediction_id,
            attempt_index=0,
        ),
        prediction_id=spec.prediction_id,
        attempt_index=0,
        status=GenerationRunStatus.ERROR,
        terminal_node_id=terminal_node_id,
        terminal_output_node_id=None,
        summary=GenerationRunSummaryPayload(
            execution_order=(terminal_node_id,),
            terminal_node_id=terminal_node_id,
            terminal_error=GenerationTerminalErrorPayload(
                node_id=terminal_node_id,
                status=GenerationRunStatus.ERROR,
                failure=FailureMetadataPayload(
                    error_type="RuntimeError",
                    message="provider failed",
                ),
            ),
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


def test_best_effort_parser_rejects_code_repr_assignment() -> None:
    result = extract_best_effort_code(
        "code='def add_one(x):\\n    return x + 1\\n'"
    )

    assert result.succeeded is False
    assert (
        result.compile_error
        == "code repr assignments are not valid HumanEval code"
    )


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
    metrics = build_metrics_payload(
        raw_generation="```python\ndef add_one(x):\n    return x + 1\n```",
        extracted_code="def add_one(x):\n    return x + 1",
        task=_task(),
        node_output_sources=(
            NodeOutputMetricsSource(
                node_id="encoder",
                field_name="description",
                text="Use return and add_one carefully.",
            ),
        ),
    )

    assert metrics.profile_id == HUMANEVAL_METRICS_PROFILE_ID
    assert metrics.task_tests is not None
    assert metrics.task_tests.case_count == 2
    assert metrics.task_tests.input_result_case_count == 2
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


def test_task_test_metrics_summarize_input_result_tests() -> None:
    metrics = task_test_metrics(_task())

    assert metrics.parse_ok is True
    assert metrics.task_id == "HumanEval/fixture"
    assert metrics.entry_point == "add_one"
    assert metrics.test_type is HumanEvalTestCaseKind.INPUT_RESULT
    assert metrics.case_count == 2
    assert metrics.input_result_case_count == 2
    assert metrics.oracle_case_count == 0
    assert metrics.input_expression_case_count == 0
    assert metrics.assertion_name == "assertion"
    assert metrics.check_name == "check"
    assert metrics.candidate_arg_name == "candidate"
    assert metrics.input_repr_character_total == len("[1]") + len("[2]")
    assert metrics.expected_output_repr_character_total == len("2") + len("3")
    assert metrics.expected_output_expr_count == 0
    assert metrics.original_test_line_count > 0


def test_task_test_metrics_summarize_oracle_tests() -> None:
    task = _task(
        test=(
            "def ref(x):\n"
            "    return x + 1\n"
            "\n"
            "def check(candidate):\n"
            "    inputs = [(1,), (2,)]\n"
            "    for inp in inputs:\n"
            "        assertion(candidate(*inp), ref(*inp))\n"
        ),
    )

    metrics = task_test_metrics(task)

    assert metrics.parse_ok is True
    assert metrics.test_type is HumanEvalTestCaseKind.INPUT_ORACLE
    assert metrics.case_count == 2
    assert metrics.oracle_case_count == 2
    assert metrics.expected_output_expr_count == 2
    assert metrics.input_result_case_count == 0
    assert metrics.input_expression_case_count == 0
    assert metrics.support_code_character_count > 0


def test_ast_metrics_include_rich_function_and_code_shape() -> None:
    source = (
        "import math\n"
        "from os import path\n"
        "\n"
        "def deco(fn):\n"
        "    return fn\n"
        "\n"
        "@deco\n"
        "def add_one(x, /, y: int = 1, *args, scale=1, **kwargs) -> int:\n"
        "    \"\"\"doc\"\"\"\n"
        "    total = x + y\n"
        "    values = [item for item in args if item]\n"
        "    if total > 0:\n"
        "        for value in values:\n"
        "            total += value\n"
        "    def helper(z):\n"
        "        return scale + z\n"
        "    return helper(total)\n"
        "\n"
        "async def later(a):\n"
        "    return await foo(a)\n"
        "\n"
        "lambda_value = lambda q: q\n"
        "class Box:\n"
        "    pass\n"
    )

    metrics = ast_metrics(source)

    assert metrics.parse_ok is True
    assert metrics.top_level_function_count == 3
    assert metrics.top_level_function_names == ("deco", "add_one", "later")
    assert metrics.function_count == 4
    assert metrics.nested_function_count == 1
    assert metrics.async_function_count == 1
    assert metrics.lambda_count == 1
    assert metrics.class_count == 1
    assert metrics.import_count == 2
    assert metrics.return_count == 4
    assert metrics.call_count >= 2
    assert metrics.assignment_count == 4
    assert metrics.comprehension_count == 1
    assert metrics.literal_count > 0
    assert metrics.max_branch_depth == 2
    assert metrics.total_argument_count == 8
    assert metrics.positional_only_argument_count == 1
    assert metrics.keyword_only_argument_count == 1
    assert metrics.vararg_count == 1
    assert metrics.kwarg_count == 1
    assert metrics.decorated_function_count == 1
    assert metrics.annotated_return_count == 1
    assert metrics.docstring_function_count == 1
    assert metrics.max_function_line_span > 0


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
    assert ast_result.function_count == 1
    assert ast_error.parse_ok is False


def test_score_generation_run_persists_passing_score_attempt() -> None:
    spec = _spec()
    run = _generation_run(spec, "def add_one(x):\n    return x + 1\n")

    score = score_generation_run(
        spec=spec,
        generation_run=run,
        node_attempts=(),
        task=_task(),
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
    assert score.metrics is not None
    assert score.metrics.task_tests is not None
    assert score.metrics.task_tests.case_count == 2
    assert score.metrics.ast is not None
    assert score.metrics.ast.function_count == 1
    assert score.metrics.custom["evaluation"] == {
        "function_names": ["add_one"],
        "total_cases": 2,
        "result_count": 2,
        "passed_count": 2,
        "failed_count": 0,
        "error_count": 0,
        "timeout_count": 0,
        "failure_count": 0,
        "passed": True,
        "status_counts": {"passed": 2},
    }


def test_score_generation_run_defaults_completed_at_after_scoring() -> None:
    spec = _spec()
    run = _generation_run(spec, "def add_one(x):\n    return x + 1\n")

    score = score_generation_run(
        spec=spec,
        generation_run=run,
        node_attempts=(),
        task=_task(),
        started_at=NOW,
    )

    assert score.completed_at > NOW


def test_score_generation_run_persists_tests_failed_as_success() -> None:
    spec = _spec()
    run = _generation_run(spec, "def add_one(x):\n    return x\n")

    score = score_generation_run(
        spec=spec,
        generation_run=run,
        node_attempts=(),
        task=_task(),
        started_at=NOW,
        completed_at=LATER,
    )

    assert score.status is ScoreAttemptStatus.SUCCESS
    assert score.score == 0.0
    assert score.generated_code_outcome is GeneratedCodeOutcome.TESTS_FAILED
    assert score.per_test_results
    assert score.metrics is not None
    assert score.metrics.custom["evaluation"]["failed_count"] == 2
    assert score.metrics.custom["evaluation"]["failure_count"] == 2


def test_score_generation_run_persists_no_top_level_functions() -> None:
    spec = _spec()
    run = _generation_run(spec, "ANSWER = 2\n")

    score = score_generation_run(
        spec=spec,
        generation_run=run,
        node_attempts=(),
        task=_task(),
        started_at=NOW,
        completed_at=LATER,
    )

    assert score.status is ScoreAttemptStatus.SUCCESS
    assert score.score == 0.0
    assert score.generated_code_outcome is (
        GeneratedCodeOutcome.NO_TOP_LEVEL_FUNCTIONS
    )
    assert score.per_test_results == ()
    assert score.metrics is not None
    assert score.metrics.task_tests is not None
    assert score.metrics.task_tests.case_count == 2
    assert score.metrics.ast is not None
    assert score.metrics.ast.top_level_function_count == 0
    assert score.metrics.custom["evaluation"] == {
        "function_names": [],
        "total_cases": 2,
        "result_count": 0,
        "passed_count": 0,
        "failed_count": 0,
        "error_count": 0,
        "timeout_count": 0,
        "failure_count": 0,
        "passed": False,
        "status_counts": {},
    }


def test_metrics_payload_round_trips_through_record_model() -> None:
    metrics = build_metrics_payload(
        raw_generation="def add_one(x):\n    return x + 1\n",
        extracted_code="def add_one(x):\n    return x + 1\n",
        task=_task(),
    )

    round_tripped = MetricsPayload.model_validate(
        metrics.model_dump(mode="json")
    )

    assert round_tripped.task_tests is not None
    assert round_tripped.task_tests.case_count == 2
    assert round_tripped.ast is not None
    assert round_tripped.ast.top_level_function_names == ("add_one",)


def test_metrics_payload_preserves_extracted_code_parse_error() -> None:
    metrics = build_metrics_payload(
        raw_generation="def add_one(x)\n    return x + 1\n",
        extracted_code="def add_one(x)\n    return x + 1\n",
        task=_task(),
    )

    assert metrics.task_tests is not None
    assert metrics.task_tests.case_count == 2
    assert metrics.ast is not None
    assert metrics.ast.parse_ok is False
    assert metrics.ast.parse_error is not None
    extracted_stage = next(
        stage for stage in metrics.stages if stage.stage_id == "extracted_code"
    )
    assert extracted_stage.ast is not None
    assert extracted_stage.ast.parse_ok is False


@pytest.mark.parametrize(
    ("raw_generation", "outcome"),
    [
        ("   ", GeneratedCodeOutcome.EMPTY_GENERATION),
        (
            "def add_one(x)\n    return x",
            GeneratedCodeOutcome.EXTRACTION_FAILED,
        ),
        (["not", "scoreable"], GeneratedCodeOutcome.EXTRACTION_FAILED),
    ],
)
def test_score_generation_run_persists_extraction_failures_as_success(
    raw_generation: Any,
    outcome: GeneratedCodeOutcome,
) -> None:
    spec = _spec()
    run = _generation_run(spec, raw_generation)

    score = score_generation_run(
        spec=spec,
        generation_run=run,
        node_attempts=(),
        task=_task(),
        started_at=NOW,
        completed_at=LATER,
    )

    assert score.status is ScoreAttemptStatus.SUCCESS
    assert score.score == 0.0
    assert score.generated_code_outcome is outcome
    assert score.per_test_results == ()
    assert score.metrics is not None
    assert score.metrics.task_tests is not None
    assert score.metrics.task_tests.case_count == 2
    assert score.metrics.ast is None


def test_score_generation_run_persists_infrastructure_error() -> None:
    spec = _spec()
    other_spec = _spec(layout="encdec")
    run = _generation_run(other_spec, "def add_one(x):\n    return x + 1\n")

    score = score_generation_run(
        spec=spec,
        generation_run=run,
        node_attempts=(),
        task=_task(),
        started_at=NOW,
        completed_at=LATER,
    )

    assert score.status is ScoreAttemptStatus.ERROR
    assert score.score is None
    assert score.metrics is None
    assert score.failure is not None
    assert score.failure.metadata["generation_run_id"] == run.generation_run_id


def test_score_generation_run_scores_encdec_terminal_output() -> None:
    spec = _spec(layout="encdec")
    raw_terminal_output = {"code": "def add_one(x):\n    return x + 1\n"}
    run = _generation_run(
        spec,
        raw_terminal_output,
    )

    score = score_generation_run(
        spec=spec,
        generation_run=run,
        node_attempts=(
            _node_attempt(
                spec,
                node_id="encoder",
                values={
                    "description": "Plain description.",
                    "plan": {"steps": ["read", "write"], "ok": True},
                },
            ),
            _node_attempt(
                spec,
                node_id="decoder",
                values={
                    "code": "def add_one(x):\n    return x + 1\n",
                    "alternatives": ["return x + 1"],
                },
            ),
        ),
        task=_task(),
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
        "node:encoder:plan",
        "node:decoder:alternatives",
        "node:decoder:code",
    }
    stages = {stage.stage_id: stage for stage in score.metrics.stages}
    assert stages["terminal"].text.character_count == len(
        '{"code":"def add_one(x):\\n    return x + 1\\n"}'
    )
    assert stages["node:encoder:plan"].text.character_count == len(
        '{"ok":true,"steps":["read","write"]}'
    )
    assert stages["node:decoder:alternatives"].text.character_count == len(
        '["return x + 1"]'
    )


def test_score_attempt_id_and_insert_are_idempotent_by_profile() -> None:
    spec = _spec()
    run = _generation_run(spec, "def add_one(x):\n    return x + 1\n")
    score = score_generation_run(
        spec=spec,
        generation_run=run,
        node_attempts=(),
        task=_task(),
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


def test_persist_score_attempt_reports_conflict_status() -> None:
    spec = _spec()
    run = _generation_run(spec, "def add_one(x):\n    return x + 1\n")
    score = score_generation_run(
        spec=spec,
        generation_run=run,
        node_attempts=(),
        task=_task(),
        started_at=NOW,
        completed_at=LATER,
    )

    class Result:
        rowcount = 0

    class Connection:
        def execute(self, statement: Any) -> Result:
            return Result()

    result = persist_score_attempt(
        cast(Any, Connection()),
        score_attempt=score,
    )

    assert result.score_attempt_id == score.score_attempt_id
    assert result.status is ScoreAttemptInsertStatus.ALREADY_PRESENT


def test_score_generation_run_persists_failed_generation_as_error() -> None:
    spec = _spec()
    run = _failed_generation_run(spec)

    score = score_generation_run(
        spec=spec,
        generation_run=run,
        node_attempts=(),
        task=_task(),
        started_at=NOW,
        completed_at=LATER,
    )

    assert score.status is ScoreAttemptStatus.ERROR
    assert score.score is None
    assert score.generated_code_outcome is None
    assert score.metrics is None
    assert score.failure is not None
    assert (
        score.failure.message
        == "generation run is not terminal success: error"
    )


def test_load_humaneval_task_step_uses_cached_task_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, Any]] = []
    rows = [{"task_id": "HumanEval/fixture"}]

    def load_rows(
        *,
        dataset_name: str,
        dataset_split: str,
    ) -> list[dict[str, str]]:
        calls.append(("load", (dataset_name, dataset_split)))
        return rows

    def parse_rows(payload: list[dict[str, str]]) -> tuple[HumanEvalTask, ...]:
        calls.append(("parse", payload))
        return (_task(),)

    monkeypatch.setattr(
        scoring_workflow,
        "load_human_eval_rows",
        load_rows,
    )
    monkeypatch.setattr(
        scoring_workflow,
        "parse_human_eval_dataset",
        parse_rows,
    )
    cached_loader = cast(Any, scoring_workflow.load_humaneval_task_map)
    cached_loader.cache_clear()
    try:
        load_step = cast(Any, scoring_workflow.load_humaneval_task_step)
        first = load_step.__wrapped__(
            "dataset",
            "split",
            "HumanEval/fixture",
        )
        second = load_step.__wrapped__(
            "dataset",
            "split",
            "HumanEval/fixture",
        )
    finally:
        cached_loader.cache_clear()

    assert first == second
    assert calls == [
        ("load", ("dataset", "split")),
        ("parse", rows),
    ]


def test_load_humaneval_task_step_raises_for_missing_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def load_rows(
        *,
        dataset_name: str,
        dataset_split: str,
    ) -> list[dict[str, str]]:
        return [{"task_id": "HumanEval/fixture"}]

    def parse_rows(payload: list[dict[str, str]]) -> tuple[HumanEvalTask, ...]:
        return (_task(),)

    monkeypatch.setattr(
        scoring_workflow,
        "load_human_eval_rows",
        load_rows,
    )
    monkeypatch.setattr(
        scoring_workflow,
        "parse_human_eval_dataset",
        parse_rows,
    )
    cached_loader = cast(Any, scoring_workflow.load_humaneval_task_map)
    cached_loader.cache_clear()
    try:
        load_step = cast(Any, scoring_workflow.load_humaneval_task_step)
        with pytest.raises(
            ValueError,
            match="HumanEval task not found: HumanEval/missing",
        ):
            load_step.__wrapped__(
                "dataset",
                "split",
                "HumanEval/missing",
            )
    finally:
        cached_loader.cache_clear()


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

    def score_step(*args: Any) -> dict[str, Any]:
        scoring_profile = HumanEvalScoringProfile.model_validate(args[4])
        calls.append(
            (
                "score",
                (
                    scoring_profile.profile_id,
                    scoring_profile.version,
                    scoring_profile.parser_profile.profile_id,
                    scoring_profile.parser_profile.version,
                    scoring_profile.timeout_seconds,
                    args[5],
                    args[6],
                ),
            )
        )
        return score_generation_run(
            spec=spec,
            generation_run=run,
            node_attempts=(),
            task=_task(),
            scoring_profile=scoring_profile,
            started_at=NOW,
            completed_at=LATER,
        ).model_dump(mode="json")

    def persist(database_url: str, payload: dict[str, Any]) -> dict[str, Any]:
        calls.append(("persist", (database_url, payload["score_attempt_id"])))
        return ScoreAttemptInsertResult(
            score_attempt_id=payload["score_attempt_id"],
            status=ScoreAttemptInsertStatus.ALREADY_PRESENT,
        ).model_dump(mode="json")

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
    assert result == {
        "score_attempt_id": expected_score_id,
        "insert_status": "already_present",
    }
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
        (
            "score",
            (
                HUMANEVAL_SCORING_PROFILE_ID,
                HUMANEVAL_SCORING_PROFILE_VERSION,
                BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_ID,
                PARSER_PROFILE_VERSION,
                2.0,
                0,
                NOW.isoformat(),
            ),
        ),
        ("persist", ("postgresql://example/db", expected_score_id)),
    ]


def test_rescore_selector_filters_and_orders_candidates() -> None:
    statement = db_io.select_rescore_generation_candidates(
        experiment_name="exp",
        generation_status=GenerationRunStatus.SUCCESS,
        generation_attempt_index=0,
        scoring_profile_id=HUMANEVAL_SCORING_PROFILE_ID,
        scoring_profile_version=HUMANEVAL_SCORING_PROFILE_VERSION,
        parser_profile_id=BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_ID,
        parser_version=PARSER_PROFILE_VERSION,
        score_attempt_index=0,
        limit=10,
        offset=2,
    )

    compiled = str(
        statement.compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"literal_binds": True},
        )
    )

    assert "LEFT OUTER JOIN dr_dspy_score_attempts" in compiled
    assert "dr_dspy_prediction_specs.experiment_name = 'exp'" in compiled
    assert "dr_dspy_generation_runs.status = 'success'" in compiled
    assert "dr_dspy_generation_runs.attempt_index = 0" in compiled
    assert (
        "dr_dspy_score_attempts.scoring_profile_id = 'humaneval'"
    ) in compiled
    assert (
        "dr_dspy_score_attempts.parser_profile_id = "
        "'humaneval-best-effort'"
    ) in compiled
    assert (
        "ORDER BY dr_dspy_prediction_specs.fair_order_key, "
        "dr_dspy_prediction_specs.prediction_id, "
        "dr_dspy_generation_runs.generation_run_id"
    ) in compiled
    assert "LIMIT 10 OFFSET 2" in compiled


def test_batch_rescore_dry_run_skips_already_scored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = (
        _rescore_candidate(0, existing_score_attempt_id="existing-score"),
        _rescore_candidate(1),
    )
    scheduler_calls: list[str] = []

    monkeypatch.setattr(
        rescoring,
        "load_rescore_generation_candidates",
        lambda connection, **kwargs: candidates,
    )

    def schedule(
        **kwargs: Any,
    ) -> scoring_workflow.ScheduledScoreGenerationWorkflow:
        scheduler_calls.append(kwargs["generation_run_id"])
        return scoring_workflow.ScheduledScoreGenerationWorkflow(
            score_attempt_id="unused",
            workflow_id="unused",
            scheduled=True,
        )

    result = rescoring.rescore_generation_runs(
        cast(Any, DummyEngine()),
        database_url="postgresql://example/db",
        experiment_name="exp",
        dry_run=True,
        schedule_workflow=schedule,
    )

    assert scheduler_calls == []
    assert result.selected_count == 2
    assert result.already_scored_count == 1
    assert result.pending_score_count == 1
    assert result.scheduled_count == 0
    assert [item.status for item in result.items] == [
        rescoring.BatchRescoreItemStatus.ALREADY_SCORED,
        rescoring.BatchRescoreItemStatus.WOULD_SCHEDULE,
    ]
    assert result.items[0].existing_score_attempt_id == "existing-score"


def test_batch_rescore_chunks_and_counts_scheduler_outcomes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = (
        _rescore_candidate(0),
        _rescore_candidate(1),
        _rescore_candidate(2),
    )
    pages: list[tuple[int, int]] = []
    scheduler_calls: list[str] = []

    def load_candidates(
        connection: DummyConnection,
        **kwargs: Any,
    ) -> tuple[rescoring.RescoreGenerationCandidate, ...]:
        pages.append((kwargs["limit"], kwargs["offset"]))
        return candidates[kwargs["offset"]:kwargs["offset"] + kwargs["limit"]]

    monkeypatch.setattr(
        rescoring,
        "load_rescore_generation_candidates",
        load_candidates,
    )

    def schedule(
        **kwargs: Any,
    ) -> scoring_workflow.ScheduledScoreGenerationWorkflow:
        generation_run_id = kwargs["generation_run_id"]
        scheduler_calls.append(generation_run_id)
        score_attempt_id = stable_score_attempt_id(
            generation_run_id=generation_run_id,
            scoring_profile_id=HUMANEVAL_SCORING_PROFILE_ID,
            scoring_profile_version=HUMANEVAL_SCORING_PROFILE_VERSION,
            parser_profile_id=BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_ID,
            parser_version=PARSER_PROFILE_VERSION,
            attempt_index=0,
        )
        if generation_run_id == "generation-run-1":
            return scoring_workflow.ScheduledScoreGenerationWorkflow(
                score_attempt_id=score_attempt_id,
                workflow_id=f"platform-score-v1:{score_attempt_id}",
                scheduled=False,
            )
        if generation_run_id == "generation-run-2":
            raise RuntimeError("dbos unavailable")
        return scoring_workflow.ScheduledScoreGenerationWorkflow(
            score_attempt_id=score_attempt_id,
            workflow_id=f"platform-score-v1:{score_attempt_id}",
            scheduled=True,
        )

    result = rescoring.rescore_generation_runs(
        cast(Any, DummyEngine()),
        database_url="postgresql://example/db",
        experiment_name="exp",
        chunk_size=2,
        schedule_workflow=schedule,
    )

    assert pages == [(2, 0), (2, 2)]
    assert scheduler_calls == [
        "generation-run-0",
        "generation-run-1",
        "generation-run-2",
    ]
    assert result.selected_count == 3
    assert result.scheduled_count == 1
    assert result.already_scheduled_count == 1
    assert result.failed_count == 1
    assert [item.status for item in result.items] == [
        rescoring.BatchRescoreItemStatus.SCHEDULED,
        rescoring.BatchRescoreItemStatus.WORKFLOW_ALREADY_PRESENT,
        rescoring.BatchRescoreItemStatus.FAILED,
    ]
    assert result.items[2].failure is not None


def test_batch_rescore_limit_caps_selected_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = tuple(_rescore_candidate(index) for index in range(5))
    pages: list[tuple[int, int]] = []

    def load_candidates(
        connection: DummyConnection,
        **kwargs: Any,
    ) -> tuple[rescoring.RescoreGenerationCandidate, ...]:
        pages.append((kwargs["limit"], kwargs["offset"]))
        return candidates[kwargs["offset"]:kwargs["offset"] + kwargs["limit"]]

    monkeypatch.setattr(
        rescoring,
        "load_rescore_generation_candidates",
        load_candidates,
    )

    result = rescoring.rescore_generation_runs(
        cast(Any, DummyEngine()),
        database_url="postgresql://example/db",
        experiment_name="exp",
        chunk_size=2,
        limit=3,
        dry_run=True,
    )

    assert pages == [(2, 0), (1, 2)]
    assert result.selected_count == 3
    assert [item.generation_run_id for item in result.items] == [
        "generation-run-0",
        "generation-run-1",
        "generation-run-2",
    ]


def test_schedule_score_generation_workflow_reports_existing_dbos_workflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec()
    run = _generation_run(spec, "def add_one(x):\n    return x + 1\n")
    expected_score_id = stable_score_attempt_id(
        generation_run_id=run.generation_run_id,
        scoring_profile_id=HUMANEVAL_SCORING_PROFILE_ID,
        scoring_profile_version=HUMANEVAL_SCORING_PROFILE_VERSION,
        parser_profile_id=BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_ID,
        parser_version=PARSER_PROFILE_VERSION,
        attempt_index=0,
    )
    expected_workflow_id = scoring_workflow.platform_scoring_workflow_id(
        expected_score_id
    )
    starts: list[Any] = []

    class FakeDbos:
        def get_workflow_status(self, workflow_id: str) -> dict[str, str]:
            assert workflow_id == expected_workflow_id
            return {"status": "PENDING"}

        def start_workflow(self, *args: Any) -> None:
            starts.append(args)

    monkeypatch.setattr(scoring_workflow, "DBOS", FakeDbos())

    result = scoring_workflow.schedule_score_generation_workflow(
        database_url="postgresql://example/db",
        generation_run_id=run.generation_run_id,
    )

    assert result == scoring_workflow.ScheduledScoreGenerationWorkflow(
        score_attempt_id=expected_score_id,
        workflow_id=expected_workflow_id,
        scheduled=False,
    )
    assert starts == []


def _rescore_candidate(
    index: int,
    *,
    existing_score_attempt_id: str | None = None,
) -> rescoring.RescoreGenerationCandidate:
    return rescoring.RescoreGenerationCandidate(
        prediction_id=f"prediction-{index}",
        fair_order_key=f"{index:04}",
        generation_run_id=f"generation-run-{index}",
        existing_score_attempt_id=existing_score_attempt_id,
    )


def test_scoring_profile_controls_parser_timeout_and_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec()
    run = _generation_run(
        spec,
        "[[ ## code ## ]]\ndef add_one(x):\n    return x + 1\n",
    )
    observed_timeouts: list[float] = []
    scoring_profile = HumanEvalScoringProfile(
        profile_id="humaneval-field-marker",
        version="v1",
        parser_profile=STRICT_FIELD_MARKER_PARSER_PROFILE,
        timeout_seconds=0.25,
        metrics_profile_id="humaneval-metrics-field-marker",
        metrics_profile_version="v1",
    )

    def evaluate(
        *,
        task: HumanEvalTask,
        candidate_code: str,
        timeout_seconds: float,
    ) -> EvaluationTaskResult:
        assert candidate_code == "def add_one(x):\n    return x + 1"
        observed_timeouts.append(timeout_seconds)
        return EvaluationTaskResult(
            task_id=task.task_id,
            entry_point=task.entry_point,
            function_names=[task.entry_point],
            total_cases=0,
            results=[],
        )

    monkeypatch.setattr(
        humaneval_scoring,
        "evaluate_human_eval_code",
        evaluate,
    )

    score = score_generation_run(
        spec=spec,
        generation_run=run,
        node_attempts=(),
        task=_task(),
        scoring_profile=scoring_profile,
        started_at=NOW,
        completed_at=LATER,
    )

    assert score.status is ScoreAttemptStatus.SUCCESS
    assert score.score == 1.0
    assert score.scoring_profile_id == "humaneval-field-marker"
    assert score.parser_profile_id == STRICT_FIELD_MARKER_PARSER_PROFILE_ID
    assert score.metrics is not None
    assert score.metrics.profile_id == "humaneval-metrics-field-marker"
    assert observed_timeouts == [0.25]
