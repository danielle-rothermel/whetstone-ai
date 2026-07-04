from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import pytest
from dbos._error import DBOSWorkflowConflictIDError
from sqlalchemy.dialects import postgresql

from whetstone.db import io as db_io
from whetstone.graph import (
    BindingRef,
    FieldRole,
    FieldSpec,
    GraphSpec,
    NodeConfig,
    NodeSpec,
    graph_digest,
)
from whetstone.humaneval import scoring as humaneval_scoring
from whetstone.humaneval.code_parsing import (
    BEST_EFFORT_HUMANEVAL_PARSER_PROFILE,
    BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_ID,
    PARSER_PROFILE_VERSION,
    STRICT_FIELD_MARKER_PARSER_PROFILE,
    STRICT_FIELD_MARKER_PARSER_PROFILE_ID,
    ExtractionMethod,
    extract_best_effort_code,
    extract_code_with_profile,
    extract_strict_field_marker_code,
    resolve_parser_profile,
)
from whetstone.humaneval.metrics import (
    HUMANEVAL_METRICS_PROFILE_ID,
    NodeOutputMetricsSource,
    ast_metrics,
    build_metrics_payload,
    python_leakage_metrics,
    task_test_metrics,
    text_metrics,
)
from whetstone.humaneval.parsed_tests import HumanEvalTestCaseKind
from whetstone.humaneval.profiles import (
    HUMANEVAL_SCORING_PROFILE_ID,
    HUMANEVAL_SCORING_PROFILE_VERSION,
    HumanEvalScoringProfile,
    resolve_humaneval_scoring_profile,
)
from whetstone.humaneval.scoring import GeneratedCodeOutcome
from whetstone.humaneval.task import (
    EvaluationCaseResult,
    EvaluationCaseStatus,
    EvaluationTaskResult,
    HumanEvalTask,
)
from whetstone.lm.boundary import EndpointKind, ProviderKind
from whetstone.platform import rescoring, scoring_workflow
from whetstone.platform.persistence import (
    ScoreAttemptInsertResult,
    ScoreAttemptInsertStatus,
    idempotent_insert_score_attempt,
    persist_score_attempt,
)
from whetstone.platform.scoring import (
    score_generation_run,
    score_metrics_payload,
)
from whetstone.platform.scoring_workflow_state import ScoringWorkflowPresence
from whetstone.records import (
    DEFAULT_SCORE_DATASET_NAME,
    DEFAULT_SCORE_DATASET_SPLIT,
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
from whetstone.records.limits import METRICS_STAGES_MAX_COUNT

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
    provider = _provider()
    prediction_id = stable_prediction_id(
        experiment_name="exp",
        task_id="HumanEval/fixture",
        graph_digest=graph_id,
        dimensions_digest=dimensions_id,
        repetition_seed=0,
        provider_kind=provider.provider_kind.value,
        endpoint_kind=provider.endpoint_kind.value,
        model=provider.model,
        throttle_key=provider.throttle_key,
    )
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
        provider_config=spec.provider_axis,
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


def test_resolve_parser_profile_rejects_unknown_ids() -> None:
    with pytest.raises(ValueError, match="unsupported parser profile id"):
        resolve_parser_profile(
            parser_profile_id="unknown",
            parser_version=PARSER_PROFILE_VERSION,
        )
    with pytest.raises(ValueError, match="unsupported parser profile version"):
        resolve_parser_profile(
            parser_profile_id=BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_ID,
            parser_version="v99",
        )


def test_extract_code_with_profile_dispatches_best_effort() -> None:
    result = extract_code_with_profile(
        "def add_one(x):\n    return x + 1\n",
        profile=BEST_EFFORT_HUMANEVAL_PARSER_PROFILE,
    )

    assert result.succeeded is True
    assert result.extraction_method is ExtractionMethod.BARE_PYTHON


def test_best_effort_parser_rejects_unsupported_raw_type() -> None:
    result = extract_best_effort_code(123)

    assert result.succeeded is False
    assert result.extraction_error == (
        "generation is not a supported code-bearing value"
    )
    assert result.metadata["raw_type"] == "int"


def test_best_effort_parser_reports_missing_code_field_in_mapping() -> None:
    result = extract_best_effort_code({"other": "value"})

    assert result.succeeded is False
    assert result.metadata["available_fields"] == ["other"]


def test_best_effort_parser_reports_no_candidates() -> None:
    result = extract_best_effort_code("plain prose without code anchors")

    assert result.succeeded is False
    assert result.extraction_error == "no code candidates extracted"


def test_best_effort_parser_reports_no_compilable_candidate() -> None:
    result = extract_best_effort_code("def bad(x)\n  pass")

    assert result.succeeded is False
    assert result.extraction_error == "no compilable extracted candidate"
    assert result.compile_error is not None


def test_strict_parser_rejects_non_string_generation() -> None:
    result = extract_strict_field_marker_code({"code": "def f(): pass"})

    assert result.succeeded is False
    assert (
        result.extraction_error
        == "strict parser requires string generation"
    )


def test_strict_parser_rejects_empty_field_marker_body() -> None:
    result = extract_strict_field_marker_code("[[ ## code ## ]]\n   \n")

    assert result.succeeded is False
    assert result.extraction_error == "empty field-marker code"


def test_strict_parser_rejects_syntax_error_in_marker_body() -> None:
    result = extract_strict_field_marker_code(
        "[[ ## code ## ]]\ndef bad(x)\n  pass\n",
    )

    assert result.succeeded is False
    assert result.extraction_error == "field-marker code is not compilable"


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


def test_task_test_metrics_reports_missing_parsed_tests() -> None:
    task = _task().model_copy(update={"parsed_tests": None})

    metrics = task_test_metrics(task)

    assert metrics.parse_ok is False
    assert metrics.parse_error == "HumanEvalTask.parsed_tests is missing"
    assert metrics.task_id == "HumanEval/fixture"
    assert metrics.entry_point == "add_one"
    assert metrics.test_type is None
    assert metrics.case_count == 0


def test_task_test_metrics_summarize_input_expression_tests() -> None:
    task = _task(
        test=(
            "def check(candidate):\n"
            "    inputs = [(1,), (2,)]\n"
            "    results = [2, 3]\n"
            "    for i, (inp, expected) in enumerate(zip(inputs, results)):\n"
            "        assert candidate(*inp) == expected\n"
        ),
    )

    metrics = task_test_metrics(task)

    assert metrics.parse_ok is True
    assert metrics.test_type is HumanEvalTestCaseKind.INPUT_EXPRESSION
    assert metrics.case_count == 2
    assert metrics.input_expression_case_count == 2
    assert metrics.input_result_case_count == 0
    assert metrics.oracle_case_count == 0
    assert metrics.expected_output_expr_count == 0


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


def test_score_metrics_payload_rejects_stage_budget_overflow() -> None:
    spec = _spec()
    max_node_sources = METRICS_STAGES_MAX_COUNT - 1
    node_attempts = tuple(
        _node_attempt(
            spec,
            node_id=f"node-{index}",
            values={"output": f"value-{index}"},
        )
        for index in range(max_node_sources + 1)
    )
    run = _generation_run(spec, "def add_one(x):\n    return x + 1\n")
    scoring_profile = resolve_humaneval_scoring_profile(
        scoring_profile_id=HUMANEVAL_SCORING_PROFILE_ID,
        scoring_profile_version=HUMANEVAL_SCORING_PROFILE_VERSION,
    )
    domain_score = humaneval_scoring.score_humaneval_generation(
        raw_generation=run.summary.terminal_output,
        task=_task(),
        parser_profile=scoring_profile.parser_profile,
        timeout_seconds=scoring_profile.timeout_seconds,
    )

    with pytest.raises(
        ValueError,
        match=r"node output metrics sources cannot exceed",
    ):
        score_metrics_payload(
            task=_task(),
            node_attempts=node_attempts,
            scoring_profile=scoring_profile,
            domain_score=domain_score,
        )


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


def test_score_generation_run_persists_evaluation_incomplete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec()
    run = _generation_run(spec, "def add_one(x):\n    return x + 1\n")
    task = _task()

    def evaluate(
        *,
        task: HumanEvalTask,
        candidate_code: str,
        timeout_seconds: float,
    ) -> EvaluationTaskResult:
        assert candidate_code == "def add_one(x):\n    return x + 1"
        return EvaluationTaskResult(
            task_id=task.task_id,
            entry_point=task.entry_point,
            function_names=[task.entry_point],
            total_cases=2,
            results=[
                EvaluationCaseResult(
                    task_id=task.task_id,
                    case_id="case_0",
                    function_name=task.entry_point,
                    status=EvaluationCaseStatus.PASSED,
                    test_type=HumanEvalTestCaseKind.INPUT_RESULT,
                ),
            ],
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
        task=task,
        started_at=NOW,
        completed_at=LATER,
    )

    assert score.status is ScoreAttemptStatus.SUCCESS
    assert score.score == 0.0
    assert score.generated_code_outcome is (
        GeneratedCodeOutcome.EVALUATION_INCOMPLETE
    )
    assert [result.test_id for result in score.per_test_results] == ["case_0"]
    assert score.metrics is not None
    evaluation_metrics = score.metrics.custom["evaluation"]
    assert evaluation_metrics["result_count"] == 1
    assert evaluation_metrics["total_cases"] == 2
    assert evaluation_metrics["failure_count"] == 0


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
    assert score.extracted_code.raw_generation == (
        '{"code":"def add_one(x):\\n    return x + 1\\n"}'
    )
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


def test_score_generation_run_rejects_task_id_mismatch() -> None:
    spec = _spec()
    run = _generation_run(spec, "def add_one(x):\n    return x + 1\n")
    mismatched_task = _task()
    mismatched_task = mismatched_task.model_copy(
        update={"task_id": "HumanEval/other"}
    )

    score = score_generation_run(
        spec=spec,
        generation_run=run,
        node_attempts=(),
        task=mismatched_task,
        started_at=NOW,
        completed_at=LATER,
    )

    assert score.status is ScoreAttemptStatus.ERROR
    assert score.failure is not None
    assert (
        score.failure.message
        == "HumanEval task_id does not match spec: 'HumanEval/other' != "
        "'HumanEval/fixture'"
    )


def test_score_attempt_id_differs_by_dataset_selection() -> None:
    spec = _spec()
    run = _generation_run(spec, "def add_one(x):\n    return x + 1\n")
    default_id = stable_score_attempt_id(
        generation_run_id=run.generation_run_id,
        scoring_profile_id=HUMANEVAL_SCORING_PROFILE_ID,
        scoring_profile_version=HUMANEVAL_SCORING_PROFILE_VERSION,
        parser_profile_id=BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_ID,
        parser_version=PARSER_PROFILE_VERSION,
        attempt_index=0,
    )
    other_dataset_id = stable_score_attempt_id(
        generation_run_id=run.generation_run_id,
        scoring_profile_id=HUMANEVAL_SCORING_PROFILE_ID,
        scoring_profile_version=HUMANEVAL_SCORING_PROFILE_VERSION,
        parser_profile_id=BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_ID,
        parser_version=PARSER_PROFILE_VERSION,
        attempt_index=0,
        dataset_name="other/dataset",
    )
    other_split_id = stable_score_attempt_id(
        generation_run_id=run.generation_run_id,
        scoring_profile_id=HUMANEVAL_SCORING_PROFILE_ID,
        scoring_profile_version=HUMANEVAL_SCORING_PROFILE_VERSION,
        parser_profile_id=BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_ID,
        parser_version=PARSER_PROFILE_VERSION,
        attempt_index=0,
        dataset_split="train",
    )

    assert default_id != other_dataset_id
    assert default_id != other_split_id
    assert other_dataset_id != other_split_id

    default_score = score_generation_run(
        spec=spec,
        generation_run=run,
        node_attempts=(),
        task=_task(),
        started_at=NOW,
        completed_at=LATER,
    )
    other_dataset_score = score_generation_run(
        spec=spec,
        generation_run=run,
        node_attempts=(),
        task=_task(),
        dataset_name="other/dataset",
        started_at=NOW,
        completed_at=LATER,
    )

    default_row = db_io.score_attempt_row(default_score)
    other_row = db_io.score_attempt_row(other_dataset_score)
    assert default_row["dataset_name"] == DEFAULT_SCORE_DATASET_NAME
    assert default_row["dataset_split"] == DEFAULT_SCORE_DATASET_SPLIT
    assert other_row["dataset_name"] == "other/dataset"
    assert other_row["dataset_split"] == DEFAULT_SCORE_DATASET_SPLIT


def test_task_name_leakage_ignores_numeric_task_suffix() -> None:
    leakage = python_leakage_metrics(
        "return 0 if x == 0 else x + 1",
        task_names=("add_one", "HumanEval/0"),
    )

    assert leakage.task_name_hit_count == 0


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
        def first(self) -> None:
            return None

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
        == "generation run is not scoreable: error"
    )


def test_score_generation_run_scores_partial_with_terminal_output() -> None:
    spec = _spec()
    run = _generation_run(
        spec,
        "def add_one(x):\n    return x + 1\n",
    )
    run = run.model_copy(update={"status": GenerationRunStatus.PARTIAL})

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


def test_score_generation_rejects_partial_without_output() -> None:
    spec = _spec()
    run = _generation_run(spec, "unused").model_copy(
        update={
            "status": GenerationRunStatus.PARTIAL,
            "summary": _generation_run(spec, "unused").summary.model_copy(
                update={"terminal_output": None}
            ),
        }
    )

    score = score_generation_run(
        spec=spec,
        generation_run=run,
        node_attempts=(),
        task=_task(),
        started_at=NOW,
        completed_at=LATER,
    )

    assert score.status is ScoreAttemptStatus.ERROR
    assert score.failure is not None
    assert (
        score.failure.message
        == "partial generation run missing terminal_output"
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
                    args[7],
                    args[8],
                ),
            )
        )
        return score_generation_run(
            spec=spec,
            generation_run=run,
            node_attempts=(),
            task=_task(),
            scoring_profile=scoring_profile,
            score_attempt_index=args[5],
            dataset_name=args[6],
            dataset_split=args[7],
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
                DEFAULT_SCORE_DATASET_NAME,
                DEFAULT_SCORE_DATASET_SPLIT,
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
                DEFAULT_SCORE_DATASET_NAME,
                DEFAULT_SCORE_DATASET_SPLIT,
                NOW.isoformat(),
            ),
        ),
        ("persist", ("postgresql://example/db", expected_score_id)),
    ]


def test_rescore_selector_filters_and_orders_candidates() -> None:
    statement = db_io.select_rescore_generation_candidates(
        experiment_name="exp",
        generation_statuses=(GenerationRunStatus.SUCCESS,),
        generation_attempt_index=0,
        scoring_profile_id=HUMANEVAL_SCORING_PROFILE_ID,
        scoring_profile_version=HUMANEVAL_SCORING_PROFILE_VERSION,
        parser_profile_id=BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_ID,
        parser_version=PARSER_PROFILE_VERSION,
        score_attempt_index=0,
        dataset_name=DEFAULT_SCORE_DATASET_NAME,
        dataset_split=DEFAULT_SCORE_DATASET_SPLIT,
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
    assert "dr_dspy_generation_runs.status IN ('success')" in compiled
    assert "dr_dspy_generation_runs.attempt_index = 0" in compiled
    assert (
        "dr_dspy_score_attempts.scoring_profile_id = 'humaneval'"
    ) in compiled
    assert (
        "dr_dspy_score_attempts.parser_profile_id = "
        "'humaneval-best-effort'"
    ) in compiled
    assert (
        "dr_dspy_score_attempts.dataset_name = "
        "'evalplus/humanevalplus'"
    ) in compiled
    assert "dr_dspy_score_attempts.dataset_split = 'test'" in compiled
    assert "dr_dspy_score_attempts.score_attempt_id IS NULL" in compiled
    assert (
        "ORDER BY dr_dspy_prediction_specs.fair_order_key, "
        "dr_dspy_prediction_specs.prediction_id, "
        "dr_dspy_generation_runs.generation_run_id"
    ) in compiled
    assert "LIMIT 10 OFFSET 2" in compiled


def test_rescore_selector_accepts_multiple_generation_statuses() -> None:
    statement = db_io.select_rescore_generation_candidates(
        experiment_name="exp",
        generation_statuses=(
            GenerationRunStatus.SUCCESS,
            GenerationRunStatus.PARTIAL,
        ),
        scoring_profile_id=HUMANEVAL_SCORING_PROFILE_ID,
        scoring_profile_version=HUMANEVAL_SCORING_PROFILE_VERSION,
        parser_profile_id=BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_ID,
        parser_version=PARSER_PROFILE_VERSION,
        score_attempt_index=0,
        dataset_name=DEFAULT_SCORE_DATASET_NAME,
        dataset_split=DEFAULT_SCORE_DATASET_SPLIT,
        limit=10,
    )

    compiled = str(
        statement.compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"literal_binds": True},
        )
    )

    assert "dr_dspy_generation_runs.status IN ('success', 'partial')" in (
        compiled
    )
    statement = db_io.select_rescore_generation_candidates(
        experiment_name="exp",
        generation_statuses=(GenerationRunStatus.SUCCESS,),
        scoring_profile_id=HUMANEVAL_SCORING_PROFILE_ID,
        scoring_profile_version=HUMANEVAL_SCORING_PROFILE_VERSION,
        parser_profile_id=BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_ID,
        parser_version=PARSER_PROFILE_VERSION,
        score_attempt_index=0,
        dataset_name="other/dataset",
        dataset_split="train",
        limit=10,
    )

    compiled = str(
        statement.compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"literal_binds": True},
        )
    )

    assert "dr_dspy_score_attempts.dataset_name = 'other/dataset'" in compiled
    assert "dr_dspy_score_attempts.dataset_split = 'train'" in compiled


def test_batch_rescore_alternate_dataset_dry_run_schedules_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = (
        _rescore_candidate(1),
    )
    scheduler_calls: list[tuple[str, str]] = []

    monkeypatch.setattr(
        rescoring,
        "load_rescore_generation_candidates",
        lambda connection, **kwargs: candidates,
    )
    monkeypatch.setattr(
        rescoring,
        "count_rescore_generation_candidates",
        _mock_rescore_candidate_count(1),
    )

    def schedule(
        **kwargs: Any,
    ) -> scoring_workflow.ScheduledScoreGenerationWorkflow:
        scheduler_calls.append(
            (kwargs["dataset_name"], kwargs["dataset_split"])
        )
        return scoring_workflow.ScheduledScoreGenerationWorkflow(
            score_attempt_id="unused",
            workflow_id="unused",
            scheduled=True,
        )

    execution = rescoring.rescore_generation_runs(
        cast(Any, DummyEngine()),
        database_url="postgresql://example/db",
        experiment_name="exp",
        dataset_name="other/dataset",
        dataset_split="train",
        dry_run=True,
        schedule_workflow=schedule,
    )
    result = execution.result

    assert scheduler_calls == []
    assert result.selected_count == 1
    assert result.total_candidates == 1
    assert result.dataset_name == "other/dataset"
    assert result.dataset_split == "train"
    assert (
        result.items[0].status
        is rescoring.BatchRescoreItemStatus.WOULD_SCHEDULE
    )


def test_batch_rescore_dry_run_counts_needed_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = (
        _rescore_candidate(1),
    )
    scheduler_calls: list[str] = []

    monkeypatch.setattr(
        rescoring,
        "load_rescore_generation_candidates",
        lambda connection, **kwargs: candidates,
    )
    monkeypatch.setattr(
        rescoring,
        "count_rescore_generation_candidates",
        _mock_rescore_candidate_count(1),
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

    execution = rescoring.rescore_generation_runs(
        cast(Any, DummyEngine()),
        database_url="postgresql://example/db",
        experiment_name="exp",
        dry_run=True,
        schedule_workflow=schedule,
    )
    result = execution.result

    assert scheduler_calls == []
    assert result.selected_count == 1
    assert result.total_candidates == 1
    assert result.already_scored_count == 0
    assert result.needs_score_count == 1
    assert result.in_flight_count == 0
    assert result.orphan_count == 0
    assert result.scheduled_count == 0
    assert [item.status for item in result.items] == [
        rescoring.BatchRescoreItemStatus.WOULD_SCHEDULE,
    ]


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
    monkeypatch.setattr(
        rescoring,
        "count_rescore_generation_candidates",
        _mock_rescore_candidate_count(3),
    )
    monkeypatch.setattr(
        rescoring,
        "classify_scoring_workflow_presence",
        lambda **kwargs: ScoringWorkflowPresence.IN_FLIGHT,
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
            workflow_handle=f"handle-{generation_run_id}",
        )

    execution = rescoring.rescore_generation_runs(
        cast(Any, DummyEngine()),
        database_url="postgresql://example/db",
        experiment_name="exp",
        chunk_size=2,
        schedule_workflow=schedule,
        await_workflows=lambda handles: None,
    )
    result = execution.result

    assert pages == [(2, 0), (2, 2)]
    assert scheduler_calls == [
        "generation-run-0",
        "generation-run-1",
        "generation-run-2",
    ]
    assert result.selected_count == 3
    assert result.total_candidates == 3
    assert result.scheduled_count == 1
    assert result.in_flight_count == 1
    assert result.orphan_count == 0
    assert result.failed_count == 1
    assert result.max_in_flight == rescoring.DEFAULT_MAX_IN_FLIGHT
    assert [item.status for item in result.items] == [
        rescoring.BatchRescoreItemStatus.SCHEDULED,
        rescoring.BatchRescoreItemStatus.WORKFLOW_IN_FLIGHT,
        rescoring.BatchRescoreItemStatus.FAILED,
    ]
    assert result.items[2].failure is not None
    assert execution.workflow_handles == ()


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
    monkeypatch.setattr(
        rescoring,
        "count_rescore_generation_candidates",
        _mock_rescore_candidate_count(5),
    )

    execution = rescoring.rescore_generation_runs(
        cast(Any, DummyEngine()),
        database_url="postgresql://example/db",
        experiment_name="exp",
        chunk_size=2,
        limit=3,
        dry_run=True,
    )
    result = execution.result

    assert pages == [(2, 0), (1, 2)]
    assert result.total_candidates == 3
    assert result.selected_count == 3
    assert [item.generation_run_id for item in result.items] == [
        "generation-run-0",
        "generation-run-1",
        "generation-run-2",
    ]


def test_batch_rescore_max_in_flight_uses_sliding_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = tuple(_rescore_candidate(index) for index in range(5))
    await_batches: list[tuple[Any, ...]] = []

    monkeypatch.setattr(
        rescoring,
        "load_rescore_generation_candidates",
        lambda connection, **kwargs: candidates,
    )
    monkeypatch.setattr(
        rescoring,
        "count_rescore_generation_candidates",
        _mock_rescore_candidate_count(5),
    )

    def schedule(
        **kwargs: Any,
    ) -> scoring_workflow.ScheduledScoreGenerationWorkflow:
        generation_run_id = kwargs["generation_run_id"]
        return scoring_workflow.ScheduledScoreGenerationWorkflow(
            score_attempt_id=f"score-{generation_run_id}",
            workflow_id=f"workflow-{generation_run_id}",
            scheduled=True,
            workflow_handle=f"handle-{generation_run_id}",
        )

    def await_workflows(handles: list[Any]) -> None:
        await_batches.append(tuple(handles))

    execution = rescoring.rescore_generation_runs(
        cast(Any, DummyEngine()),
        database_url="postgresql://example/db",
        experiment_name="exp",
        max_in_flight=2,
        schedule_workflow=schedule,
        await_workflows=await_workflows,
    )
    result = execution.result

    assert await_batches == [
        ("handle-generation-run-0",),
        ("handle-generation-run-1",),
        ("handle-generation-run-2",),
        ("handle-generation-run-3",),
        ("handle-generation-run-4",),
    ]
    assert result.max_in_flight == 2
    assert result.total_candidates == 5
    assert result.scheduled_count == 5
    assert execution.workflow_handles == ()


def test_batch_rescore_sliding_window_keeps_scheduling_under_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = tuple(_rescore_candidate(index) for index in range(4))
    schedule_order: list[str] = []
    await_order: list[str] = []
    blocked: dict[str, bool] = {
        "handle-generation-run-0": True,
        "handle-generation-run-1": True,
    }

    monkeypatch.setattr(
        rescoring,
        "load_rescore_generation_candidates",
        lambda connection, **kwargs: candidates,
    )
    monkeypatch.setattr(
        rescoring,
        "count_rescore_generation_candidates",
        _mock_rescore_candidate_count(4),
    )

    def schedule(
        **kwargs: Any,
    ) -> scoring_workflow.ScheduledScoreGenerationWorkflow:
        generation_run_id = kwargs["generation_run_id"]
        handle_label = f"handle-{generation_run_id}"
        schedule_order.append(handle_label)

        class FakeHandle:
            def __init__(self, label: str) -> None:
                self.label = label

            def get_result(self) -> None:
                if blocked.get(self.label):
                    raise AssertionError(
                        f"{self.label} should not complete before slot release"
                    )

        return scoring_workflow.ScheduledScoreGenerationWorkflow(
            score_attempt_id=f"score-{generation_run_id}",
            workflow_id=f"workflow-{generation_run_id}",
            scheduled=True,
            workflow_handle=FakeHandle(handle_label),
        )

    def await_workflows(handles: list[Any]) -> None:
        assert len(handles) == 1
        handle = handles[0]
        await_order.append(handle.label)
        blocked.pop(handle.label, None)
        handle.get_result()

    execution = rescoring.rescore_generation_runs(
        cast(Any, DummyEngine()),
        database_url="postgresql://example/db",
        experiment_name="exp",
        max_in_flight=2,
        schedule_workflow=schedule,
        await_workflows=await_workflows,
    )

    assert schedule_order == [
        "handle-generation-run-0",
        "handle-generation-run-1",
        "handle-generation-run-2",
        "handle-generation-run-3",
    ]
    assert await_order == [
        "handle-generation-run-0",
        "handle-generation-run-1",
        "handle-generation-run-2",
        "handle-generation-run-3",
    ]
    assert execution.result.scheduled_count == 4


def test_batch_rescore_dry_run_does_not_await_workflows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = (_rescore_candidate(0),)

    monkeypatch.setattr(
        rescoring,
        "load_rescore_generation_candidates",
        lambda connection, **kwargs: candidates,
    )
    monkeypatch.setattr(
        rescoring,
        "count_rescore_generation_candidates",
        _mock_rescore_candidate_count(1),
    )

    def fail_await(handles: list[Any]) -> None:
        raise AssertionError("dry-run should not await workflows")

    execution = rescoring.rescore_generation_runs(
        cast(Any, DummyEngine()),
        database_url="postgresql://example/db",
        experiment_name="exp",
        dry_run=True,
        await_workflows=fail_await,
    )

    assert execution.result.max_in_flight == rescoring.DEFAULT_MAX_IN_FLIGHT
    assert execution.workflow_handles == ()


def test_await_scheduled_score_workflows_waits_for_each_handle() -> None:
    completed: list[str] = []

    class FakeHandle:
        def __init__(self, label: str) -> None:
            self.label = label

        def get_result(self) -> None:
            completed.append(self.label)

    scoring_workflow.await_scheduled_score_workflows(
        [FakeHandle("first"), FakeHandle("second")]
    )

    assert completed == ["first", "second"]


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

    monkeypatch.setattr(
        scoring_workflow,
        "classify_scoring_workflow_presence",
        lambda **kwargs: ScoringWorkflowPresence.IN_FLIGHT,
    )

    class FakeDbos:
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


def test_schedule_score_generation_workflow_surfaces_unrelated_start_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    starts: list[Any] = []

    monkeypatch.setattr(
        scoring_workflow,
        "classify_scoring_workflow_presence",
        lambda **kwargs: ScoringWorkflowPresence.ABSENT,
    )

    class FakeDbos:
        def start_workflow(self, *args: Any) -> None:
            starts.append(args)
            raise RuntimeError("dbos unavailable")

    monkeypatch.setattr(scoring_workflow, "DBOS", FakeDbos())

    with pytest.raises(RuntimeError, match="dbos unavailable"):
        scoring_workflow.schedule_score_generation_workflow(
            database_url="postgresql://example/db",
            generation_run_id="generation-run-1",
        )


def _schedule_score_ids(
    generation_run_id: str,
) -> tuple[str, str]:
    score_attempt_id = stable_score_attempt_id(
        generation_run_id=generation_run_id,
        scoring_profile_id=HUMANEVAL_SCORING_PROFILE_ID,
        scoring_profile_version=HUMANEVAL_SCORING_PROFILE_VERSION,
        parser_profile_id=BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_ID,
        parser_version=PARSER_PROFILE_VERSION,
        attempt_index=0,
    )
    return (
        score_attempt_id,
        scoring_workflow.platform_scoring_workflow_id(score_attempt_id),
    )


@pytest.mark.parametrize(
    ("presence", "recover_orphans", "recover_result", "expected"),
    [
        (
            ScoringWorkflowPresence.COMPLETE,
            False,
            None,
            {"scheduled": False, "recovered": False},
        ),
        (
            ScoringWorkflowPresence.ABSENT,
            False,
            None,
            {"scheduled": True, "recovered": False},
        ),
        (
            ScoringWorkflowPresence.ORPHAN,
            False,
            None,
            {"scheduled": False, "recovered": False},
        ),
        (
            ScoringWorkflowPresence.ORPHAN,
            True,
            True,
            {"scheduled": True, "recovered": True},
        ),
        (
            ScoringWorkflowPresence.ORPHAN,
            True,
            False,
            {"scheduled": False, "recovered": False},
        ),
    ],
)
def test_schedule_score_generation_workflow_presence_matrix(
    monkeypatch: pytest.MonkeyPatch,
    presence: ScoringWorkflowPresence,
    recover_orphans: bool,
    recover_result: bool | None,
    expected: dict[str, bool],
) -> None:
    generation_run_id = "generation-run-schedule"
    score_attempt_id, workflow_id = _schedule_score_ids(generation_run_id)
    starts: list[Any] = []

    monkeypatch.setattr(
        scoring_workflow,
        "classify_scoring_workflow_presence",
        lambda **kwargs: presence,
    )
    if recover_result is not None:
        monkeypatch.setattr(
            scoring_workflow,
            "recover_orphan_scoring_workflow",
            lambda **kwargs: recover_result,
        )

    class FakeDbos:
        def start_workflow(self, *args: Any) -> None:
            starts.append(args)

    monkeypatch.setattr(scoring_workflow, "DBOS", FakeDbos())

    result = scoring_workflow.schedule_score_generation_workflow(
        database_url="postgresql://example/db",
        generation_run_id=generation_run_id,
        recover_orphans=recover_orphans,
    )

    assert result == scoring_workflow.ScheduledScoreGenerationWorkflow(
        score_attempt_id=score_attempt_id,
        workflow_id=workflow_id,
        scheduled=expected["scheduled"],
        recovered=expected["recovered"],
    )
    if presence is ScoringWorkflowPresence.ABSENT:
        assert len(starts) == 1
    else:
        assert starts == []


def test_schedule_score_generation_workflow_treats_start_race_as_unscheduled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generation_run_id = "generation-run-race"
    score_attempt_id, workflow_id = _schedule_score_ids(generation_run_id)

    monkeypatch.setattr(
        scoring_workflow,
        "classify_scoring_workflow_presence",
        lambda **kwargs: ScoringWorkflowPresence.ABSENT,
    )

    class FakeDbos:
        def start_workflow(self, *args: Any) -> None:
            raise DBOSWorkflowConflictIDError(workflow_id)

    monkeypatch.setattr(scoring_workflow, "DBOS", FakeDbos())

    result = scoring_workflow.schedule_score_generation_workflow(
        database_url="postgresql://example/db",
        generation_run_id=generation_run_id,
    )

    assert result == scoring_workflow.ScheduledScoreGenerationWorkflow(
        score_attempt_id=score_attempt_id,
        workflow_id=workflow_id,
        scheduled=False,
    )


def _mock_rescore_candidate_count(count: int):
    return lambda connection, **kwargs: count


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


def test_classify_scoring_workflow_presence_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from whetstone.platform import scoring_workflow_state

    monkeypatch.setattr(
        scoring_workflow_state,
        "score_attempt_exists",
        lambda database_url, score_attempt_id: score_attempt_id == "complete",
    )
    monkeypatch.setattr(
        scoring_workflow_state,
        "DBOS",
        type(
            "FakeDbos",
            (),
            {
                "get_workflow_status": staticmethod(
                    lambda workflow_id: {"status": "PENDING"}
                )
            },
        )(),
    )
    assert (
        scoring_workflow_state.classify_scoring_workflow_presence(
            database_url="postgresql://example/db",
            score_attempt_id="complete",
            workflow_id="platform-score-v1:complete",
        )
        is ScoringWorkflowPresence.COMPLETE
    )

    monkeypatch.setattr(
        scoring_workflow_state,
        "score_attempt_exists",
        lambda database_url, score_attempt_id: False,
    )
    assert (
        scoring_workflow_state.classify_scoring_workflow_presence(
            database_url="postgresql://example/db",
            score_attempt_id="missing",
            workflow_id="platform-score-v1:missing",
        )
        is ScoringWorkflowPresence.IN_FLIGHT
    )

    monkeypatch.setattr(
        scoring_workflow_state,
        "DBOS",
        type(
            "FakeDbos",
            (),
            {
                "get_workflow_status": staticmethod(
                    lambda workflow_id: {"status": "SUCCESS"}
                )
            },
        )(),
    )
    assert (
        scoring_workflow_state.classify_scoring_workflow_presence(
            database_url="postgresql://example/db",
            score_attempt_id="orphan",
            workflow_id="platform-score-v1:orphan",
        )
        is ScoringWorkflowPresence.ORPHAN
    )

    monkeypatch.setattr(
        scoring_workflow_state,
        "DBOS",
        type(
            "FakeDbos",
            (),
            {"get_workflow_status": staticmethod(lambda workflow_id: None)},
        )(),
    )
    assert (
        scoring_workflow_state.classify_scoring_workflow_presence(
            database_url="postgresql://example/db",
            score_attempt_id="absent",
            workflow_id="platform-score-v1:absent",
        )
        is ScoringWorkflowPresence.ABSENT
    )

    for failed_status in (
        "ERROR",
        "CANCELLED",
        "MAX_RECOVERY_ATTEMPTS_EXCEEDED",
    ):
        monkeypatch.setattr(
            scoring_workflow_state,
            "score_attempt_exists",
            lambda database_url, score_attempt_id: False,
        )
        monkeypatch.setattr(
            scoring_workflow_state,
            "DBOS",
            type(
                "FakeDbos",
                (),
                {
                    "get_workflow_status": staticmethod(
                        lambda workflow_id, status=failed_status: {
                            "status": status
                        }
                    )
                },
            )(),
        )
        assert (
            scoring_workflow_state.classify_scoring_workflow_presence(
                database_url="postgresql://example/db",
                score_attempt_id=f"orphan-{failed_status.lower()}",
                workflow_id=(
                    f"platform-score-v1:orphan-{failed_status.lower()}"
                ),
            )
            is ScoringWorkflowPresence.ORPHAN
        )


def test_recover_orphan_scoring_workflow_replays_workflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from whetstone.platform import scoring_workflow_state

    replay_calls: list[tuple[Any, ...]] = []
    persist_state = {"exists": False}

    def replay(*args: Any) -> None:
        replay_calls.append(args)
        persist_state["exists"] = True

    monkeypatch.setattr(
        scoring_workflow_state,
        "score_attempt_exists",
        lambda database_url, score_attempt_id: persist_state["exists"],
    )
    monkeypatch.setattr(
        scoring_workflow_state,
        "DBOS",
        type(
            "FakeDbos",
            (),
            {
                "retrieve_workflow": staticmethod(
                    lambda workflow_id: type(
                        "Handle",
                        (),
                        {"get_result": staticmethod(lambda: None)},
                    )()
                )
            },
        )(),
    )

    recovered = scoring_workflow_state.recover_orphan_scoring_workflow(
        database_url="postgresql://example/db",
        workflow_id="platform-score-v1:orphan",
        score_attempt_id="pending",
        replay_workflow=replay,
        replay_args=("db", "run-1"),
    )
    assert recovered is True
    assert replay_calls == [("db", "run-1")]


def test_recover_orphan_scoring_workflow_skips_replay_when_row_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from whetstone.platform import scoring_workflow_state

    replay_calls: list[tuple[Any, ...]] = []

    def replay(*args: Any) -> None:
        replay_calls.append(args)

    class OrphanHandle:
        @staticmethod
        def get_result() -> None:
            raise RuntimeError("orphan result unavailable")

    monkeypatch.setattr(
        scoring_workflow_state,
        "score_attempt_exists",
        lambda database_url, score_attempt_id: True,
    )
    monkeypatch.setattr(
        scoring_workflow_state,
        "DBOS",
        type(
            "FakeDbos",
            (),
            {
                "retrieve_workflow": staticmethod(
                    lambda workflow_id: OrphanHandle()
                )
            },
        )(),
    )

    recovered = scoring_workflow_state.recover_orphan_scoring_workflow(
        database_url="postgresql://example/db",
        workflow_id="platform-score-v1:orphan",
        score_attempt_id="existing",
        replay_workflow=replay,
        replay_args=("db", "run-1"),
    )
    assert recovered is True
    assert replay_calls == []


def test_recover_orphan_scoring_workflow_returns_false_when_replay_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from whetstone.platform import scoring_workflow_state

    monkeypatch.setattr(
        scoring_workflow_state,
        "score_attempt_exists",
        lambda database_url, score_attempt_id: False,
    )
    monkeypatch.setattr(
        scoring_workflow_state,
        "DBOS",
        type(
            "FakeDbos",
            (),
            {
                "retrieve_workflow": staticmethod(
                    lambda workflow_id: type(
                        "Handle",
                        (),
                        {"get_result": staticmethod(lambda: None)},
                    )()
                )
            },
        )(),
    )

    recovered = scoring_workflow_state.recover_orphan_scoring_workflow(
        database_url="postgresql://example/db",
        workflow_id="platform-score-v1:orphan",
        score_attempt_id="pending",
        replay_workflow=lambda *args: None,
        replay_args=("db", "run-1"),
    )
    assert recovered is False


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
