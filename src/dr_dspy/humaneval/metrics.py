from __future__ import annotations

import ast
import keyword
import re
import string
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    StrictStr,
)

from dr_dspy.humaneval.compression import compression_metrics
from dr_dspy.humaneval.metric_models import (
    AstMetricsPayload,
    HumanEvalTaskTestMetricsPayload,
    MetricsPayload,
    MetricsStagePayload,
    PythonLeakageMetricsPayload,
    TextMetricsPayload,
)
from dr_dspy.humaneval.parsed_tests import (
    HumanEvalTestCaseKind,
    ParsedTestsSummary,
)
from dr_dspy.humaneval.task import HumanEvalTask

HUMANEVAL_METRICS_PROFILE_ID = "humaneval-metrics"
HUMANEVAL_METRICS_PROFILE_VERSION = "v1"
TEXT_ENCODING = "utf-8"
MAX_TOP_LEVEL_FUNCTION_NAMES = 20
WORD_RE = re.compile(r"\b\w+\b")
FENCED_CODE_RE = re.compile(r"```|~~~")
CODE_LIKE_LINE_RE = re.compile(
    r"^\s*(def |async def |class |import |from |return\b|if |for |while |"
    r"try:|except\b|with |[A-Za-z_]\w*\s*=)"
)
CODE_MARKERS = frozenset({"def", "return", "import", "class"})
OPERATOR_CHARS = frozenset("+-*/%=<>!&|^~:@")
BRANCH_NODES = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.Try,
    ast.IfExp,
    ast.BoolOp,
    ast.Match,
)
ASSIGNMENT_NODES = (ast.Assign, ast.AnnAssign, ast.AugAssign)
COMPREHENSION_NODES = (
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
)
LITERAL_NODES = (ast.Constant, ast.List, ast.Tuple, ast.Set, ast.Dict)
FUNCTION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)


class NodeOutputMetricsSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: StrictStr
    field_name: StrictStr
    text: StrictStr


def build_metrics_payload(
    *,
    raw_generation: str,
    extracted_code: str | None,
    task: HumanEvalTask,
    node_output_sources: tuple[NodeOutputMetricsSource, ...] = (),
    profile_id: str = HUMANEVAL_METRICS_PROFILE_ID,
    profile_version: str = HUMANEVAL_METRICS_PROFILE_VERSION,
) -> MetricsPayload:
    stages: list[MetricsStagePayload] = [
        build_metrics_stage(
            stage_id="terminal",
            source_kind="terminal_raw_generation",
            text=raw_generation,
            task=task,
            include_ast=False,
            include_compression=True,
        )
    ]
    extracted_stage: MetricsStagePayload | None = None
    if extracted_code is not None:
        extracted_stage = build_metrics_stage(
            stage_id="extracted_code",
            source_kind="extracted_code",
            text=extracted_code,
            task=task,
            include_ast=True,
            include_compression=True,
        )
        stages.append(extracted_stage)
    stages.extend(
        node_output_stages(node_output_sources=node_output_sources, task=task)
    )
    terminal_stage = stages[0]
    return MetricsPayload(
        profile_id=profile_id,
        profile_version=profile_version,
        task_tests=task_test_metrics(task),
        text=terminal_stage.text,
        python_leakage=terminal_stage.python_leakage,
        ast=extracted_stage.ast if extracted_stage is not None else None,
        compression=(
            extracted_stage.compression
            if extracted_stage is not None
            else terminal_stage.compression
        ),
        stages=tuple(stages),
        custom={
            "stage_count": len(stages),
            "task_id": task.task_id,
            "entry_point": task.entry_point,
        },
    )


def build_metrics_stage(
    *,
    stage_id: str,
    source_kind: str,
    text: str,
    task: HumanEvalTask,
    include_ast: bool,
    include_compression: bool,
) -> MetricsStagePayload:
    return MetricsStagePayload(
        stage_id=stage_id,
        source_kind=source_kind,
        text=text_metrics(text),
        python_leakage=python_leakage_metrics(
            text,
            task_names=(task.entry_point, task.task_id.rsplit("/", 1)[-1]),
        ),
        ast=ast_metrics(text) if include_ast else None,
        compression=(
            compression_metrics_payload(
                ground_truth_code=task.ground_truth_code,
                representation_text=text,
            )
            if include_compression
            else {}
        ),
    )


def node_output_stages(
    *,
    node_output_sources: tuple[NodeOutputMetricsSource, ...],
    task: HumanEvalTask,
) -> tuple[MetricsStagePayload, ...]:
    stages: list[MetricsStagePayload] = []
    for source in sorted(
        node_output_sources,
        key=lambda item: (item.node_id, item.field_name),
    ):
        stages.append(
            build_metrics_stage(
                stage_id=f"node:{source.node_id}:{source.field_name}",
                source_kind="node_output",
                text=source.text,
                task=task,
                include_ast=True,
                include_compression=False,
            )
        )
    return tuple(stages)


def text_metrics(value: str) -> TextMetricsPayload:
    words = WORD_RE.findall(value)
    word_lengths = [len(word) for word in words]
    punctuation_count = sum(1 for char in value if char in string.punctuation)
    return TextMetricsPayload(
        character_count=len(value),
        byte_count=len(value.encode(TEXT_ENCODING)),
        line_count=len(value.split("\n")) if value else 0,
        nonempty_line_count=sum(
            1 for line in value.splitlines() if line.strip()
        ),
        word_count=len(words),
        average_word_length=(
            sum(word_lengths) / len(word_lengths) if word_lengths else None
        ),
        punctuation_count=punctuation_count,
        symbol_count=sum(1 for char in value if char in OPERATOR_CHARS),
    )


def python_leakage_metrics(
    value: str,
    *,
    task_names: tuple[str, ...] = (),
) -> PythonLeakageMetricsPayload:
    words = WORD_RE.findall(value)
    punctuation_count = sum(1 for char in value if char in string.punctuation)
    return PythonLeakageMetricsPayload(
        keyword_count=sum(1 for word in words if keyword.iskeyword(word)),
        code_marker_count=sum(1 for word in words if word in CODE_MARKERS),
        fenced_code_block_count=len(FENCED_CODE_RE.findall(value)) // 2,
        code_like_line_count=sum(
            1 for line in value.splitlines() if CODE_LIKE_LINE_RE.match(line)
        ),
        operator_count=sum(1 for char in value if char in OPERATOR_CHARS),
        punctuation_density=(
            punctuation_count / len(value) if value else None
        ),
        task_name_hit_count=sum(
            value.count(task_name) for task_name in task_names if task_name
        ),
    )


def task_test_metrics(task: HumanEvalTask) -> HumanEvalTaskTestMetricsPayload:
    if task.parsed_tests is None:
        return HumanEvalTaskTestMetricsPayload(
            parse_ok=False,
            parse_error="HumanEvalTask.parsed_tests is missing",
            task_id=task.task_id,
            entry_point=task.entry_point,
        )
    summary = task.parsed_tests.to_summary()
    return task_test_metrics_from_summary(
        task_id=task.task_id,
        entry_point=task.entry_point,
        summary=summary,
    )


def task_test_metrics_from_summary(
    *,
    task_id: str,
    entry_point: str,
    summary: ParsedTestsSummary,
) -> HumanEvalTaskTestMetricsPayload:
    return HumanEvalTaskTestMetricsPayload(
        parse_ok=True,
        parse_error=None,
        task_id=task_id,
        entry_point=entry_point,
        test_type=summary.test_type,
        case_count=len(summary.cases),
        support_code_character_count=len(summary.support_code),
        support_code_line_count=line_count(summary.support_code),
        original_test_character_count=len(summary.original_test),
        original_test_line_count=line_count(summary.original_test),
        assertion_name=summary.assertion_name,
        check_name=summary.check_name,
        candidate_arg_name=summary.candidate_arg_name,
        input_repr_character_total=sum(
            len(case.input_repr) for case in summary.cases
        ),
        expected_output_repr_character_total=sum(
            len(case.expected_output_repr) for case in summary.cases
        ),
        expected_output_expr_count=sum(
            case.expected_output_expr is not None for case in summary.cases
        ),
        oracle_case_count=sum(
            case.kind is HumanEvalTestCaseKind.INPUT_ORACLE
            for case in summary.cases
        ),
        input_result_case_count=sum(
            case.kind is HumanEvalTestCaseKind.INPUT_RESULT
            for case in summary.cases
        ),
        input_expression_case_count=sum(
            case.kind is HumanEvalTestCaseKind.INPUT_EXPRESSION
            for case in summary.cases
        ),
    )


def line_count(value: str) -> int:
    return len(value.split("\n")) if value else 0


def ast_metrics(source: str) -> AstMetricsPayload:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError) as exc:
        return AstMetricsPayload(
            parse_ok=False,
            parse_error=f"{type(exc).__name__}: {exc}",
        )
    top_level_functions = [
        node for node in tree.body if isinstance(node, FUNCTION_NODES)
    ]
    all_functions = [
        node for node in ast.walk(tree) if isinstance(node, FUNCTION_NODES)
    ]
    return AstMetricsPayload(
        parse_ok=True,
        parse_error=None,
        top_level_function_count=len(top_level_functions),
        top_level_function_names=tuple(
            node.name
            for node in top_level_functions[:MAX_TOP_LEVEL_FUNCTION_NAMES]
        ),
        nested_function_count=len(all_functions) - len(top_level_functions),
        async_function_count=sum(
            isinstance(node, ast.AsyncFunctionDef) for node in all_functions
        ),
        lambda_count=sum(
            isinstance(node, ast.Lambda) for node in ast.walk(tree)
        ),
        class_count=sum(
            isinstance(node, ast.ClassDef) for node in ast.walk(tree)
        ),
        import_count=sum(
            isinstance(node, ast.Import | ast.ImportFrom)
            for node in ast.walk(tree)
        ),
        ast_node_count=sum(1 for _node in ast.walk(tree)),
        statement_count=sum(
            isinstance(node, ast.stmt) for node in ast.walk(tree)
        ),
        branch_count=sum(
            isinstance(node, BRANCH_NODES) for node in ast.walk(tree)
        ),
        return_count=sum(
            isinstance(node, ast.Return) for node in ast.walk(tree)
        ),
        yield_count=sum(
            isinstance(node, ast.Yield | ast.YieldFrom)
            for node in ast.walk(tree)
        ),
        call_count=sum(isinstance(node, ast.Call) for node in ast.walk(tree)),
        assignment_count=sum(
            isinstance(node, ASSIGNMENT_NODES) for node in ast.walk(tree)
        ),
        comprehension_count=sum(
            isinstance(node, COMPREHENSION_NODES) for node in ast.walk(tree)
        ),
        literal_count=sum(
            isinstance(node, LITERAL_NODES) for node in ast.walk(tree)
        ),
        max_branch_depth=max_branch_depth(tree),
        function_count=len(all_functions),
        total_argument_count=sum(
            function_argument_count(node) for node in all_functions
        ),
        positional_only_argument_count=sum(
            len(node.args.posonlyargs) for node in all_functions
        ),
        keyword_only_argument_count=sum(
            len(node.args.kwonlyargs) for node in all_functions
        ),
        vararg_count=sum(
            node.args.vararg is not None for node in all_functions
        ),
        kwarg_count=sum(node.args.kwarg is not None for node in all_functions),
        decorated_function_count=sum(
            bool(node.decorator_list) for node in all_functions
        ),
        annotated_return_count=sum(
            node.returns is not None for node in all_functions
        ),
        docstring_function_count=sum(
            ast.get_docstring(node) is not None for node in all_functions
        ),
        total_function_body_statement_count=sum(
            len(node.body) for node in all_functions
        ),
        max_function_body_statement_count=max(
            (len(node.body) for node in all_functions),
            default=0,
        ),
        max_function_line_span=max(
            (function_line_span(node) for node in all_functions),
            default=0,
        ),
    )


def function_argument_count(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> int:
    return (
        len(node.args.posonlyargs)
        + len(node.args.args)
        + len(node.args.kwonlyargs)
        + int(node.args.vararg is not None)
        + int(node.args.kwarg is not None)
    )


def function_line_span(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    end_lineno = getattr(node, "end_lineno", None)
    if end_lineno is None:
        return 0
    return max(0, end_lineno - node.lineno + 1)


def max_branch_depth(node: ast.AST, *, current_depth: int = 0) -> int:
    next_depth = (
        current_depth + 1
        if isinstance(node, BRANCH_NODES)
        else current_depth
    )
    child_depth = max(
        (
            max_branch_depth(child, current_depth=next_depth)
            for child in ast.iter_child_nodes(node)
        ),
        default=next_depth,
    )
    return max(next_depth, child_depth)


def compression_metrics_payload(
    *,
    ground_truth_code: str,
    representation_text: str,
) -> dict[str, Any]:
    return {
        method.value: metric.model_dump(mode="json")
        for method, metric in compression_metrics(
            ground_truth_code=ground_truth_code,
            representation_text=representation_text,
        ).items()
    }
