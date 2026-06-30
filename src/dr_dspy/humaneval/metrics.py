from __future__ import annotations

import ast
import keyword
import re
import string
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
)

from dr_dspy.humaneval.compression import compression_metrics
from dr_dspy.humaneval.task import HumanEvalTask

HUMANEVAL_METRICS_PROFILE_ID = "humaneval-metrics"
HUMANEVAL_METRICS_PROFILE_VERSION = "v1"
TEXT_ENCODING = "utf-8"
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


class TextMetricsPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    character_count: StrictInt
    byte_count: StrictInt
    line_count: StrictInt
    nonempty_line_count: StrictInt
    word_count: StrictInt
    average_word_length: StrictFloat | None = None
    punctuation_count: StrictInt
    symbol_count: StrictInt


class PythonLeakageMetricsPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    keyword_count: StrictInt
    code_marker_count: StrictInt
    fenced_code_block_count: StrictInt
    code_like_line_count: StrictInt
    operator_count: StrictInt
    punctuation_density: StrictFloat | None = None
    task_name_hit_count: StrictInt


class AstMetricsPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parse_ok: StrictBool
    parse_error: StrictStr | None = None
    top_level_function_count: StrictInt = 0
    class_count: StrictInt = 0
    import_count: StrictInt = 0
    ast_node_count: StrictInt = 0
    statement_count: StrictInt = 0
    branch_count: StrictInt = 0


class MetricsStagePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage_id: StrictStr
    source_kind: StrictStr
    text: TextMetricsPayload
    python_leakage: PythonLeakageMetricsPayload
    ast: AstMetricsPayload | None = None
    compression: dict[StrictStr, Any] = Field(default_factory=dict)
    custom: dict[StrictStr, Any] = Field(default_factory=dict)


class MetricsPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile_id: StrictStr
    profile_version: StrictStr
    text: TextMetricsPayload | None = None
    python_leakage: PythonLeakageMetricsPayload | None = None
    ast: AstMetricsPayload | None = None
    compression: dict[StrictStr, Any] = Field(default_factory=dict)
    stages: tuple[MetricsStagePayload, ...] = ()
    custom: dict[StrictStr, Any] = Field(default_factory=dict)


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


def ast_metrics(source: str) -> AstMetricsPayload:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError) as exc:
        return AstMetricsPayload(
            parse_ok=False,
            parse_error=f"{type(exc).__name__}: {exc}",
        )
    return AstMetricsPayload(
        parse_ok=True,
        parse_error=None,
        top_level_function_count=sum(
            isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
            for node in tree.body
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
    )


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
