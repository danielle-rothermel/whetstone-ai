from __future__ import annotations

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

from dr_dspy.humaneval.parsed_tests import HumanEvalTestCaseKind


class TextMetricsPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    character_count: StrictInt
    byte_count: StrictInt
    line_count: StrictInt
    nonempty_line_count: StrictInt
    word_count: StrictInt
    average_word_length: StrictFloat | None = None
    punctuation_count: StrictInt | None = None
    symbol_count: StrictInt | None = None


class PythonLeakageMetricsPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    keyword_count: StrictInt
    code_marker_count: StrictInt
    fenced_code_block_count: StrictInt
    code_like_line_count: StrictInt
    operator_count: StrictInt
    punctuation_density: StrictFloat | None = None
    task_name_hit_count: StrictInt | None = None


class AstMetricsPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parse_ok: StrictBool
    parse_error: StrictStr | None = None
    top_level_function_count: StrictInt = 0
    top_level_function_names: tuple[StrictStr, ...] = ()
    nested_function_count: StrictInt = 0
    async_function_count: StrictInt = 0
    lambda_count: StrictInt = 0
    class_count: StrictInt = 0
    import_count: StrictInt = 0
    ast_node_count: StrictInt = 0
    statement_count: StrictInt = 0
    branch_count: StrictInt = 0
    return_count: StrictInt = 0
    yield_count: StrictInt = 0
    call_count: StrictInt = 0
    assignment_count: StrictInt = 0
    comprehension_count: StrictInt = 0
    literal_count: StrictInt = 0
    max_branch_depth: StrictInt = 0
    function_count: StrictInt = 0
    total_argument_count: StrictInt = 0
    positional_only_argument_count: StrictInt = 0
    keyword_only_argument_count: StrictInt = 0
    vararg_count: StrictInt = 0
    kwarg_count: StrictInt = 0
    decorated_function_count: StrictInt = 0
    annotated_return_count: StrictInt = 0
    docstring_function_count: StrictInt = 0
    total_function_body_statement_count: StrictInt = 0
    max_function_body_statement_count: StrictInt = 0
    max_function_line_span: StrictInt = 0


class HumanEvalTaskTestMetricsPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parse_ok: StrictBool
    parse_error: StrictStr | None = None
    task_id: StrictStr
    entry_point: StrictStr
    test_type: HumanEvalTestCaseKind | None = None
    case_count: StrictInt = 0
    support_code_character_count: StrictInt = 0
    support_code_line_count: StrictInt = 0
    original_test_character_count: StrictInt = 0
    original_test_line_count: StrictInt = 0
    assertion_name: StrictStr | None = None
    check_name: StrictStr | None = None
    candidate_arg_name: StrictStr | None = None
    input_repr_character_total: StrictInt = 0
    expected_output_repr_character_total: StrictInt = 0
    expected_output_expr_count: StrictInt = 0
    oracle_case_count: StrictInt = 0
    input_result_case_count: StrictInt = 0
    input_expression_case_count: StrictInt = 0


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
    task_tests: HumanEvalTaskTestMetricsPayload | None = None
    text: TextMetricsPayload | None = None
    python_leakage: PythonLeakageMetricsPayload | None = None
    ast: AstMetricsPayload | None = None
    compression: dict[StrictStr, Any] = Field(default_factory=dict)
    stages: tuple[MetricsStagePayload, ...] = ()
    custom: dict[StrictStr, Any] = Field(default_factory=dict)
