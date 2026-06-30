from __future__ import annotations

import ast
from collections.abc import Iterator
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

EXPECTED_ARG_INDEX = 1
TOLERANCE_ARG_INDEX = 2
PAIR_TARGET_SIZE = 2


class UnsupportedTestFormatError(ValueError):
    pass


class HumanEvalTestCaseKind(StrEnum):
    INPUT_RESULT = "input_result"
    INPUT_ORACLE = "input_oracle"
    INPUT_EXPRESSION = "input_expression"


class SingleCaseCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    code: str
    input_repr: str = ""
    expected_output_repr: str = ""
    actual_output_expr: str = ""
    expected_output_expr: str | None = None


class InputResultTestCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal[HumanEvalTestCaseKind.INPUT_RESULT] = (
        HumanEvalTestCaseKind.INPUT_RESULT
    )
    case_id: str
    args: list[Any]
    expected: Any
    atol: float = 0

    def as_check(
        self,
        *,
        candidate_name: str,
        assertion_name: str,
    ) -> SingleCaseCheck:
        return SingleCaseCheck(
            case_id=self.case_id,
            input_repr=repr(self.args),
            expected_output_repr=repr(self.expected),
            actual_output_expr=f"{candidate_name}(*{self.args!r})",
            code=(
                f"{assertion_name}("
                f"{candidate_name}(*{self.args!r}), "
                f"{self.expected!r}, {self.atol!r})"
            ),
        )


class InputOracleTestCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal[HumanEvalTestCaseKind.INPUT_ORACLE] = (
        HumanEvalTestCaseKind.INPUT_ORACLE
    )
    case_id: str
    args: list[Any]
    oracle_name: str
    atol: float = 0

    def as_check(
        self,
        *,
        candidate_name: str,
        assertion_name: str,
    ) -> SingleCaseCheck:
        expected_expr = f"{self.oracle_name}(*{self.args!r})"
        return SingleCaseCheck(
            case_id=self.case_id,
            input_repr=repr(self.args),
            expected_output_repr=expected_expr,
            actual_output_expr=f"{candidate_name}(*{self.args!r})",
            expected_output_expr=expected_expr,
            code=(
                f"{assertion_name}("
                f"{candidate_name}(*{self.args!r}), "
                f"{expected_expr}, {self.atol!r})"
            ),
        )


class InputExpressionTestCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal[HumanEvalTestCaseKind.INPUT_EXPRESSION] = (
        HumanEvalTestCaseKind.INPUT_EXPRESSION
    )
    case_id: str
    args: list[Any]
    expected: Any
    expression: str
    input_name: str
    expected_name: str
    index_name: str | None = None

    def as_check(
        self,
        *,
        candidate_name: str,
        assertion_name: str,
    ) -> SingleCaseCheck:
        _ = assertion_name
        lines = []
        if self.index_name is not None:
            index = int(self.case_id.rsplit("_", maxsplit=1)[-1])
            lines.append(f"{self.index_name} = {index!r}")
        lines.extend(
            [
                f"{self.input_name} = {self.args!r}",
                f"{self.expected_name} = {self.expected!r}",
                f"candidate = {candidate_name}",
                self.expression,
            ]
        )
        return SingleCaseCheck(
            case_id=self.case_id,
            input_repr=repr(self.args),
            expected_output_repr=repr(self.expected),
            actual_output_expr=f"{candidate_name}(*{self.args!r})",
            code="\n".join(lines),
        )


TestCase = Annotated[
    InputResultTestCase | InputOracleTestCase | InputExpressionTestCase,
    Field(discriminator="kind"),
]


class ParsedTests(BaseModel):
    model_config = ConfigDict(extra="forbid")

    test_type: HumanEvalTestCaseKind
    support_code: str
    check_name: str
    candidate_arg_name: str
    assertion_name: str
    cases: list[TestCase]
    original_test: str

    def iter_checks(
        self,
        *,
        candidate_name: str = "candidate",
    ) -> Iterator[SingleCaseCheck]:
        for case in self.cases:
            yield case.as_check(
                candidate_name=candidate_name,
                assertion_name=self.assertion_name,
            )

    def to_summary(self) -> ParsedTestsSummary:
        return ParsedTestsSummary(
            test_type=self.test_type,
            support_code=self.support_code,
            check_name=self.check_name,
            candidate_arg_name=self.candidate_arg_name,
            assertion_name=self.assertion_name,
            cases=[
                ParsedTestCaseSummary.from_case(
                    case,
                    assertion_name=self.assertion_name,
                )
                for case in self.cases
            ],
            original_test=self.original_test,
        )


class ParsedTestCaseSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: HumanEvalTestCaseKind
    case_id: str
    input_repr: str = ""
    expected_output_repr: str = ""
    actual_output_expr: str = ""
    expected_output_expr: str | None = None

    @classmethod
    def from_case(
        cls,
        case: TestCase,
        *,
        assertion_name: str,
    ) -> ParsedTestCaseSummary:
        check = case.as_check(
            candidate_name="candidate",
            assertion_name=assertion_name,
        )
        return cls(
            kind=case.kind,
            case_id=case.case_id,
            input_repr=check.input_repr,
            expected_output_repr=check.expected_output_repr,
            actual_output_expr=check.actual_output_expr,
            expected_output_expr=check.expected_output_expr,
        )


class ParsedTestsSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    test_type: HumanEvalTestCaseKind
    support_code: str
    check_name: str
    candidate_arg_name: str
    assertion_name: str
    cases: list[ParsedTestCaseSummary]
    original_test: str


def literal_assignment(function_node: ast.FunctionDef, name: str) -> Any:
    value = find_assignment_value(function_node, name)
    if value is None:
        raise UnsupportedTestFormatError(
            f"Could not find assignment for {name!r}"
        )
    try:
        return ast.literal_eval(value)
    except ValueError as exc:
        raise UnsupportedTestFormatError(
            f"Assignment for {name!r} is not a literal"
        ) from exc


def find_assignment_value(
    function_node: ast.FunctionDef,
    name: str,
) -> ast.expr | None:
    for stmt in function_node.body:
        if not isinstance(stmt, ast.Assign):
            continue
        for target in stmt.targets:
            if isinstance(target, ast.Name) and target.id == name:
                return stmt.value
    return None


def find_for_loop(function_node: ast.FunctionDef) -> ast.For:
    for stmt in function_node.body:
        if isinstance(stmt, ast.For):
            return stmt
    raise UnsupportedTestFormatError(
        f"{function_node.name} does not contain a for loop"
    )


def find_assertion_call(function_node: ast.FunctionDef) -> ast.Call | None:
    for node in ast.walk(function_node):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "assertion"
        ):
            return node
    return None


def find_assert_statement(function_node: ast.FunctionDef) -> ast.Assert:
    for node in ast.walk(function_node):
        if isinstance(node, ast.Assert):
            return node
    raise UnsupportedTestFormatError(
        f"{function_node.name} does not contain assert ..."
    )


def find_oracle_name(assertion_call: ast.Call) -> str | None:
    if len(assertion_call.args) <= EXPECTED_ARG_INDEX:
        return None
    expected_expr = assertion_call.args[EXPECTED_ARG_INDEX]
    if isinstance(expected_expr, ast.Call) and isinstance(
        expected_expr.func, ast.Name
    ):
        return expected_expr.func.id
    return None


def assertion_tolerance(assertion_call: ast.Call) -> float:
    if len(assertion_call.args) <= TOLERANCE_ARG_INDEX:
        return 0
    value = assertion_call.args[TOLERANCE_ARG_INDEX]
    try:
        tolerance = ast.literal_eval(value)
    except ValueError as exc:
        raise UnsupportedTestFormatError(
            "Assertion tolerance is not a literal"
        ) from exc
    if isinstance(tolerance, int | float):
        return float(tolerance)
    raise UnsupportedTestFormatError("Assertion tolerance must be numeric")


def for_loop_names(loop_node: ast.For) -> tuple[str | None, str, str]:
    target = loop_node.target
    if (
        isinstance(target, ast.Tuple)
        and len(target.elts) == PAIR_TARGET_SIZE
        and isinstance(target.elts[0], ast.Name)
        and isinstance(target.elts[1], ast.Tuple)
        and len(target.elts[1].elts) == PAIR_TARGET_SIZE
        and isinstance(target.elts[1].elts[0], ast.Name)
        and isinstance(target.elts[1].elts[1], ast.Name)
    ):
        return (
            target.elts[0].id,
            target.elts[1].elts[0].id,
            target.elts[1].elts[1].id,
        )
    if (
        isinstance(target, ast.Tuple)
        and len(target.elts) == PAIR_TARGET_SIZE
        and isinstance(target.elts[0], ast.Name)
        and isinstance(target.elts[1], ast.Name)
    ):
        return (None, target.elts[0].id, target.elts[1].id)
    raise UnsupportedTestFormatError("Unsupported for-loop target shape")
