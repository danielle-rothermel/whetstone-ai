from __future__ import annotations

import ast
import json
import subprocess
import sys
import textwrap
from collections import Counter
from collections.abc import Iterable, Mapping
from enum import StrEnum
from typing import Any, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    computed_field,
    model_validator,
)

from dr_dspy.humaneval.parsed_code import FunctionSignature, ParsedCode
from dr_dspy.humaneval.parsed_tests import (
    HumanEvalTestCaseKind,
    InputExpressionTestCase,
    InputOracleTestCase,
    InputResultTestCase,
    ParsedTests,
    SingleCaseCheck,
    TestCase,
    UnsupportedTestFormatError,
    assertion_tolerance,
    find_assert_statement,
    find_assertion_call,
    find_assignment_value,
    find_for_loop,
    find_oracle_name,
    for_loop_names,
    literal_assignment,
)


class EvaluationCaseStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    TIMEOUT = "timeout"


class HumanEvalTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    prompt: str
    canonical_solution: str
    entry_point: str
    test: str
    notes: list[str] = Field(default_factory=list)
    parsed: ParsedCode | None = None
    parsed_tests: ParsedTests | None = None

    @computed_field
    @property
    def ground_truth_code(self) -> str:
        return self.prompt + self.canonical_solution

    @computed_field
    @property
    def ground_truth_code_without_comments(self) -> str | None:
        if self.parsed is None:
            return None
        return self.parsed.code_without_comments

    @model_validator(mode="after")
    def parse_code(self) -> Self:
        if self.parsed is None:
            self.parsed = ParsedCode(
                display_title=self.task_id,
                code_str=self.ground_truth_code,
            )
        if self.parsed_tests is None:
            self.parsed_tests = parse_human_eval_tests(self.test)
        return self


class EvaluationCaseResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    case_id: str
    function_name: str
    status: EvaluationCaseStatus
    message: str = ""
    test_type: HumanEvalTestCaseKind
    input_repr: str = ""
    expected_output_repr: str = ""
    actual_output_repr: str = ""

    def to_summary(self) -> EvaluationCaseSummary:
        return EvaluationCaseSummary(
            task_id=self.task_id,
            case_id=self.case_id,
            function_name=self.function_name,
            status=self.status,
            message=self.message,
            test_type=self.test_type,
            input_repr=self.input_repr,
            expected_output_repr=self.expected_output_repr,
            actual_output_repr=self.actual_output_repr,
        )


class EvaluationCaseSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    case_id: str
    function_name: str
    status: EvaluationCaseStatus
    message: str = ""
    test_type: HumanEvalTestCaseKind
    input_repr: str = ""
    expected_output_repr: str = ""
    actual_output_repr: str = ""


class EvaluationTaskResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    entry_point: str
    function_names: list[str]
    total_cases: int
    results: list[EvaluationCaseResult] = Field(default_factory=list)

    @computed_field
    @property
    def failures(self) -> list[EvaluationCaseResult]:
        return [
            result
            for result in self.results
            if result.status is not EvaluationCaseStatus.PASSED
        ]

    @computed_field
    @property
    def passed(self) -> bool:
        if not self.function_names:
            return False
        return any(
            all(
                result.status is EvaluationCaseStatus.PASSED
                for result in self.results
                if result.function_name == function_name
            )
            for function_name in self.function_names
        )

    @computed_field
    @property
    def status_counts(self) -> dict[str, int]:
        return dict(Counter(result.status.value for result in self.results))

    def to_summary(self) -> EvaluationTaskSummary:
        return EvaluationTaskSummary(
            task_id=self.task_id,
            entry_point=self.entry_point,
            function_names=self.function_names,
            total_cases=self.total_cases,
            results=[result.to_summary() for result in self.results],
            passed=self.passed,
            failure_count=len(self.failures),
            status_counts=self.status_counts,
        )


class EvaluationTaskSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    entry_point: str
    function_names: list[str]
    total_cases: int
    results: list[EvaluationCaseSummary] = Field(default_factory=list)
    passed: bool
    failure_count: int
    status_counts: dict[str, int] = Field(default_factory=dict)


class HumanEvalRunnerPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    candidate_code: str
    support_code: str
    function_name: str
    test_type: HumanEvalTestCaseKind
    checks: list[SingleCaseCheck]


class HumanEvalRunnerCaseOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    status: EvaluationCaseStatus
    message: str = ""
    input_repr: str = ""
    expected_output_repr: str = ""
    actual_output_repr: str = ""


class HumanEvalOverride(BaseModel):
    model_config = ConfigDict(extra="forbid")

    notes: list[str] = Field(default_factory=list)
    canonical_solution: str | None = None
    test_replacements: dict[str, str] = Field(default_factory=dict)


HUMAN_EVAL_OVERRIDES: dict[str, HumanEvalOverride] = {
    "HumanEval/32": HumanEvalOverride(
        notes=[
            "Fixed the benchmark test assertion to evaluate the polynomial at "
            "the returned root with a scaled residual tolerance, and replaced "
            "the Newton-only canonical solution with a hybrid "
            "Newton/bisection method."
        ],
        canonical_solution="""

    dxs = [xs[i] * i for i in range(1, len(xs))]

    def func(x):
        return poly(xs, x)

    def derivative(x):
        return poly(dxs, x)

    x = 0.0
    last_step = None
    for _ in range(1000):
        fx = func(x)
        dfx = derivative(x)
        if abs(fx) < 1e-5:
            return x
        if dfx == 0:
            break
        last_step = fx / dfx
        x = x - last_step

    if last_step is not None and abs(last_step) <= 1e-7 * max(1.0, abs(x)):
        return x

    lo, hi = -1.0, 1.0
    flo, fhi = func(lo), func(hi)
    for _ in range(200):
        if flo == 0:
            return lo
        if fhi == 0:
            return hi
        if (flo < 0 < fhi) or (fhi < 0 < flo):
            break
        lo *= 2.0
        hi *= 2.0
        flo, fhi = func(lo), func(hi)

    for _ in range(200):
        mid = (lo + hi) / 2.0
        fm = func(mid)
        if fm == 0 or abs(hi - lo) <= 1e-12 * max(1.0, abs(mid)):
            return mid
        if (flo < 0 < fm) or (fm < 0 < flo):
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm

    return (lo + hi) / 2.0
""",
        test_replacements={
            "assert _poly(*candidate(*inp), inp) <= 0.0001": (
                "assert abs(_poly(*inp, (out := candidate(*inp)))) <= max("
                "1e-4, "
                "1e-12 * sum("
                "abs(coeff) * max(1.0, abs(out)) ** j "
                "for j, coeff in enumerate(inp[0])"
                ")"
                ")"
            ),
        },
    ),
}


def parse_human_eval_tests(test_str: str) -> ParsedTests:
    tree = ast.parse(test_str)
    check_node = find_check_function(tree)
    if len(check_node.args.args) != 1:
        raise UnsupportedTestFormatError(
            "Expected check(candidate) with one positional argument"
        )

    inputs = literal_assignment(check_node, "inputs")
    results_value = find_assignment_value(check_node, "results")
    assertion_call = find_assertion_call(check_node)
    tolerance = assertion_tolerance(assertion_call) if assertion_call else 0
    support_code = support_code_without_check(tree)
    candidate_arg_name = check_node.args.args[0].arg

    cases: list[TestCase]
    if results_value is not None:
        results = literal_assignment(check_node, "results")
        if len(inputs) != len(results):
            raise UnsupportedTestFormatError(
                f"len(inputs)={len(inputs)} does not match "
                f"len(results)={len(results)}"
            )
        if assertion_call is None:
            loop_node = find_for_loop(check_node)
            index_name, input_name, expected_name = for_loop_names(loop_node)
            assert_statement = find_assert_statement(check_node)
            cases = [
                InputExpressionTestCase(
                    case_id=f"case_{index}",
                    args=args,
                    expected=expected,
                    expression=ast.unparse(assert_statement),
                    input_name=input_name,
                    expected_name=expected_name,
                    index_name=index_name,
                )
                for index, (args, expected) in enumerate(
                    zip(inputs, results, strict=True)
                )
            ]
            test_type = HumanEvalTestCaseKind.INPUT_EXPRESSION
        else:
            cases = [
                InputResultTestCase(
                    case_id=f"case_{index}",
                    args=args,
                    expected=expected,
                    atol=tolerance,
                )
                for index, (args, expected) in enumerate(
                    zip(inputs, results, strict=True)
                )
            ]
            test_type = HumanEvalTestCaseKind.INPUT_RESULT
    else:
        _ = find_for_loop(check_node)
        if assertion_call is None:
            raise UnsupportedTestFormatError(
                "Expected assertion(..., ref_func(*inp), ...) for oracle tests"
            )
        oracle_name = find_oracle_name(assertion_call)
        if oracle_name is None:
            raise UnsupportedTestFormatError(
                "Expected assertion(..., ref_func(*inp), ...) for oracle tests"
            )
        cases = [
            InputOracleTestCase(
                case_id=f"case_{index}",
                args=args,
                oracle_name=oracle_name,
                atol=tolerance,
            )
            for index, args in enumerate(inputs)
        ]
        test_type = HumanEvalTestCaseKind.INPUT_ORACLE

    return ParsedTests(
        test_type=test_type,
        support_code=support_code,
        check_name=check_node.name,
        candidate_arg_name=candidate_arg_name,
        assertion_name="assertion",
        cases=cases,
        original_test=test_str,
    )


def parse_human_eval_dataset(
    rows: Iterable[Mapping[str, Any]],
    *,
    overrides: dict[str, HumanEvalOverride] | None = None,
) -> list[HumanEvalTask]:
    active_overrides = HUMAN_EVAL_OVERRIDES if overrides is None else overrides
    return [
        HumanEvalTask(**apply_human_eval_override(row, active_overrides))
        for row in rows
    ]


def apply_human_eval_override(
    row: Mapping[str, Any],
    overrides: dict[str, HumanEvalOverride],
) -> dict[str, Any]:
    task_id = str(row["task_id"])
    override = overrides.get(task_id)
    if override is None:
        return dict(row)

    updated = dict(row)
    if override.canonical_solution is not None:
        updated["canonical_solution"] = override.canonical_solution
    test = str(updated["test"])
    for old, new in override.test_replacements.items():
        if old not in test:
            raise ValueError(
                f"Override replacement text not found for {task_id}"
            )
        test = test.replace(old, new, 1)
    updated["test"] = test
    updated["notes"] = [*updated.get("notes", []), *override.notes]
    return updated


def evaluate_human_eval_code(
    *,
    task: HumanEvalTask,
    candidate_code: str,
    timeout_seconds: float,
) -> EvaluationTaskResult:
    parsed_tests = require_parsed_tests(task)
    function_names = top_level_function_names(candidate_code)
    results: list[EvaluationCaseResult] = []
    for function_name in function_names:
        results.extend(
            run_subprocess_batch(
                task=task,
                candidate_code=candidate_code,
                function_name=function_name,
                timeout_seconds=timeout_seconds,
            )
        )
    return EvaluationTaskResult(
        task_id=task.task_id,
        entry_point=task.entry_point,
        function_names=function_names,
        total_cases=len(parsed_tests.cases),
        results=results,
    )


def find_check_function(tree: ast.Module) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "check":
            return node
    raise UnsupportedTestFormatError(
        "Could not find check(candidate) function"
    )


def support_code_without_check(tree: ast.Module) -> str:
    support_nodes = [
        node
        for node in tree.body
        if not (isinstance(node, ast.FunctionDef) and node.name == "check")
    ]
    module = ast.Module(body=support_nodes, type_ignores=[])
    return ast.unparse(module)


def top_level_function_names(code_str: str) -> list[str]:
    tree = ast.parse(code_str)
    return [
        FunctionSignature(tree=node).function_name
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    ]


def run_subprocess_batch(
    *,
    task: HumanEvalTask,
    candidate_code: str,
    function_name: str,
    timeout_seconds: float,
) -> list[EvaluationCaseResult]:
    parsed_tests = require_parsed_tests(task)
    payload = HumanEvalRunnerPayload(
        task_id=task.task_id,
        candidate_code=candidate_code,
        support_code=parsed_tests.support_code,
        function_name=function_name,
        test_type=parsed_tests.test_type,
        checks=list(parsed_tests.iter_checks(candidate_name="candidate")),
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", runner_script()],
            input=payload.model_dump_json(),
            capture_output=True,
            check=False,
            encoding="utf-8",
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return timeout_results(task=task, function_name=function_name)

    if completed.returncode != 0:
        return error_results(
            task=task,
            function_name=function_name,
            message=completed.stderr.strip() or completed.stdout.strip(),
        )
    try:
        raw_results = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        return error_results(
            task=task,
            function_name=function_name,
            message=f"Could not decode runner output: {exc}",
        )
    try:
        runner_results = TypeAdapter(
            list[HumanEvalRunnerCaseOutput]
        ).validate_python(raw_results)
    except ValidationError as exc:
        return error_results(
            task=task,
            function_name=function_name,
            message=f"Invalid runner output: {exc}",
        )
    return [
        EvaluationCaseResult(
            task_id=task.task_id,
            case_id=result.case_id,
            function_name=function_name,
            status=result.status,
            message=result.message,
            test_type=parsed_tests.test_type,
            input_repr=result.input_repr,
            expected_output_repr=result.expected_output_repr,
            actual_output_repr=result.actual_output_repr,
        )
        for result in runner_results
    ]


def timeout_results(
    *,
    task: HumanEvalTask,
    function_name: str,
) -> list[EvaluationCaseResult]:
    parsed_tests = require_parsed_tests(task)
    return [
        EvaluationCaseResult(
            task_id=task.task_id,
            case_id=case.case_id,
            function_name=function_name,
            status=EvaluationCaseStatus.TIMEOUT,
            message="Batch timed out",
            test_type=parsed_tests.test_type,
            **case_metadata(parsed_tests, case),
        )
        for case in parsed_tests.cases
    ]


def error_results(
    *,
    task: HumanEvalTask,
    function_name: str,
    message: str,
) -> list[EvaluationCaseResult]:
    parsed_tests = require_parsed_tests(task)
    return [
        EvaluationCaseResult(
            task_id=task.task_id,
            case_id=case.case_id,
            function_name=function_name,
            status=EvaluationCaseStatus.ERROR,
            message=message,
            test_type=parsed_tests.test_type,
            **case_metadata(parsed_tests, case),
        )
        for case in parsed_tests.cases
    ]


def case_metadata(
    parsed_tests: ParsedTests,
    case: TestCase,
) -> dict[str, str]:
    check = case.as_check(
        candidate_name="candidate",
        assertion_name=parsed_tests.assertion_name,
    )
    return {
        "input_repr": check.input_repr,
        "expected_output_repr": check.expected_output_repr,
        "actual_output_repr": "",
    }


def require_parsed_tests(task: HumanEvalTask) -> ParsedTests:
    if task.parsed_tests is None:
        raise ValueError("HumanEvalTask.parsed_tests is required")
    return task.parsed_tests


def runner_script() -> str:
    return textwrap.dedent(
        """
        import json
        import traceback

        payload = json.loads(input())

        def assertion(actual, expected, atol=0):
            if atol:
                assert abs(actual - expected) <= atol
            else:
                assert actual == expected

        def build_namespace():
            namespace = {"assertion": assertion}
            exec(payload["support_code"], namespace)
            exec(payload["candidate_code"], namespace)
            return namespace

        def failure_metadata(check):
            metadata = {
                "input_repr": check.get("input_repr", ""),
                "expected_output_repr": check.get("expected_output_repr", ""),
                "actual_output_repr": "",
            }
            try:
                detail_namespace = build_namespace()
                detail_candidate = detail_namespace[payload["function_name"]]
            except Exception:
                metadata["actual_output_repr"] = traceback.format_exc(limit=4)
                return metadata

            try:
                if check.get("actual_output_expr"):
                    metadata["actual_output_repr"] = repr(eval(
                        check["actual_output_expr"],
                        detail_namespace | {"candidate": detail_candidate},
                    ))
            except Exception:
                metadata["actual_output_repr"] = traceback.format_exc(limit=4)

            try:
                if check.get("expected_output_expr"):
                    metadata["expected_output_repr"] = repr(eval(
                        check["expected_output_expr"],
                        detail_namespace | {"candidate": detail_candidate},
                    ))
            except Exception:
                metadata["expected_output_repr"] = traceback.format_exc(
                    limit=4,
                )

            return metadata

        namespace = build_namespace()
        candidate = namespace[payload["function_name"]]
        results = []
        for check in payload["checks"]:
            try:
                exec(
                    compile(
                        check["code"],
                        f"<generated {check['case_id']}>",
                        "exec",
                    ),
                    namespace | {"candidate": candidate},
                )
            except AssertionError as exc:
                results.append({
                    "case_id": check["case_id"],
                    "status": "failed",
                    "message": str(exc),
                    **failure_metadata(check),
                })
            except Exception:
                results.append({
                    "case_id": check["case_id"],
                    "status": "error",
                    "message": traceback.format_exc(limit=4),
                    **failure_metadata(check),
                })
            else:
                results.append({
                    "case_id": check["case_id"],
                    "status": "passed",
                    "message": "",
                    "input_repr": check.get("input_repr", ""),
                    "expected_output_repr": check.get(
                        "expected_output_repr",
                        "",
                    ),
                    "actual_output_repr": "",
                })
        print(json.dumps(results))
        """
    )
