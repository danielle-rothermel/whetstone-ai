from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from dr_dspy.humaneval.code_extraction import (
    apply_cleaning,
    validate_python_source,
)
from dr_dspy.humaneval.compression import (
    CompressionMethod,
    compression_metrics,
)
from dr_dspy.humaneval.parsed_code import ParsedCode, ParsedCodeSummary
from dr_dspy.humaneval.parsed_tests import HumanEvalTestCaseKind
from dr_dspy.humaneval.sampling import sample_human_eval_tasks_from_rows
from dr_dspy.humaneval.scoring import (
    GeneratedCodeOutcome,
    score_generated_code_for_humaneval,
    score_humaneval_prediction,
)
from dr_dspy.humaneval.task import (
    EvaluationCaseStatus,
    HumanEvalTask,
    evaluate_human_eval_code,
    parse_human_eval_tests,
    run_subprocess_batch,
)


def _task(*, test: str | None = None) -> HumanEvalTask:
    return HumanEvalTask(
        task_id="HumanEval/fixture",
        prompt="def add_one(x):\n",
        canonical_solution="    return x + 1\n",
        entry_point="add_one",
        test=test or _input_result_test(),
    )


def _row(task_id: str, offset: int) -> dict[str, str]:
    return {
        "task_id": task_id,
        "prompt": f"def f_{offset}(x):\n",
        "canonical_solution": f"    return x + {offset}\n",
        "entry_point": f"f_{offset}",
        "test": (
            "def check(candidate):\n"
            "    inputs = [(1,)]\n"
            f"    results = [{1 + offset}]\n"
            "    for inp, expected in zip(inputs, results):\n"
            "        assertion(candidate(*inp), expected)\n"
        ),
    }


def _input_result_test() -> str:
    return (
        "def check(candidate):\n"
        "    inputs = [(1,), (2,)]\n"
        "    results = [2, 3]\n"
        "    for inp, expected in zip(inputs, results):\n"
        "        assertion(candidate(*inp), expected)\n"
    )


class _CompletedProcessStub:
    def __init__(
        self,
        *,
        stdout: str,
        stderr: str = "",
        returncode: int = 0,
    ) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def test_sampling_from_rows_is_deterministic_and_indexed() -> None:
    rows = [_row(f"HumanEval/{index}", index) for index in range(5)]

    first = sample_human_eval_tasks_from_rows(
        rows,
        seed=17,
        sample_count=3,
    )
    second = sample_human_eval_tasks_from_rows(
        rows,
        seed=17,
        sample_count=3,
    )

    assert [sample.sample_index for sample in first] == [0, 1, 2]
    assert [sample.task.task_id for sample in first] == [
        sample.task.task_id for sample in second
    ]
    assert [sample.task.task_id for sample in first] == [
        "HumanEval/0",
        "HumanEval/2",
        "HumanEval/1",
    ]


def test_parse_input_result_tests_have_stable_case_ids() -> None:
    parsed = parse_human_eval_tests(_input_result_test())

    assert parsed.test_type is HumanEvalTestCaseKind.INPUT_RESULT
    assert [case.case_id for case in parsed.cases] == ["case_0", "case_1"]
    assert [case.kind for case in parsed.cases] == [
        HumanEvalTestCaseKind.INPUT_RESULT,
        HumanEvalTestCaseKind.INPUT_RESULT,
    ]
    checks = list(parsed.iter_checks(candidate_name="candidate"))
    assert checks[0].input_repr == "[1]"
    assert "candidate(*[1])" in checks[0].code

    summary = parsed.to_summary()
    assert summary.test_type is HumanEvalTestCaseKind.INPUT_RESULT
    assert [case.case_id for case in summary.cases] == ["case_0", "case_1"]
    assert summary.cases[0].input_repr == "[1]"
    assert "code" not in summary.cases[0].model_dump(mode="json")


def test_parse_oracle_tests_have_expected_expression_metadata() -> None:
    parsed = parse_human_eval_tests(
        "def ref(x):\n"
        "    return x + 1\n"
        "\n"
        "def check(candidate):\n"
        "    inputs = [(1,), (2,)]\n"
        "    for inp in inputs:\n"
        "        assertion(candidate(*inp), ref(*inp))\n"
    )

    assert parsed.test_type is HumanEvalTestCaseKind.INPUT_ORACLE
    assert [case.case_id for case in parsed.cases] == ["case_0", "case_1"]
    checks = list(parsed.iter_checks(candidate_name="candidate"))
    assert checks[0].expected_output_expr == "ref(*[1])"


def test_parse_expression_tests_preserve_indexed_assertion() -> None:
    parsed = parse_human_eval_tests(
        "def check(candidate):\n"
        "    inputs = [(1,), (2,)]\n"
        "    results = [2, 3]\n"
        "    for i, (inp, expected) in enumerate(zip(inputs, results)):\n"
        "        assert candidate(*inp) == expected\n"
    )

    assert parsed.test_type is HumanEvalTestCaseKind.INPUT_EXPRESSION
    checks = list(parsed.iter_checks(candidate_name="candidate"))
    assert checks[1].case_id == "case_1"
    assert "i = 1" in checks[1].code
    assert "assert candidate(*inp) == expected" in checks[1].code


def test_parsed_code_summary_excludes_runtime_ast() -> None:
    parsed = ParsedCode(
        display_title="fixture",
        code_str=(
            'def add_one(x: int) -> int:\n'
            '    """doc"""\n'
            '    return x + 1\n'
        ),
    )

    summary = parsed.to_summary()

    assert isinstance(summary, ParsedCodeSummary)
    assert summary.display_title == "fixture"
    assert summary.signatures[0].function_name == "add_one"
    assert summary.signatures[0].function_args[0].name == "x"
    dumped = summary.model_dump(mode="json")
    assert "tree" not in dumped
    assert "doc" in dumped["comments"]


@pytest.mark.parametrize(
    ("source", "expected_fragment"),
    [
        ("```python\ndef add_one(x):\n    return x + 1\n```", "def add_one"),
        ("> def add_one(x):\n>     return x + 1", "def add_one"),
        (
            "    def add_one(x):\n        return x + 1\n",
            "def add_one",
        ),
        (
            "def add_one(x):\n"
            "    return x + 1\n"
            "print('trailing')\n",
            "return x + 1",
        ),
        (
            "def add_one(x):\n"
            "    return x + 1\n"
            "if __name__ == '__main__':\n"
            "    print(add_one(1))\n",
            "def add_one",
        ),
    ],
)
def test_apply_cleaning_extracts_known_generation_shapes(
    source: str,
    expected_fragment: str,
) -> None:
    candidates = apply_cleaning(source, apply_dedent=True)

    assert candidates
    assert expected_fragment in candidates[0]
    assert validate_python_source(candidates[0]).compile_ok
    assert "if __name__" not in candidates[0]
    assert "print('trailing')" not in candidates[0]


def test_evaluation_passes_when_best_function_passes() -> None:
    result = evaluate_human_eval_code(
        task=_task(),
        candidate_code=(
            "def broken_helper(x):\n"
            "    return x\n"
            "\n"
            "def add_one(x):\n"
            "    return x + 1\n"
        ),
        timeout_seconds=2.0,
    )

    assert result.best_function_name == "add_one"
    assert result.passed is True
    assert result.status_counts == {"passed": 2}
    assert result.failures == []
    summary = result.to_summary()
    assert summary.passed is True
    assert summary.best_function_name == "add_one"
    assert summary.failure_count == 0


def test_score_generated_code_passes_when_best_function_passes() -> None:
    result = score_generated_code_for_humaneval(
        raw_generation=(
            "def broken_helper(x):\n"
            "    return x\n"
            "\n"
            "def add_one(x):\n"
            "    return x + 1\n"
        ),
        task=_task(),
        timeout=2.0,
    )

    assert result.outcome is GeneratedCodeOutcome.PASSED
    assert result.score == 1.0
    assert result.evaluation is not None
    assert result.evaluation.best_function_name == "add_one"


def test_evaluation_prefers_entry_point_when_pass_counts_tie() -> None:
    result = evaluate_human_eval_code(
        task=_task(),
        candidate_code=(
            "def add_one(x):\n"
            "    return x + 1\n"
            "\n"
            "def also_add_one(x):\n"
            "    return x + 1\n"
        ),
        timeout_seconds=2.0,
    )

    assert result.best_function_name == "add_one"
    assert result.passed is True


def test_evaluation_fails_when_best_function_does_not_pass_all_cases() -> None:
    result = evaluate_human_eval_code(
        task=_task(),
        candidate_code=(
            "def broken_helper(x):\n"
            "    return x\n"
            "\n"
            "def add_one(x):\n"
            "    return x + 1 if x == 1 else x\n"
        ),
        timeout_seconds=2.0,
    )

    assert result.best_function_name == "add_one"
    assert result.passed is False
    assert result.status_counts == {"passed": 1, "failed": 1}


def test_evaluation_uses_highest_pass_count() -> None:
    result = evaluate_human_eval_code(
        task=_task(),
        candidate_code=(
            "def add_one(x):\n"
            "    return x\n"
            "\n"
            "def helper(x):\n"
            "    return x + 1\n"
        ),
        timeout_seconds=2.0,
    )

    assert result.best_function_name == "helper"
    assert result.passed is True
    assert result.status_counts == {"passed": 2}


def test_score_generated_code_passes_humaneval_task() -> None:
    result = score_generated_code_for_humaneval(
        raw_generation="def add_one(x):\n    return x + 1\n",
        task=_task(),
        timeout=2.0,
    )

    assert result.outcome is GeneratedCodeOutcome.PASSED
    assert result.score == 1.0
    assert result.error is None
    assert result.evaluation is not None
    assert result.evaluation.status_counts == {"passed": 2}
    summary = result.evaluation.to_summary()
    assert summary.passed is True
    assert summary.failure_count == 0
    assert summary.results[0].status is EvaluationCaseStatus.PASSED
    assert "failures" not in summary.model_dump(mode="json")


def test_score_humaneval_prediction_flattens_score_and_compression() -> None:
    result = score_humaneval_prediction(
        prediction_id="prediction-1",
        raw_generation="def add_one(x):\n    return x + 1\n",
        task=_task(),
        compression_input="short description",
        ground_truth_code="def add_one(x):\n    return x + 1\n",
        timeout=2.0,
    )

    assert result.prediction_id == "prediction-1"
    assert result.generated_code_outcome is GeneratedCodeOutcome.PASSED
    assert result.score == 1.0
    assert result.error is None
    assert result.raw_code == "def add_one(x):\n    return x + 1"
    assert result.raw_compile_ok is True
    assert result.extracted_compile_ok is True
    assert result.evaluation_function_names == ["add_one"]
    assert result.evaluation_total_cases == 2
    assert result.evaluation_failure_count == 0
    assert result.evaluation_status_counts == {"passed": 2}
    assert result.evaluation_summary is not None
    assert result.evaluation_summary.passed is True
    assert len(result.evaluation_summary.results) == 2
    assert set(result.compression_metrics) == set(CompressionMethod)
    assert result.raw_compression_ratio is not None
    assert result.best_compression_ratio is not None
    assert result.best_compression_percent_reduction is not None


def test_score_generated_code_reports_wrong_answer_as_domain_result() -> None:
    result = score_generated_code_for_humaneval(
        raw_generation="def add_one(x):\n    return x\n",
        task=_task(),
        timeout=2.0,
    )

    assert result.outcome is GeneratedCodeOutcome.TESTS_FAILED
    assert result.score == 0.0
    assert result.error == "HumanEval tests failed"
    assert result.evaluation is not None
    assert result.evaluation.failures[0].status is EvaluationCaseStatus.FAILED


def test_score_generated_code_reports_runtime_error_as_domain_result() -> None:
    result = score_generated_code_for_humaneval(
        raw_generation=(
            "def add_one(x):\n"
            "    raise RuntimeError('boom')\n"
        ),
        task=_task(),
        timeout=2.0,
    )

    assert result.outcome is GeneratedCodeOutcome.TESTS_FAILED
    assert result.evaluation is not None
    assert result.evaluation.failures[0].status is EvaluationCaseStatus.ERROR
    assert "RuntimeError" in result.evaluation.failures[0].message


def test_score_generated_code_reports_empty_generation() -> None:
    result = score_generated_code_for_humaneval(
        raw_generation="   ",
        task=_task(),
        timeout=2.0,
    )

    assert result.outcome is GeneratedCodeOutcome.EMPTY_GENERATION
    assert result.score == 0.0
    assert result.extraction_candidate_count == 0
    assert result.extraction_error == "empty raw generation"


def test_score_generated_code_reports_invalid_generated_code() -> None:
    result = score_generated_code_for_humaneval(
        raw_generation="```python\ndef add_one(x)\n    return x + 1\n```",
        task=_task(),
        timeout=2.0,
    )

    assert result.outcome is GeneratedCodeOutcome.EXTRACTION_FAILED
    assert result.score == 0.0
    assert result.extracted_compile_ok is False
    assert result.extracted_compile_error is not None


def test_score_generated_code_reports_no_top_level_functions() -> None:
    result = score_generated_code_for_humaneval(
        raw_generation="answer = 2\n",
        task=_task(),
        timeout=2.0,
    )

    assert result.outcome is GeneratedCodeOutcome.NO_TOP_LEVEL_FUNCTIONS
    assert result.score == 0.0
    assert result.error == "no top-level candidate functions"
    assert result.evaluation is not None
    assert result.evaluation.function_names == []


def test_evaluate_humaneval_code_reports_timeout_per_case() -> None:
    result = evaluate_human_eval_code(
        task=_task(),
        candidate_code=(
            "def add_one(x):\n"
            "    while True:\n"
            "        pass\n"
        ),
        timeout_seconds=0.2,
    )

    assert result.passed is False
    assert result.status_counts == {"timeout": 2}
    assert {case.case_id for case in result.results} == {"case_0", "case_1"}


def test_run_subprocess_batch_maps_malformed_runner_output_to_errors() -> None:
    def fake_run(*args: Any, **kwargs: Any) -> _CompletedProcessStub:
        return _CompletedProcessStub(
            stdout=(
                '[{"case_id": "case_0", "status": "passed", "message": ""}, '
                '{"case_id": "case_1", "status": "nonsense"}]'
            ),
        )

    with patch("dr_dspy.humaneval.task.subprocess.run", fake_run):
        results = run_subprocess_batch(
            task=_task(),
            candidate_code="def add_one(x):\n    return x + 1\n",
            function_name="add_one",
            timeout_seconds=2.0,
        )

    by_case_id = {result.case_id: result for result in results}
    assert set(by_case_id) == {"case_0", "case_1"}
    assert by_case_id["case_0"].status is EvaluationCaseStatus.PASSED
    assert by_case_id["case_1"].status is EvaluationCaseStatus.ERROR
    assert "Invalid runner output" in by_case_id["case_1"].message


_PARTIAL_RUNNER_PASSED_CASE_0 = (
    '[{"case_id": "case_0", "status": "passed", "message": ""}]'
)


def test_evaluation_incomplete_when_runner_returns_partial_results() -> None:
    def fake_run(*args: Any, **kwargs: Any) -> _CompletedProcessStub:
        return _CompletedProcessStub(stdout=_PARTIAL_RUNNER_PASSED_CASE_0)

    with patch("dr_dspy.humaneval.task.subprocess.run", fake_run):
        result = evaluate_human_eval_code(
            task=_task(),
            candidate_code="def add_one(x):\n    return x + 1\n",
            timeout_seconds=2.0,
        )

    assert result.passed is False
    assert result.coverage_complete is False
    assert result.failures == []
    assert result.status_counts == {"passed": 1}


def test_score_generated_code_reports_incomplete_runner_output() -> None:
    def fake_run(*args: Any, **kwargs: Any) -> _CompletedProcessStub:
        return _CompletedProcessStub(stdout=_PARTIAL_RUNNER_PASSED_CASE_0)

    with patch("dr_dspy.humaneval.task.subprocess.run", fake_run):
        result = score_generated_code_for_humaneval(
            raw_generation="def add_one(x):\n    return x + 1\n",
            task=_task(),
            timeout=2.0,
        )

    assert result.outcome is GeneratedCodeOutcome.EVALUATION_INCOMPLETE
    assert result.score == 0.0
    assert result.error == "HumanEval evaluation incomplete"
    assert result.evaluation is not None
    assert result.evaluation.failures == []
    assert result.evaluation.coverage_complete is False


def test_score_generated_code_reports_test_failure_when_case_fails() -> None:
    def fake_run(*args: Any, **kwargs: Any) -> _CompletedProcessStub:
        return _CompletedProcessStub(
            stdout=(
                '[{"case_id": "case_0", "status": "failed", '
                '"message": "bad"}, '
                '{"case_id": "case_1", "status": "passed", "message": ""}]'
            ),
        )

    with patch("dr_dspy.humaneval.task.subprocess.run", fake_run):
        result = score_generated_code_for_humaneval(
            raw_generation="def add_one(x):\n    return x + 1\n",
            task=_task(),
            timeout=2.0,
        )

    assert result.outcome is GeneratedCodeOutcome.TESTS_FAILED
    assert result.error == "HumanEval tests failed"
    assert result.evaluation is not None
    assert result.evaluation.failures[0].status is EvaluationCaseStatus.FAILED


def test_compression_metrics_are_stable_for_methods_and_ratios() -> None:
    metrics = compression_metrics(
        ground_truth_code="def f():\n    return 1\n",
        representation_text="return 1",
    )

    assert set(metrics) == set(CompressionMethod)
    raw = metrics[CompressionMethod.RAW]
    assert raw.ground_truth_bytes == len(b"def f():\n    return 1\n")
    assert raw.representation_bytes == len(b"return 1")
    assert raw.compressed_bytes == raw.representation_bytes
    assert raw.ratio_to_ground_truth == pytest.approx(
        raw.representation_bytes / raw.ground_truth_bytes
    )


def test_compression_metrics_keep_empty_ground_truth_ratio_null() -> None:
    metrics = compression_metrics(
        ground_truth_code="",
        representation_text="return 1",
    )

    assert all(
        metric.ratio_to_ground_truth is None
        for metric in metrics.values()
    )
    assert all(
        metric.percent_reduction_vs_ground_truth is None
        for metric in metrics.values()
    )
