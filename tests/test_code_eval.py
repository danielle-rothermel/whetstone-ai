from __future__ import annotations

import time

import dspy

from dr_dspy.code_extraction import apply_cleaning, validate_python_source
from dr_dspy.code_eval import extract_dspy_code, run_python_check


def test_run_python_check_scores_pass_fail_and_errors() -> None:
    cases = [
        (
            "def add(a, b):\n    return a + b\n",
            "def check(candidate):\n    assert candidate(1,2)==3\n",
            "add",
            1.0,
            None,
        ),
        (
            "def add(a, b):\n    return a - b\n",
            "def check(candidate):\n    assert candidate(1,2)==3\n",
            "add",
            0.0,
            "AssertionError",
        ),
        (
            "def add(a, b: return a+b\n",
            "def check(candidate):\n    assert candidate(1,2)==3\n",
            "add",
            0.0,
            "SyntaxError",
        ),
        (
            "def add(a, b):\n    raise ValueError('x')\n",
            "def check(candidate):\n    candidate(1,2)\n",
            "add",
            0.0,
            "ValueError",
        ),
    ]

    for code, test, entry_point, expected_score, expected_error in cases:
        result = run_python_check(
            code=code,
            test=test,
            entry_point=entry_point,
            timeout=5.0,
        )
        assert result.score == expected_score
        if expected_error is not None:
            assert result.error is not None
            assert expected_error in result.error


def test_run_python_check_timeout() -> None:
    start = time.time()
    result = run_python_check(
        code="def loop():\n    while True: pass\n",
        test="def check(candidate):\n    candidate()\n",
        entry_point="loop",
        timeout=2.0,
        cpu_limit_seconds=3,
    )
    wall = time.time() - start

    assert result.score == 0.0
    assert result.error is not None
    assert "timeout" in result.error
    assert wall < 6.0


def test_run_python_check_captures_child_stdout_stderr_without_leaking(
    capsys,
) -> None:
    result = run_python_check(
        code=(
            "import sys\n"
            "def noisy():\n"
            "    print('out')\n"
            "    print('err', file=sys.stderr)\n"
        ),
        test="def check(candidate):\n    candidate()\n",
        entry_point="noisy",
        timeout=5.0,
    )

    captured = capsys.readouterr()
    assert result.score == 1.0
    assert result.stdout == "out\n"
    assert result.stderr == "err\n"
    assert captured.out == ""
    assert captured.err == ""


def test_run_python_check_bounds_captured_stdout_stderr() -> None:
    result = run_python_check(
        code=(
            "import sys\n"
            "def noisy():\n"
            "    print('x' * 5000)\n"
            "    print('y' * 5000, file=sys.stderr)\n"
        ),
        test="def check(candidate):\n    candidate()\n",
        entry_point="noisy",
        timeout=5.0,
        capture_limit_bytes=4096,
    )

    assert result.score == 1.0
    assert result.stdout == "x" * 4096
    assert result.stderr == "y" * 4096
    assert result.stdout_truncated is True
    assert result.stderr_truncated is True


def test_extract_dspy_code_handles_common_prediction_shapes() -> None:
    prediction = dspy.Prediction(code="x = 1")
    assert extract_dspy_code(prediction) == "x = 1"

    class CodeObject:
        code = "def f():\n    return 1\n"

    class PredictionObject:
        code = CodeObject()

    assert extract_dspy_code(PredictionObject()) == "def f():\n    return 1\n"


def test_apply_cleaning_extracts_fenced_python() -> None:
    candidates = apply_cleaning(
        "Here is the solution:\n```python\ndef add(a, b):\n"
        "    return a + b\n```"
    )

    assert candidates == ["def add(a, b):\n    return a + b"]


def test_apply_cleaning_extracts_prose_plus_code() -> None:
    candidates = apply_cleaning(
        "We can implement it directly.\n\n"
        "def add(a, b):\n"
        "    return a + b\n"
    )

    assert candidates == ["def add(a, b):\n    return a + b"]


def test_apply_cleaning_handles_open_fence() -> None:
    candidates = apply_cleaning(
        "```python\ndef add(a, b):\n    return a + b\n"
    )

    assert candidates == ["def add(a, b):\n    return a + b"]


def test_apply_cleaning_handles_markdown_wrapped_code() -> None:
    candidates = apply_cleaning(
        "- def add(a, b):\n-     return a + b\n"
    )

    assert candidates == ["def add(a, b):\n    return a + b"]


def test_apply_cleaning_infers_missing_imports() -> None:
    candidates = apply_cleaning(
        "def root(x):\n    return math.sqrt(x)\n"
    )

    assert candidates == [
        "import math\ndef root(x):\n    return math.sqrt(x)"
    ]


def test_validate_python_source_reports_no_compilable_candidate() -> None:
    candidates = apply_cleaning("This is not Python.")

    assert candidates == []
    validation = validate_python_source("def broken(:\n    pass\n")
    assert validation.compile_ok is False
    assert validation.compile_error is not None
    assert "SyntaxError" in validation.compile_error
