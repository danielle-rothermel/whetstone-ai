from __future__ import annotations

from dr_dspy.human_eval import HumanEvalTask
from dr_dspy.scoring import score_humaneval_prediction

TASK = HumanEvalTask(
    task_id="add/0",
    prompt='def add(a, b):\n    """Add two numbers."""\n',
    canonical_solution="def add(a, b):\n    return a + b\n",
    test=(
        "def check(candidate):\n"
        "    inputs = [[1, 2], [0, 0]]\n"
        "    results = [3, 0]\n"
        "    for inp, expected in zip(inputs, results):\n"
        "        assert candidate(*inp) == expected\n"
    ),
    entry_point="add",
)
GROUND_TRUTH = "def add(a, b):\n    return a + b\n"


def _score(raw_generation: str):
    return score_humaneval_prediction(
        prediction_id="p1",
        raw_generation=raw_generation,
        task=TASK,
        compression_input="a description of add",
        ground_truth_code=GROUND_TRUTH,
        timeout=15.0,
    )


def test_correct_solution_scores_one() -> None:
    result = _score("def add(a, b):\n    return a + b\n")
    assert result.score == 1.0
    assert result.raw_compile_ok is True
    assert result.extracted_compile_ok is True
    assert result.error is None


def test_wrong_solution_scores_zero() -> None:
    result = _score("def add(a, b):\n    return a - b\n")
    assert result.score == 0.0
    assert result.raw_compile_ok is True
    assert result.error is not None


def test_uncompilable_generation_scores_zero() -> None:
    result = _score("def add(a, b): return a +")
    assert result.score == 0.0
    assert result.extracted_compile_ok is False


def test_empty_generation_scores_zero() -> None:
    result = _score("   ")
    assert result.score == 0.0
    assert result.extraction_error is not None


def test_compression_metrics_present() -> None:
    result = _score("def add(a, b):\n    return a + b\n")
    assert result.compression_metrics
