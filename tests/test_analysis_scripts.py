from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts" / "analysis"


def load_analysis_script(module_name: str):
    path = SCRIPTS_DIR / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_estimate_repeat_stability_handles_sparse_data() -> None:
    q3 = load_analysis_script("q3_repeat_stability")
    frame = pd.DataFrame(
        [
            {
                "task_id": "HumanEval/0",
                "model": "model-a",
                "compression_target": 0.25,
                "score_status": "success",
                "score": 1.0,
                "generated_code_outcome": "passed",
            }
        ]
    )
    coverage, intervals = q3.estimate_repeat_stability(frame)
    assert len(coverage) == 1
    assert len(intervals) >= 1


def test_build_task_variation_summary_marks_useful_signal_task() -> None:
    q4 = load_analysis_script("q4_task_variation")
    frame = pd.DataFrame(
        [
            {
                "task_id": "HumanEval/1",
                "model": "model-a",
                "compression_target": 0.25,
                "score_status": "success",
                "score": 1.0,
                "generated_code_outcome": "passed",
            },
            {
                "task_id": "HumanEval/1",
                "model": "model-a",
                "compression_target": 0.5,
                "score_status": "success",
                "score": 0.0,
                "generated_code_outcome": "tests_failed",
            },
            {
                "task_id": "HumanEval/1",
                "model": "model-b",
                "compression_target": 0.75,
                "score_status": "success",
                "score": 0.0,
                "generated_code_outcome": "tests_failed",
            },
        ]
    )
    summary = q4.build_task_variation_summary(frame)
    useful = summary.loc[summary["task_id"] == "HumanEval/1"].iloc[0]
    assert useful["useful_signal"]
    assert useful["scoreable_count"] == 3


def test_build_compression_summary_aggregates_overall_rows() -> None:
    q2 = load_analysis_script("q2_compression_range")
    frame = pd.DataFrame(
        [
            {
                "model": "model-a",
                "compression_target": 0.25,
                "generation_status": "success",
                "score_status": "success",
                "score": 1.0,
                "generated_code_outcome": "passed",
                "realized_compression_ratio": 0.2,
            },
            {
                "model": "model-b",
                "compression_target": 0.25,
                "generation_status": "success",
                "score_status": "success",
                "score": 0.0,
                "generated_code_outcome": "tests_failed",
                "realized_compression_ratio": 0.3,
            },
        ]
    )
    summary = q2.build_compression_summary(frame)
    overall = summary[summary["rollup"] == "overall"].iloc[0]
    assert overall["scoreable_count"] == 2
    assert overall["pass_count"] == 1
    assert overall["pass_rate"] == 0.5
