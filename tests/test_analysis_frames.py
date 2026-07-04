from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from whetstone.analysis.figures import FigureRun
from whetstone.analysis.frames import (
    extract_encoder_decoder_models,
    is_pass_row,
    normalize_compression_target,
    parse_score_metrics,
    select_encdec_analysis_rows,
)
from whetstone.humaneval.scoring import GeneratedCodeOutcome
from whetstone.records import ScoreAttemptStatus

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts" / "analysis"


def load_analysis_script(module_name: str):
    path = SCRIPTS_DIR / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_select_encdec_analysis_rows_require_score_uses_inner_join() -> None:
    scored = select_encdec_analysis_rows(
        ["encdec-budget-full-v0"],
        require_score=True,
    )
    scored_sql = str(scored.compile(compile_kwargs={"literal_binds": True}))
    assert ScoreAttemptStatus.SUCCESS.value in scored_sql
    assert "JOIN dr_dspy_score_attempts" in scored_sql
    assert "LEFT OUTER JOIN dr_dspy_score_attempts" not in scored_sql

    all_rows = select_encdec_analysis_rows(
        ["encdec-budget-full-v0"],
        require_score=False,
    )
    all_sql = str(all_rows.compile(compile_kwargs={"literal_binds": True}))
    assert "LEFT OUTER JOIN dr_dspy_score_attempts" in all_sql


def test_figure_run_path_uses_script_folder_and_timestamp() -> None:
    output_run = FigureRun(
        script_name="q1_model_candidates",
        timestamp="20260630_120000",
    )
    fig_path = output_run.path("pass_rate_by_model")
    assert fig_path.parent.name == "q1_model_candidates"
    assert fig_path.parent.parent.name == "figs"
    assert fig_path.name == "20260630_120000_pass_rate_by_model.png"

    artifact_path = output_run.artifact_path("model_candidates", suffix="csv")
    assert artifact_path.parent.name == "q1_model_candidates"
    assert artifact_path.parent.parent.name == "artifacts"
    assert artifact_path.name == "20260630_120000_model_candidates.csv"

    log_path = output_run.run_log_path()
    assert log_path.parent.name == "q1_model_candidates"
    assert log_path.name == "20260630_120000_run.html"


def test_normalize_compression_target_prefers_compression_target() -> None:
    assert normalize_compression_target({"compression_target": 0.25}) == 0.25


def test_normalize_compression_target_falls_back_to_budget_ratio() -> None:
    assert normalize_compression_target({"budget_ratio": 0.5}) == 0.5


def test_normalize_compression_target_unwraps_dimensions_values() -> None:
    from whetstone.analysis.frames import _dimension_values

    wrapped = {"values": {"budget_ratio": 0.25}}
    assert normalize_compression_target(wrapped) is None
    assert normalize_compression_target(_dimension_values(wrapped)) == 0.25


def test_normalize_compression_target_returns_none_when_missing() -> None:
    assert normalize_compression_target({}) is None


def test_extract_encoder_decoder_models_from_dimensions() -> None:
    encoder, decoder = extract_encoder_decoder_models(
        {"encoder_model": "enc", "decoder_model": "dec"},
        None,
    )
    assert encoder == "enc"
    assert decoder == "dec"


def test_extract_encoder_decoder_models_from_provider_configs() -> None:
    encoder, decoder = extract_encoder_decoder_models(
        {},
        [
            {"config_id": "encoder", "model": "enc-model"},
            {"config_id": "decoder", "model": "dec-model"},
        ],
    )
    assert encoder == "enc-model"
    assert decoder == "dec-model"


def test_parse_score_metrics_reads_compression_and_text() -> None:
    parsed = parse_score_metrics(
        {
            "compression": {"ratio_to_ground_truth": 0.42},
            "text": {"character_count": 123},
        }
    )
    assert parsed["realized_compression_ratio"] == 0.42
    assert parsed["text_character_count"] == 123


def test_is_pass_row_accepts_score_and_outcome() -> None:
    assert is_pass_row({"score": 1.0, "generated_code_outcome": None}) is True
    assert (
        is_pass_row(
            {
                "score": 0.0,
                "generated_code_outcome": GeneratedCodeOutcome.PASSED.value,
            }
        )
        is True
    )
    assert is_pass_row(
        {"score": 0.0, "generated_code_outcome": "tests_failed"}
    ) is False


def test_build_model_candidate_summary_handles_missing_scores() -> None:
    q1 = load_analysis_script("q1_model_candidates")
    frame = pd.DataFrame(
        [
            {
                "model": "model-a",
                "provider_kind": "openrouter",
                "compression_target": 0.25,
                "generation_status": "success",
                "score_status": pd.NA,
                "score": pd.NA,
                "generated_code_outcome": pd.NA,
                "total_provider_cost": 0.01,
            },
            {
                "model": "model-a",
                "provider_kind": "openrouter",
                "compression_target": 0.25,
                "generation_status": "success",
                "score_status": "success",
                "score": 1.0,
                "generated_code_outcome": "passed",
                "total_provider_cost": 0.02,
            },
        ]
    )
    summary = q1.build_model_candidate_summary(frame)
    model_row = summary[summary["rollup"] == "model"].iloc[0]
    assert model_row["total_runs"] == 2
    assert model_row["scoreable_count"] == 1
    assert model_row["pass_count"] == 1
    assert model_row["pass_rate"] == pytest.approx(1.0)
    assert model_row["generation_success_rate"] == pytest.approx(1.0)


def test_bootstrap_interval_width_handles_sparse_group() -> None:
    q3 = load_analysis_script("q3_repeat_stability")
    width = q3.bootstrap_interval_width(np.array([1.0]), sample_size=1)
    assert not np.isnan(width)
    too_large = q3.bootstrap_interval_width(np.array([1.0]), sample_size=2)
    assert np.isnan(too_large)


def test_bootstrap_interval_width_shrinks_as_sample_size_grows() -> None:
    q3 = load_analysis_script("q3_repeat_stability")
    passes = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    width_one = q3.bootstrap_interval_width(passes, sample_size=1, seed=0)
    width_five = q3.bootstrap_interval_width(passes, sample_size=5, seed=0)
    assert width_five < width_one


def test_classify_task_signal_flags() -> None:
    q4 = load_analysis_script("q4_task_variation")
    sparse = q4.classify_task_signal(pass_rate=1.0, scoreable_count=1)
    assert sparse["sparse"] is True
    assert sparse["useful_signal"] is False

    always_pass = q4.classify_task_signal(pass_rate=1.0, scoreable_count=5)
    assert always_pass["always_pass"] is True
    assert always_pass["useful_signal"] is False

    always_fail = q4.classify_task_signal(pass_rate=0.0, scoreable_count=5)
    assert always_fail["always_fail"] is True
    assert always_fail["useful_signal"] is False

    useful = q4.classify_task_signal(pass_rate=0.5, scoreable_count=5)
    assert useful["useful_signal"] is True
    assert useful["always_pass"] is False
    assert useful["always_fail"] is False
