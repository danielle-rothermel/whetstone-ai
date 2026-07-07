"""Tests for minimal COPRO enc-dec optimizer helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from whetstone.optimization.copro import (
    CoproAttempt,
    CoproCandidate,
    CoproProposalMode,
    CoproRunConfig,
    baseline_candidate,
    build_copro_dimensions,
    instructions_digest,
    make_candidate_id,
    manual_proposal_candidates,
    parse_lm_proposal_response,
    render_attempt_history,
    select_best_candidate,
    summarize_attempts,
)
from whetstone.platform.spec_builder import (
    DEFAULT_HUMANEVAL_INSTRUCTIONS_START,
)
from whetstone.records import DimensionsPayload


def test_manual_candidates_include_baseline_and_match_breadth() -> None:
    current = baseline_candidate(depth=0)
    candidates = manual_proposal_candidates(current, breadth=3, depth=0)

    assert len(candidates) == 3
    assert (
        candidates[0].instructions_start
        == DEFAULT_HUMANEVAL_INSTRUCTIONS_START
    )
    assert candidates[0].instructions_end == ""
    assert candidates[0].proposal_source == "carry_forward"
    assert {candidate.proposal_source for candidate in candidates[1:]} <= {
        "manual",
    }


def test_candidate_ids_and_digests_are_stable() -> None:
    assert make_candidate_id(1, 2) == "d1_c2"
    digest_a = instructions_digest("start", "end")
    digest_b = instructions_digest("start", "end")
    digest_c = instructions_digest("other", "end")

    assert digest_a == digest_b
    assert digest_a != digest_c
    assert len(digest_a) == 16

    candidate = CoproCandidate(
        candidate_id="d0_c0",
        depth=0,
        instructions_start="start",
        instructions_end="end",
        proposal_source="manual",
        instructions_digest=digest_a,
    )
    assert candidate.instructions_digest == digest_a


def test_render_attempt_history_orders_by_score_desc() -> None:
    attempts = (
        CoproAttempt(
            candidate_id="d0_c1",
            depth=0,
            instructions_start="low",
            instructions_end="",
            proposal_source="manual",
            instructions_digest=instructions_digest("low", ""),
            experiment_name="exp",
            scoreable_count=2,
            pass_count=1,
            pass_rate=0.5,
        ),
        CoproAttempt(
            candidate_id="d0_c0",
            depth=0,
            instructions_start="high",
            instructions_end="",
            proposal_source="baseline",
            instructions_digest=instructions_digest("high", ""),
            experiment_name="exp",
            scoreable_count=2,
            pass_count=2,
            pass_rate=1.0,
        ),
    )

    rendered = render_attempt_history(attempts)

    assert rendered.index("d0_c0") < rendered.index("d0_c1")
    assert "pass_rate=1.000" in rendered
    assert "pass_rate=0.500" in rendered


def test_build_copro_dimensions_includes_optimizer_metadata() -> None:
    candidate = baseline_candidate()
    dimensions = build_copro_dimensions(
        candidate=candidate,
        copro_run_id="run123",
        compression_target=0.5,
        encoder_model="encoder-model",
        decoder_model="decoder-model",
    )

    assert isinstance(dimensions, DimensionsPayload)
    assert dimensions.values["optimizer"] == "copro_minimal"
    assert dimensions.values["copro_run_id"] == "run123"
    assert dimensions.values["candidate_id"] == candidate.candidate_id
    assert dimensions.values["candidate_depth"] == candidate.depth
    assert (
        dimensions.values["instructions_digest"]
        == candidate.instructions_digest
    )
    assert dimensions.values["compression_target"] == 0.5


def test_select_best_candidate_breaks_ties_deterministically() -> None:
    attempts = (
        CoproAttempt(
            candidate_id="d0_c2",
            depth=0,
            instructions_start="longer prompt text",
            instructions_end="tail",
            proposal_source="manual",
            instructions_digest=instructions_digest(
                "longer prompt text", "tail"
            ),
            experiment_name="exp",
            scoreable_count=2,
            pass_count=1,
            pass_rate=0.5,
            generation_error_count=0,
            score_error_count=0,
        ),
        CoproAttempt(
            candidate_id="d0_c1",
            depth=0,
            instructions_start="short",
            instructions_end="",
            proposal_source="manual",
            instructions_digest=instructions_digest("short", ""),
            experiment_name="exp",
            scoreable_count=2,
            pass_count=1,
            pass_rate=0.5,
            generation_error_count=0,
            score_error_count=0,
        ),
        CoproAttempt(
            candidate_id="d0_c0",
            depth=0,
            instructions_start="short",
            instructions_end="",
            proposal_source="baseline",
            instructions_digest=instructions_digest("short", ""),
            experiment_name="exp",
            scoreable_count=1,
            pass_count=1,
            pass_rate=1.0,
            generation_error_count=0,
            score_error_count=0,
        ),
    )

    best = select_best_candidate(attempts)

    assert best is not None
    assert best.candidate_id == "d0_c0"
    tied = (
        attempts[0],
        attempts[1],
    )
    second = select_best_candidate(tied)
    assert second is not None
    assert second.candidate_id == "d0_c1"


def test_parse_lm_proposal_response_rejects_malformed_json() -> None:
    with pytest.raises(ValueError, match="JSON"):
        parse_lm_proposal_response("not json at all")

    with pytest.raises(ValueError, match="candidates list"):
        parse_lm_proposal_response(json.dumps({"candidates": []}))

    with pytest.raises(ValueError, match="instructions_start"):
        parse_lm_proposal_response(
            json.dumps(
                {
                    "candidates": [
                        {"instructions_start": 1, "instructions_end": ""}
                    ]
                }
            )
        )


def test_parse_lm_proposal_response_accepts_valid_payload() -> None:
    parsed = parse_lm_proposal_response(
        json.dumps(
            {
                "candidates": [
                    {
                        "instructions_start": "Describe the code briefly.",
                        "instructions_end": "",
                    }
                ]
            }
        )
    )

    assert parsed == (
        {
            "instructions_start": "Describe the code briefly.",
            "instructions_end": "",
        },
    )


def test_summarize_attempts_groups_by_candidate_id() -> None:
    candidate = baseline_candidate()
    frame = pd.DataFrame(
        [
            {
                "dimensions": json.dumps(
                    {
                        "values": {
                            "candidate_id": candidate.candidate_id,
                            "candidate_depth": 0,
                        }
                    }
                ),
                "generation_status": "success",
                "score_status": "success",
                "score": 1.0,
                "generated_code_outcome": "passed",
            },
            {
                "dimensions": json.dumps(
                    {
                        "values": {
                            "candidate_id": candidate.candidate_id,
                            "candidate_depth": 0,
                        }
                    }
                ),
                "generation_status": "success",
                "score_status": "success",
                "score": 0.0,
                "generated_code_outcome": "failed",
            },
        ]
    )

    attempts = summarize_attempts(
        frame,
        experiment_name="copro_minimal_test",
        candidates=(candidate,),
    )

    assert len(attempts) == 1
    assert attempts[0].scoreable_count == 2
    assert attempts[0].pass_count == 1
    assert attempts[0].pass_rate == 0.5


def test_copro_run_config_requires_prompt_model_for_lm_mode(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="prompt_model"):
        CoproRunConfig(
            model_config_path=tmp_path / "model.json",
            split_path=tmp_path / "split.json",
            compression_targets=(0.5,),
            breadth=2,
            depth=1,
            repetition_seeds=(0,),
            proposal_mode=CoproProposalMode.LM,
            output_dir=tmp_path / "out",
        )
