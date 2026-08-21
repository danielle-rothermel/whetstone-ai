"""Invariants on SearchEvidence, the GEPA per-step evidence record.

Every rule below is a ``raise`` inside ``SearchEvidence._validate``. The
happy path is covered by ``tests/test_gepa_step_evidence.py``; these pin the
rejections, which nothing else exercises.
"""

from __future__ import annotations

import pytest

from whetstone.core.identity import TypedRef
from whetstone.experiment.candidate import candidate_reference
from whetstone.optim.contracts import IntentOutcome, SearchEvidence
from whetstone.testing.toy.experiment import build_toy_experiment


def _candidate_ref():
    return candidate_reference(
        build_toy_experiment(num_seeds=1).initial_candidate
    )


def _eval_result_ref() -> TypedRef:
    return TypedRef(
        schema_name="whetstone.eval_result",
        content_hash="a" * 64,
    )


def test_completed_evidence_requires_an_eval_result_ref() -> None:
    with pytest.raises(ValueError, match="requires an Evaluation Result ref"):
        SearchEvidence(
            eval_request_id="e1",
            candidate=_candidate_ref(),
            outcome=IntentOutcome.COMPLETED,
        )


def test_failed_evidence_requires_an_eval_result_ref() -> None:
    with pytest.raises(ValueError, match="requires an Evaluation Result ref"):
        SearchEvidence(
            eval_request_id="e1",
            candidate=_candidate_ref(),
            outcome=IntentOutcome.FAILED,
        )


def test_rejected_evidence_carries_no_eval_result_ref() -> None:
    with pytest.raises(ValueError, match="carries no Evaluation Result ref"):
        SearchEvidence(
            eval_request_id="e1",
            candidate=_candidate_ref(),
            outcome=IntentOutcome.REJECTED,
            eval_result_ref=_eval_result_ref(),
        )


def test_rejected_evidence_is_valid_without_an_eval_result_ref() -> None:
    evidence = SearchEvidence(
        eval_request_id="e1",
        candidate=_candidate_ref(),
        outcome=IntentOutcome.REJECTED,
    )

    assert evidence.eval_result_ref is None
    assert evidence.evidence_refs == ()


def test_rewardless_evidence_carries_no_reward_evidence_refs() -> None:
    with pytest.raises(ValueError, match="no Reward evidence refs"):
        SearchEvidence(
            eval_request_id="e1",
            candidate=_candidate_ref(),
            outcome=IntentOutcome.COMPLETED,
            eval_result_ref=_eval_result_ref(),
            reward_evidence_refs=(_eval_result_ref(),),
        )
