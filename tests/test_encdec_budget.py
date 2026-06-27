from __future__ import annotations

from typing import Any

import pytest

from dr_dspy import humaneval_encdec_dbos as encdec
from dr_dspy.lm_utils import ModelConfig

SAMPLE = encdec.EncDecSample(
    task_id="t/0",
    sample_index=0,
    prompt="def f(x):\n    pass\n",
    canonical_solution="def f(x):\n    return x\n",
    ground_truth_code="def f(x):\n    return x\n",
    test="def check(candidate):\n    inputs = [[1]]\n    results = [1]\n",
    entry_point="f",
)
PAIR = encdec.EncDecPair(
    encoder=ModelConfig(model="enc/m", reasoning={}),
    decoder=ModelConfig(model="dec/m", reasoning={}),
)


def test_parse_budget_ratios_handles_none_and_floats() -> None:
    assert encdec.parse_budget_ratios("none,0.5,1.0") == [None, 0.5, 1.0]
    assert encdec.parse_budget_ratios("NONE") == [None]


def test_build_prediction_jobs_cartesian_includes_budget() -> None:
    jobs = encdec.build_prediction_jobs(
        experiment_name="exp",
        submission_id="sub",
        samples=[SAMPLE],
        model_pairs=[PAIR],
        encoder_temperatures=[0.0],
        decoder_temperatures=[0.0],
        budget_ratios=[None, 0.5],
        repetitions=1,
    )
    assert len(jobs) == 2
    assert {job.budget_ratio for job in jobs} == {None, 0.5}
    assert len({job.prediction_id for job in jobs}) == 2


def _stub_lms(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def fake_build_lm(**_kwargs: Any) -> object:
        return object()

    def fake_run_predictor(
        *, signature: Any, input_kwargs: Any, output_field: str, **_kw: Any
    ) -> str:
        calls.append(
            {
                "signature": signature,
                "input_kwargs": dict(input_kwargs),
                "output_field": output_field,
            }
        )
        return "DESC" if output_field == "description" else "CODE"

    monkeypatch.setattr(encdec, "build_lm", fake_build_lm)
    monkeypatch.setattr(encdec, "run_predictor", fake_run_predictor)
    return calls


def _job(budget_ratio: float | None) -> encdec.EncDecJob:
    (job,) = encdec.build_prediction_jobs(
        experiment_name="exp",
        submission_id="sub",
        samples=[SAMPLE],
        model_pairs=[PAIR],
        encoder_temperatures=[0.0],
        decoder_temperatures=[0.0],
        budget_ratios=[budget_ratio],
        repetitions=1,
    )
    return job


def test_no_budget_uses_plain_encoder(
    encdec_configured: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _stub_lms(monkeypatch)
    result = encdec.generate_code_for_job(_job(None))
    encoder_call = calls[0]
    assert encoder_call["signature"] is encdec.encoder_signature()
    assert "max_characters" not in encoder_call["input_kwargs"]
    assert result.encoder_char_budget is None


def test_budget_uses_budgeted_encoder_with_derived_chars(
    encdec_configured: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _stub_lms(monkeypatch)
    job = _job(5.0)
    expected = round(5.0 * len(job.ground_truth_code))
    assert expected > encdec.MIN_ENCODER_CHAR_BUDGET
    result = encdec.generate_code_for_job(job)
    encoder_call = calls[0]
    assert encoder_call["signature"] is encdec.budgeted_encoder_signature()
    assert encoder_call["input_kwargs"]["max_characters"] == expected
    assert result.encoder_char_budget == expected


def test_budget_respects_min_floor(
    encdec_configured: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _stub_lms(monkeypatch)
    job = _job(0.1)
    assert round(0.1 * len(job.ground_truth_code)) < (
        encdec.MIN_ENCODER_CHAR_BUDGET
    )
    result = encdec.generate_code_for_job(job)
    assert (
        calls[0]["input_kwargs"]["max_characters"]
        == encdec.MIN_ENCODER_CHAR_BUDGET
    )
    assert result.encoder_char_budget == encdec.MIN_ENCODER_CHAR_BUDGET
