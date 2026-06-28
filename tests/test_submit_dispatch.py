from __future__ import annotations

import importlib.util
from pathlib import Path

from dr_dspy import humaneval_direct_dbos as direct
from dr_dspy import humaneval_encdec_dbos as encdec
from dr_dspy import submission_dispatch
from dr_dspy.lm_utils import ModelConfig

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(path: Path, module_name: str) -> None:
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def test_direct_offset_jobs_match_full_cartesian_builder() -> None:
    _load_script(
        REPO_ROOT / "scripts" / "humaneval_dspy_eval_only_dbos_v0.py",
        "direct_script_for_submit_dispatch_test",
    )
    samples = [
        direct.HumanEvalSample(
            task_id=f"task/{index}",
            sample_index=index,
            prompt="def f():\n    pass\n",
            canonical_solution="",
            ground_truth_code="def f():\n    return 1\n",
            test="def check(candidate):\n    pass\n",
            entry_point="f",
        )
        for index in range(2)
    ]
    models = [
        ModelConfig(model="model/a", reasoning={}),
        ModelConfig(model="model/b", reasoning={"effort": "low"}),
    ]
    full = direct.build_prediction_jobs(
        experiment_name="exp",
        submission_id="sub",
        samples=samples,
        model_configs=models,
        temperatures=[0.0, 0.5],
        repetitions=2,
    )
    spec = direct.build_submit_spec(
        experiment_name="exp",
        seed=0,
        sample_count=len(samples),
        model_configs=models,
        temperatures=[0.0, 0.5],
        repetitions=2,
        score_timeout=10.0,
    )
    chunked = []
    for offset in range(0, spec.total_jobs(), 3):
        chunked.extend(
            direct.build_prediction_jobs_for_offsets(
                spec=spec,
                submission_id="sub",
                samples=samples,
                start_offset=offset,
                limit=3,
            )
        )
    assert [job.model_dump() for job in chunked] == [
        job.model_dump() for job in full
    ]


def test_encdec_offset_jobs_match_full_cartesian_builder(
    encdec_configured: None,
) -> None:
    samples = [
        encdec.EncDecSample(
            task_id=f"task/{index}",
            sample_index=index,
            prompt="def f():\n    pass\n",
            canonical_solution="",
            ground_truth_code="def f():\n    return 1\n",
            test="def check(candidate):\n    pass\n",
            entry_point="f",
        )
        for index in range(2)
    ]
    pairs = [
        encdec.EncDecPair(
            encoder=ModelConfig(model="model/a", reasoning={}),
            decoder=ModelConfig(model="model/a", reasoning={}),
        ),
        encdec.EncDecPair(
            encoder=ModelConfig(model="model/b", reasoning={"x": 1}),
            decoder=ModelConfig(model="model/c", reasoning={"y": 2}),
        ),
    ]
    full = encdec.build_prediction_jobs(
        experiment_name="exp",
        submission_id="sub",
        samples=samples,
        model_pairs=pairs,
        encoder_temperatures=[0.0, 0.5],
        decoder_temperatures=[0.0],
        budget_ratios=[None, 0.5],
        repetitions=2,
    )
    spec = encdec.build_submit_spec(
        experiment_name="exp",
        seed=0,
        sample_count=len(samples),
        model_pairs=pairs,
        encoder_temperatures=[0.0, 0.5],
        decoder_temperatures=[0.0],
        budget_ratios=[None, 0.5],
        repetitions=2,
        score_timeout=10.0,
    )
    chunked = []
    for offset in range(0, spec.total_jobs(), 5):
        chunked.extend(
            encdec.build_prediction_jobs_for_offsets(
                spec=spec,
                submission_id="sub",
                samples=samples,
                start_offset=offset,
                limit=5,
            )
        )
    assert [job.model_dump() for job in chunked] == [
        job.model_dump() for job in full
    ]


def test_encdec_default_submit_plan_total_is_55104(
    encdec_configured: None,
) -> None:
    config = encdec.experiment_config()
    spec = encdec.build_submit_spec(
        experiment_name="exp",
        seed=config.default_seed,
        sample_count=config.default_sample_count,
        model_pairs=config.default_model_pairs,
        encoder_temperatures=config.default_encoder_temperatures,
        decoder_temperatures=config.default_decoder_temperatures,
        budget_ratios=config.default_budget_ratios,
        repetitions=config.default_repetitions,
        score_timeout=config.default_subprocess_timeout,
    )
    assert spec.total_jobs() == 55104


def test_submit_key_is_deterministic_and_changes_with_axes(
    encdec_configured: None,
) -> None:
    config = encdec.experiment_config()
    spec = encdec.build_submit_spec(
        experiment_name="exp",
        seed=config.default_seed,
        sample_count=1,
        model_pairs=config.default_model_pairs[:1],
        encoder_temperatures=config.default_encoder_temperatures,
        decoder_temperatures=config.default_decoder_temperatures,
        budget_ratios=[None],
        repetitions=1,
        score_timeout=config.default_subprocess_timeout,
    )
    changed = spec.model_copy(update={"budget_ratios": [0.5]})
    assert submission_dispatch.submit_key(
        spec.model_dump(mode="json")
    ) == submission_dispatch.submit_key(spec.model_dump(mode="json"))
    assert submission_dispatch.submit_key(
        spec.model_dump(mode="json")
    ) != submission_dispatch.submit_key(changed.model_dump(mode="json"))
