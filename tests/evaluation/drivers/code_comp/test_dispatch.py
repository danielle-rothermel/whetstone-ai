from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from dr_code.humaneval.plus_dataset import HF_DATASET_ID, HF_REVISION

from tests.envs.code_comp.test_mutant import _mutant_record
from tests.envs.support import synthetic_code_comp_tasks
from whetstone.envs.code_comp import (
    CodeCompMode,
    MutantExperiment,
    build_code_comp_experiment,
)
from whetstone.envs.code_comp.mutant.dataset import (
    FamilyCount,
    GenerationConfig,
    OperatorFamily,
    build_manifest,
    encode_records,
)
from whetstone.envs.factory import build_env_experiment
from whetstone.evaluation.drivers.code_comp.dispatch import run_code_comp_eval


@pytest.fixture
def mutant_dataset_dir(tmp_path: Path) -> Path:
    record = _mutant_record()
    config = GenerationConfig(
        dataset_id=HF_DATASET_ID,
        dataset_revision=HF_REVISION,
        operator_families=(OperatorFamily.COMPARISON_FLIP,),
        seeds=1,
        max_inputs_per_mutant=10,
        timeout_seconds=5.0,
        task_ids=("HumanEval/0",),
        canonical_suite_digest="opaque-schema-v1-suite-provenance",
        runner_label="whetstone-test-fixture@v1",
        runtime_label="whetstone-test-fixture-runtime@v1",
    )
    manifest = build_manifest(
        config=config,
        records=(record,),
        accepted_by_family=(
            FamilyCount(
                operator_family=OperatorFamily.COMPARISON_FLIP,
                count=1,
            ),
        ),
    )
    output_dir = tmp_path / "mutant-dataset"
    output_dir.mkdir()
    (output_dir / "mutants.jsonl").write_bytes(encode_records((record,)))
    (output_dir / "manifest.json").write_text(
        manifest.model_dump_json(indent=2) + "\n"
    )
    return output_dir


def test_run_code_comp_eval_routes_direct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = build_code_comp_experiment(
        CodeCompMode.DIRECT,
        tasks=synthetic_code_comp_tasks(1),
        internal_n=1,
        official_n=1,
    )
    sentinel: dict[str, Any] = {"called": False}

    def fake_run_direct_eval(*args: Any, **kwargs: Any) -> str:
        sentinel["called"] = True
        assert args[0] is experiment
        return "d1-result"

    monkeypatch.setattr(
        "whetstone.evaluation.drivers.code_comp.direct.run_direct_eval",
        fake_run_direct_eval,
    )
    assert run_code_comp_eval(experiment, candidate_body="x") == "d1-result"
    assert sentinel["called"]


def test_run_code_comp_eval_routes_encdec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = build_code_comp_experiment(
        CodeCompMode.ENCDEC,
        tasks=synthetic_code_comp_tasks(1),
        internal_n=1,
        official_n=1,
    )
    sentinel: dict[str, Any] = {"called": False}

    def fake_run_encdec_eval(*args: Any, **kwargs: Any) -> str:
        sentinel["called"] = True
        assert args[0] is experiment
        return "ed1-result"

    monkeypatch.setattr(
        "whetstone.evaluation.drivers.code_comp.encdec.run_encdec_eval",
        fake_run_encdec_eval,
    )
    assert (
        run_code_comp_eval(experiment, candidate_template="x") == "ed1-result"
    )
    assert sentinel["called"]


def test_run_code_comp_eval_routes_mutant_via_encdec(
    monkeypatch: pytest.MonkeyPatch,
    mutant_dataset_dir: Path,
) -> None:
    experiment = build_code_comp_experiment(
        CodeCompMode.ENCDEC_MUTANT,
        artifact_dir=mutant_dataset_dir,
        internal_n=1,
        official_n=1,
    )
    assert isinstance(experiment, MutantExperiment)
    sentinel: dict[str, Any] = {"called": False}

    def fake_run_encdec_eval(*args: Any, **kwargs: Any) -> str:
        sentinel["called"] = True
        assert args[0] is experiment
        return "ed1m-result"

    monkeypatch.setattr(
        "whetstone.evaluation.drivers.code_comp.encdec.run_encdec_eval",
        fake_run_encdec_eval,
    )
    assert (
        run_code_comp_eval(experiment, candidate_template="x") == "ed1m-result"
    )
    assert sentinel["called"]


def test_run_code_comp_eval_rejects_non_code_comp_experiment() -> None:
    experiment = build_env_experiment(
        "c18",
        model="openai/test",
        pool_n_per_stratum=1,
        split_sizes=(1, 1, 1),
    )
    with pytest.raises(TypeError, match="not a code_comp"):
        run_code_comp_eval(experiment)
