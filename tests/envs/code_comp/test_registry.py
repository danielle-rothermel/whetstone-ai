from __future__ import annotations

from pathlib import Path

import pytest
from dr_code.humaneval.plus_dataset import HF_DATASET_ID, HF_REVISION

from tests.envs.support import synthetic_ed1_tasks
from tests.envs.test_ed1m import _mutant_record
from whetstone.envs.code_comp import (
    CodeCompMode,
    DirectExperiment,
    EncDecExperiment,
    build_code_comp_experiment,
    build_direct_experiment,
    build_encdec_experiment,
    build_mutant_experiment,
    code_comp_mode_for,
)
from whetstone.envs.d1 import build_d1_experiment
from whetstone.envs.ed1 import build_ed1_experiment
from whetstone.envs.ed1m import build_ed1m_experiment
from whetstone.envs.ed1m_dataset import (
    FamilyCount,
    GenerationConfig,
    OperatorFamily,
    build_manifest,
    encode_records,
)


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
        runner_identity="whetstone-test-fixture@v1",
        runtime_identity="whetstone-test-fixture-runtime@v1",
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


def test_registry_direct_matches_legacy_builder() -> None:
    tasks = synthetic_ed1_tasks(2)
    legacy = build_d1_experiment(tasks=tasks, internal_n=1, official_n=1)
    unified = build_code_comp_experiment(
        CodeCompMode.DIRECT,
        tasks=tasks,
        internal_n=1,
        official_n=1,
    )
    assert type(unified) is type(legacy)
    assert unified.env_name == legacy.env_name
    assert unified.input_arm == legacy.input_arm


def test_registry_encdec_matches_legacy_builder() -> None:
    tasks = synthetic_ed1_tasks(2)
    legacy = build_ed1_experiment(tasks=tasks, internal_n=1, official_n=1)
    unified = build_code_comp_experiment(
        CodeCompMode.ENCDEC,
        tasks=tasks,
        internal_n=1,
        official_n=1,
    )
    assert type(unified) is type(legacy)
    assert unified.env_name == legacy.env_name
    assert unified.budget_ratio == legacy.budget_ratio


def test_registry_mutant_matches_legacy_builder(
    mutant_dataset_dir: Path,
) -> None:
    legacy = build_ed1m_experiment(
        artifact_dir=mutant_dataset_dir,
        internal_n=1,
        official_n=1,
    )
    unified = build_code_comp_experiment(
        CodeCompMode.ENCDEC_MUTANT,
        artifact_dir=mutant_dataset_dir,
        internal_n=1,
        official_n=1,
    )
    assert type(unified) is type(legacy)
    assert unified.env_name == legacy.env_name


@pytest.mark.parametrize(
    ("experiment", "expected_mode"),
    [
        (
            build_direct_experiment(
                tasks=synthetic_ed1_tasks(1),
                internal_n=1,
                official_n=1,
            ),
            CodeCompMode.DIRECT,
        ),
        (
            build_encdec_experiment(
                tasks=synthetic_ed1_tasks(1),
                internal_n=1,
                official_n=1,
            ),
            CodeCompMode.ENCDEC,
        ),
    ],
)
def test_code_comp_mode_for_direct_and_encdec(
    experiment: DirectExperiment | EncDecExperiment,
    expected_mode: CodeCompMode,
) -> None:
    assert code_comp_mode_for(experiment) is expected_mode


def test_code_comp_mode_for_mutant(mutant_dataset_dir: Path) -> None:
    experiment = build_mutant_experiment(
        artifact_dir=mutant_dataset_dir,
        internal_n=1,
        official_n=1,
    )
    assert code_comp_mode_for(experiment) is CodeCompMode.ENCDEC_MUTANT


def test_code_comp_mode_for_rejects_non_code_comp_experiment() -> None:
    from whetstone.envs.factory import build_env_experiment

    experiment = build_env_experiment(
        "c18",
        model="openai/test",
        pool_n_per_stratum=1,
        split_sizes=(1, 1, 1),
    )
    with pytest.raises(TypeError, match="not a code_comp"):
        code_comp_mode_for(experiment)


def test_registry_rejects_unknown_direct_kwargs() -> None:
    with pytest.raises(TypeError):
        build_code_comp_experiment(
            CodeCompMode.DIRECT,
            tasks=synthetic_ed1_tasks(1),
            artifact_dir=Path("."),
        )
