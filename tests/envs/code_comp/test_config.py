from __future__ import annotations

from pathlib import Path

import pytest

from tests.envs.code_comp.test_mutant import _mutant_record
from tests.envs.support import execution_policy, synthetic_code_comp_tasks
from whetstone.envs.code_comp.config import (
    CodeCompExperimentConfig,
    CodeCompModelRouteConfig,
    CodeCompModelRoutesConfig,
    DirectModeSettings,
    EncDecModeSettings,
    default_code_comp_config,
)
from whetstone.envs.code_comp.modes.direct import build_direct_experiment
from whetstone.envs.code_comp.modes.encdec import (
    EncDecExperiment,
    build_encdec_experiment,
)
from whetstone.envs.code_comp.modes.mutant import build_mutant_experiment
from whetstone.envs.code_comp.mutant.dataset import (
    FamilyCount,
    GenerationConfig,
    OperatorFamily,
    build_manifest,
    encode_records,
)
from whetstone.envs.code_comp.registry import CodeCompMode
from whetstone.envs.code_comp.runtime_config import (
    CodeCompEvaluationRuntimeConfig,
)


@pytest.fixture
def mutant_dataset_dir(tmp_path: Path) -> Path:
    from dr_code.humaneval.plus_dataset import HF_DATASET_ID, HF_REVISION

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


def test_identity_hash_is_stable_for_same_config() -> None:
    tasks = synthetic_code_comp_tasks(2)
    config = default_code_comp_config(
        CodeCompMode.DIRECT,
        pool={"tasks": tasks},
        split={"internal_n": 1, "official_n": 1},
    )
    assert config.identity_hash() == config.identity_hash()


def test_identity_hash_differs_for_different_modes() -> None:
    tasks = synthetic_code_comp_tasks(2)
    direct = default_code_comp_config(
        CodeCompMode.DIRECT,
        pool={"tasks": tasks},
        split={"internal_n": 1, "official_n": 1},
    )
    encdec = default_code_comp_config(
        CodeCompMode.ENCDEC,
        pool={"tasks": tasks},
        split={"internal_n": 1, "official_n": 1},
    )
    assert direct.identity_hash() != encdec.identity_hash()


def test_direct_config_matches_legacy_builder() -> None:
    tasks = synthetic_code_comp_tasks(2)
    legacy = build_direct_experiment(tasks=tasks, internal_n=1, official_n=1)
    config = default_code_comp_config(
        CodeCompMode.DIRECT,
        pool={"tasks": tasks},
        split={"internal_n": 1, "official_n": 1},
    )
    built = config.build_experiment()
    assert built.env_name == legacy.env_name
    assert (
        built.generation_graph.graph_hash == legacy.generation_graph.graph_hash
    )
    assert (
        built.eval_configs.internal.eval_config.config_hash
        == legacy.eval_configs.internal.eval_config.config_hash
    )


def test_encdec_config_matches_legacy_builder() -> None:
    tasks = synthetic_code_comp_tasks(2)
    legacy = build_encdec_experiment(tasks=tasks, internal_n=1, official_n=1)
    config = default_code_comp_config(
        CodeCompMode.ENCDEC,
        pool={"tasks": tasks},
        split={"internal_n": 1, "official_n": 1},
    )
    built = config.build_experiment()
    assert isinstance(built, EncDecExperiment)
    assert built.env_name == legacy.env_name
    assert (
        built.generation_graph.graph_hash == legacy.generation_graph.graph_hash
    )
    assert built.budget_ratio == legacy.budget_ratio
    assert (
        built.eval_configs.internal.eval_config.config_hash
        == legacy.eval_configs.internal.eval_config.config_hash
    )


def test_mutant_config_matches_legacy_builder(
    mutant_dataset_dir: Path,
) -> None:
    legacy = build_mutant_experiment(
        artifact_dir=mutant_dataset_dir,
        internal_n=1,
        official_n=1,
    )
    config = default_code_comp_config(
        CodeCompMode.ENCDEC_MUTANT,
        artifact_dir=mutant_dataset_dir,
        split={"internal_n": 1, "official_n": 1},
    )
    built = config.build_experiment()
    assert built.env_name == legacy.env_name
    assert (
        built.generation_graph.graph_hash == legacy.generation_graph.graph_hash
    )


def test_decoder_route_defaults_to_encoder() -> None:
    encoder = CodeCompModelRouteConfig(model="openai/test")
    routes = CodeCompModelRoutesConfig(encoder=encoder)
    assert routes.decoder_call_config().identity_hash == (
        routes.encoder_call_config().identity_hash
    )


def test_runtime_config_builds_engine_with_matching_eval_hash() -> None:
    from dr_store import MemoryBackend, ObjectStore

    tasks = synthetic_code_comp_tasks(2)
    config = default_code_comp_config(
        CodeCompMode.DIRECT,
        pool={"tasks": tasks},
        split={"internal_n": 1, "official_n": 1},
        sampling={"num_samples": 1},
    )
    experiment = config.build_experiment()
    runtime = CodeCompEvaluationRuntimeConfig(
        experiment_config=config,
        expected_eval_config_hash=(
            experiment.eval_configs.internal.eval_config.config_hash
        ),
        execution_policy=execution_policy(),
        row_job_entrypoint="tests.envs.process_workers:return_payload",
    )
    engine = runtime.build_engine(ObjectStore(MemoryBackend()))
    assert engine.experiment.env_name == "code_comp"


def test_config_rejects_mismatched_mode_settings() -> None:
    with pytest.raises(ValueError, match="requires its settings"):
        CodeCompExperimentConfig(
            mode=CodeCompMode.ENCDEC,
            models=CodeCompModelRoutesConfig(
                encoder=CodeCompModelRouteConfig(model="openai/test")
            ),
            direct=DirectModeSettings(),
        )


def test_config_rejects_cross_mode_settings() -> None:
    with pytest.raises(ValueError, match="cannot carry"):
        CodeCompExperimentConfig(
            mode=CodeCompMode.DIRECT,
            models=CodeCompModelRoutesConfig(
                encoder=CodeCompModelRouteConfig(model="openai/test")
            ),
            direct=DirectModeSettings(),
            encdec=EncDecModeSettings(),
        )


def test_compression_level_affects_procedure_hash() -> None:
    base = default_code_comp_config(CodeCompMode.ENCDEC)
    custom = default_code_comp_config(
        CodeCompMode.ENCDEC,
        compression={"level": 9},
    )
    assert base.build_procedure_config().config_hash != (
        custom.build_procedure_config().config_hash
    )


def test_direct_engine_task_model_identity_hash() -> None:
    from dr_store import MemoryBackend, ObjectStore

    from tests.envs.support import (
        code_comp_direct_experiment,
        in_process_direct_row_job_factory,
    )
    from whetstone.evaluation.engine import EvaluationEngine

    custom_experiment = code_comp_direct_experiment(
        task_count=1,
        internal_n=1,
        official_n=1,
        model="openai/custom-direct",
    )
    custom_engine = EvaluationEngine(
        store=ObjectStore(MemoryBackend()),
        experiment=custom_experiment,
        sampling=custom_experiment.eval_configs.internal,
        execution_policy=execution_policy(),
        row_job_factory=in_process_direct_row_job_factory(),
    )
    assert custom_engine.task_model_identity_hash == (
        custom_experiment.generation_graph.provider_call_config.identity_hash
    )

    default_experiment = code_comp_direct_experiment(
        task_count=1,
        internal_n=1,
        official_n=1,
    )
    default_engine = EvaluationEngine(
        store=ObjectStore(MemoryBackend()),
        experiment=default_experiment,
        sampling=default_experiment.eval_configs.internal,
        execution_policy=execution_policy(),
        row_job_factory=in_process_direct_row_job_factory(),
    )
    assert (
        custom_engine.task_model_identity_hash
        != default_engine.task_model_identity_hash
    )


def test_encdec_engine_task_model_identity_hash_uses_encoder_route() -> None:
    from dr_store import MemoryBackend, ObjectStore

    from tests.envs.support import process_row_job_factory
    from whetstone.evaluation.engine import EvaluationEngine

    tasks = synthetic_code_comp_tasks(1)
    config = default_code_comp_config(
        CodeCompMode.ENCDEC,
        pool={"tasks": tasks},
        split={"internal_n": 1, "official_n": 1},
        models=CodeCompModelRoutesConfig(
            encoder=CodeCompModelRouteConfig(model="openai/custom-encoder"),
            decoder=CodeCompModelRouteConfig(model="openai/custom-decoder"),
        ),
    )
    experiment = config.build_experiment()
    engine = EvaluationEngine(
        store=ObjectStore(MemoryBackend()),
        experiment=experiment,
        sampling=experiment.eval_configs.internal,
        execution_policy=execution_policy(),
        row_job_factory=process_row_job_factory(
            "tests.envs.process_workers:drive_ed1_success"
        ),
    )
    assert engine.task_model_identity_hash == (
        config.models.encoder_call_config().identity_hash
    )
    assert engine.task_model_identity_hash != (
        config.models.decoder_call_config().identity_hash
    )


def test_distinct_decoder_route_affects_graph_hash() -> None:
    tasks = synthetic_code_comp_tasks(1)
    shared = default_code_comp_config(
        CodeCompMode.ENCDEC,
        pool={"tasks": tasks},
    )
    split = default_code_comp_config(
        CodeCompMode.ENCDEC,
        pool={"tasks": tasks},
        models=CodeCompModelRoutesConfig(
            encoder=CodeCompModelRouteConfig(model="openai/test-a"),
            decoder=CodeCompModelRouteConfig(model="openai/test-b"),
        ),
    )
    procedure_hash = shared.build_procedure_config().config_hash
    shared_graph = shared.build_generation_graph(
        procedure_config_hash=procedure_hash
    )
    split_graph = split.build_generation_graph(
        procedure_config_hash=procedure_hash
    )
    assert shared_graph.graph_hash != split_graph.graph_hash
    assert (
        split_graph.encoder_call_config.identity_hash
        != split_graph.decoder_call_config.identity_hash
    )
