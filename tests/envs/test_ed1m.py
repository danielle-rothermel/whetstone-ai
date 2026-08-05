"""Authenticated ed1m dataset, oracle, and environment contract tests."""

from __future__ import annotations

from inspect import Parameter, signature
from pathlib import Path
from typing import cast

import pytest
from dr_code.execution import SubprocessStartError
from dr_code.mutants import load_dataset
from dr_code.mutants import provenance as provenance_module
from dr_code.mutants.dataset import (
    ExpectedOutcome,
    FamilyCount,
    GeneratedDataset,
    GenerationConfig,
    MutantRecord,
    build_record,
    publish_dataset,
)
from dr_code.mutants.operators import OperatorFamily
from dr_code.mutants.provenance import (
    canonical_suite_digest,
    resolve_canonical_suite,
)
from dr_code.synthetic.humaneval_loader import (
    HF_DATASET_ID,
    HF_REVISION,
    HumanEvalPlusTask,
)

_CANONICAL_SOURCE = "def f(x):\n    return x < 1"
_MUTATED_SOURCE = "def f(x):\n    return x <= 1"
_CANONICAL_TEST = """def check(candidate):
    inputs = [[1], [2]]
    results = [False, False]
    for i, (inp, exp) in enumerate(zip(inputs, results)):
        assertion(candidate(*inp), exp, 0)
"""


def _mutant_record() -> MutantRecord:
    return build_record(
        task_id="HumanEval/0",
        entry_point="f",
        prompt="def f(x):\n",
        canonical_full_source=_CANONICAL_SOURCE,
        mutated_full_source=_MUTATED_SOURCE,
        operator_family=OperatorFamily.COMPARISON_FLIP,
        seed=0,
        site_node_path=5,
        site_target_index=0,
        site_description="line 2: comparison operand 0 <",
        input_reprs=("(1,)", "(2,)"),
        mutant_expected=(
            ExpectedOutcome(kind="value", output_repr="True"),
            ExpectedOutcome(kind="value", output_repr="False"),
        ),
        canonical_expected=(
            ExpectedOutcome(kind="value", output_repr="False"),
            ExpectedOutcome(kind="value", output_repr="False"),
        ),
        distinct_input_indices=(0,),
        diff_summary="changed comparison from < to <=",
        canonical_test=_CANONICAL_TEST,
    )


@pytest.fixture
def mutant_dataset_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """Publish a canonical authenticated dataset without external artifacts."""
    tasks = [
        HumanEvalPlusTask(
            task_id="HumanEval/0",
            prompt="def f(x):\n",
            canonical_solution="    return x < 1\n",
            entry_point="f",
            test=_CANONICAL_TEST,
        )
    ]
    monkeypatch.setattr(
        provenance_module,
        "load_humaneval_plus",
        lambda prefer_snapshot: tasks,
    )
    suite = resolve_canonical_suite(
        task_ids=("HumanEval/0",),
        max_inputs=10,
        prefer_snapshot=True,
    )
    record = _mutant_record()
    generated = GeneratedDataset(
        config=GenerationConfig(
            dataset_id=HF_DATASET_ID,
            dataset_revision=HF_REVISION,
            operator_families=(OperatorFamily.COMPARISON_FLIP,),
            seeds=1,
            max_inputs_per_mutant=10,
            timeout_seconds=5.0,
            task_ids=("HumanEval/0",),
            canonical_suite_digest=canonical_suite_digest(suite),
            runner_identity="whetstone-test-fixture@v1",
            runtime_identity="whetstone-test-fixture-runtime@v1",
        ),
        records=(record,),
        accepted_by_family=(
            FamilyCount(
                operator_family=OperatorFamily.COMPARISON_FLIP,
                count=1,
            ),
        ),
        skipped=(),
    )
    output_dir = tmp_path / "mutant-dataset"
    publish_dataset(output_dir=output_dir, generated=generated)
    return output_dir


def test_canonical_loader_authenticates_fixture(
    mutant_dataset_dir: Path,
) -> None:
    loaded = load_dataset(mutant_dataset_dir)

    assert loaded.records == (_mutant_record(),)
    assert isinstance(loaded.records[0], MutantRecord)
    assert len(loaded.manifest.dataset_identity) == 64


def test_build_requires_explicit_path() -> None:
    from whetstone.envs.ed1m import build_ed1m_experiment

    artifact_dir = signature(build_ed1m_experiment).parameters["artifact_dir"]

    assert artifact_dir.default is Parameter.empty
    assert artifact_dir.kind is Parameter.KEYWORD_ONLY


def test_build_rejects_non_path_boundary(
    mutant_dataset_dir: Path,
) -> None:
    from whetstone.envs.ed1m import build_ed1m_experiment

    with pytest.raises(TypeError, match=r"pathlib\.Path"):
        build_ed1m_experiment(
            artifact_dir=cast(Path, str(mutant_dataset_dir)),
        )


def test_build_rejects_unauthenticated_records(
    mutant_dataset_dir: Path,
) -> None:
    from dr_code.mutants.dataset import DatasetValidationError

    from whetstone.envs.ed1m import build_ed1m_experiment

    records_path = mutant_dataset_dir / "mutants.jsonl"
    records_path.write_text(
        records_path.read_text().replace('"True"', '"tampered"')
    )

    with pytest.raises(DatasetValidationError, match="SHA-256"):
        build_ed1m_experiment(artifact_dir=mutant_dataset_dir)


def test_oracle_faithful_reconstruction_has_zero_attractor() -> None:
    from whetstone.envs.ed1m_oracle import score_ed1m_reconstruction

    score = score_ed1m_reconstruction(
        reconstruction=_MUTATED_SOURCE,
        mutant=_mutant_record(),
        timeout_seconds=5.0,
    )

    assert score.fidelity_to_mutant == pytest.approx(1.0)
    assert score.attractor_pull == pytest.approx(0.0)
    assert score.infrastructure_unknown is False


def test_oracle_canonical_reconstruction_has_full_attractor() -> None:
    from whetstone.envs.ed1m_oracle import score_ed1m_reconstruction

    score = score_ed1m_reconstruction(
        reconstruction=_CANONICAL_SOURCE,
        mutant=_mutant_record(),
        timeout_seconds=5.0,
    )

    assert score.fidelity_to_mutant == pytest.approx(0.5)
    assert score.attractor_pull == pytest.approx(1.0)


def test_oracle_definitive_mismatch_scores_zero() -> None:
    from whetstone.envs.ed1m_oracle import score_ed1m_reconstruction

    score = score_ed1m_reconstruction(
        reconstruction="def f(x):\n    return None\n",
        mutant=_mutant_record(),
        timeout_seconds=5.0,
    )

    assert score.fidelity_to_mutant == pytest.approx(0.0)
    assert score.infrastructure_unknown is False


def test_oracle_failure_is_infrastructure_unknown() -> None:
    from whetstone.envs.ed1m_oracle import score_ed1m_reconstruction

    def unavailable(*, source: str, input_text: str, timeout_seconds: float):
        raise SubprocessStartError("subprocess unavailable")

    score = score_ed1m_reconstruction(
        reconstruction=_MUTATED_SOURCE,
        mutant=_mutant_record(),
        run_in_subprocess=unavailable,
        timeout_seconds=5.0,
    )

    assert score.infrastructure_unknown is True
    assert score.fidelity_to_mutant is None
    assert score.attractor_pull is None


def test_build_uses_content_and_dataset_identities(
    mutant_dataset_dir: Path,
) -> None:
    from whetstone.envs.ed1 import build_ed1_procedure_config
    from whetstone.envs.ed1_blended import BoundedCompressionMetricConfig
    from whetstone.envs.ed1m import (
        Ed1mExperiment,
        build_ed1m_experiment,
        build_ed1m_procedure_config,
    )

    loaded = load_dataset(mutant_dataset_dir)
    record = loaded.records[0]
    experiment = build_ed1m_experiment(
        artifact_dir=mutant_dataset_dir,
        internal_n=1,
        official_n=1,
        blend_config=BoundedCompressionMetricConfig(weight=0.1),
    )

    assert isinstance(experiment, Ed1mExperiment)
    assert experiment.env_name == "ed1m"
    assert experiment.budget_ratio is None
    rollout = experiment.encdec_rollout
    assert rollout is not None and rollout.budget_rule is None
    assert tuple(experiment.mutants) == (record.content_identity,)
    assert experiment.eval_configs.internal.instances[0].id == (
        record.content_identity
    )
    assert experiment.dataset_revision == loaded.manifest.dataset_identity
    assert experiment.eval_configs.internal.task_set.dataset_revision == (
        loaded.manifest.dataset_identity
    )
    assert experiment.eval_configs.official.task_set.dataset_revision == (
        loaded.manifest.dataset_identity
    )
    assert experiment.blend_config is not None
    # The ADVERTISED policy must be the one reward time applies: a blended
    # ed1m cell advertises the blended policy (scoped to the ed1m env name),
    # never the fidelity-only one.
    from whetstone.envs.ed1 import build_ed1_blended_reward_policy
    from whetstone.envs.ed1m import ED1M_ENV_NAME, build_ed1m_reward_policy

    assert experiment.reward_policy == build_ed1_blended_reward_policy(
        experiment.blend_config, env_name=ED1M_ENV_NAME
    )
    assert experiment.reward_policy != build_ed1m_reward_policy()

    unblended = build_ed1m_experiment(
        artifact_dir=mutant_dataset_dir, internal_n=1, official_n=1
    )
    assert unblended.reward_policy == build_ed1m_reward_policy()

    ed1m_procedure = build_ed1m_procedure_config()
    assert rollout.procedure_config_hash == ed1m_procedure.config_identity_hash
    assert ed1m_procedure.definition_ref.definition_id == (
        "whetstone.ed1m.procedure"
    )
    assert ed1m_procedure.config_identity_hash != (
        build_ed1_procedure_config().config_identity_hash
    )


def test_ed1m_eval_rewards_fidelity_reports_attractor(
    mutant_dataset_dir: Path,
) -> None:
    from tests.envs.support import (
        evaluation_binding,
        execution_policy,
        process_row_job_factory,
    )
    from whetstone.envs.ed1 import ed1_initial_candidate
    from whetstone.envs.ed1_eval import run_ed1_eval
    from whetstone.envs.ed1m import (
        ED1M_FIDELITY_NAME,
        build_ed1m_experiment,
    )
    from whetstone.optimization.mutation import MUTATION_FIELD

    experiment = build_ed1m_experiment(
        artifact_dir=mutant_dataset_dir,
        internal_n=1,
        official_n=1,
        repeats=1,
    )

    evaluation = run_ed1_eval(
        experiment,
        candidate_template=str(
            ed1_initial_candidate().payload[MUTATION_FIELD]
        ),
        candidate_id="ed1m-naive",
        sampling=experiment.eval_configs.internal,
        execution_policy=execution_policy(max_attempts=1),
        row_job_factory=process_row_job_factory(
            "tests.envs.process_workers:drive_ed1_success"
        ),
        evaluation_binding=evaluation_binding(
            experiment.eval_configs.internal
        ),
    )

    assert evaluation.per_task_scores[0] >= 0.9
    assert evaluation.per_task_attractor == pytest.approx((0.0,))
    assert evaluation.primary_aggregate.name == ED1M_FIDELITY_NAME
    assert (
        evaluation.primary_aggregate.aggregation_output.value
        == pytest.approx(1.0)
    )
    assert evaluation.reward is not None
    assert evaluation.reward.input_citations[0].name == ED1M_FIDELITY_NAME
