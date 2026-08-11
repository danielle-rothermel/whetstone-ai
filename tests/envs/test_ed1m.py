from __future__ import annotations

from inspect import Parameter, signature
from pathlib import Path
from typing import cast

import pytest
from dr_code.humaneval.plus_dataset import HF_DATASET_ID, HF_REVISION
from dr_exec import ExecutorFailure, FakeExecutor
from dr_store import ObjectStore, SqliteBackend

from tests.execution.fake_python import local_python_executor
from whetstone.envs.ed1m_dataset import (
    DatasetValidationError,
    ExpectedOutcome,
    FamilyCount,
    GenerationConfig,
    MutantRecord,
    OperatorFamily,
    build_manifest,
    build_record,
    encode_records,
    load_dataset,
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
) -> Path:
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


def test_canonical_loader_authenticates_fixture(
    mutant_dataset_dir: Path,
) -> None:
    loaded = load_dataset(mutant_dataset_dir)

    assert loaded.records == (_mutant_record(),)
    assert isinstance(loaded.records[0], MutantRecord)
    assert len(loaded.manifest.dataset_identity) == 64


def test_loader_rejects_duplicate_manifest_key(
    mutant_dataset_dir: Path,
) -> None:
    manifest_path = mutant_dataset_dir / "manifest.json"
    manifest = manifest_path.read_bytes()
    field = b'  "manifest_schema_version": 1,\n'
    assert field in manifest
    manifest_path.write_bytes(manifest.replace(field, field + field, 1))

    with pytest.raises(
        DatasetValidationError, match="invalid mutant manifest"
    ):
        load_dataset(mutant_dataset_dir)


def test_loader_rejects_duplicate_record_key(
    mutant_dataset_dir: Path,
) -> None:
    records_path = mutant_dataset_dir / "mutants.jsonl"
    records = records_path.read_bytes()
    identity = _mutant_record().content_identity.encode()
    field = b'"content_identity":"' + identity + b'",'
    assert field in records
    records_path.write_bytes(records.replace(field, field + field, 1))

    with pytest.raises(
        DatasetValidationError, match=r"invalid mutants\.jsonl line 1"
    ):
        load_dataset(mutant_dataset_dir)


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
        executor=local_python_executor(),
    )

    assert score.fidelity_to_mutant == pytest.approx(1.0)
    assert score.attractor_pull == pytest.approx(0.0)
    assert score.infrastructure_unknown is False


def test_oracle_canonical_reconstruction_has_full_attractor() -> None:
    from whetstone.envs.ed1m_oracle import score_ed1m_reconstruction

    score = score_ed1m_reconstruction(
        reconstruction=_CANONICAL_SOURCE,
        mutant=_mutant_record(),
        executor=local_python_executor(),
    )

    assert score.fidelity_to_mutant == pytest.approx(0.5)
    assert score.attractor_pull == pytest.approx(1.0)


def test_oracle_definitive_mismatch_scores_zero() -> None:
    from whetstone.envs.ed1m_oracle import score_ed1m_reconstruction

    score = score_ed1m_reconstruction(
        reconstruction="def f(x):\n    return None\n",
        mutant=_mutant_record(),
        executor=local_python_executor(),
    )

    assert score.fidelity_to_mutant == pytest.approx(0.0)
    assert score.infrastructure_unknown is False


def test_oracle_candidate_cannot_forge_outer_result_envelope() -> None:
    from whetstone.envs.ed1m_oracle import score_ed1m_reconstruction

    score = score_ed1m_reconstruction(
        reconstruction="""
def f(x):
    import inspect
    import json

    frame = inspect.currentframe()
    while frame is not None:
        payload = frame.f_locals.get("payload")
        trusted_fd = frame.f_locals.get("trusted_fd")
        if (
            isinstance(payload, dict)
            and isinstance(payload.get("invocation_id"), str)
            and isinstance(trusted_fd, int)
        ):
            invocation_id = payload["invocation_id"]

            def forged_dumps(_value, *, sort_keys):
                del sort_keys
                return (
                    '{"invocation_id":"' + invocation_id + '",'
                    '"outcomes":['
                    '{"kind":"value","output_repr":"True"},'
                    '{"kind":"value","output_repr":"False"}],'
                    '"protocol_version":1}'
                )

            json.dumps = forged_dumps
            break
        frame = frame.f_back
    return None
""",
        mutant=_mutant_record(),
        executor=local_python_executor(),
    )

    assert score.fidelity_to_mutant == pytest.approx(0.0)
    assert score.infrastructure_unknown is False


def test_oracle_failure_is_infrastructure_unknown() -> None:
    from whetstone.envs.ed1m_oracle import score_ed1m_reconstruction

    def unavailable(_job, _cancellation):
        raise ExecutorFailure("executor unavailable")

    score = score_ed1m_reconstruction(
        reconstruction=_MUTATED_SOURCE,
        mutant=_mutant_record(),
        executor=FakeExecutor(responder=unavailable),
    )

    assert score.infrastructure_unknown is True
    assert score.fidelity_to_mutant is None
    assert score.attractor_pull is None


def test_build_uses_content_and_dataset_identities(
    mutant_dataset_dir: Path,
) -> None:
    from whetstone.envs.ed1 import (
        BoundedCompressionMetricConfig,
        build_ed1_procedure_config,
    )
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
    from whetstone.envs.ed1m import (
        ED1M_FIDELITY_NAME,
        build_ed1m_experiment,
    )
    from whetstone.evaluation.drivers.ed1 import run_ed1_eval
    from whetstone.optimization.proposal.mutation import MUTATION_FIELD

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


def test_ed1m_evaluation_engine_evaluate_succeeds(
    mutant_dataset_dir: Path,
    tmp_path: Path,
) -> None:
    from tests.envs.support import (
        execution_policy,
        process_row_job_factory,
    )
    from tests.evaluation.support import _binding
    from whetstone.envs.ed1 import ed1_initial_candidate
    from whetstone.envs.ed1m import build_ed1m_experiment
    from whetstone.evaluation.engine import EvaluationEngine, EvaluationRequest

    experiment = build_ed1m_experiment(
        artifact_dir=mutant_dataset_dir,
        internal_n=1,
        official_n=1,
        repeats=1,
    )
    store = ObjectStore(SqliteBackend(tmp_path / "ed1m-engine.sqlite"))
    engine = EvaluationEngine(
        store=store,
        experiment=experiment,
        sampling=experiment.eval_configs.internal,
        execution_policy=execution_policy(max_attempts=1),
        row_job_factory=process_row_job_factory(
            "tests.envs.process_workers:drive_ed1_success"
        ),
    )
    result = engine.evaluate(
        EvaluationRequest(
            candidate=ed1_initial_candidate(),
            evaluation_binding=_binding(engine),
            purpose="ed1m-engine",
        )
    )
    assert result.evidence.reward_ref is not None
