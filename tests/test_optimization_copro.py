from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from dr_code.humaneval.sampling import write_human_eval_snapshot_rows
from typer.testing import CliRunner

from whetstone.optimization import copro_operator
from whetstone.optimization.copro import (
    CoproCandidate,
    CoproCandidateResult,
    CoproLifecycle,
    CoproPin,
    CoproPinLossError,
    CoproRunConfig,
    baseline_candidate,
    manual_proposals,
    run_copro_loop,
    summarize_pinned_candidates,
)
from whetstone.optimization.copro_checkpoint import (
    CoproCheckpointStore,
    CoproInputPins,
    CoproRunManifest,
)
from whetstone.optimization.copro_operator import (
    CoproSpecConfiguration,
    build_candidate_specs,
)
from whetstone.records import PredictionSpecRecord


class _Rows:
    def rows(
        self, member: str, *, where: str = "", params: tuple[Any, ...] = ()
    ) -> tuple[dict[str, Any], ...]:
        if member == "predictions":
            assert where == "experiment_id = ?"
            assert params == ("exp",)
        return {
            "predictions": (
                {
                    "prediction_id": "task-a-repeat-0",
                    "candidate_id": "shared-a",
                },
                {
                    "prediction_id": "task-b-repeat-0",
                    "candidate_id": "shared-a",
                },
                {
                    "prediction_id": "task-a-repeat-1",
                    "candidate_id": "shared-b",
                },
            ),
            "generation_runs": (
                {"prediction_id": "task-a-repeat-0", "status": "success"},
                {"prediction_id": "task-b-repeat-0", "status": "error"},
                {"prediction_id": "task-a-repeat-1", "status": "success"},
            ),
            "score_attempts": (
                {
                    "prediction_id": "task-a-repeat-0",
                    "status": "success",
                    "score": 1.0,
                },
                {
                    "prediction_id": "task-b-repeat-0",
                    "status": "error",
                    "score": None,
                },
                {
                    "prediction_id": "task-a-repeat-1",
                    "status": "success",
                    "score": 0.0,
                },
            ),
        }[member]


def test_multi_task_predictions_aggregate_by_shared_candidate() -> None:
    results = summarize_pinned_candidates(
        cast("Any", _Rows()), experiment_name="exp"
    )
    assert results == (
        CoproCandidateResult("shared-a", 1, 1, 1, 1),
        CoproCandidateResult("shared-b", 1, 0, 0, 0),
    )


def _task(index: int) -> dict[str, str]:
    return {
        "task_id": f"HumanEval/{index}",
        "prompt": f"def add_{index}(value):\n",
        "canonical_solution": f"    return value + {index}\n",
        "entry_point": f"add_{index}",
        "test": (
            "def check(candidate):\n"
            "    inputs = [(1,)]\n"
            f"    results = [{1 + index}]\n"
            "    for inp, expected in zip(inputs, results):\n"
            "        assertion(candidate(*inp), expected)\n"
        ),
    }


def test_candidate_identity_is_shared_across_tasks_and_repeats(
    tmp_path: Path,
) -> None:
    snapshot = write_human_eval_snapshot_rows(
        [_task(1), _task(2)],
        snapshot_path=tmp_path / "snapshot.json",
        dataset_name="local/fixture",
    )
    (tmp_path / "model.json").write_text(
        json.dumps(
            {
                "name": "fixture",
                "providers": [
                    {"model": "encoder", "config_id": "encoder"},
                    {"model": "decoder", "config_id": "decoder"},
                ],
            }
        )
    )
    (tmp_path / "split.json").write_text(
        json.dumps(
            {
                "name": "fixture",
                "dataset": {
                    "name": "local/fixture",
                    "split": "test",
                    "snapshot_path": str(snapshot),
                    "sample_seed": 0,
                    "sample_count": 2,
                },
            }
        )
    )
    candidates = manual_proposals(
        baseline_candidate("run"), run_id="run", breadth=2, depth=0
    )

    specs = build_candidate_specs(
        CoproSpecConfiguration(
            model_config_path=Path("model.json"),
            split_path=Path("split.json"),
            configs_root=tmp_path,
            compression_targets=(0.5,),
            repetition_seeds=(0, 1),
        ),
        run_id="run",
        experiment_name="exp",
        candidates=candidates,
    )

    candidate_ids = [spec.dimensions.values["candidate_id"] for spec in specs]
    assert len(specs) == 8
    counts = {
        candidate_id: candidate_ids.count(candidate_id)
        for candidate_id in set(candidate_ids)
    }
    assert counts == {
        candidate.candidate_id: 4 for candidate in candidates
    }


class _Lifecycle(CoproLifecycle):
    def __init__(self, *, lose_pin: bool = False) -> None:
        self.calls: list[str] = []
        self.lose_pin = lose_pin
        self.candidate_results: tuple[CoproCandidateResult, ...] = ()

    def submit_generation(self, **kwargs: Any) -> None:
        self.calls.append("submit_generation")

    def wait(self, operation_key: str) -> None:
        suffix = (
            "generation" if operation_key.endswith("generation") else "scoring"
        )
        self.calls.append(f"wait_{suffix}")

    def submit_scoring(self, **kwargs: Any) -> None:
        self.calls.append("submit_scoring")

    def promote_acceptance(self, experiment_name: str) -> None:
        self.calls.append("promote_acceptance")

    def export_and_pin(self) -> CoproPin:
        self.calls.append("export_and_pin")
        return CoproPin("bundle", 7, object())

    def read_pinned_candidates(
        self, pin: CoproPin, *, experiment_name: str
    ) -> tuple[CoproCandidateResult, ...]:
        self.calls.append("read_pinned_candidates")
        if self.lose_pin:
            raise CoproPinLossError
        return self.candidate_results


def _specs(*_args: Any) -> tuple[PredictionSpecRecord, ...]:
    return (cast("PredictionSpecRecord", object()),)


def _factory(lifecycle: _Lifecycle):
    def factory(
        _experiment: str, candidates: tuple[CoproCandidate, ...]
    ) -> tuple[PredictionSpecRecord, ...]:
        lifecycle.candidate_results = tuple(
            CoproCandidateResult(candidate.candidate_id, 1, 1, 0, 0)
            for candidate in candidates
        )
        return _specs()

    return factory


def _pins(*, model_digest: str = "model") -> CoproInputPins:
    return CoproInputPins(
        model_config_digest=model_digest,
        split_config_digest="split",
        dataset_snapshot_digest="snapshot",
        dataset_snapshot_header_digest="header",
        compression_targets=(0.5,),
        repetition_seeds=(0,),
        min_encoder_char_budget=50,
    )


def test_iteration_uses_exact_typed_lifecycle_order() -> None:
    lifecycle = _Lifecycle()

    run_copro_loop(
        config=CoproRunConfig(run_id="run", breadth=2, depth=2),
        lifecycle=lifecycle,
        spec_factory=_factory(lifecycle),
    )

    expected = [
        "submit_generation",
        "wait_generation",
        "submit_scoring",
        "wait_scoring",
        "promote_acceptance",
        "export_and_pin",
        "read_pinned_candidates",
    ]
    assert lifecycle.calls == expected + expected


def test_pin_loss_stops_before_depth_advances() -> None:
    lifecycle = _Lifecycle(lose_pin=True)

    with pytest.raises(CoproPinLossError, match="PINNED_BUNDLE_GONE"):
        run_copro_loop(
            config=CoproRunConfig(run_id="run", breadth=2, depth=2),
            lifecycle=lifecycle,
            spec_factory=_specs,
        )

    assert lifecycle.calls[-1] == "read_pinned_candidates"
    assert lifecycle.calls.count("submit_generation") == 1


def test_dry_run_is_zero_spend_and_writes_durable_outputs(
    tmp_path: Path,
) -> None:
    lifecycle = _Lifecycle()
    checkpoints: list[Path] = []
    config = CoproRunConfig(
        run_id="zero-spend", breadth=2, depth=2, dry_run=True
    )
    store = CoproCheckpointStore(
        tmp_path,
        manifest=CoproRunManifest.create(
            run_config=config, input_pins=_pins()
        ),
    )
    assert store.load_or_initialize() is None

    def checkpoint(result: Any) -> None:
        paths = store.commit(result)
        checkpoints.append(paths["run"])

    result = run_copro_loop(
        config=config,
        lifecycle=lifecycle,
        spec_factory=_specs,
        checkpoint=checkpoint,
    )

    assert lifecycle.calls == []
    assert result.best_candidate is None
    assert len(checkpoints) == 2
    paths = store.artifact_paths(result)
    assert json.loads(paths["run"].read_text())["dry_run"] is True
    assert paths["candidates"].read_text().count("\n") == 4
    assert paths["attempts"].is_file()
    assert paths["best_prompt"].is_file()


def test_cli_dry_run_never_resolves_runtime_or_integrity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        copro_operator,
        "build_candidate_specs",
        lambda *_args, **_kwargs: _specs(),
    )
    monkeypatch.setattr(
        copro_operator,
        "prepare_copro_inputs",
        lambda _configuration: SimpleNamespace(pins=_pins()),
    )

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("zero-spend CLI touched live runtime")

    monkeypatch.setattr(
        copro_operator, "resolve_application_database_url", forbidden
    )
    monkeypatch.setattr(
        copro_operator,
        "required_bundle_integrity_configuration",
        forbidden,
    )

    result = CliRunner().invoke(
        copro_operator.APP,
        [
            "--model-config",
            "model.json",
            "--split",
            "split.json",
            "--compression-target",
            "0.5",
            "--breadth",
            "2",
            "--depth",
            "1",
            "--run-id",
            "zero-spend-cli",
            "--output-dir",
            str(tmp_path),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["dry_run"] is True
    run_path = Path(json.loads(result.output)["artifacts"]["run"])
    assert json.loads(run_path.read_text())["run_id"] == "zero-spend-cli"
