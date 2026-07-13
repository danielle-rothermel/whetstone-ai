from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, cast

import pytest

from whetstone.optimization.copro import (
    CoproCandidate,
    CoproCandidateResult,
    CoproLifecycle,
    CoproPin,
    CoproRunConfig,
    CoproRunResult,
    run_copro_loop,
)
from whetstone.optimization.copro_checkpoint import (
    CoproCheckpointError,
    CoproCheckpointStore,
    CoproInputPins,
    CoproRunManifest,
)
from whetstone.records import PredictionSpecRecord


class InjectedCrash(RuntimeError):
    pass


class CrashOnce:
    def __init__(self, boundary: str) -> None:
        self.boundary = boundary
        self.triggered = False

    def __call__(self, boundary: str) -> None:
        if boundary == self.boundary and not self.triggered:
            self.triggered = True
            raise InjectedCrash(boundary)


def _pins(*, model: str = "model") -> CoproInputPins:
    return CoproInputPins(
        model_config_digest=model,
        split_config_digest="split",
        dataset_snapshot_digest="snapshot",
        dataset_snapshot_header_digest="header",
        compression_targets=(0.5,),
        repetition_seeds=(0,),
        min_encoder_char_budget=50,
    )


def _manifest(
    config: CoproRunConfig, *, model: str = "model"
) -> CoproRunManifest:
    return CoproRunManifest.create(
        run_config=config, input_pins=_pins(model=model)
    )


def _specs(*_args: Any) -> tuple[PredictionSpecRecord, ...]:
    return (cast("PredictionSpecRecord", object()),)


def _complete(
    store: CoproCheckpointStore,
    config: CoproRunConfig,
    *,
    resume: CoproRunResult | None = None,
) -> CoproRunResult:
    return run_copro_loop(
        config=config,
        lifecycle=None,
        spec_factory=_specs,
        checkpoint=store.commit,
        resume=resume,
    )


@pytest.mark.parametrize(
    "boundary",
    CoproCheckpointStore.MANIFEST_BOUNDARIES
    + CoproCheckpointStore.CHECKPOINT_BOUNDARIES,
)
def test_every_write_boundary_recovers_to_one_complete_depth(
    tmp_path: Path, boundary: str
) -> None:
    config = CoproRunConfig(
        run_id="crash", breadth=2, depth=1, dry_run=True
    )
    crash = CrashOnce(boundary)
    store = CoproCheckpointStore(
        tmp_path, manifest=_manifest(config), failure_injector=crash
    )

    with pytest.raises(InjectedCrash, match=boundary):
        resume = store.load_or_initialize()
        _complete(store, config, resume=resume)

    recovered = CoproCheckpointStore(tmp_path, manifest=_manifest(config))
    resume = recovered.load_or_initialize()
    result = _complete(recovered, config, resume=resume)

    assert len(result.iterations) == 1
    assert recovered.load_or_initialize() == result
    assert not tuple(tmp_path.glob(".run-manifest.*.tmp"))
    assert not tuple((tmp_path / "checkpoints").glob(".depth-*.tmp"))


class RecordingLifecycle(CoproLifecycle):
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.results: tuple[CoproCandidateResult, ...] = ()

    def submit_generation(self, **kwargs: Any) -> None:
        self.calls.append(f"submit:{kwargs['operation_key']}")

    def wait(self, operation_key: str) -> None:
        self.calls.append(f"wait:{operation_key}")

    def submit_scoring(self, **kwargs: Any) -> None:
        self.calls.append(f"submit:{kwargs['operation_key']}")

    def promote_acceptance(self, experiment_name: str) -> None:
        self.calls.append(f"accept:{experiment_name}")

    def export_and_pin(self) -> CoproPin:
        self.calls.append("export")
        return CoproPin("bundle", 1, object())

    def read_pinned_candidates(
        self, pin: CoproPin, *, experiment_name: str
    ) -> tuple[CoproCandidateResult, ...]:
        self.calls.append(f"read:{experiment_name}")
        return self.results


def _live_factory(lifecycle: RecordingLifecycle):
    def factory(
        _experiment: str, candidates: tuple[CoproCandidate, ...]
    ) -> tuple[PredictionSpecRecord, ...]:
        lifecycle.results = tuple(
            CoproCandidateResult(candidate.candidate_id, 1, 1, 0, 0)
            for candidate in candidates
        )
        return _specs()

    return factory


def test_resume_starts_after_last_atomically_committed_depth(
    tmp_path: Path,
) -> None:
    config = CoproRunConfig(run_id="resume", breadth=2, depth=2)
    crash = CrashOnce("checkpoint.after_commit")
    first_lifecycle = RecordingLifecycle()
    first = CoproCheckpointStore(
        tmp_path, manifest=_manifest(config), failure_injector=crash
    )
    assert first.load_or_initialize() is None
    with pytest.raises(InjectedCrash):
        run_copro_loop(
            config=config,
            lifecycle=first_lifecycle,
            spec_factory=_live_factory(first_lifecycle),
            checkpoint=first.commit,
        )

    second_lifecycle = RecordingLifecycle()
    second = CoproCheckpointStore(tmp_path, manifest=_manifest(config))
    resume = second.load_or_initialize()
    assert resume is not None
    assert len(resume.iterations) == 1
    result = run_copro_loop(
        config=config,
        lifecycle=second_lifecycle,
        spec_factory=_live_factory(second_lifecycle),
        checkpoint=second.commit,
        resume=resume,
    )

    assert len(result.iterations) == 2
    assert all("d0" not in call for call in second_lifecycle.calls)
    assert any("d1" in call for call in second_lifecycle.calls)


def test_completed_rerun_is_exactly_idempotent(tmp_path: Path) -> None:
    config = CoproRunConfig(
        run_id="idempotent", breadth=2, depth=2, dry_run=True
    )
    store = CoproCheckpointStore(tmp_path, manifest=_manifest(config))
    assert store.load_or_initialize() is None
    result = _complete(store, config)
    evidence = {
        path: (
            path.stat().st_mtime_ns,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in tmp_path.rglob("*")
        if path.is_file()
    }

    reopened = CoproCheckpointStore(tmp_path, manifest=_manifest(config))
    resume = reopened.load_or_initialize()
    assert resume == result

    def forbidden(*_args: Any) -> tuple[PredictionSpecRecord, ...]:
        raise AssertionError("completed run rebuilt specs")

    replay = run_copro_loop(
        config=config,
        lifecycle=None,
        spec_factory=forbidden,
        checkpoint=reopened.commit,
        resume=resume,
    )
    assert replay == result
    assert evidence == {
        path: (
            path.stat().st_mtime_ns,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in evidence
    }


@pytest.mark.parametrize("change", ["config", "inputs"])
def test_incompatible_resume_fails_before_execution(
    tmp_path: Path, change: str
) -> None:
    config = CoproRunConfig(
        run_id="incompatible", breadth=2, depth=1, dry_run=True
    )
    store = CoproCheckpointStore(tmp_path, manifest=_manifest(config))
    store.load_or_initialize()
    _complete(store, config)
    requested_config = (
        config.model_copy(update={"breadth": 3})
        if change == "config"
        else config
    )
    requested_manifest = _manifest(
        requested_config, model="different" if change == "inputs" else "model"
    )

    with pytest.raises(CoproCheckpointError, match="incompatible"):
        CoproCheckpointStore(
            tmp_path, manifest=requested_manifest
        ).load_or_initialize()


@pytest.mark.parametrize("mutation", ["tamper", "partial", "gap"])
def test_mixed_or_partial_committed_artifacts_fail_closed(
    tmp_path: Path, mutation: str
) -> None:
    config = CoproRunConfig(
        run_id="mixed", breadth=2, depth=1, dry_run=True
    )
    store = CoproCheckpointStore(tmp_path, manifest=_manifest(config))
    store.load_or_initialize()
    _complete(store, config)
    depth = tmp_path / "checkpoints" / "depth-000"
    if mutation == "tamper":
        (depth / "candidates.jsonl").write_text("tampered\n")
    elif mutation == "partial":
        (depth / "attempts.csv").unlink()
    else:
        depth.rename(tmp_path / "checkpoints" / "depth-001")

    with pytest.raises(CoproCheckpointError):
        CoproCheckpointStore(
            tmp_path, manifest=_manifest(config)
        ).load_or_initialize()


def test_staging_artifacts_are_recovered_not_mixed(tmp_path: Path) -> None:
    config = CoproRunConfig(
        run_id="staging", breadth=2, depth=1, dry_run=True
    )
    store = CoproCheckpointStore(tmp_path, manifest=_manifest(config))
    store.load_or_initialize()
    staging = tmp_path / "checkpoints" / ".depth-000.crashed.tmp"
    staging.mkdir()
    (staging / "run.json").write_text("partial")

    reopened = CoproCheckpointStore(tmp_path, manifest=_manifest(config))
    assert reopened.load_or_initialize() is None
    assert not staging.exists()
