"""Atomic, immutable depth checkpoints for the COPRO operator."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Literal, cast

from dr_serialize import Jsonable, sha256_json_digest
from pydantic import (
    BaseModel,
    ConfigDict,
    StrictInt,
    StrictStr,
    model_validator,
)

from whetstone.optimization.copro import (
    CoproRunConfig,
    CoproRunResult,
    render_copro_artifacts,
)

CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_STATE_MACHINE = "copro_depth_checkpoint_v1"
_DEPTH_PATTERN = re.compile(r"depth-(\d{3})")
_LEGACY_ARTIFACTS = frozenset(
    {"run.json", "candidates.jsonl", "attempts.csv", "best_prompt.json"}
)


class CoproCheckpointError(RuntimeError):
    """Persisted COPRO state is missing, mixed, or incompatible."""


class CoproInputPins(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_config_digest: StrictStr
    split_config_digest: StrictStr
    dataset_snapshot_digest: StrictStr
    dataset_snapshot_header_digest: StrictStr
    compression_targets: tuple[float, ...]
    repetition_seeds: tuple[StrictInt, ...]
    min_encoder_char_budget: StrictInt


class CoproRunManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: StrictInt = CHECKPOINT_SCHEMA_VERSION
    state_machine: Literal["copro_depth_checkpoint_v1"] = (
        CHECKPOINT_STATE_MACHINE
    )
    run_config: CoproRunConfig
    input_pins: CoproInputPins
    manifest_digest: StrictStr

    @classmethod
    def create(
        cls, *, run_config: CoproRunConfig, input_pins: CoproInputPins
    ) -> CoproRunManifest:
        payload = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "state_machine": CHECKPOINT_STATE_MACHINE,
            "run_config": run_config.model_dump(mode="json"),
            "input_pins": input_pins.model_dump(mode="json"),
        }
        return cls(
            run_config=run_config,
            input_pins=input_pins,
            manifest_digest=sha256_json_digest(payload),
        )

    @model_validator(mode="after")
    def validate_digest(self) -> CoproRunManifest:
        if self.schema_version != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("unsupported COPRO checkpoint schema")
        payload = self.model_dump(mode="json", exclude={"manifest_digest"})
        if self.manifest_digest != sha256_json_digest(payload):
            raise ValueError("COPRO run manifest digest is invalid")
        return self


class CoproCheckpoint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: StrictInt = CHECKPOINT_SCHEMA_VERSION
    state: Literal["depth_committed"] = "depth_committed"
    manifest_digest: StrictStr
    depth: StrictInt
    result: CoproRunResult
    artifact_digests: dict[StrictStr, StrictStr]
    checkpoint_digest: StrictStr

    @classmethod
    def create(
        cls,
        *,
        manifest_digest: str,
        depth: int,
        result: CoproRunResult,
        artifact_digests: Mapping[str, str],
    ) -> CoproCheckpoint:
        payload = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "state": "depth_committed",
            "manifest_digest": manifest_digest,
            "depth": depth,
            "result": result.model_dump(mode="json"),
            "artifact_digests": dict(artifact_digests),
        }
        return cls(
            manifest_digest=manifest_digest,
            depth=depth,
            result=result,
            artifact_digests=dict(artifact_digests),
            checkpoint_digest=sha256_json_digest(cast("Jsonable", payload)),
        )

    @model_validator(mode="after")
    def validate_digest(self) -> CoproCheckpoint:
        if self.schema_version != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("unsupported COPRO checkpoint schema")
        payload = self.model_dump(mode="json", exclude={"checkpoint_digest"})
        if self.checkpoint_digest != sha256_json_digest(payload):
            raise ValueError("COPRO depth checkpoint digest is invalid")
        return self


FailureInjector = Callable[[str], None]


class CoproCheckpointStore:
    """Publish complete depth generations without overwriting evidence."""

    MANIFEST_BOUNDARIES = (
        "manifest.after_write",
        "manifest.after_commit",
    )
    CHECKPOINT_BOUNDARIES = (
        "checkpoint.after_run",
        "checkpoint.after_candidates",
        "checkpoint.after_attempts",
        "checkpoint.after_best_prompt",
        "checkpoint.after_metadata",
        "checkpoint.after_stage_fsync",
        "checkpoint.after_commit",
    )

    def __init__(
        self,
        output_dir: Path,
        *,
        manifest: CoproRunManifest,
        failure_injector: FailureInjector | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.manifest = manifest
        self._failure_injector = failure_injector
        self._manifest_path = output_dir / "run-manifest.json"
        self._checkpoints = output_dir / "checkpoints"

    def load_or_initialize(self) -> CoproRunResult | None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._recover_staging()
        legacy = _LEGACY_ARTIFACTS.intersection(
            path.name for path in self.output_dir.iterdir()
        )
        if legacy:
            raise CoproCheckpointError(
                "legacy or mixed COPRO artifacts require a new output "
                "directory"
            )
        committed = self._committed_depth_paths()
        if self._manifest_path.exists():
            observed = self._load_manifest()
            if observed != self.manifest:
                raise CoproCheckpointError(
                    "COPRO run manifest is incompatible with requested inputs"
                )
        else:
            if committed:
                raise CoproCheckpointError(
                    "COPRO checkpoints exist without a run manifest"
                )
            self._commit_manifest()
        self._checkpoints.mkdir(exist_ok=True)
        _fsync_directory(self.output_dir)
        return self._load_committed_depths()

    def commit(self, result: CoproRunResult) -> Mapping[str, Path]:
        if not result.iterations:
            raise CoproCheckpointError("cannot commit an empty COPRO result")
        if result.run_id != self.manifest.run_config.run_id:
            raise CoproCheckpointError(
                "checkpoint run ID does not match manifest"
            )
        depth = len(result.iterations) - 1
        if result.iterations[-1].depth != depth:
            raise CoproCheckpointError(
                "checkpoint result depth is inconsistent"
            )
        existing = self._generation_path(depth)
        if existing.exists():
            checkpoint = self._load_generation(existing, depth=depth)
            if checkpoint.result != result:
                raise CoproCheckpointError(
                    "completed COPRO depth cannot be overwritten"
                )
            return self._artifact_paths(existing)
        committed = self._committed_depth_paths()
        expected_prior = list(range(depth))
        if [item[0] for item in committed] != expected_prior:
            raise CoproCheckpointError(
                "COPRO depth commit is not contiguous with durable state"
            )
        if depth:
            prior = self._load_generation(committed[-1][1], depth=depth - 1)
            if result.iterations[:-1] != prior.result.iterations:
                raise CoproCheckpointError(
                    "COPRO depth does not extend the committed predecessor"
                )

        self._checkpoints.mkdir(exist_ok=True)
        staging = self._checkpoints / (
            f".depth-{depth:03d}.{uuid.uuid4().hex}.tmp"
        )
        staging.mkdir()
        artifacts = render_copro_artifacts(result)
        digests: dict[str, str] = {}
        boundary_for = {
            "run.json": "checkpoint.after_run",
            "candidates.jsonl": "checkpoint.after_candidates",
            "attempts.csv": "checkpoint.after_attempts",
            "best_prompt.json": "checkpoint.after_best_prompt",
        }
        for name, content in artifacts.items():
            _write_fsynced(staging / name, content)
            digests[name] = _sha256_text(content)
            self._inject(boundary_for[name])
        checkpoint = CoproCheckpoint.create(
            manifest_digest=self.manifest.manifest_digest,
            depth=depth,
            result=result,
            artifact_digests=digests,
        )
        _write_fsynced(
            staging / "checkpoint.json",
            checkpoint.model_dump_json(indent=2) + "\n",
        )
        self._inject("checkpoint.after_metadata")
        _fsync_directory(staging)
        self._inject("checkpoint.after_stage_fsync")
        try:
            staging.rename(existing)
        except FileExistsError:
            shutil.rmtree(staging)
            observed = self._load_generation(existing, depth=depth)
            if observed.result != result:
                raise CoproCheckpointError(
                    "concurrent COPRO checkpoint has incompatible evidence"
                ) from None
        _fsync_directory(self._checkpoints)
        self._inject("checkpoint.after_commit")
        return self._artifact_paths(existing)

    def artifact_paths(self, result: CoproRunResult) -> Mapping[str, Path]:
        if not result.iterations:
            return {}
        path = self._generation_path(len(result.iterations) - 1)
        checkpoint = self._load_generation(
            path, depth=len(result.iterations) - 1
        )
        if checkpoint.result != result:
            raise CoproCheckpointError(
                "requested COPRO result is not the committed generation"
            )
        return self._artifact_paths(path)

    def _commit_manifest(self) -> None:
        temporary = self.output_dir / f".run-manifest.{uuid.uuid4().hex}.tmp"
        _write_fsynced(
            temporary, self.manifest.model_dump_json(indent=2) + "\n"
        )
        self._inject("manifest.after_write")
        try:
            os.link(temporary, self._manifest_path)
        except FileExistsError:
            observed = self._load_manifest()
            if observed != self.manifest:
                raise CoproCheckpointError(
                    "concurrent COPRO manifest is incompatible"
                ) from None
        finally:
            temporary.unlink(missing_ok=True)
        _fsync_directory(self.output_dir)
        self._inject("manifest.after_commit")

    def _load_manifest(self) -> CoproRunManifest:
        try:
            return CoproRunManifest.model_validate_json(
                self._manifest_path.read_bytes()
            )
        except (OSError, ValueError) as error:
            raise CoproCheckpointError(
                "COPRO run manifest is invalid"
            ) from error

    def _load_committed_depths(self) -> CoproRunResult | None:
        committed = self._committed_depth_paths()
        if [item[0] for item in committed] != list(range(len(committed))):
            raise CoproCheckpointError(
                "COPRO checkpoint depths are not contiguous"
            )
        latest: CoproCheckpoint | None = None
        for depth, path in committed:
            observed = self._load_generation(path, depth=depth)
            if latest is not None and (
                observed.result.iterations[:-1] != latest.result.iterations
            ):
                raise CoproCheckpointError(
                    "COPRO checkpoint generations have mixed history"
                )
            latest = observed
        return None if latest is None else latest.result

    def _load_generation(self, path: Path, *, depth: int) -> CoproCheckpoint:
        if not path.is_dir():
            raise CoproCheckpointError(
                "COPRO checkpoint entry is not a directory"
            )
        expected_files = {*_LEGACY_ARTIFACTS, "checkpoint.json"}
        observed_files = {item.name for item in path.iterdir()}
        if observed_files != expected_files:
            raise CoproCheckpointError(
                f"COPRO depth {depth} has partial or mixed artifacts"
            )
        try:
            checkpoint = CoproCheckpoint.model_validate_json(
                (path / "checkpoint.json").read_bytes()
            )
        except (OSError, ValueError) as error:
            raise CoproCheckpointError(
                f"COPRO depth {depth} metadata is invalid"
            ) from error
        if (
            checkpoint.manifest_digest != self.manifest.manifest_digest
            or checkpoint.depth != depth
            or len(checkpoint.result.iterations) != depth + 1
            or checkpoint.result.iterations[-1].depth != depth
        ):
            raise CoproCheckpointError(
                f"COPRO depth {depth} identity is incompatible"
            )
        if set(checkpoint.artifact_digests) != _LEGACY_ARTIFACTS:
            raise CoproCheckpointError(
                f"COPRO depth {depth} artifact inventory is invalid"
            )
        for name, expected in checkpoint.artifact_digests.items():
            if _sha256_bytes((path / name).read_bytes()) != expected:
                raise CoproCheckpointError(
                    f"COPRO depth {depth} artifact digest mismatch: {name}"
                )
        run_result = CoproRunResult.model_validate_json(
            (path / "run.json").read_bytes()
        )
        if run_result != checkpoint.result:
            raise CoproCheckpointError(
                f"COPRO depth {depth} run view does not match checkpoint"
            )
        return checkpoint

    def _committed_depth_paths(self) -> list[tuple[int, Path]]:
        if not self._checkpoints.exists():
            return []
        committed: list[tuple[int, Path]] = []
        for path in self._checkpoints.iterdir():
            match = _DEPTH_PATTERN.fullmatch(path.name)
            if match is None:
                if path.name.startswith(".depth-") and path.name.endswith(
                    ".tmp"
                ):
                    continue
                raise CoproCheckpointError(
                    f"unexpected COPRO checkpoint entry: {path.name}"
                )
            committed.append((int(match.group(1)), path))
        return sorted(committed)

    def _recover_staging(self) -> None:
        for path in self.output_dir.glob(".run-manifest.*.tmp"):
            path.unlink(missing_ok=True)
        if self._checkpoints.exists():
            for path in self._checkpoints.glob(".depth-*.tmp"):
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink(missing_ok=True)

    def _generation_path(self, depth: int) -> Path:
        return self._checkpoints / f"depth-{depth:03d}"

    @staticmethod
    def _artifact_paths(path: Path) -> Mapping[str, Path]:
        return {
            name.removesuffix(Path(name).suffix): path / name
            for name in _LEGACY_ARTIFACTS
        }

    def _inject(self, boundary: str) -> None:
        if self._failure_injector is not None:
            self._failure_injector(boundary)


def _write_fsynced(path: Path, content: str) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sha256_text(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()
