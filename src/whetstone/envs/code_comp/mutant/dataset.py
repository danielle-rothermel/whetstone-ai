"""Load and verify retained ED1M behavioral-mutant artifacts.

The loader verifies artifact schemas, hashes, identities, ordering, and
internal consistency. ``canonical_suite_digest`` is opaque recorded
provenance; the external canonical suite is not independently reauthenticated.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from dr_code.humaneval.plus_dataset import HF_DATASET_ID, HF_REVISION
from dr_serialize import StrictJsonDecodeError, decode_strict_json_bytes
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from whetstone.envs.code_comp.mutant.mutation import (
    ALL_FAMILIES,
    MutationError,
    OperatorFamily,
    apply_site,
    iter_sites,
)
from whetstone.evaluation.config import identity_hash_for

DATASET_SCHEMA_VERSION: Final = 1
MANIFEST_SCHEMA_VERSION: Final = 1
GENERATOR_VERSION: Final = "mutants@v1"
RECORDS_FILENAME: Final = "mutants.jsonl"
MANIFEST_FILENAME: Final = "manifest.json"

# These labels authenticate returning immutable artifacts and therefore remain
# byte-for-byte stable even though Whetstone now owns the reader.
_CONFIG_SCHEMA: Final = "dr_code.mutants.generation_config"
_RECORD_SCHEMA: Final = "dr_code.mutants.record"
_DATASET_SCHEMA: Final = "dr_code.mutants.dataset"


class _StrictPersistedModel(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=False,
        extra="forbid",
        frozen=True,
        strict=True,
        validate_assignment=True,
    )


class ExpectedOutcome(_StrictPersistedModel):
    kind: Literal["value", "error"]
    output_repr: str


class MutantRecord(_StrictPersistedModel):
    content_hash: str
    task_id: str
    entry_point: str
    prompt: str
    canonical_full_source: str
    mutated_full_source: str
    operator_family: OperatorFamily
    seed: int
    site_node_path: int
    site_target_index: int
    site_description: str
    input_reprs: tuple[str, ...]
    mutant_expected: tuple[ExpectedOutcome, ...]
    canonical_expected: tuple[ExpectedOutcome, ...]
    distinct_input_indices: tuple[int, ...]
    diff_summary: str
    canonical_test: str

    @property
    def distinct_input_count(self) -> int:
        return len(self.distinct_input_indices)


class GenerationConfig(_StrictPersistedModel):
    generator_version: Literal["mutants@v1"] = GENERATOR_VERSION
    dataset_schema_version: Literal[1] = DATASET_SCHEMA_VERSION
    dataset_id: str
    dataset_revision: str
    operator_families: tuple[OperatorFamily, ...]
    seeds: int
    max_inputs_per_mutant: int
    timeout_seconds: float
    task_ids: tuple[str, ...]
    canonical_suite_digest: str
    runner_label: str
    runtime_label: str

    @field_validator("timeout_seconds", mode="before")
    @classmethod
    def _require_json_float(cls, value: object) -> object:
        if type(value) is not float:
            raise ValueError("timeout_seconds must be a JSON float")
        return value

    def identity_hash(self) -> str:
        return identity_hash_for(
            schema=_CONFIG_SCHEMA,
            payload=self.model_dump(mode="json"),
        )


class SkippedMutation(_StrictPersistedModel):
    task_id: str
    operator_family: OperatorFamily | Literal["*"]
    seed: int | None
    reason: str


class FamilyCount(_StrictPersistedModel):
    operator_family: OperatorFamily
    count: int


class DatasetManifest(_StrictPersistedModel):
    manifest_schema_version: Literal[1] = MANIFEST_SCHEMA_VERSION
    dataset_schema_version: Literal[1] = DATASET_SCHEMA_VERSION
    generator_version: Literal["mutants@v1"] = GENERATOR_VERSION
    config: GenerationConfig
    config_hash: str
    dataset_hash: str
    records_filename: Literal["mutants.jsonl"] = RECORDS_FILENAME
    records_sha256: str
    accepted_count: int
    accepted_by_family: tuple[FamilyCount, ...]
    skipped: tuple[SkippedMutation, ...]


@dataclass(frozen=True, slots=True)
class LoadedDataset:
    records: tuple[MutantRecord, ...]
    manifest: DatasetManifest


class DatasetValidationError(ValueError):
    """Persisted mutant artifacts failed validation."""


def build_record(
    *,
    task_id: str,
    entry_point: str,
    prompt: str,
    canonical_full_source: str,
    mutated_full_source: str,
    operator_family: OperatorFamily,
    seed: int,
    site_node_path: int,
    site_target_index: int,
    site_description: str,
    input_reprs: tuple[str, ...],
    mutant_expected: tuple[ExpectedOutcome, ...],
    canonical_expected: tuple[ExpectedOutcome, ...],
    distinct_input_indices: tuple[int, ...],
    diff_summary: str,
    canonical_test: str,
) -> MutantRecord:
    """Build an in-memory record with the persisted wire identity."""

    provisional = MutantRecord(
        content_hash="",
        task_id=task_id,
        entry_point=entry_point,
        prompt=prompt,
        canonical_full_source=canonical_full_source,
        mutated_full_source=mutated_full_source,
        operator_family=operator_family,
        seed=seed,
        site_node_path=site_node_path,
        site_target_index=site_target_index,
        site_description=site_description,
        input_reprs=input_reprs,
        mutant_expected=mutant_expected,
        canonical_expected=canonical_expected,
        distinct_input_indices=distinct_input_indices,
        diff_summary=diff_summary,
        canonical_test=canonical_test,
    )
    return provisional.model_copy(
        update={
            "content_hash": identity_hash_for(
                schema=_RECORD_SCHEMA,
                payload=provisional.model_dump(
                    mode="json", exclude={"content_hash"}
                ),
            )
        }
    )


def build_manifest(
    *,
    config: GenerationConfig,
    records: tuple[MutantRecord, ...],
    accepted_by_family: tuple[FamilyCount, ...],
    skipped: tuple[SkippedMutation, ...] = (),
) -> DatasetManifest:
    """Build an in-memory manifest for tests and artifact tooling."""

    records_bytes = encode_records(records)
    records_sha256 = _sha256(records_bytes)
    config_hash = config.identity_hash()
    return DatasetManifest(
        config=config,
        config_hash=config_hash,
        dataset_hash=_dataset_hash(
            config_hash=config_hash,
            records_sha256=records_sha256,
            accepted_count=len(records),
            accepted_by_family=accepted_by_family,
            skipped=skipped,
        ),
        records_sha256=records_sha256,
        accepted_count=len(records),
        accepted_by_family=accepted_by_family,
        skipped=skipped,
    )


def encode_records(records: tuple[MutantRecord, ...]) -> bytes:
    lines = (
        json.dumps(
            record.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
        for record in records
    )
    return "".join(f"{line}\n" for line in lines).encode("utf-8")


def load_dataset(
    output_dir: Path,
    *,
    expected_config_hash: str | None = None,
) -> LoadedDataset:
    """Load one retained ED1M dataset and verify its internal artifacts."""

    if not output_dir.is_dir():
        raise DatasetValidationError(
            f"mutant dataset directory does not exist: {output_dir}"
        )
    names = {path.name for path in output_dir.iterdir()}
    expected_names = {RECORDS_FILENAME, MANIFEST_FILENAME}
    if names != expected_names:
        raise DatasetValidationError(
            "mutant dataset directory must contain exactly "
            f"{sorted(expected_names)}"
        )
    records_bytes = (output_dir / RECORDS_FILENAME).read_bytes()
    manifest_bytes = (output_dir / MANIFEST_FILENAME).read_bytes()
    try:
        decode_strict_json_bytes(
            manifest_bytes,
            max_bytes=len(manifest_bytes),
            max_depth=len(manifest_bytes),
        )
        manifest = DatasetManifest.model_validate_json(manifest_bytes)
    except (StrictJsonDecodeError, ValidationError) as exc:
        raise DatasetValidationError("invalid mutant manifest") from exc
    records = _decode_records(records_bytes)
    _validate_components(
        records=records,
        manifest=manifest,
        records_bytes=records_bytes,
        expected_config_hash=expected_config_hash,
    )
    return LoadedDataset(records=records, manifest=manifest)


def record_order_key(
    record: MutantRecord,
) -> tuple[tuple[str, int, str], str, int, int, int]:
    return (
        _task_order_key(record.task_id),
        record.operator_family.value,
        record.seed,
        record.site_node_path,
        record.site_target_index,
    )


def _skip_order_key(
    skip: SkippedMutation,
    config: GenerationConfig,
) -> tuple[tuple[str, int, str], int, int]:
    family_rank = (
        -1
        if skip.operator_family == "*"
        else config.operator_families.index(skip.operator_family)
    )
    seed_rank = -1 if skip.seed is None else skip.seed
    return _task_order_key(skip.task_id), family_rank, seed_rank


def _validate_components(
    *,
    records: tuple[MutantRecord, ...],
    manifest: DatasetManifest,
    records_bytes: bytes,
    expected_config_hash: str | None,
) -> None:
    _validate_config(manifest.config)
    config_hash = manifest.config.identity_hash()
    if manifest.config_hash != config_hash:
        raise DatasetValidationError("manifest config identity mismatch")
    if (
        expected_config_hash is not None
        and config_hash != expected_config_hash
    ):
        raise DatasetValidationError("unexpected generation config identity")
    records_sha256 = _sha256(records_bytes)
    if manifest.records_sha256 != records_sha256:
        raise DatasetValidationError("mutants.jsonl SHA-256 mismatch")
    if manifest.accepted_count != len(records):
        raise DatasetValidationError("manifest accepted count mismatch")
    if tuple(sorted(records, key=record_order_key)) != records:
        raise DatasetValidationError("mutant records are not in stable order")
    if (
        tuple(
            sorted(
                manifest.skipped,
                key=lambda skip: _skip_order_key(skip, manifest.config),
            )
        )
        != manifest.skipped
    ):
        raise DatasetValidationError("mutant skips are not in stable order")

    _validate_coordinate_partition(
        records=records,
        skipped=manifest.skipped,
        config=manifest.config,
    )
    identities: set[str] = set()
    programs: set[tuple[str, str]] = set()
    canonical_records: dict[str, tuple[object, ...]] = {}
    for record in records:
        expected_hash = identity_hash_for(
            schema=_RECORD_SCHEMA,
            payload=record.model_dump(mode="json", exclude={"content_hash"}),
        )
        if record.content_hash != expected_hash:
            raise DatasetValidationError(
                f"record content identity mismatch: {record.task_id}"
            )
        if record.content_hash in identities:
            raise DatasetValidationError("duplicate record content identity")
        identities.add(record.content_hash)
        _validate_record(record, manifest.config)
        canonical_shape = (
            record.entry_point,
            record.prompt,
            record.canonical_full_source,
            record.canonical_test,
            record.input_reprs,
            record.canonical_expected,
        )
        prior = canonical_records.setdefault(record.task_id, canonical_shape)
        if prior != canonical_shape:
            raise DatasetValidationError(
                "records disagree on canonical task content or outcomes"
            )
        program_key = (record.task_id, record.mutated_full_source)
        if program_key in programs:
            raise DatasetValidationError("duplicate mutated program")
        programs.add(program_key)

    actual_counts = tuple(
        FamilyCount(
            operator_family=family,
            count=sum(record.operator_family is family for record in records),
        )
        for family in sorted(
            manifest.config.operator_families, key=lambda item: item.value
        )
    )
    if manifest.accepted_by_family != actual_counts:
        raise DatasetValidationError("manifest family counts mismatch")
    expected_dataset_hash = _dataset_hash(
        config_hash=config_hash,
        records_sha256=records_sha256,
        accepted_count=len(records),
        accepted_by_family=manifest.accepted_by_family,
        skipped=manifest.skipped,
    )
    if manifest.dataset_hash != expected_dataset_hash:
        raise DatasetValidationError("manifest dataset identity mismatch")


def _validate_record(record: MutantRecord, config: GenerationConfig) -> None:
    if record.operator_family not in config.operator_families:
        raise DatasetValidationError(
            "record operator family is absent from generation config"
        )
    if not 0 <= record.seed < config.seeds:
        raise DatasetValidationError("record seed is outside config")
    if record.site_node_path < 0 or record.site_target_index < 0:
        raise DatasetValidationError("record site address is negative")
    if record.task_id not in config.task_ids:
        raise DatasetValidationError("record task is outside config")
    count = len(record.input_reprs)
    if count > config.max_inputs_per_mutant:
        raise DatasetValidationError("record input count exceeds config")
    if (
        len(record.mutant_expected) != count
        or len(record.canonical_expected) != count
    ):
        raise DatasetValidationError("record outcome count mismatch")
    actual_distinct = tuple(
        index
        for index, (canonical, mutant) in enumerate(
            zip(
                record.canonical_expected,
                record.mutant_expected,
                strict=True,
            )
        )
        if canonical != mutant
    )
    if not actual_distinct:
        raise DatasetValidationError("record has no canonical divergence")
    if record.distinct_input_indices != actual_distinct:
        raise DatasetValidationError(
            "record distinct input indices are invalid"
        )
    for input_repr in record.input_reprs:
        try:
            input_value = ast.literal_eval(input_repr)
        except (SyntaxError, ValueError) as exc:
            raise DatasetValidationError(
                "record input is not a Python literal"
            ) from exc
        if not isinstance(input_value, tuple):
            raise DatasetValidationError(
                "record input is not an argument tuple"
            )
    try:
        matches = tuple(
            site
            for site in iter_sites(
                record.canonical_full_source,
                record.operator_family,
            )
            if site.node_path == record.site_node_path
            and site.target_index == record.site_target_index
        )
    except MutationError as exc:
        raise DatasetValidationError(
            "record canonical source is malformed"
        ) from exc
    if len(matches) != 1:
        raise DatasetValidationError("record mutation site is not applicable")
    site = matches[0]
    if site.description != record.site_description:
        raise DatasetValidationError(
            "record mutation site description mismatch"
        )
    try:
        expected_mutant = apply_site(record.canonical_full_source, site)
    except MutationError as exc:
        raise DatasetValidationError(
            "record mutation could not be reproduced"
        ) from exc
    if expected_mutant != record.mutated_full_source:
        raise DatasetValidationError(
            "record mutant does not match its mutation site"
        )
    if not _defines_entry_point(
        record.canonical_full_source, record.entry_point
    ):
        raise DatasetValidationError(
            "record canonical source does not define its entry point"
        )


def _validate_config(config: GenerationConfig) -> None:
    if config.dataset_id != HF_DATASET_ID:
        raise DatasetValidationError("generation config dataset id mismatch")
    if config.dataset_revision != HF_REVISION:
        raise DatasetValidationError(
            "generation config dataset revision mismatch"
        )
    if config.seeds < 1:
        raise DatasetValidationError("generation config seeds are invalid")
    if config.max_inputs_per_mutant < 1:
        raise DatasetValidationError(
            "generation config input limit is invalid"
        )
    if config.timeout_seconds <= 0 or not math.isfinite(
        config.timeout_seconds
    ):
        raise DatasetValidationError("generation config timeout is invalid")
    if not config.canonical_suite_digest:
        raise DatasetValidationError(
            "generation config canonical suite provenance is empty"
        )
    if not config.runner_label or not config.runtime_label:
        raise DatasetValidationError(
            "generation runner provenance is incomplete"
        )
    if len(set(config.task_ids)) != len(config.task_ids):
        raise DatasetValidationError(
            "generation config task ids contain duplicates"
        )
    if not config.task_ids:
        raise DatasetValidationError("generation config has no task ids")
    expected_families = tuple(
        family for family in ALL_FAMILIES if family in config.operator_families
    )
    if config.operator_families != expected_families:
        raise DatasetValidationError(
            "generation config operator family order is invalid"
        )
    if tuple(sorted(config.task_ids, key=_task_order_key)) != config.task_ids:
        raise DatasetValidationError(
            "generation config task id order is invalid"
        )


def _validate_coordinate_partition(
    *,
    records: tuple[MutantRecord, ...],
    skipped: tuple[SkippedMutation, ...],
    config: GenerationConfig,
) -> None:
    accepted = {
        (record.task_id, record.operator_family, record.seed)
        for record in records
    }
    if len(accepted) != len(records):
        raise DatasetValidationError("duplicate accepted search coordinate")
    ordinary_skips = {
        (skip.task_id, skip.operator_family, skip.seed)
        for skip in skipped
        if skip.operator_family != "*"
    }
    wildcards = [
        skip.task_id for skip in skipped if skip.operator_family == "*"
    ]
    if len(wildcards) != len(set(wildcards)):
        raise DatasetValidationError("duplicate task-wide skip")
    if accepted & ordinary_skips:
        raise DatasetValidationError(
            "accepted and skipped coordinates overlap"
        )
    if len(ordinary_skips) != sum(
        skip.operator_family != "*" for skip in skipped
    ):
        raise DatasetValidationError("duplicate skipped search coordinate")
    for task_id in config.task_ids:
        expected = {
            (task_id, family, seed)
            for family in config.operator_families
            for seed in range(config.seeds)
        }
        concrete = {
            coordinate
            for coordinate in accepted | ordinary_skips
            if coordinate[0] == task_id
        }
        if task_id in wildcards:
            if concrete:
                raise DatasetValidationError(
                    "task-wide skip overlaps concrete coordinates"
                )
        elif concrete != expected:
            raise DatasetValidationError(
                f"incomplete coordinate partition for {task_id}"
            )
    for skip in skipped:
        if skip.task_id not in config.task_ids:
            raise DatasetValidationError("skipped task is outside config")
        if skip.operator_family == "*":
            if skip.seed is not None:
                raise DatasetValidationError(
                    "task-wide skip must not have a seed"
                )
            continue
        if skip.operator_family not in config.operator_families:
            raise DatasetValidationError(
                "skipped operator family is outside config"
            )
        if skip.seed is None or not 0 <= skip.seed < config.seeds:
            raise DatasetValidationError("skipped seed is outside config")


def _decode_records(content: bytes) -> tuple[MutantRecord, ...]:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DatasetValidationError("mutants.jsonl is not UTF-8") from exc
    if text and not text.endswith("\n"):
        raise DatasetValidationError("mutants.jsonl must end with a newline")
    records: list[MutantRecord] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise DatasetValidationError(
                f"mutants.jsonl line {line_number} is blank"
            )
        raw = line.encode("utf-8")
        try:
            decode_strict_json_bytes(
                raw,
                max_bytes=len(raw),
                max_depth=len(raw),
            )
            records.append(MutantRecord.model_validate_json(raw))
        except (StrictJsonDecodeError, ValidationError) as exc:
            raise DatasetValidationError(
                f"invalid mutants.jsonl line {line_number}"
            ) from exc
    return tuple(records)


def _dataset_hash(
    *,
    config_hash: str,
    records_sha256: str,
    accepted_count: int,
    accepted_by_family: tuple[FamilyCount, ...],
    skipped: tuple[SkippedMutation, ...],
) -> str:
    return identity_hash_for(
        schema=_DATASET_SCHEMA,
        payload={
            "accepted_count": accepted_count,
            "accepted_by_family": [
                item.model_dump(mode="json") for item in accepted_by_family
            ],
            "config_hash": config_hash,
            "records_sha256": records_sha256,
            "skipped": [item.model_dump(mode="json") for item in skipped],
        },
    )


def _defines_entry_point(source: str, entry_point: str) -> bool:
    tree = ast.parse(source)
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == entry_point
        for node in tree.body
    )


def _task_order_key(task_id: str) -> tuple[str, int, str]:
    prefix, separator, suffix = task_id.rpartition("/")
    try:
        task_number = int(suffix)
    except ValueError:
        task_number = -1
    return prefix if separator else task_id, task_number, task_id


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


__all__ = (
    "DatasetManifest",
    "DatasetValidationError",
    "ExpectedOutcome",
    "FamilyCount",
    "GenerationConfig",
    "LoadedDataset",
    "MutantRecord",
    "OperatorFamily",
    "SkippedMutation",
    "build_manifest",
    "build_record",
    "encode_records",
    "load_dataset",
)
