"""Production Typer coordinator for the frozen v6 COPRO lifecycle."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import duckdb
import typer
from dr_code.humaneval import resolve_humaneval_scoring_profile
from dr_platform import (
    ExportReconciliationDependencies,
    OperationWaitOptions,
    PinnedBundleGoneError,
    PlatformSchema,
    pin_local_bundle,
    wait_operation,
)
from sqlalchemy import Engine, create_engine

from whetstone.optimization.copro import (
    CoproCandidate,
    CoproCandidateResult,
    CoproDimensions,
    CoproLifecycle,
    CoproPin,
    CoproPinLossError,
    CoproRunConfig,
    CoproRunResult,
    run_copro_loop,
    summarize_pinned_candidates,
    write_copro_artifacts,
)
from whetstone.platform.acceptance import (
    AcceptanceDisposition,
    RequiredScoringProfile,
    evaluate_strict_acceptance,
)
from whetstone.platform.cli_env import load_env_file, run_typer_app
from whetstone.platform.dataset_snapshot import load_humaneval_snapshot
from whetstone.platform.integrity import (
    BundleIntegrityConfiguration,
    required_bundle_integrity_configuration,
)
from whetstone.platform.publication import (
    build_export_reconciliation_dependencies,
)
from whetstone.platform.runtime import resolve_application_database_url
from whetstone.platform.spec_builder import (
    DEFAULT_CONFIGS_ROOT,
    ExperimentSpecConfig,
    GraphLayout,
    HumanevalEncDecConfig,
    iter_experiment_specs,
    load_model_config_fragment,
    load_split_config_fragment,
    resolve_config_path,
)
from whetstone.platform.submission import (
    scoring_target_for_generation_run,
    select_populated_scoring_generation_runs,
    submit_prediction_specs,
    submit_scoring_targets,
)
from whetstone.platform.targets import target_registry
from whetstone.publication import (
    ANALYSIS_BUNDLE_KEY,
    AnalysisBundleReader,
    export_whetstone,
)
from whetstone.records import PredictionSpecRecord

APP = typer.Typer(no_args_is_help=True)
PLATFORM_SCHEMA = PlatformSchema(prefix="whetstone")


@dataclass(frozen=True)
class CoproSpecConfiguration:
    model_config_path: Path
    split_path: Path
    configs_root: Path
    compression_targets: tuple[float, ...]
    repetition_seeds: tuple[int, ...]
    min_encoder_char_budget: int = 50


def _resolve_config_path(configs_root: Path, path: Path) -> Path:
    if path.is_absolute():
        return path
    reference = path.as_posix().removeprefix("configs/")
    return resolve_config_path(configs_root, reference)


def build_candidate_specs(
    configuration: CoproSpecConfiguration,
    *,
    run_id: str,
    experiment_name: str,
    candidates: tuple[CoproCandidate, ...],
) -> tuple[PredictionSpecRecord, ...]:
    """Load the dataset once and stamp one shared identity per candidate."""
    model = load_model_config_fragment(
        _resolve_config_path(
            configuration.configs_root, configuration.model_config_path
        )
    )
    split = load_split_config_fragment(
        _resolve_config_path(
            configuration.configs_root, configuration.split_path
        )
    )
    snapshot = load_humaneval_snapshot(
        dataset_name=split.dataset.name,
        dataset_split=split.dataset.split,
        snapshot_path=split.dataset.snapshot_path,
    )
    specs: list[PredictionSpecRecord] = []
    for candidate in candidates:
        dimensions = tuple(
            CoproDimensions(
                copro_run_id=run_id,
                candidate_id=candidate.candidate_id,
                candidate_depth=candidate.depth,
                parent_candidate_id=candidate.parent_candidate_id,
                instructions_digest=candidate.instructions_digest,
                compression_target=target,
            ).model_dump(mode="json")
            for target in configuration.compression_targets
        )
        config = ExperimentSpecConfig(
            experiment_name=experiment_name,
            graph_layout=GraphLayout.ENCDEC,
            dataset=split.dataset,
            repetition_seeds=configuration.repetition_seeds,
            dimensions_axes=dimensions,
            providers=model.providers,
            encdec_shape="humaneval",
            humaneval_encdec=HumanevalEncDecConfig(
                instructions_start=candidate.instructions_start,
                instructions_end=candidate.instructions_end,
                min_encoder_char_budget=(
                    configuration.min_encoder_char_budget
                ),
            ),
        )
        specs.extend(iter_experiment_specs(config, snapshot=snapshot))
    return tuple(specs)


class ProductionCoproLifecycle(CoproLifecycle):
    """Compose only v6 submission, lifecycle, acceptance, and export APIs."""

    def __init__(
        self,
        *,
        engine: Engine,
        reconciliation: ExportReconciliationDependencies,
        integrity: BundleIntegrityConfiguration,
        analysis_destination: Path,
        detail_destination: Path,
        wait_timeout_seconds: float,
        wait_poll_seconds: float,
        pin_ttl_seconds: int,
    ) -> None:
        self._engine = engine
        self._reconciliation = reconciliation
        self._integrity = integrity
        self._analysis_destination = analysis_destination
        self._detail_destination = detail_destination
        self._wait_options = OperationWaitOptions(
            timeout_seconds=wait_timeout_seconds,
            poll_interval_seconds=wait_poll_seconds,
            clock=lambda: datetime.now(UTC),
            sleeper=time.sleep,
        )
        self._pin_ttl_seconds = pin_ttl_seconds
        self._profile = resolve_humaneval_scoring_profile(
            scoring_profile_id="humaneval",
            scoring_profile_version="v1",
        )

    def submit_generation(
        self,
        *,
        experiment_name: str,
        operation_key: str,
        specs: tuple[PredictionSpecRecord, ...],
    ) -> None:
        submit_prediction_specs(
            self._engine,
            operation_key=operation_key,
            experiment_name=experiment_name,
            specs=specs,
            metadata={"optimizer": "copro_minimal"},
        )

    def wait(self, operation_key: str) -> None:
        wait_operation(
            operation_key,
            engine=self._engine,
            resolver=target_registry(),
            options=self._wait_options,
            schema=PLATFORM_SCHEMA,
        )

    def submit_scoring(
        self,
        *,
        experiment_name: str,
        operation_key: str,
        generation_operation_key: str,
        specs: tuple[PredictionSpecRecord, ...],
    ) -> None:
        with self._engine.begin() as connection:
            generation_runs = select_populated_scoring_generation_runs(
                connection, experiment_name=experiment_name
            )
        if not generation_runs:
            raise RuntimeError("COPRO generation produced no scoreable runs")
        specs_by_prediction = {spec.prediction_id: spec for spec in specs}
        targets = tuple(
            scoring_target_for_generation_run(
                spec=specs_by_prediction[run.prediction_id],
                generation_run=run,
                scoring_profile_id=self._profile.profile_id,
                scoring_profile_version=self._profile.version,
            )
            for run in generation_runs
        )
        submit_scoring_targets(
            self._engine,
            operation_key=operation_key,
            experiment_name=experiment_name,
            targets=targets,
            source_generation_operation_key=generation_operation_key,
            metadata={"optimizer": "copro_minimal"},
        )

    def promote_acceptance(self, experiment_name: str) -> None:
        required = RequiredScoringProfile(
            scoring_profile_id=self._profile.profile_id,
            scoring_profile_version=self._profile.version,
            parser_profile_id=self._profile.parser_profile.profile_id,
            parser_version=self._profile.parser_profile.version,
            dataset_name="evalplus/humanevalplus",
            dataset_split="test",
        )
        with self._engine.begin() as connection:
            result = evaluate_strict_acceptance(
                connection,
                experiment_name=experiment_name,
                required_profiles=(required,),
            )
        if result.disposition is not AcceptanceDisposition.PROMOTED:
            raise RuntimeError(
                "COPRO acceptance was not promoted: "
                f"{result.disposition.value}"
            )

    def export_and_pin(self) -> CoproPin:
        analysis, detail = export_whetstone(
            self._engine,
            reconciliation=self._reconciliation,
            integrity_signer=self._integrity.signer,
            destination_path=self._analysis_destination,
            detail_destination_path=self._detail_destination,
        )
        for result in (analysis, detail):
            if any(
                destination.status not in {"PROMOTED", "IDEMPOTENT"}
                for destination in result.destinations
            ):
                raise RuntimeError("COPRO export did not promote every bundle")
        destination = analysis.destinations[0]
        if destination.bundle_id is None:
            raise RuntimeError("COPRO Analysis export returned no bundle ID")
        try:
            pin = pin_local_bundle(
                self._analysis_destination,
                bundle_key=ANALYSIS_BUNDLE_KEY,
                bundle_id=destination.bundle_id,
                ttl_seconds=self._pin_ttl_seconds,
            )
            reader = AnalysisBundleReader.from_pin(
                self._analysis_destination,
                pin,
                public_key_ring=self._integrity.public_key_ring,
            )
        except PinnedBundleGoneError as error:
            raise CoproPinLossError from error
        if reader.snapshot_seq != analysis.snapshot_seq:
            raise RuntimeError("COPRO pin does not match exported snapshot")
        return CoproPin(
            bundle_id=destination.bundle_id,
            snapshot_seq=reader.snapshot_seq,
            token=reader,
        )

    def read_pinned_candidates(
        self, pin: CoproPin, *, experiment_name: str
    ) -> tuple[CoproCandidateResult, ...]:
        if not isinstance(pin.token, AnalysisBundleReader):
            raise TypeError("COPRO pin token is not an Analysis reader")
        try:
            return summarize_pinned_candidates(
                pin.token, experiment_name=experiment_name
            )
        except (duckdb.Error, OSError, PinnedBundleGoneError) as error:
            raise CoproPinLossError from error


def _run_id() -> str:
    return uuid.uuid4().hex[:12]


@APP.command("run")
def run(
    model_config: Annotated[Path, typer.Option("--model-config")],
    split: Annotated[Path, typer.Option("--split")],
    compression_target: Annotated[
        list[float], typer.Option("--compression-target")
    ],
    output_dir: Annotated[
        Path, typer.Option("--output-dir")
    ] = Path("artifacts/optimization/copro"),
    analysis_destination: Annotated[
        Path | None, typer.Option("--analysis-destination")
    ] = None,
    detail_destination: Annotated[
        Path | None, typer.Option("--detail-destination")
    ] = None,
    run_id: Annotated[str | None, typer.Option("--run-id")] = None,
    breadth: Annotated[int, typer.Option(min=1, max=4)] = 3,
    depth: Annotated[int, typer.Option(min=1)] = 2,
    repeats: Annotated[int, typer.Option(min=1)] = 1,
    configs_root: Annotated[Path, typer.Option("--configs-root")] = (
        DEFAULT_CONFIGS_ROOT
    ),
    database_url: Annotated[
        str | None, typer.Option("--database-url")
    ] = None,
    dbos_system_database_url: Annotated[
        str | None, typer.Option("--dbos-system-database-url")
    ] = None,
    wait_timeout_seconds: Annotated[float, typer.Option(min=0.1)] = 3600.0,
    wait_poll_seconds: Annotated[float, typer.Option(min=0.1)] = 1.0,
    pin_ttl_seconds: Annotated[int, typer.Option(min=1)] = 3600,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Optimize encoder instructions through the frozen v6 lifecycle."""
    if not compression_target:
        raise typer.BadParameter("--compression-target is required")
    resolved_run_id = run_id or _run_id()
    config = CoproRunConfig(
        run_id=resolved_run_id,
        breadth=breadth,
        depth=depth,
        dry_run=dry_run,
    )
    spec_configuration = CoproSpecConfiguration(
        model_config_path=model_config,
        split_path=split,
        configs_root=configs_root,
        compression_targets=tuple(compression_target),
        repetition_seeds=tuple(range(repeats)),
    )
    def factory(
        experiment: str, candidates: tuple[CoproCandidate, ...]
    ) -> tuple[PredictionSpecRecord, ...]:
        return build_candidate_specs(
            spec_configuration,
            run_id=resolved_run_id,
            experiment_name=experiment,
            candidates=candidates,
        )

    def checkpoint(result: CoproRunResult) -> None:
        write_copro_artifacts(result, output_dir=output_dir)

    if dry_run:
        result = run_copro_loop(
            config=config,
            lifecycle=None,
            spec_factory=factory,
            checkpoint=checkpoint,
        )
    else:
        load_env_file()
        application_url = resolve_application_database_url(database_url)
        integrity = required_bundle_integrity_configuration()
        analysis_path = analysis_destination or output_dir / "analysis.duckdb"
        detail_path = detail_destination or output_dir / "detail.duckdb"
        engine = create_engine(application_url)
        try:
            with build_export_reconciliation_dependencies(
                application_database_url=application_url,
                dbos_system_database_url=dbos_system_database_url,
            ) as reconciliation:
                lifecycle = ProductionCoproLifecycle(
                    engine=engine,
                    reconciliation=reconciliation,
                    integrity=integrity,
                    analysis_destination=analysis_path,
                    detail_destination=detail_path,
                    wait_timeout_seconds=wait_timeout_seconds,
                    wait_poll_seconds=wait_poll_seconds,
                    pin_ttl_seconds=pin_ttl_seconds,
                )
                result = run_copro_loop(
                    config=config,
                    lifecycle=lifecycle,
                    spec_factory=factory,
                    checkpoint=checkpoint,
                )
        finally:
            engine.dispose()
    paths = write_copro_artifacts(result, output_dir=output_dir)
    typer.echo(
        json.dumps(
            {
                "run_id": result.run_id,
                "dry_run": result.dry_run,
                "best_candidate_id": (
                    None
                    if result.best_candidate is None
                    else result.best_candidate.candidate_id
                ),
                "artifacts": {key: str(value) for key, value in paths.items()},
            },
            sort_keys=True,
        )
    )


def main() -> None:
    run_typer_app(APP)


if __name__ == "__main__":
    main()
