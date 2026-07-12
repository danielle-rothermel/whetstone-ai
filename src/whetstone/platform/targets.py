"""Versioned, top-level Whetstone execution targets."""

from __future__ import annotations

from typing import Any, cast

from dbos import DBOS
from dr_platform import (
    ExecutionIdentity,
    ExecutionRecipeEnvelope,
    ExecutionTarget,
    FailureSnapshot,
    ItemInsertStatus,
    PlatformSchema,
    TargetRegistry,
    WorkflowTopology,
)
from dr_platform import (
    FailureClass as KernelFailureClass,
)
from dr_platform.items import SubmittableItem
from dr_platform.submission import (
    RegistrationItem,
    RegistrationItemResult,
    RegistrationPageContext,
    RegistrationResult,
)
from dr_platform.targets import TargetContractDeclaration
from dr_serialize import sha256_json_digest
from pydantic import BaseModel, ConfigDict, StrictStr
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from whetstone.db import io as db_io
from whetstone.db import schema
from whetstone.platform.acceptance import (
    GenerationMembershipConflictError,
    ManifestRelationshipResult,
    accept_operation_manifest,
)
from whetstone.platform.graph_workflow import run_prediction_graph_workflow
from whetstone.platform.scoring_workflow import run_score_submission_workflow
from whetstone.records import (
    DatasetSnapshotIdentityPayload,
    ExperimentRecord,
    PredictionSpecRecord,
)

GENERATION_QUEUE_NAME = "whetstone-generation"
SCORING_QUEUE_NAME = "whetstone-scoring"
GENERATION_TARGET_KEY = "whetstone-generation"
GENERATION_TARGET_VERSION = 1
GENERATION_WORKFLOW_VERSION = 1
GENERATION_ARGUMENT_RECIPE_VERSION = 1
SCORING_TARGET_KEY = "whetstone-scoring"
SCORING_TARGET_VERSION = 1
SCORING_WORKFLOW_VERSION = 1
SCORING_ARGUMENT_RECIPE_VERSION = 1
CLASSIFIER_VERSION = 1


class ScoringTargetSpec(BaseModel):
    """A concrete, immutable scoring Item recipe input."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    prediction_id: StrictStr
    generation_run_id: StrictStr
    scoring_profile_id: StrictStr
    scoring_profile_version: StrictStr
    parser_profile_id: StrictStr
    parser_version: StrictStr
    dataset_name: StrictStr
    dataset_split: StrictStr
    dataset_snapshot: DatasetSnapshotIdentityPayload

    @property
    def item_key(self) -> str:
        return sha256_json_digest(
            {
                "generation_run_id": self.generation_run_id,
                "scoring_profile_id": self.scoring_profile_id,
                "scoring_profile_version": self.scoring_profile_version,
                "parser_profile_id": self.parser_profile_id,
                "parser_version": self.parser_version,
                "dataset_name": self.dataset_name,
                "dataset_split": self.dataset_split,
            }
        )

    @property
    def spec(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @property
    def service_class(self):
        from dr_platform import ServiceClass

        return ServiceClass.STANDARD


def enqueue_failure_from_whetstone_exception(
    error: BaseException,
) -> FailureSnapshot:
    """Translate provider taxonomy into the kernel-owned failure enum."""
    from whetstone.eval_failures import summarize_exception

    summary = summarize_exception(error)
    return FailureSnapshot(
        failure_class=KernelFailureClass(summary.failure_class.value),
        error_type=summary.failure_exception_type,
        message=summary.message,
        metadata=summary.failure_metadata,
    )


def generation_target() -> ExecutionTarget:
    declaration = TargetContractDeclaration(
        queue_name=GENERATION_QUEUE_NAME,
        workflow_role="generation",
        managed_workflow_name="whetstone_generation",
        managed_workflow_version=GENERATION_WORKFLOW_VERSION,
        topology=WorkflowTopology.TOP_LEVEL_ONLY,
        argument_recipe_version=GENERATION_ARGUMENT_RECIPE_VERSION,
        classifier_version=CLASSIFIER_VERSION,
        registration_hook_name="whetstone_prediction_spec_registration",
        registration_hook_version=1,
    )
    return ExecutionTarget(
        ref=declaration.target_ref(
            target_key=GENERATION_TARGET_KEY,
            target_version=GENERATION_TARGET_VERSION,
        ),
        queue_name=GENERATION_QUEUE_NAME,
        workflow_role="generation",
        managed_workflow_name="whetstone_generation",
        managed_workflow_version=GENERATION_WORKFLOW_VERSION,
        topology=WorkflowTopology.TOP_LEVEL_ONLY,
        argument_recipe_version=GENERATION_ARGUMENT_RECIPE_VERSION,
        classifier_version=CLASSIFIER_VERSION,
        registration_hook_name="whetstone_prediction_spec_registration",
        registration_hook_version=1,
        workflow=run_prediction_graph_workflow,
        execution_for=_generation_execution_identity,
        args_for=_generation_arguments,
        recipe_for=_generation_recipe,
        classify_error=enqueue_failure_from_whetstone_exception,
        registration_hook=_register_prediction_specs,
    )


def scoring_target() -> ExecutionTarget:
    """The final managed scoring target.

    Its durable arguments are deliberately limited to a generation identity,
    frozen profile/dataset axes, and the platform ordinal.  Database and
    provider configuration are resolved inside DBOS steps.
    """
    declaration = TargetContractDeclaration(
        queue_name=SCORING_QUEUE_NAME,
        workflow_role="scoring",
        managed_workflow_name="whetstone_scoring",
        managed_workflow_version=SCORING_WORKFLOW_VERSION,
        topology=WorkflowTopology.TOP_LEVEL_ONLY,
        argument_recipe_version=SCORING_ARGUMENT_RECIPE_VERSION,
        classifier_version=CLASSIFIER_VERSION,
        registration_hook_name="whetstone_score_target_registration",
        registration_hook_version=1,
    )
    return ExecutionTarget(
        ref=declaration.target_ref(
            target_key=SCORING_TARGET_KEY,
            target_version=SCORING_TARGET_VERSION,
        ),
        queue_name=SCORING_QUEUE_NAME,
        workflow_role="scoring",
        managed_workflow_name="whetstone_scoring",
        managed_workflow_version=SCORING_WORKFLOW_VERSION,
        topology=WorkflowTopology.TOP_LEVEL_ONLY,
        argument_recipe_version=SCORING_ARGUMENT_RECIPE_VERSION,
        classifier_version=CLASSIFIER_VERSION,
        registration_hook_name="whetstone_score_target_registration",
        registration_hook_version=1,
        workflow=run_score_submission_workflow,
        execution_for=_scoring_execution_identity,
        args_for=_scoring_arguments,
        recipe_for=_scoring_recipe,
        classify_error=enqueue_failure_from_whetstone_exception,
        registration_hook=_register_score_targets,
    )


def target_registry() -> TargetRegistry:
    registry = TargetRegistry()
    registry.register_all((generation_target(), scoring_target()))
    return registry


def register_execution_queues(*, worker_concurrency: int) -> None:
    for queue_name in (GENERATION_QUEUE_NAME, SCORING_QUEUE_NAME):
        DBOS.register_queue(
            queue_name,
            worker_concurrency=worker_concurrency,
            priority_enabled=True,
            on_conflict="always_update",
        )


def listen_to_execution_queues() -> None:
    DBOS.listen_queues([GENERATION_QUEUE_NAME, SCORING_QUEUE_NAME])


def _generation_recipe(item: SubmittableItem) -> ExecutionRecipeEnvelope:
    spec = PredictionSpecRecord.model_validate(item.spec)
    target = generation_target()
    return ExecutionRecipeEnvelope(
        target_ref=target.ref,
        managed_workflow_name=target.managed_workflow_name,
        managed_workflow_version=target.managed_workflow_version,
        topology=WorkflowTopology.TOP_LEVEL_ONLY,
        argument_recipe_version=GENERATION_ARGUMENT_RECIPE_VERSION,
        payload={
            "prediction_spec": spec.model_dump(mode="json"),
            "application_version": "whetstone-v6",
        },
    )


def _generation_execution_identity(
    item: Any, attempt: int
) -> ExecutionIdentity:
    recipe_digest = _generation_recipe(cast("SubmittableItem", item)).digest()
    key = sha256_json_digest({"recipe": recipe_digest, "attempt": attempt})
    return ExecutionIdentity(
        execution_key=key, workflow_id=f"whetstone-generation:{key}"
    )


def _generation_arguments(
    item: Any, attempt: int
) -> tuple[str, int, str, str]:
    recipe_digest = _generation_recipe(cast("SubmittableItem", item)).digest()
    return (
        str(item.spec["prediction_id"]),
        attempt,
        recipe_digest,
        item.item_id,
    )


def _scoring_recipe(item: SubmittableItem) -> ExecutionRecipeEnvelope:
    target = scoring_target()
    spec = ScoringTargetSpec.model_validate(item.spec).model_dump(mode="json")
    return ExecutionRecipeEnvelope(
        target_ref=target.ref,
        managed_workflow_name=target.managed_workflow_name,
        managed_workflow_version=target.managed_workflow_version,
        topology=WorkflowTopology.TOP_LEVEL_ONLY,
        argument_recipe_version=SCORING_ARGUMENT_RECIPE_VERSION,
        payload={
            "generation_run_id": spec["generation_run_id"],
            "scoring_profile_id": spec["scoring_profile_id"],
            "scoring_profile_version": spec["scoring_profile_version"],
            "parser_profile_id": spec["parser_profile_id"],
            "parser_version": spec["parser_version"],
            "dataset_name": spec["dataset_name"],
            "dataset_split": spec["dataset_split"],
            "dataset_snapshot": spec["dataset_snapshot"],
            "application_version": "whetstone-v6",
        },
    )


def _scoring_execution_identity(item: Any, attempt: int) -> ExecutionIdentity:
    recipe_digest = _scoring_recipe(cast("SubmittableItem", item)).digest()
    key = sha256_json_digest({"recipe": recipe_digest, "attempt": attempt})
    return ExecutionIdentity(
        execution_key=key, workflow_id=f"whetstone-scoring:{key}"
    )


def _scoring_arguments(item: Any, attempt: int) -> tuple[Any, ...]:
    spec = ScoringTargetSpec.model_validate(item.spec)
    return (
        spec.generation_run_id,
        attempt,
        spec.scoring_profile_id,
        spec.scoring_profile_version,
        spec.parser_profile_id,
        spec.parser_version,
        spec.dataset_name,
        spec.dataset_split,
        spec.dataset_snapshot.model_dump(mode="json"),
        _scoring_recipe(cast("SubmittableItem", item)).digest(),
        item.item_id,
    )


def _register_prediction_specs(
    connection: Any,
    *,
    operation_key: str,
    items: tuple[RegistrationItem, ...],
    page: RegistrationPageContext,
) -> RegistrationResult:
    platform = PlatformSchema(prefix="whetstone")
    operation = (
        connection.execute(
            select(platform.operations).where(
                platform.operations.c.operation_key == operation_key
            )
        )
        .mappings()
        .one()
    )
    experiment_name = str(operation["group_key"])
    existing_relationship = (
        connection.execute(
            select(schema.experiment_operation_manifests).where(
                schema.experiment_operation_manifests.c.experiment_name
                == experiment_name,
                schema.experiment_operation_manifests.c.workflow_role
                == "generation",
            )
        )
        .mappings()
        .first()
    )
    if existing_relationship is not None and (
        existing_relationship["operation_key"] != operation_key
        or existing_relationship["manifest_digest"] != page.manifest_digest
        or dict(existing_relationship["target_ref"])
        != _target_ref_from_operation(operation)
    ):
        raise GenerationMembershipConflictError(
            "generation membership is already fixed by a different Manifest"
        )
    if page.is_final_page:
        connection.execute(
            insert(schema.experiments)
            .values(
                db_io.experiment_row(
                    ExperimentRecord(experiment_name=experiment_name)
                )
            )
            .on_conflict_do_nothing(index_elements=["experiment_name"])
        )
        relationship_result = accept_operation_manifest(
            connection,
            experiment_name=experiment_name,
            workflow_role="generation",
            operation_key=operation_key,
            manifest_digest=page.manifest_digest,
            target_ref=_target_ref_from_operation(operation),
        )
        if (
            relationship_result
            is ManifestRelationshipResult.GENERATION_MEMBERSHIP_CONFLICT
        ):
            raise GenerationMembershipConflictError(
                "generation membership is already fixed by a different "
                "Manifest"
            )
    results: list[RegistrationItemResult] = []
    for item in items:
        spec = PredictionSpecRecord.model_validate(item.spec)
        if spec.experiment_name != experiment_name:
            raise ValueError(
                "prediction spec Experiment does not match the Operation group"
            )
        existing = (
            connection.execute(
                select(schema.prediction_specs).where(
                    schema.prediction_specs.c.prediction_id
                    == spec.prediction_id
                )
            )
            .mappings()
            .first()
        )
        if existing is None:
            connection.execute(
                insert(schema.experiments)
                .values(
                    db_io.experiment_row(
                        ExperimentRecord(experiment_name=spec.experiment_name)
                    )
                )
                .on_conflict_do_nothing(index_elements=["experiment_name"])
            )
            connection.execute(
                insert(schema.prediction_specs).values(
                    db_io.prediction_spec_row(spec)
                )
            )
            status = ItemInsertStatus.INSERTED
        else:
            persisted = db_io.prediction_spec_record_from_row(dict(existing))
            if persisted != spec:
                raise ValueError(
                    "prediction spec conflicts with the canonical persisted "
                    f"spec: {spec.prediction_id!r}"
                )
            status = ItemInsertStatus.ALREADY_PRESENT
        results.append(
            RegistrationItemResult(
                item_key=item.item_key, insert_status=status
            )
        )
    return RegistrationResult(items=tuple(results))


def _register_score_targets(
    connection: Any,
    *,
    operation_key: str,
    items: tuple[RegistrationItem, ...],
    page: RegistrationPageContext,
) -> RegistrationResult:
    """Validate the frozen target rows without recreating legacy scheduling.

    Score outcomes are append-only and are intentionally *not* inserted at
    registration time.  A retry receives a new platform ordinal and persists
    either one immutable ScoreAttempt or one harness-failure record.
    """
    platform = PlatformSchema(prefix="whetstone")
    operation = (
        connection.execute(
            select(platform.operations).where(
                platform.operations.c.operation_key == operation_key
            )
        )
        .mappings()
        .one()
    )
    experiment_name = str(operation["group_key"])
    results: list[RegistrationItemResult] = []
    for item in items:
        target = ScoringTargetSpec.model_validate(item.spec)
        canonical_spec = target.model_dump(mode="json")
        if item.spec != canonical_spec:
            raise ValueError(
                "scoring target is not in exact canonical form: "
                f"{item.item_key!r}"
            )
        if item.item_key != target.item_key:
            raise ValueError(
                "scoring target item key does not match its recipe"
            )
        canonical_recipe = _scoring_recipe(cast("SubmittableItem", item))
        if (
            item.execution_recipe != canonical_recipe
            or item.execution_recipe_digest != canonical_recipe.digest()
        ):
            raise ValueError(
                "scoring target execution recipe does not match its canonical "
                "profile/parser/snapshot inputs"
            )
        _validate_registered_scoring_target(
            connection,
            target=target,
            experiment_name=experiment_name,
        )
        results.append(
            RegistrationItemResult(
                item_key=item.item_key,
                insert_status=ItemInsertStatus.INSERTED,
            )
        )
    if page.is_final_page:
        spec = dict(operation["spec"])
        accept_operation_manifest(
            connection,
            experiment_name=str(spec["experiment_name"]),
            workflow_role="scoring",
            operation_key=operation_key,
            manifest_digest=page.manifest_digest,
            selection_digest=str(spec["selection_digest"]),
            target_ref=_target_ref_from_operation(operation),
        )
    return RegistrationResult(items=tuple(results))


def _validate_registered_scoring_target(
    connection: Any,
    *,
    target: ScoringTargetSpec,
    experiment_name: str,
) -> None:
    generation_run = (
        connection.execute(
            select(schema.generation_runs).where(
                schema.generation_runs.c.generation_run_id
                == target.generation_run_id
            )
        )
        .mappings()
        .one()
    )
    if generation_run["prediction_id"] != target.prediction_id:
        raise ValueError(
            "scoring target Generation Run does not belong to its Prediction"
        )
    prediction = (
        connection.execute(
            select(schema.prediction_specs).where(
                schema.prediction_specs.c.prediction_id == target.prediction_id
            )
        )
        .mappings()
        .one()
    )
    canonical_prediction = db_io.prediction_spec_record_from_row(
        dict(prediction)
    )
    if canonical_prediction.experiment_name != experiment_name:
        raise ValueError(
            "scoring target Prediction does not belong to the Operation group"
        )
    snapshot = canonical_prediction.task.metadata.get("dataset_snapshot")
    if snapshot != target.dataset_snapshot.model_dump(mode="json"):
        raise ValueError(
            "scoring target dataset snapshot does not match the canonical "
            "Prediction snapshot"
        )
    if target.dataset_snapshot.header.dataset_id != target.dataset_name:
        raise ValueError(
            "scoring target dataset name does not match its snapshot identity"
        )
    from dr_code.humaneval import resolve_humaneval_scoring_profile

    profile = resolve_humaneval_scoring_profile(
        scoring_profile_id=target.scoring_profile_id,
        scoring_profile_version=target.scoring_profile_version,
    )
    if (
        profile.profile_id != target.scoring_profile_id
        or profile.version != target.scoring_profile_version
        or profile.parser_profile.profile_id != target.parser_profile_id
        or profile.parser_profile.version != target.parser_version
    ):
        raise ValueError(
            "scoring target profile/parser axes do not match the canonical "
            "scoring profile"
        )


def _target_ref_from_operation(operation: Any) -> dict[str, Any]:
    return {
        "target_key": operation["target_key"],
        "target_version": operation["target_version"],
        "target_contract_digest": operation["target_contract_digest"],
    }
