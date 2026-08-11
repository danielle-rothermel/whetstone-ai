from __future__ import annotations

import hashlib
import io
import math
import pickle
import random
from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    field_serializer,
    model_validator,
)

from whetstone.core.identity import (
    compute_identity_hash,
    require_full_hash,
)
from whetstone.optimization.miprov2.demo import (
    BootstrapAcceptance,
    ComponentDemo,
    ComponentDemoSequence,
    ComponentDemoSet,
    DemoSourceKind,
    LabeledTaskDemo,
    MetricValue,
    ObservedTraceStep,
)
from whetstone.optimization.miprov2.rng import (
    Miprov2DurableBindings,
    Miprov2RngCheckpoint,
)

MIPROV2_BOOTSTRAP_PLAN_SCHEMA = "whetstone.miprov2_bootstrap_plan"
MIPROV2_BOOTSTRAP_ATTEMPT_SCHEMA = "whetstone.miprov2_bootstrap_attempt"
MIPROV2_BOOTSTRAP_SCHEMA_VERSION = 1
MIPROV2_TRACE_SELECTION_PROJECTION_VERSION = (
    "dspy_example_pickle_protocol4_cpython/v1"
)
ZERO_SHOT_BOOTSTRAPPED_DEMOS_IN_PROPOSAL = 3
ZERO_SHOT_LABELED_DEMOS_IN_PROPOSAL = 0


class FewshotSeedKind(StrEnum):
    RESET = "reset"
    LABELS_ONLY = "labels_only"
    BOOTSTRAP = "bootstrap"


class TeacherSource(StrEnum):
    EXPLICIT = "explicit_teacher"
    STUDENT = "student"


class LabeledSelectionPlan(BaseModel):
    """DSPy ``LabeledFewShot`` selections, in predictor iteration order."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    component_ids: tuple[StrictStr, ...]
    per_component_task_indices: tuple[tuple[StrictInt, ...], ...]
    sample: StrictBool

    @model_validator(mode="after")
    def _validate_selection(self) -> LabeledSelectionPlan:
        if len(self.component_ids) != len(self.per_component_task_indices):
            raise ValueError(
                "labeled selections must match the component count"
            )
        if len(self.component_ids) != len(set(self.component_ids)):
            raise ValueError("component_ids must be unique")
        if any(not component for component in self.component_ids):
            raise ValueError("component_ids must be non-empty")
        if any(
            index < 0
            for indices in self.per_component_task_indices
            for index in indices
        ):
            raise ValueError("labeled task indices cannot be negative")
        return self


def plan_labeled_selection(
    *,
    component_ids: tuple[str, ...],
    trainset_size: int,
    k: int,
    sample: bool,
) -> LabeledSelectionPlan:
    """Reproduce ``LabeledFewShot.compile``'s local ``Random(0)`` stream."""

    if trainset_size < 0:
        raise ValueError("trainset_size cannot be negative")
    rng = random.Random(0)
    count = min(k, trainset_size)
    selections: list[tuple[int, ...]] = []
    population = list(range(trainset_size))
    for _component_id in component_ids:
        if sample:
            selections.append(tuple(rng.sample(population, count)))
        else:
            selections.append(tuple(population[:count]))
    return LabeledSelectionPlan(
        component_ids=component_ids,
        per_component_task_indices=tuple(selections),
        sample=sample,
    )


class TeacherPreparationPlan(BaseModel):
    """Pure representation of DSPy's teacher copy/reset behavior."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: TeacherSource
    initial_copy: StrictStr = "deepcopy"
    reset_before_labeled_compile: StrictBool
    labeled_selection: LabeledSelectionPlan | None

    @model_validator(mode="after")
    def _validate_teacher(self) -> TeacherPreparationPlan:
        if self.initial_copy != "deepcopy":
            raise ValueError("the frozen MIPROv2 teacher uses deepcopy")
        if self.reset_before_labeled_compile != (
            self.labeled_selection is not None
        ):
            raise ValueError(
                "teacher reset and labeled compilation must occur together"
            )
        return self


class FewshotCandidatePlan(BaseModel):
    """One candidate from the exact ``range(-3, N - 3)`` sequence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_ordinal: StrictInt
    candidate_seed: StrictInt
    bindings: Miprov2DurableBindings
    kind: FewshotSeedKind
    component_ids: tuple[StrictStr, ...]
    trainset_task_hashes: tuple[StrictStr, ...]
    max_bootstrapped_demos: StrictInt
    max_labeled_demos: StrictInt
    max_rounds: StrictInt
    max_errors: StrictInt
    metric_threshold: float | None
    teacher: TeacherPreparationPlan | None
    labels_only_selection: LabeledSelectionPlan | None
    trace_selection_projection_version: StrictStr = (
        MIPROV2_TRACE_SELECTION_PROJECTION_VERSION
    )

    @model_validator(mode="after")
    def _validate_plan(self) -> FewshotCandidatePlan:
        if self.candidate_ordinal < 0:
            raise ValueError("candidate_ordinal cannot be negative")
        for index, task_hash in enumerate(self.trainset_task_hashes):
            require_full_hash(
                task_hash,
                field=f"trainset_task_hashes[{index}]",
            )
        if not self.component_ids:
            raise ValueError("at least one component is required")
        if len(self.component_ids) != len(set(self.component_ids)):
            raise ValueError("component_ids must be unique")
        if any(not component for component in self.component_ids):
            raise ValueError("component_ids must be non-empty")
        if self.max_bootstrapped_demos < 0:
            raise ValueError("max_bootstrapped_demos cannot be negative")
        if self.max_labeled_demos < 0:
            raise ValueError("max_labeled_demos cannot be negative")
        if self.max_errors < 0:
            raise ValueError("max_errors cannot be negative")
        if self.max_rounds <= 0:
            raise ValueError("max_rounds must be positive")
        if self.metric_threshold is not None and not math.isfinite(
            self.metric_threshold
        ):
            raise ValueError("metric_threshold must be finite")
        if self.kind is FewshotSeedKind.RESET:
            if (
                self.teacher is not None
                or self.labels_only_selection is not None
            ):
                raise ValueError("a reset candidate has no teacher or labels")
        elif self.kind is FewshotSeedKind.LABELS_ONLY:
            if self.teacher is not None or self.labels_only_selection is None:
                raise ValueError(
                    "a labels-only candidate needs only label selections"
                )
        elif self.teacher is None or self.labels_only_selection is not None:
            raise ValueError("a bootstrap candidate needs exactly one teacher")
        if (
            self.trace_selection_projection_version
            != MIPROV2_TRACE_SELECTION_PROJECTION_VERSION
        ):
            raise ValueError("MIPROv2 trace selection projection is fixed")
        return self

    def identity_payload(self) -> dict[str, Any]:
        # Persisted identity keys are an explicit wire contract. Nested
        # records use their canonical JSON projection.
        return {
            "candidate_ordinal": self.candidate_ordinal,
            "candidate_seed": self.candidate_seed,
            "bindings": self.bindings.model_dump(mode="json"),
            "kind": self.kind.value,
            "component_ids": list(self.component_ids),
            "trainset_task_hashes": list(self.trainset_task_hashes),
            "max_bootstrapped_demos": self.max_bootstrapped_demos,
            "max_labeled_demos": self.max_labeled_demos,
            "max_rounds": self.max_rounds,
            "max_errors": self.max_errors,
            "metric_threshold": self.metric_threshold,
            "teacher": (
                None
                if self.teacher is None
                else self.teacher.model_dump(mode="json")
            ),
            "labels_only_selection": (
                None
                if self.labels_only_selection is None
                else self.labels_only_selection.model_dump(mode="json")
            ),
            "trace_selection_projection_version": (
                self.trace_selection_projection_version
            ),
        }

    def identity_hash(self) -> str:
        return compute_identity_hash(
            schema=MIPROV2_BOOTSTRAP_PLAN_SCHEMA,
            schema_version=MIPROV2_BOOTSTRAP_SCHEMA_VERSION,
            payload=self.identity_payload(),
        )


class FewshotCandidatePlanningInputs(BaseModel):
    """All authorities needed to replay the special-seed plan sequence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    bindings: Miprov2DurableBindings
    component_ids: tuple[StrictStr, ...]
    trainset_task_hashes: tuple[StrictStr, ...]
    num_candidate_sets: StrictInt
    max_bootstrapped_demos: StrictInt
    max_labeled_demos: StrictInt
    max_errors: StrictInt
    max_rounds: StrictInt
    labeled_sample: StrictBool
    min_num_samples: StrictInt
    metric_threshold: float | None
    explicit_teacher: StrictBool
    teacher_compiled: StrictBool
    include_non_bootstrapped: StrictBool
    zeroshot_opt: StrictBool


class FewshotCandidatePlanningResult(BaseModel):
    """Candidate plans plus the post-bootstrap shared-RNG checkpoint."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    inputs: FewshotCandidatePlanningInputs
    initial_rng_checkpoint: Miprov2RngCheckpoint
    plans: tuple[FewshotCandidatePlan, ...]
    rng_checkpoint: Miprov2RngCheckpoint
    zeroshot_opt: StrictBool
    proposal_max_bootstrapped_demos: StrictInt
    proposal_max_labeled_demos: StrictInt
    study_uses_demo_candidates: StrictBool

    @model_validator(mode="after")
    def _validate_result(self) -> FewshotCandidatePlanningResult:
        if self.study_uses_demo_candidates == self.zeroshot_opt:
            raise ValueError("zero-shot demo projection is inconsistent")
        if self.zeroshot_opt and (
            self.proposal_max_bootstrapped_demos
            != ZERO_SHOT_BOOTSTRAPPED_DEMOS_IN_PROPOSAL
            or self.proposal_max_labeled_demos
            != ZERO_SHOT_LABELED_DEMOS_IN_PROPOSAL
        ):
            raise ValueError("zero-shot proposal grounding caps are fixed")
        if self.zeroshot_opt != self.inputs.zeroshot_opt:
            raise ValueError(
                "planning result zero-shot mode conflicts with inputs"
            )
        (
            expected_plans,
            expected_checkpoint,
            expected_bootstrapped,
            expected_labeled,
        ) = _build_fewshot_candidate_plans(
            inputs=self.inputs,
            initial_rng_checkpoint=self.initial_rng_checkpoint,
        )
        if self.plans != expected_plans:
            raise ValueError(
                "candidate plan count, order, seeds, kinds, or bindings "
                "do not match canonical replay"
            )
        if self.rng_checkpoint != expected_checkpoint:
            raise ValueError(
                "candidate planning RNG cursor does not match canonical replay"
            )
        if (
            self.proposal_max_bootstrapped_demos != expected_bootstrapped
            or self.proposal_max_labeled_demos != expected_labeled
        ):
            raise ValueError(
                "proposal demo caps do not match canonical candidate planning"
            )
        return self


def create_fewshot_candidate_plans(
    *,
    bindings: Miprov2DurableBindings,
    component_ids: tuple[str, ...],
    trainset_task_hashes: tuple[str, ...],
    num_candidate_sets: int,
    max_bootstrapped_demos: int,
    max_labeled_demos: int,
    max_errors: int,
    max_rounds: int = 1,
    labeled_sample: bool = True,
    min_num_samples: int = 1,
    metric_threshold: float | None = None,
    explicit_teacher: bool = False,
    teacher_compiled: bool = False,
    include_non_bootstrapped: bool = True,
    rng_checkpoint: Miprov2RngCheckpoint,
    zeroshot_opt: bool = False,
) -> FewshotCandidatePlanningResult:
    """Plan DSPy's complete special-seed sequence and shared RNG draws.

    ``metric_threshold`` does not influence RNG draws, but is folded into each
    candidate plan so replay cannot silently change the acceptance rule.
    """

    inputs = FewshotCandidatePlanningInputs(
        bindings=bindings,
        component_ids=component_ids,
        trainset_task_hashes=trainset_task_hashes,
        num_candidate_sets=num_candidate_sets,
        max_bootstrapped_demos=max_bootstrapped_demos,
        max_labeled_demos=max_labeled_demos,
        max_errors=max_errors,
        max_rounds=max_rounds,
        labeled_sample=labeled_sample,
        min_num_samples=min_num_samples,
        metric_threshold=metric_threshold,
        explicit_teacher=explicit_teacher,
        teacher_compiled=teacher_compiled,
        include_non_bootstrapped=include_non_bootstrapped,
        zeroshot_opt=zeroshot_opt,
    )
    plans, checkpoint, proposal_bootstrapped, proposal_labeled = (
        _build_fewshot_candidate_plans(
            inputs=inputs,
            initial_rng_checkpoint=rng_checkpoint,
        )
    )
    return FewshotCandidatePlanningResult(
        inputs=inputs,
        initial_rng_checkpoint=rng_checkpoint,
        plans=plans,
        rng_checkpoint=checkpoint,
        zeroshot_opt=zeroshot_opt,
        proposal_max_bootstrapped_demos=proposal_bootstrapped,
        proposal_max_labeled_demos=proposal_labeled,
        study_uses_demo_candidates=not zeroshot_opt,
    )


def _build_fewshot_candidate_plans(
    *,
    inputs: FewshotCandidatePlanningInputs,
    initial_rng_checkpoint: Miprov2RngCheckpoint,
) -> tuple[
    tuple[FewshotCandidatePlan, ...],
    Miprov2RngCheckpoint,
    int,
    int,
]:
    component_ids = inputs.component_ids
    trainset_task_hashes = inputs.trainset_task_hashes
    if not component_ids:
        raise ValueError("at least one component is required")
    if len(set(component_ids)) != len(component_ids):
        raise ValueError("component_ids must be unique")
    if inputs.num_candidate_sets <= 0:
        raise ValueError("num_candidate_sets must be positive")
    if inputs.max_bootstrapped_demos < 0:
        raise ValueError("max_bootstrapped_demos cannot be negative")
    if inputs.max_labeled_demos < 0:
        raise ValueError("max_labeled_demos cannot be negative")
    if inputs.max_errors < 0:
        raise ValueError("max_errors cannot be negative")
    if inputs.max_rounds <= 0:
        raise ValueError("max_rounds must be positive")
    if inputs.min_num_samples <= 0:
        raise ValueError("min_num_samples must be positive")
    if inputs.metric_threshold is not None and not math.isfinite(
        inputs.metric_threshold
    ):
        raise ValueError("metric_threshold must be finite")
    for index, task_hash in enumerate(trainset_task_hashes):
        require_full_hash(
            task_hash,
            field=f"trainset_task_hashes[{index}]",
        )

    max_bootstrapped_demos = inputs.max_bootstrapped_demos
    max_labeled_demos = inputs.max_labeled_demos
    if inputs.zeroshot_opt:
        if max_bootstrapped_demos != 0 or max_labeled_demos != 0:
            raise ValueError(
                "zeroshot_opt requires zero study demonstration caps"
            )
        max_bootstrapped_demos = ZERO_SHOT_BOOTSTRAPPED_DEMOS_IN_PROPOSAL
        max_labeled_demos = ZERO_SHOT_LABELED_DEMOS_IN_PROPOSAL

    upper_bound = inputs.num_candidate_sets - 3
    candidate_seeds = range(-3, upper_bound)
    generic_seed_is_used = any(
        not (
            (seed == -3 and inputs.include_non_bootstrapped)
            or (
                seed == -2
                and max_labeled_demos > 0
                and inputs.include_non_bootstrapped
            )
            or seed == -1
        )
        for seed in candidate_seeds
    )
    if (
        generic_seed_is_used
        and max_bootstrapped_demos < inputs.min_num_samples
    ):
        raise ValueError(
            "max_bootstrapped_demos must be at least min_num_samples "
            "when a shuffled bootstrap candidate is planned"
        )

    checkpoint = initial_rng_checkpoint
    rng = checkpoint.state.restore()
    plans: list[FewshotCandidatePlan] = []
    for candidate_seed in range(-3, upper_bound):
        trainset = list(trainset_task_hashes)
        ordinal = len(plans)
        if candidate_seed == -3 and inputs.include_non_bootstrapped:
            plans.append(
                FewshotCandidatePlan(
                    candidate_ordinal=ordinal,
                    candidate_seed=candidate_seed,
                    bindings=inputs.bindings,
                    kind=FewshotSeedKind.RESET,
                    component_ids=component_ids,
                    trainset_task_hashes=tuple(trainset),
                    max_bootstrapped_demos=max_bootstrapped_demos,
                    max_labeled_demos=max_labeled_demos,
                    max_rounds=inputs.max_rounds,
                    max_errors=inputs.max_errors,
                    metric_threshold=inputs.metric_threshold,
                    teacher=None,
                    labels_only_selection=None,
                )
            )
            continue
        if (
            candidate_seed == -2
            and max_labeled_demos > 0
            and inputs.include_non_bootstrapped
        ):
            labels = plan_labeled_selection(
                component_ids=component_ids,
                trainset_size=len(trainset),
                k=max_labeled_demos,
                sample=inputs.labeled_sample,
            )
            plans.append(
                FewshotCandidatePlan(
                    candidate_ordinal=ordinal,
                    candidate_seed=candidate_seed,
                    bindings=inputs.bindings,
                    kind=FewshotSeedKind.LABELS_ONLY,
                    component_ids=component_ids,
                    trainset_task_hashes=tuple(trainset),
                    max_bootstrapped_demos=max_bootstrapped_demos,
                    max_labeled_demos=max_labeled_demos,
                    max_rounds=inputs.max_rounds,
                    max_errors=inputs.max_errors,
                    metric_threshold=inputs.metric_threshold,
                    teacher=None,
                    labels_only_selection=labels,
                )
            )
            continue

        candidate_max_bootstrapped = max_bootstrapped_demos
        if candidate_seed != -1:
            before_shuffle = tuple(trainset)
            rng.shuffle(trainset)
            checkpoint = checkpoint.append(
                rng=rng,
                phase="bootstrap",
                operation="shuffle",
                arguments=(before_shuffle,),
                result=tuple(trainset),
            )
            candidate_max_bootstrapped = rng.randint(
                inputs.min_num_samples,
                max_bootstrapped_demos,
            )
            checkpoint = checkpoint.append(
                rng=rng,
                phase="bootstrap",
                operation="randint",
                arguments=(inputs.min_num_samples, max_bootstrapped_demos),
                result=candidate_max_bootstrapped,
            )
        reset_for_labels = bool(
            max_labeled_demos and not inputs.teacher_compiled
        )
        labeled_selection = (
            plan_labeled_selection(
                component_ids=component_ids,
                trainset_size=len(trainset),
                k=max_labeled_demos,
                sample=True,
            )
            if reset_for_labels
            else None
        )
        plans.append(
            FewshotCandidatePlan(
                candidate_ordinal=ordinal,
                candidate_seed=candidate_seed,
                bindings=inputs.bindings,
                kind=FewshotSeedKind.BOOTSTRAP,
                component_ids=component_ids,
                trainset_task_hashes=tuple(trainset),
                max_bootstrapped_demos=candidate_max_bootstrapped,
                max_labeled_demos=max_labeled_demos,
                max_rounds=inputs.max_rounds,
                max_errors=inputs.max_errors,
                metric_threshold=inputs.metric_threshold,
                teacher=TeacherPreparationPlan(
                    source=(
                        TeacherSource.EXPLICIT
                        if inputs.explicit_teacher
                        else TeacherSource.STUDENT
                    ),
                    reset_before_labeled_compile=reset_for_labels,
                    labeled_selection=labeled_selection,
                ),
                labels_only_selection=None,
            )
        )
    return (
        tuple(plans),
        checkpoint,
        max_bootstrapped_demos,
        max_labeled_demos,
    )


class BootstrapCompilerState(BaseModel):
    """Folded state for exactly one ``BootstrapFewShot`` compiler."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    plan_identity_hash: StrictStr
    task_cursor: StrictInt = 0
    round_cursor: StrictInt = 0
    error_count: StrictInt = 0
    attempt_count: StrictInt = 0
    bootstrapped_task_indices: tuple[StrictInt, ...] = ()
    augmented_demos: Mapping[StrictStr, tuple[ComponentDemo, ...]] = Field(
        default_factory=lambda: MappingProxyType({})
    )
    evidence: tuple[BootstrapFoldEvidence, ...] = ()
    terminal_failure: BootstrapTerminalFailure | None = None

    def model_post_init(self, _context: Any) -> None:
        if not isinstance(self.augmented_demos, MappingProxyType):
            object.__setattr__(
                self,
                "augmented_demos",
                MappingProxyType(
                    {
                        component_id: tuple(demos)
                        for component_id, demos in self.augmented_demos.items()
                    }
                ),
            )

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        if deep:
            payload = self.model_dump(mode="json")
            payload.update(update or {})
            return type(self).model_validate(payload)
        copied = super().model_copy(update=update, deep=deep)
        copied.model_post_init(None)
        return copied

    @field_serializer("augmented_demos")
    def _serialize_augmented_demos(
        self,
        value: Mapping[str, tuple[ComponentDemo, ...]],
    ) -> dict[str, list[dict[str, Any]]]:
        return {
            component_id: [demo.model_dump(mode="json") for demo in demos]
            for component_id, demos in value.items()
        }

    @model_validator(mode="after")
    def _validate_state(self) -> BootstrapCompilerState:
        require_full_hash(self.plan_identity_hash, field="plan_identity_hash")
        if (
            min(
                self.task_cursor,
                self.round_cursor,
                self.error_count,
                self.attempt_count,
            )
            < 0
        ):
            raise ValueError("bootstrap counters cannot be negative")
        if tuple(sorted(set(self.bootstrapped_task_indices))) != (
            self.bootstrapped_task_indices
        ):
            raise ValueError(
                "bootstrapped task indices must be unique and ordered"
            )
        if self.attempt_count != len(self.evidence):
            raise ValueError(
                "bootstrap attempt count must equal append-only evidence count"
            )
        if self.terminal_failure is not None:
            if not self.evidence or self.evidence[-1].result.error is None:
                raise ValueError(
                    "terminal bootstrap failure requires final error evidence"
                )
        return self


def initial_compiler_state(
    plan: FewshotCandidatePlan,
) -> BootstrapCompilerState:
    if plan.kind is not FewshotSeedKind.BOOTSTRAP:
        raise ValueError("only bootstrap candidates have compiler state")
    return BootstrapCompilerState(
        plan_identity_hash=plan.identity_hash(),
        augmented_demos={
            component_id: () for component_id in plan.component_ids
        },
    )


class BootstrapAttemptPlan(BaseModel):
    """One task-model effect request selected by the pure state machine."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    bindings: Miprov2DurableBindings
    plan_identity_hash: StrictStr
    task_index: StrictInt
    task_hash: StrictStr
    round_index: StrictInt
    exclude_equal_task_from_all_teacher_components: StrictBool = True
    restore_teacher_demos_after_effect: StrictBool = True
    copy_task_model: StrictBool
    rollout_id: StrictInt | None
    temperature: float | None

    @model_validator(mode="after")
    def _validate_attempt(self) -> BootstrapAttemptPlan:
        if self.task_index < 0 or self.round_index < 0:
            raise ValueError("attempt indices cannot be negative")
        require_full_hash(self.plan_identity_hash, field="plan_identity_hash")
        require_full_hash(self.task_hash, field="task_hash")
        if not self.exclude_equal_task_from_all_teacher_components:
            raise ValueError(
                "the current task must be excluded by equality from every "
                "teacher component"
            )
        if not self.restore_teacher_demos_after_effect:
            raise ValueError("teacher demos must be restored after the effect")
        retry = self.round_index > 0
        if (
            self.copy_task_model != retry
            or (self.rollout_id is not None) != retry
            or (self.temperature is not None) != retry
        ):
            raise ValueError("only rounds after zero copy the task model")
        if retry and (
            self.rollout_id != self.round_index or self.temperature != 1.0
        ):
            raise ValueError(
                "retry rollout_id must equal round and temperature must be 1.0"
            )
        return self

    def identity_payload(self) -> dict[str, Any]:
        # Persisted identity keys are an explicit wire contract. The durable
        # bindings use their canonical JSON projection.
        return {
            "bindings": self.bindings.model_dump(mode="json"),
            "plan_identity_hash": self.plan_identity_hash,
            "task_index": self.task_index,
            "task_hash": self.task_hash,
            "round_index": self.round_index,
            "exclude_equal_task_from_all_teacher_components": (
                self.exclude_equal_task_from_all_teacher_components
            ),
            "restore_teacher_demos_after_effect": (
                self.restore_teacher_demos_after_effect
            ),
            "copy_task_model": self.copy_task_model,
            "rollout_id": self.rollout_id,
            "temperature": self.temperature,
        }

    def identity_hash(self) -> str:
        return compute_identity_hash(
            schema=MIPROV2_BOOTSTRAP_ATTEMPT_SCHEMA,
            schema_version=MIPROV2_BOOTSTRAP_SCHEMA_VERSION,
            payload=self.identity_payload(),
        )


def next_bootstrap_attempt(
    plan: FewshotCandidatePlan,
    state: BootstrapCompilerState,
) -> BootstrapAttemptPlan | None:
    """Return the next effect, or ``None`` once DSPy's loops would stop."""

    _validate_state_for_plan(plan, state)
    if state.terminal_failure is not None:
        failure = state.terminal_failure
        raise BootstrapErrorLimitReached(
            error_count=failure.error_count,
            max_errors=failure.max_errors,
            error=failure.error,
            state=state,
        )
    return _next_bootstrap_attempt_unchecked(plan, state)


def _next_bootstrap_attempt_unchecked(
    plan: FewshotCandidatePlan,
    state: BootstrapCompilerState,
) -> BootstrapAttemptPlan | None:
    if (
        plan.max_rounds <= 0
        or len(state.bootstrapped_task_indices) >= plan.max_bootstrapped_demos
        or state.task_cursor >= len(plan.trainset_task_hashes)
    ):
        return None
    round_index = state.round_cursor
    return BootstrapAttemptPlan(
        bindings=plan.bindings,
        plan_identity_hash=state.plan_identity_hash,
        task_index=state.task_cursor,
        task_hash=plan.trainset_task_hashes[state.task_cursor],
        round_index=round_index,
        copy_task_model=round_index > 0,
        rollout_id=round_index if round_index > 0 else None,
        temperature=1.0 if round_index > 0 else None,
    )


class BootstrapRolloutResult(BaseModel):
    """Normalized evidence returned by one canonical bootstrap evaluation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    attempt_identity_hash: StrictStr
    source_rollout_identity: StrictStr
    source_trace_identity: StrictStr
    source_output_identity: StrictStr
    source_score_identity: StrictStr
    metric_present: StrictBool
    score: MetricValue | None
    trace_steps: tuple[ObservedTraceStep, ...] = ()
    error: StrictStr | None = None

    @model_validator(mode="after")
    def _validate_result(self) -> BootstrapRolloutResult:
        for field in (
            "attempt_identity_hash",
            "source_rollout_identity",
            "source_trace_identity",
            "source_output_identity",
            "source_score_identity",
        ):
            require_full_hash(getattr(self, field), field=field)
        if self.error is not None:
            if not self.error:
                raise ValueError("error must be non-empty when present")
            if self.score is not None or self.trace_steps:
                raise ValueError(
                    "failed rollout cannot carry score or trace steps"
                )
        elif self.metric_present and self.score is None:
            raise ValueError("a present metric must have a score")
        elif not self.metric_present and self.score is not None:
            raise ValueError("an absent metric cannot have a score")
        return self


class BootstrapTerminalFailure(BaseModel):
    """Durable terminal error reached after charging the final attempt."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    attempt_identity_hash: StrictStr
    error_count: StrictInt
    max_errors: StrictInt
    error: StrictStr

    @model_validator(mode="after")
    def _validate_terminal(self) -> BootstrapTerminalFailure:
        require_full_hash(
            self.attempt_identity_hash,
            field="attempt_identity_hash",
        )
        if self.error_count < self.max_errors or not self.error:
            raise ValueError(
                "terminal bootstrap failure must reach the error limit"
            )
        return self


class BootstrapFoldEvidence(BaseModel):
    """One append-only attempted rollout and its derived acceptance."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    attempt: BootstrapAttemptPlan
    result: BootstrapRolloutResult
    acceptance: BootstrapAcceptance | None

    @model_validator(mode="after")
    def _validate_fold(self) -> BootstrapFoldEvidence:
        if self.result.attempt_identity_hash != self.attempt.identity_hash():
            raise ValueError(
                "bootstrap evidence result belongs to another attempt"
            )
        if self.result.error is not None:
            if self.acceptance is not None:
                raise ValueError(
                    "failed bootstrap evidence cannot be accepted"
                )
            return self
        if self.acceptance is None:
            raise ValueError(
                "completed bootstrap evidence requires acceptance"
            )
        expected = _acceptance_for(
            attempt=self.attempt,
            result=self.result,
            metric_threshold=self.acceptance.metric_threshold,
        )
        if self.acceptance != expected:
            raise ValueError(
                "bootstrap acceptance does not match rollout evidence"
            )
        return self


BootstrapCompilerState.model_rebuild()


class BootstrapErrorLimitReached(RuntimeError):
    """DSPy's per-compiler error limit was reached."""

    def __init__(
        self,
        *,
        error_count: int,
        max_errors: int,
        error: str,
        state: BootstrapCompilerState | None = None,
    ):
        self.error_count = error_count
        self.max_errors = max_errors
        self.error = error
        self.state = state
        super().__init__(
            f"bootstrap error {error_count} reached max_errors "
            f"{max_errors}: {error}"
        )


def fold_bootstrap_result(
    *,
    plan: FewshotCandidatePlan,
    state: BootstrapCompilerState,
    attempt: BootstrapAttemptPlan,
    result: BootstrapRolloutResult,
    metric_threshold: float | None,
    component_ids: tuple[str, ...],
) -> BootstrapCompilerState:
    """Fold one effect using DSPy's success, trace, and cursor rules."""

    _validate_state_for_plan(plan, state)
    if component_ids != plan.component_ids:
        raise ValueError("component_ids do not match the candidate plan")
    if metric_threshold != plan.metric_threshold:
        raise ValueError("metric_threshold does not match the candidate plan")
    expected = _next_bootstrap_attempt_unchecked(plan, state)
    if expected is None or attempt != expected:
        raise ValueError("bootstrap result does not match the next attempt")
    if result.attempt_identity_hash != attempt.identity_hash():
        raise ValueError("bootstrap result belongs to another attempt")
    acceptance = (
        None
        if result.error is not None
        else _acceptance_for(
            attempt=attempt,
            result=result,
            metric_threshold=metric_threshold,
        )
    )
    event = BootstrapFoldEvidence(
        attempt=attempt,
        result=result,
        acceptance=acceptance,
    )
    advanced = _apply_bootstrap_event_unchecked(
        plan=plan,
        state=state,
        event=event,
    )
    _validate_state_for_plan(plan, advanced)
    return advanced


def _apply_bootstrap_event_unchecked(
    *,
    plan: FewshotCandidatePlan,
    state: BootstrapCompilerState,
    event: BootstrapFoldEvidence,
) -> BootstrapCompilerState:
    attempt = event.attempt
    result = event.result
    evidence = (*state.evidence, event)
    if result.error is not None:
        error_count = state.error_count + 1
        attempted = state.model_copy(
            update={
                "error_count": error_count,
                "attempt_count": state.attempt_count + 1,
                "evidence": evidence,
            }
        )
        if error_count >= plan.max_errors:
            return attempted.model_copy(
                update={
                    "terminal_failure": BootstrapTerminalFailure(
                        attempt_identity_hash=attempt.identity_hash(),
                        error_count=error_count,
                        max_errors=plan.max_errors,
                        error=result.error,
                    )
                }
            )
        return _advance_state(
            plan=plan,
            state=attempted,
            success=False,
        )

    acceptance = event.acceptance
    assert acceptance is not None
    if not acceptance.accepted:
        return _advance_state(
            plan=plan,
            state=state.model_copy(
                update={
                    "attempt_count": state.attempt_count + 1,
                    "evidence": evidence,
                }
            ),
            success=False,
        )

    known_components = set(plan.component_ids)
    grouped: dict[str, list[ObservedTraceStep]] = {}
    for step in result.trace_steps:
        if step.component_id not in known_components:
            continue
        assert step.component_id is not None
        grouped.setdefault(step.component_id, []).append(step)

    augmented = dict(state.augmented_demos)
    for component_id, steps in grouped.items():
        chosen = _choose_trace_step(steps)
        demo = ComponentDemo(
            component_id=component_id,
            source_kind=DemoSourceKind.BOOTSTRAPPED,
            inputs=chosen.inputs,
            outputs=chosen.outputs,
            augmented=True,
            source_task_hash=attempt.task_hash,
            source_rollout_identity=result.source_rollout_identity,
            source_trace_identity=result.source_trace_identity,
            source_output_identity=result.source_output_identity,
            source_score_identity=result.source_score_identity,
            source_trace_index=chosen.trace_index,
            score=result.score,
            acceptance_identity_hash=acceptance.identity_hash(),
        )
        augmented[component_id] = (*augmented.get(component_id, ()), demo)

    accepted_indices = (
        *state.bootstrapped_task_indices,
        attempt.task_index,
    )
    accepted = state.model_copy(
        update={
            "attempt_count": state.attempt_count + 1,
            "bootstrapped_task_indices": accepted_indices,
            "augmented_demos": augmented,
            "evidence": evidence,
        }
    )
    return _advance_state(plan=plan, state=accepted, success=True)


def _acceptance_for(
    *,
    attempt: BootstrapAttemptPlan,
    result: BootstrapRolloutResult,
    metric_threshold: float | None,
) -> BootstrapAcceptance:
    return BootstrapAcceptance(
        source_task_hash=attempt.task_hash,
        source_rollout_identity=result.source_rollout_identity,
        source_trace_identity=result.source_trace_identity,
        source_output_identity=result.source_output_identity,
        source_score_identity=result.source_score_identity,
        metric_present=result.metric_present,
        score=result.score,
        metric_threshold=metric_threshold,
        accepted=_accept_result(result, metric_threshold),
    )


def _accept_result(
    result: BootstrapRolloutResult,
    metric_threshold: float | None,
) -> bool:
    if not result.metric_present:
        return True
    assert result.score is not None
    if metric_threshold:
        return bool(result.score >= metric_threshold)
    return bool(result.score)


def _choose_trace_step(
    steps: list[ObservedTraceStep],
) -> ObservedTraceStep:
    if len(steps) == 1:
        return steps[0]
    # Frozen DSPy constructs ``Example(augmented=True, **inputs, **outputs)``
    # for every repeated trace, then seeds ``Random`` with
    # ``Hasher.hash(tuple(demos))``. ``Hasher.hash`` is SHA-256 over the
    # standard-library pickle. Keep the exact hash algorithm and demo field
    # order while projecting away Whetstone-only trace/component metadata.
    group_identity = hashlib.sha256(
        _frozen_dspy_demo_tuple_pickle(steps)
    ).hexdigest()
    rng = random.Random(group_identity)
    return rng.choice(steps[:-1]) if rng.random() < 0.5 else steps[-1]


class _FrozenDspyExample:
    """Pickle-compatible projection of DSPy's frozen ``Example`` class.

    Only the object state and global class reference participate in hashing.
    This proxy is never unpickled and intentionally carries no DSPy behavior.
    """

    def __init__(self, **kwargs: Any) -> None:
        self._store: dict[str, Any] = {}
        self._demos: list[Any] = []
        self._input_keys: set[str] | None = None
        self._store.update(kwargs)


_PICKLE_SAVE_TYPE = pickle._Pickler.dispatch[type]


class _FrozenDspyExamplePickler(pickle._Pickler):
    """Emit DSPy's class global while preserving stock protocol-4 semantics."""

    dispatch: Any = (  # ty: ignore[invalid-attribute-override]
        pickle._Pickler.dispatch.copy()
    )

    def save_type(self: Any, obj: type[Any]) -> None:
        if obj is not _FrozenDspyExample:
            _PICKLE_SAVE_TYPE(self, obj)
            return
        self.save("dspy.primitives.example")
        self.save("Example")
        self.write(pickle.STACK_GLOBAL)
        self.memoize(obj)

    dispatch[type] = save_type


def _frozen_dspy_demo_tuple_pickle(
    steps: list[ObservedTraceStep],
) -> bytes:
    """Serialize repeated traces exactly as frozen DSPy ``Hasher.hash``.

    Protocol 4 is pinned because DSPy's ambient ``pickle.dumps`` default at
    the reference commit was protocol 4. Field mapping order and shared
    reference topology are preserved exactly.
    """

    demos: list[_FrozenDspyExample] = []
    for step in steps:
        demos.append(
            _FrozenDspyExample(
                augmented=True,
                **step.inputs.to_json(),
                **step.outputs.to_json(),
            )
        )
    buffer = io.BytesIO()
    _FrozenDspyExamplePickler(buffer, protocol=4).dump(tuple(demos))
    return buffer.getvalue()


def _advance_state(
    *,
    plan: FewshotCandidatePlan,
    state: BootstrapCompilerState,
    success: bool,
) -> BootstrapCompilerState:
    if success or state.round_cursor + 1 >= plan.max_rounds:
        return state.model_copy(
            update={
                "task_cursor": state.task_cursor + 1,
                "round_cursor": 0,
            }
        )
    return state.model_copy(update={"round_cursor": state.round_cursor + 1})


def _validate_state_for_plan(
    plan: FewshotCandidatePlan,
    state: BootstrapCompilerState,
) -> None:
    if plan.kind is not FewshotSeedKind.BOOTSTRAP:
        raise ValueError("only bootstrap candidates can be folded")
    if state.plan_identity_hash != plan.identity_hash():
        raise ValueError("bootstrap state belongs to another candidate plan")
    if set(state.augmented_demos) != set(plan.component_ids):
        raise ValueError(
            "bootstrap state's component mapping does not match the plan"
        )
    if any(
        demo.component_id != component_id
        for component_id, demos in state.augmented_demos.items()
        for demo in demos
    ):
        raise ValueError(
            "bootstrap state contains a demo under another component"
        )
    replay = BootstrapCompilerState.model_construct(
        plan_identity_hash=plan.identity_hash(),
        task_cursor=0,
        round_cursor=0,
        error_count=0,
        attempt_count=0,
        bootstrapped_task_indices=(),
        augmented_demos={
            component_id: () for component_id in plan.component_ids
        },
        evidence=(),
        terminal_failure=None,
    )
    for event in state.evidence:
        if replay.terminal_failure is not None:
            raise ValueError(
                "bootstrap evidence continues after terminal failure"
            )
        expected = _next_bootstrap_attempt_unchecked(plan, replay)
        if expected is None or event.attempt != expected:
            raise ValueError(
                "bootstrap evidence skips or rewrites the next attempt"
            )
        if event.result.attempt_identity_hash != expected.identity_hash():
            raise ValueError(
                "bootstrap evidence result belongs to another attempt"
            )
        if event.result.error is None:
            expected_acceptance = _acceptance_for(
                attempt=expected,
                result=event.result,
                metric_threshold=plan.metric_threshold,
            )
            if event.acceptance != expected_acceptance:
                raise ValueError(
                    "bootstrap evidence acceptance conflicts with "
                    "candidate plan"
                )
        elif event.acceptance is not None:
            raise ValueError("failed bootstrap evidence cannot be accepted")
        replay = _apply_bootstrap_event_unchecked(
            plan=plan,
            state=replay,
            event=event,
        )
    if replay.model_dump(mode="json") != state.model_dump(mode="json"):
        raise ValueError(
            "bootstrap state cursors or demos do not match evidence replay"
        )


def materialize_reset_demo_set(
    *,
    plan: FewshotCandidatePlan,
    component_ids: tuple[str, ...],
) -> ComponentDemoSet:
    if plan.kind is not FewshotSeedKind.RESET:
        raise ValueError("plan is not the reset candidate")
    if component_ids != plan.component_ids:
        raise ValueError("component_ids do not match the candidate plan")
    return ComponentDemoSet(
        candidate_seed=plan.candidate_seed,
        components=tuple(
            ComponentDemoSequence(component_id=component_id)
            for component_id in component_ids
        ),
    )


def materialize_labels_only_demo_set(
    *,
    plan: FewshotCandidatePlan,
    labeled_trainset: tuple[LabeledTaskDemo, ...],
) -> ComponentDemoSet:
    if (
        plan.kind is not FewshotSeedKind.LABELS_ONLY
        or plan.labels_only_selection is None
    ):
        raise ValueError("plan is not the labels-only candidate")
    _require_plan_ordered_labeled_trainset(plan, labeled_trainset)
    return _materialize_labeled_selection(
        candidate_seed=plan.candidate_seed,
        selection=plan.labels_only_selection,
        labeled_trainset=labeled_trainset,
    )


def materialize_bootstrap_demo_set(
    *,
    plan: FewshotCandidatePlan,
    state: BootstrapCompilerState,
    labeled_trainset: tuple[LabeledTaskDemo, ...],
    component_ids: tuple[str, ...],
) -> ComponentDemoSet:
    """Match ``BootstrapFewShot._train``, including raw-demo narrowing."""

    _validate_state_for_plan(plan, state)
    if component_ids != plan.component_ids:
        raise ValueError("component_ids do not match the candidate plan")
    _require_plan_ordered_labeled_trainset(plan, labeled_trainset)
    validation = [
        task
        for index, task in enumerate(labeled_trainset)
        if index not in set(state.bootstrapped_task_indices)
    ]
    random.Random(0).shuffle(validation)

    rng = random.Random(0)
    raw_demos = validation
    components: list[ComponentDemoSequence] = []
    for component_id in component_ids:
        augmented = state.augmented_demos.get(component_id, ())[
            : plan.max_bootstrapped_demos
        ]
        sample_size = min(
            plan.max_labeled_demos - len(augmented), len(raw_demos)
        )
        sample_size = max(0, sample_size)
        raw_demos = rng.sample(raw_demos, sample_size)
        labeled = tuple(task.for_component(component_id) for task in raw_demos)
        components.append(
            ComponentDemoSequence(
                component_id=component_id,
                demos=(*augmented, *labeled),
            )
        )
    return ComponentDemoSet(
        candidate_seed=plan.candidate_seed,
        components=tuple(components),
    )


def _require_plan_ordered_labeled_trainset(
    plan: FewshotCandidatePlan,
    labeled_trainset: tuple[LabeledTaskDemo, ...],
) -> None:
    actual = tuple(task.source_task_hash for task in labeled_trainset)
    if actual != plan.trainset_task_hashes:
        raise ValueError(
            "labeled_trainset order does not match candidate plan task order"
        )


def _materialize_labeled_selection(
    *,
    candidate_seed: int,
    selection: LabeledSelectionPlan,
    labeled_trainset: tuple[LabeledTaskDemo, ...],
) -> ComponentDemoSet:
    components: list[ComponentDemoSequence] = []
    for component_id, indices in zip(
        selection.component_ids,
        selection.per_component_task_indices,
        strict=True,
    ):
        try:
            demos = tuple(
                labeled_trainset[index].for_component(component_id)
                for index in indices
            )
        except IndexError as exc:
            raise ValueError(
                "labeled selection index is outside the trainset"
            ) from exc
        components.append(
            ComponentDemoSequence(
                component_id=component_id,
                demos=demos,
            )
        )
    return ComponentDemoSet(
        candidate_seed=candidate_seed,
        components=tuple(components),
    )


__all__ = [
    "MIPROV2_BOOTSTRAP_ATTEMPT_SCHEMA",
    "MIPROV2_BOOTSTRAP_PLAN_SCHEMA",
    "MIPROV2_TRACE_SELECTION_PROJECTION_VERSION",
    "ZERO_SHOT_BOOTSTRAPPED_DEMOS_IN_PROPOSAL",
    "ZERO_SHOT_LABELED_DEMOS_IN_PROPOSAL",
    "BootstrapAttemptPlan",
    "BootstrapCompilerState",
    "BootstrapErrorLimitReached",
    "BootstrapFoldEvidence",
    "BootstrapRolloutResult",
    "BootstrapTerminalFailure",
    "FewshotCandidatePlan",
    "FewshotCandidatePlanningInputs",
    "FewshotCandidatePlanningResult",
    "FewshotSeedKind",
    "LabeledSelectionPlan",
    "TeacherPreparationPlan",
    "TeacherSource",
    "create_fewshot_candidate_plans",
    "fold_bootstrap_result",
    "initial_compiler_state",
    "materialize_bootstrap_demo_set",
    "materialize_labels_only_demo_set",
    "materialize_reset_demo_set",
    "next_bootstrap_attempt",
    "plan_labeled_selection",
]
