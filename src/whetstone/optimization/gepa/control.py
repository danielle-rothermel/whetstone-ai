from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    StrictBool,
    StrictInt,
    StrictStr,
    model_validator,
)

from whetstone.core.identity import (
    compute_identity_hash,
    require_full_hash,
)
from whetstone.experiment.binding import EvalConfigRef
from whetstone.optimization.gepa.prompts import (
    GEPA_REFLECTION_PROMPT_SCHEMA,
    GEPA_REFLECTION_PROMPT_SCHEMA_VERSION,
    GEPA_REFLECTION_RESPONSE_PARSER_SCHEMA,
    GEPA_REFLECTION_RESPONSE_PARSER_SCHEMA_VERSION,
)
from whetstone.optimization.gepa.source import (
    GEPA_DISTRIBUTION_VERSION,
    GEPA_REPOSITORY_COMMIT,
    GEPA_SDIST_SHA256,
    GEPA_SOURCE_MANIFEST_HASH,
    GEPA_WHEEL_SHA256,
)
from whetstone.optimization.proposal.proposer import ProposerConfig

GEPA_ALGORITHM_VERSION = "gepa_upstream_0_1_1/v1"
GEPA_DSPY_REFERENCE_COMMIT = "6f68dcdb3ef46d70bf0c12596699ebc44e82d6b0"
GEPA_CONTROL_SCHEMA = "whetstone.gepa_optimizer_config"
GEPA_CONTROL_SCHEMA_VERSION = 1
GEPA_ADAPTER_SCHEMA_VERSION = "whetstone.gepa_upstream_adapter/v1"
GEPA_RESULT_SCHEMA_VERSION = "whetstone.gepa_detailed_result/v1"
GEPA_AUTO_CANDIDATES: dict[str, int] = {
    "light": 6,
    "medium": 12,
    "heavy": 18,
}
GEPA_MERGE_POLICY_IDENTITY_HASH = compute_identity_hash(
    schema="whetstone.gepa.merge_policy",
    schema_version=1,
    payload={
        "distribution_version": GEPA_DISTRIBUTION_VERSION,
        "repository_commit": GEPA_REPOSITORY_COMMIT,
        "source_manifest_hash": GEPA_SOURCE_MANIFEST_HASH,
        "source_file": "gepa/proposer/merge.py",
        "policy": "upstream_component_recombination_evaluation_only",
    },
)

GepaAutoMode = Literal["light", "medium", "heavy"]
GepaCandidateSelection = Literal["pareto", "current_best"]
GepaComponentSelector = Literal["round_robin", "all"]


def gepa_auto_budget(
    *,
    num_predictors: int,
    num_candidates: int,
    valset_size: int,
    minibatch_size: int = 35,
    full_eval_steps: int = 5,
) -> int:
    """Copy DSPy's frozen GEPA ``auto_budget`` arithmetic exactly."""

    num_trials = int(
        max(
            2 * (num_predictors * 2) * math.log2(num_candidates),
            1.5 * num_candidates,
        )
    )
    if num_trials < 0 or valset_size < 0 or minibatch_size < 0:
        raise ValueError(
            "num_trials, valset_size, and minibatch_size must be >= 0."
        )
    if full_eval_steps < 1:
        raise ValueError("full_eval_steps must be >= 1.")

    total = valset_size
    total += num_candidates * 5
    total += num_trials * minibatch_size
    if num_trials == 0:
        return total
    periodic_fulls = (num_trials + 1) // full_eval_steps + 1
    extra_final = 1 if num_trials < full_eval_steps else 0
    total += (periodic_fulls + extra_final) * valset_size
    return total


class GepaControl(BaseModel):
    """Fully resolved identity document for one canonical GEPA run.

    No Python callback, sampler, policy, selector, logger, or filesystem
    checkpoint object is accepted here. All upstream behavior is selected by
    frozen, serializable controls and runs through public ``gepa.optimize``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    reflection_model: ProposerConfig
    metric: EvalConfigRef
    reward_policy_hash: StrictStr
    evaluation_execution_policy_hash: StrictStr
    proposal_execution_policy_hash: StrictStr
    proposal_prompt_adapter_identity_hash: StrictStr
    proposal_durability_policy_identity_hash: StrictStr
    task_model_identity_hash: StrictStr
    prompt_format_identity_hash: StrictStr
    prompt_binding_identity_hash: StrictStr
    source_trainset_task_identities: tuple[StrictStr, ...]
    source_valset_task_identities: tuple[StrictStr, ...] | None
    trainset_task_identities: tuple[StrictStr, ...]
    valset_task_identities: tuple[StrictStr, ...]
    component_names: tuple[StrictStr, ...]
    num_predictors: StrictInt

    auto: GepaAutoMode | None
    max_full_evals: StrictInt | None
    max_metric_calls: StrictInt | None
    resolved_max_metric_calls: StrictInt
    reflection_minibatch_size: StrictInt = 3
    candidate_selection_strategy: GepaCandidateSelection = "pareto"
    skip_perfect_score: StrictBool = True
    add_format_failure_as_feedback: StrictBool = False
    component_selector: GepaComponentSelector = "round_robin"
    use_merge: StrictBool = True
    max_merge_invocations: StrictInt = 5
    merge_val_overlap_floor: StrictInt = 5
    num_threads: StrictInt | None = None
    failure_score: float = 0.0
    perfect_score: float = 1.0
    track_stats: StrictBool = False
    track_best_outputs: StrictBool = False
    warn_on_score_mismatch: StrictBool = True
    seed: StrictInt = 0

    frontier_type: Literal["instance"] = "instance"
    batch_sampler: Literal["epoch_shuffled"] = "epoch_shuffled"
    val_evaluation_policy: Literal["full_eval"] = "full_eval"
    use_cloudpickle: Literal[False] = False
    cache_evaluation: Literal[False] = False

    algorithm_version: StrictStr = GEPA_ALGORITHM_VERSION
    dspy_reference_commit: StrictStr = GEPA_DSPY_REFERENCE_COMMIT
    gepa_distribution_version: StrictStr = GEPA_DISTRIBUTION_VERSION
    gepa_repository_commit: StrictStr = GEPA_REPOSITORY_COMMIT
    gepa_wheel_sha256: StrictStr = GEPA_WHEEL_SHA256
    gepa_sdist_sha256: StrictStr = GEPA_SDIST_SHA256
    gepa_source_manifest_hash: StrictStr = GEPA_SOURCE_MANIFEST_HASH
    merge_policy_identity_hash: StrictStr = GEPA_MERGE_POLICY_IDENTITY_HASH
    adapter_schema_version: StrictStr = GEPA_ADAPTER_SCHEMA_VERSION
    result_schema_version: StrictStr = GEPA_RESULT_SCHEMA_VERSION
    reflection_prompt_schema: StrictStr = GEPA_REFLECTION_PROMPT_SCHEMA
    reflection_prompt_schema_version: StrictInt = (
        GEPA_REFLECTION_PROMPT_SCHEMA_VERSION
    )
    reflection_parser_schema: StrictStr = (
        GEPA_REFLECTION_RESPONSE_PARSER_SCHEMA
    )
    reflection_parser_schema_version: StrictInt = (
        GEPA_REFLECTION_RESPONSE_PARSER_SCHEMA_VERSION
    )

    @model_validator(mode="after")
    def _validate(self) -> GepaControl:
        for field in (
            "reward_policy_hash",
            "evaluation_execution_policy_hash",
            "proposal_execution_policy_hash",
            "proposal_prompt_adapter_identity_hash",
            "proposal_durability_policy_identity_hash",
            "task_model_identity_hash",
            "prompt_format_identity_hash",
            "prompt_binding_identity_hash",
            "gepa_wheel_sha256",
            "gepa_sdist_sha256",
            "gepa_source_manifest_hash",
            "merge_policy_identity_hash",
        ):
            require_full_hash(getattr(self, field), field=field)
        for field, identities in (
            (
                "source_trainset_task_identities",
                self.source_trainset_task_identities,
            ),
            ("trainset_task_identities", self.trainset_task_identities),
            ("valset_task_identities", self.valset_task_identities),
        ):
            if not identities:
                raise ValueError(f"{field} must be non-empty")
            for identity in identities:
                require_full_hash(identity, field=field)
            if len(set(identities)) != len(identities):
                raise ValueError(f"{field} must contain unique identities")
        if self.source_valset_task_identities is not None:
            if not self.source_valset_task_identities:
                raise ValueError(
                    "source_valset_task_identities must be non-empty when set"
                )
            for identity in self.source_valset_task_identities:
                require_full_hash(
                    identity, field="source_valset_task_identities"
                )
            if len(set(self.source_valset_task_identities)) != len(
                self.source_valset_task_identities
            ):
                raise ValueError(
                    "source_valset_task_identities must contain unique "
                    "identities"
                )
        if self.trainset_task_identities != (
            self.source_trainset_task_identities
        ):
            raise ValueError(
                "canonical GEPA does not silently resample the trainset"
            )
        expected_valset = (
            self.source_trainset_task_identities
            if self.source_valset_task_identities is None
            else self.source_valset_task_identities
        )
        if self.valset_task_identities != expected_valset:
            raise ValueError(
                "valset identities do not match the ordered source binding"
            )
        if self.num_predictors < 1:
            raise ValueError("num_predictors must be positive")
        if (
            not self.component_names
            or any(not name for name in self.component_names)
            or len(set(self.component_names)) != len(self.component_names)
        ):
            raise ValueError(
                "component_names must preserve unique non-empty order"
            )
        if self.num_predictors != len(self.component_names):
            raise ValueError(
                "num_predictors must equal the ordered component count"
            )
        budget_count = sum(
            value is not None
            for value in (
                self.auto,
                self.max_full_evals,
                self.max_metric_calls,
            )
        )
        if budget_count != 1:
            raise ValueError(
                "Exactly one of max_metric_calls, max_full_evals, auto "
                "must be set."
            )
        if self.max_full_evals is not None and self.max_full_evals < 0:
            raise ValueError("max_full_evals must be non-negative")
        if self.max_metric_calls is not None and self.max_metric_calls < 0:
            raise ValueError("max_metric_calls must be non-negative")
        if self.resolved_max_metric_calls < 0:
            raise ValueError("resolved_max_metric_calls must be non-negative")
        if self.resolved_max_metric_calls != self._expected_metric_calls():
            raise ValueError(
                "resolved_max_metric_calls conflicts with the budget mode"
            )
        if self.reflection_minibatch_size < 1:
            raise ValueError("reflection_minibatch_size must be positive")
        if self.max_merge_invocations < 0:
            raise ValueError("max_merge_invocations must be non-negative")
        if self.merge_val_overlap_floor < 1:
            raise ValueError("merge_val_overlap_floor must be positive")
        if self.num_threads is not None:
            raise ValueError(
                "num_threads is not yet supported by canonical GEPA"
            )
        if not math.isfinite(self.failure_score):
            raise ValueError("failure_score must be finite")
        if not math.isfinite(self.perfect_score):
            raise ValueError("perfect_score must be finite")
        if self.track_best_outputs and not self.track_stats:
            raise ValueError(
                "track_stats must be True if track_best_outputs is True."
            )
        fixed_values = {
            "algorithm_version": GEPA_ALGORITHM_VERSION,
            "dspy_reference_commit": GEPA_DSPY_REFERENCE_COMMIT,
            "gepa_distribution_version": GEPA_DISTRIBUTION_VERSION,
            "gepa_repository_commit": GEPA_REPOSITORY_COMMIT,
            "gepa_wheel_sha256": GEPA_WHEEL_SHA256,
            "gepa_sdist_sha256": GEPA_SDIST_SHA256,
            "gepa_source_manifest_hash": GEPA_SOURCE_MANIFEST_HASH,
            "merge_policy_identity_hash": GEPA_MERGE_POLICY_IDENTITY_HASH,
            "adapter_schema_version": GEPA_ADAPTER_SCHEMA_VERSION,
            "result_schema_version": GEPA_RESULT_SCHEMA_VERSION,
            "reflection_prompt_schema": GEPA_REFLECTION_PROMPT_SCHEMA,
            "reflection_prompt_schema_version": (
                GEPA_REFLECTION_PROMPT_SCHEMA_VERSION
            ),
            "reflection_parser_schema": (
                GEPA_REFLECTION_RESPONSE_PARSER_SCHEMA
            ),
            "reflection_parser_schema_version": (
                GEPA_REFLECTION_RESPONSE_PARSER_SCHEMA_VERSION
            ),
        }
        for field, expected in fixed_values.items():
            if getattr(self, field) != expected:
                raise ValueError(f"{field} is fixed")
        return self

    def _expected_metric_calls(self) -> int:
        if self.auto is not None:
            return gepa_auto_budget(
                num_predictors=self.num_predictors,
                num_candidates=GEPA_AUTO_CANDIDATES[self.auto],
                valset_size=len(self.valset_task_identities),
            )
        if self.max_full_evals is not None:
            # This intentionally copies DSPy's asymmetric valset=None branch.
            denominator = len(self.trainset_task_identities)
            if self.source_valset_task_identities is not None:
                denominator += len(self.valset_task_identities)
            return self.max_full_evals * denominator
        assert self.max_metric_calls is not None
        return self.max_metric_calls

    def identity_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload["algorithm"] = "gepa"
        payload["reflection_model_identity_hash"] = (
            self.reflection_model.identity_hash()
        )
        payload["metric_identity_hash"] = self.metric.identity_hash
        return payload

    def identity_hash(self) -> str:
        return compute_identity_hash(
            schema=GEPA_CONTROL_SCHEMA,
            schema_version=GEPA_CONTROL_SCHEMA_VERSION,
            payload=self.identity_payload(),
        )

    def require_identity_hash(self, persisted_hash: str) -> None:
        require_full_hash(persisted_hash, field="optimizer_config_hash")
        if persisted_hash != self.identity_hash():
            raise ValueError(
                "optimizer_config_hash conflicts with resolved GEPA control"
            )

    def upstream_kwargs(self) -> dict[str, Any]:
        """Project only frozen public options into ``gepa.optimize``."""

        return {
            "candidate_selection_strategy": (
                self.candidate_selection_strategy
            ),
            "frontier_type": self.frontier_type,
            "skip_perfect_score": self.skip_perfect_score,
            "batch_sampler": self.batch_sampler,
            "reflection_minibatch_size": self.reflection_minibatch_size,
            "perfect_score": self.perfect_score,
            "module_selector": self.component_selector,
            "use_merge": self.use_merge,
            "max_merge_invocations": self.max_merge_invocations,
            "merge_val_overlap_floor": self.merge_val_overlap_floor,
            "max_metric_calls": self.resolved_max_metric_calls,
            "run_dir": None,
            "use_wandb": False,
            "use_mlflow": False,
            "track_best_outputs": self.track_best_outputs,
            "display_progress_bar": False,
            "use_cloudpickle": self.use_cloudpickle,
            "cache_evaluation": self.cache_evaluation,
            "seed": self.seed,
            "raise_on_exception": True,
            "val_evaluation_policy": self.val_evaluation_policy,
        }


def configure_gepa(
    *,
    reflection_model: ProposerConfig,
    metric: EvalConfigRef,
    reward_policy_hash: str,
    evaluation_execution_policy_hash: str,
    proposal_execution_policy_hash: str,
    proposal_prompt_adapter_identity_hash: str,
    proposal_durability_policy_identity_hash: str,
    task_model_identity_hash: str,
    prompt_format_identity_hash: str,
    prompt_binding_identity_hash: str,
    trainset_task_identities: tuple[str, ...],
    valset_task_identities: tuple[str, ...] | None,
    component_names: tuple[str, ...],
    num_predictors: int,
    auto: GepaAutoMode | None = None,
    max_full_evals: int | None = None,
    max_metric_calls: int | None = None,
    reflection_minibatch_size: int = 3,
    candidate_selection_strategy: GepaCandidateSelection = "pareto",
    skip_perfect_score: bool = True,
    add_format_failure_as_feedback: bool = False,
    component_selector: GepaComponentSelector = "round_robin",
    use_merge: bool = True,
    max_merge_invocations: int = 5,
    merge_val_overlap_floor: int = 5,
    num_threads: int | None = None,
    failure_score: float = 0.0,
    perfect_score: float = 1.0,
    track_stats: bool = False,
    track_best_outputs: bool = False,
    warn_on_score_mismatch: bool = True,
    seed: int = 0,
    teacher: object | None = None,
) -> GepaControl:
    """Resolve DSPy-compatible arguments before any observable effect."""

    if teacher is not None:
        raise ValueError("Teacher is not supported in GEPA.")
    resolved_valset = (
        trainset_task_identities
        if valset_task_identities is None
        else valset_task_identities
    )
    budget_count = sum(
        value is not None for value in (auto, max_full_evals, max_metric_calls)
    )
    if budget_count != 1:
        raise ValueError(
            "Exactly one of max_metric_calls, max_full_evals, auto "
            "must be set."
        )
    if auto is not None:
        if auto not in GEPA_AUTO_CANDIDATES:
            raise ValueError(
                "auto must be one of 'light', 'medium', or 'heavy'"
            )
        resolved_metric_calls = gepa_auto_budget(
            num_predictors=num_predictors,
            num_candidates=GEPA_AUTO_CANDIDATES[auto],
            valset_size=len(resolved_valset),
        )
    elif max_full_evals is not None:
        denominator = len(trainset_task_identities)
        if valset_task_identities is not None:
            denominator += len(resolved_valset)
        resolved_metric_calls = max_full_evals * denominator
    else:
        assert max_metric_calls is not None
        resolved_metric_calls = max_metric_calls
    return GepaControl(
        reflection_model=reflection_model,
        metric=metric,
        reward_policy_hash=reward_policy_hash,
        evaluation_execution_policy_hash=evaluation_execution_policy_hash,
        proposal_execution_policy_hash=proposal_execution_policy_hash,
        proposal_prompt_adapter_identity_hash=(
            proposal_prompt_adapter_identity_hash
        ),
        proposal_durability_policy_identity_hash=(
            proposal_durability_policy_identity_hash
        ),
        task_model_identity_hash=task_model_identity_hash,
        prompt_format_identity_hash=prompt_format_identity_hash,
        prompt_binding_identity_hash=prompt_binding_identity_hash,
        source_trainset_task_identities=trainset_task_identities,
        source_valset_task_identities=valset_task_identities,
        trainset_task_identities=trainset_task_identities,
        valset_task_identities=resolved_valset,
        component_names=component_names,
        num_predictors=num_predictors,
        auto=auto,
        max_full_evals=max_full_evals,
        max_metric_calls=max_metric_calls,
        resolved_max_metric_calls=resolved_metric_calls,
        reflection_minibatch_size=reflection_minibatch_size,
        candidate_selection_strategy=candidate_selection_strategy,
        skip_perfect_score=skip_perfect_score,
        add_format_failure_as_feedback=add_format_failure_as_feedback,
        component_selector=component_selector,
        use_merge=use_merge,
        max_merge_invocations=max_merge_invocations,
        merge_val_overlap_floor=merge_val_overlap_floor,
        num_threads=num_threads,
        failure_score=failure_score,
        perfect_score=perfect_score,
        track_stats=track_stats,
        track_best_outputs=track_best_outputs,
        warn_on_score_mismatch=warn_on_score_mismatch,
        seed=seed,
    )


__all__ = [
    "GEPA_ADAPTER_SCHEMA_VERSION",
    "GEPA_ALGORITHM_VERSION",
    "GEPA_AUTO_CANDIDATES",
    "GEPA_CONTROL_SCHEMA",
    "GEPA_CONTROL_SCHEMA_VERSION",
    "GEPA_DSPY_REFERENCE_COMMIT",
    "GEPA_MERGE_POLICY_IDENTITY_HASH",
    "GEPA_RESULT_SCHEMA_VERSION",
    "GepaAutoMode",
    "GepaCandidateSelection",
    "GepaComponentSelector",
    "GepaControl",
    "configure_gepa",
    "gepa_auto_budget",
]
