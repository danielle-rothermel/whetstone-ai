from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    StrictInt,
    StrictStr,
    model_validator,
)

from whetstone.core.identity import reject_non_json, require_full_hash
from whetstone.optimization.gepa.control import (
    GEPA_RESULT_SCHEMA_VERSION,
    GepaControl,
)
from whetstone.optimization.gepa.source import (
    GEPA_SOURCE_MANIFEST_HASH,
    verify_installed_gepa_source,
)
from whetstone.optimization.gepa.upstream_adapter import (
    GEPA_UPSTREAM_ADAPTER_IDENTITY_HASH,
)
from whetstone.optimization.proposal.proposer import ProposerConfig


class _Logger(Protocol):
    def log(self, message: str) -> None: ...


class _EffectContext(Protocol):
    @property
    def control_identity_hash(self) -> str: ...

    @property
    def source_manifest_identity_hash(self) -> str: ...

    @property
    def adapter_identity_hash(self) -> str: ...


class _EvaluationAuthorityBinding(Protocol):
    @property
    def evaluation_config_hash(self) -> str: ...

    @property
    def reward_policy_identity_hash(self) -> str: ...

    @property
    def provider_route_identity_hash(self) -> str: ...

    @property
    def execution_policy_identity_hash(self) -> str: ...

    @property
    def failure_score(self) -> float: ...

    @property
    def add_format_failure_as_feedback(self) -> bool: ...

    @property
    def warn_on_score_mismatch(self) -> bool: ...

    @property
    def selection_seed(self) -> int: ...


class _ProposalAuthorityBinding(Protocol):
    @property
    def prompt_binding_identity_hash(self) -> str: ...

    @property
    def proposer_config(self) -> ProposerConfig: ...

    @property
    def execution_policy_identity_hash(self) -> str: ...

    @property
    def prompt_adapter_identity_hash(self) -> str: ...

    @property
    def durability_policy_identity_hash(self) -> str: ...


class GepaEngineAdapter(Protocol):
    @property
    def effect_context(self) -> _EffectContext: ...

    @property
    def evaluation_authority(self) -> _EvaluationAuthorityBinding: ...

    @property
    def proposal_authority(self) -> _ProposalAuthorityBinding: ...

    @property
    def prompt_format_identity_hash(self) -> str: ...

    def reset_effect_ordinal(self) -> None: ...


class _QuietLogger:
    """Suppress operational stdout without participating in decisions."""

    def log(self, message: str) -> None:
        del message


class GepaDetailedResult(BaseModel):
    """Lossless JSON-facing projection of frozen ``GEPAResult`` schema v2."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidates: tuple[dict[StrictStr, StrictStr], ...]
    parents: tuple[tuple[StrictInt | None, ...], ...]
    val_aggregate_scores: tuple[float, ...]
    val_subscores: tuple[dict[StrictStr, float], ...]
    per_val_instance_best_candidates: dict[StrictStr, tuple[StrictInt, ...]]
    discovery_eval_counts: tuple[StrictInt, ...]
    best_outputs_valset: (
        dict[StrictStr, tuple[tuple[StrictInt, Any], ...]] | None
    ) = None
    val_aggregate_subscores: tuple[dict[StrictStr, float], ...] | None = None
    per_objective_best_candidates: (
        dict[StrictStr, tuple[StrictInt, ...]] | None
    ) = None
    objective_pareto_front: dict[StrictStr, float] | None = None
    total_metric_calls: StrictInt | None = None
    num_full_val_evals: StrictInt | None = None
    seed: StrictInt
    best_idx: StrictInt
    control_identity_hash: StrictStr
    source_manifest_hash: StrictStr = GEPA_SOURCE_MANIFEST_HASH
    result_schema_version: StrictStr = GEPA_RESULT_SCHEMA_VERSION
    upstream_validation_schema_version: StrictInt = 2

    @model_validator(mode="after")
    def _validate(self) -> GepaDetailedResult:
        candidate_count = len(self.candidates)
        for field, values in (
            ("parents", self.parents),
            ("val_aggregate_scores", self.val_aggregate_scores),
            ("val_subscores", self.val_subscores),
            ("discovery_eval_counts", self.discovery_eval_counts),
        ):
            if len(values) != candidate_count:
                raise ValueError(
                    f"{field} must align with upstream candidates"
                )
        if not 0 <= self.best_idx < candidate_count:
            raise ValueError("best_idx must identify an upstream candidate")
        require_full_hash(
            self.control_identity_hash, field="control_identity_hash"
        )
        require_full_hash(
            self.source_manifest_hash, field="source_manifest_hash"
        )
        if self.best_outputs_valset is not None:
            reject_non_json(
                self.best_outputs_valset,
                field="best_outputs_valset",
            )
        if self.source_manifest_hash != GEPA_SOURCE_MANIFEST_HASH:
            raise ValueError("GEPA result source-manifest identity drift")
        if self.result_schema_version != GEPA_RESULT_SCHEMA_VERSION:
            raise ValueError("GEPA result schema identity drift")
        return self


def _val_identity(control: GepaControl, upstream_id: object) -> str:
    if type(upstream_id) is not int:
        raise ValueError(
            "canonical GEPA requires list-backed integer validation ids"
        )
    index = upstream_id
    if index < 0 or index >= len(control.valset_task_hashes):
        raise ValueError(f"unknown upstream validation id {index}")
    return control.valset_task_hashes[index]


def _project_result(
    result: Any,
    *,
    control: GepaControl,
) -> GepaDetailedResult:
    val_subscores = tuple(
        {
            _val_identity(control, val_id): score
            for val_id, score in scores.items()
        }
        for scores in result.val_subscores
    )
    per_val_best = {
        _val_identity(control, val_id): tuple(sorted(programs))
        for val_id, programs in (
            result.per_val_instance_best_candidates.items()
        )
    }
    best_outputs = None
    if result.best_outputs_valset is not None:
        best_outputs = {
            _val_identity(control, val_id): tuple(outputs)
            for val_id, outputs in result.best_outputs_valset.items()
        }
    objective_best = None
    if result.per_objective_best_candidates is not None:
        objective_best = {
            objective: tuple(sorted(programs))
            for objective, programs in (
                result.per_objective_best_candidates.items()
            )
        }
    aggregate_subscores = None
    if result.val_aggregate_subscores is not None:
        aggregate_subscores = tuple(
            dict(scores) for scores in result.val_aggregate_subscores
        )
    return GepaDetailedResult(
        candidates=tuple(dict(candidate) for candidate in result.candidates),
        parents=tuple(tuple(parents) for parents in result.parents),
        val_aggregate_scores=tuple(result.val_aggregate_scores),
        val_subscores=val_subscores,
        per_val_instance_best_candidates=per_val_best,
        discovery_eval_counts=tuple(result.discovery_eval_counts),
        best_outputs_valset=best_outputs,
        val_aggregate_subscores=aggregate_subscores,
        per_objective_best_candidates=objective_best,
        objective_pareto_front=(
            dict(result.objective_pareto_front)
            if result.objective_pareto_front is not None
            else None
        ),
        total_metric_calls=result.total_metric_calls,
        num_full_val_evals=result.num_full_val_evals,
        seed=control.seed,
        best_idx=result.best_idx,
        control_identity_hash=control.identity_hash(),
    )


def _validate_adapter_authorities(
    *,
    control: GepaControl,
    adapter: GepaEngineAdapter,
) -> None:
    context = adapter.effect_context
    if context.control_identity_hash != control.identity_hash():
        raise ValueError(
            "GEPA adapter effect context conflicts with GepaControl"
        )
    if (
        context.source_manifest_identity_hash
        != control.gepa_source_manifest_hash
    ):
        raise ValueError(
            "GEPA adapter effect context conflicts with frozen source"
        )
    if context.adapter_identity_hash != GEPA_UPSTREAM_ADAPTER_IDENTITY_HASH:
        raise ValueError(
            "GEPA adapter effect context conflicts with canonical adapter"
        )

    evaluation = adapter.evaluation_authority
    expected_evaluation = {
        "evaluation_config_hash": control.metric.config_hash,
        "reward_policy_identity_hash": control.reward_policy_hash,
        "provider_route_identity_hash": control.task_model_identity_hash,
        "execution_policy_identity_hash": (
            control.evaluation_execution_policy_hash
        ),
        "failure_score": control.failure_score,
        "add_format_failure_as_feedback": (
            control.add_format_failure_as_feedback
        ),
        "warn_on_score_mismatch": control.warn_on_score_mismatch,
        "selection_seed": control.seed,
    }
    for field, expected in expected_evaluation.items():
        if getattr(evaluation, field) != expected:
            raise ValueError(
                f"GEPA adapter evaluation authority {field} "
                "conflicts with GepaControl"
            )

    proposal = adapter.proposal_authority
    if proposal.proposer_config != control.reflection_model:
        raise ValueError(
            "GEPA adapter proposal authority proposer_config conflicts "
            "with GepaControl"
        )
    if (
        proposal.prompt_binding_identity_hash
        != control.prompt_binding_identity_hash
    ):
        raise ValueError(
            "GEPA adapter proposal authority prompt binding conflicts "
            "with GepaControl"
        )
    expected_proposal = {
        "execution_policy_identity_hash": (
            control.proposal_execution_policy_hash
        ),
        "prompt_adapter_identity_hash": (
            control.proposal_prompt_adapter_identity_hash
        ),
        "durability_policy_identity_hash": (
            control.proposal_durability_policy_identity_hash
        ),
    }
    for field, expected in expected_proposal.items():
        if getattr(proposal, field) != expected:
            raise ValueError(
                f"GEPA adapter proposal authority {field} "
                "conflicts with GepaControl"
            )
    if (
        adapter.prompt_format_identity_hash
        != control.prompt_format_identity_hash
    ):
        raise ValueError(
            "GEPA adapter prompt format conflicts with GepaControl"
        )


def run_gepa_engine[DataInst](
    *,
    control: GepaControl,
    seed_candidate: Mapping[str, str],
    trainset: Sequence[DataInst],
    valset: Sequence[DataInst] | None,
    adapter: GepaEngineAdapter,
    logger: _Logger | None = None,
) -> GepaDetailedResult:
    """Run the upstream engine without replacing any algorithmic decision."""

    verify_installed_gepa_source()
    from gepa import optimize

    _validate_adapter_authorities(control=control, adapter=adapter)
    adapter.reset_effect_ordinal()

    ordered_seed = dict(seed_candidate)
    if not ordered_seed:
        raise ValueError("seed_candidate must contain a component")
    if tuple(ordered_seed) != control.component_names:
        raise ValueError(
            "seed_candidate component order conflicts with GepaControl"
        )
    observed_train_ids = tuple(cast(Any, item).data_id for item in trainset)
    if observed_train_ids != control.trainset_task_hashes:
        raise ValueError("trainset order/identity conflicts with GepaControl")
    if valset is None:
        if control.source_valset_task_hashes is not None:
            raise ValueError("GepaControl binds an explicit valset")
    elif control.source_valset_task_hashes is None:
        raise ValueError("GepaControl binds valset omission to the trainset")
    else:
        observed_val_ids = tuple(cast(Any, item).data_id for item in valset)
        if observed_val_ids != control.valset_task_hashes:
            raise ValueError(
                "valset order/identity conflicts with GepaControl"
            )

    result = optimize(
        seed_candidate=ordered_seed,
        trainset=list(trainset),
        valset=None if valset is None else list(valset),
        adapter=cast(Any, adapter),
        reflection_lm=None,
        custom_candidate_proposer=None,
        logger=_QuietLogger() if logger is None else logger,
        callbacks=None,
        **control.upstream_kwargs(),
    )
    return _project_result(result, control=control)


__all__ = [
    "GepaDetailedResult",
    "GepaEngineAdapter",
    "run_gepa_engine",
]
