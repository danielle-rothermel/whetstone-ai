"""Resolved, identity-bearing MIPROv2 construction and run controls.

This module is deliberately effect free. It reproduces the configuration,
dataset, and auto/manual resolution order of the frozen DSPy implementation
before bootstrap, proposal, or evaluation effects are allowed to run.
"""

from __future__ import annotations

import math
import random
from copy import deepcopy
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    model_validator,
)

from whetstone.lm.boundary import PlainPromptAdapter
from whetstone.optimization.identity import (
    IdentityRef,
    ImmutableJsonObject,
    compute_identity_hash,
    require_full_hash,
    typed_ref_for_record,
)
from whetstone.optimization.miprov2_bootstrap import (
    MIPROV2_TRACE_SELECTION_PROJECTION_VERSION,
)
from whetstone.optimization.miprov2_proposal import (
    COMPONENT_DESCRIPTION_SCHEMA_TAG,
    DATASET_FINAL_SCHEMA_TAG,
    DATASET_FOLLOWUP_SCHEMA_TAG,
    DATASET_INITIAL_SCHEMA_TAG,
    INSTRUCTION_PROPOSAL_SCHEMA_TAG,
    PROGRAM_DESCRIPTION_SCHEMA_TAG,
)
from whetstone.optimization.mutation import MUTATION_FIELD
from whetstone.optimization.proposer import (
    ProposerConfig,
    prompt_adapter_identity_hash,
)
from whetstone.optimization.reward import RewardPolicy
from whetstone.optimization.schema import (
    CandidateRef,
    EvalConfigRef,
    EvaluationBinding,
    TemplateRenderContract,
)

MIPROV2_ALGORITHM_VERSION = "dspy_miprov2/v2"
MIPROV2_REFERENCE_COMMIT = "6f68dcdb3ef46d70bf0c12596699ebc44e82d6b0"
MIPROV2_OPTUNA_VERSION = "4.8.0"
MIPROV2_CONTROL_SCHEMA = "whetstone.miprov2_optimizer_config"
MIPROV2_CONTROL_SCHEMA_VERSION = 4
MIPROV2_COMPONENT_SPEC_SCHEMA = "whetstone.miprov2_component_spec"
MIPROV2_COMPONENT_SPEC_SCHEMA_VERSION = 1
MIPROV2_PROGRAM_LAYOUT_SCHEMA = "whetstone.miprov2_program_layout"
MIPROV2_PROGRAM_LAYOUT_SCHEMA_VERSION = 1
MIPROV2_PROMPT_FORMAT_ADAPTER_VERSION = "whetstone_prompt_components/v1"
MIPROV2_CANDIDATE_RENDERER_VERSION = "whetstone_native_prompt_components/v1"
MIPROV2_PROPOSER_OUTPUT_PARSER_SCHEMA = (
    "whetstone.miprov2.proposer_output_parser"
)
MIPROV2_PROPOSER_OUTPUT_PARSER_SCHEMA_VERSION = 1
MIPROV2_STATE_SCHEMA = "whetstone.miprov2_runtime"
MIPROV2_STATE_SCHEMA_VERSION = 1
MIPROV2_RESULT_SCHEMA = "whetstone.miprov2_result"
MIPROV2_RESULT_SCHEMA_VERSION = 1
MIPROV2_PHASE_SCHEMA_MANIFEST: tuple[tuple[str, int], ...] = (
    ("whetstone.miprov2_grounded_proposal", 1),
    (DATASET_INITIAL_SCHEMA_TAG, 1),
    (DATASET_FOLLOWUP_SCHEMA_TAG, 1),
    (DATASET_FINAL_SCHEMA_TAG, 1),
    (PROGRAM_DESCRIPTION_SCHEMA_TAG, 1),
    (COMPONENT_DESCRIPTION_SCHEMA_TAG, 1),
    (INSTRUCTION_PROPOSAL_SCHEMA_TAG, 1),
    ("whetstone.miprov2_bootstrap_plan", 1),
    ("whetstone.miprov2_bootstrap_attempt", 1),
    (MIPROV2_TRACE_SELECTION_PROJECTION_VERSION, 1),
    ("whetstone.miprov2_component_demo_set", 1),
    ("whetstone.miprov2_candidate_rendering", 1),
    ("whetstone.miprov2_candidate_program", 1),
    ("whetstone.miprov2_candidate_assembly", 2),
    ("whetstone.miprov2_evaluation_execution_policy", 1),
    ("whetstone.miprov2_eval_binding_request", 1),
    ("whetstone.miprov2_eval_binding", 1),
    ("whetstone.miprov2_intent_context", 2),
    ("whetstone.miprov2_study_transcript", 3),
)

Miprov2AutoMode = Literal["light", "medium", "heavy"]

_AUTO_RUN_SETTINGS: dict[Miprov2AutoMode, tuple[int, int]] = {
    "light": (6, 100),
    "medium": (12, 300),
    "heavy": (18, 1000),
}
_MIN_MINIBATCH_SIZE = 50


class Miprov2ComponentSpec(BaseModel):
    """One ordered, identity-bound Whetstone prompt component."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    component_id: StrictStr
    prompt_format_identity_hash: StrictStr

    @model_validator(mode="after")
    def _validate_component(self) -> Miprov2ComponentSpec:
        if not self.component_id:
            raise ValueError(
                "Whetstone safety: component_id must be non-empty"
            )
        require_full_hash(
            self.prompt_format_identity_hash,
            field="prompt_format_identity_hash",
        )
        return self

    def identity_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def identity_hash(self) -> str:
        return compute_identity_hash(
            schema=MIPROV2_COMPONENT_SPEC_SCHEMA,
            schema_version=MIPROV2_COMPONENT_SPEC_SCHEMA_VERSION,
            payload=self.identity_payload(),
        )


class Miprov2ProgramLayout(BaseModel):
    """Authoritative ordered prompt components optimized as one program."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    layout_id: StrictStr
    component_specs: tuple[Miprov2ComponentSpec, ...]

    @model_validator(mode="after")
    def _validate_layout(self) -> Miprov2ProgramLayout:
        if not self.layout_id:
            raise ValueError("Whetstone safety: layout_id must be non-empty")
        if len(self.component_specs) != 1:
            raise ValueError(
                "MIPROv2 requires exactly one optimizable component"
            )
        component_ids = tuple(
            spec.component_id for spec in self.component_specs
        )
        if len(set(component_ids)) != len(component_ids):
            raise ValueError("Whetstone safety: component_ids must be unique")
        if component_ids[0] not in {"generate", "encode"}:
            raise ValueError(
                "MIPROv2 optimizes generate (Internal/D1) or encode (ED1)"
            )
        return self

    def identity_payload(self) -> dict[str, Any]:
        return {
            "layout_id": self.layout_id,
            "component_specs": [
                {
                    "identity_hash": spec.identity_hash(),
                    "config": spec.identity_payload(),
                }
                for spec in self.component_specs
            ],
        }

    def identity_hash(self) -> str:
        return compute_identity_hash(
            schema=MIPROV2_PROGRAM_LAYOUT_SCHEMA,
            schema_version=MIPROV2_PROGRAM_LAYOUT_SCHEMA_VERSION,
            payload=self.identity_payload(),
        )


class Miprov2InjectedDefaults(BaseModel):
    """Explicit Whetstone bindings replacing DSPy's ambient settings.

    Hash and prompt-adapter checks are Whetstone safety validations, not
    validations performed by DSPy.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    prompt_model: ProposerConfig
    # Source authorities, not fixed per-effect configs. Bootstrap rollouts and
    # each randomly sampled validation minibatch derive an exact ordered-subset
    # Eval Config from the corresponding source before issuing an Intent.
    bootstrap_eval_source: EvalConfigRef
    validation_eval_source: EvalConfigRef
    reward_policy: RewardPolicy
    evaluation_binding: EvaluationBinding
    provider_execution_policy_hash: StrictStr
    task_model_identity_hash: StrictStr
    prompt_adapter: PlainPromptAdapter
    template_render_contract: TemplateRenderContract
    max_errors: StrictInt
    validation_eval_source_is_metric_authority: StrictBool = False

    @model_validator(mode="after")
    def _validate_whetstone_bindings(self) -> Miprov2InjectedDefaults:
        require_full_hash(
            self.provider_execution_policy_hash,
            field="provider_execution_policy_hash",
        )
        require_full_hash(
            self.task_model_identity_hash,
            field="task_model_identity_hash",
        )
        _require_positive_int(self.max_errors, field="max_errors")
        return self


class Miprov2Control(BaseModel):
    """Fully resolved MIPROv2 control persisted before observable effects."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    base_candidate: CandidateRef
    teacher_candidate: CandidateRef
    teacher_compiled: StrictBool
    program_layout: Miprov2ProgramLayout
    source_trainset_task_identities: tuple[StrictStr, ...]
    source_valset_task_identities: tuple[StrictStr, ...] | None
    trainset_task_identities: tuple[StrictStr, ...]
    valset_task_identities: tuple[StrictStr, ...]
    auto_validation_sample_indices: tuple[StrictInt, ...] | None

    prompt_model: ProposerConfig
    bootstrap_eval_source: EvalConfigRef
    validation_eval_source: EvalConfigRef
    reward_policy: RewardPolicy
    evaluation_binding: EvaluationBinding
    provider_execution_policy_hash: StrictStr
    task_model_identity_hash: StrictStr
    prompt_adapter_identity_hash: StrictStr
    template_render_contract: TemplateRenderContract
    metric_authority: Literal["explicit", "injected_default"]

    teacher_settings: ImmutableJsonObject = Field(
        default_factory=lambda: ImmutableJsonObject({})
    )
    max_bootstrapped_demos: StrictInt
    max_labeled_demos: StrictInt
    auto: Miprov2AutoMode | None
    num_candidates: StrictInt | None
    num_instruct_candidates: StrictInt
    num_fewshot_candidates: StrictInt
    num_trials: StrictInt
    num_threads: StrictInt | None
    max_errors: StrictInt
    seed: StrictInt
    init_temperature: float
    verbose: StrictBool
    track_stats: StrictBool
    log_dir: StrictStr | None
    metric_threshold: float | None
    minibatch: StrictBool
    minibatch_size: StrictInt
    minibatch_full_eval_steps: StrictInt
    program_aware_proposer: StrictBool
    data_aware_proposer: StrictBool
    view_data_batch_size: StrictInt
    tip_aware_proposer: StrictBool
    fewshot_aware_proposer: StrictBool
    provide_traceback: StrictBool | None
    zeroshot_opt: StrictBool

    algorithm_version: StrictStr = MIPROV2_ALGORITHM_VERSION
    reference_commit: StrictStr = MIPROV2_REFERENCE_COMMIT
    optuna_version: StrictStr = MIPROV2_OPTUNA_VERSION
    prompt_format_adapter_version: StrictStr = (
        MIPROV2_PROMPT_FORMAT_ADAPTER_VERSION
    )
    candidate_renderer_version: StrictStr = MIPROV2_CANDIDATE_RENDERER_VERSION
    proposer_output_parser_schema: StrictStr = (
        MIPROV2_PROPOSER_OUTPUT_PARSER_SCHEMA
    )
    proposer_output_parser_schema_version: StrictInt = (
        MIPROV2_PROPOSER_OUTPUT_PARSER_SCHEMA_VERSION
    )
    phase_schema_manifest: tuple[tuple[StrictStr, StrictInt], ...] = (
        MIPROV2_PHASE_SCHEMA_MANIFEST
    )
    state_schema: StrictStr = MIPROV2_STATE_SCHEMA
    state_schema_version: StrictInt = MIPROV2_STATE_SCHEMA_VERSION
    result_schema: StrictStr = MIPROV2_RESULT_SCHEMA
    result_schema_version: StrictInt = MIPROV2_RESULT_SCHEMA_VERSION

    @model_validator(mode="after")
    def _validate_resolved_control(self) -> Miprov2Control:
        # These are Whetstone safety validations. DSPy operates on process
        # objects and therefore has no corresponding identity checks.
        for spec in self.component_specs:
            if (
                spec.prompt_format_identity_hash
                != self.prompt_adapter_identity_hash
            ):
                raise ValueError(
                    "Whetstone safety: component prompt format conflicts with "
                    "the bound prompt adapter"
                )
        _validate_candidate_components(
            self.base_candidate,
            self.component_specs,
            role="base candidate",
        )
        _validate_candidate_components(
            self.teacher_candidate,
            self.component_specs,
            role="teacher candidate",
        )
        self.template_render_contract.validate_template(
            self.base_candidate.record.payload.get(MUTATION_FIELD)
        )
        self.template_render_contract.validate_template(
            self.teacher_candidate.record.payload.get(MUTATION_FIELD)
        )
        for field, identities in (
            (
                "source_trainset_task_identities",
                self.source_trainset_task_identities,
            ),
            ("trainset_task_identities", self.trainset_task_identities),
            ("valset_task_identities", self.valset_task_identities),
        ):
            for identity in identities:
                require_full_hash(identity, field=field)
        if self.source_valset_task_identities is not None and any(
            not identity for identity in self.source_valset_task_identities
        ):
            raise ValueError(
                "Whetstone safety: source_valset_task_identities contains "
                "an empty identity"
            )
        if self.source_valset_task_identities is not None:
            for identity in self.source_valset_task_identities:
                require_full_hash(
                    identity,
                    field="source_valset_task_identities",
                )
        _validate_teacher_settings(self.teacher_settings.to_json())
        if self.evaluation_binding.eval_config != self.validation_eval_source:
            raise ValueError(
                "MIPROv2 Evaluation Binding must match validation source"
            )
        require_full_hash(
            self.provider_execution_policy_hash,
            field="provider_execution_policy_hash",
        )
        require_full_hash(
            self.task_model_identity_hash,
            field="task_model_identity_hash",
        )
        require_full_hash(
            self.prompt_adapter_identity_hash,
            field="prompt_adapter_identity_hash",
        )
        if self.prompt_model.temperature != self.init_temperature:
            raise ValueError(
                "Whetstone safety: prompt_model temperature conflicts with "
                "init_temperature"
            )
        _validate_resolved_numeric_controls(self)
        if self.algorithm_version != MIPROV2_ALGORITHM_VERSION:
            raise ValueError("MIPROv2 algorithm_version is fixed")
        if self.reference_commit != MIPROV2_REFERENCE_COMMIT:
            raise ValueError("MIPROv2 reference_commit is fixed")
        if self.optuna_version != MIPROV2_OPTUNA_VERSION:
            raise ValueError("MIPROv2 optuna_version is fixed")
        if (
            self.prompt_format_adapter_version
            != MIPROV2_PROMPT_FORMAT_ADAPTER_VERSION
        ):
            raise ValueError("MIPROv2 prompt_format_adapter_version is fixed")
        if (
            self.candidate_renderer_version
            != MIPROV2_CANDIDATE_RENDERER_VERSION
        ):
            raise ValueError("MIPROv2 candidate renderer is fixed")
        if (
            self.proposer_output_parser_schema
            != MIPROV2_PROPOSER_OUTPUT_PARSER_SCHEMA
            or self.proposer_output_parser_schema_version
            != MIPROV2_PROPOSER_OUTPUT_PARSER_SCHEMA_VERSION
        ):
            raise ValueError(
                "MIPROv2 proposer output parser identity is fixed"
            )
        if self.phase_schema_manifest != MIPROV2_PHASE_SCHEMA_MANIFEST:
            raise ValueError("MIPROv2 phase schema manifest is fixed")
        if (
            self.state_schema != MIPROV2_STATE_SCHEMA
            or self.state_schema_version != MIPROV2_STATE_SCHEMA_VERSION
        ):
            raise ValueError("MIPROv2 state schema identity is fixed")
        if (
            self.result_schema != MIPROV2_RESULT_SCHEMA
            or self.result_schema_version != MIPROV2_RESULT_SCHEMA_VERSION
        ):
            raise ValueError("MIPROv2 result schema identity is fixed")
        _validate_resolved_derivations(self)
        return self

    @property
    def component_ids(self) -> tuple[str, ...]:
        """Return ordered component ids derived from bound component specs."""

        return tuple(spec.component_id for spec in self.component_specs)

    @property
    def component_specs(self) -> tuple[Miprov2ComponentSpec, ...]:
        """Return the exact ordered component authority."""

        return self.program_layout.component_specs

    def identity_payload(self) -> dict[str, Any]:
        """Return every resolved decision and Whetstone effect authority."""

        return {
            "algorithm": "miprov2",
            "algorithm_version": self.algorithm_version,
            "reference_commit": self.reference_commit,
            "optuna_version": self.optuna_version,
            "prompt_format_adapter_version": (
                self.prompt_format_adapter_version
            ),
            "candidate_renderer_version": self.candidate_renderer_version,
            "proposer_output_parser": {
                "schema": self.proposer_output_parser_schema,
                "schema_version": self.proposer_output_parser_schema_version,
            },
            "phase_schema_manifest": [
                {"schema": schema, "schema_version": version}
                for schema, version in self.phase_schema_manifest
            ],
            "state_schema": {
                "schema": self.state_schema,
                "schema_version": self.state_schema_version,
            },
            "result_schema": {
                "schema": self.result_schema,
                "schema_version": self.result_schema_version,
            },
            "base_candidate": self.base_candidate.model_dump(mode="json"),
            "teacher_candidate": self.teacher_candidate.model_dump(
                mode="json"
            ),
            "teacher_compiled": self.teacher_compiled,
            "program_layout": {
                "identity_hash": self.program_layout.identity_hash(),
                "config": self.program_layout.identity_payload(),
            },
            "source_trainset_task_identities": list(
                self.source_trainset_task_identities
            ),
            "source_valset_task_identities": (
                list(self.source_valset_task_identities)
                if self.source_valset_task_identities is not None
                else None
            ),
            "trainset_task_identities": list(self.trainset_task_identities),
            "valset_task_identities": list(self.valset_task_identities),
            "auto_validation_sample_indices": (
                list(self.auto_validation_sample_indices)
                if self.auto_validation_sample_indices is not None
                else None
            ),
            "prompt_model": {
                "identity_hash": self.prompt_model.identity_hash(),
                "config": self.prompt_model.identity_payload(),
            },
            "bootstrap_eval_source": self.bootstrap_eval_source.model_dump(
                mode="json"
            ),
            "validation_eval_source": self.validation_eval_source.model_dump(
                mode="json"
            ),
            "reward_policy": self.reward_policy.model_dump(mode="json"),
            "evaluation_binding": self.evaluation_binding.model_dump(
                mode="json"
            ),
            "provider_execution_policy_hash": (
                self.provider_execution_policy_hash
            ),
            "task_model_identity_hash": self.task_model_identity_hash,
            "prompt_adapter_identity_hash": (
                self.prompt_adapter_identity_hash
            ),
            "template_render_contract": (
                self.template_render_contract.model_dump(mode="json")
            ),
            "metric_authority": self.metric_authority,
            "teacher_settings": self.teacher_settings.to_json(),
            "max_bootstrapped_demos": self.max_bootstrapped_demos,
            "max_labeled_demos": self.max_labeled_demos,
            "auto": self.auto,
            "num_candidates": self.num_candidates,
            "num_instruct_candidates": self.num_instruct_candidates,
            "num_fewshot_candidates": self.num_fewshot_candidates,
            "num_trials": self.num_trials,
            "num_threads": self.num_threads,
            "max_errors": self.max_errors,
            "seed": self.seed,
            "init_temperature": self.init_temperature,
            "verbose": self.verbose,
            "track_stats": self.track_stats,
            "log_dir": self.log_dir,
            "metric_threshold": self.metric_threshold,
            "minibatch": self.minibatch,
            "minibatch_size": self.minibatch_size,
            "minibatch_full_eval_steps": (self.minibatch_full_eval_steps),
            "program_aware_proposer": self.program_aware_proposer,
            "data_aware_proposer": self.data_aware_proposer,
            "view_data_batch_size": self.view_data_batch_size,
            "tip_aware_proposer": self.tip_aware_proposer,
            "fewshot_aware_proposer": self.fewshot_aware_proposer,
            "provide_traceback": self.provide_traceback,
            "zeroshot_opt": self.zeroshot_opt,
        }

    def identity_hash(self) -> str:
        return compute_identity_hash(
            schema=MIPROV2_CONTROL_SCHEMA,
            schema_version=MIPROV2_CONTROL_SCHEMA_VERSION,
            payload=self.identity_payload(),
        )

    @property
    def reward_policy_hash(self) -> str:
        return self.reward_policy.identity_hash()

    def reference(self) -> IdentityRef:
        """Return the exact persisted-record and identity binding."""

        return IdentityRef(
            record_ref=typed_ref_for_record(
                MIPROV2_CONTROL_SCHEMA,
                self.model_dump(mode="json"),
            ),
            identity_hash=self.identity_hash(),
        )

    def require_identity_hash(self, persisted_hash: str) -> None:
        require_full_hash(persisted_hash, field="optimizer_config_hash")
        if persisted_hash != self.identity_hash():
            raise ValueError(
                "optimizer_config_hash conflicts with resolved MIPROv2 control"
            )

    def replay_dataset_rng(self) -> random.Random:
        """Recreate the RNG immediately after auto validation sampling."""

        rng = random.Random(self.seed)
        if self.auto_validation_sample_indices is not None:
            indices = tuple(
                rng.sample(
                    range(len(_pre_auto_valset(self))),
                    len(self.auto_validation_sample_indices),
                )
            )
            if indices != self.auto_validation_sample_indices:
                raise ValueError(
                    "resolved MIPROv2 validation sample conflicts with seed"
                )
        return rng


def _pre_auto_valset(control: Miprov2Control) -> tuple[str, ...]:
    if control.source_valset_task_identities is not None:
        return control.source_valset_task_identities
    size = min(
        1000,
        max(1, int(len(control.source_trainset_task_identities) * 0.80)),
    )
    return control.source_trainset_task_identities[-size:]


def _validate_resolved_derivations(control: Miprov2Control) -> None:
    """Reject a persisted control that configure_miprov2 could not produce."""

    source_trainset = control.source_trainset_task_identities
    source_valset = control.source_valset_task_identities
    if not source_trainset:
        raise ValueError("resolved MIPROv2 source trainset cannot be empty")
    if source_valset is None:
        if len(source_trainset) < 2:
            raise ValueError(
                "resolved MIPROv2 source trainset requires at least two "
                "tasks when no source valset is present"
            )
        expected_pre_auto_valset = _pre_auto_valset(control)
        expected_trainset = source_trainset[
            : len(source_trainset) - len(expected_pre_auto_valset)
        ]
    else:
        if not source_valset:
            raise ValueError("resolved MIPROv2 source valset cannot be empty")
        expected_trainset = source_trainset
        expected_pre_auto_valset = source_valset

    if control.trainset_task_identities != expected_trainset:
        raise ValueError(
            "resolved MIPROv2 trainset conflicts with source datasets"
        )

    expected_zeroshot = (
        control.max_bootstrapped_demos == 0 and control.max_labeled_demos == 0
    )
    if control.zeroshot_opt is not expected_zeroshot:
        raise ValueError(
            "resolved MIPROv2 zeroshot_opt conflicts with demo maxima"
        )

    if control.auto is None:
        if control.num_candidates is None:
            raise ValueError("resolved manual MIPROv2 requires num_candidates")
        if control.auto_validation_sample_indices is not None:
            raise ValueError(
                "resolved manual MIPROv2 cannot carry auto sample indices"
            )
        if control.valset_task_identities != expected_pre_auto_valset:
            raise ValueError(
                "resolved manual MIPROv2 valset conflicts with source datasets"
            )
        if (
            control.num_instruct_candidates != control.num_candidates
            or control.num_fewshot_candidates != control.num_candidates
        ):
            raise ValueError(
                "resolved manual MIPROv2 candidate counts conflict with "
                "num_candidates"
            )
    else:
        if control.num_candidates is not None:
            raise ValueError(
                "resolved auto MIPROv2 cannot carry num_candidates"
            )
        n, requested_val_size = _AUTO_RUN_SETTINGS[control.auto]
        expected_batch_size = min(
            requested_val_size, len(expected_pre_auto_valset)
        )
        rng = random.Random(control.seed)
        expected_indices = tuple(
            rng.sample(
                range(len(expected_pre_auto_valset)),
                expected_batch_size,
            )
        )
        if control.auto_validation_sample_indices != expected_indices:
            raise ValueError(
                "resolved MIPROv2 validation sample conflicts with seed "
                "or auto mode"
            )
        expected_valset = tuple(
            expected_pre_auto_valset[index] for index in expected_indices
        )
        if control.valset_task_identities != expected_valset:
            raise ValueError(
                "resolved MIPROv2 valset order conflicts with sample indices"
            )
        expected_instruct = n if expected_zeroshot else int(n * 0.5)
        if (
            control.num_instruct_candidates != expected_instruct
            or control.num_fewshot_candidates != n
        ):
            raise ValueError(
                "resolved auto MIPROv2 candidate counts conflict with mode"
            )
        expected_trials = _recommended_num_trials(
            component_count=len(control.component_specs),
            zeroshot_opt=expected_zeroshot,
            num_candidates=n,
        )
        if control.num_trials != expected_trials:
            raise ValueError(
                "resolved auto MIPROv2 num_trials conflicts with mode"
            )
        expected_minibatch = len(expected_valset) > _MIN_MINIBATCH_SIZE
        if control.minibatch is not expected_minibatch:
            raise ValueError(
                "resolved auto MIPROv2 minibatch conflicts with valset size"
            )

    if control.minibatch and control.minibatch_size > len(
        control.valset_task_identities
    ):
        raise ValueError(
            "resolved MIPROv2 minibatch_size exceeds resolved valset"
        )


def _require_strict_int(value: Any, *, field: str) -> int:
    if type(value) is not int:
        raise ValueError(f"Whetstone safety: {field} must be an integer")
    return value


def _require_positive_int(value: Any, *, field: str) -> int:
    resolved = _require_strict_int(value, field=field)
    if resolved <= 0:
        raise ValueError(f"Whetstone safety: {field} must be positive")
    return resolved


def _require_nonnegative_int(value: Any, *, field: str) -> int:
    resolved = _require_strict_int(value, field=field)
    if resolved < 0:
        raise ValueError(f"Whetstone safety: {field} must be nonnegative")
    return resolved


def _require_finite(value: Any, *, field: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"Whetstone safety: {field} must be finite")
    return float(value)


def _validate_teacher_settings(settings: dict[str, Any]) -> None:
    """Validate settings at configuration while preserving arbitrary LM kwargs.

    Native controls are validated here; all other JSON keys are carried as
    provider body extensions. An explicit ``extra_body`` may not duplicate a
    top-level extension because that would make translation ambiguous.
    """

    extra_body = settings.get("extra_body")
    if extra_body is not None and not isinstance(extra_body, dict):
        raise ValueError(
            "Whetstone safety: teacher_settings.extra_body must be an object"
        )
    if isinstance(extra_body, dict):
        conflicts = (set(settings) - {"extra_body"}) & set(extra_body)
        if conflicts:
            raise ValueError(
                "Whetstone safety: teacher_settings duplicates extra_body "
                f"keys: {', '.join(sorted(conflicts))}"
            )
    if "temperature" in settings and settings["temperature"] is not None:
        _require_finite(
            settings["temperature"],
            field="teacher_settings.temperature",
        )
    if "token_limit" in settings and settings["token_limit"] is not None:
        _require_positive_int(
            settings["token_limit"],
            field="teacher_settings.token_limit",
        )
    if "reasoning" in settings and settings["reasoning"] is not None:
        if not isinstance(settings["reasoning"], str):
            raise ValueError(
                "Whetstone safety: teacher_settings.reasoning must be a string"
            )


def _validate_resolved_numeric_controls(control: Miprov2Control) -> None:
    _require_nonnegative_int(
        control.max_bootstrapped_demos,
        field="max_bootstrapped_demos",
    )
    _require_nonnegative_int(
        control.max_labeled_demos,
        field="max_labeled_demos",
    )
    if control.num_candidates is not None:
        _require_positive_int(control.num_candidates, field="num_candidates")
    _require_positive_int(
        control.num_instruct_candidates,
        field="num_instruct_candidates",
    )
    _require_positive_int(
        control.num_fewshot_candidates,
        field="num_fewshot_candidates",
    )
    _require_positive_int(control.num_trials, field="num_trials")
    if control.num_threads is not None:
        _require_positive_int(control.num_threads, field="num_threads")
    _require_positive_int(control.max_errors, field="max_errors")
    _require_strict_int(control.seed, field="seed")
    _require_finite(control.init_temperature, field="init_temperature")
    if control.metric_threshold is not None:
        _require_finite(control.metric_threshold, field="metric_threshold")
    _require_positive_int(control.minibatch_size, field="minibatch_size")
    _require_positive_int(
        control.minibatch_full_eval_steps,
        field="minibatch_full_eval_steps",
    )
    _require_positive_int(
        control.view_data_batch_size,
        field="view_data_batch_size",
    )


def _normalize_program_layout(
    program_layout: Miprov2ProgramLayout | None,
    *,
    base_candidate: CandidateRef,
    prompt_format_identity_hash: str,
) -> Miprov2ProgramLayout:
    if program_layout is None:
        field = MUTATION_FIELD
        template = base_candidate.record.payload.get(field)
        if type(template) is not str or not template:
            raise ValueError(
                "Whetstone safety: base candidate requires a non-empty "
                "string user_prompt_template"
            )
        return Miprov2ProgramLayout(
            layout_id="canonical-mutation-surface",
            component_specs=(
                Miprov2ComponentSpec(
                    component_id="generate",
                    prompt_format_identity_hash=prompt_format_identity_hash,
                ),
            ),
        )
    return Miprov2ProgramLayout.model_validate(
        program_layout.model_dump(mode="python")
    )


def _validate_candidate_components(
    candidate: CandidateRef,
    component_specs: tuple[Miprov2ComponentSpec, ...],
    *,
    role: str,
) -> None:
    if len(component_specs) != 1:
        raise ValueError("MIPROv2 requires exactly one optimizable component")
    template = candidate.record.payload.get(MUTATION_FIELD)
    if type(template) is not str or not template:
        raise ValueError(
            f"Whetstone safety: {role} component field "
            f"{MUTATION_FIELD!r} must be a non-empty string"
        )


def _validate_input_numeric_controls(
    *,
    max_bootstrapped_demos: int,
    max_labeled_demos: int,
    run_max_bootstrapped_demos: int | None,
    run_max_labeled_demos: int | None,
    num_candidates: int | None,
    num_trials: int | None,
    num_threads: int | None,
    max_errors: int | None,
    seed: int,
    run_seed: int | None,
    init_temperature: float,
    metric_threshold: float | None,
    minibatch_size: int,
    minibatch_full_eval_steps: int,
    view_data_batch_size: int,
) -> None:
    _require_nonnegative_int(
        max_bootstrapped_demos,
        field="max_bootstrapped_demos",
    )
    _require_nonnegative_int(
        max_labeled_demos,
        field="max_labeled_demos",
    )
    if run_max_bootstrapped_demos is not None:
        _require_nonnegative_int(
            run_max_bootstrapped_demos,
            field="run_max_bootstrapped_demos",
        )
    if run_max_labeled_demos is not None:
        _require_nonnegative_int(
            run_max_labeled_demos,
            field="run_max_labeled_demos",
        )
    if num_candidates is not None:
        _require_positive_int(num_candidates, field="num_candidates")
    if num_trials is not None:
        _require_positive_int(num_trials, field="num_trials")
    if num_threads is not None:
        _require_positive_int(num_threads, field="num_threads")
    if max_errors is not None:
        _require_positive_int(max_errors, field="max_errors")
    _require_strict_int(seed, field="seed")
    if run_seed is not None:
        _require_strict_int(run_seed, field="run_seed")
    _require_finite(init_temperature, field="init_temperature")
    if metric_threshold is not None:
        _require_finite(metric_threshold, field="metric_threshold")
    _require_positive_int(minibatch_size, field="minibatch_size")
    _require_positive_int(
        minibatch_full_eval_steps,
        field="minibatch_full_eval_steps",
    )
    _require_positive_int(
        view_data_batch_size,
        field="view_data_batch_size",
    )


def _recommended_num_trials(
    *,
    component_count: int,
    zeroshot_opt: bool,
    num_candidates: int,
) -> int:
    num_vars = component_count * (1 if zeroshot_opt else 2)
    return int(
        max(
            2 * num_vars * math.log2(num_candidates),
            1.5 * num_candidates,
        )
    )


def configure_miprov2(
    metric: EvalConfigRef | None = None,
    prompt_model: ProposerConfig | None = None,
    task_model: str | None = None,
    teacher_settings: dict[str, Any] | None = None,
    max_bootstrapped_demos: int = 4,
    max_labeled_demos: int = 4,
    auto: Miprov2AutoMode | None = "light",
    num_candidates: int | None = None,
    num_threads: int | None = None,
    max_errors: int | None = None,
    seed: int = 9,
    init_temperature: float = 1.0,
    verbose: bool = False,
    track_stats: bool = True,
    log_dir: str | None = None,
    metric_threshold: float | None = None,
    *,
    base_candidate: CandidateRef,
    program_layout: Miprov2ProgramLayout | None = None,
    trainset: tuple[str, ...],
    teacher: CandidateRef | None = None,
    teacher_compiled: bool | None = None,
    valset: tuple[str, ...] | None = None,
    num_trials: int | None = None,
    run_max_bootstrapped_demos: int | None = None,
    run_max_labeled_demos: int | None = None,
    run_seed: int | None = None,
    minibatch: bool = True,
    minibatch_size: int = 35,
    minibatch_full_eval_steps: int = 5,
    program_aware_proposer: bool = True,
    data_aware_proposer: bool = True,
    view_data_batch_size: int = 10,
    tip_aware_proposer: bool = True,
    fewshot_aware_proposer: bool = True,
    provide_traceback: bool | None = None,
    defaults: Miprov2InjectedDefaults,
) -> Miprov2Control:
    """Resolve DSPy's constructor and compile controls without effects.

    Validation and error precedence through minibatch-size checking follows
    the frozen DSPy implementation. Whetstone-specific identity and route
    checks are explicitly labeled as safety validations.
    """

    allowed_modes = {None, "light", "medium", "heavy"}
    if auto not in allowed_modes:
        raise ValueError(
            f"Invalid value for auto: {auto}. Must be one of {allowed_modes}."
        )

    if (
        metric is None
        and not defaults.validation_eval_source_is_metric_authority
    ):
        raise ValueError(
            "metric is required unless the injected validation Eval Config "
            "is explicitly declared as the metric authority"
        )
    metric_authority: Literal["explicit", "injected_default"] = (
        "injected_default" if metric is None else "explicit"
    )

    effective_max_errors = (
        defaults.max_errors if max_errors is None else max_errors
    )
    effective_max_bootstrapped_demos = (
        max_bootstrapped_demos
        if run_max_bootstrapped_demos is None
        else run_max_bootstrapped_demos
    )
    effective_max_labeled_demos = (
        max_labeled_demos
        if run_max_labeled_demos is None
        else run_max_labeled_demos
    )
    zeroshot_opt = (
        effective_max_bootstrapped_demos == 0
        and effective_max_labeled_demos == 0
    )
    component_count = (
        1 if program_layout is None else len(program_layout.component_specs)
    )

    if auto is None and num_candidates is not None and num_trials is None:
        if type(num_candidates) is not int or num_candidates <= 0:
            _require_positive_int(num_candidates, field="num_candidates")
        recommendation = _recommended_num_trials(
            component_count=component_count,
            zeroshot_opt=zeroshot_opt,
            num_candidates=num_candidates,
        )
        raise ValueError(
            "If auto is None, num_trials must also be provided. Given "
            f"num_candidates={num_candidates}, we'd recommend setting "
            f"num_trials to ~{recommendation}."
        )
    if auto is None and (num_candidates is None or num_trials is None):
        raise ValueError(
            "If auto is None, num_candidates must also be provided."
        )
    if auto is not None and (
        num_candidates is not None or num_trials is not None
    ):
        raise ValueError(
            "If auto is not None, num_candidates and num_trials cannot be "
            "set, since they would be overridden by the auto settings. "
            "Please either set auto to None, or do not specify "
            "num_candidates and num_trials."
        )

    resolved_seed = run_seed or seed
    rng = random.Random(resolved_seed)

    source_trainset = tuple(trainset)
    source_valset = tuple(valset) if valset is not None else None
    if not source_trainset:
        raise ValueError("Trainset cannot be empty.")
    if source_valset is None:
        if len(source_trainset) < 2:
            raise ValueError(
                "Trainset must have at least 2 examples if no valset "
                "specified."
            )
        valset_size = min(1000, max(1, int(len(source_trainset) * 0.80)))
        cutoff = len(source_trainset) - valset_size
        resolved_trainset = source_trainset[:cutoff]
        pre_auto_valset = source_trainset[cutoff:]
    else:
        if len(source_valset) < 1:
            raise ValueError("Validation set must have at least 1 example.")
        resolved_trainset = source_trainset
        pre_auto_valset = source_valset

    sample_indices: tuple[int, ...] | None = None
    if auto is None:
        assert num_candidates is not None
        assert num_trials is not None
        num_instruct_candidates = num_candidates
        num_fewshot_candidates = num_candidates
        resolved_num_trials = num_trials
        resolved_valset = pre_auto_valset
        resolved_minibatch = minibatch
    else:
        n, val_size = _AUTO_RUN_SETTINGS[auto]
        batch_size = min(val_size, len(pre_auto_valset))
        sample_indices = tuple(
            rng.sample(range(len(pre_auto_valset)), batch_size)
        )
        resolved_valset = tuple(
            pre_auto_valset[index] for index in sample_indices
        )
        resolved_minibatch = len(resolved_valset) > _MIN_MINIBATCH_SIZE
        num_instruct_candidates = n if zeroshot_opt else int(n * 0.5)
        num_fewshot_candidates = n
        resolved_num_trials = _recommended_num_trials(
            component_count=component_count,
            zeroshot_opt=zeroshot_opt,
            num_candidates=n,
        )

    if resolved_minibatch and minibatch_size > len(resolved_valset):
        raise ValueError(
            "Minibatch size cannot exceed the size of the valset. Valset "
            f"size: {len(resolved_valset)}."
        )

    adapter_identity_hash = prompt_adapter_identity_hash(
        defaults.prompt_adapter
    )
    resolved_program_layout = _normalize_program_layout(
        program_layout,
        base_candidate=base_candidate,
        prompt_format_identity_hash=adapter_identity_hash,
    )
    resolved_component_specs = resolved_program_layout.component_specs

    _validate_input_numeric_controls(
        max_bootstrapped_demos=max_bootstrapped_demos,
        max_labeled_demos=max_labeled_demos,
        run_max_bootstrapped_demos=run_max_bootstrapped_demos,
        run_max_labeled_demos=run_max_labeled_demos,
        num_candidates=num_candidates,
        num_trials=num_trials,
        num_threads=num_threads,
        max_errors=max_errors,
        seed=seed,
        run_seed=run_seed,
        init_temperature=init_temperature,
        metric_threshold=metric_threshold,
        minibatch_size=minibatch_size,
        minibatch_full_eval_steps=minibatch_full_eval_steps,
        view_data_batch_size=view_data_batch_size,
    )
    for spec in resolved_component_specs:
        if spec.prompt_format_identity_hash != adapter_identity_hash:
            raise ValueError(
                "Whetstone safety: component prompt format conflicts with "
                "the bound prompt adapter"
            )

    resolved_prompt_model = (
        defaults.prompt_model if not prompt_model else prompt_model
    )
    resolved_validation_source = (
        defaults.validation_eval_source if metric is None else metric
    )
    resolved_task_model_hash = task_model or defaults.task_model_identity_hash
    require_full_hash(
        resolved_task_model_hash,
        field="task_model_identity_hash",
    )

    resolved_teacher = base_candidate if teacher is None else teacher
    if teacher_compiled is not None and type(teacher_compiled) is not bool:
        raise ValueError(
            "Whetstone safety: teacher_compiled must be a boolean"
        )
    resolved_teacher_compiled = bool(teacher_compiled)
    _validate_candidate_components(
        base_candidate,
        resolved_component_specs,
        role="base candidate",
    )
    _validate_candidate_components(
        resolved_teacher,
        resolved_component_specs,
        role="teacher candidate",
    )

    base_snapshot = CandidateRef.model_validate(
        base_candidate.model_dump(mode="python")
    )
    teacher_snapshot = CandidateRef.model_validate(
        resolved_teacher.model_dump(mode="python")
    )
    layout_snapshot = Miprov2ProgramLayout.model_validate(
        resolved_program_layout.model_dump(mode="python")
    )
    prompt_model_snapshot = ProposerConfig.model_validate(
        resolved_prompt_model.model_dump(mode="python")
    )
    bootstrap_source_snapshot = EvalConfigRef.model_validate(
        defaults.bootstrap_eval_source.model_dump(mode="python")
    )
    validation_source_snapshot = EvalConfigRef.model_validate(
        resolved_validation_source.model_dump(mode="python")
    )

    return Miprov2Control(
        base_candidate=base_snapshot,
        teacher_candidate=teacher_snapshot,
        teacher_compiled=resolved_teacher_compiled,
        program_layout=layout_snapshot,
        source_trainset_task_identities=source_trainset,
        source_valset_task_identities=source_valset,
        trainset_task_identities=resolved_trainset,
        valset_task_identities=resolved_valset,
        auto_validation_sample_indices=sample_indices,
        prompt_model=prompt_model_snapshot,
        bootstrap_eval_source=bootstrap_source_snapshot,
        validation_eval_source=validation_source_snapshot,
        reward_policy=defaults.reward_policy,
        evaluation_binding=EvaluationBinding.model_validate(
            {
                **defaults.evaluation_binding.model_dump(mode="json"),
                "eval_config": validation_source_snapshot.model_dump(
                    mode="json"
                ),
            }
        ),
        provider_execution_policy_hash=(
            defaults.provider_execution_policy_hash
        ),
        task_model_identity_hash=resolved_task_model_hash,
        prompt_adapter_identity_hash=adapter_identity_hash,
        template_render_contract=defaults.template_render_contract,
        metric_authority=metric_authority,
        teacher_settings=ImmutableJsonObject(deepcopy(teacher_settings or {})),
        max_bootstrapped_demos=effective_max_bootstrapped_demos,
        max_labeled_demos=effective_max_labeled_demos,
        auto=auto,
        num_candidates=num_candidates,
        num_instruct_candidates=num_instruct_candidates,
        num_fewshot_candidates=num_fewshot_candidates,
        num_trials=resolved_num_trials,
        num_threads=num_threads,
        max_errors=effective_max_errors,
        seed=resolved_seed,
        init_temperature=init_temperature,
        verbose=verbose,
        track_stats=track_stats,
        log_dir=log_dir,
        metric_threshold=metric_threshold,
        minibatch=resolved_minibatch,
        minibatch_size=minibatch_size,
        minibatch_full_eval_steps=minibatch_full_eval_steps,
        program_aware_proposer=program_aware_proposer,
        data_aware_proposer=data_aware_proposer,
        view_data_batch_size=view_data_batch_size,
        tip_aware_proposer=tip_aware_proposer,
        fewshot_aware_proposer=fewshot_aware_proposer,
        provide_traceback=provide_traceback,
        zeroshot_opt=zeroshot_opt,
    )


__all__ = [
    "MIPROV2_ALGORITHM_VERSION",
    "MIPROV2_COMPONENT_SPEC_SCHEMA",
    "MIPROV2_COMPONENT_SPEC_SCHEMA_VERSION",
    "MIPROV2_CONTROL_SCHEMA",
    "MIPROV2_CONTROL_SCHEMA_VERSION",
    "MIPROV2_OPTUNA_VERSION",
    "MIPROV2_PHASE_SCHEMA_MANIFEST",
    "MIPROV2_PROGRAM_LAYOUT_SCHEMA",
    "MIPROV2_PROGRAM_LAYOUT_SCHEMA_VERSION",
    "MIPROV2_PROMPT_FORMAT_ADAPTER_VERSION",
    "MIPROV2_PROPOSER_OUTPUT_PARSER_SCHEMA",
    "MIPROV2_PROPOSER_OUTPUT_PARSER_SCHEMA_VERSION",
    "MIPROV2_REFERENCE_COMMIT",
    "MIPROV2_RESULT_SCHEMA",
    "MIPROV2_RESULT_SCHEMA_VERSION",
    "MIPROV2_STATE_SCHEMA",
    "MIPROV2_STATE_SCHEMA_VERSION",
    "Miprov2AutoMode",
    "Miprov2ComponentSpec",
    "Miprov2Control",
    "Miprov2InjectedDefaults",
    "Miprov2ProgramLayout",
    "configure_miprov2",
]
