"""Effect-free, replay-stable Optuna ownership for MIPROv2.

The immutable transcript is the authority.  It binds the frozen DSPy/Optuna
contract, exact ordered candidate pools, evaluation schedule, candidate
identities, task batches, Eval Configs, and evidence references.  A fresh
in-memory Optuna study is reconstructed for every operation.
"""

from __future__ import annotations

import importlib
import math
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Literal, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    StrictBool,
    StrictInt,
    StrictStr,
    ValidationError,
    model_validator,
)

from whetstone.optimization.identity import (
    TypedRef,
    compute_identity_hash,
    require_full_hash,
)
from whetstone.optimization.miprov2_control import (
    MIPROV2_CANDIDATE_RENDERER_VERSION,
    Miprov2ProgramLayout,
)
from whetstone.optimization.miprov2_demo import ComponentDemoSet
from whetstone.optimization.miprov2_eval_config import (
    Miprov2EvalConfigBinding,
)
from whetstone.optimization.prompt_program import (
    PROMPT_PROGRAM_PAYLOAD_FIELD,
    PromptProgram,
    PromptProgramComponent,
    PromptProgramExample,
)
from whetstone.optimization.schema import (
    Candidate,
    CandidateRef,
    EvalConfigRef,
    candidate_reference,
)

MIPROV2_ALGORITHM_VERSION = "dspy_miprov2_prompt_program/v1"
MIPROV2_REFERENCE_COMMIT = "6f68dcdb3ef46d70bf0c12596699ebc44e82d6b0"
MIPROV2_STUDY_SCHEMA = "whetstone.miprov2_study_transcript"
MIPROV2_STUDY_SCHEMA_VERSION = 1
OPTUNA_VERSION = "4.8.0"
EVALUATION_EVIDENCE_SCHEMA = "whetstone.evaluation_evidence"
EVALUATION_FAILURE_SCHEMA = "whetstone.evaluation_failure"
REWARD_SCHEMA = "whetstone.reward"
MIPROV2_CANDIDATE_ASSEMBLY_SCHEMA = "whetstone.miprov2_candidate_assembly"
MIPROV2_CANDIDATE_ASSEMBLY_SCHEMA_VERSION = 1
MIPROV2_CANDIDATE_RENDERING_SCHEMA = "whetstone.miprov2_candidate_rendering"
MIPROV2_CANDIDATE_RENDERING_SCHEMA_VERSION = 1
MIPROV2_COMPONENT_DEMO_SET_SCHEMA = "whetstone.miprov2_component_demo_set"
MIPROV2_INSTRUCTION_SCHEMA = "whetstone.miprov2_instruction"
MIPROV2_RENDERER_VERSION = MIPROV2_CANDIDATE_RENDERER_VERSION

type TrialParams = tuple[tuple[StrictStr, StrictInt], ...]
type EvaluationPurpose = Literal[
    "miprov2_baseline",
    "miprov2_sample",
    "miprov2_promotion",
]


class StudyTranscriptMismatch(ValueError):
    """A durable transcript cannot reproduce the frozen study contract."""


class _IdentityRecord(BaseModel):
    """Frozen strict-JSON record with a canonical versioned identity."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    _identity_schema: ClassVar[str]
    _identity_schema_version: ClassVar[int] = 1

    def identity_hash(self) -> str:
        return compute_identity_hash(
            schema=self._identity_schema,
            schema_version=self._identity_schema_version,
            payload=self.model_dump(mode="json"),
        )


class Miprov2ParameterSpace(_IdentityRecord):
    """Exact candidate pools in predictor-major suggestion order."""

    _identity_schema = "whetstone.miprov2_parameter_space"

    instruction_pool_identity_hashes: tuple[tuple[StrictStr, ...], ...]
    demo_pool_identity_hashes: tuple[tuple[StrictStr, ...], ...] | None = None

    @model_validator(mode="after")
    def _validate_pools(self) -> Miprov2ParameterSpace:
        if not self.instruction_pool_identity_hashes:
            raise ValueError("at least one predictor is required")
        for predictor_index, pool in enumerate(
            self.instruction_pool_identity_hashes
        ):
            if not pool:
                raise ValueError(
                    "instruction candidate pools must all be non-empty"
                )
            for candidate_index, identity_hash in enumerate(pool):
                require_full_hash(
                    identity_hash,
                    field=(
                        "instruction_pool_identity_hashes"
                        f"[{predictor_index}][{candidate_index}]"
                    ),
                )
        demos = self.demo_pool_identity_hashes
        if demos is not None:
            if len(demos) != len(self.instruction_pool_identity_hashes):
                raise ValueError(
                    "demo candidate pools must match predictor count"
                )
            for predictor_index, pool in enumerate(demos):
                if not pool:
                    raise ValueError(
                        "demo candidate pools must all be non-empty"
                    )
                for candidate_index, identity_hash in enumerate(pool):
                    require_full_hash(
                        identity_hash,
                        field=(
                            "demo_pool_identity_hashes"
                            f"[{predictor_index}][{candidate_index}]"
                        ),
                    )
        return self

    @property
    def instruction_candidate_counts(self) -> tuple[int, ...]:
        return tuple(
            len(pool) for pool in self.instruction_pool_identity_hashes
        )

    @property
    def demo_candidate_counts(self) -> tuple[int, ...] | None:
        if self.demo_pool_identity_hashes is None:
            return None
        return tuple(len(pool) for pool in self.demo_pool_identity_hashes)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        names: list[str] = []
        for predictor_index in range(
            len(self.instruction_pool_identity_hashes)
        ):
            names.append(f"{predictor_index}_predictor_instruction")
            if self.demo_pool_identity_hashes is not None:
                names.append(f"{predictor_index}_predictor_demos")
        return tuple(names)

    @property
    def baseline_params(self) -> TrialParams:
        return tuple((name, 0) for name in self.parameter_names)

    def candidate_count(self, name: str) -> int:
        parts = name.split("_", maxsplit=2)
        if len(parts) != 3 or parts[1] != "predictor":
            raise ValueError(f"unknown MIPROv2 parameter {name!r}")
        try:
            predictor_index = int(parts[0])
        except ValueError as exc:
            raise ValueError(f"unknown MIPROv2 parameter {name!r}") from exc
        if (
            not 0
            <= predictor_index
            < len(self.instruction_pool_identity_hashes)
        ):
            raise ValueError(f"unknown MIPROv2 parameter {name!r}")
        if parts[2] == "instruction":
            return len(self.instruction_pool_identity_hashes[predictor_index])
        if parts[2] == "demos" and self.demo_pool_identity_hashes is not None:
            return len(self.demo_pool_identity_hashes[predictor_index])
        raise ValueError(f"unknown MIPROv2 parameter {name!r}")

    def normalize(
        self,
        params: Mapping[str, int] | TrialParams,
    ) -> TrialParams:
        if isinstance(params, tuple):
            normalized = tuple(params)
        else:
            if set(params) != set(self.parameter_names):
                raise ValueError(
                    "parameter mapping does not match the MIPROv2 space"
                )
            normalized = tuple(
                (name, params[name]) for name in self.parameter_names
            )
        if tuple(name for name, _ in normalized) != self.parameter_names:
            raise ValueError(
                "parameters are not in predictor-major instruction/demo order"
            )
        for name, value in normalized:
            if type(value) is not int:
                raise ValueError(f"{name} must be an integer category")
            if not 0 <= value < self.candidate_count(name):
                raise ValueError(
                    f"{name} category {value} is outside its search space"
                )
        return cast("TrialParams", normalized)

    def as_dict(self, params: TrialParams) -> dict[str, int]:
        return dict(self.normalize(params))

    def distribution_identity_hash(self) -> str:
        return compute_identity_hash(
            schema="whetstone.miprov2_optuna_distributions",
            schema_version=1,
            payload={
                "parameters": [
                    {
                        "name": name,
                        "distribution": "categorical",
                        "choices": list(range(self.candidate_count(name))),
                    }
                    for name in self.parameter_names
                ]
            },
        )

    def combination_identity_hash(self, params: TrialParams) -> str:
        normalized = self.normalize(params)
        selected: list[dict[str, Any]] = []
        values = dict(normalized)
        for predictor_index, instruction_pool in enumerate(
            self.instruction_pool_identity_hashes
        ):
            instruction_index = values[
                f"{predictor_index}_predictor_instruction"
            ]
            item: dict[str, Any] = {
                "predictor_index": predictor_index,
                "instruction_index": instruction_index,
                "instruction_identity_hash": instruction_pool[
                    instruction_index
                ],
            }
            if self.demo_pool_identity_hashes is not None:
                demo_index = values[f"{predictor_index}_predictor_demos"]
                item["demo_index"] = demo_index
                item["demo_identity_hash"] = self.demo_pool_identity_hashes[
                    predictor_index
                ][demo_index]
            selected.append(item)
        return compute_identity_hash(
            schema="whetstone.miprov2_trial_combination",
            schema_version=1,
            payload={"selected": selected},
        )


class Miprov2StudySchedule(_IdentityRecord):
    """Persisted controls from which objective and promotion timing derives."""

    _identity_schema = "whetstone.miprov2_study_schedule"

    num_trials: StrictInt
    minibatch: StrictBool
    minibatch_size: StrictInt
    valset_size: StrictInt
    minibatch_full_eval_steps: StrictInt

    @model_validator(mode="after")
    def _validate_schedule(self) -> Miprov2StudySchedule:
        if self.num_trials <= 0:
            raise ValueError("num_trials must be positive")
        if self.minibatch_size <= 0:
            raise ValueError("minibatch_size must be positive")
        if self.valset_size <= 0:
            raise ValueError("valset_size must be positive")
        if self.minibatch_full_eval_steps <= 0:
            raise ValueError("minibatch_full_eval_steps must be positive")
        if self.minibatch and self.minibatch_size > self.valset_size:
            raise ValueError("minibatch_size cannot exceed valset_size")
        return self

    @property
    def adjusted_num_trials(self) -> int:
        if not self.minibatch:
            return self.num_trials
        extra_at_end = (
            1 if self.num_trials % self.minibatch_full_eval_steps != 0 else 0
        )
        return (
            self.num_trials
            + self.num_trials // self.minibatch_full_eval_steps
            + 1
            + extra_at_end
        )

    def promotion_due(self, *, optuna_trial_number: int) -> bool:
        if not self.minibatch:
            return False
        display_trial_number = optuna_trial_number + 1
        return (
            display_trial_number % (self.minibatch_full_eval_steps + 1) == 0
            or display_trial_number == self.adjusted_num_trials - 1
        )


class VerifiedEvaluationCitation(_IdentityRecord):
    """Identity-bound projection of externally verified evaluation evidence.

    The evaluation engine owns the full evidence record.  Before this pure
    study seam accepts its reference, the caller projects the fields whose
    equality is algorithmically significant.  Keeping that projection inside
    the transcript makes a cross-run, cross-candidate, reordered-task, or
    changed-config substitution detectable without loading the object store
    during Optuna replay.
    """

    _identity_schema = "whetstone.miprov2_verified_evaluation_citation"

    run_id: StrictStr
    intent_id: StrictStr
    effect_identity_hash: StrictStr
    purpose: EvaluationPurpose
    candidate_identity_hash: StrictStr
    task_batch_identities: tuple[StrictStr, ...]
    validation_eval_source_identity_hash: StrictStr
    eval_config_identity_hash: StrictStr
    eval_config_binding_identity_hash: StrictStr
    reward_policy_hash: StrictStr
    evidence_ref: TypedRef
    reward_ref: TypedRef
    normalized_score: float

    @model_validator(mode="after")
    def _validate_citation(self) -> VerifiedEvaluationCitation:
        if not self.run_id or not self.intent_id:
            raise ValueError(
                "evaluation citation run_id and intent_id are required"
            )
        require_full_hash(
            self.effect_identity_hash,
            field="effect_identity_hash",
        )
        require_full_hash(
            self.candidate_identity_hash,
            field="candidate_identity_hash",
        )
        require_full_hash(
            self.validation_eval_source_identity_hash,
            field="validation_eval_source_identity_hash",
        )
        require_full_hash(
            self.eval_config_identity_hash,
            field="eval_config_identity_hash",
        )
        require_full_hash(
            self.eval_config_binding_identity_hash,
            field="eval_config_binding_identity_hash",
        )
        require_full_hash(
            self.reward_policy_hash,
            field="reward_policy_hash",
        )
        if not self.task_batch_identities:
            raise ValueError(
                "citation task_batch_identities must not be empty"
            )
        for index, identity_hash in enumerate(self.task_batch_identities):
            require_full_hash(
                identity_hash,
                field=f"task_batch_identities[{index}]",
            )
        if self.evidence_ref.schema_name not in {
            EVALUATION_EVIDENCE_SCHEMA,
            EVALUATION_FAILURE_SCHEMA,
        }:
            raise ValueError(
                "evaluation citation must reference canonical "
                "evaluation evidence"
            )
        if self.reward_ref.schema_name != REWARD_SCHEMA:
            raise ValueError(
                "evaluation citation must reference a canonical reward"
            )
        _require_finite(self.normalized_score, field="normalized_score")
        return self


class EvaluationBinding(_IdentityRecord):
    """Exact durable provenance for one candidate evaluation."""

    _identity_schema = "whetstone.miprov2_evaluation_binding"

    run_id: StrictStr
    intent_id: StrictStr
    effect_identity_hash: StrictStr
    purpose: EvaluationPurpose
    candidate_identity_hash: StrictStr
    task_batch_identities: tuple[StrictStr, ...]
    eval_config: EvalConfigRef
    eval_config_binding: Miprov2EvalConfigBinding
    reward_policy_hash: StrictStr
    reward_ref: TypedRef
    evidence_citations: tuple[VerifiedEvaluationCitation, ...]
    normalized_score: float

    @model_validator(mode="after")
    def _validate_evidence(self) -> EvaluationBinding:
        if not self.run_id or not self.intent_id:
            raise ValueError(
                "evaluation binding run_id and intent_id are required"
            )
        require_full_hash(
            self.effect_identity_hash,
            field="effect_identity_hash",
        )
        require_full_hash(
            self.candidate_identity_hash,
            field="candidate_identity_hash",
        )
        require_full_hash(
            self.reward_policy_hash,
            field="reward_policy_hash",
        )
        if not self.task_batch_identities:
            raise ValueError("task_batch_identities must not be empty")
        for index, identity_hash in enumerate(self.task_batch_identities):
            require_full_hash(
                identity_hash,
                field=f"task_batch_identities[{index}]",
            )
        if self.reward_ref.schema_name != REWARD_SCHEMA:
            raise ValueError(
                "evaluation binding must reference a canonical reward"
            )
        if not self.evidence_citations:
            raise ValueError("evaluation evidence_citations must not be empty")
        if self.eval_config != self.eval_config_binding.eval_config:
            raise ValueError(
                "evaluation Eval Config differs from its derivation binding"
            )
        eval_config_binding_identity = self.eval_config_binding.identity_hash()
        validation_eval_source_identity = (
            self.eval_config_binding.request.source_eval_config.identity_hash
        )
        expected = (
            self.run_id,
            self.intent_id,
            self.effect_identity_hash,
            self.purpose,
            self.candidate_identity_hash,
            self.task_batch_identities,
            validation_eval_source_identity,
            self.eval_config.identity_hash,
            eval_config_binding_identity,
            self.reward_policy_hash,
            self.reward_ref,
            self.normalized_score,
        )
        for citation in self.evidence_citations:
            actual = (
                citation.run_id,
                citation.intent_id,
                citation.effect_identity_hash,
                citation.purpose,
                citation.candidate_identity_hash,
                citation.task_batch_identities,
                citation.validation_eval_source_identity_hash,
                citation.eval_config_identity_hash,
                citation.eval_config_binding_identity_hash,
                citation.reward_policy_hash,
                citation.reward_ref,
                citation.normalized_score,
            )
            if actual != expected:
                raise ValueError(
                    "evaluation citation does not match its bound evaluation"
                )
        _require_finite(self.normalized_score, field="normalized_score")
        return self

    @property
    def evidence_refs(self) -> tuple[TypedRef, ...]:
        return tuple(
            citation.evidence_ref for citation in self.evidence_citations
        )


class BaselineObservation(_IdentityRecord):
    """Trial-zero candidate and its full-validation evidence."""

    _identity_schema = "whetstone.miprov2_baseline_observation"

    categorical_combination_identity_hash: StrictStr
    evaluated_base_candidate: CandidateRef
    score: float
    evaluation: EvaluationBinding

    @model_validator(mode="after")
    def _validate_baseline(self) -> BaselineObservation:
        require_full_hash(
            self.categorical_combination_identity_hash,
            field="categorical_combination_identity_hash",
        )
        require_full_hash(
            self.evaluated_base_candidate.identity_hash,
            field="evaluated_candidate_identity_hash",
        )
        if (
            self.evaluation.candidate_identity_hash
            != self.evaluated_base_candidate.identity_hash
        ):
            raise ValueError(
                "baseline evidence does not match evaluated candidate"
            )
        if self.evaluation.purpose != "miprov2_baseline":
            raise ValueError("baseline evidence has the wrong purpose")
        _require_finite(self.score, field="score")
        if self.score != self.evaluation.normalized_score:
            raise ValueError(
                "baseline score does not match verified evaluation"
            )
        return self


class Miprov2CandidateAssemblyBinding(_IdentityRecord):
    """Canonical params-to-native-program assembly persisted for evaluation."""

    _identity_schema = MIPROV2_CANDIDATE_ASSEMBLY_SCHEMA
    _identity_schema_version = MIPROV2_CANDIDATE_ASSEMBLY_SCHEMA_VERSION

    params: TrialParams
    categorical_combination_identity_hash: StrictStr
    candidate: CandidateRef
    program_identity_hash: StrictStr
    control_identity_hash: StrictStr
    base_candidate: CandidateRef
    program_layout: Miprov2ProgramLayout
    prompt_adapter_identity_hash: StrictStr

    @model_validator(mode="after")
    def _validate_assembly(self) -> Miprov2CandidateAssemblyBinding:
        for field in (
            "categorical_combination_identity_hash",
            "program_identity_hash",
            "control_identity_hash",
            "prompt_adapter_identity_hash",
        ):
            require_full_hash(getattr(self, field), field=field)
        rendering = self.candidate.record.payload.get(
            "miprov2_candidate_rendering"
        )
        if not isinstance(rendering, dict):
            raise ValueError(
                "assembled candidate requires canonical MIPROv2 rendering"
            )
        if set(rendering) != {
            "control_identity_hash",
            "base_candidate_identity_hash",
            "categorical_combination_identity_hash",
            "renderer_version",
            "components",
        }:
            raise ValueError("candidate rendering has a non-canonical shape")
        if rendering.get("renderer_version") != MIPROV2_RENDERER_VERSION:
            raise ValueError(
                "candidate rendering has the wrong renderer version"
            )
        expected_rendering_context = (
            self.control_identity_hash,
            self.base_candidate.identity_hash,
            self.categorical_combination_identity_hash,
        )
        actual_rendering_context = (
            rendering.get("control_identity_hash"),
            rendering.get("base_candidate_identity_hash"),
            rendering.get("categorical_combination_identity_hash"),
        )
        if actual_rendering_context != expected_rendering_context:
            raise ValueError(
                "candidate rendering does not match its assembly context"
            )
        expected_program_identity = compute_identity_hash(
            schema=MIPROV2_CANDIDATE_RENDERING_SCHEMA,
            schema_version=MIPROV2_CANDIDATE_RENDERING_SCHEMA_VERSION,
            payload=rendering,
        )
        if self.program_identity_hash != expected_program_identity:
            raise ValueError(
                "program identity does not match canonical candidate rendering"
            )
        if (
            self.candidate.record.candidate_id
            != f"miprov2-{self.program_identity_hash[:24]}"
        ):
            raise ValueError(
                "assembled candidate id does not match its program identity"
            )
        components = rendering.get("components")
        if not isinstance(components, list):
            raise ValueError("candidate rendering components must be ordered")
        if any(not isinstance(component, dict) for component in components):
            raise ValueError(
                "candidate rendering component must be a JSON object"
            )
        canonical_components = cast("list[dict[str, Any]]", components)
        component_ids = [
            component.get("component_id") for component in canonical_components
        ]
        candidate_fields = [
            component.get("candidate_field")
            for component in canonical_components
        ]
        if len(component_ids) != len(set(component_ids)) or len(
            candidate_fields
        ) != len(set(candidate_fields)):
            raise ValueError("candidate rendering components must be unique")
        values = dict(self.params)
        if len(canonical_components) * (
            2
            if any(name.endswith("_predictor_demos") for name in values)
            else 1
        ) != len(values):
            raise ValueError(
                "candidate rendering component count does not match parameters"
            )
        for index, component in enumerate(canonical_components):
            if set(component) != {
                "component_id",
                "candidate_field",
                "instruction_index",
                "instruction",
                "instruction_identity_hash",
                "demo_index",
                "demo_set",
                "demo_identity_hash",
            }:
                raise ValueError(
                    "candidate rendering component has a non-canonical shape"
                )
            if not component.get("component_id") or not component.get(
                "candidate_field"
            ):
                raise ValueError(
                    "candidate rendering component identifiers are required"
                )
            if component.get("instruction_index") != values.get(
                f"{index}_predictor_instruction"
            ):
                raise ValueError(
                    "candidate rendering instruction index differs from params"
                )
            instruction = component.get("instruction")
            if not isinstance(instruction, str):
                raise ValueError(
                    "candidate rendering instruction must be a string"
                )
            expected_instruction_identity = compute_identity_hash(
                schema=MIPROV2_INSTRUCTION_SCHEMA,
                schema_version=1,
                payload={"instruction": instruction},
            )
            if (
                component.get("instruction_identity_hash")
                != expected_instruction_identity
            ):
                raise ValueError(
                    "candidate rendering instruction identity differs "
                    "from text"
                )
            demo_name = f"{index}_predictor_demos"
            if demo_name in values:
                if component.get("demo_index") != values[demo_name]:
                    raise ValueError(
                        "candidate rendering demo index differs from params"
                    )
                demo_set = component.get("demo_set")
                if not isinstance(demo_set, dict):
                    raise ValueError(
                        "few-shot candidate rendering requires a demo set"
                    )
                expected_demo_identity = compute_identity_hash(
                    schema=MIPROV2_COMPONENT_DEMO_SET_SCHEMA,
                    schema_version=1,
                    payload=demo_set,
                )
                if (
                    component.get("demo_identity_hash")
                    != expected_demo_identity
                ):
                    raise ValueError(
                        "candidate rendering demo identity differs "
                        "from content"
                    )
            elif component.get("demo_index") is not None:
                raise ValueError(
                    "zero-shot candidate rendering cannot contain demo index"
                )
            elif (
                component.get("demo_set") is not None
                or component.get("demo_identity_hash") is not None
            ):
                raise ValueError(
                    "zero-shot candidate rendering cannot contain demo data"
                )
        return self


class Promotion(_IdentityRecord):
    """A full evaluation inserted while its sampled trial remains open."""

    _identity_schema = "whetstone.miprov2_promotion"

    trial_number: StrictInt
    params: TrialParams
    candidate_combination_identity_hash: StrictStr
    evaluated_candidate_identity_hash: StrictStr
    candidate_assembly: Miprov2CandidateAssemblyBinding
    source_sample_trial_number: StrictInt
    minibatch_mean: float
    full_score: float
    evaluation: EvaluationBinding

    @model_validator(mode="after")
    def _validate_promotion(self) -> Promotion:
        if self.trial_number < 1:
            raise ValueError("promotion trial_number must be positive")
        if self.source_sample_trial_number < 1:
            raise ValueError("source_sample_trial_number must be positive")
        require_full_hash(
            self.candidate_combination_identity_hash,
            field="candidate_combination_identity_hash",
        )
        require_full_hash(
            self.evaluated_candidate_identity_hash,
            field="evaluated_candidate_identity_hash",
        )
        if (
            self.evaluation.candidate_identity_hash
            != self.evaluated_candidate_identity_hash
        ):
            raise ValueError(
                "promotion evidence does not match evaluated candidate"
            )
        if self.evaluation.purpose != "miprov2_promotion":
            raise ValueError("promotion evidence has the wrong purpose")
        if (
            self.candidate_assembly.candidate.identity_hash
            != self.evaluated_candidate_identity_hash
        ):
            raise ValueError(
                "promotion candidate does not match its assembly binding"
            )
        _require_finite(self.minibatch_mean, field="minibatch_mean")
        _require_finite(self.full_score, field="full_score")
        if self.full_score != self.evaluation.normalized_score:
            raise ValueError(
                "promotion score does not match verified evaluation"
            )
        return self


class SampleObservation(_IdentityRecord):
    """One Optuna-sampled combination and its exact evaluation provenance."""

    _identity_schema = "whetstone.miprov2_sample_observation"

    trial_number: StrictInt
    params: TrialParams
    candidate_combination_identity_hash: StrictStr
    evaluated_candidate_identity_hash: StrictStr
    candidate_assembly: Miprov2CandidateAssemblyBinding
    score: float
    evaluation: EvaluationBinding
    batch_full_evaluation: StrictBool
    promotion: Promotion | None = None

    @model_validator(mode="after")
    def _validate_sample(self) -> SampleObservation:
        if self.trial_number < 1:
            raise ValueError("sample trial_number must be positive")
        require_full_hash(
            self.candidate_combination_identity_hash,
            field="candidate_combination_identity_hash",
        )
        require_full_hash(
            self.evaluated_candidate_identity_hash,
            field="evaluated_candidate_identity_hash",
        )
        if (
            self.evaluation.candidate_identity_hash
            != self.evaluated_candidate_identity_hash
        ):
            raise ValueError(
                "sample evidence does not match evaluated candidate"
            )
        if self.evaluation.purpose != "miprov2_sample":
            raise ValueError("sample evidence has the wrong purpose")
        if (
            self.candidate_assembly.candidate.identity_hash
            != self.evaluated_candidate_identity_hash
        ):
            raise ValueError(
                "sample candidate does not match its assembly binding"
            )
        _require_finite(self.score, field="score")
        if self.score != self.evaluation.normalized_score:
            raise ValueError("sample score does not match verified evaluation")
        return self


class StudyTranscript(_IdentityRecord):
    """Complete information required to reconstruct and verify the study."""

    _identity_schema = MIPROV2_STUDY_SCHEMA

    schema_name: Literal["whetstone.miprov2_study_transcript"] = (
        MIPROV2_STUDY_SCHEMA
    )
    schema_version: Literal[1] = MIPROV2_STUDY_SCHEMA_VERSION
    algorithm_version: Literal["dspy_miprov2_prompt_program/v1"] = (
        MIPROV2_ALGORITHM_VERSION
    )
    reference_commit: Literal["6f68dcdb3ef46d70bf0c12596699ebc44e82d6b0"] = (
        MIPROV2_REFERENCE_COMMIT
    )
    optuna_version: Literal["4.8.0"] = OPTUNA_VERSION
    seed: StrictInt
    run_id: StrictStr
    validation_task_identities: tuple[StrictStr, ...]
    validation_eval_source: EvalConfigRef
    reward_policy_hash: StrictStr
    control_identity_hash: StrictStr
    prompt_adapter_identity_hash: StrictStr
    expected_base_candidate: CandidateRef
    program_layout: Miprov2ProgramLayout
    instruction_pool_identity_hashes: tuple[tuple[StrictStr, ...], ...]
    demo_pool_identity_hashes: tuple[tuple[StrictStr, ...], ...] | None = None
    parameter_space_identity_hash: StrictStr
    distribution_identity_hash: StrictStr
    schedule: Miprov2StudySchedule
    baseline: BaselineObservation
    samples: tuple[SampleObservation, ...] = ()

    @model_validator(mode="after")
    def _validate_contract(self) -> StudyTranscript:
        space = self.parameter_space
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if not self.validation_task_identities:
            raise ValueError("validation_task_identities must not be empty")
        for index, identity_hash in enumerate(self.validation_task_identities):
            require_full_hash(
                identity_hash,
                field=f"validation_task_identities[{index}]",
            )
        if len(set(self.validation_task_identities)) != len(
            self.validation_task_identities
        ):
            raise ValueError("validation task identities must be unique")
        if len(self.validation_task_identities) != self.schedule.valset_size:
            raise ValueError(
                "validation task identities do not match persisted valset size"
            )
        require_full_hash(
            self.reward_policy_hash,
            field="reward_policy_hash",
        )
        require_full_hash(
            self.control_identity_hash,
            field="control_identity_hash",
        )
        require_full_hash(
            self.prompt_adapter_identity_hash,
            field="prompt_adapter_identity_hash",
        )
        require_full_hash(
            self.parameter_space_identity_hash,
            field="parameter_space_identity_hash",
        )
        require_full_hash(
            self.distribution_identity_hash,
            field="distribution_identity_hash",
        )
        if self.parameter_space_identity_hash != space.identity_hash():
            raise ValueError(
                "parameter_space_identity_hash does not match exact pools"
            )
        if self.distribution_identity_hash != (
            space.distribution_identity_hash()
        ):
            raise ValueError(
                "distribution_identity_hash does not match exact space"
            )
        if len(self.program_layout.component_specs) != len(
            space.instruction_pool_identity_hashes
        ):
            raise ValueError(
                "program topology does not match the frozen search space"
            )
        if any(
            spec.prompt_format_identity_hash
            != self.prompt_adapter_identity_hash
            for spec in self.program_layout.component_specs
        ):
            raise ValueError(
                "program topology conflicts with the prompt adapter"
            )
        if self.baseline.categorical_combination_identity_hash != (
            space.combination_identity_hash(space.baseline_params)
        ):
            raise ValueError(
                "baseline categorical identity does not match all-zero params"
            )
        if (
            self.baseline.evaluated_base_candidate
            != self.expected_base_candidate
        ):
            raise ValueError(
                "baseline candidate does not match the expected native base"
            )
        self._validate_evaluation_binding(
            self.baseline.evaluation,
            expected_purpose="miprov2_baseline",
            expected_tasks=self.validation_task_identities,
        )
        if len(self.samples) > self.schedule.num_trials:
            raise ValueError("sample count exceeds persisted num_trials")
        seen_intent_ids = {self.baseline.evaluation.intent_id}
        for sample in self.samples:
            try:
                normalized = space.normalize(sample.params)
            except ValueError as exc:
                raise ValueError(
                    "sample parameters do not match the frozen search space"
                ) from exc
            expected_candidate = space.combination_identity_hash(normalized)
            if (
                sample.candidate_combination_identity_hash
                != expected_candidate
            ):
                raise ValueError(
                    "sample candidate identity does not match parameters"
                )
            self._validate_candidate_assembly(
                sample.candidate_assembly,
                expected_params=normalized,
                expected_combination=expected_candidate,
            )
            batch_size = len(sample.evaluation.task_batch_identities)
            expected_batch_size = (
                self.schedule.minibatch_size
                if self.schedule.minibatch
                else self.schedule.valset_size
            )
            if batch_size != expected_batch_size:
                raise ValueError(
                    "sample task batch does not match persisted schedule"
                )
            if sample.batch_full_evaluation != (
                batch_size >= self.schedule.valset_size
            ):
                raise ValueError(
                    "batch_full_evaluation does not match task batch size"
                )
            if sample.batch_full_evaluation:
                expected_tasks = self.validation_task_identities
            else:
                valid_subset = len(
                    set(sample.evaluation.task_batch_identities)
                ) == batch_size and set(
                    sample.evaluation.task_batch_identities
                ).issubset(self.validation_task_identities)
                if not valid_subset:
                    raise ValueError(
                        "minibatch tasks must be a unique ordered subset "
                        "of the validation set"
                    )
                expected_tasks = sample.evaluation.task_batch_identities
            self._validate_evaluation_binding(
                sample.evaluation,
                expected_purpose="miprov2_sample",
                expected_tasks=expected_tasks,
            )
            if sample.evaluation.intent_id in seen_intent_ids:
                raise ValueError("evaluation intent IDs must be unique")
            seen_intent_ids.add(sample.evaluation.intent_id)
            promotion_due = self.schedule.promotion_due(
                optuna_trial_number=sample.trial_number
            )
            if promotion_due != (sample.promotion is not None):
                raise ValueError(
                    "promotion presence does not match persisted cadence"
                )
            promotion = sample.promotion
            if promotion is not None:
                self._validate_evaluation_binding(
                    promotion.evaluation,
                    expected_purpose="miprov2_promotion",
                    expected_tasks=self.validation_task_identities,
                )
                if promotion.evaluation.intent_id in seen_intent_ids:
                    raise ValueError("evaluation intent IDs must be unique")
                seen_intent_ids.add(promotion.evaluation.intent_id)
                expected_promotion_candidate = space.combination_identity_hash(
                    promotion.params
                )
                if (
                    promotion.candidate_combination_identity_hash
                    != expected_promotion_candidate
                ):
                    raise ValueError(
                        "promotion candidate identity does not match "
                        "parameters"
                    )
                self._validate_candidate_assembly(
                    promotion.candidate_assembly,
                    expected_params=promotion.params,
                    expected_combination=expected_promotion_candidate,
                )
        return self

    def _validate_candidate_assembly(
        self,
        assembly: Miprov2CandidateAssemblyBinding,
        *,
        expected_params: TrialParams,
        expected_combination: str,
    ) -> None:
        _require_candidate_assembly(
            assembly,
            space=self.parameter_space,
            expected_params=expected_params,
            expected_combination=expected_combination,
            control_identity_hash=self.control_identity_hash,
            expected_base_candidate=self.expected_base_candidate,
            prompt_adapter_identity_hash=self.prompt_adapter_identity_hash,
            program_layout=self.program_layout,
        )

    def _validate_evaluation_binding(
        self,
        binding: EvaluationBinding,
        *,
        expected_purpose: EvaluationPurpose,
        expected_tasks: tuple[str, ...],
    ) -> None:
        if binding.run_id != self.run_id:
            raise ValueError("evaluation binding belongs to another run")
        if binding.purpose != expected_purpose:
            raise ValueError("evaluation binding has the wrong purpose")
        if binding.task_batch_identities != expected_tasks:
            raise ValueError(
                "evaluation task order does not match the persisted "
                "study contract"
            )
        derivation = binding.eval_config_binding
        request = derivation.request
        expected_derivation_purpose = expected_purpose.removeprefix("miprov2_")
        if (
            request.control_identity_hash != self.control_identity_hash
            or request.source_eval_config != self.validation_eval_source
            or request.purpose != expected_derivation_purpose
            or request.effect_identity_hash != binding.effect_identity_hash
            or request.task_batch_identities != expected_tasks
            or request.repeat_count != 1
        ):
            raise ValueError(
                "evaluation derivation does not match the persisted "
                "study contract"
            )
        if binding.reward_policy_hash != self.reward_policy_hash:
            raise ValueError(
                "evaluation reward policy does not match the persisted "
                "study contract"
            )

    @property
    def parameter_space(self) -> Miprov2ParameterSpace:
        return Miprov2ParameterSpace(
            instruction_pool_identity_hashes=(
                self.instruction_pool_identity_hashes
            ),
            demo_pool_identity_hashes=self.demo_pool_identity_hashes,
        )


def _require_candidate_assembly(
    assembly: Miprov2CandidateAssemblyBinding,
    *,
    space: Miprov2ParameterSpace,
    expected_params: TrialParams,
    expected_combination: str,
    control_identity_hash: str,
    expected_base_candidate: CandidateRef,
    prompt_adapter_identity_hash: str,
    program_layout: Miprov2ProgramLayout,
) -> None:
    """Verify one rendered program against its exact durable study inputs."""

    if assembly.params != expected_params:
        raise ValueError(
            "candidate assembly parameters do not match the observation"
        )
    if assembly.categorical_combination_identity_hash != expected_combination:
        raise ValueError(
            "candidate assembly does not match the categorical combination"
        )
    expected_context = (
        control_identity_hash,
        expected_base_candidate.identity_hash,
        prompt_adapter_identity_hash,
    )
    actual_context = (
        assembly.control_identity_hash,
        assembly.base_candidate.identity_hash,
        assembly.prompt_adapter_identity_hash,
    )
    if actual_context != expected_context:
        raise ValueError(
            "candidate assembly context does not match the study contract"
        )
    if assembly.base_candidate != expected_base_candidate:
        raise ValueError(
            "candidate assembly does not bind the exact native base candidate"
        )
    if assembly.program_layout != program_layout:
        raise ValueError(
            "candidate assembly does not bind the exact program topology"
        )
    if (
        assembly.candidate.record.base_ref
        != expected_base_candidate.record.base_ref
    ):
        raise ValueError(
            "assembled candidate does not preserve the native base reference"
        )
    rendering = assembly.candidate.record.payload[
        "miprov2_candidate_rendering"
    ]
    components = rendering["components"]
    values = dict(expected_params)
    expected_payload = dict(expected_base_candidate.record.payload)
    prompt_components: list[PromptProgramComponent] = []
    for predictor_index, (spec, component) in enumerate(
        zip(program_layout.component_specs, components, strict=True)
    ):
        if (
            component["component_id"] != spec.component_id
            or component["candidate_field"] != spec.candidate_field
        ):
            raise ValueError(
                "candidate rendering differs from the bound program topology"
            )
        instruction_index = values[f"{predictor_index}_predictor_instruction"]
        expected_instruction_identity = space.instruction_pool_identity_hashes[
            predictor_index
        ][instruction_index]
        if (
            component["instruction_identity_hash"]
            != expected_instruction_identity
        ):
            raise ValueError(
                "assembled instruction does not match the frozen pool"
            )
        instruction = cast("str", component["instruction"])
        expected_payload[spec.candidate_field] = instruction
        examples: tuple[PromptProgramExample, ...] = ()
        demo_pools = space.demo_pool_identity_hashes
        if demo_pools is not None:
            demo_index = values[f"{predictor_index}_predictor_demos"]
            if (
                component["demo_identity_hash"]
                != demo_pools[predictor_index][demo_index]
            ):
                raise ValueError(
                    "assembled demonstrations do not match the frozen pool"
                )
            demo_set = ComponentDemoSet.model_validate(component["demo_set"])
            examples = tuple(
                PromptProgramExample(
                    inputs=demo.inputs,
                    outputs=demo.outputs,
                )
                for demo in demo_set.demos_for(spec.component_id)
            )
        prompt_components.append(
            PromptProgramComponent(
                component_id=spec.component_id,
                candidate_field=spec.candidate_field,
                examples=examples,
            )
        )
    expected_payload["miprov2_candidate_rendering"] = rendering
    expected_payload[PROMPT_PROGRAM_PAYLOAD_FIELD] = PromptProgram(
        components=tuple(prompt_components)
    ).model_dump(mode="json")
    expected_candidate = candidate_reference(
        Candidate(
            candidate_id=f"miprov2-{assembly.program_identity_hash[:24]}",
            base_ref=expected_base_candidate.record.base_ref,
            payload=expected_payload,
        )
    )
    if assembly.candidate != expected_candidate:
        raise ValueError(
            "assembled candidate differs from canonical native rendering"
        )


class StudySuggestion(_IdentityRecord):
    """The next replay-stable categorical suggestion."""

    _identity_schema = "whetstone.miprov2_study_suggestion"

    trial_number: StrictInt
    params: TrialParams
    candidate_combination_identity_hash: StrictStr

    @model_validator(mode="after")
    def _validate_suggestion(self) -> StudySuggestion:
        if self.trial_number < 1:
            raise ValueError("suggestion trial_number must be positive")
        require_full_hash(
            self.candidate_combination_identity_hash,
            field="candidate_combination_identity_hash",
        )
        return self


class PromotionCandidate(_IdentityRecord):
    """The highest-mean combination not previously promoted."""

    _identity_schema = "whetstone.miprov2_promotion_candidate"

    params: TrialParams
    candidate_combination_identity_hash: StrictStr
    evaluated_candidate_identity_hash: StrictStr
    candidate_assembly: Miprov2CandidateAssemblyBinding
    source_sample_trial_number: StrictInt
    minibatch_mean: float

    @model_validator(mode="after")
    def _validate_candidate(self) -> PromotionCandidate:
        require_full_hash(
            self.candidate_combination_identity_hash,
            field="candidate_combination_identity_hash",
        )
        require_full_hash(
            self.evaluated_candidate_identity_hash,
            field="evaluated_candidate_identity_hash",
        )
        if (
            self.candidate_assembly.candidate.identity_hash
            != self.evaluated_candidate_identity_hash
        ):
            raise ValueError(
                "promotion candidate does not match its assembly binding"
            )
        if self.source_sample_trial_number < 1:
            raise ValueError("source_sample_trial_number must be positive")
        _require_finite(self.minibatch_mean, field="minibatch_mean")
        return self


class FullEvaluation(_IdentityRecord):
    """A candidate eligible for DSPy's strict winner update."""

    _identity_schema = "whetstone.miprov2_full_evaluation"

    source: Literal["baseline", "sample", "promotion"]
    trial_number: StrictInt
    params: TrialParams
    candidate_combination_identity_hash: StrictStr
    evaluated_candidate_identity_hash: StrictStr
    score: float

    @model_validator(mode="after")
    def _validate_full_evaluation(self) -> FullEvaluation:
        if self.trial_number < 0:
            raise ValueError("trial_number cannot be negative")
        require_full_hash(
            self.candidate_combination_identity_hash,
            field="candidate_combination_identity_hash",
        )
        require_full_hash(
            self.evaluated_candidate_identity_hash,
            field="evaluated_candidate_identity_hash",
        )
        _require_finite(self.score, field="score")
        return self


def select_promotion(
    samples: Sequence[SampleObservation],
) -> PromotionCandidate:
    """Match DSPy's stable highest-average combination selection."""

    scores: OrderedDict[str, list[float]] = OrderedDict()
    first_observation: dict[str, SampleObservation] = {}
    promoted: set[str] = set()
    for sample in samples:
        key = sample.candidate_combination_identity_hash
        scores.setdefault(key, []).append(sample.score)
        first_observation.setdefault(key, sample)
        if sample.promotion is not None:
            promoted.add(sample.promotion.candidate_combination_identity_hash)

    ranked = sorted(
        (
            PromotionCandidate(
                params=first_observation[key].params,
                candidate_combination_identity_hash=key,
                evaluated_candidate_identity_hash=(
                    first_observation[key].evaluated_candidate_identity_hash
                ),
                candidate_assembly=first_observation[key].candidate_assembly,
                source_sample_trial_number=first_observation[key].trial_number,
                minibatch_mean=sum(values) / len(values),
            )
            for key, values in scores.items()
        ),
        key=lambda candidate: candidate.minibatch_mean,
        reverse=True,
    )
    for candidate in ranked:
        if candidate.candidate_combination_identity_hash not in promoted:
            return candidate
    raise ValueError("No valid program found in param_score_dict")


class Miprov2Study:
    """Replay-only ownership seam around frozen Optuna 4.8.0."""

    def __init__(
        self,
        *,
        seed: int,
        space: Miprov2ParameterSpace,
        schedule: Miprov2StudySchedule,
        run_id: str,
        validation_task_identities: tuple[str, ...],
        validation_eval_source: EvalConfigRef,
        reward_policy_hash: str,
        control_identity_hash: str,
        prompt_adapter_identity_hash: str,
        expected_base_candidate: CandidateRef,
        program_layout: Miprov2ProgramLayout,
    ) -> None:
        self.seed = seed
        self.space = space
        self.schedule = schedule
        self.run_id = run_id
        self.validation_task_identities = validation_task_identities
        self.validation_eval_source = validation_eval_source
        self.reward_policy_hash = reward_policy_hash
        self.control_identity_hash = control_identity_hash
        self.prompt_adapter_identity_hash = prompt_adapter_identity_hash
        self.expected_base_candidate = expected_base_candidate
        self.program_layout = program_layout

    def initial_transcript(
        self,
        *,
        baseline_score: float,
        baseline_evaluation: EvaluationBinding,
    ) -> StudyTranscript:
        """Create the bound trial-zero durability record."""

        baseline = BaselineObservation(
            categorical_combination_identity_hash=(
                self.space.combination_identity_hash(
                    self.space.baseline_params
                )
            ),
            evaluated_base_candidate=self.expected_base_candidate,
            score=baseline_score,
            evaluation=baseline_evaluation,
        )
        return StudyTranscript(
            seed=self.seed,
            run_id=self.run_id,
            validation_task_identities=self.validation_task_identities,
            validation_eval_source=self.validation_eval_source,
            reward_policy_hash=self.reward_policy_hash,
            control_identity_hash=self.control_identity_hash,
            prompt_adapter_identity_hash=self.prompt_adapter_identity_hash,
            expected_base_candidate=self.expected_base_candidate,
            program_layout=self.program_layout,
            instruction_pool_identity_hashes=(
                self.space.instruction_pool_identity_hashes
            ),
            demo_pool_identity_hashes=self.space.demo_pool_identity_hashes,
            parameter_space_identity_hash=self.space.identity_hash(),
            distribution_identity_hash=(
                self.space.distribution_identity_hash()
            ),
            schedule=self.schedule,
            baseline=baseline,
        )

    def suggest_next(self, transcript: StudyTranscript) -> StudySuggestion:
        self._require_bound_transcript(transcript)
        if len(transcript.samples) >= self.schedule.num_trials:
            raise ValueError("MIPROv2 sampled trial schedule is exhausted")
        study = self._reconstruct(transcript)
        trial = study.ask()
        params = self._suggest(trial)
        return StudySuggestion(
            trial_number=trial.number,
            params=params,
            candidate_combination_identity_hash=(
                self.space.combination_identity_hash(params)
            ),
        )

    def promotion_candidate(
        self,
        transcript: StudyTranscript,
        suggestion: StudySuggestion,
        *,
        score: float,
        evaluation: EvaluationBinding,
        candidate_assembly: Miprov2CandidateAssemblyBinding,
    ) -> PromotionCandidate | None:
        """Return the exact candidate requiring a promotion effect, if due."""

        self._validate_next_suggestion(transcript, suggestion)
        provisional = self._provisional_sample(
            suggestion=suggestion,
            score=score,
            evaluation=evaluation,
            candidate_assembly=candidate_assembly,
        )
        if not self.schedule.promotion_due(
            optuna_trial_number=suggestion.trial_number
        ):
            return None
        return select_promotion((*transcript.samples, provisional))

    def record_sample(
        self,
        transcript: StudyTranscript,
        suggestion: StudySuggestion,
        *,
        score: float,
        evaluation: EvaluationBinding,
        candidate_assembly: Miprov2CandidateAssemblyBinding,
        promotion_full_score: float | None = None,
        promotion_evaluation: EvaluationBinding | None = None,
    ) -> StudyTranscript:
        """Append a sample using DSPy's promotion-before-tell ordering."""

        self._validate_next_suggestion(transcript, suggestion)
        study = self._reconstruct(transcript)
        trial = study.ask()
        reproduced = self._suggestion(trial)
        if reproduced != suggestion:
            raise StudyTranscriptMismatch(
                "stored suggestion does not match reconstructed Optuna trial"
            )

        provisional = self._provisional_sample(
            suggestion=suggestion,
            score=score,
            evaluation=evaluation,
            candidate_assembly=candidate_assembly,
        )
        promotion_due = self.schedule.promotion_due(
            optuna_trial_number=suggestion.trial_number
        )
        has_promotion_result = (
            promotion_full_score is not None
            or promotion_evaluation is not None
        )
        if promotion_due and (
            promotion_full_score is None or promotion_evaluation is None
        ):
            raise ValueError(
                "a promotion evaluation is required at this trial"
            )
        if not promotion_due and has_promotion_result:
            raise ValueError("promotion evaluation supplied off cadence")

        promotion: Promotion | None = None
        if promotion_due:
            assert promotion_full_score is not None
            assert promotion_evaluation is not None
            if (
                len(promotion_evaluation.task_batch_identities)
                != self.schedule.valset_size
            ):
                raise ValueError(
                    "promotion task batch must cover validation set"
                )
            selected = select_promotion((*transcript.samples, provisional))
            self._validate_evaluation_binding(
                promotion_evaluation,
                expected_purpose="miprov2_promotion",
                expected_tasks=self.validation_task_identities,
            )
            if (
                promotion_evaluation.candidate_identity_hash
                != selected.evaluated_candidate_identity_hash
            ):
                raise ValueError(
                    "promotion evaluation does not match the selected "
                    "first-observed candidate"
                )
            promoted_trial = self._add_completed_trial(
                study,
                params=selected.params,
                value=promotion_full_score,
            )
            promotion = Promotion(
                trial_number=promoted_trial.number,
                params=selected.params,
                candidate_combination_identity_hash=(
                    selected.candidate_combination_identity_hash
                ),
                evaluated_candidate_identity_hash=(
                    selected.evaluated_candidate_identity_hash
                ),
                candidate_assembly=selected.candidate_assembly,
                source_sample_trial_number=(
                    selected.source_sample_trial_number
                ),
                minibatch_mean=selected.minibatch_mean,
                full_score=promotion_full_score,
                evaluation=promotion_evaluation,
            )

        # The sampled objective is told only after an optional promotion trial.
        study.tell(trial, score)
        completed = provisional.model_copy(update={"promotion": promotion})
        return StudyTranscript.model_validate(
            {
                **transcript.model_dump(mode="json"),
                "samples": [
                    *[
                        sample.model_dump(mode="json")
                        for sample in transcript.samples
                    ],
                    completed.model_dump(mode="json"),
                ],
            }
        )

    def reconstruct_study(self, transcript: StudyTranscript) -> Any:
        """Return a verified in-memory study, primarily for diagnostics."""

        self._require_bound_transcript(transcript)
        return self._reconstruct(transcript)

    def best_full_evaluation(
        self,
        transcript: StudyTranscript,
    ) -> FullEvaluation:
        """Return the best eligible full result, retaining the first tie."""

        self._require_bound_transcript(transcript)
        best = FullEvaluation(
            source="baseline",
            trial_number=0,
            params=self.space.baseline_params,
            candidate_combination_identity_hash=(
                transcript.baseline.categorical_combination_identity_hash
            ),
            evaluated_candidate_identity_hash=(
                transcript.baseline.evaluated_base_candidate.identity_hash
            ),
            score=transcript.baseline.score,
        )
        for sample in transcript.samples:
            # Batch-size full_eval classification does not make a minibatch
            # objective winner-eligible, including size == validation size.
            if not self.schedule.minibatch and sample.score > best.score:
                best = FullEvaluation(
                    source="sample",
                    trial_number=sample.trial_number,
                    params=sample.params,
                    candidate_combination_identity_hash=(
                        sample.candidate_combination_identity_hash
                    ),
                    evaluated_candidate_identity_hash=(
                        sample.evaluated_candidate_identity_hash
                    ),
                    score=sample.score,
                )
            promotion = sample.promotion
            if promotion is not None and promotion.full_score > best.score:
                best = FullEvaluation(
                    source="promotion",
                    trial_number=promotion.trial_number,
                    params=promotion.params,
                    candidate_combination_identity_hash=(
                        promotion.candidate_combination_identity_hash
                    ),
                    evaluated_candidate_identity_hash=(
                        promotion.evaluated_candidate_identity_hash
                    ),
                    score=promotion.full_score,
                )
        return best

    def _validate_next_suggestion(
        self,
        transcript: StudyTranscript,
        suggestion: StudySuggestion,
    ) -> None:
        self._require_bound_transcript(transcript)
        if len(transcript.samples) >= self.schedule.num_trials:
            raise ValueError("MIPROv2 sampled trial schedule is exhausted")
        expected = self.suggest_next(transcript)
        if expected != suggestion:
            raise StudyTranscriptMismatch(
                "stored suggestion does not match reconstructed Optuna trial"
            )

    def _provisional_sample(
        self,
        *,
        suggestion: StudySuggestion,
        score: float,
        evaluation: EvaluationBinding,
        candidate_assembly: Miprov2CandidateAssemblyBinding,
    ) -> SampleObservation:
        _require_candidate_assembly(
            candidate_assembly,
            space=self.space,
            expected_params=suggestion.params,
            expected_combination=(
                suggestion.candidate_combination_identity_hash
            ),
            control_identity_hash=self.control_identity_hash,
            expected_base_candidate=self.expected_base_candidate,
            prompt_adapter_identity_hash=self.prompt_adapter_identity_hash,
            program_layout=self.program_layout,
        )
        expected_batch_size = (
            self.schedule.minibatch_size
            if self.schedule.minibatch
            else self.schedule.valset_size
        )
        if len(evaluation.task_batch_identities) != expected_batch_size:
            raise ValueError(
                "sample task batch does not match persisted schedule"
            )
        expected_tasks = evaluation.task_batch_identities
        if len(expected_tasks) >= self.schedule.valset_size:
            expected_tasks = self.validation_task_identities
        elif len(set(expected_tasks)) != len(expected_tasks) or not set(
            expected_tasks
        ).issubset(self.validation_task_identities):
            raise ValueError(
                "sample tasks must be a unique ordered validation subset"
            )
        self._validate_evaluation_binding(
            evaluation,
            expected_purpose="miprov2_sample",
            expected_tasks=expected_tasks,
        )
        return SampleObservation(
            trial_number=suggestion.trial_number,
            params=suggestion.params,
            candidate_combination_identity_hash=(
                suggestion.candidate_combination_identity_hash
            ),
            evaluated_candidate_identity_hash=(
                candidate_assembly.candidate.identity_hash
            ),
            candidate_assembly=candidate_assembly,
            score=score,
            evaluation=evaluation,
            batch_full_evaluation=(
                len(evaluation.task_batch_identities)
                >= self.schedule.valset_size
            ),
        )

    def _validate_evaluation_binding(
        self,
        binding: EvaluationBinding,
        *,
        expected_purpose: EvaluationPurpose,
        expected_tasks: tuple[str, ...],
    ) -> None:
        if binding.run_id != self.run_id:
            raise ValueError("evaluation binding belongs to another run")
        if binding.purpose != expected_purpose:
            raise ValueError("evaluation binding has the wrong purpose")
        if binding.task_batch_identities != expected_tasks:
            raise ValueError(
                "evaluation task order does not match the persisted "
                "study contract"
            )
        request = binding.eval_config_binding.request
        if (
            request.control_identity_hash != self.control_identity_hash
            or request.source_eval_config != self.validation_eval_source
            or request.purpose != expected_purpose.removeprefix("miprov2_")
            or request.effect_identity_hash != binding.effect_identity_hash
            or request.task_batch_identities != expected_tasks
            or request.repeat_count != 1
        ):
            raise ValueError(
                "evaluation derivation does not match the persisted "
                "study contract"
            )
        if binding.reward_policy_hash != self.reward_policy_hash:
            raise ValueError(
                "evaluation reward policy does not match the persisted "
                "study contract"
            )

    def _require_bound_transcript(self, transcript: StudyTranscript) -> None:
        try:
            validated = StudyTranscript.model_validate(
                transcript.model_dump(mode="json")
            )
        except ValidationError as exc:
            raise StudyTranscriptMismatch(
                "transcript fails its identity and evidence contract"
            ) from exc
        if validated != transcript:
            raise StudyTranscriptMismatch(
                "transcript changes under canonical validation"
            )
        expected = (
            self.seed,
            self.run_id,
            self.validation_task_identities,
            self.validation_eval_source,
            self.reward_policy_hash,
            self.control_identity_hash,
            self.prompt_adapter_identity_hash,
            self.expected_base_candidate,
            self.program_layout,
            self.space.instruction_pool_identity_hashes,
            self.space.demo_pool_identity_hashes,
            self.space.identity_hash(),
            self.space.distribution_identity_hash(),
            self.schedule,
        )
        actual = (
            transcript.seed,
            transcript.run_id,
            transcript.validation_task_identities,
            transcript.validation_eval_source,
            transcript.reward_policy_hash,
            transcript.control_identity_hash,
            transcript.prompt_adapter_identity_hash,
            transcript.expected_base_candidate,
            transcript.program_layout,
            transcript.instruction_pool_identity_hashes,
            transcript.demo_pool_identity_hashes,
            transcript.parameter_space_identity_hash,
            transcript.distribution_identity_hash,
            transcript.schedule,
        )
        if actual != expected:
            raise StudyTranscriptMismatch(
                "transcript does not match the bound MIPROv2 study contract"
            )
        expected_trial_number = 1
        replayed: list[SampleObservation] = []
        for sample in transcript.samples:
            if sample.trial_number != expected_trial_number:
                raise StudyTranscriptMismatch(
                    "sample trial chronology differs from promotion schedule"
                )
            promotion_due = self.schedule.promotion_due(
                optuna_trial_number=sample.trial_number
            )
            if promotion_due != (sample.promotion is not None):
                raise StudyTranscriptMismatch(
                    "promotion presence differs from persisted cadence"
                )
            provisional = sample.model_copy(update={"promotion": None})
            if sample.promotion is not None:
                selected = select_promotion((*replayed, provisional))
                promotion = sample.promotion
                if promotion.trial_number != sample.trial_number + 1:
                    raise StudyTranscriptMismatch(
                        "promotion trial chronology differs from "
                        "promotion-before-tell ordering"
                    )
                if (
                    promotion.params != selected.params
                    or promotion.candidate_combination_identity_hash
                    != selected.candidate_combination_identity_hash
                    or promotion.source_sample_trial_number
                    != selected.source_sample_trial_number
                    or promotion.minibatch_mean != selected.minibatch_mean
                    or promotion.evaluated_candidate_identity_hash
                    != selected.evaluated_candidate_identity_hash
                    or promotion.candidate_assembly
                    != selected.candidate_assembly
                ):
                    raise StudyTranscriptMismatch(
                        "promotion does not match stable mean ranking"
                    )
                expected_trial_number += 1
            replayed.append(sample)
            expected_trial_number += 1

    def _reconstruct(self, transcript: StudyTranscript) -> Any:
        optuna = _import_optuna()
        sampler = optuna.samplers.TPESampler(
            seed=self.seed,
            multivariate=True,
        )
        study = optuna.create_study(direction="maximize", sampler=sampler)
        baseline = optuna.trial.create_trial(
            params=self.space.as_dict(self.space.baseline_params),
            distributions=self._distributions(optuna),
            value=transcript.baseline.score,
        )
        study.add_trial(baseline)
        if study.trials[0].number != 0:
            raise StudyTranscriptMismatch(
                "Optuna did not assign the baseline to trial zero"
            )

        replayed: list[SampleObservation] = []
        for recorded in transcript.samples:
            trial = study.ask()
            params = self._suggest(trial)
            if trial.number != recorded.trial_number:
                raise StudyTranscriptMismatch(
                    "sample trial number differs during reconstruction"
                )
            try:
                expected_params = self.space.normalize(recorded.params)
            except ValueError as exc:
                raise StudyTranscriptMismatch(
                    "sample parameters do not match the frozen search space"
                ) from exc
            if params != expected_params:
                raise StudyTranscriptMismatch(
                    "sample parameters differ during reconstruction"
                )
            expected_combination = self.space.combination_identity_hash(params)
            if (
                recorded.candidate_combination_identity_hash
                != expected_combination
            ):
                raise StudyTranscriptMismatch(
                    "sample candidate identity differs during reconstruction"
                )

            provisional = recorded.model_copy(update={"promotion": None})
            promotion_due = self.schedule.promotion_due(
                optuna_trial_number=trial.number
            )
            if promotion_due != (recorded.promotion is not None):
                raise StudyTranscriptMismatch(
                    "promotion presence differs from persisted cadence"
                )
            if recorded.promotion is not None:
                selected = select_promotion((*replayed, provisional))
                promoted = recorded.promotion
                if (
                    promoted.params != selected.params
                    or promoted.candidate_combination_identity_hash
                    != selected.candidate_combination_identity_hash
                    or promoted.source_sample_trial_number
                    != selected.source_sample_trial_number
                    or promoted.minibatch_mean != selected.minibatch_mean
                    or promoted.evaluated_candidate_identity_hash
                    != selected.evaluated_candidate_identity_hash
                    or promoted.candidate_assembly
                    != selected.candidate_assembly
                ):
                    raise StudyTranscriptMismatch(
                        "promotion does not match stable mean ranking"
                    )
                added = self._add_completed_trial(
                    study,
                    params=promoted.params,
                    value=promoted.full_score,
                )
                if added.number != promoted.trial_number:
                    raise StudyTranscriptMismatch(
                        "promotion trial number differs during reconstruction"
                    )

            study.tell(trial, recorded.score)
            replayed.append(recorded)
        return study

    def _suggestion(self, trial: Any) -> StudySuggestion:
        params = self._suggest(trial)
        return StudySuggestion(
            trial_number=trial.number,
            params=params,
            candidate_combination_identity_hash=(
                self.space.combination_identity_hash(params)
            ),
        )

    def _suggest(self, trial: Any) -> TrialParams:
        params: list[tuple[str, int]] = []
        for predictor_index, instruction_count in enumerate(
            self.space.instruction_candidate_counts
        ):
            instruction_name = f"{predictor_index}_predictor_instruction"
            instruction = trial.suggest_categorical(
                instruction_name,
                range(instruction_count),
            )
            params.append((instruction_name, instruction))
            demos = self.space.demo_candidate_counts
            if demos is not None:
                demo_name = f"{predictor_index}_predictor_demos"
                demo = trial.suggest_categorical(
                    demo_name,
                    range(demos[predictor_index]),
                )
                params.append((demo_name, demo))
        return self.space.normalize(tuple(params))

    def _distributions(self, optuna: Any) -> dict[str, Any]:
        categorical = optuna.distributions.CategoricalDistribution
        return {
            name: categorical(range(self.space.candidate_count(name)))
            for name in self.space.parameter_names
        }

    def _add_completed_trial(
        self,
        study: Any,
        *,
        params: TrialParams,
        value: float,
    ) -> Any:
        optuna = _import_optuna()
        frozen = optuna.trial.create_trial(
            params=self.space.as_dict(params),
            distributions=self._distributions(optuna),
            value=value,
        )
        study.add_trial(frozen)
        return study.trials[-1]


def _require_finite(value: float, *, field: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{field} must be finite")
    return value


def _import_optuna() -> Any:
    try:
        optuna = importlib.import_module("optuna")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "MIPROv2 requires the frozen optuna==4.8.0 dependency"
        ) from exc
    version = getattr(optuna, "__version__", None)
    if version != OPTUNA_VERSION:
        raise RuntimeError(
            f"MIPROv2 requires optuna=={OPTUNA_VERSION}; found {version!r}"
        )
    return optuna


__all__ = [
    "EVALUATION_EVIDENCE_SCHEMA",
    "EVALUATION_FAILURE_SCHEMA",
    "MIPROV2_ALGORITHM_VERSION",
    "MIPROV2_CANDIDATE_ASSEMBLY_SCHEMA",
    "MIPROV2_CANDIDATE_ASSEMBLY_SCHEMA_VERSION",
    "MIPROV2_CANDIDATE_RENDERING_SCHEMA",
    "MIPROV2_CANDIDATE_RENDERING_SCHEMA_VERSION",
    "MIPROV2_REFERENCE_COMMIT",
    "MIPROV2_STUDY_SCHEMA",
    "MIPROV2_STUDY_SCHEMA_VERSION",
    "OPTUNA_VERSION",
    "REWARD_SCHEMA",
    "BaselineObservation",
    "EvaluationBinding",
    "FullEvaluation",
    "Miprov2CandidateAssemblyBinding",
    "Miprov2ParameterSpace",
    "Miprov2Study",
    "Miprov2StudySchedule",
    "Promotion",
    "PromotionCandidate",
    "SampleObservation",
    "StudySuggestion",
    "StudyTranscript",
    "StudyTranscriptMismatch",
    "TrialParams",
    "VerifiedEvaluationCitation",
    "select_promotion",
]
