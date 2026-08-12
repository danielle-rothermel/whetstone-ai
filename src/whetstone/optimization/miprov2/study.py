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

from whetstone.core.identity import (
    IdentityRef,
    TypedRef,
    compute_identity_hash,
    require_full_hash,
)
from whetstone.evaluation.schema_names import (
    EVALUATION_EVIDENCE_SCHEMA as _EVALUATION_EVIDENCE_SCHEMA,
)
from whetstone.evaluation.schema_names import (
    EVALUATION_FAILURE_SCHEMA as _EVALUATION_FAILURE_SCHEMA,
)
from whetstone.experiment.binding import (
    EvalConfigRef,
    EvaluationBinding,
)
from whetstone.experiment.candidate import (
    CandidateRef,
    candidate_reference,
)
from whetstone.experiment.reward import RewardRef
from whetstone.optimization.contracts import OptimizationRunRef
from whetstone.optimization.miprov2.control import (
    MIPROV2_ALGORITHM_VERSION,
    MIPROV2_CANDIDATE_RENDERER_VERSION,
    MIPROV2_OPTUNA_VERSION,
    MIPROV2_REFERENCE_COMMIT,
    Miprov2ProgramLayout,
)
from whetstone.optimization.miprov2.demo import ComponentDemoSet
from whetstone.optimization.miprov2.eval_config import (
    Miprov2EvalConfigBinding,
)
from whetstone.optimization.miprov2.render import candidate_from_components
from whetstone.optimization.proposal.mutation import diff_check

MIPROV2_STUDY_SCHEMA = "whetstone.miprov2_study_transcript"
MIPROV2_STUDY_SCHEMA_VERSION = 5
OPTUNA_VERSION = MIPROV2_OPTUNA_VERSION
MIPROV2_CANDIDATE_ASSEMBLY_SCHEMA = "whetstone.miprov2_candidate_assembly"
MIPROV2_CANDIDATE_ASSEMBLY_SCHEMA_VERSION = 4
MIPROV2_CANDIDATE_RENDERING_SCHEMA = "whetstone.miprov2_candidate_rendering"
MIPROV2_CANDIDATE_RENDERING_SCHEMA_VERSION = 1
MIPROV2_CANDIDATE_PROGRAM_SCHEMA = "whetstone.miprov2_candidate_program"
MIPROV2_CANDIDATE_PROGRAM_SCHEMA_VERSION = 1
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
    pass


class _IdentityRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    _identity_schema: ClassVar[str]
    _identity_schema_version: ClassVar[int] = 1

    def identity_payload(self) -> dict[str, Any]:
        raise NotImplementedError

    def identity_hash(self) -> str:
        return compute_identity_hash(
            schema=self._identity_schema,
            schema_version=self._identity_schema_version,
            payload=self.identity_payload(),
        )


class Miprov2ParameterSpace(_IdentityRecord):
    _identity_schema = "whetstone.miprov2_parameter_space"

    instruction_pool_identity_hashes: tuple[tuple[StrictStr, ...], ...]
    demo_pool_identity_hashes: tuple[tuple[StrictStr, ...], ...] | None = None

    def identity_payload(self) -> dict[str, Any]:
        return {
            "instruction_pool_identity_hashes": [
                list(pool) for pool in self.instruction_pool_identity_hashes
            ],
            "demo_pool_identity_hashes": (
                None
                if self.demo_pool_identity_hashes is None
                else [list(pool) for pool in self.demo_pool_identity_hashes]
            ),
        }

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
    _identity_schema = "whetstone.miprov2_study_schedule"

    num_trials: StrictInt
    minibatch: StrictBool
    minibatch_size: StrictInt
    valset_size: StrictInt
    minibatch_full_eval_steps: StrictInt

    def identity_payload(self) -> dict[str, Any]:
        return {
            "num_trials": self.num_trials,
            "minibatch": self.minibatch,
            "minibatch_size": self.minibatch_size,
            "valset_size": self.valset_size,
            "minibatch_full_eval_steps": self.minibatch_full_eval_steps,
        }

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


class Miprov2EvaluationObservation(_IdentityRecord):
    _identity_schema = "whetstone.miprov2_evaluation_observation"

    run_id: StrictStr
    intent_id: StrictStr
    effect_identity_hash: StrictStr
    purpose: EvaluationPurpose
    candidate: CandidateRef
    task_batch_hashes: tuple[StrictStr, ...]
    eval_config: EvalConfigRef
    eval_config_binding: Miprov2EvalConfigBinding
    evaluation_binding: EvaluationBinding
    evaluation_result_ref: TypedRef
    expected_reward_policy_hash: StrictStr
    reward_ref: RewardRef | None
    normalized_score: float

    def identity_payload(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "intent_id": self.intent_id,
            "effect_identity_hash": self.effect_identity_hash,
            "purpose": self.purpose,
            "candidate": self.candidate.model_dump(mode="json"),
            "task_batch_hashes": list(self.task_batch_hashes),
            "eval_config": self.eval_config.model_dump(mode="json"),
            "eval_config_binding": self.eval_config_binding.model_dump(
                mode="json"
            ),
            "evaluation_binding": self.evaluation_binding.model_dump(
                mode="json"
            ),
            "evaluation_result_ref": self.evaluation_result_ref.model_dump(
                mode="json"
            ),
            "expected_reward_policy_hash": (self.expected_reward_policy_hash),
            "reward_ref": (
                None
                if self.reward_ref is None
                else self.reward_ref.model_dump(mode="json")
            ),
            "normalized_score": self.normalized_score,
        }

    @model_validator(mode="after")
    def _validate_evidence(self) -> Miprov2EvaluationObservation:
        CandidateRef.model_validate(self.candidate.model_dump(mode="json"))
        EvaluationBinding.model_validate(
            self.evaluation_binding.model_dump(mode="json")
        )
        Miprov2EvalConfigBinding.model_validate(
            self.eval_config_binding.model_dump(mode="json")
        )
        if self.reward_ref is not None:
            RewardRef.model_validate(self.reward_ref.model_dump(mode="json"))
        if not self.run_id or not self.intent_id:
            raise ValueError(
                "evaluation observation run_id and intent_id are required"
            )
        require_full_hash(
            self.effect_identity_hash,
            field="effect_identity_hash",
        )
        require_full_hash(
            self.expected_reward_policy_hash,
            field="expected_reward_policy_hash",
        )
        if not self.task_batch_hashes:
            raise ValueError("task_batch_hashes must not be empty")
        for index, identity_hash in enumerate(self.task_batch_hashes):
            require_full_hash(
                identity_hash,
                field=f"task_batch_hashes[{index}]",
            )
        if self.eval_config != self.eval_config_binding.eval_config:
            raise ValueError(
                "evaluation Eval Config differs from its derivation binding"
            )
        if self.evaluation_binding.eval_config != self.eval_config:
            raise ValueError(
                "canonical Evaluation Binding differs from exact Eval Config"
            )
        if (
            self.evaluation_result_ref.schema_name
            == _EVALUATION_EVIDENCE_SCHEMA
        ):
            if self.reward_ref is None:
                raise ValueError(
                    "measured observation requires exact RewardRef"
                )
            if (
                self.reward_ref.record.reward_policy_hash
                != self.expected_reward_policy_hash
            ):
                raise ValueError("observation Reward uses another policy")
            if (
                round(self.reward_ref.record.value * 100, 2)
                != self.normalized_score
            ):
                raise ValueError("observation score differs from exact Reward")
        elif (
            self.evaluation_result_ref.schema_name
            == _EVALUATION_FAILURE_SCHEMA
        ):
            if self.reward_ref is not None or self.normalized_score != 0.0:
                raise ValueError(
                    "failed observation has zero score and no Reward"
                )
        else:
            raise ValueError(
                "observation has the wrong Evaluation Result schema"
            )
        _require_finite(self.normalized_score, field="normalized_score")
        return self

    @property
    def reward_policy_hash(self) -> str:
        return self.expected_reward_policy_hash


class BaselineObservation(_IdentityRecord):
    _identity_schema = "whetstone.miprov2_baseline_observation"

    categorical_combination_identity_hash: StrictStr
    evaluated_base_candidate: CandidateRef
    score: float
    evaluation: Miprov2EvaluationObservation

    def identity_payload(self) -> dict[str, Any]:
        return {
            "categorical_combination_identity_hash": (
                self.categorical_combination_identity_hash
            ),
            "evaluated_base_candidate": (
                self.evaluated_base_candidate.model_dump(mode="json")
            ),
            "score": self.score,
            "evaluation": self.evaluation.model_dump(mode="json"),
        }

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
            self.evaluation.candidate.identity_hash
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


class Miprov2ComponentSelection(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    component_id: StrictStr
    instruction_index: StrictInt
    instruction: StrictStr
    instruction_identity_hash: StrictStr
    demo_index: StrictInt | None
    demo_set: ComponentDemoSet | None
    demo_identity_hash: StrictStr | None

    @model_validator(mode="after")
    def _validate_selection(self) -> Miprov2ComponentSelection:
        if not self.component_id:
            raise ValueError("component_id must be non-empty")
        if self.instruction_index < 0:
            raise ValueError("instruction_index cannot be negative")
        expected_instruction = compute_identity_hash(
            schema=MIPROV2_INSTRUCTION_SCHEMA,
            schema_version=1,
            payload={"instruction": self.instruction},
        )
        if self.instruction_identity_hash != expected_instruction:
            raise ValueError("instruction identity does not match its text")
        if (self.demo_index is None) != (self.demo_set is None):
            raise ValueError("demo index and structured demo must be paired")
        if (self.demo_set is None) != (self.demo_identity_hash is None):
            raise ValueError("structured demo and identity must be paired")
        if self.demo_index is not None and self.demo_index < 0:
            raise ValueError("demo_index cannot be negative")
        if self.demo_set is not None:
            if self.demo_identity_hash != self.demo_set.identity_hash():
                raise ValueError(
                    "demo identity does not match structured demo"
                )
            if tuple(
                sequence.component_id for sequence in self.demo_set.components
            ) != (self.component_id,):
                raise ValueError(
                    "structured demo must contain only its selected component"
                )
        return self


class Miprov2CandidateRendering(_IdentityRecord):
    _identity_schema = MIPROV2_CANDIDATE_RENDERING_SCHEMA
    _identity_schema_version = MIPROV2_CANDIDATE_RENDERING_SCHEMA_VERSION

    control_identity_hash: StrictStr
    base_candidate_identity_hash: StrictStr
    categorical_combination_identity_hash: StrictStr
    renderer_version: Literal["whetstone_native_prompt_components/v1"] = (
        MIPROV2_CANDIDATE_RENDERER_VERSION
    )
    components: tuple[Miprov2ComponentSelection, ...]

    def identity_payload(self) -> dict[str, Any]:
        return {
            "control_identity_hash": self.control_identity_hash,
            "base_candidate_identity_hash": self.base_candidate_identity_hash,
            "categorical_combination_identity_hash": (
                self.categorical_combination_identity_hash
            ),
            "renderer_version": self.renderer_version,
            "components": [
                component.model_dump(mode="json")
                for component in self.components
            ],
        }

    @model_validator(mode="after")
    def _validate_rendering(self) -> Miprov2CandidateRendering:
        for field in (
            "control_identity_hash",
            "base_candidate_identity_hash",
            "categorical_combination_identity_hash",
        ):
            require_full_hash(getattr(self, field), field=field)
        if len(self.components) != 1:
            raise ValueError(
                "candidate rendering requires exactly one component"
            )
        return self


def _require_run_authorities(
    run: OptimizationRunRef,
    *,
    optimizer_config: IdentityRef,
    reward_policy_hash: str | None = None,
) -> None:
    if run.record.optimizer_config != optimizer_config:
        raise ValueError(
            "optimization run optimizer_config does not bind the exact "
            "MIPROv2 control"
        )
    if reward_policy_hash is None:
        return
    require_full_hash(
        reward_policy_hash,
        field="reward_policy_hash",
    )
    run_reward_policy = run.record.reward_policy
    if (
        run_reward_policy is None
        or run_reward_policy.identity_hash() != reward_policy_hash
    ):
        raise ValueError(
            "optimization run reward policy does not bind the exact "
            "MIPROv2 reward policy"
        )


class Miprov2CandidateAssemblyBinding(_IdentityRecord):
    _identity_schema = MIPROV2_CANDIDATE_ASSEMBLY_SCHEMA
    _identity_schema_version = MIPROV2_CANDIDATE_ASSEMBLY_SCHEMA_VERSION

    params: TrialParams
    categorical_combination_identity_hash: StrictStr
    candidate: CandidateRef
    program_identity_hash: StrictStr
    rendering: Miprov2CandidateRendering
    optimizer_config: IdentityRef
    base_candidate: CandidateRef
    program_layout: Miprov2ProgramLayout
    prompt_adapter_identity_hash: StrictStr
    run: OptimizationRunRef

    def identity_payload(self) -> dict[str, Any]:
        return {
            "params": [[name, value] for name, value in self.params],
            "categorical_combination_identity_hash": (
                self.categorical_combination_identity_hash
            ),
            "candidate": self.candidate.model_dump(mode="json"),
            "program_identity_hash": self.program_identity_hash,
            "rendering": self.rendering.model_dump(mode="json"),
            "optimizer_config": self.optimizer_config.model_dump(mode="json"),
            "base_candidate": self.base_candidate.model_dump(mode="json"),
            "program_layout": self.program_layout.model_dump(mode="json"),
            "prompt_adapter_identity_hash": (
                self.prompt_adapter_identity_hash
            ),
            "run": self.run.model_dump(mode="json"),
        }

    @model_validator(mode="after")
    def _validate_assembly(self) -> Miprov2CandidateAssemblyBinding:
        for field in (
            "categorical_combination_identity_hash",
            "program_identity_hash",
            "prompt_adapter_identity_hash",
        ):
            require_full_hash(getattr(self, field), field=field)
        _require_run_authorities(
            self.run,
            optimizer_config=self.optimizer_config,
        )
        diff_check(
            base=self.base_candidate.record,
            proposed=self.candidate.record,
            run=self.run,
        )
        if self.candidate.record.base_ref != self.base_candidate.record_ref:
            raise ValueError("assembled candidate must bind its exact base")
        if (
            self.rendering.control_identity_hash
            != self.optimizer_config.record_hash
            or self.rendering.base_candidate_identity_hash
            != self.base_candidate.identity_hash
            or self.rendering.categorical_combination_identity_hash
            != self.categorical_combination_identity_hash
        ):
            raise ValueError(
                "candidate rendering context does not match assembly"
            )
        expected_candidate = candidate_from_components(
            base=self.base_candidate,
            candidate_id=f"miprov2-{self.rendering.identity_hash()[:24]}",
            components=self.rendering.model_dump(mode="json")["components"],
            run=self.run,
        )
        if candidate_reference(expected_candidate) != self.candidate:
            raise ValueError(
                "assembled candidate differs from deterministic rendering"
            )
        expected_program_hash = compute_identity_hash(
            schema=MIPROV2_CANDIDATE_PROGRAM_SCHEMA,
            schema_version=MIPROV2_CANDIDATE_PROGRAM_SCHEMA_VERSION,
            payload={"candidate": self.candidate.model_dump(mode="json")},
        )
        if self.program_identity_hash != expected_program_hash:
            raise ValueError(
                "program identity does not match canonical candidate rendering"
            )
        return self


class Promotion(_IdentityRecord):
    _identity_schema = "whetstone.miprov2_promotion"

    trial_number: StrictInt
    params: TrialParams
    candidate_combination_identity_hash: StrictStr
    evaluated_candidate_identity_hash: StrictStr
    candidate_assembly: Miprov2CandidateAssemblyBinding
    source_sample_trial_number: StrictInt
    minibatch_mean: float
    full_score: float
    evaluation: Miprov2EvaluationObservation

    def identity_payload(self) -> dict[str, Any]:
        return {
            "trial_number": self.trial_number,
            "params": [[name, value] for name, value in self.params],
            "candidate_combination_identity_hash": (
                self.candidate_combination_identity_hash
            ),
            "evaluated_candidate_identity_hash": (
                self.evaluated_candidate_identity_hash
            ),
            "candidate_assembly": self.candidate_assembly.model_dump(
                mode="json"
            ),
            "source_sample_trial_number": self.source_sample_trial_number,
            "minibatch_mean": self.minibatch_mean,
            "full_score": self.full_score,
            "evaluation": self.evaluation.model_dump(mode="json"),
        }

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
            self.evaluation.candidate.identity_hash
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
    _identity_schema = "whetstone.miprov2_sample_observation"

    trial_number: StrictInt
    params: TrialParams
    candidate_combination_identity_hash: StrictStr
    evaluated_candidate_identity_hash: StrictStr
    candidate_assembly: Miprov2CandidateAssemblyBinding
    score: float
    evaluation: Miprov2EvaluationObservation
    batch_full_evaluation: StrictBool
    promotion: Promotion | None = None

    def identity_payload(self) -> dict[str, Any]:
        return {
            "trial_number": self.trial_number,
            "params": [[name, value] for name, value in self.params],
            "candidate_combination_identity_hash": (
                self.candidate_combination_identity_hash
            ),
            "evaluated_candidate_identity_hash": (
                self.evaluated_candidate_identity_hash
            ),
            "candidate_assembly": self.candidate_assembly.model_dump(
                mode="json"
            ),
            "score": self.score,
            "evaluation": self.evaluation.model_dump(mode="json"),
            "batch_full_evaluation": self.batch_full_evaluation,
            "promotion": (
                None
                if self.promotion is None
                else self.promotion.model_dump(mode="json")
            ),
        }

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
            self.evaluation.candidate.identity_hash
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
    _identity_schema = MIPROV2_STUDY_SCHEMA
    _identity_schema_version = MIPROV2_STUDY_SCHEMA_VERSION

    schema_name: Literal["whetstone.miprov2_study_transcript"] = (
        MIPROV2_STUDY_SCHEMA
    )
    schema_version: Literal[5] = MIPROV2_STUDY_SCHEMA_VERSION
    algorithm_version: Literal["dspy_miprov2/v2"] = MIPROV2_ALGORITHM_VERSION
    reference_commit: Literal["6f68dcdb3ef46d70bf0c12596699ebc44e82d6b0"] = (
        MIPROV2_REFERENCE_COMMIT
    )
    optuna_version: Literal["4.8.0"] = OPTUNA_VERSION
    seed: StrictInt
    run_id: StrictStr
    validation_task_hashes: tuple[StrictStr, ...]
    validation_eval_source: EvalConfigRef
    reward_policy_hash: StrictStr
    optimizer_config: IdentityRef
    prompt_adapter_identity_hash: StrictStr
    expected_base_candidate: CandidateRef
    program_layout: Miprov2ProgramLayout
    run: OptimizationRunRef
    instruction_pool_identity_hashes: tuple[tuple[StrictStr, ...], ...]
    demo_pool_identity_hashes: tuple[tuple[StrictStr, ...], ...] | None = None
    parameter_space_identity_hash: StrictStr
    distribution_identity_hash: StrictStr
    schedule: Miprov2StudySchedule
    baseline: BaselineObservation
    samples: tuple[SampleObservation, ...] = ()

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "algorithm_version": self.algorithm_version,
            "reference_commit": self.reference_commit,
            "optuna_version": self.optuna_version,
            "seed": self.seed,
            "run_id": self.run_id,
            "validation_task_hashes": list(self.validation_task_hashes),
            "validation_eval_source": self.validation_eval_source.model_dump(
                mode="json"
            ),
            "reward_policy_hash": self.reward_policy_hash,
            "optimizer_config": self.optimizer_config.model_dump(mode="json"),
            "prompt_adapter_identity_hash": (
                self.prompt_adapter_identity_hash
            ),
            "expected_base_candidate": (
                self.expected_base_candidate.model_dump(mode="json")
            ),
            "program_layout": self.program_layout.model_dump(mode="json"),
            "run": self.run.model_dump(mode="json"),
            "instruction_pool_identity_hashes": [
                list(pool) for pool in self.instruction_pool_identity_hashes
            ],
            "demo_pool_identity_hashes": (
                None
                if self.demo_pool_identity_hashes is None
                else [list(pool) for pool in self.demo_pool_identity_hashes]
            ),
            "parameter_space_identity_hash": (
                self.parameter_space_identity_hash
            ),
            "distribution_identity_hash": self.distribution_identity_hash,
            "schedule": self.schedule.model_dump(mode="json"),
            "baseline": self.baseline.model_dump(mode="json"),
            "samples": [
                sample.model_dump(mode="json") for sample in self.samples
            ],
        }

    @model_validator(mode="after")
    def _validate_contract(self) -> StudyTranscript:
        space = self.parameter_space
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if self.run.record.run_id != self.run_id:
            raise ValueError("study run_id conflicts with the exact run")
        if not self.validation_task_hashes:
            raise ValueError("validation_task_hashes must not be empty")
        for index, identity_hash in enumerate(self.validation_task_hashes):
            require_full_hash(
                identity_hash,
                field=f"validation_task_hashes[{index}]",
            )
        if len(set(self.validation_task_hashes)) != len(
            self.validation_task_hashes
        ):
            raise ValueError("validation task identities must be unique")
        if len(self.validation_task_hashes) != self.schedule.valset_size:
            raise ValueError(
                "validation task identities do not match persisted valset size"
            )
        require_full_hash(
            self.reward_policy_hash,
            field="reward_policy_hash",
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
        _require_run_authorities(
            self.run,
            optimizer_config=self.optimizer_config,
            reward_policy_hash=self.reward_policy_hash,
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
            expected_tasks=self.validation_task_hashes,
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
            batch_size = len(sample.evaluation.task_batch_hashes)
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
                expected_tasks = self.validation_task_hashes
            else:
                valid_subset = len(
                    set(sample.evaluation.task_batch_hashes)
                ) == batch_size and set(
                    sample.evaluation.task_batch_hashes
                ).issubset(self.validation_task_hashes)
                if not valid_subset:
                    raise ValueError(
                        "minibatch tasks must be a unique ordered subset "
                        "of the validation set"
                    )
                expected_tasks = sample.evaluation.task_batch_hashes
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
                    expected_tasks=self.validation_task_hashes,
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
            optimizer_config=self.optimizer_config,
            expected_base_candidate=self.expected_base_candidate,
            prompt_adapter_identity_hash=self.prompt_adapter_identity_hash,
            program_layout=self.program_layout,
            run=self.run,
        )

    def _validate_evaluation_binding(
        self,
        binding: Miprov2EvaluationObservation,
        *,
        expected_purpose: EvaluationPurpose,
        expected_tasks: tuple[str, ...],
    ) -> None:
        if binding.run_id != self.run_id:
            raise ValueError("evaluation binding belongs to another run")
        if binding.purpose != expected_purpose:
            raise ValueError("evaluation binding has the wrong purpose")
        if binding.task_batch_hashes != expected_tasks:
            raise ValueError(
                "evaluation task order does not match the persisted "
                "study contract"
            )
        derivation = binding.eval_config_binding
        request = derivation.request
        expected_derivation_purpose = expected_purpose.removeprefix("miprov2_")
        if (
            request.control_identity_hash != self.optimizer_config.record_hash
            or request.source_eval_config != self.validation_eval_source
            or request.purpose != expected_derivation_purpose
            or request.effect_identity_hash != binding.effect_identity_hash
            or request.task_batch_hashes != expected_tasks
            or request.num_samples != 1
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
    optimizer_config: IdentityRef,
    expected_base_candidate: CandidateRef,
    prompt_adapter_identity_hash: str,
    program_layout: Miprov2ProgramLayout,
    run: OptimizationRunRef,
) -> None:

    if assembly.params != expected_params:
        raise ValueError(
            "candidate assembly parameters do not match the observation"
        )
    if assembly.categorical_combination_identity_hash != expected_combination:
        raise ValueError(
            "candidate assembly does not match the categorical combination"
        )
    expected_context = (
        optimizer_config,
        expected_base_candidate.identity_hash,
        prompt_adapter_identity_hash,
    )
    actual_context = (
        assembly.optimizer_config,
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
    if assembly.run != run:
        raise ValueError(
            "candidate assembly does not bind the exact optimization run"
        )
    components = assembly.rendering.components
    if tuple(component.component_id for component in components) != tuple(
        spec.component_id for spec in program_layout.component_specs
    ):
        raise ValueError("candidate rendering does not match program topology")
    values = dict(expected_params)
    for index, component in enumerate(components):
        instruction_index = values[f"{index}_predictor_instruction"]
        if (
            component.instruction_index != instruction_index
            or space.instruction_pool_identity_hashes[index][instruction_index]
            != component.instruction_identity_hash
        ):
            raise ValueError(
                "candidate instruction differs from the selected category"
            )
        if space.demo_pool_identity_hashes is None:
            if component.demo_index is not None:
                raise ValueError("zeroshot candidate cannot select a demo")
        else:
            demo_index = values[f"{index}_predictor_demos"]
            if (
                component.demo_index != demo_index
                or space.demo_pool_identity_hashes[index][demo_index]
                != component.demo_identity_hash
            ):
                raise ValueError(
                    "candidate demo differs from the selected category"
                )
    if (
        assembly.candidate.record.base_ref
        != expected_base_candidate.record_ref
    ):
        raise ValueError(
            "assembled candidate does not bind the exact input CandidateRef"
        )
    diff_check(
        base=expected_base_candidate.record,
        proposed=assembly.candidate.record,
        run=run,
    )


class StudySuggestion(_IdentityRecord):
    _identity_schema = "whetstone.miprov2_study_suggestion"

    trial_number: StrictInt
    params: TrialParams
    candidate_combination_identity_hash: StrictStr

    def identity_payload(self) -> dict[str, Any]:
        return {
            "trial_number": self.trial_number,
            "params": [[name, value] for name, value in self.params],
            "candidate_combination_identity_hash": (
                self.candidate_combination_identity_hash
            ),
        }

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
    _identity_schema = "whetstone.miprov2_promotion_candidate"

    params: TrialParams
    candidate_combination_identity_hash: StrictStr
    evaluated_candidate_identity_hash: StrictStr
    candidate_assembly: Miprov2CandidateAssemblyBinding
    source_sample_trial_number: StrictInt
    minibatch_mean: float

    def identity_payload(self) -> dict[str, Any]:
        return {
            "params": [[name, value] for name, value in self.params],
            "candidate_combination_identity_hash": (
                self.candidate_combination_identity_hash
            ),
            "evaluated_candidate_identity_hash": (
                self.evaluated_candidate_identity_hash
            ),
            "candidate_assembly": self.candidate_assembly.model_dump(
                mode="json"
            ),
            "source_sample_trial_number": self.source_sample_trial_number,
            "minibatch_mean": self.minibatch_mean,
        }

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
    _identity_schema = "whetstone.miprov2_full_evaluation"

    source: Literal["baseline", "sample", "promotion"]
    trial_number: StrictInt
    params: TrialParams
    candidate_combination_identity_hash: StrictStr
    evaluated_candidate_identity_hash: StrictStr
    score: float

    def identity_payload(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "trial_number": self.trial_number,
            "params": [[name, value] for name, value in self.params],
            "candidate_combination_identity_hash": (
                self.candidate_combination_identity_hash
            ),
            "evaluated_candidate_identity_hash": (
                self.evaluated_candidate_identity_hash
            ),
            "score": self.score,
        }

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
    def __init__(
        self,
        *,
        seed: int,
        space: Miprov2ParameterSpace,
        schedule: Miprov2StudySchedule,
        run_id: str,
        validation_task_hashes: tuple[str, ...],
        validation_eval_source: EvalConfigRef,
        reward_policy_hash: str,
        optimizer_config: IdentityRef,
        prompt_adapter_identity_hash: str,
        expected_base_candidate: CandidateRef,
        program_layout: Miprov2ProgramLayout,
        run: OptimizationRunRef,
    ) -> None:
        self.seed = seed
        self.space = space
        self.schedule = schedule
        self.run_id = run_id
        self.validation_task_hashes = validation_task_hashes
        self.validation_eval_source = validation_eval_source
        self.reward_policy_hash = reward_policy_hash
        self.optimizer_config = optimizer_config
        self.prompt_adapter_identity_hash = prompt_adapter_identity_hash
        self.expected_base_candidate = expected_base_candidate
        self.program_layout = program_layout
        self.run = run
        if self.run.record.run_id != self.run_id:
            raise ValueError("study run_id conflicts with the exact run")
        _require_run_authorities(
            self.run,
            optimizer_config=self.optimizer_config,
            reward_policy_hash=self.reward_policy_hash,
        )

    def initial_transcript(
        self,
        *,
        baseline_score: float,
        baseline_evaluation: Miprov2EvaluationObservation,
    ) -> StudyTranscript:

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
            validation_task_hashes=self.validation_task_hashes,
            validation_eval_source=self.validation_eval_source,
            reward_policy_hash=self.reward_policy_hash,
            optimizer_config=self.optimizer_config,
            prompt_adapter_identity_hash=self.prompt_adapter_identity_hash,
            expected_base_candidate=self.expected_base_candidate,
            program_layout=self.program_layout,
            run=self.run,
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
        evaluation: Miprov2EvaluationObservation,
        candidate_assembly: Miprov2CandidateAssemblyBinding,
    ) -> PromotionCandidate | None:

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
        evaluation: Miprov2EvaluationObservation,
        candidate_assembly: Miprov2CandidateAssemblyBinding,
        promotion_full_score: float | None = None,
        promotion_evaluation: Miprov2EvaluationObservation | None = None,
    ) -> StudyTranscript:

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
                len(promotion_evaluation.task_batch_hashes)
                != self.schedule.valset_size
            ):
                raise ValueError(
                    "promotion task batch must cover validation set"
                )
            selected = select_promotion((*transcript.samples, provisional))
            self._validate_evaluation_binding(
                promotion_evaluation,
                expected_purpose="miprov2_promotion",
                expected_tasks=self.validation_task_hashes,
            )
            if (
                promotion_evaluation.candidate.identity_hash
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

        self._require_bound_transcript(transcript)
        return self._reconstruct(transcript)

    def best_full_evaluation(
        self,
        transcript: StudyTranscript,
    ) -> FullEvaluation:

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
        evaluation: Miprov2EvaluationObservation,
        candidate_assembly: Miprov2CandidateAssemblyBinding,
    ) -> SampleObservation:
        _require_candidate_assembly(
            candidate_assembly,
            space=self.space,
            expected_params=suggestion.params,
            expected_combination=(
                suggestion.candidate_combination_identity_hash
            ),
            optimizer_config=self.optimizer_config,
            expected_base_candidate=self.expected_base_candidate,
            prompt_adapter_identity_hash=self.prompt_adapter_identity_hash,
            program_layout=self.program_layout,
            run=self.run,
        )
        expected_batch_size = (
            self.schedule.minibatch_size
            if self.schedule.minibatch
            else self.schedule.valset_size
        )
        if len(evaluation.task_batch_hashes) != expected_batch_size:
            raise ValueError(
                "sample task batch does not match persisted schedule"
            )
        expected_tasks = evaluation.task_batch_hashes
        if len(expected_tasks) >= self.schedule.valset_size:
            expected_tasks = self.validation_task_hashes
        elif len(set(expected_tasks)) != len(expected_tasks) or not set(
            expected_tasks
        ).issubset(self.validation_task_hashes):
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
                len(evaluation.task_batch_hashes) >= self.schedule.valset_size
            ),
        )

    def _validate_evaluation_binding(
        self,
        binding: Miprov2EvaluationObservation,
        *,
        expected_purpose: EvaluationPurpose,
        expected_tasks: tuple[str, ...],
    ) -> None:
        if binding.run_id != self.run_id:
            raise ValueError("evaluation binding belongs to another run")
        if binding.purpose != expected_purpose:
            raise ValueError("evaluation binding has the wrong purpose")
        if binding.task_batch_hashes != expected_tasks:
            raise ValueError(
                "evaluation task order does not match the persisted "
                "study contract"
            )
        request = binding.eval_config_binding.request
        if (
            request.control_identity_hash != self.optimizer_config.record_hash
            or request.source_eval_config != self.validation_eval_source
            or request.purpose != expected_purpose.removeprefix("miprov2_")
            or request.effect_identity_hash != binding.effect_identity_hash
            or request.task_batch_hashes != expected_tasks
            or request.num_samples != 1
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
            self.validation_task_hashes,
            self.validation_eval_source,
            self.reward_policy_hash,
            self.optimizer_config,
            self.prompt_adapter_identity_hash,
            self.expected_base_candidate,
            self.program_layout,
            self.run,
            self.space.instruction_pool_identity_hashes,
            self.space.demo_pool_identity_hashes,
            self.space.identity_hash(),
            self.space.distribution_identity_hash(),
            self.schedule,
        )
        actual = (
            transcript.seed,
            transcript.run_id,
            transcript.validation_task_hashes,
            transcript.validation_eval_source,
            transcript.reward_policy_hash,
            transcript.optimizer_config,
            transcript.prompt_adapter_identity_hash,
            transcript.expected_base_candidate,
            transcript.program_layout,
            transcript.run,
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
    "MIPROV2_ALGORITHM_VERSION",
    "MIPROV2_CANDIDATE_ASSEMBLY_SCHEMA",
    "MIPROV2_CANDIDATE_ASSEMBLY_SCHEMA_VERSION",
    "MIPROV2_CANDIDATE_PROGRAM_SCHEMA",
    "MIPROV2_CANDIDATE_PROGRAM_SCHEMA_VERSION",
    "MIPROV2_CANDIDATE_RENDERING_SCHEMA",
    "MIPROV2_CANDIDATE_RENDERING_SCHEMA_VERSION",
    "MIPROV2_REFERENCE_COMMIT",
    "MIPROV2_STUDY_SCHEMA",
    "MIPROV2_STUDY_SCHEMA_VERSION",
    "OPTUNA_VERSION",
    "BaselineObservation",
    "FullEvaluation",
    "Miprov2CandidateAssemblyBinding",
    "Miprov2CandidateRendering",
    "Miprov2ComponentSelection",
    "Miprov2EvaluationObservation",
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
    "select_promotion",
]
