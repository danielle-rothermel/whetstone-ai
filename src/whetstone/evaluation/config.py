"""Whetstone-owned evaluation definition and configuration contracts."""

from __future__ import annotations

import json
from typing import Self, cast

from dr_serialize import validate_strict_json
from pydantic import (
    BaseModel,
    ConfigDict,
    JsonValue,
    field_validator,
    model_validator,
)

from whetstone.core.identity import compute_identity_hash

SCHEMA_VERSION = 1

SCHEMA_SAMPLING_DEFINITION = "whetstone.sampling.definition"
SCHEMA_SAMPLING_CONFIG = "whetstone.sampling.config"
SCHEMA_PREPROCESSING_DEFINITION = "whetstone.preprocessing.definition"
SCHEMA_PREPROCESSING_CONFIG = "whetstone.preprocessing.config"
SCHEMA_METRIC_EXTRACTION_DEFINITION = "whetstone.metric_extraction.definition"
SCHEMA_METRIC_EXTRACTION_CONFIG = "whetstone.metric_extraction.config"
SCHEMA_EVALUATION_PROCEDURE_DEFINITION = (
    "whetstone.evaluation_procedure.definition"
)
SCHEMA_EVALUATION_PROCEDURE_CONFIG = "whetstone.evaluation_procedure.config"
SCHEMA_AGGREGATION_DEFINITION = "whetstone.aggregation.definition"
SCHEMA_AGGREGATION_CONFIG = "whetstone.aggregation.config"
SCHEMA_EVAL_DEFINITION = "whetstone.eval.definition"
SCHEMA_EVAL_CONFIG = "whetstone.eval.config"
SCHEMA_TASK_SET = "whetstone.task_set"
SCHEMA_SAMPLE_PLAN = "whetstone.sample_plan"
SCHEMA_SAMPLE_ID = "whetstone.sample_id"
SCHEMA_COMPRESSION_REFERENCE_KEY = "whetstone.compression_reference.key"
SCHEMA_COMPRESSION_REFERENCE_ARTIFACT = (
    "whetstone.compression_reference.artifact"
)


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


def identity_hash_for(
    *,
    schema: str,
    payload: object,
    schema_version: int = SCHEMA_VERSION,
) -> str:
    """Hash one complete finite identity-bearing payload."""

    return compute_identity_hash(
        schema=schema,
        schema_version=schema_version,
        payload=payload,
    )


class VariableError(ValueError):
    """A config assignment does not satisfy its definition."""


class VariableSpec(_FrozenModel):
    """One serialized variable declaration."""

    name: str
    allowed: tuple[JsonValue, ...] | None = None
    default: JsonValue | None = None
    has_default: bool = False

    @model_validator(mode="after")
    def validate_default(self) -> Self:
        if (
            self.has_default
            and self.allowed is not None
            and self.default not in self.allowed
        ):
            raise VariableError(
                f"default for {self.name!r} is not in its allowed values"
            )
        return self


def resolve_assignment(
    specs: tuple[VariableSpec, ...],
    assignment: dict[str, JsonValue],
) -> dict[str, JsonValue]:
    """Validate and complete an assignment in declaration order."""

    spec_by_name = {spec.name: spec for spec in specs}
    if len(spec_by_name) != len(specs):
        raise VariableError("variable names must be unique")
    unknown = set(assignment) - set(spec_by_name)
    if unknown:
        raise VariableError("unknown variables: " + ", ".join(sorted(unknown)))

    resolved: dict[str, JsonValue] = {}
    for spec in specs:
        if spec.name in assignment:
            value = assignment[spec.name]
        elif spec.has_default:
            value = spec.default
        else:
            raise VariableError(f"variable {spec.name!r} is unassigned")
        if spec.allowed is not None and value not in spec.allowed:
            raise VariableError(
                f"value for {spec.name!r} is not an allowed value"
            )
        resolved[spec.name] = value
    return resolved


class DefinitionRef(_FrozenModel):
    definition_id: str
    version: str
    schema_name: str
    identity_hash: str


def _definition_identity(
    *,
    schema: str,
    definition_id: str,
    version: str,
    variables: tuple[VariableSpec, ...],
    extra: dict[str, JsonValue],
) -> str:
    payload: dict[str, JsonValue] = {
        "definition_id": definition_id,
        "version": version,
        "variables": [
            {
                "name": spec.name,
                "allowed": (
                    None if spec.allowed is None else list(spec.allowed)
                ),
                "has_default": spec.has_default,
                "default": spec.default if spec.has_default else None,
            }
            for spec in variables
        ],
    }
    payload.update(extra)
    return identity_hash_for(schema=schema, payload=payload)


def _assignment_payload(
    assignment: dict[str, JsonValue],
) -> list[list[JsonValue]]:
    return [[name, value] for name, value in assignment.items()]


def _canonical_json_value(value: object) -> JsonValue:
    validated = cast(JsonValue, validate_strict_json(value))
    return _canonicalize_validated_json(validated)


def _canonicalize_validated_json(value: JsonValue) -> JsonValue:
    if isinstance(value, dict):
        return {
            name: _canonicalize_validated_json(value[name])
            for name in sorted(value)
        }
    if isinstance(value, list):
        return [_canonicalize_validated_json(item) for item in value]
    return value


class SamplingDefinition(_FrozenModel):
    definition_id: str
    version: str
    variables: tuple[VariableSpec, ...] = (
        VariableSpec(name="task_set_hash"),
        VariableSpec(name="sample_plan_hash"),
    )

    def identity_hash(self) -> str:
        return _definition_identity(
            schema=SCHEMA_SAMPLING_DEFINITION,
            definition_id=self.definition_id,
            version=self.version,
            variables=self.variables,
            extra={},
        )

    def ref(self) -> DefinitionRef:
        return DefinitionRef(
            definition_id=self.definition_id,
            version=self.version,
            schema_name=SCHEMA_SAMPLING_DEFINITION,
            identity_hash=self.identity_hash(),
        )

    def materialize(self, assignment: dict[str, JsonValue]) -> SamplingConfig:
        return SamplingConfig._create(
            definition=self,
            assignment=resolve_assignment(self.variables, assignment),
        )


class SamplingConfig(_FrozenModel):
    definition_ref: DefinitionRef
    assignment: tuple[tuple[str, JsonValue], ...]
    config_hash: str

    @classmethod
    def _create(
        cls,
        *,
        definition: SamplingDefinition,
        assignment: dict[str, JsonValue],
    ) -> SamplingConfig:
        config_hash = identity_hash_for(
            schema=SCHEMA_SAMPLING_CONFIG,
            payload={
                "definition_identity": definition.identity_hash(),
                "assignment": _assignment_payload(assignment),
            },
        )
        return cls(
            definition_ref=definition.ref(),
            assignment=tuple(assignment.items()),
            config_hash=config_hash,
        )

    def assignment_dict(self) -> dict[str, JsonValue]:
        return dict(self.assignment)


class PreprocessingStepBinding(_FrozenModel):
    instance_name: str
    step: str
    settings: tuple[tuple[str, JsonValue], ...] = ()

    @field_validator("settings", mode="before")
    @classmethod
    def normalize_settings(cls, value: object) -> object:
        if isinstance(value, dict):
            return tuple(value.items())
        return value

    @model_validator(mode="after")
    def reject_duplicate_settings(self) -> Self:
        names = [name for name, _value in self.settings]
        if len(names) != len(set(names)):
            raise ValueError("preprocessing setting names must be unique")
        return self


class PreprocessingDefinition(_FrozenModel):
    definition_id: str
    version: str
    steps: tuple[PreprocessingStepBinding, ...]
    variables: tuple[VariableSpec, ...] = ()

    @model_validator(mode="after")
    def reject_duplicate_tasks(self) -> Self:
        names = [binding.instance_name for binding in self.steps]
        if len(names) != len(set(names)):
            raise ValueError("preprocessing instance names must be unique")
        return self

    def identity_hash(self) -> str:
        return _definition_identity(
            schema=SCHEMA_PREPROCESSING_DEFINITION,
            definition_id=self.definition_id,
            version=self.version,
            variables=self.variables,
            extra={
                "steps": [
                    {
                        "instance_name": binding.instance_name,
                        "step": binding.step,
                        "settings": [list(pair) for pair in binding.settings],
                    }
                    for binding in self.steps
                ]
            },
        )

    def ref(self) -> DefinitionRef:
        return DefinitionRef(
            definition_id=self.definition_id,
            version=self.version,
            schema_name=SCHEMA_PREPROCESSING_DEFINITION,
            identity_hash=self.identity_hash(),
        )

    def materialize(
        self,
        assignment: dict[str, JsonValue] | None = None,
        *,
        resolved_steps: tuple[tuple[str, str, str], ...] = (),
    ) -> PreprocessingConfig:
        if len(resolved_steps) != len(self.steps):
            raise ValueError(
                "resolved preprocessing steps must match definition steps"
            )
        expected = tuple(
            (binding.instance_name, binding.step) for binding in self.steps
        )
        actual = tuple((name, step) for name, step, _version in resolved_steps)
        if actual != expected:
            raise ValueError(
                "resolved preprocessing steps must preserve definition order"
            )
        return PreprocessingConfig._create(
            definition=self,
            assignment=resolve_assignment(self.variables, assignment or {}),
            resolved_steps=resolved_steps,
        )


class PreprocessingConfig(_FrozenModel):
    definition_ref: DefinitionRef
    assignment: tuple[tuple[str, JsonValue], ...]
    resolved_step_versions: tuple[tuple[str, str, str], ...]
    config_hash: str

    @classmethod
    def _create(
        cls,
        *,
        definition: PreprocessingDefinition,
        assignment: dict[str, JsonValue],
        resolved_steps: tuple[tuple[str, str, str], ...],
    ) -> PreprocessingConfig:
        config_hash = identity_hash_for(
            schema=SCHEMA_PREPROCESSING_CONFIG,
            payload={
                "definition_identity": definition.identity_hash(),
                "assignment": _assignment_payload(assignment),
                "resolved_step_versions": [
                    list(item) for item in resolved_steps
                ],
            },
        )
        return cls(
            definition_ref=definition.ref(),
            assignment=tuple(assignment.items()),
            resolved_step_versions=resolved_steps,
            config_hash=config_hash,
        )


class MetricQuestionBinding(_FrozenModel):
    metric: str
    on: str
    settings: tuple[tuple[str, JsonValue], ...] = ()

    @field_validator("settings", mode="before")
    @classmethod
    def normalize_settings(cls, value: object) -> object:
        if isinstance(value, dict):
            canonical = _canonical_json_value(value)
            if not isinstance(canonical, dict):
                raise TypeError("metric settings must be a JSON object")
            return tuple(canonical.items())
        if isinstance(value, (list, tuple)):
            normalized: list[tuple[str, JsonValue]] = []
            for pair in value:
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    return value
                name, setting_value = pair
                if not isinstance(name, str):
                    return value
                normalized.append((name, _canonical_json_value(setting_value)))
            return tuple(sorted(normalized, key=lambda pair: pair[0]))
        return value

    @model_validator(mode="after")
    def reject_duplicate_settings(self) -> Self:
        names = [name for name, _value in self.settings]
        if len(names) != len(set(names)):
            raise ValueError("metric setting names must be unique")
        return self

    def settings_dict(self) -> dict[str, JsonValue]:
        return dict(self.settings)


class MetricExtractionDefinition(_FrozenModel):
    definition_id: str
    version: str
    questions: tuple[MetricQuestionBinding, ...]
    variables: tuple[VariableSpec, ...] = ()

    @model_validator(mode="after")
    def reject_duplicate_questions(self) -> Self:
        identities = [
            (
                question.metric,
                question.on,
                json.dumps(
                    list(question.settings),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
            for question in self.questions
        ]
        if len(identities) != len(set(identities)):
            raise ValueError(
                "metric questions must have unique "
                "(metric, on, settings) triples"
            )
        return self

    def identity_hash(self) -> str:
        return _definition_identity(
            schema=SCHEMA_METRIC_EXTRACTION_DEFINITION,
            definition_id=self.definition_id,
            version=self.version,
            variables=self.variables,
            extra={
                "questions": [
                    {
                        "metric": question.metric,
                        "on": question.on,
                        "settings": [list(pair) for pair in question.settings],
                    }
                    for question in self.questions
                ]
            },
        )

    def ref(self) -> DefinitionRef:
        return DefinitionRef(
            definition_id=self.definition_id,
            version=self.version,
            schema_name=SCHEMA_METRIC_EXTRACTION_DEFINITION,
            identity_hash=self.identity_hash(),
        )

    def materialize(
        self,
        assignment: dict[str, JsonValue] | None = None,
        *,
        resolved_operators: tuple[tuple[str, str], ...] = (),
    ) -> MetricExtractionConfig:
        if len(resolved_operators) != len(self.questions):
            raise ValueError(
                "resolved metric operators must match definition questions"
            )
        if tuple(name for name, _version in resolved_operators) != tuple(
            question.metric for question in self.questions
        ):
            raise ValueError(
                "resolved metric operators must preserve question order"
            )
        return MetricExtractionConfig._create(
            definition=self,
            assignment=resolve_assignment(self.variables, assignment or {}),
            resolved_operators=resolved_operators,
        )


class MetricExtractionConfig(_FrozenModel):
    definition_ref: DefinitionRef
    assignment: tuple[tuple[str, JsonValue], ...]
    resolved_operator_versions: tuple[tuple[str, str], ...]
    config_hash: str

    @classmethod
    def _create(
        cls,
        *,
        definition: MetricExtractionDefinition,
        assignment: dict[str, JsonValue],
        resolved_operators: tuple[tuple[str, str], ...],
    ) -> MetricExtractionConfig:
        config_hash = identity_hash_for(
            schema=SCHEMA_METRIC_EXTRACTION_CONFIG,
            payload={
                "definition_identity": definition.identity_hash(),
                "assignment": _assignment_payload(assignment),
                "resolved_operator_versions": [
                    list(item) for item in resolved_operators
                ],
            },
        )
        return cls(
            definition_ref=definition.ref(),
            assignment=tuple(assignment.items()),
            resolved_operator_versions=resolved_operators,
            config_hash=config_hash,
        )


class EvaluationProcedureDefinition(_FrozenModel):
    definition_id: str
    version: str
    variables: tuple[VariableSpec, ...] = (
        VariableSpec(
            name="zero_denominator",
            allowed=("not_applicable", "error"),
        ),
    )

    def identity_hash(self) -> str:
        return _definition_identity(
            schema=SCHEMA_EVALUATION_PROCEDURE_DEFINITION,
            definition_id=self.definition_id,
            version=self.version,
            variables=self.variables,
            extra={},
        )

    def ref(self) -> DefinitionRef:
        return DefinitionRef(
            definition_id=self.definition_id,
            version=self.version,
            schema_name=SCHEMA_EVALUATION_PROCEDURE_DEFINITION,
            identity_hash=self.identity_hash(),
        )

    def materialize(
        self,
        *,
        preprocessing: PreprocessingConfig,
        metric_extraction: MetricExtractionConfig,
        assignment: dict[str, JsonValue] | None = None,
    ) -> EvaluationProcedureConfig:
        return EvaluationProcedureConfig._create(
            definition=self,
            preprocessing=preprocessing,
            metric_extraction=metric_extraction,
            assignment=resolve_assignment(self.variables, assignment or {}),
        )


class EvaluationProcedureConfig(_FrozenModel):
    definition_ref: DefinitionRef
    preprocessing_config_hash: str
    metric_extraction_config_hash: str
    assignment: tuple[tuple[str, JsonValue], ...]
    config_hash: str

    @classmethod
    def _create(
        cls,
        *,
        definition: EvaluationProcedureDefinition,
        preprocessing: PreprocessingConfig,
        metric_extraction: MetricExtractionConfig,
        assignment: dict[str, JsonValue],
    ) -> EvaluationProcedureConfig:
        config_hash = identity_hash_for(
            schema=SCHEMA_EVALUATION_PROCEDURE_CONFIG,
            payload={
                "definition_identity": definition.identity_hash(),
                "preprocessing_config": preprocessing.config_hash,
                "metric_extraction_config": (metric_extraction.config_hash),
                "assignment": _assignment_payload(assignment),
            },
        )
        return cls(
            definition_ref=definition.ref(),
            preprocessing_config_hash=preprocessing.config_hash,
            metric_extraction_config_hash=(metric_extraction.config_hash),
            assignment=tuple(assignment.items()),
            config_hash=config_hash,
        )


class AggregationDefinition(_FrozenModel):
    definition_id: str
    version: str
    variables: tuple[VariableSpec, ...] = (
        VariableSpec(name="reduction", allowed=("mean", "sum")),
        VariableSpec(
            name="missing_data",
            allowed=("propagate", "skip"),
            default="propagate",
            has_default=True,
        ),
        VariableSpec(
            name="zero_denominator",
            allowed=("not_applicable", "error"),
            default="not_applicable",
            has_default=True,
        ),
    )

    def identity_hash(self) -> str:
        return _definition_identity(
            schema=SCHEMA_AGGREGATION_DEFINITION,
            definition_id=self.definition_id,
            version=self.version,
            variables=self.variables,
            extra={},
        )

    def ref(self) -> DefinitionRef:
        return DefinitionRef(
            definition_id=self.definition_id,
            version=self.version,
            schema_name=SCHEMA_AGGREGATION_DEFINITION,
            identity_hash=self.identity_hash(),
        )

    def materialize(
        self,
        assignment: dict[str, JsonValue],
    ) -> AggregationConfig:
        return AggregationConfig._create(
            definition=self,
            assignment=resolve_assignment(self.variables, assignment),
        )


class AggregationConfig(_FrozenModel):
    definition_ref: DefinitionRef
    assignment: tuple[tuple[str, JsonValue], ...]
    config_hash: str

    @classmethod
    def _create(
        cls,
        *,
        definition: AggregationDefinition,
        assignment: dict[str, JsonValue],
    ) -> AggregationConfig:
        config_hash = identity_hash_for(
            schema=SCHEMA_AGGREGATION_CONFIG,
            payload={
                "definition_identity": definition.identity_hash(),
                "assignment": _assignment_payload(assignment),
            },
        )
        return cls(
            definition_ref=definition.ref(),
            assignment=tuple(assignment.items()),
            config_hash=config_hash,
        )

    def assignment_dict(self) -> dict[str, JsonValue]:
        return dict(self.assignment)


class EvalDefinition(_FrozenModel):
    definition_id: str
    version: str
    variables: tuple[VariableSpec, ...] = (
        VariableSpec(name="sampling_config_hash"),
        VariableSpec(name="evaluation_procedure_config_hash"),
        VariableSpec(name="aggregation_config_hash"),
    )

    def identity_hash(self) -> str:
        return _definition_identity(
            schema=SCHEMA_EVAL_DEFINITION,
            definition_id=self.definition_id,
            version=self.version,
            variables=self.variables,
            extra={},
        )

    def ref(self) -> DefinitionRef:
        return DefinitionRef(
            definition_id=self.definition_id,
            version=self.version,
            schema_name=SCHEMA_EVAL_DEFINITION,
            identity_hash=self.identity_hash(),
        )

    def materialize(
        self,
        *,
        sampling: SamplingConfig,
        evaluation_procedure: EvaluationProcedureConfig,
        aggregation: AggregationConfig,
    ) -> EvalConfig:
        return EvalConfig._create(
            definition=self,
            sampling=sampling,
            evaluation_procedure=evaluation_procedure,
            aggregation=aggregation,
        )


class EvalConfig(_FrozenModel):
    definition_ref: DefinitionRef
    sampling_config_hash: str
    evaluation_procedure_config_hash: str
    aggregation_config_hash: str
    config_hash: str

    @classmethod
    def _create(
        cls,
        *,
        definition: EvalDefinition,
        sampling: SamplingConfig,
        evaluation_procedure: EvaluationProcedureConfig,
        aggregation: AggregationConfig,
    ) -> EvalConfig:
        config_hash = identity_hash_for(
            schema=SCHEMA_EVAL_CONFIG,
            payload={
                "definition_identity": definition.identity_hash(),
                "sampling_config": sampling.config_hash,
                "evaluation_procedure_config": (
                    evaluation_procedure.config_hash
                ),
                "aggregation_config": aggregation.config_hash,
            },
        )
        return cls(
            definition_ref=definition.ref(),
            sampling_config_hash=sampling.config_hash,
            evaluation_procedure_config_hash=(
                evaluation_procedure.config_hash
            ),
            aggregation_config_hash=aggregation.config_hash,
            config_hash=config_hash,
        )


__all__ = [
    "SCHEMA_AGGREGATION_CONFIG",
    "SCHEMA_EVALUATION_PROCEDURE_CONFIG",
    "SCHEMA_EVAL_CONFIG",
    "SCHEMA_SAMPLING_CONFIG",
    "AggregationConfig",
    "AggregationDefinition",
    "DefinitionRef",
    "EvalConfig",
    "EvalDefinition",
    "EvaluationProcedureConfig",
    "EvaluationProcedureDefinition",
    "MetricExtractionConfig",
    "MetricExtractionDefinition",
    "MetricQuestionBinding",
    "PreprocessingConfig",
    "PreprocessingDefinition",
    "PreprocessingStepBinding",
    "SamplingConfig",
    "SamplingDefinition",
    "VariableError",
    "VariableSpec",
    "identity_hash_for",
    "resolve_assignment",
]
