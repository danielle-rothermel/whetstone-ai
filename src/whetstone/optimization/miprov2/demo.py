from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Any, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    model_validator,
)

from whetstone.core.identity import (
    FiniteFloat,
    ImmutableJsonObject,
    compute_identity_hash,
    require_full_hash,
)

MIPROV2_ACCEPTANCE_SCHEMA = "whetstone.miprov2_bootstrap_acceptance"
MIPROV2_COMPONENT_DEMO_SCHEMA = "whetstone.miprov2_component_demo"
MIPROV2_COMPONENT_DEMO_SET_SCHEMA = "whetstone.miprov2_component_demo_set"
MIPROV2_DEMO_SCHEMA_VERSION = 1

type MetricValue = StrictBool | FiniteFloat


def _component_fields(
    by_component: ImmutableJsonObject,
    component_id: str,
    *,
    field: str,
) -> dict[str, Any]:
    """Return one component's field mapping as a fresh strict-JSON object."""

    try:
        fields = by_component[component_id]
    except KeyError:
        raise ValueError(
            f"{field} has no component {component_id!r}"
        ) from None
    if not isinstance(fields, ImmutableJsonObject):
        raise ValueError(f"{field}[{component_id!r}] must be a JSON object")
    return fields.to_json()


def _require_disjoint_fields(
    inputs: Mapping[str, Any],
    outputs: Mapping[str, Any],
) -> None:
    """Require input and output field names to name distinct prompt fields.

    Demonstration rendering merges inputs and outputs into one flat field
    mapping, so an overlapping name silently discards the input value.
    """

    duplicate_fields = set(inputs) & set(outputs)
    if duplicate_fields:
        raise ValueError(
            "demo input/output fields overlap: "
            f"{', '.join(sorted(duplicate_fields))}"
        )


class DemoSourceKind(StrEnum):
    """How a demonstration entered a component's prompt."""

    BOOTSTRAPPED = "bootstrapped"
    LABELED = "labeled"


class BootstrapAcceptance(BaseModel):
    """The exact metric decision attached to one bootstrap rollout.

    ``metric_threshold`` deliberately uses DSPy's truthiness rule.  In
    particular, ``0.0`` takes the truthiness branch rather than the numeric
    comparison branch, while positive *and negative* non-zero thresholds take
    the comparison branch.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_task_hash: StrictStr
    source_rollout_identity: StrictStr
    source_trace_identity: StrictStr
    source_output_identity: StrictStr
    source_score_identity: StrictStr
    metric_present: StrictBool
    score: MetricValue | None
    metric_threshold: FiniteFloat | None
    accepted: StrictBool

    @model_validator(mode="after")
    def _validate_decision(self) -> BootstrapAcceptance:
        for field in (
            "source_task_hash",
            "source_rollout_identity",
            "source_trace_identity",
            "source_output_identity",
            "source_score_identity",
        ):
            require_full_hash(getattr(self, field), field=field)
        if self.metric_present and self.score is None:
            raise ValueError("a present metric must have a score")
        if not self.metric_present and self.score is not None:
            raise ValueError("an absent metric cannot have a score")
        if self.accepted != bootstrap_accepts(
            metric_present=self.metric_present,
            score=self.score,
            metric_threshold=self.metric_threshold,
        ):
            raise ValueError("accepted does not match DSPy's threshold rule")
        return self

    def identity_payload(self) -> dict[str, Any]:
        return {
            "source_task_hash": self.source_task_hash,
            "source_rollout_identity": self.source_rollout_identity,
            "source_trace_identity": self.source_trace_identity,
            "source_output_identity": self.source_output_identity,
            "source_score_identity": self.source_score_identity,
            "metric_present": self.metric_present,
            "score": self.score,
            "metric_threshold": self.metric_threshold,
            "accepted": self.accepted,
        }

    def identity_hash(self) -> str:
        return compute_identity_hash(
            schema=MIPROV2_ACCEPTANCE_SCHEMA,
            schema_version=MIPROV2_DEMO_SCHEMA_VERSION,
            payload=self.identity_payload(),
        )


def bootstrap_accepts(
    *,
    metric_present: bool,
    score: bool | float | None,
    metric_threshold: float | None,
) -> bool:
    """Return the frozen DSPy bootstrap acceptance decision."""

    if not metric_present:
        return True
    if score is None:
        raise ValueError("a present metric must have a score")
    if metric_threshold:
        return bool(score >= metric_threshold)
    return bool(score)


class ObservedTraceStep(BaseModel):
    """One normalized trace entry emitted by the task-model executor.

    ``component_id=None`` represents a trace predictor that is not part of the
    student/teacher component mapping.  DSPy silently skips such entries.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    trace_index: StrictInt
    component_id: StrictStr | None
    inputs: ImmutableJsonObject = Field(
        default_factory=lambda: ImmutableJsonObject({})
    )
    outputs: ImmutableJsonObject = Field(
        default_factory=lambda: ImmutableJsonObject({})
    )

    def model_post_init(self, _context: Any) -> None:
        self._freeze_json_fields()

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
        copied._freeze_json_fields()
        return copied

    def _freeze_json_fields(self) -> None:
        for field in ("inputs", "outputs"):
            value = getattr(self, field)
            if not isinstance(value, ImmutableJsonObject):
                object.__setattr__(self, field, ImmutableJsonObject(value))

    @model_validator(mode="after")
    def _validate_step(self) -> ObservedTraceStep:
        if self.trace_index < 0:
            raise ValueError("trace_index cannot be negative")
        if self.component_id == "":
            raise ValueError("component_id must be non-empty when present")
        _require_disjoint_fields(self.inputs, self.outputs)
        return self

    def identity_payload(self) -> dict[str, Any]:
        return {
            "trace_index": self.trace_index,
            "component_id": self.component_id,
            "inputs": self.inputs.to_json(),
            "outputs": self.outputs.to_json(),
        }


class ComponentDemo(BaseModel):
    """One prompt-ready demonstration for one Whetstone component."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    component_id: StrictStr
    source_kind: DemoSourceKind
    inputs: ImmutableJsonObject = Field(
        default_factory=lambda: ImmutableJsonObject({})
    )
    outputs: ImmutableJsonObject = Field(
        default_factory=lambda: ImmutableJsonObject({})
    )
    augmented: StrictBool

    source_task_hash: StrictStr
    source_rollout_identity: StrictStr
    source_trace_identity: StrictStr
    source_output_identity: StrictStr
    source_score_identity: StrictStr
    source_trace_index: StrictInt | None
    score: MetricValue | None
    acceptance_identity_hash: StrictStr

    def model_post_init(self, _context: Any) -> None:
        self._freeze_json_fields()

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
        copied._freeze_json_fields()
        return copied

    def _freeze_json_fields(self) -> None:
        for field in ("inputs", "outputs"):
            value = getattr(self, field)
            if not isinstance(value, ImmutableJsonObject):
                object.__setattr__(self, field, ImmutableJsonObject(value))

    @model_validator(mode="after")
    def _validate_demo(self) -> ComponentDemo:
        if not self.component_id:
            raise ValueError("component_id must be non-empty")
        for field in (
            "source_task_hash",
            "source_rollout_identity",
            "source_trace_identity",
            "source_output_identity",
            "source_score_identity",
            "acceptance_identity_hash",
        ):
            require_full_hash(getattr(self, field), field=field)
        if self.source_trace_index is not None and self.source_trace_index < 0:
            raise ValueError("source_trace_index cannot be negative")
        if self.source_kind is DemoSourceKind.BOOTSTRAPPED:
            if not self.augmented:
                raise ValueError("bootstrapped demos must be augmented")
            if self.source_trace_index is None:
                raise ValueError(
                    "bootstrapped demos require a source trace index"
                )
        else:
            if self.augmented:
                raise ValueError("labeled demos cannot be augmented")
            if self.source_trace_index is not None:
                raise ValueError(
                    "labeled demos cannot have a source trace index"
                )
        _require_disjoint_fields(self.inputs, self.outputs)
        return self

    def identity_payload(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "source_kind": self.source_kind.value,
            "inputs": self.inputs.to_json(),
            "outputs": self.outputs.to_json(),
            "augmented": self.augmented,
            "source_task_hash": self.source_task_hash,
            "source_rollout_identity": self.source_rollout_identity,
            "source_trace_identity": self.source_trace_identity,
            "source_output_identity": self.source_output_identity,
            "source_score_identity": self.source_score_identity,
            "source_trace_index": self.source_trace_index,
            "score": self.score,
            "acceptance_identity_hash": self.acceptance_identity_hash,
        }

    def identity_hash(self) -> str:
        return compute_identity_hash(
            schema=MIPROV2_COMPONENT_DEMO_SCHEMA,
            schema_version=MIPROV2_DEMO_SCHEMA_VERSION,
            payload=self.identity_payload(),
        )


class LabeledTaskDemo(BaseModel):
    """A raw labeled task adapted into component-addressed prompt fields."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_task_hash: StrictStr
    inputs_by_component: ImmutableJsonObject
    outputs_by_component: ImmutableJsonObject

    def model_post_init(self, _context: Any) -> None:
        self._freeze_json_fields()

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
        copied._freeze_json_fields()
        return copied

    def _freeze_json_fields(self) -> None:
        for field in ("inputs_by_component", "outputs_by_component"):
            value = getattr(self, field)
            if not isinstance(value, ImmutableJsonObject):
                object.__setattr__(self, field, ImmutableJsonObject(value))

    @model_validator(mode="after")
    def _validate_task(self) -> LabeledTaskDemo:
        require_full_hash(self.source_task_hash, field="source_task_hash")
        if set(self.inputs_by_component) != set(self.outputs_by_component):
            raise ValueError(
                "labeled input and output component sets must match"
            )
        if any(not component for component in self.inputs_by_component):
            raise ValueError("labeled component ids must be non-empty")
        return self

    def inputs_for(self, component_id: str) -> dict[str, Any]:
        """Return one component's labeled input fields as fresh strict JSON."""

        return _component_fields(
            self.inputs_by_component, component_id, field="inputs_by_component"
        )

    def outputs_for(self, component_id: str) -> dict[str, Any]:
        """Return one component's labeled outputs as fresh strict JSON."""

        return _component_fields(
            self.outputs_by_component,
            component_id,
            field="outputs_by_component",
        )

    def for_component(self, component_id: str) -> ComponentDemo:
        """Adapt this task to one component without fake trace data."""

        if component_id not in self.inputs_by_component:
            raise ValueError(f"labeled task has no component {component_id!r}")
        inputs = self.inputs_for(component_id)
        outputs = self.outputs_for(component_id)
        source_identity = compute_identity_hash(
            schema="whetstone.miprov2_labeled_demo_source",
            schema_version=MIPROV2_DEMO_SCHEMA_VERSION,
            payload={
                "source_task_hash": self.source_task_hash,
                "component_id": component_id,
                "inputs": inputs,
                "outputs": outputs,
            },
        )
        acceptance_identity = compute_identity_hash(
            schema=MIPROV2_ACCEPTANCE_SCHEMA,
            schema_version=MIPROV2_DEMO_SCHEMA_VERSION,
            payload={
                "source_kind": DemoSourceKind.LABELED,
                "source_identity": source_identity,
                "accepted": True,
            },
        )
        return ComponentDemo(
            component_id=component_id,
            source_kind=DemoSourceKind.LABELED,
            inputs=inputs,
            outputs=outputs,
            augmented=False,
            source_task_hash=self.source_task_hash,
            source_rollout_identity=source_identity,
            source_trace_identity=source_identity,
            source_output_identity=source_identity,
            source_score_identity=source_identity,
            source_trace_index=None,
            score=None,
            acceptance_identity_hash=acceptance_identity,
        )


class ComponentDemoSequence(BaseModel):
    """Ordered demos for exactly one component."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    component_id: StrictStr
    demos: tuple[ComponentDemo, ...] = ()

    @model_validator(mode="after")
    def _validate_sequence(self) -> ComponentDemoSequence:
        if not self.component_id:
            raise ValueError("component_id must be non-empty")
        if any(demo.component_id != self.component_id for demo in self.demos):
            raise ValueError("all demos must belong to the sequence component")
        return self


class ComponentDemoSet(BaseModel):
    """A predictor-major, prompt-format-neutral MIPROv2 demo candidate."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_seed: StrictInt
    components: tuple[ComponentDemoSequence, ...]

    @model_validator(mode="after")
    def _validate_set(self) -> ComponentDemoSet:
        component_ids = [sequence.component_id for sequence in self.components]
        if not component_ids:
            raise ValueError("a demo set needs at least one component")
        if len(component_ids) != len(set(component_ids)):
            raise ValueError("demo set components must be unique")
        return self

    def identity_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def identity_hash(self) -> str:
        return compute_identity_hash(
            schema=MIPROV2_COMPONENT_DEMO_SET_SCHEMA,
            schema_version=MIPROV2_DEMO_SCHEMA_VERSION,
            payload=self.identity_payload(),
        )

    def demos_for(self, component_id: str) -> tuple[ComponentDemo, ...]:
        for sequence in self.components:
            if sequence.component_id == component_id:
                return sequence.demos
        raise KeyError(component_id)


def _json_object_at(
    value: ImmutableJsonObject,
    key: str,
) -> dict[str, Any]:
    nested = value[key]
    if not isinstance(nested, ImmutableJsonObject):
        raise ValueError(f"{key!r} must address a JSON object")
    return nested.to_json()


def proposal_demo_context(
    demo_candidates: tuple[ComponentDemoSet, ...],
    *,
    zeroshot_opt: bool,
) -> tuple[ComponentDemoSet, ...]:
    """Preserve proposal grounding even during zero-shot optimization.

    DSPy always passes the bootstrapped candidates into instruction proposal.
    ``zeroshot_opt`` is accepted here so the durable orchestrator can make the
    phase boundary explicit without changing proposal behavior.
    """

    del zeroshot_opt
    return demo_candidates


def study_demo_context(
    demo_candidates: tuple[ComponentDemoSet, ...],
    *,
    zeroshot_opt: bool,
) -> tuple[ComponentDemoSet, ...] | None:
    """Discard demonstrations only after instruction proposal in zero-shot."""

    return None if zeroshot_opt else demo_candidates


__all__ = [
    "MIPROV2_ACCEPTANCE_SCHEMA",
    "MIPROV2_COMPONENT_DEMO_SCHEMA",
    "MIPROV2_COMPONENT_DEMO_SET_SCHEMA",
    "BootstrapAcceptance",
    "ComponentDemo",
    "ComponentDemoSequence",
    "ComponentDemoSet",
    "DemoSourceKind",
    "LabeledTaskDemo",
    "MetricValue",
    "ObservedTraceStep",
    "bootstrap_accepts",
    "proposal_demo_context",
    "study_demo_context",
]
