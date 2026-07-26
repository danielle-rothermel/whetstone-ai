"""Structured, identity-bearing demonstrations for MIPROv2.

DSPy stores demonstrations as mutable ``Example`` objects attached directly to
predictors.  Whetstone cannot persist those process objects.  This module is
the prompt-format adaptation seam: it keeps the same per-predictor ordering,
but represents every demonstration as immutable JSON with the evidence that
produced it.

The types in this module perform no model, storage, or network effects.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    model_validator,
)

from whetstone.optimization.identity import (
    compute_identity_hash,
    reject_non_json,
    require_full_hash,
)

MIPROV2_ACCEPTANCE_SCHEMA = "whetstone.miprov2_bootstrap_acceptance"
MIPROV2_COMPONENT_DEMO_SCHEMA = "whetstone.miprov2_component_demo"
MIPROV2_COMPONENT_DEMO_SET_SCHEMA = "whetstone.miprov2_component_demo_set"
MIPROV2_DEMO_SCHEMA_VERSION = 1

type MetricValue = StrictBool | float


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

    source_task_identity: StrictStr
    source_rollout_identity: StrictStr
    source_trace_identity: StrictStr
    source_output_identity: StrictStr
    source_score_identity: StrictStr
    metric_present: StrictBool
    score: MetricValue | None
    metric_threshold: float | None
    accepted: StrictBool

    @model_validator(mode="after")
    def _validate_decision(self) -> BootstrapAcceptance:
        for field in (
            "source_task_identity",
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
        return self.model_dump(mode="json")

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
    inputs: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_step(self) -> ObservedTraceStep:
        if self.trace_index < 0:
            raise ValueError("trace_index cannot be negative")
        if self.component_id == "":
            raise ValueError("component_id must be non-empty when present")
        reject_non_json(self.inputs, field="inputs")
        reject_non_json(self.outputs, field="outputs")
        return self

    def identity_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class ComponentDemo(BaseModel):
    """One prompt-ready demonstration for one Whetstone component."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    component_id: StrictStr
    source_kind: DemoSourceKind
    inputs: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, Any] = Field(default_factory=dict)
    augmented: StrictBool

    source_task_identity: StrictStr
    source_rollout_identity: StrictStr
    source_trace_identity: StrictStr
    source_output_identity: StrictStr
    source_score_identity: StrictStr
    source_trace_index: StrictInt | None
    score: MetricValue | None
    acceptance_identity_hash: StrictStr

    @model_validator(mode="after")
    def _validate_demo(self) -> ComponentDemo:
        if not self.component_id:
            raise ValueError("component_id must be non-empty")
        for field in (
            "source_task_identity",
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
        reject_non_json(self.inputs, field="inputs")
        reject_non_json(self.outputs, field="outputs")
        return self

    def identity_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def identity_hash(self) -> str:
        return compute_identity_hash(
            schema=MIPROV2_COMPONENT_DEMO_SCHEMA,
            schema_version=MIPROV2_DEMO_SCHEMA_VERSION,
            payload=self.identity_payload(),
        )


class LabeledTaskDemo(BaseModel):
    """A raw labeled task adapted into component-addressed prompt fields."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_task_identity: StrictStr
    inputs_by_component: dict[str, dict[str, Any]]
    outputs_by_component: dict[str, dict[str, Any]]

    @model_validator(mode="after")
    def _validate_task(self) -> LabeledTaskDemo:
        require_full_hash(
            self.source_task_identity, field="source_task_identity"
        )
        if set(self.inputs_by_component) != set(self.outputs_by_component):
            raise ValueError(
                "labeled input and output component sets must match"
            )
        if any(not component for component in self.inputs_by_component):
            raise ValueError("labeled component ids must be non-empty")
        reject_non_json(self.inputs_by_component, field="inputs_by_component")
        reject_non_json(
            self.outputs_by_component, field="outputs_by_component"
        )
        return self

    def for_component(self, component_id: str) -> ComponentDemo:
        """Adapt this task to one component without fake trace data."""

        if component_id not in self.inputs_by_component:
            raise ValueError(f"labeled task has no component {component_id!r}")
        source_identity = compute_identity_hash(
            schema="whetstone.miprov2_labeled_demo_source",
            schema_version=MIPROV2_DEMO_SCHEMA_VERSION,
            payload={
                "source_task_identity": self.source_task_identity,
                "component_id": component_id,
                "inputs": self.inputs_by_component[component_id],
                "outputs": self.outputs_by_component[component_id],
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
            inputs=self.inputs_by_component[component_id],
            outputs=self.outputs_by_component[component_id],
            augmented=False,
            source_task_identity=self.source_task_identity,
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
