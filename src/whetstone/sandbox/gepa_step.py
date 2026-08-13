from __future__ import annotations

from pydantic import BaseModel, ConfigDict, StrictBool, StrictStr

__all__ = ["GepaStepPreview", "run_gepa_step_preview"]


class GepaStepPreview(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    component_names: tuple[StrictStr, ...]
    selected_component: StrictStr
    mutation_field: StrictStr
    current_text: StrictStr
    evaluation_intent_boundary: StrictStr
    show_intent: StrictBool


def run_gepa_step_preview(
    *,
    show_intent: bool = False,
    component_name: str = "generate",
) -> GepaStepPreview:
    from whetstone.testing.toy.experiment import (
        TOY_MUTATION_FIELD,
        build_toy_experiment,
    )

    experiment = build_toy_experiment()
    current = str(experiment.initial_candidate.payload[TOY_MUTATION_FIELD])
    intent = (
        "Would issue an internal Optim Eval Request for the reflected candidate "
        "after proposal acceptance."
        if show_intent
        else "Evaluation intent emission is omitted in preview mode."
    )
    return GepaStepPreview(
        component_names=("generate",),
        selected_component=component_name,
        mutation_field=TOY_MUTATION_FIELD,
        current_text=current,
        evaluation_intent_boundary=intent,
        show_intent=show_intent,
    )
