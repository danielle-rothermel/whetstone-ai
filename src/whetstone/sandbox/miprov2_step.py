from __future__ import annotations

from pydantic import BaseModel, ConfigDict, StrictInt, StrictStr

__all__ = ["Miprov2PlanPreview", "run_miprov2_plan_preview"]


class Miprov2PlanPreview(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    round_index: StrictInt
    surface: StrictStr
    proposal_mode: StrictStr
    mutation_field: StrictStr
    base_template: StrictStr
    component_count: StrictInt
    message: StrictStr


def run_miprov2_plan_preview(*, round_index: int = 0) -> Miprov2PlanPreview:
    from whetstone.optim.miprov2.adapter import MIPROV2_PROPOSAL
    from whetstone.testing.toy.experiment import (
        DEFAULT_TOY_TEMPLATE,
        TOY_MUTATION_FIELD,
        build_toy_experiment,
    )

    experiment = build_toy_experiment()
    template = experiment.initial_candidate.payload[TOY_MUTATION_FIELD]
    return Miprov2PlanPreview(
        round_index=round_index,
        surface="instruction_proposal",
        proposal_mode=MIPROV2_PROPOSAL,
        mutation_field=TOY_MUTATION_FIELD,
        base_template=str(template),
        component_count=1,
        message=(
            "MIPROv2 preview stops before Optuna trials, bootstrap generations, "
            f"or evaluation intents (toy template: {DEFAULT_TOY_TEMPLATE!r})."
        ),
    )
