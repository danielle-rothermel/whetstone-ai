"""Build a runnable toy MIPROv2 control, adapter, and opening state.

MIPROv2's control is fully resolved before any effect happens, and its
opening state binds the run, the control, every durable route, and the
labeled/proposal datasets the search reads. That is a lot of wiring, and
every harness test and smoke script needs the same wiring, so it lives
here once rather than being re-derived per call site.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from whetstone.core.identity import ImmutableJsonObject, compute_identity_hash
from whetstone.core.roles import EvalRole
from whetstone.eval.protocol import EvalEngine
from whetstone.experiment.candidate import candidate_reference
from whetstone.optim.contracts import OptimRunRef
from whetstone.optim.miprov2.control import (
    Miprov2ComponentSpec,
    Miprov2Control,
    Miprov2DemoMode,
    Miprov2InjectedDefaults,
    Miprov2ProgramLayout,
    configure_miprov2,
)
from whetstone.optim.miprov2.demo import LabeledTaskDemo
from whetstone.optim.miprov2.proposal import (
    Miprov2DatasetExample,
    Miprov2PromptComponent,
)
from whetstone.optim.miprov2.rng import Miprov2DurableBindings
from whetstone.optim.miprov2.runtime import (
    Miprov2Driver,
    Miprov2EffectBudget,
    Miprov2State,
)
from whetstone.optim.proposal.proposer import (
    ProposerConfig,
    prompt_adapter_identity_hash,
)
from whetstone.provider.language_model import PlainPromptAdapter
from whetstone.testing.toy.experiment import (
    TOY_MUTATION_FIELD,
    build_toy_experiment,
    toy_template_render_contract,
)

if TYPE_CHECKING:
    from whetstone.experiment.env import Experiment

__all__ = [
    "TOY_MIPROV2_COMPONENT_ID",
    "build_toy_miprov2_control",
    "build_toy_miprov2_state",
    "toy_miprov2_budget",
    "toy_proposal_policy_identity_hash",
]

#: The single optimizable component. Whetstone's generation primitive
#: exposes exactly one provider trace, so MIPROv2 optimizes one component.
TOY_MIPROV2_COMPONENT_ID = "generate"


def build_toy_miprov2_control(
    *,
    engine: EvalEngine,
    experiment: Experiment | None = None,
    demo_mode: Miprov2DemoMode = Miprov2DemoMode.FEWSHOT,
    num_trials: int = 2,
    # Seeds -3/-2 are RESET/LABELS_ONLY; 3 admits seed -1, the first
    # bootstrap candidate. Two candidates never reach that seed.
    num_candidates: int = 3,
    max_bootstrapped_demos: int | None = None,
    max_labeled_demos: int | None = None,
    seed: int = 9,
    minibatch: bool = False,
    minibatch_full_eval_steps: int = 1,
    mutation_field: str = TOY_MUTATION_FIELD,
) -> Miprov2Control:
    """Resolve a toy MIPROv2 control bound to ``engine``'s exact authorities.

    The demo maxima follow ``demo_mode`` unless overridden: a zero-shot run
    must carry zero maxima, and the bootstrapping modes need at least one
    demo to bootstrap.
    """

    resolved_experiment = experiment or build_toy_experiment(num_seeds=1)
    bootstrapped, labeled = _demo_maxima(
        demo_mode,
        max_bootstrapped_demos=max_bootstrapped_demos,
        max_labeled_demos=max_labeled_demos,
    )
    prompt_adapter = PlainPromptAdapter()
    task_hashes = tuple(engine.sampling.task_hashes)
    if len(task_hashes) < 2:
        raise ValueError(
            "toy MIPROv2 needs at least two tasks to split train and val"
        )
    defaults = Miprov2InjectedDefaults(
        prompt_model=ProposerConfig(
            provider_call_config=engine.provider_execution_policy_ref,
            temperature=1.0,
        ),
        bootstrap_eval_source=engine.eval_config_ref,
        validation_eval_source=engine.eval_config_ref,
        reward_policy=resolved_experiment.reward_policy,
        eval_role=EvalRole.INTERNAL,
        provider_execution_policy_ref=engine.provider_execution_policy_ref,
        provider_execution_policy_hash=(
            engine.execution_policy_identity_hash()
        ),
        task_model_identity_hash=engine.task_model_identity_hash(),
        prompt_adapter=prompt_adapter,
        template_render_contract=toy_template_render_contract(),
        mutation_field=mutation_field,
        max_errors=4,
        validation_eval_source_is_metric_authority=True,
    )
    base_candidate = candidate_reference(resolved_experiment.initial_candidate)
    layout = Miprov2ProgramLayout(
        layout_id="whetstone.toy.miprov2",
        component_specs=(
            Miprov2ComponentSpec(
                component_id=TOY_MIPROV2_COMPONENT_ID,
                prompt_format_identity_hash=prompt_adapter_identity_hash(
                    prompt_adapter
                ),
            ),
        ),
    )
    return configure_miprov2(
        base_candidate=base_candidate,
        program_layout=layout,
        # The bound engine's split is the repeat authority; the control
        # records the count it was resolved against.
        num_seeds=engine.sampling.num_seeds,
        trainset=task_hashes[:1],
        valset=task_hashes[1:],
        max_bootstrapped_demos=bootstrapped,
        max_labeled_demos=labeled,
        auto=None,
        num_candidates=num_candidates,
        num_trials=num_trials,
        seed=seed,
        init_temperature=1.0,
        minibatch=minibatch,
        minibatch_size=len(task_hashes[1:]),
        minibatch_full_eval_steps=minibatch_full_eval_steps,
        demo_mode=demo_mode,
        defaults=defaults,
    )


def _demo_maxima(
    demo_mode: Miprov2DemoMode,
    *,
    max_bootstrapped_demos: int | None,
    max_labeled_demos: int | None,
) -> tuple[int, int]:
    if demo_mode is Miprov2DemoMode.ZEROSHOT:
        bootstrapped = 0 if max_bootstrapped_demos is None else max_bootstrapped_demos
        labeled = 0 if max_labeled_demos is None else max_labeled_demos
        return bootstrapped, labeled
    return (
        1 if max_bootstrapped_demos is None else max_bootstrapped_demos,
        1 if max_labeled_demos is None else max_labeled_demos,
    )


def toy_miprov2_budget(
    *,
    bootstrap_generations: int = 32,
    proposal_calls: int = 32,
    evaluations: int = 32,
    task_rows: int = 256,
) -> Miprov2EffectBudget:
    """A ceiling generous enough that a toy run terminates on its schedule.

    These are budgets, not expectations: a toy run should end because its
    trial schedule is exhausted, not because it ran out of budget, so a
    budget-exhaustion failure in a test is a real signal.
    """

    return Miprov2EffectBudget(
        bootstrap_generations=bootstrap_generations,
        proposal_calls=proposal_calls,
        evaluations=evaluations,
        task_rows=task_rows,
    )


def build_toy_miprov2_state(
    *,
    run: OptimRunRef,
    control: Miprov2Control,
    engine: EvalEngine,
    proposal_executor_policy_identity_hash: str,
    proposal_transport_durability_identity_hash: str,
    budget: Miprov2EffectBudget | None = None,
    driver: Miprov2Driver | None = None,
) -> Miprov2State:
    """Build the opening MIPROv2 state for ``run`` under ``control``."""

    bindings = Miprov2DurableBindings(
        control_identity_hash=control.identity_hash(),
        prompt_route_identity_hash=control.prompt_model.identity_hash(),
        task_route_identity_hash=control.task_model_identity_hash,
        execution_policy_identity_hash=(
            control.provider_execution_policy_hash
        ),
        prompt_adapter_identity_hash=control.prompt_adapter_identity_hash,
        proposal_executor_policy_identity_hash=(
            proposal_executor_policy_identity_hash
        ),
        proposal_transport_durability_identity_hash=(
            proposal_transport_durability_identity_hash
        ),
        base_candidate_identity_hash=control.base_candidate.identity_hash,
        teacher_candidate_identity_hash=(
            control.teacher_candidate.identity_hash
        ),
    )
    component_id = control.component_ids[0]
    inputs_by_hash = _task_inputs_by_hash(engine)
    labeled_trainset = tuple(
        LabeledTaskDemo(
            source_task_hash=task_hash,
            inputs_by_component=ImmutableJsonObject(
                {component_id: dict(inputs_by_hash[task_hash])}
            ),
            outputs_by_component=ImmutableJsonObject(
                {component_id: {"response": ""}}
            ),
        )
        for task_hash in control.trainset_task_hashes
    )
    template = control.base_candidate.record.payload[control.mutation_field]
    proposal_components = (
        Miprov2PromptComponent(
            component_id=component_id,
            template=str(template),
            template_render_contract=control.template_render_contract,
            rendering_rules=(
                "Render the template with the task's prompt inputs "
                "substituted for its placeholders."
            ),
            example_execution=(
                "The rendered prompt is sent to the task model and its reply "
                "is scored by the run's Reward Policy."
            ),
        ),
    )
    proposal_trainset = tuple(
        Miprov2DatasetExample(
            task_hash=task_hash,
            rendered_record=_rendered_record(inputs_by_hash[task_hash]),
        )
        for task_hash in control.trainset_task_hashes
    )
    resolved_driver = driver or Miprov2Driver()
    return resolved_driver.start(
        run=run,
        control=control,
        bindings=bindings,
        labeled_trainset=labeled_trainset,
        proposal_components=proposal_components,
        proposal_trainset=proposal_trainset,
        component_field_order={component_id: ("prompt",)},
        budget=budget or toy_miprov2_budget(),
    )


def _task_inputs_by_hash(engine: EvalEngine) -> dict[str, dict[str, str]]:
    return {
        task.task_hash: dict(task.prompt_inputs)
        for task in engine.sampling.tasks
    }


def _rendered_record(prompt_inputs: dict[str, str]) -> str:
    return "\n".join(
        f"{name}: {value}" for name, value in sorted(prompt_inputs.items())
    )


def toy_proposal_policy_identity_hash() -> str:
    """The identity of the inline proposal executor policy used in tests."""

    return compute_identity_hash(
        schema="whetstone.testing.inline_proposal_executor",
        schema_version=1,
        payload={"mode": "inline"},
    )
