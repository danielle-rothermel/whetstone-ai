"""Pin auto-mode instruct / trial derivation for each demo regime."""

from __future__ import annotations

from dr_store.sync import open_sqlite

from whetstone.core.roles import EvalRole
from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
from whetstone.experiment.candidate import candidate_reference
from whetstone.optim.miprov2.control import (
    Miprov2ComponentSpec,
    Miprov2InjectedDefaults,
    Miprov2ProgramLayout,
    configure_miprov2,
)
from whetstone.optim.miprov2.demo_mode import Miprov2DemoMode
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
from whetstone.testing.toy.miprov2 import TOY_MIPROV2_COMPONENT_ID


def _auto_control(engine, *, auto: str, demo_mode: Miprov2DemoMode):
    experiment = build_toy_experiment(num_seeds=1)
    prompt_adapter = PlainPromptAdapter()
    task_hashes = tuple(engine.sampling.task_hashes)
    maxima = (0, 0) if demo_mode is Miprov2DemoMode.ZEROSHOT else (1, 1)
    return configure_miprov2(
        base_candidate=candidate_reference(experiment.initial_candidate),
        program_layout=Miprov2ProgramLayout(
            layout_id="whetstone.toy.miprov2",
            component_specs=(
                Miprov2ComponentSpec(
                    component_id=TOY_MIPROV2_COMPONENT_ID,
                    prompt_format_identity_hash=prompt_adapter_identity_hash(
                        prompt_adapter
                    ),
                ),
            ),
        ),
        trainset=task_hashes[:1],
        valset=task_hashes[1:],
        max_bootstrapped_demos=maxima[0],
        max_labeled_demos=maxima[1],
        auto=auto,
        seed=9,
        demo_mode=demo_mode,
        defaults=Miprov2InjectedDefaults(
            prompt_model=ProposerConfig(
                provider_call_config=engine.provider_execution_policy_ref,
                temperature=1.0,
            ),
            bootstrap_eval_source=engine.eval_config_ref,
            validation_eval_source=engine.eval_config_ref,
            reward_policy=experiment.reward_policy,
            eval_role=EvalRole.INTERNAL,
            provider_execution_policy_ref=engine.provider_execution_policy_ref,
            provider_execution_policy_hash=(
                engine.execution_policy_identity_hash()
            ),
            task_model_identity_hash=engine.task_model_identity_hash(),
            prompt_adapter=prompt_adapter,
            template_render_contract=toy_template_render_contract(),
            mutation_field=TOY_MUTATION_FIELD,
            max_errors=4,
            validation_eval_source_is_metric_authority=True,
        ),
    )


def test_auto_mode_splits_instruct_counts_by_demo_search(tmp_path) -> None:
    with open_sqlite(str(tmp_path / "auto.sqlite")) as store:
        engine = ReferenceEvalRuntimeConfig().build_engine(store)

    expected = {
        ("light", Miprov2DemoMode.FEWSHOT): (3, 6, 10),
        ("light", Miprov2DemoMode.ZEROSHOT): (6, 6, 9),
        ("light", Miprov2DemoMode.GROUND_ONLY): (6, 6, 9),
        ("medium", Miprov2DemoMode.FEWSHOT): (6, 12, 18),
        ("medium", Miprov2DemoMode.ZEROSHOT): (12, 12, 18),
        ("heavy", Miprov2DemoMode.FEWSHOT): (9, 18, 27),
        ("heavy", Miprov2DemoMode.ZEROSHOT): (18, 18, 27),
    }
    for (auto, demo_mode), (
        instruct,
        fewshot,
        trials,
    ) in expected.items():
        control = _auto_control(
            engine, auto=auto, demo_mode=demo_mode
        )
        assert control.auto == auto
        assert control.num_candidates is None
        assert control.num_instruct_candidates == instruct
        assert control.num_fewshot_candidates == fewshot
        assert control.num_trials == trials
        assert control.demo_mode is demo_mode
        assert control.eval_role is EvalRole.INTERNAL
