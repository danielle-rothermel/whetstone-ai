from __future__ import annotations

import pytest
from dr_store.sync import BlockingObjectStore, open_sqlite

from whetstone.coordination.runtime_bootstrap import register_runtime
from whetstone.core.effects.authority import ReplayPolicy
from whetstone.optim.contracts import StepMode
from whetstone.optim.gepa.harness_adapter import GEPA_ADAPTER_KEY


class _StubGepaAdapter:
    @property
    def key(self) -> str:
        return GEPA_ADAPTER_KEY

    @property
    def mode(self) -> StepMode:
        return StepMode.PROPOSAL_ONLY

    @property
    def required_replay_policy(self) -> ReplayPolicy:
        return ReplayPolicy.DURABLE_WORKFLOW

    def invoke(self, request, handles):
        raise AssertionError("prepare_gepa_run must not invoke the adapter")


def _toy_gepa_control(*, sqlite_path: str):
    from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
    from whetstone.optim.gepa.control import configure_gepa
    from whetstone.optim.proposal.proposer import (
        ProposerConfig,
        prompt_adapter_identity_hash,
    )
    from whetstone.provider.language_model import PlainPromptAdapter
    from whetstone.testing.toy.experiment import build_toy_experiment

    experiment = build_toy_experiment(num_seeds=1)
    with open_sqlite(sqlite_path) as store:
        engine = ReferenceEvalRuntimeConfig().build_engine(store)
        task_hashes = engine.sampling.task_hashes[:1]
        prompt_adapter = PlainPromptAdapter()
        return configure_gepa(
            reflection_model=ProposerConfig(
                provider_call_config=engine.provider_execution_policy_ref,
            ),
            metric=engine.eval_config_ref,
            reward_policy_hash=experiment.reward_policy.identity_hash(),
            evaluation_execution_policy_hash=engine.execution_policy_identity_hash(),
            proposal_execution_policy_hash=engine.execution_policy_identity_hash(),
            proposal_prompt_adapter_identity_hash=prompt_adapter_identity_hash(
                prompt_adapter
            ),
            proposal_durability_policy_identity_hash="c" * 64,
            task_model_identity_hash=engine.task_model_identity_hash(),
            prompt_format_identity_hash="d" * 64,
            prompt_binding_identity_hash="e" * 64,
            trainset_task_hashes=task_hashes,
            valset_task_hashes=None,
            component_names=("generate",),
            num_predictors=1,
            max_metric_calls=2,
        )


def test_register_runtime_is_idempotent_for_same_store(sqlite_store) -> None:
    first = register_runtime(store=sqlite_store)
    second = register_runtime(store=sqlite_store)
    assert first.harness is not second.harness
    assert first.controller.runtime_hash != second.controller.runtime_hash


def test_register_runtime_opens_store_when_omitted(tmp_path) -> None:
    runtime = register_runtime(sqlite_path=str(tmp_path / "runtime.sqlite"))
    assert isinstance(runtime.store, BlockingObjectStore)
    assert runtime.store.resolve("absent") is None


def test_register_runtime_accepts_caller_engine(sqlite_store) -> None:
    from whetstone.coordination.runtime_bootstrap import build_toy_copro_control
    from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
    from whetstone.testing.toy.experiment import build_toy_experiment

    experiment = build_toy_experiment(num_seeds=1)
    engine = ReferenceEvalRuntimeConfig().build_engine(
        sqlite_store,
        experiment=experiment,
    )
    control = build_toy_copro_control(breadth=2, depth=1, engine=engine)
    runtime = register_runtime(
        store=sqlite_store,
        engine=engine,
        copro_control=control,
    )
    assert runtime.eval_service is not None


def test_register_runtime_requires_store_when_engine_is_supplied(
    sqlite_store,
) -> None:
    from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
    from whetstone.testing.toy.experiment import build_toy_experiment

    experiment = build_toy_experiment(num_seeds=1)
    engine = ReferenceEvalRuntimeConfig().build_engine(
        sqlite_store,
        experiment=experiment,
    )
    with pytest.raises(ValueError, match="requires store="):
        register_runtime(engine=engine)


def test_register_runtime_requires_control_when_engine_is_supplied(
    sqlite_store,
) -> None:
    from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
    from whetstone.testing.toy.experiment import build_toy_experiment

    experiment = build_toy_experiment(num_seeds=1)
    engine = ReferenceEvalRuntimeConfig().build_engine(
        sqlite_store,
        experiment=experiment,
    )
    with pytest.raises(ValueError, match="requires copro_control="):
        register_runtime(store=sqlite_store, engine=engine)


def test_prepare_copro_run_uses_caller_experiment(sqlite_store) -> None:
    from whetstone.coordination.runtime_bootstrap import (
        build_toy_copro_control,
        prepare_copro_run,
    )
    from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
    from whetstone.testing.toy.experiment import (
        TOY_MUTATION_FIELD,
        build_toy_experiment,
        toy_template_render_contract,
    )

    experiment = build_toy_experiment(num_seeds=1)
    engine = ReferenceEvalRuntimeConfig().build_engine(
        sqlite_store,
        experiment=experiment,
    )
    control = build_toy_copro_control(breadth=2, depth=1, engine=engine)
    runtime = register_runtime(
        store=sqlite_store,
        engine=engine,
        copro_control=control,
    )
    launch = prepare_copro_run(
        runtime,
        run_id="caller-experiment",
        control=control,
        experiment=experiment,
        render_contract=toy_template_render_contract(),
        mutation_field=TOY_MUTATION_FIELD,
    )
    assert launch.run.reward_policy == experiment.reward_policy
    assert launch.initial_candidate == experiment.initial_candidate


def test_register_runtime_accepts_caller_proposer_transport(sqlite_store) -> None:
    from whetstone.coordination.runtime_bootstrap import build_toy_copro_control
    from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
    from whetstone.testing.fakes.proposer import DummyProposerTransport
    from whetstone.testing.toy.experiment import build_toy_experiment

    experiment = build_toy_experiment(num_seeds=1)
    engine = ReferenceEvalRuntimeConfig().build_engine(
        sqlite_store,
        experiment=experiment,
    )
    control = build_toy_copro_control(breadth=2, depth=1, engine=engine)
    runtime_config = ReferenceEvalRuntimeConfig()
    prompt_hash = "c" * 64
    transport = DummyProposerTransport(
        scripted_bodies=("body",),
        execution_policy_hash=runtime_config.execution_policy.identity_hash,
        prompt_adapter_identity_hash=prompt_hash,
    )
    runtime = register_runtime(
        store=sqlite_store,
        engine=engine,
        copro_control=control,
        proposer_transport=transport,
    )
    assert runtime.adapter_registry.resolve("copro") is not None


def test_prepare_gepa_run_requires_registered_adapter(
    sqlite_store, tmp_path
) -> None:
    from whetstone.coordination.runtime_bootstrap import prepare_gepa_run

    runtime = register_runtime(store=sqlite_store)
    control = _toy_gepa_control(sqlite_path=str(tmp_path / "gepa.sqlite"))
    with pytest.raises(ValueError, match="extra_adapters"):
        prepare_gepa_run(runtime, run_id="missing-gepa", control=control)


def test_prepare_gepa_run_binds_when_adapter_is_registered(
    sqlite_store, tmp_path
) -> None:
    from whetstone.coordination.runtime_bootstrap import prepare_gepa_run

    runtime = register_runtime(
        store=sqlite_store,
        extra_adapters={GEPA_ADAPTER_KEY: _StubGepaAdapter()},
    )
    control = _toy_gepa_control(sqlite_path=str(tmp_path / "gepa.sqlite"))
    launch = prepare_gepa_run(
        runtime,
        run_id="registered-gepa",
        control=control,
    )
    assert launch.run.adapter_key == GEPA_ADAPTER_KEY
    assert launch.run.terminal_output_contract.returned_proposal_count == 1
