from __future__ import annotations

from dataclasses import replace

import pytest
from dr_store.sync import BlockingObjectStore, open_sqlite

from whetstone.coordination.runtime_bootstrap import prepare_copro_run, prepare_gepa_run
from whetstone.core.leasing import ReplayPolicy
from whetstone.optim.contracts import StepMode
from whetstone.optim.gepa.harness_adapter import GEPA_ADAPTER_KEY
from whetstone.testing.runtime import build_toy_copro_control, register_toy_runtime
from whetstone.testing.toy.experiment import (
    TOY_MUTATION_FIELD,
    build_toy_experiment,
    toy_template_render_contract,
)


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


def _toy_gepa_control(*, sqlite_path: str, max_metric_calls: int = 2):
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
            max_metric_calls=max_metric_calls,
        )


def _gepa_harness_adapter(control, *, seed_text: str):
    from unittest.mock import MagicMock

    from whetstone.optim.gepa.harness_adapter import (
        GepaHarnessAdapter,
        GepaHarnessAdapterFactory,
    )

    return GepaHarnessAdapter(
        control=control,
        seed_candidate={"generate": seed_text},
        trainset=(),
        valset=None,
        adapter_factory=GepaHarnessAdapterFactory(factory=MagicMock()),
    )


def _toy_launch_kwargs(experiment=None):
    resolved = experiment or build_toy_experiment(num_seeds=1)
    return {
        "experiment": resolved,
        "render_contract": toy_template_render_contract(),
        "mutation_field": TOY_MUTATION_FIELD,
    }


def test_register_runtime_is_idempotent_for_same_store(sqlite_store) -> None:
    first = register_toy_runtime(store=sqlite_store)
    second = register_toy_runtime(store=sqlite_store)
    assert first.harness is not second.harness
    assert first.controller.runtime_hash != second.controller.runtime_hash


def test_register_runtime_opens_store_when_omitted(tmp_path) -> None:
    runtime = register_toy_runtime(sqlite_path=str(tmp_path / "runtime.sqlite"))
    assert isinstance(runtime.store, BlockingObjectStore)
    assert runtime.store.resolve("absent") is None


def test_register_runtime_accepts_caller_engine(sqlite_store) -> None:
    from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig

    experiment = build_toy_experiment(num_seeds=1)
    engine = ReferenceEvalRuntimeConfig().build_engine(
        sqlite_store,
        experiment=experiment,
    )
    control = build_toy_copro_control(breadth=2, depth=1, engine=engine)
    runtime = register_toy_runtime(
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
        register_toy_runtime(engine=engine)


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
        register_toy_runtime(store=sqlite_store, engine=engine)


def test_prepare_copro_run_uses_caller_experiment(sqlite_store) -> None:
    from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig

    experiment = build_toy_experiment(num_seeds=1)
    engine = ReferenceEvalRuntimeConfig().build_engine(
        sqlite_store,
        experiment=experiment,
    )
    control = build_toy_copro_control(breadth=2, depth=1, engine=engine)
    runtime = register_toy_runtime(
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


def test_build_toy_copro_control_binds_engine_hashes(sqlite_store) -> None:
    from dr_providers import ProviderKind

    from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig

    experiment = build_toy_experiment(num_seeds=1)
    engine = ReferenceEvalRuntimeConfig(
        provider_kind=ProviderKind.ANTHROPIC,
    ).build_engine(
        sqlite_store,
        experiment=experiment,
    )
    control = build_toy_copro_control(breadth=2, depth=1, engine=engine)
    assert (
        control.expected_reward_policy_hash
        == engine.reward_policy_identity_hash()
    )
    assert (
        control.provider_execution_policy_hash
        == engine.execution_policy_identity_hash()
    )


def test_prepare_copro_run_rejects_mismatched_reward_policy(
    sqlite_store,
) -> None:
    from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
    from whetstone.experiment.reward import RewardPolicy, RewardTerm

    experiment = build_toy_experiment(num_seeds=1)
    engine = ReferenceEvalRuntimeConfig().build_engine(
        sqlite_store,
        experiment=experiment,
    )
    control = build_toy_copro_control(breadth=2, depth=1, engine=engine)
    runtime = register_toy_runtime(
        store=sqlite_store,
        engine=engine,
        copro_control=control,
    )
    other = replace(
        experiment,
        reward_policy=RewardPolicy(
            policy_name="other-reward",
            terms=(RewardTerm(name="score", weight=1.0),),
        ),
    )
    with pytest.raises(ValueError, match="reward policy must match"):
        prepare_copro_run(
            runtime,
            run_id="mismatched-reward",
            control=control,
            **_toy_launch_kwargs(other),
        )


def test_register_runtime_accepts_caller_proposer_transport(sqlite_store) -> None:
    from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
    from whetstone.testing.fakes.proposer import DummyProposerTransport

    experiment = build_toy_experiment(num_seeds=1)
    engine = ReferenceEvalRuntimeConfig().build_engine(
        sqlite_store,
        experiment=experiment,
    )
    control = build_toy_copro_control(breadth=2, depth=1, engine=engine)
    transport = DummyProposerTransport(
        scripted_bodies=("body",),
        execution_policy_hash=control.provider_execution_policy_hash,
        prompt_adapter_identity_hash=control.prompt_adapter_identity_hash,
    )
    runtime = register_toy_runtime(
        store=sqlite_store,
        engine=engine,
        copro_control=control,
        proposer_transport=transport,
    )
    assert runtime.adapter_registry.resolve("copro") is not None


def test_prepare_gepa_run_requires_registered_adapter(
    sqlite_store, tmp_path
) -> None:
    runtime = register_toy_runtime(store=sqlite_store)
    control = _toy_gepa_control(sqlite_path=str(tmp_path / "gepa.sqlite"))
    with pytest.raises(ValueError, match="adapter registry"):
        prepare_gepa_run(
            runtime,
            run_id="missing-gepa",
            control=control,
            **_toy_launch_kwargs(),
        )


def test_prepare_gepa_run_binds_when_adapter_is_registered(
    sqlite_store, tmp_path
) -> None:
    runtime = register_toy_runtime(
        store=sqlite_store,
        extra_adapters={GEPA_ADAPTER_KEY: _StubGepaAdapter()},
    )
    control = _toy_gepa_control(sqlite_path=str(tmp_path / "gepa.sqlite"))
    launch = prepare_gepa_run(
        runtime,
        run_id="registered-gepa",
        control=control,
        **_toy_launch_kwargs(),
    )
    assert launch.run.adapter_key == GEPA_ADAPTER_KEY
    assert launch.run.terminal_output_contract.returned_proposal_count == 1


def test_prepare_gepa_run_rejects_mismatched_control(
    sqlite_store, tmp_path
) -> None:
    experiment = build_toy_experiment(num_seeds=1)
    seed_text = str(experiment.initial_candidate.payload[TOY_MUTATION_FIELD])
    control = _toy_gepa_control(sqlite_path=str(tmp_path / "gepa-a.sqlite"))
    other = _toy_gepa_control(
        sqlite_path=str(tmp_path / "gepa-b.sqlite"),
        max_metric_calls=3,
    )
    runtime = register_toy_runtime(
        store=sqlite_store,
        extra_adapters={
            GEPA_ADAPTER_KEY: _gepa_harness_adapter(control, seed_text=seed_text),
        },
    )
    with pytest.raises(ValueError, match="must match the registered GEPA adapter"):
        prepare_gepa_run(
            runtime,
            run_id="mismatched-gepa-control",
            control=other,
            **_toy_launch_kwargs(experiment),
        )


def test_prepare_gepa_run_rejects_mismatched_seed(
    sqlite_store, tmp_path
) -> None:
    experiment = build_toy_experiment(num_seeds=1)
    control = _toy_gepa_control(sqlite_path=str(tmp_path / "gepa-seed.sqlite"))
    runtime = register_toy_runtime(
        store=sqlite_store,
        extra_adapters={
            GEPA_ADAPTER_KEY: _gepa_harness_adapter(
                control,
                seed_text="not-the-launch-seed",
            ),
        },
    )
    with pytest.raises(ValueError, match="must match the adapter seed"):
        prepare_gepa_run(
            runtime,
            run_id="mismatched-gepa-seed",
            control=control,
            **_toy_launch_kwargs(experiment),
        )


def test_prepare_gepa_run_binds_matching_adapter_seed(
    sqlite_store, tmp_path
) -> None:
    experiment = build_toy_experiment(num_seeds=1)
    seed_text = str(experiment.initial_candidate.payload[TOY_MUTATION_FIELD])
    control = _toy_gepa_control(sqlite_path=str(tmp_path / "gepa-ok.sqlite"))
    runtime = register_toy_runtime(
        store=sqlite_store,
        extra_adapters={
            GEPA_ADAPTER_KEY: _gepa_harness_adapter(control, seed_text=seed_text),
        },
    )
    launch = prepare_gepa_run(
        runtime,
        run_id="matching-gepa",
        control=control,
        **_toy_launch_kwargs(experiment),
    )
    assert launch.initial_candidate == experiment.initial_candidate
    assert launch.run.optimizer_config == control.reference()


def test_prepare_copro_run_rejects_mismatched_control(sqlite_store) -> None:
    from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig

    experiment = build_toy_experiment(num_seeds=1)
    engine = ReferenceEvalRuntimeConfig().build_engine(
        sqlite_store,
        experiment=experiment,
    )
    control = build_toy_copro_control(breadth=2, depth=1, engine=engine)
    other = build_toy_copro_control(breadth=3, depth=1, engine=engine)
    runtime = register_toy_runtime(
        store=sqlite_store,
        engine=engine,
        copro_control=control,
    )
    with pytest.raises(ValueError, match="must match the registered COPRO adapter"):
        prepare_copro_run(
            runtime,
            run_id="mismatched-copro-control",
            control=other,
            **_toy_launch_kwargs(experiment),
        )


def test_register_runtime_rejects_control_that_disagrees_with_engine(
    sqlite_store,
) -> None:
    from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
    from whetstone.experiment.reward import RewardPolicy, RewardTerm

    experiment = build_toy_experiment(num_seeds=1)
    other = replace(
        experiment,
        reward_policy=RewardPolicy(
            policy_name="other-reward",
            terms=(RewardTerm(name="score", weight=1.0),),
        ),
    )
    engine = ReferenceEvalRuntimeConfig().build_engine(
        sqlite_store,
        experiment=experiment,
    )
    other_engine = ReferenceEvalRuntimeConfig().build_engine(
        sqlite_store,
        experiment=other,
    )
    control = build_toy_copro_control(breadth=2, depth=1, engine=other_engine)
    with pytest.raises(ValueError, match="reward policy must match"):
        register_toy_runtime(
            store=sqlite_store,
            engine=engine,
            copro_control=control,
        )


def test_copro_adapter_rejects_mismatched_transport(sqlite_store) -> None:
    from whetstone.core.identity import compute_identity_hash
    from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
    from whetstone.optim.copro.adapter import CoproAdapter
    from whetstone.optim.proposal.proposer import build_inline_proposal_executor
    from whetstone.testing.fakes.proposer import DummyProposerTransport
    from whetstone.testing.toy.experiment import build_toy_experiment

    experiment = build_toy_experiment(num_seeds=1)
    engine = ReferenceEvalRuntimeConfig().build_engine(
        sqlite_store,
        experiment=experiment,
    )
    control = build_toy_copro_control(breadth=2, depth=1, engine=engine)
    transport = DummyProposerTransport(
        scripted_bodies=("body",),
        execution_policy_hash="a" * 64,
        prompt_adapter_identity_hash=control.prompt_adapter_identity_hash,
    )
    with pytest.raises(ValueError, match="provider execution policy"):
        CoproAdapter(
            control=control,
            transport=transport,
            proposal_executor=build_inline_proposal_executor(
                policy_identity_hash=compute_identity_hash(
                    schema="whetstone.testing.inline_proposal_executor",
                    schema_version=1,
                    payload={"mode": "inline"},
                ),
            ),
        )


def test_build_runtime_accepts_a_non_toy_experiment(sqlite_store) -> None:
    from whetstone.coordination.runtime_bootstrap import build_runtime
    from whetstone.core.leasing import EffectLeaseAuthority
    from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
    from whetstone.experiment.reward import RewardPolicy, RewardTerm
    from whetstone.optim.adapters import MappingAdapterRegistry
    from whetstone.optim.copro.adapter import COPRO_ADAPTER_KEY, CoproAdapter
    from whetstone.core.identity import compute_identity_hash
    from whetstone.optim.proposal.proposer import (
        build_inline_proposal_executor,
        prompt_adapter_identity_hash,
    )
    from whetstone.provider.language_model import PlainPromptAdapter
    from whetstone.testing.fakes.proposer import DummyProposerTransport

    experiment = replace(
        build_toy_experiment(num_seeds=1),
        env_name="whetstone.factory_probe",
        reward_policy=RewardPolicy(
            policy_name="factory-probe-reward",
            terms=(RewardTerm(name="score", weight=1.0),),
        ),
    )
    engine = ReferenceEvalRuntimeConfig().build_engine(
        sqlite_store,
        experiment=experiment,
    )
    control = build_toy_copro_control(breadth=2, depth=1, engine=engine)
    prompt_adapter = PlainPromptAdapter()
    adapter = CoproAdapter(
        control=control,
        transport=DummyProposerTransport(
            scripted_bodies=("body",),
            execution_policy_hash=control.provider_execution_policy_hash,
            prompt_adapter_identity_hash=prompt_adapter_identity_hash(
                prompt_adapter
            ),
        ),
        proposal_executor=build_inline_proposal_executor(
            policy_identity_hash=compute_identity_hash(
                schema="whetstone.testing.inline_proposal_executor",
                schema_version=1,
                payload={"mode": "inline"},
            ),
        ),
    )
    runtime = build_runtime(
        store=sqlite_store,
        engine=engine,
        adapter_registry=MappingAdapterRegistry(
            {COPRO_ADAPTER_KEY: adapter}
        ),
        effect_authority=EffectLeaseAuthority.memory(),
    )
    assert runtime.engine is engine
    assert runtime.ledger_engine is None
    launch = prepare_copro_run(
        runtime,
        run_id="factory-probe",
        control=control,
        experiment=experiment,
        render_contract=toy_template_render_contract(),
        mutation_field=TOY_MUTATION_FIELD,
    )
    assert launch.run.reward_policy == experiment.reward_policy


def test_build_runtime_platform_mode_requires_ledger_engine(
    sqlite_store,
) -> None:
    from whetstone.coordination.runtime_bootstrap import build_runtime
    from whetstone.core.leasing import EffectLeaseAuthority
    from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
    from whetstone.optim.adapters import MappingAdapterRegistry
    from whetstone.optim.copro.adapter import COPRO_ADAPTER_KEY, CoproAdapter
    from whetstone.core.identity import compute_identity_hash
    from whetstone.optim.proposal.proposer import (
        build_inline_proposal_executor,
        prompt_adapter_identity_hash,
    )
    from whetstone.provider.language_model import PlainPromptAdapter
    from whetstone.testing.fakes.proposer import DummyProposerTransport

    experiment = build_toy_experiment(num_seeds=1)
    engine = ReferenceEvalRuntimeConfig().build_engine(
        sqlite_store,
        experiment=experiment,
    )
    control = build_toy_copro_control(breadth=2, depth=1, engine=engine)
    prompt_adapter = PlainPromptAdapter()
    adapter = CoproAdapter(
        control=control,
        transport=DummyProposerTransport(
            scripted_bodies=("body",),
            execution_policy_hash=control.provider_execution_policy_hash,
            prompt_adapter_identity_hash=prompt_adapter_identity_hash(
                prompt_adapter
            ),
        ),
        proposal_executor=build_inline_proposal_executor(
            policy_identity_hash=compute_identity_hash(
                schema="whetstone.testing.inline_proposal_executor",
                schema_version=1,
                payload={"mode": "inline"},
            ),
        ),
    )
    with pytest.raises(ValueError, match="platform mode requires ledger_engine"):
        build_runtime(
            store=sqlite_store,
            engine=engine,
            adapter_registry=MappingAdapterRegistry(
                {COPRO_ADAPTER_KEY: adapter}
            ),
            effect_authority=EffectLeaseAuthority.memory(),
            platform=True,
        )


def test_build_runtime_owner_id_is_part_of_controller_identity(
    sqlite_store,
) -> None:
    from whetstone.coordination.runtime_bootstrap import build_runtime
    from whetstone.core.identity import compute_identity_hash
    from whetstone.core.leasing import EffectLeaseAuthority
    from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
    from whetstone.optim.adapters import MappingAdapterRegistry
    from whetstone.optim.copro.adapter import COPRO_ADAPTER_KEY, CoproAdapter
    from whetstone.optim.proposal.proposer import (
        build_inline_proposal_executor,
        prompt_adapter_identity_hash,
    )
    from whetstone.provider.language_model import PlainPromptAdapter
    from whetstone.testing.fakes.proposer import DummyProposerTransport

    experiment = build_toy_experiment(num_seeds=1)
    engine = ReferenceEvalRuntimeConfig().build_engine(
        sqlite_store,
        experiment=experiment,
    )
    control = build_toy_copro_control(breadth=2, depth=1, engine=engine)
    prompt_adapter = PlainPromptAdapter()
    adapter = CoproAdapter(
        control=control,
        transport=DummyProposerTransport(
            scripted_bodies=("body",),
            execution_policy_hash=control.provider_execution_policy_hash,
            prompt_adapter_identity_hash=prompt_adapter_identity_hash(
                prompt_adapter
            ),
        ),
        proposal_executor=build_inline_proposal_executor(
            policy_identity_hash=compute_identity_hash(
                schema="whetstone.testing.inline_proposal_executor",
                schema_version=1,
                payload={"mode": "inline"},
            ),
        ),
    )
    registry = MappingAdapterRegistry({COPRO_ADAPTER_KEY: adapter})
    first = build_runtime(
        store=sqlite_store,
        engine=engine,
        adapter_registry=registry,
        effect_authority=EffectLeaseAuthority.memory(),
        owner_id="owner-a",
    )
    second = build_runtime(
        store=sqlite_store,
        engine=engine,
        adapter_registry=registry,
        effect_authority=EffectLeaseAuthority.memory(),
        owner_id="owner-a",
    )
    other = build_runtime(
        store=sqlite_store,
        engine=engine,
        adapter_registry=registry,
        effect_authority=EffectLeaseAuthority.memory(),
        owner_id="owner-b",
    )
    assert first.controller.runtime_hash == second.controller.runtime_hash
    assert first.controller.runtime_hash != other.controller.runtime_hash
    first.close()
    second.close()
    other.close()


def test_registered_runtime_close_stops_child_workers(sqlite_store) -> None:
    import os
    import subprocess

    from whetstone.eval.protocol import EvalRequest
    from whetstone.eval.metadata import metadata_with_purpose
    from whetstone.eval.protocol import eval_is_success
    from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig

    def child_pids() -> frozenset[str]:
        listing = subprocess.run(
            ["pgrep", "-P", str(os.getpid())],
            capture_output=True,
            text=True,
            check=False,
        )
        return frozenset(listing.stdout.split())

    before = child_pids()
    engine = ReferenceEvalRuntimeConfig(
        driver_mode="subprocess",
    ).build_engine(sqlite_store)
    control = build_toy_copro_control(breadth=2, depth=1, engine=engine)
    runtime = register_toy_runtime(
        store=sqlite_store,
        engine=engine,
        copro_control=control,
    )
    evaluated = runtime.engine.evaluate(
        EvalRequest(
            request_id="runtime-close:run",
            candidate=runtime.engine.experiment.initial_candidate,
            metadata=metadata_with_purpose("test"),
        )
    )
    assert eval_is_success(evaluated)
    assert child_pids() - before
    runtime.close()
    assert child_pids() - before == frozenset()


def test_prepare_miprov2_run_rejects_a_differing_mutation_field(
    sqlite_store,
) -> None:
    from whetstone.coordination.runtime_bootstrap import prepare_miprov2_run
    from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
    from whetstone.optim.miprov2.adapter import (
        MIPROV2_ADAPTER_KEY,
        MIPROV2_STATE_KEY,
    )
    from whetstone.optim.miprov2.runtime import Miprov2State
    from whetstone.testing.runtime import (
        build_miprov2_adapter,
        prepare_toy_miprov2_run,
    )
    from whetstone.testing.toy.miprov2 import build_toy_miprov2_control

    experiment = build_toy_experiment(num_seeds=1)
    engine = ReferenceEvalRuntimeConfig().build_engine(
        sqlite_store,
        experiment=experiment,
    )
    control = build_toy_miprov2_control(engine=engine, experiment=experiment)
    adapter = build_miprov2_adapter(
        store=sqlite_store, control=control, engine=engine
    )
    runtime = register_toy_runtime(
        store=sqlite_store,
        engine=engine,
        copro_control=build_toy_copro_control(engine=engine),
        extra_adapters={MIPROV2_ADAPTER_KEY: adapter},
    )
    with pytest.raises(ValueError, match="mutation_field must match"):
        prepare_toy_miprov2_run(
            runtime,
            run_id="toy-wrong-field",
            control=control,
            engine=engine,
            experiment=experiment,
            mutation_field="other_prompt_field",
        )

    launch = prepare_toy_miprov2_run(
        runtime,
        run_id="miprov2-field-ok",
        control=control,
        engine=engine,
        experiment=experiment,
        mutation_field=TOY_MUTATION_FIELD,
    )
    assert launch.run.mutation_field == control.mutation_field
    opened = Miprov2State.model_validate(
        launch.extra_pools[MIPROV2_STATE_KEY]
    )
    with pytest.raises(ValueError, match="mutation_field must match"):
        prepare_miprov2_run(
            runtime,
            run_id="prod-wrong-field",
            control=control,
            experiment=experiment,
            initial_state=opened,
            mutation_field="other_prompt_field",
        )
