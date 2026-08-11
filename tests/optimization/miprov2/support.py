from __future__ import annotations

import hashlib
from typing import Any

from dr_providers import openrouter_chat_config
from dr_serialize import Jsonable
from dr_store import ObjectStore

from tests.optimization.support import (
    FULL_A,
    FULL_B,
    FULL_C,
    FULL_D,
    candidate,
    eval_config,
    internal_reward_policy,
    python_format_contract,
)
from whetstone.core.identity import IdentityRef, TypedRef, typed_ref_for_record
from whetstone.core.roles import EvaluationRole
from whetstone.evaluation import (
    DefinitionRef,
    EvalConfig,
    SamplePlan,
    SamplingDefinition,
    TaskSet,
)
from whetstone.evaluation.config import SCHEMA_EVAL_CONFIG, identity_hash_for
from whetstone.experiment.binding import (
    EvaluationBinding,
    eval_config_reference,
)
from whetstone.experiment.candidate import candidate_reference
from whetstone.optimization.contracts import (
    EvaluationIntent,
    OptimizationRun,
    OutputContract,
    StepMode,
    optimization_run_reference,
)
from whetstone.optimization.miprov2.bootstrap import BootstrapAttemptPlan
from whetstone.optimization.miprov2.control import (
    Miprov2InjectedDefaults,
    configure_miprov2,
)
from whetstone.optimization.miprov2.demo import LabeledTaskDemo
from whetstone.optimization.miprov2.eval_config import (
    Miprov2EvalConfigBinding,
    Miprov2EvalConfigBindingRequest,
    Miprov2EvaluationExecutionPolicy,
    derive_eval_config_reference,
)
from whetstone.optimization.miprov2.evidence import (
    Miprov2IntentContext,
    persist_miprov2_intent_context,
)
from whetstone.optimization.miprov2.proposal import (
    Miprov2DatasetExample,
    Miprov2PromptComponent,
)
from whetstone.optimization.miprov2.rng import Miprov2DurableBindings
from whetstone.optimization.miprov2.runtime import (
    Miprov2Driver,
    Miprov2EffectBudget,
    Miprov2State,
)
from whetstone.optimization.proposal.proposer import (
    FakeProposerTransport,
    ProposerConfig,
)
from whetstone.provider.language_model import PlainPromptAdapter

MIPROV2_TASK_IDENTITIES = tuple(f"{index:064x}" for index in range(1, 8))
MIPROV2_EVIDENCE_TASK_IDENTITY = hashlib.sha256(
    b"miprov2-evidence-task"
).hexdigest()


def miprov2_injected_defaults() -> Miprov2InjectedDefaults:

    provider = openrouter_chat_config(model="proposal-model")
    validation = eval_config_reference(eval_config(FULL_B))
    return Miprov2InjectedDefaults(
        prompt_model=ProposerConfig(
            provider_call_config=IdentityRef(
                record_ref=typed_ref_for_record(
                    "dr_providers.provider_call_config",
                    provider.model_dump(mode="json"),
                ),
                record_hash=provider.identity_hash,
            )
        ),
        bootstrap_eval_source=eval_config_reference(eval_config(FULL_A)),
        validation_eval_source=validation,
        reward_policy=internal_reward_policy(),
        evaluation_binding=EvaluationBinding(
            schema_version=3,
            eval_config=validation,
            role=EvaluationRole.INTERNAL,
            campaign="miprov2-control-test",
        ),
        provider_execution_policy_hash=FULL_B,
        task_model_identity_hash=FULL_C,
        prompt_adapter=PlainPromptAdapter(),
        template_render_contract=python_format_contract(),
        max_errors=3,
        validation_eval_source_is_metric_authority=True,
    )


def configure_test_miprov2(**updates: Any):

    values: dict[str, Any] = {
        "base_candidate": candidate_reference(
            candidate("base", text="Answer {query}.")
        ),
        "trainset": MIPROV2_TASK_IDENTITIES[:4],
        "valset": MIPROV2_TASK_IDENTITIES[4:],
        "auto": None,
        "num_candidates": 2,
        "num_trials": 2,
        "minibatch": False,
        "defaults": miprov2_injected_defaults(),
    }
    values.update(updates)
    return configure_miprov2(**values)


def persist_test_record(
    store: ObjectStore,
    schema: str,
    record: Jsonable,
) -> TypedRef:

    ref, _ = store.put(schema, record)
    return TypedRef(schema_name=ref.schema, content_hash=ref.content_hash)


def miprov2_evidence_source_eval_config():

    definition = DefinitionRef(
        definition_id="miprov2-evidence",
        version="1",
        schema_name="whetstone.eval.definition",
        identity_hash=FULL_A,
    )
    identity = identity_hash_for(
        schema=SCHEMA_EVAL_CONFIG,
        payload={
            "definition_identity": FULL_A,
            "sampling_config": FULL_B,
            "evaluation_procedure_config": FULL_C,
            "aggregation_config": FULL_D,
        },
    )
    return eval_config_reference(
        EvalConfig(
            definition_ref=definition,
            sampling_config_hash=FULL_B,
            evaluation_procedure_config_hash=FULL_C,
            aggregation_config_hash=FULL_D,
            config_hash=identity,
        )
    )


def miprov2_evidence_bindings(
    control_identity_hash: str = FULL_A,
) -> Miprov2DurableBindings:

    return Miprov2DurableBindings(
        control_identity_hash=control_identity_hash,
        prompt_route_identity_hash=FULL_B,
        task_route_identity_hash=FULL_C,
        execution_policy_identity_hash=FULL_D,
        prompt_adapter_identity_hash=FULL_A,
        proposal_executor_policy_identity_hash=FULL_B,
        proposal_transport_durability_identity_hash=FULL_C,
        base_candidate_identity_hash=FULL_B,
        teacher_candidate_identity_hash=FULL_C,
    )


def make_miprov2_evidence_fixture(
    store: ObjectStore,
    *,
    reward_policy_hash: str,
    control_identity_hash: str = FULL_A,
) -> tuple[EvaluationIntent, Miprov2IntentContext]:

    attempt = BootstrapAttemptPlan(
        bindings=miprov2_evidence_bindings(control_identity_hash),
        plan_identity_hash=FULL_D,
        task_index=0,
        task_hash=MIPROV2_EVIDENCE_TASK_IDENTITY,
        round_index=0,
        copy_task_model=False,
        rollout_id=None,
        temperature=None,
    )
    policy = Miprov2EvaluationExecutionPolicy(
        num_threads=1,
        max_errors=1,
        provide_traceback=None,
        task_model_identity_hash=FULL_C,
        provider_execution_policy_hash=FULL_D,
    )
    request = Miprov2EvalConfigBindingRequest(
        control_identity_hash=control_identity_hash,
        source_eval_config=miprov2_evidence_source_eval_config(),
        purpose="bootstrap",
        effect_identity_hash=attempt.identity_hash(),
        execution_policy=policy,
        task_batch_hashes=(MIPROV2_EVIDENCE_TASK_IDENTITY,),
    )
    task_set = TaskSet(
        manifest_id="miprov2-evidence-tasks",
        version="1",
        dataset_revision="test",
        task_hashes=(MIPROV2_EVIDENCE_TASK_IDENTITY,),
    )
    sample_plan = SamplePlan(
        plan_id="miprov2-evidence-repeats",
        version="1",
        task_hashes=(MIPROV2_EVIDENCE_TASK_IDENTITY,),
        num_samples=1,
    )
    sampling = SamplingDefinition(
        definition_id="miprov2-evidence-sampling",
        version="1",
    ).materialize(
        {
            "task_set_hash": task_set.identity_hash(),
            "sample_plan_hash": sample_plan.identity_hash(),
        }
    )
    eval_binding = Miprov2EvalConfigBinding(
        request=request,
        task_set=task_set,
        sample_plan=sample_plan,
        sampling_config=sampling,
        eval_config=derive_eval_config_reference(
            request.source_eval_config,
            sampling,
        ),
    )
    exact_binding = EvaluationBinding(
        schema_version=3,
        eval_config=eval_binding.eval_config,
        role=EvaluationRole.INTERNAL,
        campaign="miprov2-evidence",
    )
    candidate_ref = candidate_reference(
        candidate("teacher", text="Encode {query}.")
    )
    intent = EvaluationIntent(
        intent_id="run:miprov2:bootstrap:evidence",
        candidate=candidate_ref,
        target_eval_config=eval_binding.eval_config,
        evaluation_binding=exact_binding,
        purpose="miprov2_bootstrap",
        run_id="run",
        step_index=0,
        expected_reward_policy_hash=reward_policy_hash,
    )
    context = Miprov2IntentContext(
        control_identity_hash=control_identity_hash,
        run_id="run",
        effect_kind="bootstrap",
        effect_identity_hash=attempt.identity_hash(),
        intent_id=intent.intent_id,
        candidate=candidate_ref,
        task_batch_hashes=(MIPROV2_EVIDENCE_TASK_IDENTITY,),
        eval_config=eval_binding.eval_config,
        eval_config_binding=eval_binding,
        evaluation_binding=exact_binding,
        execution_policy=policy,
        reward_policy_hash=reward_policy_hash,
        bootstrap_attempt=attempt,
        optimizable_component_id="encode",
        optimizable_trace_index=0,
    )
    persist_miprov2_intent_context(store, context)
    return intent, context


def canonical_miprov2_eval_source(sampling_hash: str):

    definition = DefinitionRef(
        definition_id="runtime-eval",
        version="1",
        schema_name="whetstone.eval.definition",
        identity_hash=FULL_A,
    )
    identity = identity_hash_for(
        schema=SCHEMA_EVAL_CONFIG,
        payload={
            "definition_identity": definition.identity_hash,
            "sampling_config": sampling_hash,
            "evaluation_procedure_config": FULL_C,
            "aggregation_config": FULL_D,
        },
    )
    return eval_config_reference(
        EvalConfig(
            definition_ref=definition,
            sampling_config_hash=sampling_hash,
            evaluation_procedure_config_hash=FULL_C,
            aggregation_config_hash=FULL_D,
            config_hash=identity,
        )
    )


def make_minimal_miprov2_runtime(
    *,
    proposal_calls: int = 2,
    track_stats: bool = True,
) -> tuple[Miprov2Driver, Miprov2State]:

    bootstrap_source = canonical_miprov2_eval_source("1" * 64)
    validation_source = canonical_miprov2_eval_source("2" * 64)
    defaults = miprov2_injected_defaults().model_copy(
        update={
            "bootstrap_eval_source": bootstrap_source,
            "validation_eval_source": validation_source,
            "evaluation_binding": EvaluationBinding(
                schema_version=3,
                eval_config=validation_source,
                role=EvaluationRole.INTERNAL,
                campaign="miprov2-runtime-test",
            ),
        }
    )
    control = configure_test_miprov2(
        max_bootstrapped_demos=0,
        max_labeled_demos=1,
        program_aware_proposer=False,
        data_aware_proposer=False,
        tip_aware_proposer=False,
        fewshot_aware_proposer=False,
        num_trials=1,
        track_stats=track_stats,
        defaults=defaults,
    )
    component_id = control.component_ids[0]
    labeled = tuple(
        LabeledTaskDemo(
            source_task_hash=task_hash,
            inputs_by_component={component_id: {"query": f"q-{index}"}},
            outputs_by_component={component_id: {"answer": f"a-{index}"}},
        )
        for index, task_hash in enumerate(control.trainset_task_hashes)
    )
    proposal_trainset = tuple(
        Miprov2DatasetExample(
            task_hash=task_hash,
            rendered_record=f"query=q-{index}; answer=a-{index}",
        )
        for index, task_hash in enumerate(control.trainset_task_hashes)
    )
    bindings = Miprov2DurableBindings(
        control_identity_hash=control.identity_hash(),
        prompt_route_identity_hash=control.prompt_model.identity_hash(),
        task_route_identity_hash=control.task_model_identity_hash,
        execution_policy_identity_hash=(
            control.provider_execution_policy_hash
        ),
        prompt_adapter_identity_hash=control.prompt_adapter_identity_hash,
        proposal_executor_policy_identity_hash="c" * 64,
        proposal_transport_durability_identity_hash=(
            FakeProposerTransport(
                {},
                execution_policy_hash=(control.provider_execution_policy_hash),
                prompt_adapter_identity_hash=(
                    control.prompt_adapter_identity_hash
                ),
            ).durability_identity_hash
        ),
        base_candidate_identity_hash=control.base_candidate.identity_hash,
        teacher_candidate_identity_hash=control.teacher_candidate.identity_hash,
    )
    driver = Miprov2Driver()
    run = optimization_run_reference(
        OptimizationRun(
            run_id="miprov2-runtime-test",
            optimizer_config=control.reference(),
            adapter_key="miprov2",
            mode=StepMode.PROPOSAL_ONLY,
            terminal_output_contract=OutputContract(returned_proposal_count=1),
            template_render_contract=control.template_render_contract,
            reward_policy=control.reward_policy,
        )
    )
    state = driver.start(
        run=run,
        control=control,
        bindings=bindings,
        labeled_trainset=labeled,
        proposal_components=(
            Miprov2PromptComponent(
                component_id=component_id,
                template=control.base_candidate.record.payload[
                    "user_prompt_template"
                ],
                template_render_contract=control.template_render_contract,
                rendering_rules="Substitute the native query field.",
                example_execution="Answer q-0.",
            ),
        ),
        proposal_trainset=proposal_trainset,
        component_field_order={component_id: ("query", "answer")},
        budget=Miprov2EffectBudget(
            bootstrap_rollouts=0,
            proposal_calls=proposal_calls,
            evaluations=2,
            task_rows=6,
        ),
    )
    return driver, Miprov2State.model_validate_json(state.model_dump_json())


def resolve_miprov2_eval_config_binding(
    request: Miprov2EvalConfigBindingRequest,
) -> Miprov2EvalConfigBinding:

    suffix = request.identity_hash()[:20]
    task_set = TaskSet(
        manifest_id=f"miprov2-runtime-tasks-{suffix}",
        version="1",
        dataset_revision="test",
        task_hashes=request.task_batch_hashes,
    )
    sample_plan = SamplePlan(
        plan_id=f"miprov2-runtime-repeats-{suffix}",
        version="1",
        task_hashes=request.task_batch_hashes,
        num_samples=request.num_samples,
    )
    sampling = SamplingDefinition(
        definition_id="miprov2-runtime-sampling",
        version="1",
    ).materialize(
        {
            "task_set_hash": task_set.identity_hash(),
            "sample_plan_hash": sample_plan.identity_hash(),
        }
    )
    return Miprov2EvalConfigBinding(
        request=request,
        task_set=task_set,
        sample_plan=sample_plan,
        sampling_config=sampling,
        eval_config=derive_eval_config_reference(
            request.source_eval_config,
            sampling,
        ),
    )
