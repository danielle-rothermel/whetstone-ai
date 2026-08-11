from __future__ import annotations

from dr_providers import (
    ControlConstraints,
    GenerationControls,
    ModelRoute,
    Protocol,
    ProviderCallConfig,
    ProviderCallDefinition,
    ProviderKind,
    ReasoningRequestShape,
    RequestControl,
    TokenLimitParameter,
)

from tests.envs.support import ReplyFn, execution_policy, row_job_factory
from whetstone.core.identity import IdentityHash, TypedRef
from whetstone.core.roles import EvaluationRole
from whetstone.envs.factory import EnvExperiment
from whetstone.envs.oracle_operator import env_exact_match_score
from whetstone.envs.registry import env_spec
from whetstone.envs.rollout_definition import LLM_NODE_ID
from whetstone.evaluation.drivers.internal import (
    InternalRowOutcome,
    InternalRowRequest,
    ProcessTask,
)
from whetstone.evaluation.traces import ExecutedRowState, _llm_component_step
from whetstone.experiment.binding import (
    EVALUATION_BINDING_SCHEMA_VERSION,
    EvaluationBinding,
    eval_config_reference,
)
from whetstone.experiment.candidate import Candidate


def _successful_internal_outcome(
    *, prompt: str = "test prompt", output_text: str = "test output"
) -> InternalRowOutcome:
    return InternalRowOutcome(
        score=1.0,
        row_state=ExecutedRowState.SUCCESS,
        executed_component_steps=(
            _llm_component_step(
                trace_index=0,
                component_id=LLM_NODE_ID,
                prompt=prompt,
                generation=output_text,
            ),
        ),
        output_text=output_text,
    )


def _binding(
    exp: EnvExperiment,
    *,
    role: EvaluationRole = EvaluationRole.INTERNAL,
) -> EvaluationBinding:
    sampling = (
        exp.eval_configs.internal
        if role is EvaluationRole.INTERNAL
        else exp.eval_configs.official
    )
    return EvaluationBinding(
        schema_version=EVALUATION_BINDING_SCHEMA_VERSION,
        eval_config=eval_config_reference(sampling.eval_config),
        role=role,
        authority_principal=(
            "test-authority" if role is EvaluationRole.OFFICIAL else None
        ),
        campaign="env-test",
    )


def _correct_reply(env_name: str, tasks) -> ReplyFn:
    env = env_spec(env_name)
    from whetstone.envs.rollout_definition import (
        initial_candidate,
        render_prompt,
    )

    naive = initial_candidate(env)
    correct_by_prompt: dict[str, str] = {}
    for inst in tasks:
        prompt = render_prompt(env, naive, inst)
        correct_by_prompt[prompt] = _correct_generation(env, inst)

    def reply(prompt: str) -> str:
        return correct_by_prompt.get(prompt, "")

    return reply


def _correct_generation(env, instance) -> str:
    if env.name == "c22":
        return instance.gold
    return instance.gold


def _internal_jobs(
    experiment: EnvExperiment,
    reply: ReplyFn,
    *,
    candidate: Candidate | None = None,
    served: list[str] | None = None,
):
    env = env_spec(experiment.env_name)
    active_candidate = candidate or experiment.initial_candidate
    procedure_hash = experiment.eval_configs.procedure_config_hash

    def outcome(instance, _repeat: int, _drive_ordinal: int):
        from whetstone.envs.rollout_definition import render_prompt

        prompt = render_prompt(env, active_candidate, instance)
        if served is not None:
            served.append(prompt)
        text = reply(prompt)
        if not text.strip():
            return InternalRowOutcome(
                score=None,
                row_state=ExecutedRowState.FAILED,
                executed_component_steps=(),
                failure_code="blank_generation",
            )
        score = env_exact_match_score(
            env=env,
            generation=text,
            gold=instance.gold,
            evaluation_procedure_config_hash=procedure_hash,
        )
        outcome = _successful_internal_outcome(prompt=prompt, output_text=text)
        return outcome.model_copy(update={"score": float(score.value)})

    return row_job_factory(outcome)


def _fixed_unordered_provider_request() -> InternalRowRequest:
    controls = frozenset(
        {
            RequestControl.REASONING,
            RequestControl.TEMPERATURE,
            RequestControl.TOKEN_LIMIT,
            RequestControl.TOP_P,
        }
    )
    provider_call_config = ProviderCallConfig(
        definition=ProviderCallDefinition(
            definition_id="test.chat_completions",
            route=ModelRoute(
                provider=ProviderKind.OPENROUTER,
                protocol=Protocol.CHAT_COMPLETIONS,
                model="test/model",
            ),
            constraints=ControlConstraints(
                supported_controls=controls,
                token_limit_parameter=(
                    TokenLimitParameter.MAX_COMPLETION_TOKENS
                ),
                reasoning_shape=ReasoningRequestShape.REASONING_OBJECT,
            ),
            required_controls=frozenset(
                {
                    RequestControl.TEMPERATURE,
                    RequestControl.TOKEN_LIMIT,
                }
            ),
            extension_keys=frozenset({"alpha", "omega"}),
        ),
        controls=GenerationControls(temperature=0.0, token_limit=1),
    )
    return InternalRowRequest(
        env_name="c18",
        candidate=Candidate(
            candidate_id="fixed-candidate",
            base_ref=TypedRef(
                schema_name="fixed-candidate-base",
                content_hash="1" * 64,
            ),
        ),
        instance=ProcessTask(
            id="fixed-instance",
            seed=1,
            strata=("fixed",),
            prompt_inputs={"question": "Fixed question?", "query": "True"},
            gold="True",
        ),
        provider_call_config=provider_call_config,
        execution_policy=execution_policy(),
        procedure_config_hash="2" * 64,
        evaluation_binding_hash=IdentityHash("3" * 64),
        logical_call_id="fixed-call#0",
        sample_index=0,
        drive_ordinal=0,
        cache_phase="internal_eval",
        cache_unit="fixed-candidate",
        cache_root=None,
        render_guard=False,
    )
