"""Non-collected builders for internal evaluation driver tests."""

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
from whetstone.envs.factory import EnvExperiment, build_env_experiment
from whetstone.envs.oracle_operator import env_exact_match_score
from whetstone.envs.registry import env_spec
from whetstone.envs.rollout_definition import LLM_NODE_ID
from whetstone.evaluation.drivers.internal import (
    InternalRowOutcome,
    InternalRowRequest,
    ProcessInstance,
)
from whetstone.evaluation.traces import ExecutedRowState, _llm_component_step
from whetstone.experiment.binding import (
    EVALUATION_BINDING_SCHEMA_VERSION,
    EvaluationBinding,
    eval_config_reference,
)
from whetstone.experiment.candidate import Candidate

_MODEL = "openai/gpt-5-nano"
_SPLIT = (2, 2, 2)
# At n_per_stratum=sum(_SPLIT), even one stratum supplies the complete split;
# additional strata can only increase capacity. This bound is independent of
# the generated pool sizes observed by the fit loop.
_SPLIT_FIT_CEILING = sum(_SPLIT)


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


def _tiny_experiment(env_name: str) -> EnvExperiment:
    # n_per_stratum=1 gives >= 4 instances (all envs have >= 4 strata except
    # c18 which has 4); a (2,2,2) split needs >= 6, so grow the pool until it
    # is large enough. For a stratified-split env (c22, whose pool is blocked)
    # each stratum must independently hold its per-stratum quota, so grow until
    # the stratified split is satisfiable rather than only until the total
    # instance count clears sum(_SPLIT).
    env = env_spec(env_name)
    attempted_sizes: list[int] = []
    for n in range(1, _SPLIT_FIT_CEILING + 1):
        attempted_sizes.append(n)
        if _split_fits(env, n):
            break
    else:
        raise AssertionError(
            f"{env_name} could not fit split {_SPLIT} by independently "
            f"derived n_per_stratum ceiling {_SPLIT_FIT_CEILING}; "
            f"attempted_sizes={attempted_sizes}; "
            f"final_attempted_size={attempted_sizes[-1]}"
        )
    return build_env_experiment(
        env_name,
        model=_MODEL,
        pool_n_per_stratum=n,
        split_sizes=_SPLIT,
        repeats=2,
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


def _split_fits(env, n: int) -> bool:
    """True once a pool at ``n_per_stratum=n`` can serve the ``_SPLIT`` totals.

    For a contiguous-split env the whole pool need only exceed ``sum(_SPLIT)``;
    for a stratified-split env each stratum must hold its per-stratum quota, so
    ``n`` must clear the largest single-stratum draw.
    """
    pool = env.generate_pool(n_per_stratum=n)
    if not env.stratified_split:
        return len(pool) >= sum(_SPLIT)
    n_strata = len(pool.strata)
    per_stratum_max = sum(
        -(-part // n_strata)
        for part in _SPLIT  # ceil division per split part
    )
    return n >= per_stratum_max


def _correct_reply(env_name: str, instances) -> ReplyFn:
    """A reply fn that returns the correct answer for the matching task.

    The env oracle grades the generation against each task's gold; the fake
    returns each task's own correct answer keyed off its rendered prompt so
    the internal eval scores a clean 1.0.
    """
    env = env_spec(env_name)
    from whetstone.envs.rollout_definition import (
        initial_candidate,
        render_prompt,
    )

    naive = initial_candidate(env)
    # Map rendered-naive-prompt -> the correct generation for that instance.
    correct_by_prompt: dict[str, str] = {}
    for inst in instances:
        prompt = render_prompt(env, naive, inst)
        correct_by_prompt[prompt] = _correct_generation(env, inst)

    def reply(prompt: str) -> str:
        return correct_by_prompt.get(prompt, "")

    return reply


def _correct_generation(env, instance) -> str:
    """The known-correct generation for an instance (per env)."""
    if env.name == "c22":
        # A response satisfying whatever stack the instance carries is
        # instance-specific; the c22 internal-eval test uses a hand-built
        # single-instance fixture instead (see the c22-specific test below).
        return instance.gold
    # For the re-derive envs the gold IS the correct answer.
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
        instance=ProcessInstance(
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
        repeat_index=0,
        drive_ordinal=0,
        cache_phase="internal_eval",
        cache_unit="fixed-candidate",
        cache_root=None,
        render_guard=False,
    )
