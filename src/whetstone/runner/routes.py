"""Provider route registry for the validation runner.

A **route** binds one native dr-providers Provider Call Config (the encoder /
decoder / proposer wire shape) to one native Provider Transport Policy (base
URL, key env, timeout, native-retry pin) plus the Whetstone
:class:`~whetstone.provider.policy.ProviderExecutionPolicy` that wraps the
transport policy with semantic-retry concerns.

Two families are registered:

* **Canonical (OpenRouter, chat-completions):** the ``task`` route
  (``openai/gpt-5-nano``) and the ``proposer`` route
  (``openai/gpt-5.4-nano``), keyed off ``OPENROUTER_API_KEY``.
* **Plan lanes (anthropic-messages protocol):** the four exhaustible free
  windows -- ``kimi`` / ``glm`` / ``minimax`` / ``stepfun``. These are
  alternate routes selectable per run (``--lane``) as proposer/task stand-ins
  for debug iterations; identity changes from a model swap are fine under
  internal contexts.

Transport policy is progress-aware: an absolute wall-clock cap
(``timeout_seconds``) plus a progress/idle timeout (``idle_timeout_seconds``)
so a legitimate long streaming response from a reasoning model -- steady tokens
over many minutes -- is bounded by inactivity, not by total wall-clock. Native
retries are pinned to ``0``, because Whetstone owns all semantic retry.
Everything here is config-identity only: no live call is made by constructing a
route.
"""

from __future__ import annotations

from dataclasses import dataclass

from dr_providers import (
    DEFAULT_IDLE_TIMEOUT_SECONDS,
    GenerationControls,
    ProviderCallConfig,
    ProviderKind,
    ProviderTransportPolicy,
    ReasoningEffort,
    anthropic_messages_config,
    openai_chat_config,
    openrouter_chat_config,
    policy_for,
)

from whetstone.provider.policy import (
    BackoffSchedule,
    ProviderExecutionPolicy,
    default_retry_eligibility,
)

__all__ = [
    "CANONICAL_PROPOSER_MODEL",
    "CANONICAL_TASK_MODEL",
    "COMPLETENESS_BY_ENV",
    "DEEPSEEK_TASK_MODEL",
    "DEFAULT_IDLE_SECONDS",
    "DEFAULT_TIMEOUT_SECONDS",
    "ENCDEC_DEFAULT_TASK_MODEL",
    "LANE_NAMES",
    "OPENAI_BASE_URL",
    "OPENAI_KEY_ENV",
    "OPENROUTER_BASE_URL",
    "OPENROUTER_KEY_ENV",
    "PLAN_LANES",
    "REASONING_EFFORT_CHOICES",
    "TASK_MODEL_BY_ENV",
    "PlanLane",
    "ProviderRoute",
    "canonical_proposer_route",
    "canonical_task_route",
    "completeness_for_env",
    "lane_route",
    "openai_direct_route",
    "reasoning_effort_for",
    "route_for",
    "task_model_for_env",
]

#: The absolute wall-clock cap (seconds) for a single wire call. It is sized
#: so a legitimate long reasoning-model generation is not capped mid-stream:
#: the cap is only the dribble backstop, and the idle timeout below is the real
#: stall detector.
DEFAULT_TIMEOUT_SECONDS = 600.0

#: The progress/idle timeout (seconds): a call fails ``stalled_response`` only
#: when no bytes arrive for this long. A stream making steady progress, even
#: for many minutes, never trips it; a genuinely wedged edge fails quickly.
DEFAULT_IDLE_SECONDS = DEFAULT_IDLE_TIMEOUT_SECONDS

#: OpenRouter canonical model slugs.
CANONICAL_TASK_MODEL = "openai/gpt-5-nano"
CANONICAL_PROPOSER_MODEL = "openai/gpt-5.4-nano"

#: An alternate task model for the constraint-heavy envs.
DEEPSEEK_TASK_MODEL = "deepseek/deepseek-v4-flash"

#: Default task model for the enc-dec family. The deepseek model is by far the
#: slowest in the funnel latency preview, so it is an explicit choice via
#: ``--task-model`` (the contamination axis), never a default.
ENCDEC_DEFAULT_TASK_MODEL = "google/gemini-3.1-flash-lite"

#: Per-env default task model. The chosen model folds into the Provider Call
#: Config (hence the graph hash) and is recorded on the cell line under
#: ``models.task``. The ``--task-model`` CLI flag overrides this default.
TASK_MODEL_BY_ENV: dict[str, str] = {
    # The code_comp env (direct, encdec, encdec_mutant) shares one task-model
    # family so anchor cells pair on the same model.
    "code_comp": ENCDEC_DEFAULT_TASK_MODEL,
}


#: Per-env default completeness tolerance. A missing entry is the strict,
#: untolerant default -- propagate with ``max_skip_fraction`` 0.0 -- so any
#: missing or failed row makes the official arm incomplete. The tolerance is
#: identity-bearing: a tolerant cell has a distinct ``eval_config_hash`` from a
#: strict one. Value: ``(missing_data, fraction)``.
COMPLETENESS_BY_ENV: dict[str, tuple[str, float]] = {
    # code_comp rows declare a higher tolerance for stochastic model behavior.
    "code_comp": ("skip", 0.15),
}


def completeness_for_env(env: str) -> tuple[str, float]:
    """The ``(missing_data, max_skip_fraction)`` matrix default for an env.

    Returns the env's declared completeness tolerance, or the strict
    untolerant default (``("propagate", 0.0)``) for any env not listed.
    """
    return COMPLETENESS_BY_ENV.get(env, ("propagate", 0.0))


def task_model_for_env(env: str, *, override: str | None = None) -> str:
    """The task model for an env: explicit override, else the matrix default.

    ``override`` (the ``--task-model`` flag) wins when given; otherwise the
    per-env matrix default (:data:`TASK_MODEL_BY_ENV`) applies, falling back to
    the canonical task model for any env not listed. The returned slug folds
    into the task route's Provider Call Config identity, so a deepseek cell's
    route identity differs from a canonical one's.
    """
    if override:
        return override
    return TASK_MODEL_BY_ENV.get(env, CANONICAL_TASK_MODEL)


#: The env var carrying the OpenRouter credential.
OPENROUTER_KEY_ENV = "OPENROUTER_API_KEY"

#: The OpenRouter API base URL (chat-completions). Every canonical route pins
#: this so the transport policy has a non-None base_url and pre-flight passes.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

#: The OpenAI direct lane (``lane="openai"``): the OpenAI API keyed off
#: ``OPENAI_API_KEY`` over the same chat-completions protocol as OpenRouter,
#: through OpenAI's own provider. The OpenAI provider respects ``temperature``
#: for ``gpt-5.4-nano`` where OpenRouter ignores it, so a temperature-sensitive
#: study must go direct. The lane folds into route and config identity -- its
#: ProviderKind is OPENAI, a distinct identity hash from the OpenRouter config
#: for the same model -- so the cell line records the provider distinctly.
OPENAI_KEY_ENV = "OPENAI_API_KEY"
OPENAI_BASE_URL = "https://api.openai.com/v1"


@dataclass(frozen=True, slots=True)
class PlanLane:
    """One anthropic-messages plan lane.

    ``base_url`` and ``key_env`` name the endpoint and its credential; the
    model is the endpoint's advertised model. Plan lanes are exhaustible free
    windows used as proposer/task stand-ins for debug iterations only.
    """

    name: str
    model: str
    base_url: str
    key_env: str


#: The four plan lanes with their base URLs and key envs.
PLAN_LANES: dict[str, PlanLane] = {
    "kimi": PlanLane(
        name="kimi",
        model="k2p7",
        base_url="https://api.kimi.com/coding",
        key_env="KIMI_CODE_API_KEY",
    ),
    "glm": PlanLane(
        name="glm",
        model="glm-5.1",
        base_url="https://api.z.ai/api/anthropic",
        key_env="ZAI_API_KEY",
    ),
    "minimax": PlanLane(
        name="minimax",
        model="MiniMax-M3",
        base_url="https://api.minimax.io/anthropic",
        key_env="MINIMAX_API_KEY",
    ),
    "stepfun": PlanLane(
        name="stepfun",
        model="step-3.7-flash",
        base_url="https://api.stepfun.ai/step_plan",
        key_env="STEPFUN_API_KEY",
    ),
}

#: Ordered plan-lane names (the priming order).
LANE_NAMES: tuple[str, ...] = ("kimi", "glm", "minimax", "stepfun")


@dataclass(frozen=True, slots=True)
class ProviderRoute:
    """A selectable route: Provider Call Config plus policies.

    ``call_config`` is the native dr-providers config, whose identity hash is
    the graph route identity. ``transport_policy`` carries the base URL, key
    env, timeout, and native-retry pin. ``execution_policy`` is the Whetstone
    semantic policy the attempt-loop driver consumes. ``lane`` is
    ``"openrouter"`` for the canonical routes, ``"openai"`` for the direct
    lane, or a plan-lane name.
    """

    role: str
    lane: str
    model: str
    call_config: ProviderCallConfig
    transport_policy: ProviderTransportPolicy
    execution_policy: ProviderExecutionPolicy

    @property
    def key_env(self) -> str:
        return self.transport_policy.api_key_env

    def identity_summary(self) -> dict[str, object]:
        """A config-identity summary (no secret material) for the report."""
        return {
            "role": self.role,
            "lane": self.lane,
            "model": self.model,
            "call_config_hash": self.call_config.identity_hash,
            "execution_policy_hash": self.execution_policy.identity_hash,
            "key_env": self.key_env,
            "base_url": self.transport_policy.base_url,
            "timeout_seconds": self.transport_policy.timeout_seconds,
            "native_retry_count": self.transport_policy.native_retry_count,
        }


def _execution_policy(
    transport_policy: ProviderTransportPolicy,
    *,
    max_attempts: int,
) -> ProviderExecutionPolicy:
    """Wrap a transport policy with the semantic-retry defaults."""
    return ProviderExecutionPolicy(
        transport_policy=transport_policy,
        max_attempts=max_attempts,
        retry_eligibility=default_retry_eligibility(),
        backoff=BackoffSchedule(),
    )


def _controls(
    temperature: float | None,
    reasoning: ReasoningEffort | None = None,
    token_limit: int | None = None,
) -> GenerationControls | None:
    """The GenerationControls for a route, or ``None`` when nothing is set.

    ``reasoning`` (the ``--reasoning-effort`` dial) is output-affecting: it
    serializes on the wire per the config's reasoning shape (OpenRouter emits a
    ``reasoning`` object, OpenAI a ``reasoning_effort`` field) and folds into
    the Provider Call Config identity hash, so a distinct effort is a distinct
    route variant. ``None`` leaves the control unset, so the provider default
    applies.
    """
    if temperature is None and reasoning is None and token_limit is None:
        return None
    return GenerationControls(
        temperature=temperature,
        reasoning=reasoning,
        token_limit=token_limit,
    )


def canonical_task_route(
    *,
    model: str = CANONICAL_TASK_MODEL,
    temperature: float | None = 0.0,
    reasoning: ReasoningEffort | None = None,
    token_limit: int | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    idle_timeout_seconds: float = DEFAULT_IDLE_SECONDS,
    max_attempts: int = 3,
) -> ProviderRoute:
    """The canonical OpenRouter task (encoder/decoder) route.

    ``openai/gpt-5-nano`` over chat-completions, keyed off
    ``OPENROUTER_API_KEY``. Temperature defaults to 0; pass ``None`` to leave
    the control unset. The absolute cap accommodates reasoning-model
    generations while the idle timeout is the real stall detector.
    ``reasoning`` and ``token_limit`` are output-affecting controls that fold
    into the config identity.
    """
    call_config = openrouter_chat_config(
        model=model,
        controls=_controls(temperature, reasoning, token_limit),
    )
    transport_policy = policy_for(
        ProviderKind.OPENROUTER,
        api_key_env=OPENROUTER_KEY_ENV,
        base_url=OPENROUTER_BASE_URL,
        timeout_seconds=timeout_seconds,
        idle_timeout_seconds=idle_timeout_seconds,
        native_retry_count=0,
    )
    return ProviderRoute(
        role="task",
        lane="openrouter",
        model=model,
        call_config=call_config,
        transport_policy=transport_policy,
        execution_policy=_execution_policy(
            transport_policy, max_attempts=max_attempts
        ),
    )


def openai_direct_route(
    *,
    role: str = "task",
    model: str = CANONICAL_TASK_MODEL,
    temperature: float | None = 0.0,
    reasoning: ReasoningEffort | None = None,
    token_limit: int | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    idle_timeout_seconds: float = DEFAULT_IDLE_SECONDS,
    max_attempts: int = 3,
) -> ProviderRoute:
    """The OpenAI direct route (``lane="openai"``): OpenAI's own API.

    The same chat-completions protocol and shape as the OpenRouter transport,
    but keyed off ``OPENAI_API_KEY`` at :data:`OPENAI_BASE_URL` via
    ``ProviderKind.OPENAI`` -- so the config identity, hence the graph route
    identity, is distinct from the OpenRouter route for the same model. Chosen
    when temperature must hold. ``reasoning`` serializes as
    ``reasoning_effort``; it and ``token_limit`` fold into config identity.
    """
    call_config = openai_chat_config(
        model=model,
        controls=_controls(temperature, reasoning, token_limit),
    )
    transport_policy = policy_for(
        ProviderKind.OPENAI,
        api_key_env=OPENAI_KEY_ENV,
        base_url=OPENAI_BASE_URL,
        timeout_seconds=timeout_seconds,
        idle_timeout_seconds=idle_timeout_seconds,
        native_retry_count=0,
    )
    return ProviderRoute(
        role=role,
        lane="openai",
        model=model,
        call_config=call_config,
        transport_policy=transport_policy,
        execution_policy=_execution_policy(
            transport_policy, max_attempts=max_attempts
        ),
    )


def canonical_proposer_route(
    *,
    model: str = CANONICAL_PROPOSER_MODEL,
    temperature: float | None = 1.0,
    reasoning: ReasoningEffort | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    idle_timeout_seconds: float = DEFAULT_IDLE_SECONDS,
    max_attempts: int = 3,
) -> ProviderRoute:
    """The canonical OpenRouter proposer route.

    ``openai/gpt-5.4-nano`` over chat-completions, keyed off
    ``OPENROUTER_API_KEY``. Its config identity is distinct from the task
    route's -- a different model and temperature -- so it never collides with
    an encoder/decoder route hash.
    """
    call_config = openrouter_chat_config(
        model=model, controls=_controls(temperature, reasoning)
    )
    transport_policy = policy_for(
        ProviderKind.OPENROUTER,
        api_key_env=OPENROUTER_KEY_ENV,
        base_url=OPENROUTER_BASE_URL,
        timeout_seconds=timeout_seconds,
        idle_timeout_seconds=idle_timeout_seconds,
        native_retry_count=0,
    )
    return ProviderRoute(
        role="proposer",
        lane="openrouter",
        model=model,
        call_config=call_config,
        transport_policy=transport_policy,
        execution_policy=_execution_policy(
            transport_policy, max_attempts=max_attempts
        ),
    )


def lane_route(
    lane: str,
    *,
    role: str = "task",
    temperature: float | None = 0.0,
    reasoning: ReasoningEffort | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    idle_timeout_seconds: float = DEFAULT_IDLE_SECONDS,
    max_attempts: int = 3,
) -> ProviderRoute:
    """An anthropic-messages plan-lane route (kimi/glm/minimax/stepfun).

    Base URL and key env come from the :data:`PLAN_LANES` table. These are
    alternate routes selectable per run: debug stand-ins for the OpenRouter
    canonical routes, never burned as canonical.
    """
    if lane not in PLAN_LANES:
        raise ValueError(
            f"unknown plan lane {lane!r}; expected one of {LANE_NAMES}"
        )
    spec = PLAN_LANES[lane]
    call_config = anthropic_messages_config(
        model=spec.model,
        controls=_controls(temperature, reasoning, token_limit=4096),
    )
    transport_policy = policy_for(
        ProviderKind.ANTHROPIC,
        api_key_env=spec.key_env,
        base_url=spec.base_url,
        timeout_seconds=timeout_seconds,
        idle_timeout_seconds=idle_timeout_seconds,
        native_retry_count=0,
    )
    return ProviderRoute(
        role=role,
        lane=lane,
        model=spec.model,
        call_config=call_config,
        transport_policy=transport_policy,
        execution_policy=_execution_policy(
            transport_policy, max_attempts=max_attempts
        ),
    )


#: The CLI ``--reasoning-effort`` choices mapped to the typed cross-provider
#: effort. ``none`` maps to ``ReasoningEffort.NONE`` (OpenRouter serializes it
#: as ``{reasoning: {enabled: false}}``; OpenAI as the minimal effort the API
#: allows). An absent flag leaves the control at the provider default.
REASONING_EFFORT_CHOICES: tuple[str, ...] = ("none", "low", "medium", "high")
_REASONING_BY_NAME: dict[str, ReasoningEffort] = {
    "none": ReasoningEffort.NONE,
    "low": ReasoningEffort.LOW,
    "medium": ReasoningEffort.MEDIUM,
    "high": ReasoningEffort.HIGH,
}


def reasoning_effort_for(name: str | None) -> ReasoningEffort | None:
    """Map a ``--reasoning-effort`` choice to the typed effort, or ``None``.

    ``None`` leaves the control unset, so the provider default applies.
    """
    if name is None:
        return None
    if name not in _REASONING_BY_NAME:
        raise ValueError(
            f"unknown reasoning effort {name!r}; expected one of "
            f"{REASONING_EFFORT_CHOICES}"
        )
    return _REASONING_BY_NAME[name]


def route_for(
    lane: str,
    *,
    role: str = "task",
    temperature: float | None = 0.0,
    reasoning: ReasoningEffort | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    idle_timeout_seconds: float = DEFAULT_IDLE_SECONDS,
    max_attempts: int = 3,
    task_model: str | None = None,
    proposer_model: str | None = None,
) -> ProviderRoute:
    """Select the route for a lane and role.

    ``lane="openrouter"`` returns the canonical task or proposer route by role;
    ``lane="openai"`` returns the direct OpenAI route; any plan-lane name
    returns that lane's anthropic-messages route. ``task_model`` selects a
    per-env task model, folding into the route's config identity so a deepseek
    route differs from a canonical one. ``proposer_model`` overrides the
    canonical proposer model on the proposer role. ``reasoning`` is
    output-affecting and folds into the config identity; ``None`` leaves it at
    the provider default.
    """
    if lane == "openrouter":
        if role == "proposer":
            return canonical_proposer_route(
                model=proposer_model or CANONICAL_PROPOSER_MODEL,
                temperature=1.0 if temperature is None else temperature,
                reasoning=reasoning,
                timeout_seconds=timeout_seconds,
                idle_timeout_seconds=idle_timeout_seconds,
                max_attempts=max_attempts,
            )
        return canonical_task_route(
            model=task_model or CANONICAL_TASK_MODEL,
            temperature=temperature,
            reasoning=reasoning,
            timeout_seconds=timeout_seconds,
            idle_timeout_seconds=idle_timeout_seconds,
            max_attempts=max_attempts,
        )
    if lane == "openai":
        # OpenAI direct: the same chat-completions shape through OpenAI's own
        # provider. The proposer default keeps the canonical proposer model.
        if role == "proposer":
            return openai_direct_route(
                role="proposer",
                model=proposer_model or CANONICAL_PROPOSER_MODEL,
                temperature=1.0 if temperature is None else temperature,
                reasoning=reasoning,
                timeout_seconds=timeout_seconds,
                idle_timeout_seconds=idle_timeout_seconds,
                max_attempts=max_attempts,
            )
        return openai_direct_route(
            role=role,
            model=task_model or CANONICAL_TASK_MODEL,
            temperature=temperature,
            reasoning=reasoning,
            timeout_seconds=timeout_seconds,
            idle_timeout_seconds=idle_timeout_seconds,
            max_attempts=max_attempts,
        )
    return lane_route(
        lane,
        role=role,
        temperature=temperature,
        reasoning=reasoning,
        timeout_seconds=timeout_seconds,
        idle_timeout_seconds=idle_timeout_seconds,
        max_attempts=max_attempts,
    )
