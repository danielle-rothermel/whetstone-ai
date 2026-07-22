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
  windows -- ``kimi`` / ``glm`` / ``minimax`` / ``stepfun`` -- whose base URLs
  and key envs come from ``reports/window-starts.json``. These are ALTERNATE
  routes selectable per run (``--lane``) as proposer/task stand-ins for debug
  iterations; identity changes from a model swap are fine under internal
  contexts.

Transport policy is sane by default: ``timeout ~120s``, native retries ``0``
(Whetstone owns all semantic retry). Everything here is config-identity only --
no live call is made by constructing a route.
"""

from __future__ import annotations

from dataclasses import dataclass

from dr_providers import (
    GenerationControls,
    ProviderCallConfig,
    ProviderTransportPolicy,
    anthropic_messages_config,
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
    "LANE_NAMES",
    "OPENROUTER_KEY_ENV",
    "PLAN_LANES",
    "PlanLane",
    "ProviderRoute",
    "canonical_proposer_route",
    "canonical_task_route",
    "lane_route",
    "route_for",
]

#: The default transport timeout (seconds). The validation plan pins ~120s.
DEFAULT_TIMEOUT_SECONDS = 120.0

#: OpenRouter canonical model slugs (window-starts.json).
CANONICAL_TASK_MODEL = "openai/gpt-5-nano"
CANONICAL_PROPOSER_MODEL = "openai/gpt-5.4-nano"

#: The env var carrying the OpenRouter credential.
OPENROUTER_KEY_ENV = "OPENROUTER_API_KEY"


@dataclass(frozen=True, slots=True)
class PlanLane:
    """One anthropic-messages plan lane from ``window-starts.json``.

    ``base_url`` and ``key_env`` are read verbatim from the window snapshot;
    the model is the endpoint's advertised model. Plan lanes are exhaustible
    free windows used as proposer/task stand-ins for debug iterations only.
    """

    name: str
    model: str
    base_url: str
    key_env: str


#: The four plan lanes, base URLs + key envs from window-starts.json.
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

#: Ordered plan-lane names (the priming order from the plan windows section).
LANE_NAMES: tuple[str, ...] = ("kimi", "glm", "minimax", "stepfun")


@dataclass(frozen=True, slots=True)
class ProviderRoute:
    """A selectable route: Provider Call Config + policies.

    ``call_config`` is the native dr-providers Config (its Identity Hash is the
    graph route identity). ``transport_policy`` carries the base URL, key env,
    timeout, native-retry pin. ``execution_policy`` is the Whetstone semantic
    policy the attempt-loop driver consumes. ``lane`` is ``"openrouter"`` for
    the canonical routes or a plan-lane name.
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
    """Wrap a transport policy with the sane semantic-retry defaults."""
    return ProviderExecutionPolicy(
        transport_policy=transport_policy,
        max_attempts=max_attempts,
        retry_eligibility=default_retry_eligibility(),
        backoff=BackoffSchedule(),
    )


def _controls(temperature: float | None) -> GenerationControls | None:
    if temperature is None:
        return None
    return GenerationControls(temperature=temperature)


def canonical_task_route(
    *,
    model: str = CANONICAL_TASK_MODEL,
    temperature: float | None = 0.0,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_attempts: int = 3,
) -> ProviderRoute:
    """The canonical OpenRouter task (encoder/decoder) route.

    ``openai/gpt-5-nano`` over chat-completions, keyed off
    ``OPENROUTER_API_KEY``. Temperature defaults to 0 (the pilots use temp-0
    for agreement checks); pass ``None`` to leave the control unset.
    """
    call_config = openrouter_chat_config(
        model=model, controls=_controls(temperature)
    )
    transport_policy = policy_for(
        api_key_env=OPENROUTER_KEY_ENV,
        timeout_seconds=timeout_seconds,
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


def canonical_proposer_route(
    *,
    model: str = CANONICAL_PROPOSER_MODEL,
    temperature: float | None = 1.0,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_attempts: int = 3,
) -> ProviderRoute:
    """The canonical OpenRouter proposer route.

    ``openai/gpt-5.4-nano`` over chat-completions, keyed off
    ``OPENROUTER_API_KEY``. Its Config identity is distinct from the task
    route's (different model, different temperature), so it never collides with
    an encoder/decoder route hash.
    """
    call_config = openrouter_chat_config(
        model=model, controls=_controls(temperature)
    )
    transport_policy = policy_for(
        api_key_env=OPENROUTER_KEY_ENV,
        timeout_seconds=timeout_seconds,
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
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_attempts: int = 3,
) -> ProviderRoute:
    """An anthropic-messages plan-lane route (kimi/glm/minimax/stepfun).

    Base URL + key env come from ``window-starts.json`` (the :data:`PLAN_LANES`
    table). These are ALTERNATE routes selectable per run: debug stand-ins for
    the OpenRouter canonical routes, never burned as canonical.
    """
    if lane not in PLAN_LANES:
        raise ValueError(
            f"unknown plan lane {lane!r}; expected one of {LANE_NAMES}"
        )
    spec = PLAN_LANES[lane]
    call_config = anthropic_messages_config(
        model=spec.model, controls=_controls(temperature)
    )
    transport_policy = policy_for(
        api_key_env=spec.key_env,
        base_url=spec.base_url,
        timeout_seconds=timeout_seconds,
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


def route_for(
    lane: str,
    *,
    role: str = "task",
    temperature: float | None = 0.0,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_attempts: int = 3,
) -> ProviderRoute:
    """Select the route for a lane + role.

    ``lane="openrouter"`` returns the canonical task/proposer route by role;
    any plan-lane name returns that lane's anthropic-messages route.
    """
    if lane == "openrouter":
        if role == "proposer":
            return canonical_proposer_route(
                temperature=1.0 if temperature is None else temperature,
                timeout_seconds=timeout_seconds,
                max_attempts=max_attempts,
            )
        return canonical_task_route(
            temperature=temperature,
            timeout_seconds=timeout_seconds,
            max_attempts=max_attempts,
        )
    return lane_route(
        lane,
        role=role,
        temperature=temperature,
        timeout_seconds=timeout_seconds,
        max_attempts=max_attempts,
    )
