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

Transport policy is progress-aware by default: an absolute wall-clock CAP
(``timeout_seconds``) plus a PROGRESS/IDLE timeout (``idle_timeout_seconds``,
default ~90s) so a legitimate long streaming response from a reasoning model
(steady tokens over many minutes) is bounded by inactivity, not by total
wall-clock. Native retries ``0`` (Whetstone owns all semantic retry).
Everything here is config-identity only -- no live call is made by
constructing a route.
"""

from __future__ import annotations

from dataclasses import dataclass

from dr_providers import (
    DEFAULT_IDLE_TIMEOUT_SECONDS,
    GenerationControls,
    ProviderCallConfig,
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
    "DEEPSEEK_TASK_MODEL",
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
    "lane_route",
    "openai_direct_route",
    "reasoning_effort_for",
    "route_for",
    "task_model_for_env",
]

#: The absolute wall-clock CAP (seconds) for a single wire call. Raised from
#: the old flat 120s to 600s so a LEGITIMATE long reasoning-model generation
#: (c23 ran 18k reasoning tokens, >360s) is not capped mid-stream: the flat
#: 120s deadline killed such streams as false stalls. The CAP is now only the
#: dribble backstop; the IDLE timeout (below) is the real stall detector.
DEFAULT_TIMEOUT_SECONDS = 600.0

#: The PROGRESS/IDLE timeout (seconds): a call fails ``stalled_response`` only
#: when NO bytes arrive for this long. A stream making steady progress (even
#: for many minutes) never trips it; a genuinely wedged edge fails in ~90s.
DEFAULT_IDLE_SECONDS = DEFAULT_IDLE_TIMEOUT_SECONDS

#: OpenRouter canonical model slugs (window-starts.json).
CANONICAL_TASK_MODEL = "openai/gpt-5-nano"
CANONICAL_PROPOSER_MODEL = "openai/gpt-5.4-nano"

#: An alternate task model for the constraint-heavy envs (per user directive).
DEEPSEEK_TASK_MODEL = "deepseek/deepseek-v4-flash"

#: Default task model for the enc-dec family (user directive 2026-07-23):
#: deepseek is by far the slowest model in the funnel latency preview
#: (~6.4s vs ~1.1s median/call), so it must be an EXPLICIT choice (the
#: contamination axis via --task-model), never a default. gemini is the
#: standing evidence favorite; the funnel phase-3 pick may revise this.
ENCDEC_DEFAULT_TASK_MODEL = "google/gemini-3.1-flash-lite"

#: Per-env DEFAULT task model (the matrix config). c18 + c22 default to the
#: deepseek model per user directive; every other env keeps the canonical nano
#: task model. The chosen model folds into the Provider Call Config (hence the
#: graph_hash) and is recorded in ``cells.jsonl`` ``models.task``. The
#: ``--task-model`` CLI flag overrides this default for a given cell.
TASK_MODEL_BY_ENV: dict[str, str] = {
    "c18": DEEPSEEK_TASK_MODEL,
    "c22": DEEPSEEK_TASK_MODEL,
    # c22h inherits c22's constraint-heavy deepseek default (the c22-column
    # convention; overridable via --task-model for the pilot / anchor).
    "c22h": DEEPSEEK_TASK_MODEL,
    # c18h inherits base c18's deepseek default (same entailment task family);
    # overridable via --task-model (the c18h headroom pilot/anchor runs nano).
    "c18h": DEEPSEEK_TASK_MODEL,
    # ed1 (enc-dec HumanEval compression): default enc/dec model, overridable
    # via --task-model. The same route plays both encoder and decoder.
    # Deepseek (the contamination axis) is explicit-only, never the default.
    "ed1": ENCDEC_DEFAULT_TASK_MODEL,
    # ed1m (behavioral-mutant enc-dec): same enc/dec model family as ed1.
    "ed1m": ENCDEC_DEFAULT_TASK_MODEL,
    # d1 (direct-generation precursor, task 23): the matrix default mirrors
    # ed1's so a d1 anchor pairs with the corresponding ed1 anchor on the
    # same model family; --task-model selects the clean-vs-deepseek axis.
    "d1": ENCDEC_DEFAULT_TASK_MODEL,
}


#: Per-env DEFAULT completeness tolerance (the matrix config). A missing entry
#: is the strict, untolerant default: PROPAGATE (``max_skip_fraction`` 0.0),
#: any missing/failed row makes the official arm incomplete. c18 declares a
#: SKIP-with-visible-counts policy tolerating up to 2% skipped rows -- deepseek
#: is ~1% non-retryably flaky under concurrency, so a strict PROPAGATE anchor
#: never certifies; the bounded tolerance certifies while the skipped rows stay
#: explicit counts on the aggregate + cell line (never silently dropped). The
#: tolerance is identity-bearing: a c18 tolerant cell has a DISTINCT
#: ``eval_config_hash`` from a strict one. Value: ``(missing_data, fraction)``.
COMPLETENESS_BY_ENV: dict[str, tuple[str, float]] = {
    "c18": ("skip", 0.02),
    # c18h shares c18's deepseek matrix default, so it inherits the same
    # bounded skip tolerance for the flaky-under-concurrency deepseek anchor.
    # (A nano cell is not flaky and simply never exercises the tolerance.)
    "c18h": ("skip", 0.02),
    # ed1 (enc-dec) declares a HIGHER SKIP tolerance than c18. Its per-row
    # failures are GENUINE stochastic model behavior at tight budgets, NOT
    # brittle plumbing (the dr-code extractor correctly strips fenced/prose
    # decoder output -- verified): at r=0.10 the model sometimes emits an EMPTY
    # completion (encoder or decoder) -> a PERMANENT response_parse_error, and
    # the tight budget can drop the entry-point NAME so the decoder writes ok
    # code under a wrong function name -> a harness timeout (infra-unknown).
    # eval:ed1:a2 measured 25/240 = 10.4% such rows. A 15% tolerance
    # covers the observed rate with ~1.4x margin while keeping the skipped rows
    # explicit counts on the aggregate + cell line (never silently dropped);
    # override per-cell with --missing-data / --max-skip-fraction.
    "ed1": ("skip", 0.15),
    # ed1m (behavioral-mutant enc-dec) inherits ed1's SKIP tolerance: the same
    # deepseek enc/dec produces the same stochastic empty-completion tail, and
    # the mutant oracle adds a small subprocess tail. 15% covers both;
    # tune per-cell once the first ed1m anchor measures the actual rate.
    "ed1m": ("skip", 0.15),
    # d1 (direct-generation) inherits the ed1 SKIP tolerance: the same
    # deepseek task model produces the same stochastic empty-completion tail on
    # a direct generation (a single call, so a somewhat smaller tail than the
    # enc-dec 2-call rows). 15% covers it with margin; tune per-cell with
    # --missing-data / --max-skip-fraction once a d1 anchor measures the rate.
    "d1": ("skip", 0.15),
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
    per-env matrix default (``TASK_MODEL_BY_ENV``) applies, falling back to the
    canonical nano task model for any env not listed. The returned slug folds
    into the task route's Provider Call Config identity (graph_hash), so a
    deepseek cell's route identity differs from a nano cell's.
    """
    if override:
        return override
    return TASK_MODEL_BY_ENV.get(env, CANONICAL_TASK_MODEL)

#: The env var carrying the OpenRouter credential.
OPENROUTER_KEY_ENV = "OPENROUTER_API_KEY"

#: The OpenRouter API base URL (chat-completions). Every canonical route pins
#: this so the transport policy has a non-None base_url and pre-flight passes.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

#: The OpenAI DIRECT lane (``lane="openai"``): the OpenAI API keyed off
#: ``OPENAI_API_KEY`` over the SAME chat-completions protocol as OpenRouter,
#: through OpenAI's own provider. RATIONALE (run lesson): the OpenAI provider
#: RESPECTS ``temperature`` for gpt-5.4-nano, whereas OpenRouter IGNORES it for
#: that model -- so a temperature-sensitive study (or the screen at a chosen
#: temp) must go direct. The lane folds into route/config identity (its
#: ProviderKind is OPENAI, a DISTINCT identity_hash from the openrouter config
#: for the same model), so ``cells.jsonl`` / the screen artifact record the
#: provider distinctly (``lane="openai"``).
OPENAI_KEY_ENV = "OPENAI_API_KEY"
OPENAI_BASE_URL = "https://api.openai.com/v1"


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


def _controls(
    temperature: float | None,
    reasoning: ReasoningEffort | None = None,
) -> GenerationControls | None:
    """The GenerationControls for a route, or ``None`` when nothing is set.

    ``reasoning`` (the ``--reasoning-effort`` dial) is OUTPUT-AFFECTING: it
    serializes on the wire per the config's reasoning shape (openrouter ->
    ``reasoning`` object; openai -> ``reasoning_effort`` field) AND folds into
    the Provider Call Config identity_hash (the c23-era rule), so a distinct
    effort is a distinct route/graph variant. ``None`` leaves the control
    UNSET -> the provider default -> byte-identical to a run without the flag.
    """
    if temperature is None and reasoning is None:
        return None
    return GenerationControls(temperature=temperature, reasoning=reasoning)


def canonical_task_route(
    *,
    model: str = CANONICAL_TASK_MODEL,
    temperature: float | None = 0.0,
    reasoning: ReasoningEffort | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    idle_timeout_seconds: float = DEFAULT_IDLE_SECONDS,
    max_attempts: int = 3,
) -> ProviderRoute:
    """The canonical OpenRouter task (encoder/decoder) route.

    ``openai/gpt-5-nano`` over chat-completions, keyed off
    ``OPENROUTER_API_KEY``. Temperature defaults to 0 (the pilots use temp-0
    for agreement checks); pass ``None`` to leave the control unset. The
    absolute cap is 600s to accommodate reasoning-model generations; the
    ~90s idle timeout is the real stall detector. ``reasoning`` sets the
    OUTPUT-AFFECTING reasoning effort (folds into the Config identity).
    """
    call_config = openrouter_chat_config(
        model=model, controls=_controls(temperature, reasoning)
    )
    transport_policy = policy_for(
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
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    idle_timeout_seconds: float = DEFAULT_IDLE_SECONDS,
    max_attempts: int = 3,
) -> ProviderRoute:
    """The OpenAI DIRECT route (``lane="openai"``): OpenAI's own API.

    Same chat-completions protocol/shape as the OpenRouter transport, but keyed
    off ``OPENAI_API_KEY`` at ``OPENAI_BASE_URL`` via ``ProviderKind.OPENAI``
    -- so the config identity (hence graph route identity) is DISTINCT from the
    openrouter route for the same model. Chosen when temperature must hold
    (OpenAI respects it for gpt-5.4-nano; OpenRouter ignores it). ``reasoning``
    serializes as ``reasoning_effort`` and folds into the Config identity.
    """
    call_config = openai_chat_config(
        model=model, controls=_controls(temperature, reasoning)
    )
    transport_policy = policy_for(
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
    ``OPENROUTER_API_KEY``. Its Config identity is distinct from the task
    route's (different model, different temperature), so it never collides with
    an encoder/decoder route hash.
    """
    call_config = openrouter_chat_config(
        model=model, controls=_controls(temperature, reasoning)
    )
    transport_policy = policy_for(
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
        model=spec.model, controls=_controls(temperature, reasoning)
    )
    transport_policy = policy_for(
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


#: The CLI ``--reasoning-effort`` choices -> the typed cross-provider effort.
#: ``none`` maps to ``ReasoningEffort.NONE`` (openrouter serializes it as
#: ``{reasoning: {enabled: false}}``; openai as the minimal/none effort the API
#: allows). Absent flag -> ``None`` -> provider default (byte-identical).
REASONING_EFFORT_CHOICES: tuple[str, ...] = ("none", "low", "medium", "high")
_REASONING_BY_NAME: dict[str, ReasoningEffort] = {
    "none": ReasoningEffort.NONE,
    "low": ReasoningEffort.LOW,
    "medium": ReasoningEffort.MEDIUM,
    "high": ReasoningEffort.HIGH,
}


def reasoning_effort_for(name: str | None) -> ReasoningEffort | None:
    """Map a ``--reasoning-effort`` choice to the typed effort, or ``None``.

    ``None`` / absent leaves the control UNSET (the provider default), so a run
    without the flag is byte-identical to the historical behavior.
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
    """Select the route for a lane + role.

    ``lane="openrouter"`` returns the canonical task/proposer route by role;
    any plan-lane name returns that lane's anthropic-messages route.
    ``task_model`` (openrouter task role only) selects a per-env task model
    (e.g. the deepseek model for c18/c22), folding into the route's Config
    identity (graph_hash) so a deepseek route differs from a nano route.
    ``proposer_model`` (openrouter proposer role only) overrides the canonical
    ``gpt-5.4-nano`` proposer model, folding into the proposer route's Config
    identity so a non-default proposer route differs from the canonical one.
    ``reasoning`` (the ``--reasoning-effort`` dial) is OUTPUT-AFFECTING and
    folds into the Config identity; ``None`` leaves it at the provider default
    (byte-identical to a run without the flag).
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
        # OpenAI DIRECT: same chat-completions shape, OpenAI's own provider.
        # ``proposer_model``/``task_model`` select the model by role; the
        # proposer default keeps the canonical proposer model + temp 1.0.
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
