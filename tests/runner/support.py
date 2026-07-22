"""Shared fixtures for the validation-runner tests.

Everything drives the injected FAKE transports -- a scripted rollout transport
(reused from the env-adapter tests) and a scripted proposer transport -- so no
test makes a live paid LLM call. Two rollout scripts are provided: an
IMPROVEMENT script (a specific template renders a correct answer, others do
not) and a NO-IMPROVEMENT script (every candidate scores the same).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from dr_providers import (
    FailureClass,
    ProviderCallRequest,
    ProviderInvocationEvidence,
    ProviderTransportFailure,
    ProviderTransportPolicy,
    RawHttpRequest,
)

from tests.envs.support import (
    FakeTransport,
    ReplyFn,
    execution_policy,
    transport_policy,
)
from whetstone.envs.factory import EnvExperiment, build_env_experiment
from whetstone.envs.registry import env_spec
from whetstone.envs.rollout_definition import (
    ceiling_candidate,
    initial_candidate,
    render_prompt,
)
from whetstone.optimization.proposer import (
    ProposalDraft,
    ProposalRequest,
    ProposerConfig,
)
from whetstone.provider.policy import ProviderExecutionPolicy
from whetstone.runner.budget import CreditsSnapshot

TASK_MODEL = "openai/gpt-5-nano"
PROPOSER_MODEL = "openai/gpt-5.4-nano"
SPLIT = (2, 2, 2)

__all__ = [
    "PROPOSER_MODEL",
    "SPLIT",
    "TASK_MODEL",
    "FailingTransport",
    "FakeTransport",
    "ScriptedProposer",
    "correct_reply",
    "credits_fetcher",
    "improvement_reply",
    "no_improvement_reply",
    "proposer_config",
    "runner_execution_policy",
    "tiny_experiment",
]


@dataclass
class FailingTransport:
    """A transport that fails EVERY call with a fixed transport-failure code.

    Models the live round-1 blocker (100% of calls rejected pre-flight, e.g.
    ``missing_base_url``) so tests can assert the loud zero-success handling in
    the pilot + cell commands. Records every request it served. No network.
    """

    code: str = "missing_base_url"
    policy: ProviderTransportPolicy = field(default_factory=transport_policy)
    served: list[ProviderCallRequest] = field(default_factory=list)

    def __call__(
        self, request: ProviderCallRequest
    ) -> ProviderInvocationEvidence:
        self.served.append(request)
        raw_request = RawHttpRequest.build(
            url="https://example.test/v1/chat/completions",
            headers={"content-type": "json"},
            body={"model": "test-model"},
        )
        failure = ProviderTransportFailure(
            failure_class=FailureClass.PERMANENT,
            code=self.code,
            message=f"scripted transport failure: {self.code}",
            retryable=False,
        )
        return ProviderInvocationEvidence.build(
            request=request,
            policy=self.policy,
            raw_request=raw_request,
            outcome=failure,
        )


def _split_fits(env, n: int) -> bool:
    try:
        build_env_experiment(
            env.name, model=TASK_MODEL, pool_n_per_stratum=n,
            split_sizes=SPLIT,
        )
    except Exception:
        return False
    return True


def tiny_experiment(env_name: str) -> EnvExperiment:
    env = env_spec(env_name)
    n = 1
    while not _split_fits(env, n):
        n += 1
        if n > 40:  # pragma: no cover - safety valve
            raise RuntimeError(f"could not size a tiny pool for {env_name}")
    return build_env_experiment(
        env_name, model=TASK_MODEL, pool_n_per_stratum=n, split_sizes=SPLIT
    )


def runner_execution_policy() -> ProviderExecutionPolicy:
    return execution_policy(max_attempts=1)


def _all_instances(experiment: EnvExperiment):
    cfgs = experiment.eval_configs
    return (
        tuple(cfgs.internal.instances)
        + tuple(cfgs.official.instances)
    )


def correct_reply(experiment: EnvExperiment) -> ReplyFn:
    """A reply keyed on the naive/ceiling rendered prompt -> the gold answer.

    Both probes render distinct prompts for the same instance, so the map keys
    on the rendered prompt of each probe; every task's own gold is returned so
    the oracle scores a clean 1.0 for either probe.
    """
    env = env_spec(experiment.env_name)
    naive = initial_candidate(env)
    ceiling = ceiling_candidate(env)
    by_prompt: dict[str, str] = {}
    for inst in _all_instances(experiment):
        by_prompt[render_prompt(env, naive, inst)] = inst.gold
        by_prompt[render_prompt(env, ceiling, inst)] = inst.gold

    def reply(prompt: str) -> str:
        return by_prompt.get(prompt, "definitely-not-a-label")

    return reply


def improvement_reply(
    experiment: EnvExperiment, winning_template: str
) -> ReplyFn:
    """Only a proposed template's rendered prompt yields the gold answer.

    The naive/ceiling probes and every other proposal score 0; the candidate
    built from ``winning_template`` scores the gold answer -> the optimizer's
    best candidate beats the naive baseline on the official split.
    """
    env = env_spec(experiment.env_name)
    from whetstone.optimization.mutation import MUTATION_FIELD
    from whetstone.optimization.schema import Candidate

    winner = Candidate(
        candidate_id="winner",
        base_ref=initial_candidate(env).base_ref,
        payload={MUTATION_FIELD: winning_template},
    )
    winning_prompts: dict[str, str] = {}
    for inst in _all_instances(experiment):
        winning_prompts[render_prompt(env, winner, inst)] = inst.gold

    def reply(prompt: str) -> str:
        return winning_prompts.get(prompt, "definitely-not-a-label")

    return reply


def no_improvement_reply(experiment: EnvExperiment) -> ReplyFn:
    """Every candidate scores the same (a constant wrong answer)."""

    def reply(_prompt: str) -> str:
        return "definitely-not-a-label"

    return reply


class ScriptedProposer:
    """A proposer transport returning fixed templates (records its calls)."""

    def __init__(self, templates: tuple[str, ...]) -> None:
        self._templates = templates
        self.calls: list[tuple[str, ProposalRequest, int]] = []

    def draft(
        self,
        config: ProposerConfig,
        request: ProposalRequest,
        count: int,
    ) -> tuple[ProposalDraft, ...]:
        self.calls.append((config.identity_hash(), request, count))
        drafts: list[ProposalDraft] = []
        for index in range(count):
            template = (
                self._templates[index]
                if index < len(self._templates)
                else f"{request.base_template}::pad::{index}"
            )
            drafts.append(ProposalDraft(template=template))
        return tuple(drafts)


def proposer_config() -> ProposerConfig:
    return ProposerConfig(
        provider_call_config_ref="pcc://openai/gpt-5.4-nano",
        provider_call_config_hash="f" * 64,
        temperature=1.0,
    )


def credits_fetcher(
    values: list[tuple[float, float]],
) -> Callable[[], CreditsSnapshot]:
    """A scripted credits fetcher yielding (total_credits, total_usage) pairs.

    Each call pops the next pair, so a before/after pair models one cell's
    OpenRouter spend without any network.
    """
    queue = list(values)

    def fetch() -> CreditsSnapshot:
        total_credits, total_usage = queue.pop(0)
        return CreditsSnapshot(
            total_credits=total_credits, total_usage=total_usage
        )

    return fetch
