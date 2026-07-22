"""The checklist-B pilot: token sanity, agreement, direction, extraction.

Per ``reports/validation-plan.md`` "Spend order" step 1, the pilot runs ~10
instances x both probes (naive + ceiling) x repeats ``{0, 1, 2}`` and records:

* **token counts vs spec estimate** -- observed prompt/completion tokens per
  call against a caller-supplied spec estimate;
* **temp-0 agreement rate** -- how often the three temp-0 repeats of one
  (instance, probe) agree on the 0/1 oracle score (a temp-0 sanity check);
* **naive-vs-ceiling direction** -- whether the ceiling probe's mean score is
  ``>=`` the naive probe's (the headroom line points the right way);
* **per-call extraction spot-record** -- for each call the raw response text,
  the extracted answer the oracle saw, and the resulting 0/1 score;
* **spend** -- the OpenRouter credits before/after (when lane=openrouter).

The transport is injected (a scripted fake in tests, dr-providers' real client
in a live run), so importing/running the pilot logic makes no live paid call by
itself. Output is written to ``validation/pilots/<env>.json``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from dr_providers import (
    MessageRole,
    PromptMessage,
    ProviderCallConfig,
    ProviderCallRequest,
    Transcript,
)
from whetstone_envs.core import Instance

from whetstone.envs.factory import EnvExperiment, build_env_experiment
from whetstone.envs.oracle_operator import env_exact_match_score
from whetstone.envs.registry import env_spec
from whetstone.envs.rollout_definition import render_prompt
from whetstone.envs.task import EnvTask
from whetstone.optimization.schema import Candidate
from whetstone.provider.driver import TransportCall, run_provider_call
from whetstone.provider.policy import ProviderExecutionPolicy
from whetstone.runner.budget import CreditsSnapshot

__all__ = [
    "PILOT_REPEATS",
    "PilotCallRecord",
    "PilotProbeSummary",
    "PilotReport",
    "run_pilot",
]

#: The temp-0 repeat ids the pilot runs per (instance, probe).
PILOT_REPEATS: tuple[int, ...] = (0, 1, 2)

#: The default number of instances the pilot draws from the internal split.
DEFAULT_PILOT_INSTANCES = 10


@dataclass(frozen=True, slots=True)
class PilotCallRecord:
    """One per-call extraction spot-record (raw + extracted + score)."""

    instance_id: str
    probe: str
    repeat_id: int
    raw_response: str
    extracted: str
    score: float | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    failed: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "instance_id": self.instance_id,
            "probe": self.probe,
            "repeat_id": self.repeat_id,
            "raw_response": self.raw_response,
            "extracted": self.extracted,
            "score": self.score,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "failed": self.failed,
        }


@dataclass(frozen=True, slots=True)
class PilotProbeSummary:
    """The aggregate for one probe (naive or ceiling)."""

    probe: str
    mean_score: float | None
    agreement_rate: float
    call_count: int
    failed_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "probe": self.probe,
            "mean_score": self.mean_score,
            "agreement_rate": self.agreement_rate,
            "call_count": self.call_count,
            "failed_count": self.failed_count,
        }


@dataclass(slots=True)
class PilotReport:
    """The full pilot report written to ``validation/pilots/<env>.json``."""

    env: str
    lane: str
    instances: int
    spec_estimate_tokens: int | None
    naive: PilotProbeSummary
    ceiling: PilotProbeSummary
    direction_ok: bool
    token_mean_total: float | None
    token_vs_spec: float | None
    calls: list[PilotCallRecord] = field(default_factory=list)
    spend_before: CreditsSnapshot | None = None
    spend_after: CreditsSnapshot | None = None

    @property
    def spend_usd(self) -> float | None:
        if self.spend_before is None or self.spend_after is None:
            return None
        before = self.spend_before.remaining_usd
        after = self.spend_after.remaining_usd
        if before is None or after is None:
            return None
        return before - after

    def as_dict(self) -> dict[str, object]:
        return {
            "env": self.env,
            "lane": self.lane,
            "instances": self.instances,
            "spec_estimate_tokens": self.spec_estimate_tokens,
            "naive": self.naive.as_dict(),
            "ceiling": self.ceiling.as_dict(),
            "direction_ok": self.direction_ok,
            "token_mean_total": self.token_mean_total,
            "token_vs_spec": self.token_vs_spec,
            "spend_usd": self.spend_usd,
            "spend_before": (
                None if self.spend_before is None
                else {
                    "total_credits": self.spend_before.total_credits,
                    "total_usage": self.spend_before.total_usage,
                    "remaining_usd": self.spend_before.remaining_usd,
                }
            ),
            "spend_after": (
                None if self.spend_after is None
                else {
                    "total_credits": self.spend_after.total_credits,
                    "total_usage": self.spend_after.total_usage,
                    "remaining_usd": self.spend_after.remaining_usd,
                }
            ),
            "calls": [c.as_dict() for c in self.calls],
        }

    def write(self, root: Path) -> Path:
        """Write ``<root>/validation/pilots/<env>.json``; return the path."""
        out_dir = root / "validation" / "pilots"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{self.env}.json"
        path.write_text(json.dumps(self.as_dict(), indent=2, sort_keys=True))
        return path


def _request(config: ProviderCallConfig, prompt: str) -> ProviderCallRequest:
    return ProviderCallRequest(
        config=config,
        transcript=Transcript(
            messages=(PromptMessage(role=MessageRole.USER, content=prompt),)
        ),
    )


def _agreement_rate(
    scores_by_instance: dict[str, list[float | None]],
) -> float:
    """Fraction of instances whose temp-0 repeats all agree on the score."""
    if not scores_by_instance:
        return 0.0
    agreeing = 0
    for scores in scores_by_instance.values():
        present = [s for s in scores if s is not None]
        if present and len(set(present)) == 1:
            agreeing += 1
    return agreeing / len(scores_by_instance)


def _run_probe(
    experiment: EnvExperiment,
    *,
    probe: str,
    candidate: Candidate,
    instances: tuple[Instance, ...],
    transport: TransportCall,
    execution_policy: ProviderExecutionPolicy,
    procedure_hash: str,
    calls: list[PilotCallRecord],
) -> PilotProbeSummary:
    env = env_spec(experiment.env_name)
    config = experiment.rollout_definition.provider_call_config
    scores_by_instance: dict[str, list[float | None]] = {}
    all_scores: list[float] = []
    failed = 0
    for instance in instances:
        task = EnvTask.from_instance(env.name, instance)
        prompt = render_prompt(env, candidate, instance)
        per_instance: list[float | None] = []
        for repeat_id in PILOT_REPEATS:
            result = run_provider_call(
                request=_request(config, prompt),
                policy=execution_policy,
                transport=transport,
                logical_call_id=f"pilot::{task.task_identity()}::"
                f"{probe}::{repeat_id}",
            )
            if not result.succeeded or result.generation is None:
                calls.append(
                    PilotCallRecord(
                        instance_id=str(instance.id),
                        probe=probe,
                        repeat_id=repeat_id,
                        raw_response="",
                        extracted="",
                        score=None,
                        prompt_tokens=None,
                        completion_tokens=None,
                        total_tokens=None,
                        failed=True,
                    )
                )
                per_instance.append(None)
                failed += 1
                continue
            text = result.generation.text
            score = env_exact_match_score(
                env=env,
                generation=text,
                gold=instance.gold,
                evaluation_procedure_config_hash=procedure_hash,
            )
            usage = result.generation.response.usage
            calls.append(
                PilotCallRecord(
                    instance_id=str(instance.id),
                    probe=probe,
                    repeat_id=repeat_id,
                    raw_response=text,
                    extracted=text,
                    score=float(score.value),
                    prompt_tokens=(
                        usage.prompt_tokens if usage is not None else None
                    ),
                    completion_tokens=(
                        usage.completion_tokens
                        if usage is not None
                        else None
                    ),
                    total_tokens=(
                        usage.total_tokens if usage is not None else None
                    ),
                    failed=False,
                )
            )
            per_instance.append(float(score.value))
            all_scores.append(float(score.value))
        scores_by_instance[str(instance.id)] = per_instance
    mean_score = (
        sum(all_scores) / len(all_scores) if all_scores else None
    )
    return PilotProbeSummary(
        probe=probe,
        mean_score=mean_score,
        agreement_rate=_agreement_rate(scores_by_instance),
        call_count=len(instances) * len(PILOT_REPEATS),
        failed_count=failed,
    )


def run_pilot(
    *,
    env: str,
    lane: str,
    model: str,
    transport: TransportCall,
    execution_policy: ProviderExecutionPolicy,
    instance_count: int = DEFAULT_PILOT_INSTANCES,
    pool_n_per_stratum: int | None = None,
    split_sizes: tuple[int, int, int] | None = None,
    spec_estimate_tokens: int | None = None,
    spend_before: CreditsSnapshot | None = None,
    spend_after: CreditsSnapshot | None = None,
) -> PilotReport:
    """Run the checklist-B pilot for one env and return its report.

    Builds the env experiment, draws ``instance_count`` instances from the
    internal split, and runs both probes x 3 temp-0 repeats through the
    injected transport, collecting token counts, agreement, direction, and
    per-call extraction spot-records.
    """
    experiment = build_env_experiment(
        env,
        model=model,
        pool_n_per_stratum=pool_n_per_stratum,
        split_sizes=split_sizes,
    )
    procedure_hash = experiment.eval_configs.procedure_config_hash
    internal = experiment.eval_configs.internal.instances
    instances = internal[:instance_count]

    calls: list[PilotCallRecord] = []
    naive = _run_probe(
        experiment,
        probe="naive",
        candidate=experiment.initial_candidate,
        instances=instances,
        transport=transport,
        execution_policy=execution_policy,
        procedure_hash=procedure_hash,
        calls=calls,
    )
    ceiling = _run_probe(
        experiment,
        probe="ceiling",
        candidate=experiment.ceiling_candidate,
        instances=instances,
        transport=transport,
        execution_policy=execution_policy,
        procedure_hash=procedure_hash,
        calls=calls,
    )
    direction_ok = (
        naive.mean_score is not None
        and ceiling.mean_score is not None
        and ceiling.mean_score >= naive.mean_score
    )
    totals = [c.total_tokens for c in calls if c.total_tokens is not None]
    token_mean_total = sum(totals) / len(totals) if totals else None
    token_vs_spec = (
        token_mean_total / spec_estimate_tokens
        if token_mean_total is not None
        and spec_estimate_tokens
        else None
    )
    return PilotReport(
        env=env,
        lane=lane,
        instances=len(instances),
        spec_estimate_tokens=spec_estimate_tokens,
        naive=naive,
        ceiling=ceiling,
        direction_ok=direction_ok,
        token_mean_total=token_mean_total,
        token_vs_spec=token_vs_spec,
        calls=calls,
        spend_before=spend_before,
        spend_after=spend_after,
    )
