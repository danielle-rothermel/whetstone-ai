"""The ed1 (enc-dec) pilot: both encoder probes on a small HumanEval slice.

The QA pilot drives a single LLM call per probe against a string oracle; the
ed1
pilot instead drives the 3-node encoder->decoder->code-eval rollout for BOTH
encoder templates (naive A "concise" and ceiling-ish B "compress for
reconstruction") on a small task slice via
:func:`whetstone.envs.ed1_eval.run_ed1_eval`,
and reports each probe's Average Binary Test Pass Rate AND Mean Compression
Ratio, plus the naive->ceiling direction (does the informative template help /
compress better). It writes ``<root>/pilots/ed1.json`` so it lands alongside
the
QA pilot reports. Reuses the ed1 eval drive + code scorer + zstd scoring; the
transport is injected (a scripted fake in tests, the live route in a real run).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from whetstone.envs.ed1 import (
    ED1_CANONICAL_MODEL,
    ED1_DATASET_ID,
    ED1_DATASET_REVISION,
    ED1_DEFAULT_BUDGET_RATIO,
    ED1_ENV_NAME,
    build_ed1_experiment,
    ed1_ceiling_candidate,
    ed1_initial_candidate,
)
from whetstone.envs.ed1_scoring import CodeScore
from whetstone.execution.fanout import FanoutConfig
from whetstone.optimization.mutation import MUTATION_FIELD
from whetstone.provider.driver import TransportCall
from whetstone.provider.policy import ProviderExecutionPolicy


@dataclass(frozen=True, slots=True)
class Ed1ProbeSummary:
    """One encoder probe's dual measurement over the pilot slice."""

    probe: str
    pass_rate: float | None
    mean_compression: float | None
    task_count: int
    repeat_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "probe": self.probe,
            "pass_rate": self.pass_rate,
            "mean_compression": self.mean_compression,
            "task_count": self.task_count,
            "repeat_count": self.repeat_count,
        }


@dataclass(frozen=True, slots=True)
class Ed1PilotReport:
    """The ed1 pilot report (both probes, dual scores)."""

    env: str
    model: str
    budget_ratio: float
    dataset_id: str
    dataset_revision: str
    tasks: int
    repeats: int
    naive: Ed1ProbeSummary
    ceiling: Ed1ProbeSummary

    @property
    def pass_direction_ok(self) -> bool:
        """Whether the ceiling probe's pass rate is >= the naive's."""
        if self.naive.pass_rate is None or self.ceiling.pass_rate is None:
            return False
        return self.ceiling.pass_rate >= self.naive.pass_rate

    @property
    def compression_direction_ok(self) -> bool:
        """Whether the ceiling probe compresses at least as well (ratio <=)."""
        n = self.naive.mean_compression
        c = self.ceiling.mean_compression
        if n is None or c is None:
            return False
        return c <= n

    def as_dict(self) -> dict[str, object]:
        return {
            "env": self.env,
            "model": self.model,
            "budget_ratio": self.budget_ratio,
            "dataset_id": self.dataset_id,
            "dataset_revision": self.dataset_revision,
            "tasks": self.tasks,
            "repeats": self.repeats,
            "naive": self.naive.as_dict(),
            "ceiling": self.ceiling.as_dict(),
            "pass_direction_ok": self.pass_direction_ok,
            "compression_direction_ok": self.compression_direction_ok,
        }

    def write(self, root: Path) -> Path:
        out_dir = root / "pilots"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{self.env}.json"
        path.write_text(json.dumps(self.as_dict(), indent=2, sort_keys=True))
        return path


def run_ed1_pilot(
    *,
    transport: TransportCall,
    execution_policy: ProviderExecutionPolicy,
    model: str = ED1_CANONICAL_MODEL,
    budget_ratio: float = ED1_DEFAULT_BUDGET_RATIO,
    tasks: int = 8,
    repeats: int = 1,
    prefer_snapshot: bool = True,
    concurrency: int = 5,
    scorer: Callable[..., CodeScore] | None = None,
) -> Ed1PilotReport:
    """Run the ed1 pilot: both encoder probes on a small task slice.

    Drives the enc-dec rollout for the naive (A) + ceiling (B) encoder
    templates
    over the first-``tasks`` HumanEval+ tasks, reporting each probe's pass rate
    +
    Mean Compression Ratio. The transport + code scorer are injected (a fake in
    tests, the live route in a real run). No live call is made by this function
    itself.
    """
    from whetstone.envs.ed1_eval import run_ed1_eval

    experiment = build_ed1_experiment(
        model=model,
        budget_ratio=budget_ratio,
        scorer=scorer,
        prefer_snapshot=prefer_snapshot,
        limit=tasks,
        internal_n=tasks,
        official_n=tasks,
    )
    instances = experiment.eval_configs.internal.instances
    fanout = FanoutConfig(concurrency=concurrency)

    def _probe(candidate) -> Ed1ProbeSummary:
        ed = run_ed1_eval(
            experiment,
            candidate_template=str(candidate.payload[MUTATION_FIELD]),
            candidate_id=candidate.candidate_id,
            instances=instances,
            execution_policy=execution_policy,
            transport=transport,
            scorer=scorer,
            repeats=repeats,
            fanout=fanout,
            apply_reward=False,
        )
        return Ed1ProbeSummary(
            probe=candidate.candidate_id,
            pass_rate=ed.pass_aggregate.aggregation_output.value,
            mean_compression=ed.compression_aggregate.aggregation_output.value,
            task_count=len(instances),
            repeat_count=repeats,
        )

    naive = _probe(ed1_initial_candidate())
    ceiling = _probe(ed1_ceiling_candidate())
    return Ed1PilotReport(
        env=ED1_ENV_NAME,
        model=model,
        budget_ratio=budget_ratio,
        dataset_id=ED1_DATASET_ID,
        dataset_revision=ED1_DATASET_REVISION,
        tasks=len(instances),
        repeats=repeats,
        naive=naive,
        ceiling=ceiling,
    )


__all__ = [
    "Ed1PilotReport",
    "Ed1ProbeSummary",
    "run_ed1_pilot",
]
