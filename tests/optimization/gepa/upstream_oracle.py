from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

# Run in a fresh interpreter so PYTHONHASHSEED is fixed before GEPA imports.
from gepa import optimize
from gepa.core.adapter import EvaluationBatch, ProposalFn


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _level(value: str) -> int:
    return int(value[1:])


@dataclass(frozen=True)
class _Datum:
    id: str
    group: str


class _QuietLogger:
    def log(self, message: str) -> None:
        del message


class _Trace:
    def __init__(self) -> None:
        self.timeline: list[dict[str, Any]] = []

    def append(self, kind: str, **payload: Any) -> None:
        self.timeline.append({"kind": kind, **payload})


class _Callback:
    def __init__(self, trace: _Trace) -> None:
        self._trace = trace

    def on_iteration_start(self, event: dict[str, Any]) -> None:
        self._trace.append("iteration_start", iteration=event["iteration"])

    def on_candidate_accepted(self, event: dict[str, Any]) -> None:
        self._trace.append(
            "candidate_accepted",
            iteration=event["iteration"],
            candidate_idx=event["new_candidate_idx"],
            parents=list(event["parent_ids"]),
        )

    def on_merge_attempted(self, event: dict[str, Any]) -> None:
        self._trace.append(
            "merge_attempted",
            iteration=event["iteration"],
            parents=list(event["parent_ids"]),
            candidate=list(event["merged_candidate"].items()),
        )

    def on_merge_accepted(self, event: dict[str, Any]) -> None:
        self._trace.append(
            "merge_accepted",
            iteration=event["iteration"],
            candidate_idx=event["new_candidate_idx"],
            parents=list(event["parent_ids"]),
        )


class _Adapter:
    propose_new_texts: ProposalFn | None

    def __init__(
        self,
        trace: _Trace,
        *,
        effect_log: Path | None,
        crash_after: int | None,
    ) -> None:
        self._trace = trace
        self._effect_log = effect_log
        self._crash_after = crash_after
        self._records: list[dict[str, Any]] = (
            json.loads(effect_log.read_text())
            if effect_log is not None and effect_log.exists()
            else []
        )
        self.effect_identities: list[str] = []
        self.effect_kinds: list[str] = []
        self.propose_new_texts = self.propose

    def _effect(
        self,
        kind: str,
        semantic: dict[str, Any],
        execute: Any,
    ) -> Any:
        ordinal = len(self.effect_identities)
        request = json.loads(
            _canonical(
                {
                    "run": "gepa-replay-spike-v1",
                    "source": "gepa==0.1.1",
                    "kind": kind,
                    "ordinal": ordinal,
                    "semantic": semantic,
                }
            )
        )
        identity = _digest(request)
        self.effect_identities.append(identity)
        self.effect_kinds.append(kind)
        self._trace.append(kind, ordinal=ordinal, semantic=semantic)
        if ordinal < len(self._records):
            prior = self._records[ordinal]
            if prior["request"] != request or prior["identity"] != identity:
                raise RuntimeError(
                    f"semantic effect conflict at replay ordinal {ordinal}"
                )
            return prior["result"]
        if ordinal != len(self._records):
            raise RuntimeError("persisted effect log has an ordinal gap")
        result = execute()
        self._records.append(
            {
                "identity": identity,
                "request": request,
                "result": result,
            }
        )
        if self._effect_log is not None:
            temporary = self._effect_log.with_suffix(".tmp")
            temporary.write_text(_canonical(self._records))
            temporary.replace(self._effect_log)
        if self._crash_after == ordinal:
            os._exit(86)
        return result

    def evaluate(
        self,
        batch: list[_Datum],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[dict[str, Any], dict[str, Any]]:
        def execute() -> dict[str, Any]:
            alpha_level = _level(candidate["alpha"])
            beta_level = _level(candidate["beta"])
            scores: list[float] = []
            outputs: list[dict[str, Any]] = []
            trajectories: list[dict[str, Any]] = []
            for row in batch:
                score = 0.5
                score += (0.2 if row.group == "A" else -0.2) * alpha_level
                score += (0.2 if row.group == "B" else -0.2) * beta_level
                scores.append(score)
                output = {
                    "id": row.id,
                    "candidate": list(candidate.items()),
                    "score": score,
                }
                outputs.append(output)
                trajectories.append(
                    {
                        "Inputs": {"id": row.id, "group": row.group},
                        "Generated Outputs": output,
                        "Feedback": f"score={score}",
                    }
                )
            return {
                "outputs": outputs,
                "scores": scores,
                "trajectories": (trajectories if capture_traces else None),
            }

        result = self._effect(
            "evaluate",
            {
                "candidate": list(candidate.items()),
                "data_ids": [row.id for row in batch],
                "capture_traces": capture_traces,
            },
            execute,
        )
        return EvaluationBatch(
            outputs=result["outputs"],
            scores=result["scores"],
            trajectories=result["trajectories"],
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[dict[str, Any], dict[str, Any]],
        components_to_update: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        del candidate
        assert eval_batch.trajectories is not None
        return {
            component: list(eval_batch.trajectories)
            for component in components_to_update
        }

    def propose(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[
            str,
            Sequence[Mapping[str, Any]],
        ],
        components_to_update: list[str],
    ) -> dict[str, str]:
        concrete_dataset = {
            name: [dict(example) for example in examples]
            for name, examples in reflective_dataset.items()
        }

        def execute() -> dict[str, str]:
            proposed: dict[str, str] = {}
            for component in components_to_update:
                examples = concrete_dataset[component]
                alpha_count = sum(
                    example["Inputs"]["group"] == "A" for example in examples
                )
                beta_count = len(examples) - alpha_count
                favored_count = (
                    alpha_count if component == "alpha" else beta_count
                )
                direction = (
                    1 if favored_count >= len(examples) - favored_count else -1
                )
                proposed[component] = candidate[component][0] + str(
                    _level(candidate[component]) + direction
                )
            return proposed

        return self._effect(
            "propose",
            {
                "candidate": list(candidate.items()),
                "reflective_dataset": concrete_dataset,
                "components_to_update": components_to_update,
            },
            execute,
        )


def _result_payload(result: Any) -> dict[str, Any]:
    return {
        # Preserve candidate insertion order.
        "candidates": [
            list(candidate.items()) for candidate in result.candidates
        ],
        "parents": result.parents,
        "val_aggregate_scores": result.val_aggregate_scores,
        "val_subscores": result.val_subscores,
        "frontier": {
            key: sorted(value)
            for key, value in (result.per_val_instance_best_candidates.items())
        },
        "discovery_eval_counts": result.discovery_eval_counts,
        "total_metric_calls": result.total_metric_calls,
        "num_full_val_evals": result.num_full_val_evals,
        "best_idx": result.best_idx,
    }


def run_oracle(
    *,
    component_selector: str,
    use_merge: bool,
    effect_log: Path | None = None,
    crash_after: int | None = None,
) -> dict[str, Any]:
    trace = _Trace()
    adapter = _Adapter(
        trace,
        effect_log=effect_log,
        crash_after=crash_after,
    )
    train = [
        _Datum("train-A-0", "A"),
        _Datum("train-B-0", "B"),
        _Datum("train-A-1", "A"),
        _Datum("train-B-1", "B"),
    ]
    val = [
        _Datum("val-A-0", "A"),
        _Datum("val-B-0", "B"),
        _Datum("val-A-1", "A"),
        _Datum("val-B-1", "B"),
        _Datum("val-A-2", "A"),
        _Datum("val-B-2", "B"),
    ]
    result = optimize(
        seed_candidate={"alpha": "A0", "beta": "B0"},
        trainset=train,
        valset=val,
        adapter=adapter,
        reflection_lm=None,
        candidate_selection_strategy="pareto",
        module_selector=component_selector,
        use_merge=use_merge,
        max_merge_invocations=3,
        merge_val_overlap_floor=1,
        reflection_minibatch_size=2,
        max_metric_calls=100,
        skip_perfect_score=False,
        seed=0,
        logger=_QuietLogger(),
        callbacks=cast(Any, [_Callback(trace)]),
        run_dir=None,
        display_progress_bar=False,
        use_wandb=False,
        use_mlflow=False,
        cache_evaluation=False,
    )
    return {
        "effect_identities": adapter.effect_identities,
        "effect_kinds": adapter.effect_kinds,
        "result": _result_payload(result),
        "timeline": trace.timeline,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--component-selector",
        choices=("round_robin", "all"),
        required=True,
    )
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--effect-log", type=Path)
    parser.add_argument("--crash-after", type=int, default=-1)
    arguments = parser.parse_args()
    print(
        _canonical(
            run_oracle(
                component_selector=arguments.component_selector,
                use_merge=arguments.merge,
                effect_log=arguments.effect_log,
                crash_after=(
                    None
                    if arguments.crash_after < 0
                    else arguments.crash_after
                ),
            )
        )
    )


if __name__ == "__main__":
    main()
