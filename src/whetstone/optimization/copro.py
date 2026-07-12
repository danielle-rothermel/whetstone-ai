"""COPRO selection over a pinned Whetstone Analysis Bundle."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from whetstone.publication import AnalysisBundleReader


@dataclass(frozen=True)
class CoproCandidateResult:
    candidate_id: str
    scoreable_count: int
    pass_count: int
    generation_error_count: int

    @property
    def pass_rate(self) -> float | None:
        if self.scoreable_count == 0:
            return None
        return self.pass_count / self.scoreable_count


def summarize_pinned_candidates(
    reader: AnalysisBundleReader,
    *,
    experiment_name: str,
) -> tuple[CoproCandidateResult, ...]:
    """Summarize only one immutable Analysis snapshot for COPRO ranking."""

    predictions = reader.rows(
        "predictions", where="experiment_name = ?", params=(experiment_name,)
    )
    candidate_for_prediction = {
        str(row["prediction_id"]): str(row.get("candidate_id", ""))
        for row in predictions
    }
    totals: dict[str, dict[str, int]] = defaultdict(
        lambda: {"scoreable": 0, "passed": 0, "generation_errors": 0}
    )
    for run in reader.rows("generation_runs"):
        candidate = candidate_for_prediction.get(str(run["prediction_id"]))
        if candidate and run["status"] != "success":
            totals[candidate]["generation_errors"] += 1
    for score in reader.rows("score_attempts"):
        candidate = candidate_for_prediction.get(str(score["prediction_id"]))
        if not candidate or score["status"] != "success":
            continue
        totals[candidate]["scoreable"] += 1
        if float(score["score"]) > 0:
            totals[candidate]["passed"] += 1
    return tuple(
        CoproCandidateResult(
            candidate_id=candidate,
            scoreable_count=counts["scoreable"],
            pass_count=counts["passed"],
            generation_error_count=counts["generation_errors"],
        )
        for candidate, counts in sorted(totals.items())
    )


def select_best_candidate(
    results: Sequence[CoproCandidateResult],
) -> CoproCandidateResult | None:
    """Use a deterministic ranking without reading operational tables."""

    if not results:
        return None
    return min(
        results,
        key=lambda result: (
            -(result.pass_rate if result.pass_rate is not None else -1.0),
            -result.scoreable_count,
            result.generation_error_count,
            result.candidate_id,
        ),
    )
