"""COPRO proposal, ranking, lifecycle coordination, and operator artifacts."""

from __future__ import annotations

import csv
import io
import json
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from dr_serialize import sha256_json_digest
from pydantic import (
    BaseModel,
    ConfigDict,
    StrictInt,
    StrictStr,
    model_validator,
)

from whetstone.platform.spec_builder import (
    DEFAULT_HUMANEVAL_INSTRUCTIONS_START,
)
from whetstone.publication import AnalysisBundleReader
from whetstone.records import PredictionSpecRecord

OPTIMIZER_NAME = "copro_minimal"
MANUAL_PROPOSALS: tuple[tuple[str, str], ...] = (
    ("Summarize the code's purpose and main logic steps concisely.", ""),
    ("Describe inputs, outputs, and algorithm in plain English.", ""),
    (
        DEFAULT_HUMANEVAL_INSTRUCTIONS_START,
        "Focus on the entry point behavior.",
    ),
)


@dataclass(frozen=True)
class CoproCandidateResult:
    candidate_id: str
    scoreable_count: int
    pass_count: int
    generation_error_count: int
    score_error_count: int = 0

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
    """Aggregate every task and repeat by one published prompt candidate."""
    predictions = reader.rows(
        "predictions", where="experiment_id = ?", params=(experiment_name,)
    )
    candidate_for_prediction: dict[str, str] = {}
    for row in predictions:
        candidate_id = row.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError(
                "published COPRO prediction has no shared candidate identity"
            )
        candidate_for_prediction[str(row["prediction_id"])] = candidate_id
    totals: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "scoreable": 0,
            "passed": 0,
            "generation_errors": 0,
            "score_errors": 0,
        }
    )
    for candidate_id in candidate_for_prediction.values():
        if candidate_id:
            totals[candidate_id]
    for run in reader.rows("generation_runs"):
        candidate = candidate_for_prediction.get(str(run["prediction_id"]))
        if candidate and run["status"] != "success":
            totals[candidate]["generation_errors"] += 1
    for score in reader.rows("score_attempts"):
        candidate = candidate_for_prediction.get(str(score["prediction_id"]))
        if not candidate:
            continue
        if score["status"] != "success":
            totals[candidate]["score_errors"] += 1
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
            score_error_count=counts["score_errors"],
        )
        for candidate, counts in sorted(totals.items())
    )


def select_best_candidate(
    results: Sequence[CoproCandidateResult],
) -> CoproCandidateResult | None:
    """Rank one pinned snapshot deterministically."""
    if not results:
        return None
    return min(
        results,
        key=lambda result: (
            -(result.pass_rate if result.pass_rate is not None else -1.0),
            -result.scoreable_count,
            result.generation_error_count + result.score_error_count,
            result.candidate_id,
        ),
    )


class CoproCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: StrictStr
    depth: StrictInt
    parent_candidate_id: StrictStr | None
    instructions_start: StrictStr
    instructions_end: StrictStr
    proposal_source: StrictStr
    instructions_digest: StrictStr

    @model_validator(mode="after")
    def validate_identity(self) -> CoproCandidate:
        if self.depth < -1:
            raise ValueError("candidate depth must be at least -1")
        expected = instructions_digest(
            self.instructions_start, self.instructions_end
        )
        if self.instructions_digest != expected:
            raise ValueError("instructions digest does not match candidate")
        if not self.candidate_id:
            raise ValueError("candidate ID must not be empty")
        return self


class CoproDimensions(BaseModel):
    """The validated shared identity published for every candidate cell."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    optimizer: StrictStr = OPTIMIZER_NAME
    copro_run_id: StrictStr
    candidate_id: StrictStr
    candidate_depth: StrictInt
    parent_candidate_id: StrictStr | None
    instructions_digest: StrictStr
    compression_target: float

    @model_validator(mode="after")
    def validate_dimensions(self) -> CoproDimensions:
        if self.optimizer != OPTIMIZER_NAME:
            raise ValueError("COPRO optimizer identity is fixed")
        if not self.copro_run_id or not self.candidate_id:
            raise ValueError("COPRO shared identities must not be empty")
        if not 0 < self.compression_target <= 1:
            raise ValueError("compression target must be in (0, 1]")
        return self


class CoproAttempt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate: CoproCandidate
    experiment_name: StrictStr
    scoreable_count: StrictInt = 0
    pass_count: StrictInt = 0
    generation_error_count: StrictInt = 0
    score_error_count: StrictInt = 0

    @property
    def pass_rate(self) -> float | None:
        if self.scoreable_count == 0:
            return None
        return self.pass_count / self.scoreable_count


class CoproIteration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    depth: StrictInt
    experiment_name: StrictStr
    generation_operation_key: StrictStr | None = None
    scoring_operation_key: StrictStr | None = None
    bundle_id: StrictStr | None = None
    snapshot_seq: StrictInt | None = None
    attempts: tuple[CoproAttempt, ...]


class CoproRunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: StrictStr
    breadth: StrictInt = 3
    depth: StrictInt = 2
    dry_run: bool = False

    @model_validator(mode="after")
    def validate_bounds(self) -> CoproRunConfig:
        if not self.run_id:
            raise ValueError("run ID must not be empty")
        if self.breadth < 1 or self.depth < 1:
            raise ValueError("breadth and depth must be positive")
        return self


class CoproRunResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: StrictStr
    dry_run: bool
    iterations: tuple[CoproIteration, ...]
    best_candidate: CoproCandidate | None = None
    best_attempt: CoproAttempt | None = None


@dataclass(frozen=True)
class CoproPin:
    bundle_id: str
    snapshot_seq: int
    token: object = field(repr=False, compare=False)


class CoproPinLossError(RuntimeError):
    """The immutable Analysis snapshot disappeared before ranking finished."""

    code = "PINNED_BUNDLE_GONE"

    def __init__(self) -> None:
        super().__init__(self.code)


class CoproLifecycle(Protocol):
    def submit_generation(
        self,
        *,
        experiment_name: str,
        operation_key: str,
        specs: tuple[PredictionSpecRecord, ...],
    ) -> None: ...

    def wait(self, operation_key: str) -> None: ...

    def submit_scoring(
        self,
        *,
        experiment_name: str,
        operation_key: str,
        generation_operation_key: str,
        specs: tuple[PredictionSpecRecord, ...],
    ) -> None: ...

    def promote_acceptance(self, experiment_name: str) -> None: ...

    def export_and_pin(self) -> CoproPin: ...

    def read_pinned_candidates(
        self, pin: CoproPin, *, experiment_name: str
    ) -> tuple[CoproCandidateResult, ...]: ...


SpecFactory = Callable[
    [str, tuple[CoproCandidate, ...]], tuple[PredictionSpecRecord, ...]
]
Checkpoint = Callable[[CoproRunResult], object]


def instructions_digest(start: str, end: str) -> str:
    return sha256_json_digest(
        {"instructions_start": start, "instructions_end": end}, length=16
    )


def _candidate(
    *,
    run_id: str,
    depth: int,
    index: int,
    start: str,
    end: str,
    source: str,
    parent: str | None,
) -> CoproCandidate:
    digest = instructions_digest(start, end)
    return CoproCandidate(
        candidate_id=f"{run_id}-d{depth}-c{index}-{digest[:8]}",
        depth=depth,
        parent_candidate_id=parent,
        instructions_start=start,
        instructions_end=end,
        proposal_source=source,
        instructions_digest=digest,
    )


def baseline_candidate(run_id: str) -> CoproCandidate:
    return _candidate(
        run_id=run_id,
        depth=-1,
        index=0,
        start=DEFAULT_HUMANEVAL_INSTRUCTIONS_START,
        end="",
        source="baseline",
        parent=None,
    )


def manual_proposals(
    current_best: CoproCandidate,
    *,
    run_id: str,
    breadth: int,
    depth: int,
) -> tuple[CoproCandidate, ...]:
    pairs = [(current_best.instructions_start, current_best.instructions_end)]
    pairs.extend(pair for pair in MANUAL_PROPOSALS if pair not in pairs)
    if breadth > len(pairs):
        raise ValueError("breadth exceeds the finite manual proposal pool")
    return tuple(
        _candidate(
            run_id=run_id,
            depth=depth,
            index=index,
            start=start,
            end=end,
            source="carry_forward" if index == 0 else "manual",
            parent=current_best.candidate_id,
        )
        for index, (start, end) in enumerate(pairs[:breadth])
    )


def _attempts(
    *,
    candidates: tuple[CoproCandidate, ...],
    experiment_name: str,
    results: Sequence[CoproCandidateResult],
) -> tuple[CoproAttempt, ...]:
    result_by_id = {result.candidate_id: result for result in results}
    attempts: list[CoproAttempt] = []
    for candidate in candidates:
        result = result_by_id.get(candidate.candidate_id)
        attempts.append(
            CoproAttempt(
                candidate=candidate,
                experiment_name=experiment_name,
                scoreable_count=(
                    0 if result is None else result.scoreable_count
                ),
                pass_count=0 if result is None else result.pass_count,
                generation_error_count=(
                    0 if result is None else result.generation_error_count
                ),
                score_error_count=(
                    0 if result is None else result.score_error_count
                ),
            )
        )
    return tuple(attempts)


def _best_attempt(attempts: Sequence[CoproAttempt]) -> CoproAttempt | None:
    scoreable = [
        attempt for attempt in attempts if attempt.scoreable_count > 0
    ]
    if not scoreable:
        return None
    return min(
        scoreable,
        key=lambda attempt: (
            -(attempt.pass_rate if attempt.pass_rate is not None else -1.0),
            -attempt.scoreable_count,
            attempt.generation_error_count + attempt.score_error_count,
            len(attempt.candidate.instructions_start)
            + len(attempt.candidate.instructions_end),
            attempt.candidate.candidate_id,
        ),
    )


def _result(
    config: CoproRunConfig, iterations: Sequence[CoproIteration]
) -> CoproRunResult:
    attempts = [attempt for item in iterations for attempt in item.attempts]
    best = _best_attempt(attempts)
    return CoproRunResult(
        run_id=config.run_id,
        dry_run=config.dry_run,
        iterations=tuple(iterations),
        best_candidate=None if best is None else best.candidate,
        best_attempt=best,
    )


def run_copro_loop(
    *,
    config: CoproRunConfig,
    lifecycle: CoproLifecycle | None,
    spec_factory: SpecFactory,
    checkpoint: Checkpoint | None = None,
    resume: CoproRunResult | None = None,
) -> CoproRunResult:
    """Run the exact wait/export/pinned-read lifecycle once per depth."""
    if not config.dry_run and lifecycle is None:
        raise ValueError("live COPRO requires lifecycle dependencies")
    iterations = [] if resume is None else list(resume.iterations)
    if resume is not None:
        expected_resume = _result(config, iterations)
        if resume != expected_resume:
            raise ValueError(
                "COPRO resume result is not internally consistent"
            )
        if len(iterations) > config.depth:
            raise ValueError("COPRO resume exceeds configured depth")
        for depth, iteration in enumerate(iterations):
            if (
                iteration.depth != depth
                or iteration.experiment_name
                != f"copro_minimal_{config.run_id}_d{depth}"
            ):
                raise ValueError("COPRO resume depths are not contiguous")
    current_best = baseline_candidate(config.run_id)
    if iterations:
        resumed_best = _best_attempt(iterations[-1].attempts)
        if resumed_best is not None:
            current_best = resumed_best.candidate
    for depth in range(len(iterations), config.depth):
        experiment_name = f"copro_minimal_{config.run_id}_d{depth}"
        candidates = manual_proposals(
            current_best,
            run_id=config.run_id,
            breadth=config.breadth,
            depth=depth,
        )
        specs = spec_factory(experiment_name, candidates)
        if not specs:
            raise ValueError("COPRO iteration produced no Prediction specs")
        generation_key = f"copro-{config.run_id}-d{depth}-generation"
        scoring_key = f"copro-{config.run_id}-d{depth}-scoring"
        pin: CoproPin | None = None
        candidate_results: tuple[CoproCandidateResult, ...] = ()
        if not config.dry_run:
            assert lifecycle is not None
            lifecycle.submit_generation(
                experiment_name=experiment_name,
                operation_key=generation_key,
                specs=specs,
            )
            lifecycle.wait(generation_key)
            lifecycle.submit_scoring(
                experiment_name=experiment_name,
                operation_key=scoring_key,
                generation_operation_key=generation_key,
                specs=specs,
            )
            lifecycle.wait(scoring_key)
            lifecycle.promote_acceptance(experiment_name)
            pin = lifecycle.export_and_pin()
            candidate_results = lifecycle.read_pinned_candidates(
                pin, experiment_name=experiment_name
            )
            if {result.candidate_id for result in candidate_results} != {
                candidate.candidate_id for candidate in candidates
            }:
                raise RuntimeError(
                    "pinned COPRO results do not cover every candidate exactly"
                )
        attempts = _attempts(
            candidates=candidates,
            experiment_name=experiment_name,
            results=candidate_results,
        )
        iteration = CoproIteration(
            depth=depth,
            experiment_name=experiment_name,
            generation_operation_key=(
                None if config.dry_run else generation_key
            ),
            scoring_operation_key=None if config.dry_run else scoring_key,
            bundle_id=None if pin is None else pin.bundle_id,
            snapshot_seq=None if pin is None else pin.snapshot_seq,
            attempts=attempts,
        )
        iterations.append(iteration)
        best = _best_attempt(attempts)
        if best is not None:
            current_best = best.candidate
        partial = _result(config, iterations)
        if checkpoint is not None:
            checkpoint(partial)
    return _result(config, iterations)


def render_copro_artifacts(result: CoproRunResult) -> Mapping[str, str]:
    """Render the complete human-facing view of one committed generation."""
    candidates = [
        attempt.candidate
        for iteration in result.iterations
        for attempt in iteration.attempts
    ]
    attempts = [
        attempt
        for iteration in result.iterations
        for attempt in iteration.attempts
    ]
    run_json = result.model_dump_json(indent=2) + "\n"
    candidates_jsonl = "".join(
        candidate.model_dump_json() + "\n" for candidate in candidates
    )
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=(
            "candidate_id",
            "depth",
            "experiment_name",
            "scoreable_count",
            "pass_count",
            "pass_rate",
            "generation_error_count",
            "score_error_count",
        ),
    )
    writer.writeheader()
    for attempt in attempts:
        writer.writerow(
            {
                "candidate_id": attempt.candidate.candidate_id,
                "depth": attempt.candidate.depth,
                "experiment_name": attempt.experiment_name,
                "scoreable_count": attempt.scoreable_count,
                "pass_count": attempt.pass_count,
                "pass_rate": attempt.pass_rate,
                "generation_error_count": attempt.generation_error_count,
                "score_error_count": attempt.score_error_count,
            }
        )
    best_prompt_json = (
        json.dumps(
            {
                "best_candidate": (
                    None
                    if result.best_candidate is None
                    else result.best_candidate.model_dump(mode="json")
                ),
                "best_attempt": (
                    None
                    if result.best_attempt is None
                    else result.best_attempt.model_dump(mode="json")
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return {
        "run.json": run_json,
        "candidates.jsonl": candidates_jsonl,
        "attempts.csv": buffer.getvalue(),
        "best_prompt.json": best_prompt_json,
    }
