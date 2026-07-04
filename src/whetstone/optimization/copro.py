"""Minimal COPRO-style encoder prompt optimizer for v1 enc-dec experiments."""

from __future__ import annotations

import csv
import json
import re
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import pandas as pd
from dr_serialize import sha256_json_digest
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    StrictStr,
    model_validator,
)
from sqlalchemy import Engine, func, select
from sqlalchemy.engine import Connection

from whetstone.analysis.frames import (
    load_encdec_analysis_frame,
    pass_mask,
    score_success_mask,
)
from whetstone.db import schema
from whetstone.lm.boundary import (
    EndpointKind,
    PlainPromptAdapter,
    ProviderKind,
    call_provider_request,
    parse_provider_response,
)
from whetstone.platform.graph_workflow import (
    run_prediction_graph_workflow_once,
)
from whetstone.platform.node_execution import (
    build_provider_request,
    create_provider_client,
    runtime_provider_config,
)
from whetstone.platform.rescoring import rescore_generation_runs
from whetstone.platform.scoring_workflow import (
    run_score_generation_workflow_once,
)
from whetstone.platform.spec_builder import (
    DEFAULT_CONFIGS_ROOT,
    DEFAULT_HUMANEVAL_INSTRUCTIONS_START,
    ExperimentSpecConfig,
    GraphLayout,
    HumanevalEncDecConfig,
    ModelConfigFragment,
    SplitConfigFragment,
    graph_for_layout,
    humaneval_encdec_task_snapshot,
    load_model_config_fragment,
    load_split_config_fragment,
    prediction_spec,
    provider_ref_from_config,
    resolve_config_path,
    sample_tasks_for_config,
)
from whetstone.platform.submission import (
    bulk_insert_prediction_specs,
    idempotent_insert_experiment,
    submit_prediction_specs,
)
from whetstone.records import (
    DimensionsPayload,
    ExperimentRecord,
    GenerationRunStatus,
    PredictionSpecRecord,
    ProviderConfigRef,
    ScoreAttemptStatus,
    stable_generation_run_id,
)

OPTIMIZER_NAME = "copro_minimal"
INSTRUCTIONS_DIGEST_LENGTH = 16
COPRO_RUN_ID_LENGTH = 12
MANUAL_PROPOSAL_POOL: tuple[tuple[str, str], ...] = (
    (
        "Summarize the code's purpose and main logic steps concisely.",
        "",
    ),
    (
        "Describe inputs, outputs, and algorithm in plain English.",
        "",
    ),
    (
        DEFAULT_HUMANEVAL_INSTRUCTIONS_START,
        "Focus on the entry point behavior.",
    ),
)
LM_PROPOSAL_PROMPT_TEMPLATE = """\
You are optimizing encoder instructions for a code-compression experiment.
The encoder sees Python ground-truth code and must describe it within a fixed
character budget. The decoder will write Python code from the description.

Return JSON only:
{{"candidates":[{{"instructions_start":"...","instructions_end":"..."}}]}}

Baseline instructions_start:
{baseline_start}

Baseline instructions_end:
{baseline_end}

Prior attempts, ordered by score:
{attempt_history}
"""
TERMINAL_GENERATION_STATUSES = frozenset(
    status.value for status in GenerationRunStatus
)


class CoproProposalMode(StrEnum):
    MANUAL = "manual"
    LM = "lm"


class CoproExecutionMode(StrEnum):
    SYNC = "sync"
    QUEUE = "queue"


class CoproCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: StrictStr
    depth: StrictInt
    parent_candidate_id: StrictStr | None = None
    instructions_start: StrictStr
    instructions_end: StrictStr
    proposal_source: StrictStr
    instructions_digest: StrictStr

    @model_validator(mode="after")
    def validate_digest(self) -> CoproCandidate:
        expected = instructions_digest(
            self.instructions_start,
            self.instructions_end,
        )
        if self.instructions_digest != expected:
            raise ValueError("instructions_digest must match instruction pair")
        return self


class CoproAttempt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: StrictStr
    depth: StrictInt
    parent_candidate_id: StrictStr | None = None
    instructions_start: StrictStr
    instructions_end: StrictStr
    proposal_source: StrictStr
    instructions_digest: StrictStr
    experiment_name: StrictStr
    scoreable_count: StrictInt = 0
    pass_count: StrictInt = 0
    pass_rate: float | None = None
    generation_error_count: StrictInt = 0
    score_error_count: StrictInt = 0


class CoproRunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_config_path: Path
    split_path: Path
    compression_targets: tuple[StrictFloat, ...]
    breadth: StrictInt
    depth: StrictInt
    repetition_seeds: tuple[StrictInt, ...]
    proposal_mode: CoproProposalMode = CoproProposalMode.MANUAL
    execution_mode: CoproExecutionMode = CoproExecutionMode.SYNC
    prompt_model: StrictStr | None = None
    prompt_provider_kind: ProviderKind = ProviderKind.OPENAI
    prompt_endpoint_kind: EndpointKind = EndpointKind.RESPONSES
    output_dir: Path
    configs_root: Path = DEFAULT_CONFIGS_ROOT
    fair_order_seed: StrictStr = "copro-minimal-v1"
    min_encoder_char_budget: StrictInt = 50
    rescore_max_in_flight: StrictInt = 100
    generation_poll_interval_seconds: float = 2.0
    generation_poll_timeout_seconds: float = 3600.0
    dry_run: bool = False

    @model_validator(mode="after")
    def validate_run_config(self) -> CoproRunConfig:
        if self.breadth < 1:
            raise ValueError("breadth must be at least 1")
        if self.depth < 1:
            raise ValueError("depth must be at least 1")
        if not self.compression_targets:
            raise ValueError("compression_targets must not be empty")
        if not self.repetition_seeds:
            raise ValueError("repetition_seeds must not be empty")
        if (
            self.proposal_mode is CoproProposalMode.LM
            and not self.prompt_model
        ):
            raise ValueError("prompt_model is required for lm proposal mode")
        return self


class CoproRunResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: StrictStr
    experiment_name: StrictStr
    config: CoproRunConfig
    candidates: tuple[CoproCandidate, ...]
    attempts: tuple[CoproAttempt, ...]
    best_candidate: CoproCandidate | None = None
    best_attempt: CoproAttempt | None = None
    command: StrictStr
    caveats: tuple[StrictStr, ...] = Field(default_factory=tuple)


def make_copro_run_id() -> str:
    return sha256_json_digest(
        {
            "run_id": str(uuid.uuid4()),
            "created_at": datetime.now(UTC).isoformat(),
        },
        length=COPRO_RUN_ID_LENGTH,
    )


def make_candidate_id(depth: int, index: int) -> str:
    return f"d{depth}_c{index}"


def instructions_digest(instructions_start: str, instructions_end: str) -> str:
    return sha256_json_digest(
        {
            "instructions_start": instructions_start,
            "instructions_end": instructions_end,
        },
        length=INSTRUCTIONS_DIGEST_LENGTH,
    )


def baseline_candidate(*, depth: int = 0) -> CoproCandidate:
    start = DEFAULT_HUMANEVAL_INSTRUCTIONS_START
    end = ""
    return CoproCandidate(
        candidate_id=make_candidate_id(depth, 0),
        depth=depth,
        parent_candidate_id=None,
        instructions_start=start,
        instructions_end=end,
        proposal_source="baseline",
        instructions_digest=instructions_digest(start, end),
    )


def build_copro_dimensions(
    *,
    candidate: CoproCandidate,
    copro_run_id: str,
    compression_target: float,
    encoder_model: str | None = None,
    decoder_model: str | None = None,
) -> DimensionsPayload:
    values: dict[str, Any] = {
        "compression_target": compression_target,
        "temperature": 0,
        "optimizer": OPTIMIZER_NAME,
        "copro_run_id": copro_run_id,
        "candidate_id": candidate.candidate_id,
        "candidate_depth": candidate.depth,
        "parent_candidate_id": candidate.parent_candidate_id,
        "instructions_digest": candidate.instructions_digest,
    }
    if encoder_model is not None:
        values["encoder_model"] = encoder_model
    if decoder_model is not None:
        values["decoder_model"] = decoder_model
    return DimensionsPayload(values=values)


def _candidate_from_pair(
    *,
    depth: int,
    index: int,
    instructions_start: str,
    instructions_end: str,
    proposal_source: str,
    parent_candidate_id: str | None,
) -> CoproCandidate:
    return CoproCandidate(
        candidate_id=make_candidate_id(depth, index),
        depth=depth,
        parent_candidate_id=parent_candidate_id,
        instructions_start=instructions_start,
        instructions_end=instructions_end,
        proposal_source=proposal_source,
        instructions_digest=instructions_digest(
            instructions_start,
            instructions_end,
        ),
    )


def manual_proposal_candidates(
    current_best: CoproCandidate,
    *,
    breadth: int,
    depth: int,
) -> tuple[CoproCandidate, ...]:
    if breadth < 1:
        raise ValueError("breadth must be at least 1")
    parent_id = current_best.candidate_id
    candidates: list[CoproCandidate] = [
        _candidate_from_pair(
            depth=depth,
            index=0,
            instructions_start=current_best.instructions_start,
            instructions_end=current_best.instructions_end,
            proposal_source="carry_forward",
            parent_candidate_id=parent_id,
        )
    ]
    seen = {
        (
            current_best.instructions_start,
            current_best.instructions_end,
        )
    }
    pool_index = 0
    while len(candidates) < breadth:
        start, end = MANUAL_PROPOSAL_POOL[
            pool_index % len(MANUAL_PROPOSAL_POOL)
        ]
        pool_index += 1
        key = (start, end)
        if key in seen:
            if pool_index > len(MANUAL_PROPOSAL_POOL) * breadth:
                break
            continue
        seen.add(key)
        candidates.append(
            _candidate_from_pair(
                depth=depth,
                index=len(candidates),
                instructions_start=start,
                instructions_end=end,
                proposal_source="manual",
                parent_candidate_id=parent_id,
            )
        )
    return tuple(candidates)


def _extract_json_object(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if match is None:
        raise ValueError("LM proposal response did not contain JSON object")
    return match.group(0)


def parse_lm_proposal_response(text: str) -> tuple[dict[str, str], ...]:
    try:
        payload = json.loads(_extract_json_object(text))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"LM proposal response is not valid JSON: {error.msg}"
        ) from error
    if not isinstance(payload, dict):
        raise ValueError("LM proposal response root must be a JSON object")
    raw_candidates = payload.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("LM proposal response must include candidates list")
    parsed: list[dict[str, str]] = []
    for index, item in enumerate(raw_candidates):
        if not isinstance(item, dict):
            raise ValueError(f"candidate {index} must be a JSON object")
        start = item.get("instructions_start")
        end = item.get("instructions_end")
        if not isinstance(start, str) or not isinstance(end, str):
            raise ValueError(
                f"candidate {index} must include string "
                "instructions_start and instructions_end"
            )
        parsed.append(
            {
                "instructions_start": start,
                "instructions_end": end,
            }
        )
    return tuple(parsed)


def render_attempt_history(attempts: Sequence[CoproAttempt]) -> str:
    if not attempts:
        return "(none)"
    ordered = sorted(
        attempts,
        key=lambda attempt: (
            -(attempt.pass_rate or -1.0),
            -attempt.scoreable_count,
            attempt.generation_error_count + attempt.score_error_count,
            len(attempt.instructions_start) + len(attempt.instructions_end),
            attempt.candidate_id,
        ),
    )
    lines: list[str] = []
    for attempt in ordered:
        rate = (
            "n/a" if attempt.pass_rate is None else f"{attempt.pass_rate:.3f}"
        )
        lines.append(
            f"- candidate_id={attempt.candidate_id} depth={attempt.depth} "
            f"pass_rate={rate} scoreable={attempt.scoreable_count} "
            f"instructions_start={attempt.instructions_start!r} "
            f"instructions_end={attempt.instructions_end!r}"
        )
    return "\n".join(lines)


def propose_lm_candidates(
    current_best: CoproCandidate,
    *,
    breadth: int,
    depth: int,
    prior_attempts: Sequence[CoproAttempt],
    prompt_model: str,
    prompt_provider_kind: ProviderKind,
    prompt_endpoint_kind: EndpointKind,
) -> tuple[CoproCandidate, ...]:
    if breadth < 1:
        raise ValueError("breadth must be at least 1")
    parent_id = current_best.candidate_id
    candidates: list[CoproCandidate] = [
        _candidate_from_pair(
            depth=depth,
            index=0,
            instructions_start=current_best.instructions_start,
            instructions_end=current_best.instructions_end,
            proposal_source="carry_forward",
            parent_candidate_id=parent_id,
        )
    ]
    if breadth == 1:
        return tuple(candidates)

    provider_ref = ProviderConfigRef(
        provider_kind=prompt_provider_kind,
        endpoint_kind=prompt_endpoint_kind,
        model=prompt_model,
        config_id="prompt",
        throttle_key=(
            f"{prompt_provider_kind.value}:"
            f"{prompt_endpoint_kind.value}:{prompt_model}"
        ),
        parameters={"temperature": 1.0},
    )
    provider_config = runtime_provider_config(provider_ref)
    prompt = LM_PROPOSAL_PROMPT_TEMPLATE.format(
        baseline_start=current_best.instructions_start,
        baseline_end=current_best.instructions_end,
        attempt_history=render_attempt_history(prior_attempts),
    )
    adapter = PlainPromptAdapter()
    messages = adapter.messages(user_content=prompt)
    request = build_provider_request(
        config=provider_config,
        messages=messages,
        parameters={"temperature": 1.0},
    )
    client = create_provider_client(provider_config)
    response = call_provider_request(client, request)
    result = parse_provider_response(
        response,
        config=provider_config,
        output_field=adapter.output_field,
    )
    proposals = parse_lm_proposal_response(result.text)
    seen = {
        (
            current_best.instructions_start,
            current_best.instructions_end,
        )
    }
    for proposal in proposals:
        if len(candidates) >= breadth:
            break
        key = (proposal["instructions_start"], proposal["instructions_end"])
        if key in seen:
            continue
        seen.add(key)
        candidates.append(
            _candidate_from_pair(
                depth=depth,
                index=len(candidates),
                instructions_start=proposal["instructions_start"],
                instructions_end=proposal["instructions_end"],
                proposal_source="lm",
                parent_candidate_id=parent_id,
            )
        )
    if len(candidates) < breadth:
        for start, end in MANUAL_PROPOSAL_POOL:
            if len(candidates) >= breadth:
                break
            key = (start, end)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                _candidate_from_pair(
                    depth=depth,
                    index=len(candidates),
                    instructions_start=start,
                    instructions_end=end,
                    proposal_source="manual_fallback",
                    parent_candidate_id=parent_id,
                )
            )
    return tuple(candidates)


def resolve_user_config_path(configs_root: Path, path: Path) -> Path:
    if path.is_absolute():
        return path
    text = str(path).replace("\\", "/")
    if text.startswith("configs/"):
        text = text.removeprefix("configs/")
    return resolve_config_path(configs_root, text)


def load_copro_model_and_split(
    config: CoproRunConfig,
) -> tuple[ModelConfigFragment, SplitConfigFragment]:
    model = load_model_config_fragment(
        resolve_user_config_path(config.configs_root, config.model_config_path)
    )
    split = load_split_config_fragment(
        resolve_user_config_path(config.configs_root, config.split_path)
    )
    return model, split


def build_experiment_spec_config(
    *,
    config: CoproRunConfig,
    experiment_name: str,
    model: ModelConfigFragment,
    split: SplitConfigFragment,
) -> ExperimentSpecConfig:
    dimensions_axes = tuple(
        {"compression_target": target} for target in config.compression_targets
    )
    return ExperimentSpecConfig(
        experiment_name=experiment_name,
        graph_layout=GraphLayout.ENCDEC,
        dataset=split.dataset,
        fair_order_seed=config.fair_order_seed,
        repetition_seeds=config.repetition_seeds,
        dimensions_axes=dimensions_axes,
        providers=model.providers,
        encdec_shape="humaneval",
        humaneval_encdec=HumanevalEncDecConfig(
            min_encoder_char_budget=config.min_encoder_char_budget,
        ),
    )


def iter_candidate_specs(
    *,
    config: CoproRunConfig,
    candidate: CoproCandidate,
    experiment_name: str,
    copro_run_id: str,
    model: ModelConfigFragment,
    split: SplitConfigFragment,
) -> Iterator[PredictionSpecRecord]:
    experiment_config = build_experiment_spec_config(
        config=config,
        experiment_name=experiment_name,
        model=model,
        split=split,
    )
    humaneval_cfg = HumanevalEncDecConfig(
        instructions_start=candidate.instructions_start,
        instructions_end=candidate.instructions_end,
        min_encoder_char_budget=config.min_encoder_char_budget,
    )
    graph = graph_for_layout(
        GraphLayout.ENCDEC,
        encdec_shape="humaneval",
        humaneval_encdec=humaneval_cfg,
    )
    providers = tuple(
        provider_ref_from_config(provider) for provider in model.providers
    )
    provider_axis = providers[0]
    encoder_model = next(
        (
            provider.model
            for provider in providers
            if provider.config_id == "encoder"
        ),
        providers[0].model,
    )
    decoder_model = next(
        (
            provider.model
            for provider in providers
            if provider.config_id == "decoder"
        ),
        providers[-1].model,
    )
    sampled_tasks = sample_tasks_for_config(experiment_config)
    for sampled in sampled_tasks:
        for repetition_seed in config.repetition_seeds:
            for compression_target in config.compression_targets:
                dimensions = build_copro_dimensions(
                    candidate=candidate,
                    copro_run_id=copro_run_id,
                    compression_target=float(compression_target),
                    encoder_model=encoder_model,
                    decoder_model=decoder_model,
                )
                task_snapshot = humaneval_encdec_task_snapshot(
                    sampled.task,
                    compression_target=float(compression_target),
                    humaneval_encdec=humaneval_cfg,
                )
                yield prediction_spec(
                    graph,
                    providers=providers,
                    provider_axis=provider_axis,
                    layout=GraphLayout.ENCDEC.value,
                    task=task_snapshot,
                    task_id=sampled.task.task_id,
                    dimensions=dimensions,
                    experiment_name=experiment_name,
                    repetition_seed=repetition_seed,
                    fair_order_seed=config.fair_order_seed,
                )


def _dimension_values(dimensions: Any) -> dict[str, Any]:
    if dimensions is None:
        return {}
    if isinstance(dimensions, str):
        dimensions = json.loads(dimensions)
    if not isinstance(dimensions, dict):
        return {}
    values = dimensions.get("values")
    if isinstance(values, dict):
        return values
    return dimensions


def summarize_attempts(
    frame: pd.DataFrame,
    *,
    experiment_name: str,
    candidates: Sequence[CoproCandidate],
) -> tuple[CoproAttempt, ...]:
    candidate_by_id = {
        candidate.candidate_id: candidate for candidate in candidates
    }
    if frame.empty:
        return tuple(
            CoproAttempt(
                candidate_id=candidate.candidate_id,
                depth=candidate.depth,
                parent_candidate_id=candidate.parent_candidate_id,
                instructions_start=candidate.instructions_start,
                instructions_end=candidate.instructions_end,
                proposal_source=candidate.proposal_source,
                instructions_digest=candidate.instructions_digest,
                experiment_name=experiment_name,
            )
            for candidate in candidates
        )

    frame = frame.copy()
    frame["candidate_id"] = frame["dimensions"].apply(
        lambda value: _dimension_values(value).get("candidate_id")
    )
    attempts: list[CoproAttempt] = []
    for candidate_id, group in frame.groupby("candidate_id", dropna=False):
        if candidate_id is None or pd.isna(candidate_id):
            continue
        candidate = candidate_by_id.get(str(candidate_id))
        if candidate is None:
            continue
        score_success = score_success_mask(group)
        scoreable = group.loc[score_success]
        passed = (
            pass_mask(scoreable)
            if score_success.any()
            else pd.Series([], dtype=bool)
        )
        pass_rate = float(passed.mean()) if score_success.any() else None
        gen_errors = (
            group["generation_status"] == GenerationRunStatus.ERROR.value
        ).sum()
        score_errors = (
            group["score_status"] == ScoreAttemptStatus.ERROR.value
        ).sum()
        attempts.append(
            CoproAttempt(
                candidate_id=candidate.candidate_id,
                depth=candidate.depth,
                parent_candidate_id=candidate.parent_candidate_id,
                instructions_start=candidate.instructions_start,
                instructions_end=candidate.instructions_end,
                proposal_source=candidate.proposal_source,
                instructions_digest=candidate.instructions_digest,
                experiment_name=experiment_name,
                scoreable_count=int(score_success.sum()),
                pass_count=int(passed.sum()) if score_success.any() else 0,
                pass_rate=pass_rate,
                generation_error_count=int(gen_errors),
                score_error_count=int(score_errors),
            )
        )
    known_ids = {attempt.candidate_id for attempt in attempts}
    for candidate in candidates:
        if candidate.candidate_id not in known_ids:
            attempts.append(
                CoproAttempt(
                    candidate_id=candidate.candidate_id,
                    depth=candidate.depth,
                    parent_candidate_id=candidate.parent_candidate_id,
                    instructions_start=candidate.instructions_start,
                    instructions_end=candidate.instructions_end,
                    proposal_source=candidate.proposal_source,
                    instructions_digest=candidate.instructions_digest,
                    experiment_name=experiment_name,
                )
            )
    return tuple(attempts)


def _combined_instruction_length(attempt: CoproAttempt) -> int:
    return len(attempt.instructions_start) + len(attempt.instructions_end)


def select_best_candidate(
    attempts: Sequence[CoproAttempt],
) -> CoproAttempt | None:
    if not attempts:
        return None
    ordered = sorted(
        attempts,
        key=lambda attempt: (
            -(attempt.pass_rate if attempt.pass_rate is not None else -1.0),
            -attempt.scoreable_count,
            attempt.generation_error_count + attempt.score_error_count,
            _combined_instruction_length(attempt),
            attempt.candidate_id,
        ),
    )
    return ordered[0]


def insert_experiment_and_specs(
    engine: Engine,
    *,
    experiment_name: str,
    specs: Sequence[PredictionSpecRecord],
    copro_run_id: str,
) -> None:
    with engine.begin() as connection:
        connection.execute(
            idempotent_insert_experiment(
                ExperimentRecord(
                    experiment_name=experiment_name,
                    description="Minimal COPRO enc-dec optimizer run",
                    config_metadata={
                        "optimizer": OPTIMIZER_NAME,
                        "copro_run_id": copro_run_id,
                    },
                )
            )
        )
        bulk_insert_prediction_specs(connection, specs)


def evaluate_specs_sync(
    *,
    database_url: str,
    specs: Sequence[PredictionSpecRecord],
) -> list[str]:
    generation_run_ids: list[str] = []
    for spec in specs:
        run_prediction_graph_workflow_once(
            database_url=database_url,
            prediction_id=spec.prediction_id,
            attempt_index=0,
        )
        generation_run_id = stable_generation_run_id(
            prediction_id=spec.prediction_id,
            attempt_index=0,
        )
        generation_run_ids.append(generation_run_id)
        run_score_generation_workflow_once(
            database_url=database_url,
            generation_run_id=generation_run_id,
        )
    return generation_run_ids


def _count_non_terminal_generation_runs(
    connection: Connection,
    *,
    experiment_name: str,
    prediction_ids: Sequence[str],
) -> int:
    if not prediction_ids:
        return 0
    statement = (
        select(func.count())
        .select_from(schema.generation_runs.join(schema.prediction_specs))
        .where(
            schema.prediction_specs.c.experiment_name == experiment_name,
            schema.prediction_specs.c.prediction_id.in_(prediction_ids),
            schema.generation_runs.c.prediction_id
            == schema.prediction_specs.c.prediction_id,
            schema.generation_runs.c.status.notin_(
                TERMINAL_GENERATION_STATUSES
            ),
        )
    )
    return int(connection.execute(statement).scalar_one())


def _count_missing_generation_runs(
    connection: Connection,
    *,
    experiment_name: str,
    prediction_ids: Sequence[str],
) -> int:
    if not prediction_ids:
        return 0
    existing = select(schema.generation_runs.c.prediction_id).where(
        schema.generation_runs.c.prediction_id.in_(prediction_ids)
    )
    statement = (
        select(func.count())
        .select_from(schema.prediction_specs)
        .where(
            schema.prediction_specs.c.experiment_name == experiment_name,
            schema.prediction_specs.c.prediction_id.in_(prediction_ids),
            schema.prediction_specs.c.prediction_id.notin_(existing),
        )
    )
    return int(connection.execute(statement).scalar_one())


def wait_for_generation_runs(
    engine: Engine,
    *,
    experiment_name: str,
    prediction_ids: Sequence[str],
    poll_interval_seconds: float,
    poll_timeout_seconds: float,
) -> None:
    deadline = time.monotonic() + poll_timeout_seconds
    while time.monotonic() < deadline:
        with engine.connect() as connection:
            missing = _count_missing_generation_runs(
                connection,
                experiment_name=experiment_name,
                prediction_ids=prediction_ids,
            )
            non_terminal = _count_non_terminal_generation_runs(
                connection,
                experiment_name=experiment_name,
                prediction_ids=prediction_ids,
            )
        if missing == 0 and non_terminal == 0:
            return
        time.sleep(poll_interval_seconds)
    raise TimeoutError(
        "timed out waiting for generation runs to complete; "
        "ensure `python -m whetstone.platform.worker worker` is running"
    )


def evaluate_specs_queue(
    engine: Engine,
    *,
    database_url: str,
    experiment_name: str,
    specs: Sequence[PredictionSpecRecord],
    operation_key: str,
    max_in_flight: int,
    poll_interval_seconds: float,
    poll_timeout_seconds: float,
) -> None:
    submit_prediction_specs(
        engine,
        database_url=database_url,
        operation_key=operation_key,
        experiment_name=experiment_name,
        specs=specs,
        metadata={"optimizer": OPTIMIZER_NAME},
    )
    prediction_ids = [spec.prediction_id for spec in specs]
    wait_for_generation_runs(
        engine,
        experiment_name=experiment_name,
        prediction_ids=prediction_ids,
        poll_interval_seconds=poll_interval_seconds,
        poll_timeout_seconds=poll_timeout_seconds,
    )
    rescore_generation_runs(
        engine,
        database_url=database_url,
        experiment_name=experiment_name,
        max_in_flight=max_in_flight,
    )


def evaluate_candidate_specs(
    engine: Engine,
    *,
    database_url: str,
    config: CoproRunConfig,
    experiment_name: str,
    copro_run_id: str,
    specs: Sequence[PredictionSpecRecord],
    depth: int,
) -> None:
    if config.dry_run:
        return
    insert_experiment_and_specs(
        engine,
        experiment_name=experiment_name,
        specs=specs,
        copro_run_id=copro_run_id,
    )
    if config.execution_mode is CoproExecutionMode.SYNC:
        evaluate_specs_sync(database_url=database_url, specs=specs)
        return
    operation_key = f"copro-{copro_run_id}-d{depth}"
    evaluate_specs_queue(
        engine,
        database_url=database_url,
        experiment_name=experiment_name,
        specs=specs,
        operation_key=operation_key,
        max_in_flight=config.rescore_max_in_flight,
        poll_interval_seconds=config.generation_poll_interval_seconds,
        poll_timeout_seconds=config.generation_poll_timeout_seconds,
    )


def propose_candidates_for_depth(
    config: CoproRunConfig,
    *,
    depth: int,
    current_best: CoproCandidate,
    prior_attempts: Sequence[CoproAttempt],
) -> tuple[CoproCandidate, ...]:
    if config.proposal_mode is CoproProposalMode.MANUAL:
        return manual_proposal_candidates(
            current_best,
            breadth=config.breadth,
            depth=depth,
        )
    assert config.prompt_model is not None
    return propose_lm_candidates(
        current_best,
        breadth=config.breadth,
        depth=depth,
        prior_attempts=prior_attempts,
        prompt_model=config.prompt_model,
        prompt_provider_kind=config.prompt_provider_kind,
        prompt_endpoint_kind=config.prompt_endpoint_kind,
    )


def _matches_candidate_depth(value: Any, depth: int) -> bool:
    return _dimension_values(value).get("candidate_depth") == depth


def run_copro_loop(
    engine: Engine,
    *,
    database_url: str,
    config: CoproRunConfig,
    run_id: str | None = None,
    on_depth_complete: Callable[[int, tuple[CoproAttempt, ...]], None]
    | None = None,
) -> CoproRunResult:
    resolved_run_id = run_id or make_copro_run_id()
    experiment_name = f"copro_minimal_{resolved_run_id}"
    model, split = load_copro_model_and_split(config)
    all_candidates: list[CoproCandidate] = []
    all_attempts: list[CoproAttempt] = []
    current_best = baseline_candidate(depth=0)
    caveats: list[str] = []

    for depth in range(config.depth):
        candidates = propose_candidates_for_depth(
            config,
            depth=depth,
            current_best=current_best,
            prior_attempts=all_attempts,
        )
        all_candidates.extend(candidates)
        specs = [
            spec
            for candidate in candidates
            for spec in iter_candidate_specs(
                config=config,
                candidate=candidate,
                experiment_name=experiment_name,
                copro_run_id=resolved_run_id,
                model=model,
                split=split,
            )
        ]
        evaluate_candidate_specs(
            engine,
            database_url=database_url,
            config=config,
            experiment_name=experiment_name,
            copro_run_id=resolved_run_id,
            specs=specs,
            depth=depth,
        )
        if config.dry_run:
            depth_attempts = summarize_attempts(
                pd.DataFrame(),
                experiment_name=experiment_name,
                candidates=candidates,
            )
        else:
            frame = load_encdec_analysis_frame(engine, [experiment_name])
            frame = frame[
                frame["dimensions"].apply(
                    lambda value, target_depth=depth: _matches_candidate_depth(
                        value,
                        target_depth,
                    )
                )
            ]
            depth_attempts = summarize_attempts(
                frame,
                experiment_name=experiment_name,
                candidates=candidates,
            )
        all_attempts.extend(depth_attempts)
        if on_depth_complete is not None:
            on_depth_complete(depth, depth_attempts)
        best_depth_attempt = select_best_candidate(depth_attempts)
        if best_depth_attempt is None:
            caveats.append(f"depth {depth}: no scoreable attempts")
            continue
        if best_depth_attempt.scoreable_count == 0:
            caveats.append(
                f"depth {depth}: sparse data; selected "
                f"{best_depth_attempt.candidate_id} without score-success rows"
            )
        matching = next(
            (
                candidate
                for candidate in candidates
                if candidate.candidate_id == best_depth_attempt.candidate_id
            ),
            None,
        )
        if matching is not None:
            current_best = matching

    best_attempt = select_best_candidate(all_attempts)
    best_candidate = None
    if best_attempt is not None:
        best_candidate = next(
            (
                candidate
                for candidate in all_candidates
                if candidate.candidate_id == best_attempt.candidate_id
            ),
            None,
        )
    if (
        config.execution_mode is CoproExecutionMode.QUEUE
        and not config.dry_run
    ):
        caveats.append(
            "queue execution requires an external platform worker "
            "for generation"
        )
    return CoproRunResult(
        run_id=resolved_run_id,
        experiment_name=experiment_name,
        config=config,
        candidates=tuple(all_candidates),
        attempts=tuple(all_attempts),
        best_candidate=best_candidate,
        best_attempt=best_attempt,
        command="",
        caveats=tuple(caveats),
    )


def render_summary_markdown(result: CoproRunResult) -> str:
    lines = [
        "# COPRO minimal enc-dec run",
        "",
        "## Run config",
        "",
        f"- run_id: `{result.run_id}`",
        f"- experiment_name: `{result.experiment_name}`",
        f"- model_config: `{result.config.model_config_path}`",
        f"- split: `{result.config.split_path}`",
        f"- compression_targets: {list(result.config.compression_targets)}",
        f"- breadth: {result.config.breadth}",
        f"- depth: {result.config.depth}",
        f"- repeats: {list(result.config.repetition_seeds)}",
        f"- proposal_mode: {result.config.proposal_mode.value}",
        f"- execution_mode: {result.config.execution_mode.value}",
        f"- dry_run: {result.config.dry_run}",
        "",
        "## Candidate table",
        "",
        (
            "| candidate_id | depth | proposal_source | pass_rate "
            "| scoreable | pass | gen_err | score_err |"
        ),
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for attempt in sorted(
        result.attempts, key=lambda item: (item.depth, item.candidate_id)
    ):
        rate = "" if attempt.pass_rate is None else f"{attempt.pass_rate:.3f}"
        row = (
            f"| {attempt.candidate_id} | {attempt.depth} "
            f"| {attempt.proposal_source} | {rate} "
            f"| {attempt.scoreable_count} | {attempt.pass_count} "
            f"| {attempt.generation_error_count} "
            f"| {attempt.score_error_count} |"
        )
        lines.append(row)
    lines.extend(["", "## Best candidate", ""])
    if result.best_candidate is None or result.best_attempt is None:
        lines.append("No best candidate selected.")
    else:
        rate = (
            "n/a"
            if result.best_attempt.pass_rate is None
            else f"{result.best_attempt.pass_rate:.3f}"
        )
        best = result.best_candidate
        lines.extend(
            [
                f"- candidate_id: `{best.candidate_id}`",
                f"- depth: {best.depth}",
                f"- pass_rate: {rate}",
                f"- instructions_start: {best.instructions_start!r}",
                f"- instructions_end: {best.instructions_end!r}",
            ]
        )
    if result.caveats:
        lines.extend(["", "## Caveats", ""])
        lines.extend(f"- {caveat}" for caveat in result.caveats)
    if result.command:
        lines.extend(["", "## Command", "", "```bash", result.command, "```"])
    return "\n".join(lines) + "\n"


def write_copro_artifacts(
    result: CoproRunResult,
    *,
    output_dir: Path,
    commands: Sequence[str] = (),
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = output_dir / "candidates.jsonl"
    candidates_path.write_text(
        "\n".join(
            candidate.model_dump_json() for candidate in result.candidates
        )
        + ("\n" if result.candidates else ""),
        encoding="utf-8",
    )
    attempts_path = output_dir / "attempts.csv"
    with attempts_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(CoproAttempt.model_fields.keys()),
        )
        writer.writeheader()
        for attempt in result.attempts:
            writer.writerow(attempt.model_dump(mode="json"))
    best_prompt_path = output_dir / "best_prompt.json"
    if result.best_candidate is None:
        best_payload: dict[str, Any] = {"best_candidate": None}
    else:
        best_payload = {
            "best_candidate": result.best_candidate.model_dump(mode="json"),
            "best_attempt": (
                result.best_attempt.model_dump(mode="json")
                if result.best_attempt is not None
                else None
            ),
        }
    best_prompt_path.write_text(
        json.dumps(best_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary_path = output_dir / "summary.md"
    summary_path.write_text(render_summary_markdown(result), encoding="utf-8")
    commands_path = output_dir / "commands.log"
    commands_path.write_text(
        "\n".join(commands) + ("\n" if commands else ""), encoding="utf-8"
    )
    return {
        "candidates": candidates_path,
        "attempts": attempts_path,
        "best_prompt": best_prompt_path,
        "summary": summary_path,
        "commands": commands_path,
    }


def append_testing_log_entry(
    *,
    testing_log_path: Path,
    result: CoproRunResult,
    artifact_paths: Mapping[str, Path],
    verdict: str,
) -> None:
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    best_rate = (
        "n/a"
        if result.best_attempt is None or result.best_attempt.pass_rate is None
        else f"{result.best_attempt.pass_rate:.3f}"
    )
    best_id = (
        result.best_candidate.candidate_id
        if result.best_candidate is not None
        else "none"
    )
    repeats = list(result.config.repetition_seeds)
    config_line = (
        f"- breadth/depth/repeats: {result.config.breadth}/"
        f"{result.config.depth}/{repeats}"
    )
    section = f"""
## COPRO minimal enc-dec ({timestamp})

### Command

```bash
{result.command}
```

### Config

{config_line}
- model_config: `{result.config.model_config_path}`
- split: `{result.config.split_path}`
- compression_targets: {list(result.config.compression_targets)}
- proposal_mode: {result.config.proposal_mode.value}
- execution_mode: {result.config.execution_mode.value}

### Results

- experiment_name: `{result.experiment_name}`
- candidates evaluated: {len(result.candidates)}
- best candidate: `{best_id}` pass_rate={best_rate}

### Artifacts

- candidates: `{artifact_paths["candidates"]}`
- attempts: `{artifact_paths["attempts"]}`
- best_prompt: `{artifact_paths["best_prompt"]}`
- summary: `{artifact_paths["summary"]}`

### Verdict

{verdict}
"""
    with testing_log_path.open("a", encoding="utf-8") as handle:
        if testing_log_path.stat().st_size > 0:
            handle.write("\n")
        handle.write(section.lstrip())
