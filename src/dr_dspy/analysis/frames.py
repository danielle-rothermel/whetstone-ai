"""Pandas frame loading and normalization for enc-dec analysis."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd
from sqlalchemy import Float, Select, and_, cast, func, select
from sqlalchemy.engine import Engine

from dr_dspy.db import schema
from dr_dspy.humaneval.profiles import (
    HUMANEVAL_SCORING_PROFILE_ID,
    HUMANEVAL_SCORING_PROFILE_VERSION,
)
from dr_dspy.humaneval.scoring import GeneratedCodeOutcome
from dr_dspy.records import GenerationRunStatus, ScoreAttemptStatus

BASE_FRAME_COLUMNS = (
    "experiment_name",
    "prediction_id",
    "generation_run_id",
    "score_attempt_id",
    "task_id",
    "model",
    "provider_kind",
    "endpoint_kind",
    "graph_layout",
    "generation_status",
    "score_status",
    "score",
    "generated_code_outcome",
    "repetition_seed",
    "dimensions",
    "compression_target",
    "encoder_model",
    "decoder_model",
    "total_provider_cost",
    "realized_compression_ratio",
    "text_character_count",
)


def normalize_compression_target(
    dimensions: Mapping[str, Any],
) -> float | None:
    if "compression_target" in dimensions:
        value = dimensions["compression_target"]
        if value is not None:
            return float(value)
    if "budget_ratio" in dimensions:
        value = dimensions["budget_ratio"]
        if value is not None:
            return float(value)
    return None


def extract_encoder_decoder_models(
    dimensions: Mapping[str, Any],
    provider_configs: Sequence[Mapping[str, Any]] | None,
) -> tuple[str | None, str | None]:
    encoder = dimensions.get("encoder_model")
    decoder = dimensions.get("decoder_model")
    if encoder is not None and decoder is not None:
        return str(encoder), str(decoder)
    encoder_model: str | None = str(encoder) if encoder is not None else None
    decoder_model: str | None = str(decoder) if decoder is not None else None
    if provider_configs:
        for config in provider_configs:
            config_id = config.get("config_id")
            model = config.get("model")
            if model is None:
                continue
            if config_id == "encoder" and encoder_model is None:
                encoder_model = str(model)
            if config_id == "decoder" and decoder_model is None:
                decoder_model = str(model)
    return encoder_model, decoder_model


def parse_score_metrics(metrics: Mapping[str, Any] | None) -> dict[str, Any]:
    if not metrics:
        return {
            "realized_compression_ratio": None,
            "text_character_count": None,
        }
    compression = metrics.get("compression") or {}
    realized_ratio = compression.get("ratio_to_ground_truth")
    if realized_ratio is None:
        custom = metrics.get("custom") or {}
        evaluation = custom.get("evaluation") or {}
        realized_ratio = evaluation.get("best_compression_ratio")
    text = metrics.get("text") or {}
    return {
        "realized_compression_ratio": (
            float(realized_ratio) if realized_ratio is not None else None
        ),
        "text_character_count": text.get("character_count"),
    }


def is_pass_row(row: Mapping[str, Any]) -> bool:
    outcome = row.get("generated_code_outcome")
    if outcome == GeneratedCodeOutcome.PASSED.value:
        return True
    score = row.get("score")
    if score is not None and float(score) == 1.0:
        return True
    return False


def _coerce_json(value: Any) -> Any:
    if value is None or isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        return json.loads(value)
    return value


def _dimension_values(dimensions: Any) -> dict[str, Any]:
    coerced = _coerce_json(dimensions)
    if not isinstance(coerced, dict):
        return {}
    values = coerced.get("values")
    if isinstance(values, dict):
        return values
    return coerced


def _provider_config_list(provider_configs: Any) -> list[dict[str, Any]]:
    coerced = _coerce_json(provider_configs)
    if isinstance(coerced, list):
        return [item for item in coerced if isinstance(item, dict)]
    return []


def _dedupe_score_attempts(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    if "score_attempt_index" not in frame.columns:
        return frame
    sorted_frame = frame.sort_values(
        ["generation_run_id", "score_attempt_index"],
        ascending=[True, False],
    )
    return sorted_frame.drop_duplicates(
        subset=["generation_run_id"],
        keep="first",
    ).reset_index(drop=True)


def _enrich_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    dimensions_series = frame["dimensions"].apply(_coerce_json)
    provider_configs_series = frame["provider_configs"].apply(_coerce_json)
    metrics_series = frame["metrics"].apply(_coerce_json)

    compression_targets: list[float | None] = []
    encoder_models: list[str | None] = []
    decoder_models: list[str | None] = []
    realized_ratios: list[float | None] = []
    text_counts: list[int | None] = []

    for dimensions, provider_configs, metrics in zip(
        dimensions_series,
        provider_configs_series,
        metrics_series,
        strict=True,
    ):
        dim_map = _dimension_values(dimensions)
        configs = _provider_config_list(provider_configs)
        compression_targets.append(normalize_compression_target(dim_map))
        encoder, decoder = extract_encoder_decoder_models(dim_map, configs)
        encoder_models.append(encoder)
        decoder_models.append(decoder)
        metrics_dict = metrics if isinstance(metrics, dict) else None
        parsed = parse_score_metrics(metrics_dict)
        realized_ratios.append(parsed["realized_compression_ratio"])
        text_count = parsed["text_character_count"]
        text_counts.append(int(text_count) if text_count is not None else None)

    frame = frame.copy()
    frame["compression_target"] = compression_targets
    frame["encoder_model"] = encoder_models
    frame["decoder_model"] = decoder_models
    frame["realized_compression_ratio"] = realized_ratios
    frame["text_character_count"] = text_counts
    return frame


def select_encdec_analysis_rows(
    experiment_names: Sequence[str],
    *,
    scoring_profile_id: str = HUMANEVAL_SCORING_PROFILE_ID,
    scoring_profile_version: str = HUMANEVAL_SCORING_PROFILE_VERSION,
    limit: int | None = None,
) -> Select[tuple[Any, ...]]:
    provider_cost = cast(
        schema.node_attempts.c.usage_cost["provider_cost"].astext,
        Float,
    )
    cost_subquery = (
        select(
            schema.node_attempts.c.generation_run_id.label("generation_run_id"),
            func.sum(provider_cost).label("total_provider_cost"),
        )
        .group_by(schema.node_attempts.c.generation_run_id)
        .subquery("run_costs")
    )
    score_match = and_(
        schema.score_attempts.c.generation_run_id
        == schema.generation_runs.c.generation_run_id,
        schema.score_attempts.c.prediction_id
        == schema.generation_runs.c.prediction_id,
        schema.score_attempts.c.scoring_profile_id == scoring_profile_id,
        schema.score_attempts.c.scoring_profile_version
        == scoring_profile_version,
    )
    statement = (
        select(
            schema.prediction_specs.c.experiment_name,
            schema.prediction_specs.c.prediction_id,
            schema.generation_runs.c.generation_run_id,
            schema.score_attempts.c.score_attempt_id,
            schema.prediction_specs.c.task_id,
            schema.prediction_specs.c.model,
            schema.prediction_specs.c.provider_kind,
            schema.prediction_specs.c.endpoint_kind,
            schema.prediction_specs.c.graph_layout,
            schema.generation_runs.c.status.label("generation_status"),
            schema.score_attempts.c.status.label("score_status"),
            schema.score_attempts.c.score,
            schema.score_attempts.c.generated_code_outcome,
            schema.prediction_specs.c.repetition_seed,
            schema.prediction_specs.c.dimensions,
            schema.prediction_specs.c.provider_configs,
            schema.score_attempts.c.metrics,
            schema.score_attempts.c.attempt_index.label("score_attempt_index"),
            cost_subquery.c.total_provider_cost,
        )
        .select_from(
            schema.prediction_specs.join(
                schema.generation_runs,
                schema.generation_runs.c.prediction_id
                == schema.prediction_specs.c.prediction_id,
            )
            .outerjoin(schema.score_attempts, score_match)
            .outerjoin(
                cost_subquery,
                cost_subquery.c.generation_run_id
                == schema.generation_runs.c.generation_run_id,
            )
        )
        .where(schema.prediction_specs.c.experiment_name.in_(experiment_names))
        .where(schema.prediction_specs.c.graph_layout == "encdec")
        .order_by(
            schema.prediction_specs.c.fair_order_key,
            schema.prediction_specs.c.prediction_id,
            schema.generation_runs.c.generation_run_id,
            schema.score_attempts.c.attempt_index,
        )
    )
    if limit is not None:
        statement = statement.limit(limit)
    return statement


def load_encdec_analysis_frame(
    engine: Engine,
    experiment_names: Sequence[str],
    *,
    limit: int | None = None,
    scoring_profile_id: str = HUMANEVAL_SCORING_PROFILE_ID,
    scoring_profile_version: str = HUMANEVAL_SCORING_PROFILE_VERSION,
) -> pd.DataFrame:
    statement = select_encdec_analysis_rows(
        experiment_names,
        scoring_profile_id=scoring_profile_id,
        scoring_profile_version=scoring_profile_version,
        limit=limit,
    )
    with engine.connect() as connection:
        frame = pd.read_sql(statement, connection)
    frame = _dedupe_score_attempts(frame)
    frame = _enrich_frame(frame)
    return frame.reindex(columns=[*BASE_FRAME_COLUMNS], fill_value=None)


def generation_success_mask(frame: pd.DataFrame) -> pd.Series:
    return frame["generation_status"] == GenerationRunStatus.SUCCESS.value


def score_success_mask(frame: pd.DataFrame) -> pd.Series:
    return frame["score_status"] == ScoreAttemptStatus.SUCCESS.value


def pass_mask(frame: pd.DataFrame) -> pd.Series:
    return frame.apply(is_pass_row, axis=1)
