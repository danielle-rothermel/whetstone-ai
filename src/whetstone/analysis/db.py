"""Database connection helpers for analysis scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dr_code.humaneval.profiles import (
    HUMANEVAL_SCORING_PROFILE_ID,
    HUMANEVAL_SCORING_PROFILE_VERSION,
)
from dr_platform import resolve_database_url
from sqlalchemy import Engine, create_engine

from whetstone.platform.cli_env import load_env_file


@dataclass(frozen=True)
class AnalysisRunConfig:
    experiment_names: tuple[str, ...]
    database_url: str
    limit: int | None = None
    scoring_profile_id: str = HUMANEVAL_SCORING_PROFILE_ID
    scoring_profile_version: str = HUMANEVAL_SCORING_PROFILE_VERSION


def create_analysis_engine(
    database_url: str | None,
    env_file: Path | None,
) -> Engine:
    if env_file is not None:
        load_env_file(env_file)
    else:
        load_env_file()
    resolved = resolve_database_url(database_url)
    return create_engine(resolved)


def resolve_analysis_config(
    *,
    experiment_names: list[str],
    database_url: str | None,
    env_file: Path | None,
    limit: int | None,
    scoring_profile_id: str = HUMANEVAL_SCORING_PROFILE_ID,
    scoring_profile_version: str = HUMANEVAL_SCORING_PROFILE_VERSION,
) -> AnalysisRunConfig:
    if not experiment_names:
        raise ValueError("At least one --experiment-name is required")
    engine = create_analysis_engine(database_url, env_file)
    try:
        resolved_url = str(engine.url)
    finally:
        engine.dispose()
    return AnalysisRunConfig(
        experiment_names=tuple(experiment_names),
        database_url=resolved_url,
        limit=limit,
        scoring_profile_id=scoring_profile_id,
        scoring_profile_version=scoring_profile_version,
    )
