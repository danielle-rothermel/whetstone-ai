"""Legacy v0 worker resource helpers for DBOS experiment workers."""

from __future__ import annotations

import os
import resource
from enum import StrEnum

import httpx
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, StrictInt, StrictStr

from dr_dspy.harness.dbos import QueueSelection
from dr_dspy.lm.openrouter import (
    OPENROUTER_API_KEY_ENV,
    OPENROUTER_BASE_URL,
)

OPEN_FILE_LIMIT_AUTO = "auto"
DEFAULT_HTTP_POOL_MARGIN = 8
DEFAULT_HTTP_KEEPALIVE_EXPIRY_SECONDS = 30.0
DEFAULT_NON_POOL_FD_MARGIN = 256
SCORING_SUBPROCESS_FDS_PER_WORKER = 4


class OpenFileLimitMode(StrEnum):
    AUTO = "auto"


class HttpClientConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_connections: StrictInt
    max_keepalive_connections: StrictInt
    base_url: StrictStr = OPENROUTER_BASE_URL


class WorkerResourceBudget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    http_max_connections: StrictInt
    db_pool_max_size: StrictInt
    scoring_subprocess_fds: StrictInt
    margin: StrictInt

    @property
    def estimated_open_files(self) -> int:
        return (
            self.http_max_connections
            + self.db_pool_max_size
            + self.scoring_subprocess_fds
            + self.margin
        )


_OPENROUTER_CLIENT: OpenAI | None = None
_HTTP_CLIENT: httpx.Client | None = None
_HTTP_CLIENT_CONFIG: HttpClientConfig | None = None


def http_pool_size(
    *,
    queue: QueueSelection,
    generation_concurrency: int,
    margin: int = DEFAULT_HTTP_POOL_MARGIN,
) -> int:
    if queue is QueueSelection.SCORING:
        return margin
    return generation_concurrency + margin


def scoring_subprocess_fd_budget(
    *, queue: QueueSelection, scoring_concurrency: int
) -> int:
    if queue is QueueSelection.GENERATION:
        return 0
    return scoring_concurrency * SCORING_SUBPROCESS_FDS_PER_WORKER


def build_worker_resource_budget(
    *,
    queue: QueueSelection,
    generation_concurrency: int,
    scoring_concurrency: int,
    db_pool_max_size: int,
) -> WorkerResourceBudget:
    return WorkerResourceBudget(
        http_max_connections=http_pool_size(
            queue=queue, generation_concurrency=generation_concurrency
        ),
        db_pool_max_size=db_pool_max_size,
        scoring_subprocess_fds=scoring_subprocess_fd_budget(
            queue=queue, scoring_concurrency=scoring_concurrency
        ),
        margin=DEFAULT_NON_POOL_FD_MARGIN,
    )


def resolve_open_file_limit_request(
    raw_open_file_limit: str,
    *,
    budget: WorkerResourceBudget,
) -> int:
    if raw_open_file_limit == OPEN_FILE_LIMIT_AUTO:
        return budget.estimated_open_files
    requested = int(raw_open_file_limit)
    if requested < 1:
        raise ValueError("--open-file-limit must be positive or 'auto'")
    return requested


def configure_openrouter_client(
    *,
    max_connections: int,
    max_keepalive_connections: int | None = None,
    api_key: str | None = None,
    base_url: str = OPENROUTER_BASE_URL,
) -> OpenAI:
    global _HTTP_CLIENT, _HTTP_CLIENT_CONFIG, _OPENROUTER_CLIENT

    resolved_api_key = api_key or os.getenv(OPENROUTER_API_KEY_ENV)
    if not resolved_api_key:
        raise ValueError(f"{OPENROUTER_API_KEY_ENV} is not set")

    close_openrouter_client()
    resolved_keepalive = (
        max_keepalive_connections
        if max_keepalive_connections is not None
        else max_connections
    )
    _HTTP_CLIENT_CONFIG = HttpClientConfig(
        max_connections=max_connections,
        max_keepalive_connections=resolved_keepalive,
        base_url=base_url,
    )
    _HTTP_CLIENT = httpx.Client(
        limits=httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=resolved_keepalive,
            keepalive_expiry=DEFAULT_HTTP_KEEPALIVE_EXPIRY_SECONDS,
        )
    )
    _OPENROUTER_CLIENT = OpenAI(
        api_key=resolved_api_key,
        base_url=base_url,
        http_client=_HTTP_CLIENT,
    )
    return _OPENROUTER_CLIENT


def openrouter_client() -> OpenAI | None:
    return _OPENROUTER_CLIENT


def openrouter_client_config() -> HttpClientConfig | None:
    return _HTTP_CLIENT_CONFIG


def close_openrouter_client() -> None:
    global _HTTP_CLIENT, _HTTP_CLIENT_CONFIG, _OPENROUTER_CLIENT

    if _OPENROUTER_CLIENT is not None:
        _OPENROUTER_CLIENT.close()
    elif _HTTP_CLIENT is not None:
        _HTTP_CLIENT.close()
    _OPENROUTER_CLIENT = None
    _HTTP_CLIENT = None
    _HTTP_CLIENT_CONFIG = None


def current_open_file_count() -> int | None:
    for fd_dir in ("/proc/self/fd", "/dev/fd"):
        try:
            return len(os.listdir(fd_dir))
        except OSError:
            continue
    return None


def current_open_file_soft_limit() -> int:
    soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    return int(soft)


def resource_budget_line(budget: WorkerResourceBudget) -> str:
    return (
        f"{'FD Budget':<14} | "
        f"estimated={budget.estimated_open_files:>5} | "
        f"http={budget.http_max_connections:>4} | "
        f"db={budget.db_pool_max_size:>4} | "
        f"score_fds={budget.scoring_subprocess_fds:>4} | "
        f"margin={budget.margin:>4}"
    )


def http_client_line(config: HttpClientConfig) -> str:
    return (
        f"{'HTTP Client':<14} | "
        f"max_connections={config.max_connections:>5} | "
        f"keepalive={config.max_keepalive_connections:>5}"
    )
