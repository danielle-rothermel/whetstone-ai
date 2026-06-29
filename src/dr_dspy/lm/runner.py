from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

import dspy
from dr_dspy.eval_failures.exceptions import (
    EvalFailureError,
    PredictionParseError,
)
from dr_dspy.eval_failures.generation import require_generation_text
from dr_dspy.eval_failures.recording import ensure_recordable
from dr_dspy.lm.openrouter import OPENROUTER_API_KEY_ENV, LoggingOpenRouterLM
from dr_dspy.lm.utils import (
    LmEventBuffer,
    provider_cost_from_response,
    usage_metadata_from_response,
)

DEFAULT_MAX_TRACE_SIZE = 10_000
LM_RESPONSE_PREVIEW_LIMIT = 512


class PredictorRunResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    response_metadata: dict[str, Any] = Field(default_factory=dict)
    usage_metadata: dict[str, Any] = Field(default_factory=dict)
    provider_cost: float | None = None


def build_logged_lm(
    *,
    model: str,
    reasoning: Mapping[str, Any],
    temperature: float | None,
    event_buffer: LmEventBuffer,
    max_completion_tokens: int,
    client: Any = None,
) -> dspy.BaseLM:
    if not os.environ.get(OPENROUTER_API_KEY_ENV) and client is None:
        raise ValueError(f"{OPENROUTER_API_KEY_ENV} is not set")
    return LoggingOpenRouterLM(
        model,
        log=event_buffer.put_event,
        client=client,
        cache=False,
        reasoning=dict(reasoning),
        max_completion_tokens=max_completion_tokens,
        temperature=temperature,
    )


def prediction_field_text(prediction: Any, field_name: str) -> str | None:
    value = getattr(prediction, field_name, None)
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def _predictor_failure_metadata(
    event_buffer: LmEventBuffer,
    *,
    output_field: str,
) -> dict[str, str]:
    metadata: dict[str, str] = {"output_field": output_field}
    response_preview = event_buffer.latest_response_text()
    if response_preview is not None:
        metadata["lm_response_preview"] = response_preview[
            :LM_RESPONSE_PREVIEW_LIMIT
        ]
    return metadata


def run_predictor(
    *,
    signature: type[dspy.Signature],
    input_kwargs: Mapping[str, Any],
    output_field: str,
    lm: dspy.BaseLM,
    event_buffer: LmEventBuffer,
    max_trace_size: int = DEFAULT_MAX_TRACE_SIZE,
    after_prediction: Callable[[Any], object] | None = None,
) -> str:
    with dspy.context(
        lm=lm,
        callbacks=[],
        track_usage=True,
        max_trace_size=max_trace_size,
    ):
        try:
            prediction = dspy.Predict(signature)(**dict(input_kwargs))
            if after_prediction is not None:
                after_prediction(prediction)
        except EvalFailureError:
            raise
        except Exception as error:
            raise PredictionParseError(
                f"predictor failed for output field {output_field!r}",
                underlying=error,
                metadata=_predictor_failure_metadata(
                    event_buffer,
                    output_field=output_field,
                ),
            ) from error
        text = prediction_field_text(prediction, output_field)
        return require_generation_text(text, output_field=output_field)


def predictor_run_result(
    text: str, event_buffer: LmEventBuffer
) -> PredictorRunResult:
    response_metadata = ensure_recordable(
        event_buffer.latest_response_metadata()
    )
    usage_metadata = ensure_recordable(
        usage_metadata_from_response(response_metadata)
    )
    return PredictorRunResult(
        text=text,
        response_metadata=response_metadata,
        usage_metadata=usage_metadata,
        provider_cost=provider_cost_from_response(response_metadata),
    )
