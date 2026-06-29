"""Shared helpers for DSPy LM request/response telemetry."""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from typing import Any

from dr_dspy.eval_failures.recording import ensure_recordable
from dr_dspy.serialization import sanitize_lm_kwargs

PutEventFn = Callable[..., None]

__all__ = ["PutEventFn", "_LoggingMixin"]


class _LoggingMixin:
    """Shared lm.request/lm.response/lm.error logging for DSPy LM wrappers."""

    _log: PutEventFn

    def _log_request(
        self, req_id: str, messages: Any, kwargs: dict[str, Any]
    ) -> None:
        self._log(
            "lm.request",
            payload={
                "req_id": req_id,
                "messages": ensure_recordable(messages),
                "kwargs": sanitize_lm_kwargs(kwargs),
            },
        )

    def _log_response(self, req_id: str, resp: Any, dt: float) -> None:
        self._log(
            "lm.response",
            payload={
                "req_id": req_id,
                "dt": dt,
                "response": ensure_recordable(resp),
            },
        )

    def _log_error(self, req_id: str, exc: BaseException, dt: float) -> None:
        self._log(
            "lm.error",
            payload={"req_id": req_id, "dt": dt, "error": repr(exc)},
            error=repr(exc),
        )

    def _run_logged_forward(
        self,
        forward_fn: Callable[[], Any],
        *,
        messages: Any,
        kwargs: dict[str, Any],
    ) -> Any:
        req_id = uuid.uuid4().hex
        t0 = time.time()
        self._log_request(req_id, messages, kwargs)
        try:
            resp = forward_fn()
        except BaseException as e:
            self._log_error(req_id, e, time.time() - t0)
            raise
        self._log_response(req_id, resp, time.time() - t0)
        return resp
