from __future__ import annotations

from typing import Any

import dr_dspy.openrouter_lm as openrouter_lm
import dspy
from dr_dspy.lm_utils import LmEventBuffer
from dr_dspy.openrouter_lm import LoggingOpenRouterLM


class FakeCompletions:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] | None = None

    def create(self, **kwargs: Any) -> dict[str, Any]:
        self.kwargs = kwargs
        return {
            "id": "cmpl-test",
            "model": kwargs["model"],
            "choices": [
                {"message": {"content": "ok"}, "finish_reason": "stop"}
            ],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 2,
                "total_tokens": 3,
            },
        }


class FakeOpenRouterClient:
    def __init__(self) -> None:
        self.completions = FakeCompletions()
        self.chat = type("FakeChat", (), {"completions": self.completions})()


def test_openrouter_lm_exposes_no_legacy_base_lm() -> None:
    assert "OpenRouterLM" not in openrouter_lm.__all__
    assert not hasattr(openrouter_lm, "OpenRouterLM")


def test_logging_openrouter_lm_uses_typed_forward_contract() -> None:
    client = FakeOpenRouterClient()
    events = LmEventBuffer()
    lm = LoggingOpenRouterLM(
        "model/test",
        log=events.put_event,
        client=client,
        cache=False,
        max_completion_tokens=12,
        reasoning={"enabled": False},
    )
    request = dspy.LMRequest.from_call(
        model="model/test",
        prompt="hello",
        max_completion_tokens=12,
        cache=False,
    )

    response = lm.forward(request)

    assert lm.forward_contract == "typed_lm"
    assert isinstance(response, dspy.LMResponse)
    assert response.text == "ok"
    assert client.completions.kwargs == {
        "model": "model/test",
        "messages": [{"role": "user", "content": "hello"}],
        "max_completion_tokens": 12,
        "extra_body": {"reasoning": {"enabled": False}},
    }
    assert events.latest_response_text() == "ok"
    assert events.latest_response_metadata()["usage"]["total_tokens"] == 3
