from __future__ import annotations

import pytest

from whetstone.lm.utils import (
    content_to_text,
    provider_cost_from_response,
    sanitize_lm_kwargs,
)


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("hello", "hello"),
        ([{"text": "a"}, {"text": "b"}], "ab"),
        (["plain", {"text": "b"}], "plainb"),
        (None, None),
        (42, None),
        ({}, None),
        ([], None),
        ([{"other": "x"}], None),
        (b"bytes", None),
    ],
)
def test_content_to_text(content: object, expected: str | None) -> None:
    assert content_to_text(content) == expected


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        ({"cost": 1.25}, 1.25),
        ({"total_cost": 2.5}, 2.5),
        ({"cost": 1.0, "total_cost": 2.0}, 1.0),
        ({"usage": {"cost": 0.75}}, 0.75),
        ({"cost": "not-a-number", "usage": {"cost": 0.5}}, 0.5),
        ({}, None),
        ({"usage": {"cost": "free"}}, None),
    ],
)
def test_provider_cost_from_response(
    metadata: dict[str, object],
    expected: float | None,
) -> None:
    assert provider_cost_from_response(metadata) == expected


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (None, {}),
        ({}, {}),
        (
            {"api_key": "secret", "temperature": 0.7},
            {"api_key": "<redacted>", "temperature": 0.7},
        ),
        (
            {"API_BASE": "https://x", "max_tokens": 100},
            {"API_BASE": "<redacted>", "max_tokens": 100},
        ),
        (
            {
                "authorization": "Bearer x",
                "model_list": ["a"],
                "base_url": "https://y",
                "other": "keep",
            },
            {
                "authorization": "<redacted>",
                "model_list": "<redacted>",
                "base_url": "<redacted>",
                "other": "keep",
            },
        ),
    ],
    ids=[
        "none",
        "empty",
        "api_key_mixed",
        "case_insensitive_key",
        "all_sensitive_keys",
    ],
)
def test_sanitize_lm_kwargs(
    kwargs: dict[str, object] | None,
    expected: dict[str, object],
) -> None:
    assert sanitize_lm_kwargs(kwargs) == expected
