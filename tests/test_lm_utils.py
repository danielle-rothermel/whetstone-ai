from __future__ import annotations

import pytest

from dr_dspy.lm.utils import content_to_text, provider_cost_from_response


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
