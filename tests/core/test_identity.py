import pickle
from typing import Any

import pytest

from whetstone.core.identity import (
    ImmutableJsonObject,
    canonical_json_equal,
)


def test_json_fields_survive_pickle_round_trips() -> None:
    original = ImmutableJsonObject(
        {
            "nested": {"enabled": True, "depth": {"count": 2}},
            "items": [1, 2.5, "three", None, {"name": "first"}],
            "flag": False,
        }
    )

    restored = pickle.loads(pickle.dumps(original))

    assert type(restored) is ImmutableJsonObject
    assert restored == original
    assert restored.to_json() == original.to_json()
    restored_nested: Any = restored["nested"]
    restored_items: Any = restored["items"]
    assert type(restored_nested) is ImmutableJsonObject
    assert type(restored_items) is tuple
    with pytest.raises(TypeError):
        restored_nested["enabled"] = False
    with pytest.raises(AttributeError):
        restored._items = ()


def test_canonical_json_comparison_preserves_json_types() -> None:
    assert not canonical_json_equal({"value": True}, {"value": 1})
    assert not canonical_json_equal({"value": 1}, {"value": 1.0})
    assert canonical_json_equal(
        {"nested": [{"value": 1}]},
        {"nested": [{"value": 1}]},
    )
