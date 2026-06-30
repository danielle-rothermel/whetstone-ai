"""Legacy v0 stable ordering contract tests.

These lock the legacy join-based key format used by v0 submit shuffles and
repair ORDER BY md5(...) selection. New platform fair-order keys should use a
separate helper rather than changing this function.
"""

from __future__ import annotations

from dr_dspy.harness.ordering import ORDER_KEY_SEPARATOR, stable_order_key


def test_legacy_stable_order_key_joins_stringified_parts() -> None:
    expected = ORDER_KEY_SEPARATOR.join(("repair", "batch-1", "42"))
    assert stable_order_key("repair", "batch-1", 42) == expected


def test_legacy_stable_order_key_matches_v0_submit_seed_shape() -> None:
    seed = stable_order_key(
        "submit",
        "humaneval_direct",
        "baseline",
        "seed-1",
        "submission-abc",
    )
    assert seed == ORDER_KEY_SEPARATOR.join(
        (
            "submit",
            "humaneval_direct",
            "baseline",
            "seed-1",
            "submission-abc",
        )
    )
