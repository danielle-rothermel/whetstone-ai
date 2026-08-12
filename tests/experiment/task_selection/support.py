from __future__ import annotations

from whetstone.experiment.task_selection import TASK_SELECTION_SCHEMA


def manifest_payload(
    *,
    encdec_test: tuple[str, ...] = ("Synthetic/3", "Synthetic/4"),
) -> dict[str, object]:
    return {
        "schema": TASK_SELECTION_SCHEMA,
        "seed": 7,
        "pools": {
            "encdec": {
                "arm": "encdec_naive",
                "train": ["Synthetic/0", "Synthetic/1"],
                "val": ["Synthetic/2"],
                "test": list(encdec_test),
            },
            "direct": {
                "arm": "direct_original",
                "train": ["Synthetic/0"],
                "val": ["Synthetic/1"],
                "test": ["Synthetic/2", "Synthetic/3"],
            },
        },
    }
