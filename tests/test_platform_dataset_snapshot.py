from __future__ import annotations

import inspect
from pathlib import Path

import pytest
from dr_code.humaneval.sampling import write_human_eval_snapshot_rows
from pydantic import ValidationError

from whetstone.platform.dataset_snapshot import (
    HumanEvalSnapshot,
    load_humaneval_snapshot,
)
from whetstone.platform.spec_builder import (
    iter_experiment_specs,
    load_experiment_spec_config,
)

DATASET_NAME = "local/fixture"
DATASET_SPLIT = "test"
CONFIG_PATH = (
    Path(__file__).parent
    / "fixtures"
    / "experiment_configs"
    / "direct_minimal.json"
)


def _row(offset: int) -> dict[str, str]:
    return {
        "task_id": f"HumanEval/{offset}",
        "prompt": f"def add_{offset}(x):\n",
        "canonical_solution": f"    return x + {offset}\n",
        "entry_point": f"add_{offset}",
        "test": (
            "def check(candidate):\n"
            "    inputs = [(1,)]\n"
            f"    results = [{1 + offset}]\n"
            "    for inp, expected in zip(inputs, results):\n"
            "        assertion(candidate(*inp), expected)\n"
        ),
    }


def _write_snapshot(path: Path, *, offset: int = 1) -> Path:
    return write_human_eval_snapshot_rows(
        [_row(offset)],
        snapshot_path=path,
        dataset_name=DATASET_NAME,
    )


def test_same_snapshot_bytes_have_one_identity_across_paths(
    tmp_path: Path,
) -> None:
    first_path = _write_snapshot(tmp_path / "first" / "snapshot.json")
    second_path = tmp_path / "second" / "renamed.json"
    second_path.parent.mkdir()
    second_path.write_bytes(first_path.read_bytes())

    first = load_humaneval_snapshot(
        dataset_name=DATASET_NAME,
        dataset_split=DATASET_SPLIT,
        snapshot_path=first_path,
    ).identity
    second = load_humaneval_snapshot(
        dataset_name=DATASET_NAME,
        dataset_split=DATASET_SPLIT,
        snapshot_path=second_path,
    ).identity

    assert first == second
    assert "source_path" not in first.model_dump(mode="json")
    assert str(first_path.resolve()) not in first.model_dump_json()
    assert str(second_path.resolve()) not in second.model_dump_json()


def test_different_snapshot_bytes_have_distinct_identity_for_same_axis(
    tmp_path: Path,
) -> None:
    first_path = _write_snapshot(tmp_path / "first.json", offset=1)
    second_path = _write_snapshot(tmp_path / "second.json", offset=2)

    first = load_humaneval_snapshot(
        dataset_name=DATASET_NAME,
        dataset_split=DATASET_SPLIT,
        snapshot_path=first_path,
    )
    second = load_humaneval_snapshot(
        dataset_name=DATASET_NAME,
        dataset_split=DATASET_SPLIT,
        snapshot_path=second_path,
    )

    assert first.identity != second.identity
    assert first.identity.sha256 != second.identity.sha256


def test_corrupt_snapshot_and_header_mismatch_fail_closed(
    tmp_path: Path,
) -> None:
    corrupt_path = tmp_path / "corrupt.json"
    corrupt_path.write_text("{", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_humaneval_snapshot(
            dataset_name=DATASET_NAME,
            dataset_split=DATASET_SPLIT,
            snapshot_path=corrupt_path,
        )

    mismatched_path = write_human_eval_snapshot_rows(
        [_row(1)],
        snapshot_path=tmp_path / "mismatched.json",
        dataset_name="other/dataset",
    )
    with pytest.raises(ValueError, match="dataset mismatch"):
        load_humaneval_snapshot(
            dataset_name=DATASET_NAME,
            dataset_split=DATASET_SPLIT,
            snapshot_path=mismatched_path,
        )


def test_mutation_after_registration_is_rejected(tmp_path: Path) -> None:
    snapshot_path = _write_snapshot(tmp_path / "snapshot.json", offset=1)
    registered_identity = load_humaneval_snapshot(
        dataset_name=DATASET_NAME,
        dataset_split=DATASET_SPLIT,
        snapshot_path=snapshot_path,
    ).identity
    _write_snapshot(snapshot_path, offset=2)

    with pytest.raises(ValueError, match="does not match registration"):
        load_humaneval_snapshot(
            dataset_name=DATASET_NAME,
            dataset_split=DATASET_SPLIT,
            snapshot_path=snapshot_path,
            expected_identity=registered_identity,
        )


def test_injected_snapshot_carries_verified_identity(
    tmp_path: Path,
) -> None:
    snapshot_path = _write_snapshot(tmp_path / "snapshot.json")
    snapshot = load_humaneval_snapshot(
        dataset_name=DATASET_NAME,
        dataset_split=DATASET_SPLIT,
        snapshot_path=snapshot_path,
    )
    config = load_experiment_spec_config(CONFIG_PATH)

    assert "rows" not in inspect.signature(iter_experiment_specs).parameters

    specs = tuple(
        iter_experiment_specs(
            config,
            snapshot=snapshot,
        )
    )

    assert specs
    assert all(
        spec.task.metadata["dataset_snapshot"]
        == snapshot.identity.model_dump(mode="json")
        for spec in specs
    )


def test_injected_snapshot_rejects_another_dataset(
    tmp_path: Path,
) -> None:
    snapshot_path = write_human_eval_snapshot_rows(
        [_row(1)],
        snapshot_path=tmp_path / "snapshot.json",
        dataset_name="other/dataset",
    )
    snapshot = load_humaneval_snapshot(
        dataset_name="other/dataset",
        dataset_split=DATASET_SPLIT,
        snapshot_path=snapshot_path,
    )
    config = load_experiment_spec_config(CONFIG_PATH)

    with pytest.raises(ValueError, match="configured dataset"):
        tuple(
            iter_experiment_specs(
                config,
                snapshot=snapshot,
            )
        )


def test_same_dataset_different_content_cannot_be_detached(
    tmp_path: Path,
) -> None:
    first = load_humaneval_snapshot(
        dataset_name=DATASET_NAME,
        dataset_split=DATASET_SPLIT,
        snapshot_path=_write_snapshot(tmp_path / "first.json", offset=1),
    )
    second = load_humaneval_snapshot(
        dataset_name=DATASET_NAME,
        dataset_split=DATASET_SPLIT,
        snapshot_path=_write_snapshot(tmp_path / "second.json", offset=2),
    )

    with pytest.raises(ValueError, match="identity must match snapshot bytes"):
        HumanEvalSnapshot(
            identity=first.identity,
            rows=second.rows,
            snapshot_bytes=second.snapshot_bytes,
        )


def test_nested_row_mutation_is_rejected_at_consumption(
    tmp_path: Path,
) -> None:
    snapshot = load_humaneval_snapshot(
        dataset_name=DATASET_NAME,
        dataset_split=DATASET_SPLIT,
        snapshot_path=_write_snapshot(tmp_path / "snapshot.json"),
    )
    config = load_experiment_spec_config(CONFIG_PATH)
    snapshot.rows[0]["prompt"] = "def tampered(x):\n"

    with pytest.raises(ValueError, match="rows must match snapshot bytes"):
        tuple(iter_experiment_specs(config, snapshot=snapshot))


def test_row_field_replacement_is_rejected_at_consumption(
    tmp_path: Path,
) -> None:
    snapshot = load_humaneval_snapshot(
        dataset_name=DATASET_NAME,
        dataset_split=DATASET_SPLIT,
        snapshot_path=_write_snapshot(tmp_path / "snapshot.json"),
    )
    config = load_experiment_spec_config(CONFIG_PATH)
    replacement = dict(snapshot.rows[0])
    replacement["prompt"] = "def replaced(x):\n"
    snapshot.rows = (replacement,)

    with pytest.raises(ValueError, match="rows must match snapshot bytes"):
        tuple(iter_experiment_specs(config, snapshot=snapshot))
