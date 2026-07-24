"""Tests for the offline attractor-pull recompute tool (task 28 item 4).

No network, no Docker, no LLM: the tool replays the ed1m dual oracle over a
persisted rollout_outputs sidecar using the LOCAL subprocess oracle. These
tests build a tiny synthetic mutants.jsonl + a matching sidecar so they run
without the full behavioral-mutant artifact.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from whetstone.tools.recompute_attractor import (
    extract_decoder_text,
    recompute_attractor,
    write_result,
)

# A tiny mutant: f(x)=x-1 (mutant) vs f(x)=x+1 (canonical); both inputs
# discriminate. A canonical reconstruction -> attractor 1.0; the mutant -> 0.0.
_CANONICAL = "def f(x):\n    return x + 1\n"
_MUTANT = "def f(x):\n    return x - 1\n"
_MUTANT_RECORD = {
    "task_id": "Synthetic/0",
    "entry_point": "f",
    "prompt": "def f(x): ...",
    "canonical_full_source": _CANONICAL,
    "mutated_full_source": _MUTANT,
    "operator_family": "synthetic",
    "seed": 0,
    "site_description": "line 1",
    "diff_summary": "",
    "input_reprs": ["[1]", "[5]"],
    "mutant_expected": [
        {"kind": "value", "output_repr": "0"},
        {"kind": "value", "output_repr": "4"},
    ],
    "canonical_expected": [
        {"kind": "value", "output_repr": "2"},
        {"kind": "value", "output_repr": "6"},
    ],
    "distinct_input_indices": [0, 1],
}
# The mutant_id the loader derives (task::family::sSEED::nSITE).
_MUTANT_ID = "Synthetic/0::synthetic::s0::nline_1"


def _sidecar_row(reconstruction: str, *, split_role: str, repeat: int) -> dict:
    return {
        "schema": "whetstone.runner.rollout_outputs/v1",
        "env": "ed1m",
        "instance_id": _MUTANT_ID,
        "split_role": split_role,
        "candidate_id": "ed1-naive",
        "repeat": repeat,
        # The sidecar format: encoder blob + DECODER reconstruction.
        "output_text": (
            f"ENCODER:\nsome description\n\nDECODER:\n{reconstruction}"
        ),
        "score": None,
        "failure_code": "",
    }


def test_extract_decoder_text_recovers_reconstruction() -> None:
    blob = "ENCODER:\nblah blah\n\nDECODER:\ndef g():\n    return 1\n"
    assert extract_decoder_text(blob) == "def g():\n    return 1\n"
    # No decoder section / missing text -> None (never a bogus reconstruction).
    assert extract_decoder_text("ENCODER:\nonly encoder") is None
    assert extract_decoder_text(None) is None


def test_recompute_canonical_reconstruction_full_attractor(
    tmp_path: Path,
) -> None:
    mutants = tmp_path / "mutants.jsonl"
    mutants.write_text(json.dumps(_MUTANT_RECORD) + "\n")
    outputs = tmp_path / "eval__ed1m__a0.jsonl"
    # The official_best arm reconstructs the CANONICAL source -> attractor 1.0.
    lines = [
        _sidecar_row(_CANONICAL, split_role="official_best", repeat=0),
        # A different arm reconstructs the MUTANT -> attractor 0.0 (sanity that
        # arms are broken out separately).
        _sidecar_row(_MUTANT, split_role="official_naive", repeat=0),
    ]
    outputs.write_text("\n".join(json.dumps(r) for r in lines) + "\n")

    result = recompute_attractor(outputs, mutants_path=mutants)
    reported = result.reported
    assert reported is not None
    assert reported.mean_attractor_pull == pytest.approx(1.0)
    assert reported.sampled_task_count == 1
    assert reported.per_task[_MUTANT_ID] == pytest.approx(1.0)
    # The naive arm reconstructed the mutant -> attractor 0.0.
    naive = result.arms["official_naive"]
    assert naive.mean_attractor_pull == pytest.approx(0.0)


def test_recompute_averages_repeats_then_tasks(tmp_path: Path) -> None:
    mutants = tmp_path / "mutants.jsonl"
    mutants.write_text(json.dumps(_MUTANT_RECORD) + "\n")
    outputs = tmp_path / "out.jsonl"
    # Two repeats of the same task: one canonical (attractor 1.0), one mutant
    # (attractor 0.0) -> task mean 0.5 -> reported mean 0.5.
    lines = [
        _sidecar_row(_CANONICAL, split_role="official_best", repeat=0),
        _sidecar_row(_MUTANT, split_role="official_best", repeat=1),
    ]
    outputs.write_text("\n".join(json.dumps(r) for r in lines) + "\n")

    result = recompute_attractor(outputs, mutants_path=mutants)
    assert result.reported is not None
    assert result.reported.mean_attractor_pull == pytest.approx(0.5)


def test_recompute_unmatched_instance_is_reported_not_scored(
    tmp_path: Path,
) -> None:
    mutants = tmp_path / "mutants.jsonl"
    mutants.write_text(json.dumps(_MUTANT_RECORD) + "\n")
    outputs = tmp_path / "out.jsonl"
    row = _sidecar_row(_CANONICAL, split_role="official_best", repeat=0)
    row["instance_id"] = "NoSuch/9::x::s0::nsite"
    outputs.write_text(json.dumps(row) + "\n")

    result = recompute_attractor(outputs, mutants_path=mutants)
    arm = result.arms["official_best"]
    # No mutant matched -> no sampled task, unmatched id surfaced.
    assert arm.mean_attractor_pull is None
    assert arm.sampled_task_count == 0
    assert "NoSuch/9::x::s0::nsite" in arm.unmatched_task_ids


def test_write_result_persists_record(tmp_path: Path) -> None:
    mutants = tmp_path / "mutants.jsonl"
    mutants.write_text(json.dumps(_MUTANT_RECORD) + "\n")
    outputs = tmp_path / "out.jsonl"
    row = _sidecar_row(_CANONICAL, split_role="official_best", repeat=0)
    outputs.write_text(json.dumps(row) + "\n")
    result = recompute_attractor(outputs, mutants_path=mutants)
    out = write_result(result, tmp_path / "recomputed.json")
    data = json.loads(out.read_text())
    assert data["reported_role"] == "official_best"
    assert data["reported_mean_attractor_pull"] == pytest.approx(1.0)
    assert "official_best" in data["arms"]
