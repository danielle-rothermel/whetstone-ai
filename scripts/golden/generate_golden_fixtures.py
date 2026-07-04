"""Generate golden identity fixtures for the composable migration.

Freezes whetstone's identity contract before any extraction stage:
canonical-JSON strings and digests, graph digests, record ID axes, and
parser/scoring outputs under the v1 profiles. The fixtures written to
``tests/fixtures/golden/`` are the acceptance oracle for every later
migration stage; ``tests/test_golden_fixtures.py`` asserts the current
code reproduces them byte-for-byte.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer
from dr_code.humaneval.code_parsing import (
    BEST_EFFORT_HUMANEVAL_PARSER_PROFILE,
    STRICT_FIELD_MARKER_PARSER_PROFILE,
    CodeParserProfile,
    extract_code_with_profile,
)
from dr_code.humaneval.profiles import DEFAULT_HUMANEVAL_SCORING_PROFILE
from dr_code.humaneval.scoring import score_humaneval_generation
from dr_code.humaneval.task import HumanEvalTask
from dr_serialize import canonical_json, sha256_json_digest

from whetstone.eval_failures.recording import recordable_text
from whetstone.graph import canonical_graph_payload, graph_digest
from whetstone.platform.spec_builder import (
    direct_graph,
    encdec_graph,
    humaneval_encdec_graph,
)
from whetstone.records.hashing import (
    dimensions_digest,
    fair_order_key,
    stable_generation_run_id,
    stable_node_attempt_id,
    stable_prediction_id,
    stable_score_attempt_id,
)
from whetstone.records.models import DimensionsPayload

GOLDEN_FIXTURES_DIR = (
    Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "golden"
)
HASHING_FIXTURE = "hashing.json"
GRAPH_DIGESTS_FIXTURE = "graph_digests.json"
RECORD_IDS_FIXTURE = "record_ids.json"
PARSER_SCORING_FIXTURE = "parser_scoring.json"
TRUNCATED_DIGEST_LENGTH = 16

HASHING_VALUES: dict[str, Any] = {
    "empty_dict": {},
    "empty_list": [],
    "empty_string": "",
    "null": None,
    "booleans": [True, False],
    "integers": [0, -1, 42, 2**53],
    "floats": [0.1, 2.5, -0.001, 1e300, 1.0],
    "unicode_text": "héllo wörld — 日本語 🎯",
    "escapes": 'line one\nline two\t"quoted" \\backslash',
    "unsorted_keys": {"zebra": 1, "alpha": 2, "Mango": 3, "_under": 4},
    "nested": {
        "outer": {
            "inner": [{"k": "v"}, [1, 2, [3]], None],
            "flag": False,
        },
        "sibling": "value",
    },
    "mixed_list": ["text", 7, 1.5, None, True, {"key": []}],
    "numeric_string_keys": {"10": "ten", "2": "two", "1": "one"},
}

PREDICTION_ID_AXES: dict[str, Any] = {
    "experiment_name": "golden_experiment",
    "task_id": "HumanEval/0",
    "graph_digest": "0123456789abcdef",
    "dimensions_digest": "fedcba9876543210",
    "repetition_seed": 3,
    "provider_kind": "openai",
    "endpoint_kind": "responses",
    "model": "gpt-test",
    "throttle_key": "openai:gpt-test",
}

FAIR_ORDER_AXES: dict[str, Any] = {
    "experiment_seed": "golden-seed",
    "prediction_id": "abc123def456abc123def456",
    "provider": "openai",
    "endpoint_kind": "responses",
    "model": "gpt-test",
    "throttle_key": "openai:gpt-test",
    "graph_layout": "encdec",
    "task_id": "HumanEval/0",
    "repetition_seed": 3,
    "config_axis": "temperature=0.2",
}

GENERATION_RUN_AXES: dict[str, Any] = {
    "prediction_id": "abc123def456abc123def456",
    "attempt_index": 2,
}

NODE_ATTEMPT_AXES: dict[str, Any] = {
    "generation_run_id": "run0123456789abcdef01234",
    "node_id": "decoder",
    "attempt_index": 1,
}

SCORE_ATTEMPT_AXES_DEFAULT_DATASET: dict[str, Any] = {
    "generation_run_id": "run0123456789abcdef01234",
    "scoring_profile_id": "humaneval",
    "scoring_profile_version": "v1",
    "parser_profile_id": "humaneval-best-effort",
    "parser_version": "v1",
    "attempt_index": 0,
}

SCORE_ATTEMPT_AXES_EXPLICIT_DATASET: dict[str, Any] = {
    **SCORE_ATTEMPT_AXES_DEFAULT_DATASET,
    "attempt_index": 4,
    "dataset_name": "openai/humaneval",
    "dataset_split": "validation",
}

DIMENSIONS_CASES: dict[str, dict[str, Any]] = {
    "empty": {},
    "mixed": {
        "compression_target": 0.25,
        "encoder_model": "gpt-test",
        "flags": {"strict": True},
        "seeds": [1, 2, 3],
    },
}

GOLDEN_TASK_INPUTS: dict[str, str] = {
    "task_id": "HumanEval/golden",
    "prompt": "def add_one(x):\n",
    "canonical_solution": "    return x + 1\n",
    "entry_point": "add_one",
    "test": (
        "def check(candidate):\n"
        "    inputs = [(1,), (2,)]\n"
        "    results = [2, 3]\n"
        "    for inp, expected in zip(inputs, results):\n"
        "        assertion(candidate(*inp), expected)\n"
    ),
}

PASSING_CODE = "def add_one(x):\n    return x + 1\n"
FAILING_CODE = "def add_one(x):\n    return x - 1\n"

EXTRACTION_SAMPLES: dict[str, str] = {
    "bare_python": PASSING_CODE,
    "fenced_block": (
        "Here is the solution:\n"
        "```python\n"
        "def add_one(x):\n"
        "    return x + 1\n"
        "```\n"
        "Hope this helps!"
    ),
    "field_marker": (
        "[[ ## code ## ]]\n"
        "def add_one(x):\n"
        "    return x + 1\n"
        "\n"
        "[[ ## completed ## ]]"
    ),
    "json_code_field": '{"code": "def add_one(x):\\n    return x + 1\\n"}',
    "code_repr_assignment": (
        "code = 'def add_one(x):\\n    return x + 1\\n'"
    ),
    "empty": "",
    "prose_only": "I cannot solve this task, sorry.",
    "broken_syntax": "def add_one(x:\n    return x + 1",
}

SCORING_SAMPLES: dict[str, str] = {
    "passing": PASSING_CODE,
    "tests_failed": FAILING_CODE,
    "fenced_passing": EXTRACTION_SAMPLES["fenced_block"],
    "extraction_failed": EXTRACTION_SAMPLES["prose_only"],
    "empty_generation": "",
}

PARSER_PROFILES: dict[str, CodeParserProfile] = {
    "best_effort": BEST_EFFORT_HUMANEVAL_PARSER_PROFILE,
    "field_marker": STRICT_FIELD_MARKER_PARSER_PROFILE,
}


def hashing_payload() -> dict[str, Any]:
    cases = {}
    for name, value in HASHING_VALUES.items():
        cases[name] = {
            "value": value,
            "canonical_json": canonical_json(value),
            "digest": sha256_json_digest(value),
            "truncated_digest": sha256_json_digest(
                value, length=TRUNCATED_DIGEST_LENGTH
            ),
        }
    return {"cases": cases}


def graph_digests_payload() -> dict[str, Any]:
    graphs = {
        "direct_graph": direct_graph(),
        "encdec_graph": encdec_graph(),
        "humaneval_encdec_graph": humaneval_encdec_graph(),
    }
    return {
        name: {
            "canonical_payload": canonical_json(
                canonical_graph_payload(graph)
            ),
            "digest": graph_digest(graph),
        }
        for name, graph in graphs.items()
    }


def record_ids_payload() -> dict[str, Any]:
    dimensions_cases = {
        name: {
            "values": values,
            "digest": dimensions_digest(DimensionsPayload(values=values)),
        }
        for name, values in DIMENSIONS_CASES.items()
    }
    return {
        "dimensions_digest": dimensions_cases,
        "stable_prediction_id": {
            "inputs": PREDICTION_ID_AXES,
            "expected": stable_prediction_id(**PREDICTION_ID_AXES),
        },
        "fair_order_key": {
            "inputs": FAIR_ORDER_AXES,
            "expected": fair_order_key(**FAIR_ORDER_AXES),
        },
        "stable_generation_run_id": {
            "inputs": GENERATION_RUN_AXES,
            "expected": stable_generation_run_id(**GENERATION_RUN_AXES),
        },
        "stable_node_attempt_id": {
            "inputs": NODE_ATTEMPT_AXES,
            "expected": stable_node_attempt_id(**NODE_ATTEMPT_AXES),
        },
        "stable_score_attempt_id_default_dataset": {
            "inputs": SCORE_ATTEMPT_AXES_DEFAULT_DATASET,
            "expected": stable_score_attempt_id(
                **SCORE_ATTEMPT_AXES_DEFAULT_DATASET
            ),
        },
        "stable_score_attempt_id_explicit_dataset": {
            "inputs": SCORE_ATTEMPT_AXES_EXPLICIT_DATASET,
            "expected": stable_score_attempt_id(
                **SCORE_ATTEMPT_AXES_EXPLICIT_DATASET
            ),
        },
    }


def golden_task() -> HumanEvalTask:
    return HumanEvalTask.model_validate(GOLDEN_TASK_INPUTS)


def parser_scoring_payload() -> dict[str, Any]:
    extraction_cases: dict[str, Any] = {}
    for sample_name, raw_generation in EXTRACTION_SAMPLES.items():
        per_profile = {}
        for profile_name, profile in PARSER_PROFILES.items():
            result = extract_code_with_profile(
                raw_generation, profile=profile
            )
            per_profile[profile_name] = result.model_dump(mode="json")
        extraction_cases[sample_name] = {
            "raw_generation": raw_generation,
            "profiles": per_profile,
        }

    task = golden_task()
    scoring_profile = DEFAULT_HUMANEVAL_SCORING_PROFILE
    scoring_cases = {
        sample_name: {
            "raw_generation": raw_generation,
            "score": score_humaneval_generation(
                raw_generation=raw_generation,
                task=task,
                parser_profile=scoring_profile.parser_profile,
                timeout_seconds=scoring_profile.timeout_seconds,
                recordable_text=recordable_text,
            ).model_dump(mode="json"),
        }
        for sample_name, raw_generation in SCORING_SAMPLES.items()
    }
    return {
        "task": GOLDEN_TASK_INPUTS,
        "scoring_profile": scoring_profile.model_dump(mode="json"),
        "extraction": extraction_cases,
        "scoring": scoring_cases,
    }


def golden_payloads() -> dict[str, dict[str, Any]]:
    return {
        HASHING_FIXTURE: hashing_payload(),
        GRAPH_DIGESTS_FIXTURE: graph_digests_payload(),
        RECORD_IDS_FIXTURE: record_ids_payload(),
        PARSER_SCORING_FIXTURE: parser_scoring_payload(),
    }


def main(
    output_dir: Annotated[
        Path,
        typer.Option(help="Directory to write golden fixture files into."),
    ] = GOLDEN_FIXTURES_DIR,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payloads = golden_payloads()
    for filename, payload in payloads.items():
        path = output_dir / filename
        path.write_text(json.dumps(payload, indent=2) + "\n")
        typer.echo(f"wrote {path}")
    typer.echo(f"generated {len(payloads)} fixture files: OK")


if __name__ == "__main__":
    typer.run(main)
