from __future__ import annotations

from pathlib import Path

import pytest

from dr_dspy.humaneval.sampling import SampledHumanEvalTask
from dr_dspy.humaneval.task import HumanEvalTask, parse_human_eval_dataset
from dr_dspy.platform import jsonl_specs, spec_builder
from dr_dspy.platform.spec_builder import (
    HUMANEVAL_DECODER_USER_PROMPT_TEMPLATE,
    HUMANEVAL_ENCODER_USER_PROMPT_TEMPLATE,
    ExperimentSpecConfig,
    GraphLayout,
    encoder_char_budget,
    humaneval_gt_code,
    iter_experiment_specs,
    load_experiment_spec_config,
    prediction_spec,
    task_snapshot_from_humaneval,
)
from dr_dspy.records import PredictionSpecRecord

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "experiment_configs"


def _humaneval_row(task_id: str, offset: int) -> dict[str, str]:
    return {
        "task_id": task_id,
        "prompt": f"def f_{offset}(x):\n",
        "canonical_solution": f"    return x + {offset}\n",
        "entry_point": f"f_{offset}",
        "test": (
            "def check(candidate):\n"
            "    inputs = [(1,)]\n"
            f"    results = [{1 + offset}]\n"
            "    for inp, expected in zip(inputs, results):\n"
            "        assertion(candidate(*inp), expected)\n"
        ),
    }


def _fixture_rows() -> tuple[dict[str, str], ...]:
    return (
        _humaneval_row("HumanEval/0", 0),
        _humaneval_row("HumanEval/1", 1),
        _humaneval_row("HumanEval/2", 2),
    )


def test_direct_config_generates_validated_specs(tmp_path: Path) -> None:
    config = load_experiment_spec_config(FIXTURES_DIR / "direct_minimal.json")
    specs = tuple(iter_experiment_specs(config, rows=_fixture_rows()))

    assert len(specs) == 8
    assert all(
        PredictionSpecRecord.model_validate(spec.model_dump(mode="json"))
        for spec in specs
    )
    assert {spec.experiment_name for spec in specs} == {"direct-exp"}
    assert {spec.graph.layout for spec in specs} == {"direct"}


def test_encdec_config_generates_validated_specs() -> None:
    config = load_experiment_spec_config(FIXTURES_DIR / "encdec_minimal.json")
    specs = tuple(iter_experiment_specs(config, rows=_fixture_rows()))

    assert len(specs) == 1
    spec = specs[0]
    assert spec.graph.layout == GraphLayout.ENCDEC.value
    assert {provider.config_id for provider in spec.provider_configs} == {
        "encoder",
        "decoder",
    }


def test_iter_experiment_specs_is_deterministic() -> None:
    config = load_experiment_spec_config(FIXTURES_DIR / "direct_minimal.json")
    rows = _fixture_rows()

    first = tuple(
        spec.prediction_id
        for spec in iter_experiment_specs(config, rows=rows)
    )
    second = tuple(
        spec.prediction_id
        for spec in iter_experiment_specs(config, rows=rows)
    )

    assert first == second
    assert len(first) == len(set(first))


def test_task_snapshot_from_humaneval_matches_v0_inputs() -> None:
    task = parse_human_eval_dataset([_humaneval_row("HumanEval/0", 0)])[0]
    snapshot = task_snapshot_from_humaneval(task)

    assert snapshot.task_id == "HumanEval/0"
    assert snapshot.inputs.values["prompt"] == task.prompt
    assert snapshot.inputs.values["test"] == task.test
    assert snapshot.inputs.values["entry_point"] == task.entry_point
    assert snapshot.metadata["canonical_solution"] == task.canonical_solution


def test_generated_jsonl_indexes_for_submit(tmp_path: Path) -> None:
    config = load_experiment_spec_config(FIXTURES_DIR / "direct_minimal.json")
    specs = tuple(iter_experiment_specs(config, rows=_fixture_rows()))
    specs_file = tmp_path / "specs.jsonl"
    spec_builder.write_prediction_specs_jsonl(specs, specs_file)

    refs = jsonl_specs.index_jsonl_prediction_specs(
        specs_file,
        experiment_name="direct-exp",
    )

    assert len(refs) == len(specs)


def test_experiment_spec_config_rejects_invalid_layout_providers() -> None:
    with pytest.raises(ValueError, match="exactly one provider"):
        ExperimentSpecConfig.model_validate(
            {
                "experiment_name": "bad",
                "graph_layout": "direct",
                "dataset": {
                    "name": "local/fixture",
                    "split": "test",
                    "sample_count": 1,
                },
                "providers": [
                    {"model": "a", "config_id": "encoder"},
                    {"model": "b", "config_id": "decoder"},
                ],
            }
        )


def test_prediction_spec_supports_task_snapshot() -> None:
    task = HumanEvalTask.model_validate(
        {
            "task_id": "HumanEval/fixture",
            "prompt": "def add_one(x):\n",
            "canonical_solution": "    return x + 1\n",
            "entry_point": "add_one",
            "test": _humaneval_row("HumanEval/fixture", 1)["test"],
        }
    )
    snapshot = task_snapshot_from_humaneval(task)
    spec = prediction_spec(
        spec_builder.direct_graph(),
        task=snapshot,
        task_id=task.task_id,
    )

    assert spec.task.inputs.values["entry_point"] == "add_one"


def test_sample_tasks_for_config_uses_injected_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_experiment_spec_config(FIXTURES_DIR / "encdec_minimal.json")
    rows = _fixture_rows()

    def fail_load(**kwargs: object) -> list[SampledHumanEvalTask]:
        raise AssertionError("load_human_eval_tasks should not be called")

    monkeypatch.setattr(spec_builder, "sample_human_eval_tasks", fail_load)
    sampled = spec_builder.sample_tasks_for_config(config, rows=rows)

    assert len(sampled) == 1
    assert sampled[0].task.task_id == "HumanEval/1"


def test_encoder_char_budget_uses_configurable_min_floor() -> None:
    assert encoder_char_budget(
        compression_target=0.1,
        gt_code="short",
        min_budget=100,
    ) == 100
    assert encoder_char_budget(
        compression_target=0.5,
        gt_code="x" * 200,
        min_budget=50,
    ) == 100


def test_humaneval_encdec_config_builds_target_spec() -> None:
    config = load_experiment_spec_config(
        FIXTURES_DIR / "encdec_humaneval_smoke.json"
    )
    rows = _fixture_rows()
    specs = tuple(iter_experiment_specs(config, rows=rows))

    assert len(specs) == 1
    spec = specs[0]
    assert spec.graph.layout == GraphLayout.ENCDEC.value
    assert {provider.config_id for provider in spec.provider_configs} == {
        "encoder",
        "decoder",
    }
    assert spec.provider_configs[0].model == spec.provider_configs[1].model
    assert all(
        provider.parameters.get("temperature") == 0
        for provider in spec.provider_configs
    )

    inputs = spec.task.inputs.values
    assert "gt_code" in inputs
    assert "budget" in inputs
    assert "instructions_start" in inputs
    assert "instructions_end" in inputs
    assert inputs["prompt"]
    assert inputs["test"]
    assert inputs["entry_point"]

    task = parse_human_eval_dataset([rows[1]])[0]
    gt_code = humaneval_gt_code(task)
    assert config.humaneval_encdec is not None
    expected_budget = encoder_char_budget(
        compression_target=0.5,
        gt_code=gt_code,
        min_budget=config.humaneval_encdec.min_encoder_char_budget,
    )
    assert inputs["gt_code"] == gt_code
    assert inputs["budget"] == expected_budget
    assert inputs["instructions_start"] == (
        "Provide a concise description of the following code."
    )
    assert inputs["instructions_end"] == ""

    graph = spec.graph.graph
    encoder = graph.node("encoder")
    decoder = graph.node("decoder")
    assert encoder.config.input_bindings["gt_code"].ref == "task.gt_code"
    assert encoder.config.input_bindings["budget"].ref == "task.budget"
    assert (
        encoder.config.input_bindings["instructions_start"].ref
        == "task.instructions_start"
    )
    assert (
        encoder.config.input_bindings["instructions_end"].ref
        == "task.instructions_end"
    )
    assert decoder.config.input_bindings["encoded_desc"].ref == (
        "encoder.description"
    )
    assert (
        encoder.config.metadata["user_prompt_template"]
        == HUMANEVAL_ENCODER_USER_PROMPT_TEMPLATE
    )
    assert (
        decoder.config.metadata["user_prompt_template"]
        == HUMANEVAL_DECODER_USER_PROMPT_TEMPLATE
    )
    assert spec.dimensions.values["compression_target"] == 0.5
