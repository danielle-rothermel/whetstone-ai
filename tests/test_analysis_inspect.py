from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from dr_code.humaneval import (
    EvaluationCaseStatus,
    HumanEvalTestCaseKind,
    SubmissionOutcome,
)
from dr_graph import graph_digest
from dr_providers import EndpointKind, ProviderKind

from tests.support.platform_scoring_fixtures import dataset_snapshot_identity
from whetstone.analysis.inspect import (
    RunBundle,
    SampleIndexError,
    build_debug_metadata,
    reconstruct_prompts,
    resolve_sample_index,
    summarize_test_results,
)
from whetstone.analysis.sample_html import _code_block, render_sample_html
from whetstone.platform.spec_builder import humaneval_encdec_graph
from whetstone.records import (
    DimensionsPayload,
    ExtractedSubmissionPayload,
    GenerationRunRecord,
    GenerationRunStatus,
    GenerationRunSummaryPayload,
    GraphSnapshotPayload,
    NodeAttemptRecord,
    NodeAttemptStatus,
    NodeOutputPayload,
    PerTestResultPayload,
    PredictionSpecRecord,
    ProviderConfigRef,
    ScoreAttemptRecord,
    ScoreAttemptStatus,
    TaskInputsPayload,
    TaskSnapshotPayload,
    UsageCostPayload,
    dimensions_digest,
    fair_order_key,
    stable_generation_run_id,
    stable_node_attempt_id,
    stable_prediction_id,
)

NOW = datetime(2026, 6, 30, 12, 0, tzinfo=UTC)


def _provider(
    *,
    config_id: str,
    model: str = "test-model",
) -> ProviderConfigRef:
    return ProviderConfigRef(
        config_id=config_id,
        provider_kind=ProviderKind.OPENAI,
        endpoint_kind=EndpointKind.CHAT_COMPLETIONS,
        model=model,
        throttle_key=f"openai:chat.completions:{model}",
        parameters={"temperature": 0},
    )


def _encdec_spec(
    *,
    task_inputs: dict[str, Any] | None = None,
    task_metadata: dict[str, Any] | None = None,
) -> PredictionSpecRecord:
    graph = humaneval_encdec_graph()
    graph_id = graph_digest(graph)
    dimensions = DimensionsPayload(
        values={
            "compression_target": 0.25,
            "encoder_model": "enc",
            "decoder_model": "dec",
        }
    )
    dimensions_id = dimensions_digest(dimensions)
    encoder = _provider(config_id="encoder", model="enc")
    decoder = _provider(config_id="decoder", model="dec")
    prediction_id = stable_prediction_id(
        experiment_name="exp",
        task_id="HumanEval/0",
        graph_digest=graph_id,
        dimensions_digest=dimensions_id,
        repetition_seed=0,
        provider_kind=decoder.provider_kind.value,
        endpoint_kind=decoder.endpoint_kind.value,
        model=decoder.model,
        throttle_key=decoder.throttle_key,
    )
    return PredictionSpecRecord(
        prediction_id=prediction_id,
        experiment_name="exp",
        task_id="HumanEval/0",
        repetition_seed=0,
        graph=GraphSnapshotPayload(
            graph=graph,
            graph_digest=graph_id,
            layout="encdec",
        ),
        dimensions=dimensions,
        dimensions_digest=dimensions_id,
        task=TaskSnapshotPayload(
            task_id="HumanEval/0",
            inputs=TaskInputsPayload(
                values=task_inputs
                or {
                    "gt_code": "def add(a, b):\n    return a + b",
                    "budget": 120,
                    "instructions_start": "Describe briefly.",
                    "instructions_end": "",
                }
            ),
            metadata=task_metadata or {},
        ),
        provider_configs=(encoder, decoder),
        provider_axis=decoder,
        fair_order_seed="seed",
        fair_order_key=fair_order_key(
            experiment_seed="seed",
            prediction_id=prediction_id,
            provider=decoder.provider_kind.value,
            endpoint_kind=decoder.endpoint_kind.value,
            model=decoder.model,
            throttle_key=decoder.throttle_key,
            graph_layout="encdec",
            task_id="HumanEval/0",
            repetition_seed=0,
            config_axis=dimensions_id,
        ),
        created_at=NOW,
    )


def _encdec_bundle(
    *,
    description: str = "Adds two numbers.",
    decoder_code: str = "def add(a, b):\n    return a + b",
) -> RunBundle:
    spec = _encdec_spec()
    generation_run_id = stable_generation_run_id(
        prediction_id=spec.prediction_id,
        attempt_index=0,
    )
    generation_run = GenerationRunRecord(
        generation_run_id=generation_run_id,
        prediction_id=spec.prediction_id,
        attempt_index=0,
        status=GenerationRunStatus.SUCCESS,
        terminal_node_id="decoder",
        terminal_output_node_id="decoder",
        summary=GenerationRunSummaryPayload(
            execution_order=("encoder", "decoder"),
            terminal_node_id="decoder",
            terminal_output=decoder_code,
            terminal_submission_text=decoder_code,
        ),
        started_at=NOW,
        completed_at=NOW,
    )
    encoder = _provider(config_id="encoder", model="enc")
    decoder = _provider(config_id="decoder", model="dec")
    node_attempts = (
        NodeAttemptRecord(
            node_attempt_id=stable_node_attempt_id(
                generation_run_id=generation_run_id,
                node_id="encoder",
                attempt_index=0,
            ),
            generation_run_id=generation_run_id,
            prediction_id=spec.prediction_id,
            node_id="encoder",
            attempt_index=0,
            status=NodeAttemptStatus.SUCCESS,
            provider_config=encoder,
            output=NodeOutputPayload(values={"description": description}),
            usage_cost=UsageCostPayload(provider_cost=0.01),
            started_at=NOW,
            completed_at=NOW,
        ),
        NodeAttemptRecord(
            node_attempt_id=stable_node_attempt_id(
                generation_run_id=generation_run_id,
                node_id="decoder",
                attempt_index=0,
            ),
            generation_run_id=generation_run_id,
            prediction_id=spec.prediction_id,
            node_id="decoder",
            attempt_index=0,
            status=NodeAttemptStatus.SUCCESS,
            provider_config=decoder,
            output=NodeOutputPayload(values={"code": decoder_code}),
            usage_cost=UsageCostPayload(provider_cost=0.02),
            started_at=NOW,
            completed_at=NOW,
        ),
    )
    score_attempt = ScoreAttemptRecord(
        score_attempt_id="score-1",
        prediction_id=spec.prediction_id,
        generation_run_id=generation_run_id,
        attempt_index=0,
        scoring_profile_id="humaneval",
        scoring_profile_version="v1",
        parser_profile_id="default",
        parser_version="v1",
        dataset_name="humaneval",
        dataset_split="test",
        dataset_snapshot=dataset_snapshot_identity(),
        status=ScoreAttemptStatus.SUCCESS,
        submission_outcome=SubmissionOutcome.PASSED,
        score=1.0,
        extracted_submission=ExtractedSubmissionPayload(
            raw_submission=decoder_code,
            extracted_code=decoder_code,
            extraction_method="fenced",
            parser_profile_id="default",
            parser_version="v1",
            metadata={"compile_ok": True},
        ),
        per_test_results=(
            PerTestResultPayload(
                task_id="HumanEval/0",
                test_id="t0",
                function_name="add",
                status=EvaluationCaseStatus.PASSED,
                test_type=HumanEvalTestCaseKind.INPUT_RESULT,
            ),
            PerTestResultPayload(
                task_id="HumanEval/0",
                test_id="t1",
                function_name="add",
                status=EvaluationCaseStatus.FAILED,
                message="assert 1 == 2",
                test_type=HumanEvalTestCaseKind.INPUT_RESULT,
            ),
        ),
        started_at=NOW,
        completed_at=NOW,
    )
    return RunBundle(
        spec=spec,
        generation_run=generation_run,
        node_attempts=node_attempts,
        score_attempt=score_attempt,
        sample_index=2,
        sample_count=5,
    )


def test_reconstruct_prompts_includes_gt_code_and_description() -> None:
    bundle = _encdec_bundle(description="Adds a and b.")
    prompts, errors = reconstruct_prompts(bundle)
    assert not errors
    encoder_user = next(
        message.content
        for message in prompts["encoder"]
        if message.role.value == "user"
    )
    decoder_user = next(
        message.content
        for message in prompts["decoder"]
        if message.role.value == "user"
    )
    assert "def add(a, b)" in encoder_user
    assert "120" in encoder_user
    assert "Adds a and b." in decoder_user


def test_build_debug_metadata_includes_stable_keys_and_test_summary() -> None:
    bundle = _encdec_bundle()
    prompts, errors = reconstruct_prompts(bundle)
    metadata = build_debug_metadata(
        bundle,
        reconstructed_prompts=prompts,
        reconstruction_errors=errors,
    )
    assert metadata["spec"]["prediction_id"] == bundle.spec.prediction_id
    assert metadata["generation_run"]["generation_run_id"] == (
        bundle.generation_run.generation_run_id
    )
    assert metadata["score_attempt"]["test_summary"]["failed"] == 1
    assert metadata["score_attempt"]["test_summary"]["total"] == 2
    assert "reconstructed_prompts" in metadata


def test_code_block_wraps_python_in_highlight_pre() -> None:
    block = _code_block("def foo():\n    pass\n", language="python")
    assert '<div class="highlight">' in block
    assert "<pre>" in block
    assert 'class="k"' in block


def test_render_sample_html_contains_cards_and_metadata() -> None:
    bundle = _encdec_bundle()
    prompts, errors = reconstruct_prompts(bundle)
    metadata = build_debug_metadata(
        bundle,
        reconstructed_prompts=prompts,
        reconstruction_errors=errors,
    )
    html = render_sample_html(
        bundle,
        metadata,
        prompts,
        errors,
        json_path=__import__("pathlib").Path("/tmp/sample.json"),
    )
    assert "Run summary" in html
    assert "Encoder prompt" in html
    assert "Decoder output" in html
    assert 'id="debug-metadata"' in html
    assert bundle.spec.prediction_id in html
    assert bundle.generation_run.generation_run_id in html
    assert "ui-monospace" in html
    assert '<div class="highlight"><pre>' in html


def test_ground_truth_falls_back_to_metadata() -> None:
    gt = "def migrated_gt():\n    return 42\n"
    spec = _encdec_spec(
        task_inputs={
            "gt_code": "",
            "budget": 120,
            "instructions_start": "Describe briefly.",
            "instructions_end": "",
            "prompt": "stub prompt",
        },
        task_metadata={"ground_truth_code": gt},
    )
    bundle = _encdec_bundle()
    bundle = RunBundle(
        spec=spec,
        generation_run=bundle.generation_run,
        node_attempts=bundle.node_attempts,
        score_attempt=bundle.score_attempt,
        sample_index=bundle.sample_index,
        sample_count=bundle.sample_count,
    )
    prompts, errors = reconstruct_prompts(bundle)
    metadata = build_debug_metadata(
        bundle,
        reconstructed_prompts=prompts,
        reconstruction_errors=errors,
    )
    html = render_sample_html(
        bundle,
        metadata,
        prompts,
        errors,
        json_path=__import__("pathlib").Path("/tmp/sample.json"),
    )
    assert "migrated_gt" in html
    assert "return 42" in html


def test_resolve_sample_index_out_of_range_lists_available_count() -> None:
    from whetstone.analysis.inspect import RunIndexRow

    rows = [
        RunIndexRow(
            prediction_id="p0",
            generation_run_id="g0",
            score_attempt_id="s0",
            fair_order_key="k0",
            task_id="HumanEval/0",
            generation_status="success",
            score_status="success",
        )
    ]
    with pytest.raises(SampleIndexError, match="1 enc-dec runs available"):
        resolve_sample_index(rows, 3)


def test_summarize_test_results_counts_failures() -> None:
    bundle = _encdec_bundle()
    summary = summarize_test_results(bundle.score_attempt)
    assert summary["total"] == 2
    assert summary["failed"] == 1
    assert summary["passed"] == 1
