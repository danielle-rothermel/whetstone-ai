"""No duplicate dr-code types in the Whetstone code-eval adapters.

The adapters consume the released dr-code kernel and add only Whetstone policy
and boundary roles. They must not redefine dr-code's evaluation-kernel
contracts (TextArtifact, Code Artifact, candidates, MetricFact, Score,
compression key/artifact/resolver, Aggregation Output).
"""

from __future__ import annotations

import dr_code.eval as dr_eval
import dr_code.trace as dr_trace

from whetstone.code_eval import aggregate as agg_module
from whetstone.code_eval import (
    binary_test_pass_score,
    compressed_description_length_fact,
    compression_ratio_score,
    compression_selection,
    scoring,
    select_compression_reference,
    submission,
    submission_text_artifact,
)

from .support import generation, operator_lineage


def test_boundary_reuses_dr_code_text_artifact() -> None:
    assert submission.TextArtifact is dr_trace.TextArtifact


def test_scoring_returns_dr_code_score_and_fact_types() -> None:
    score = binary_test_pass_score(
        _passed(), evaluation_procedure_config_hash="0" * 64
    )
    assert type(score) is dr_eval.Score

    fact = compressed_description_length_fact(
        "code", lineage=operator_lineage()
    )
    assert type(fact) is dr_eval.MetricFact

    ratio_score = compression_ratio_score(
        compressed_description_length=1,
        reference=dr_eval.CompressionReferenceArtifact(content=b"abcd"),
        evaluation_procedure_config_hash="0" * 64,
    )
    assert type(ratio_score) is dr_eval.Score


def test_compression_selection_returns_generic_types() -> None:
    from pydantic import BaseModel

    class _Task(BaseModel):
        gt_code_wo_comments: str

    artifact = select_compression_reference(_Task(gt_code_wo_comments="x"))
    assert type(artifact) is dr_eval.CompressionReferenceArtifact


def test_no_module_redefines_a_dr_code_kernel_type() -> None:
    dr_code_names = set(dr_eval.__all__) | {"TextArtifact", "CodeArtifact"}
    for module in (
        submission,
        scoring,
        compression_selection,
        agg_module,
    ):
        for name in getattr(module, "__all__", []):
            # A Whetstone export may share a *concept* but must not shadow a
            # dr-code kernel type name with a new class.
            if name in dr_code_names:
                exported = getattr(module, name)
                # If the name collides, it must be the dr-code object itself
                # (re-exported), not a Whetstone redefinition.
                assert getattr(exported, "__module__", "").startswith(
                    "dr_code"
                ), f"{module.__name__}.{name} shadows a dr-code type"


def _passed():
    from whetstone.code_eval import (
        CandidateVerdict,
        evaluate_candidate_correctness,
    )

    cs = dr_eval.CodeCandidateSet.from_sources(
        ("def f():\n    return 1\n",), origin="t"
    )
    return evaluate_candidate_correctness(
        cs, runner=lambda _a: CandidateVerdict.PASSED
    )


def test_submission_generation_is_whetstone_owned() -> None:
    # Code Generation stays a Whetstone Generation; dr-code never learns it.
    gen = generation(text="x = 1\n")
    assert gen.__class__.__module__.startswith("whetstone")
    artifact = submission_text_artifact(gen)
    assert artifact.__class__.__module__.startswith("dr_code")
