"""ed1m (behavioral-mutant enc-dec) tests -- no network, no Docker.

Covers the direct-JSONL loader, the per-input dual oracle (fidelity vs
attractor), the env binding (mutant enc-dec reusing the ed1 pipeline), and the
attractor-reported / fidelity-rewarded wiring. Uses the committed behavioral-
mutant artifact + the LOCAL subprocess oracle (no dr_code.mutants import).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from whetstone.envs.ed1m_dataset import (
    ED1M_ARTIFACT_DIR,
    Ed1mMutant,
    ExpectedOutcome,
    ed1m_manifest_identity,
    load_ed1m_mutants,
)

_ARTIFACT_MISSING = not (ED1M_ARTIFACT_DIR / "mutants.jsonl").exists()
_skip_no_artifact = pytest.mark.skipif(
    _ARTIFACT_MISSING, reason="behavioral-mutant artifact not present"
)


# --- loader (direct JSONL, no dr_code.mutants import) ------------------------


@_skip_no_artifact
def test_loader_reads_mutants_directly() -> None:
    mutants = load_ed1m_mutants(limit=5)
    assert len(mutants) == 5
    m = mutants[0]
    assert isinstance(m, Ed1mMutant)
    assert m.task_id and m.entry_point and m.mutated_full_source
    # Aligned dual oracle vectors.
    assert len(m.input_reprs) == len(m.mutant_expected)
    assert len(m.input_reprs) == len(m.canonical_expected)
    assert all(isinstance(e, ExpectedOutcome) for e in m.mutant_expected)
    # A stable, unique mutant id.
    assert m.mutant_id.startswith(m.task_id)


@_skip_no_artifact
def test_loader_mutant_ids_unique_and_manifest_pinned() -> None:
    mutants = load_ed1m_mutants()
    assert len(mutants) == 204  # the pinned suite
    assert len({m.mutant_id for m in mutants}) == len(mutants)
    identity = ed1m_manifest_identity()
    assert identity is not None and identity.startswith("d0e082fe")


def test_loader_does_not_import_dr_code_mutants() -> None:
    # The loader must NOT IMPORT dr_code.mutants (not in the current dr-code
    # checkout) -- build/test flip-free. Check the AST imports, not prose.
    import ast

    import whetstone.envs.ed1m_dataset as mod

    tree = ast.parse(Path(mod.__file__).read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not any(name.startswith("dr_code") for name in imported)


# --- the per-input dual oracle (fidelity vs attractor) -----------------------


@_skip_no_artifact
def test_oracle_faithful_reconstruction_high_fidelity_zero_attractor() -> None:
    from whetstone.envs.ed1m_oracle import score_ed1m_reconstruction

    # A mutant with at least one discriminating input.
    m = next(
        x for x in load_ed1m_mutants(limit=30) if x.distinct_input_count >= 1
    )
    # The MUTANT source reconstructs the mutant behavior -> fidelity 1.0 and
    # (since it never matches canonical on the distinct inputs) attractor 0.0.
    s = score_ed1m_reconstruction(
        reconstruction=m.mutated_full_source, mutant=m, timeout_seconds=5.0
    )
    assert s.fidelity_to_mutant == pytest.approx(1.0)
    assert s.attractor_pull == pytest.approx(0.0)
    assert s.infrastructure_unknown is False


@_skip_no_artifact
def test_oracle_canonical_reconstruction_full_attractor() -> None:
    from whetstone.envs.ed1m_oracle import score_ed1m_reconstruction

    m = next(
        x for x in load_ed1m_mutants(limit=30) if x.distinct_input_count >= 1
    )
    # The CANONICAL source "fixes" the bug -> attractor pull 1.0 (matches
    # canonical on every discriminating input) and fidelity < 1.0 (differs from
    # the mutant exactly on the distinct inputs).
    s = score_ed1m_reconstruction(
        reconstruction=m.canonical_full_source, mutant=m, timeout_seconds=5.0
    )
    assert s.attractor_pull == pytest.approx(1.0)
    assert s.fidelity_to_mutant is not None and s.fidelity_to_mutant < 1.0


@_skip_no_artifact
def test_oracle_unrunnable_reconstruction() -> None:
    from whetstone.envs.ed1m_oracle import score_ed1m_reconstruction

    m = load_ed1m_mutants(limit=1)[0]
    # A reconstruction defining the WRONG function name -> every input errors
    # (NameError), a definitive mismatch: fidelity 0, attractor 0, not infra.
    s = score_ed1m_reconstruction(
        reconstruction="def wrong_name():\n    return None\n",
        mutant=m,
        timeout_seconds=5.0,
    )
    assert s.fidelity_to_mutant == pytest.approx(0.0)
    assert s.infrastructure_unknown is False


def test_oracle_infrastructure_unknown_when_runner_always_fails() -> None:
    # An injected runner that ALWAYS raises -> every input is infra-failed ->
    # the row is infrastructure-unknown (fails, never scores 0).
    from whetstone.envs.ed1m_oracle import score_ed1m_reconstruction

    m = _synthetic_mutant()

    def _boom(*, source, input_json, timeout_seconds):
        raise RuntimeError("subprocess unavailable")

    s = score_ed1m_reconstruction(
        reconstruction=m.mutated_full_source,
        mutant=m,
        run_in_subprocess=_boom,
        timeout_seconds=5.0,
    )
    assert s.infrastructure_unknown is True
    assert s.fidelity_to_mutant is None
    assert s.attractor_pull is None


# --- env binding (reuses the ed1 pipeline) -----------------------------------


@_skip_no_artifact
def test_build_ed1m_experiment_no_budget_default() -> None:
    from whetstone.envs.ed1_blended import BoundedCompressionMetricConfig
    from whetstone.envs.ed1m import Ed1mExperiment, build_ed1m_experiment

    mutants = load_ed1m_mutants(limit=6)
    exp = build_ed1m_experiment(
        mutants=mutants,
        internal_n=3,
        official_n=3,
        blend_config=BoundedCompressionMetricConfig(weight=0.1),
    )
    assert isinstance(exp, Ed1mExperiment)
    assert exp.env_name == "ed1m"
    # No-budget frame is the ed1m default (task 22.4).
    assert exp.budget_ratio is None
    rd = exp.encdec_rollout
    assert rd is not None and rd.budget_rule is None
    # The mutant map is carried; dataset revision pins the suite identity.
    assert len(exp.mutants) == 6
    assert exp.dataset_revision.startswith("d0e082fe")
    assert exp.blend_config is not None


@_skip_no_artifact
def test_ed1m_eval_rewards_fidelity_reports_attractor() -> None:
    # The full ed1m eval: fidelity drives the (blended) reward; attractor
    # pull is reported separately per task, never in the reward.
    from tests.envs.support import FakeTransport, execution_policy
    from whetstone.envs.ed1 import ed1_initial_candidate
    from whetstone.envs.ed1_blended import BoundedCompressionMetricConfig
    from whetstone.envs.ed1_eval import run_ed1_eval
    from whetstone.envs.ed1m import build_ed1m_experiment
    from whetstone.optimization.mutation import MUTATION_FIELD

    mutants = tuple(
        m for m in load_ed1m_mutants(limit=40) if m.distinct_input_count >= 1
    )[:2]
    exp = build_ed1m_experiment(
        mutants=mutants,
        internal_n=2,
        official_n=2,
        blend_config=BoundedCompressionMetricConfig(weight=0.1),
    )
    by_id = {m.mutant_id: m for m in mutants}

    def reply(prompt: str) -> str:
        if prompt.startswith("Provide") or prompt.startswith("Compress"):
            for mid, m in by_id.items():
                if m.mutated_full_source[:40] in prompt:
                    return f"REBUILD:{mid}"
            return "REBUILD:x"
        for mid, m in by_id.items():
            if f"REBUILD:{mid}" in prompt:
                return m.mutated_full_source  # faithful reconstruction
        return "def x():\n    return None\n"

    ed = run_ed1_eval(
        exp,
        candidate_template=ed1_initial_candidate().payload[MUTATION_FIELD],
        candidate_id="ed1m-naive",
        sampling=exp.eval_configs.official,
        execution_policy=execution_policy(max_attempts=1),
        transport=FakeTransport(reply=reply),
        apply_reward=True,
    )
    # Fidelity is the reward-bearing per-task vector (high for faithful recon).
    assert all(s >= 0.9 for s in ed.per_task_scores)
    # Attractor pull is reported per task, separate from the reward.
    assert len(ed.per_task_attractor) == 2
    # A blended reward object was derived (fidelity blended with compression).
    assert ed.reward is not None


# --- helpers -----------------------------------------------------------------


def _synthetic_mutant() -> Ed1mMutant:
    """A tiny in-memory mutant for oracle tests that need no artifact."""
    return Ed1mMutant(
        task_id="Synthetic/0",
        entry_point="f",
        prompt="def f(x): ...",
        canonical_full_source="def f(x):\n    return x + 1\n",
        mutated_full_source="def f(x):\n    return x - 1\n",
        operator_family="synthetic",
        seed=0,
        site_description="line 1",
        diff_summary="",
        input_reprs=("[1]", "[5]"),
        mutant_expected=(
            ExpectedOutcome("value", "0"),
            ExpectedOutcome("value", "4"),
        ),
        canonical_expected=(
            ExpectedOutcome("value", "2"),
            ExpectedOutcome("value", "6"),
        ),
        distinct_input_indices=(0, 1),
    )


def test_synthetic_mutant_oracle_dual_scoring() -> None:
    # No artifact needed: score the synthetic mutant + canonical directly.
    from whetstone.envs.ed1m_oracle import score_ed1m_reconstruction

    m = _synthetic_mutant()
    s_mut = score_ed1m_reconstruction(
        reconstruction=m.mutated_full_source, mutant=m, timeout_seconds=5.0
    )
    assert s_mut.fidelity_to_mutant == pytest.approx(1.0)
    assert s_mut.attractor_pull == pytest.approx(0.0)
    s_can = score_ed1m_reconstruction(
        reconstruction=m.canonical_full_source, mutant=m, timeout_seconds=5.0
    )
    assert s_can.fidelity_to_mutant == pytest.approx(0.0)
    assert s_can.attractor_pull == pytest.approx(1.0)
