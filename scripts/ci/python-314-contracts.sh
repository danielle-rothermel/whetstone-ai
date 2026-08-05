#!/usr/bin/env bash
set -euo pipefail

uv run --python 3.14 pytest -q \
  tests/optimization/test_schema_identity.py::test_candidate_ref_binds_exact_record_content_and_identity \
  tests/provider/test_driver.py::TestReplayDeterminism::test_same_recorded_outcomes_produce_identical_sequence \
  tests/execution/test_resume.py::test_pending_ordinal_zero_requires_exact_ordinal_one \
  tests/execution/test_prompt_cache.py::test_hit_preserves_original_entry_provenance_and_nulls_latency \
  tests/evaluation/test_engine.py::test_engine_persists_exact_evidence_and_reward \
  tests/optimization/miprov2/test_runtime.py::test_proposal_restart_reconstructs_exact_next_effect \
  'tests/core/effects/test_authority.py::test_success_and_failure_are_exact_immutable_and_replayed[memory]'
