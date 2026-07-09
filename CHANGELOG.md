# Changelog

## 2026-07-08 - Phase 2 Lockstep Scoring Schema And Submission Vocabulary

- Pinned `dr-code` to the Phase 2 contract SHA and moved production scoring
  imports to the curated `dr_code.humaneval` root.
- Split completed evaluations from harness failures: `score_attempts` stores
  completed scores with `submission_outcome`; `score_harness_failures` stores
  harness trouble with `failure_class` and cause detail.
- Renamed scoped scoring vocabulary from generation/code outcome to submission
  outcome and added `terminal_submission_text` at the generation persistence
  boundary.
- Collapsed the Alembic chain to a pristine `20260708_0001` initial schema and
  added `scripts/db/archive_current_v1_tables.py` for one-time archival of the
  prior v1 tables.
- Removed the golden fixture mechanism.

## 2026-07-08 - Phase 2 Cutover And Docs Sweep

- Removed the `docs/composable/` migration narrative, stale DBOS flow export,
  v0 cleanup checklist, backfill inspection guide, testing run history, and
  completed/deferred planning docs from the active documentation set.
- Rewrote README, AGENTS, TESTING, and limit-reference docs as current-state
  operator material with no legacy/golden-gate/composable/v0 guidance.
- Added the required ecosystem block: whetstone-ai orchestrates DBOS workflows,
  Postgres persistence, and DSPy optimization; it works alongside
  dr-serialize, dr-providers, dr-graph, dr-platform, dr-code, and unitbench;
  it consumes dr-code's evaluator library while unitbench reads whetstone
  Postgres.
- Removed stale dependency comments from `pyproject.toml`.

## 2026-07-04 - Golden Identity Fixtures For The Composable Migration

- Added `tests/fixtures/golden/` and `tests/test_golden_fixtures.py` with
  canonical JSON strings, graph digests, record ID axes, and parser/scoring
  outputs.

## 2026-06-30 - Enc-Dec Smoke And Scoring Payload Sizing

- Captured live smoke run notes for model routing, rescoring await behavior,
  URL normalization, and score-row payload sizing.
- Confirmed the score payload cap was sufficient for HumanEval per-test result
  storage.

## 2026-06-30 - CI And Coverage Gates

- Added lint, unit, and integration CI scripts and the standalone repository
  GitHub workflow.
- Removed the coverage CI job and local coverage script from the default gate.
- Pinned `dspy==3.3.0b1` for serialization contract coverage.

## 2026-06-30 - Platform Integration Coverage

- Added tiered integration fixtures, DBOS workflow tests, recovery tests, and
  pipeline E2E coverage.
- Added platform worker CLI coverage for generation, scoring, submission, and
  rescoring commands.

## 2026-06-30 - v0 Runtime Archive

- Removed v0 experiment, harness, repair, reporting, and legacy LM runtime code
  from the active platform path.
- Kept the reshape and backfill bridge long enough to carry archived rows into
  the append-only record model.
