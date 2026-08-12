# Changelog

All notable changes to Whetstone are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Add `CodeCompExperimentConfig` as the unified typed input for HumanEval
  code-compression experiments, with `CodeCompEvaluationRuntimeConfig` for
  subprocess engine reconstruction and Codex MCP cutover.
- Add shared evaluation driver helpers (`row_common`, `eval_result`) extracted
  from the retired internal row path.
- Add a Codex optimization adapter with a typed output artifact, one bounded
  MCP evaluation tool backed by the pinned official protocol SDK, and
  fail-closed macOS filesystem isolation.
- Add a guarded local PostgreSQL 17 integration runner that creates and
  removes a unique least-privilege role and disposable database per run.
- Add code-grounded repository terms and binding contracts under `.defs/`,
  with semantic validation and a GitHub Pages reference that renders both
  authoritative TOML files.
- Add one local pre-check entrypoint shared by pre-commit and pre-push hooks.
- Add isolated PostgreSQL/DBOS, process, SQLite-time, and SQLite-contention CI
  lanes alongside installed-wheel smoke and Python 3.14 contract checks.
- Expand deterministic coverage for recovery, conflict, cache-accounting,
  optimization-adapter, and real-environment behavior.

### Changed

- Cut `EvaluationEngine` over to code_comp experiments only; row execution
  flows through `run_code_comp_eval` and mode-specific row requests rather
  than the generic internal driver.
- Migrate evaluation, coordination, runner, Codex, and preview fixtures to
  `code_comp` candidates (`MUTATION_FIELD`) and canonical task hashes.
- Launch the sandbox-wrapped Codex command through pinned `dr-exec` typed
  `PROCESS_BOUNDARY_ONLY` execution with caller-owned durable run records.
- Own generic evaluation configuration, planning, aggregation, measurement,
  compression, and ED1M dataset contracts in Whetstone; use released
  `dr-code` HumanEval/trace APIs and explicit `dr-exec` executors.
- Pin the released graph, serialization, and storage foundations; publish
  optimization traces atomically and admit persisted JSON only through the
  strict decoder boundary.
- Adopt the released provider evidence identities and version affected
  Whetstone provider-result persistence schemas.
- Reorganize production and test packages around the canonical core,
  experiment, environment, provider, execution, evaluation, optimization, and
  coordination boundaries, with a hard cutover to the new imports.
- Rewrite the README and package metadata around the current system,
  repository boundaries, and complete local test entrypoints.
- Require locked CI environments, cancel superseded Depot runs, and bound every
  CI job with an explicit watchdog.
- Document the current issued-call identity concurrency limitation and the
  atomic `dr-store` multi-binding required to remove it safely.
- Preserve structured prompt identity and immutability, use canonical UTC
  partial timestamps, and apply one character-budget rounding rule.

### Fixed

- Align direct-mode rendered prompts and evidence validation with per-instance
  input-arm text rather than the arm token alone.
- Map HumanEval task IDs to canonical sampling task hashes in anchor preview
  and calibration paths.
- Restrict Codex runtime reconstruction to internal sampling, validate
  evaluation-tool inputs before launch, surface durable tool failures to the
  model, stage complete namespace packages, and retain failed PostgreSQL
  integration state.
- Make immutable JSON values safe for DBOS checkpoint serialization and keep
  DBOS tests independent of registry teardown order.
- Bind proposal transport atomically and strengthen durable recovery,
  contention, partial-log, and prompt-cache accounting behavior.

### Removed

- Remove the generic internal evaluation row driver (`internal.py`) and its
  `run_internal_eval` entrypoint; `InternalEvalResult` lives under
  `evaluation.drivers.eval_result`.
- Remove QA env registry, probes, oracle operators, `build_env_experiment`,
  and the associated env test suite.
