# Changelog

All notable changes to `whetstone-ai` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- `run_anchor_calibration` subsets the engine by the caller's task IDs and
  checks anchor evidence against the subset's task hashes; it previously
  passed task hashes to an ID-keyed lookup and could not run against any
  engine whose task IDs differ from their hashes. `run_baseline_preview` no
  longer pre-converts IDs to hashes before calling it.

### Added

- Closed-form tests for `eval/analysis`: bootstrap mean and paired-delta
  intervals, power decomposition and minimum detectable difference, and
  anchor calibration.

### Changed

- The effect-authority and tool-admission SQL backends now verify their
  owned SQLite and PostgreSQL tables through the shared
  `dr_store.relational` primitives instead of duplicating column and
  constraint introspection, the transaction observer protocol, and the
  persisted-type guards in each tree. No schema, table, version, or
  wire-format change.
- `EffectAuthoritySchemaMismatchError` now carries structured `table`,
  `aspect`, `expected`, and `actual` fields, matching
  `ToolAdmissionSchemaMismatchError`. It remains an
  `EffectAuthorityError` subclass.
- `SubprocessGraphRolloutEvalDriver` runs rollout rows on a dr-exec
  `WorkerPoolImportableJsonExecutor` instead of the bespoke fanout scheduler.
  Workers import the row entry point once at startup rather than once per row,
  wall budgets apply per row and per batch, and the driver owns its pool: close
  it (or use it as a context manager) to stop its workers promptly. Closing is
  terminal — a closed driver refuses further runs instead of respawning workers.
  The per-row budget is configurable through
  `ReferenceEvalRuntimeConfig.unit_deadline_seconds`.
- A row killed by the operation deadline is reported as `deadline`;
  `not-dispatched` is reserved for rows that never reached a worker. Both
  drivers now use that vocabulary: `GraphRolloutEvalDriver` reports a row the
  deadline stopped before submission as `not-dispatched` instead of `deadline`.
  A broken worker-pool scheduler surfaces as `RowWorkerError` rather than a
  dr-exec exception.
- `RuntimeEvalEngine` owns its driver's lifetime: it has `close()` and works as
  a context manager, forwarding to drivers that implement the new
  `ClosableEvalDriver` capability (the in-process driver has nothing to close).
  This makes the worker pool built by `ReferenceEvalRuntimeConfig.build_engine`
  releasable, which it previously was not. Engines derived through
  `for_task_ids` or `for_task_seed` share the root engine's driver and do not
  close it — only the root engine does.

### Known issues

- When a rollout row's entry point raises, dr-exec's worker discards the
  exception and reports the fixed detail "the importable JSON entry point
  raised". `SubprocessGraphRolloutEvalDriver` therefore surfaces the failing
  row's coordinates and a payload attribution, but not the exception's type,
  message, or traceback. A dr-exec fix will carry that detail; until it lands,
  reproduce the failing row on the in-process `GraphRolloutEvalDriver`, which
  propagates the original exception.
- `EvalEvidence` is at `schema_version` 4. The worker pool cannot produce
  `concurrency_halved` or `guard_timeouts`, so both fields are gone;
  `deadline_reached` is unchanged. Evidence written at version 3 is not read.

### Removed

- `whetstone.execution.fanout`, `whetstone.execution.process_worker`, and
  `whetstone.execution.process_guardian`, with their re-exports from
  `whetstone.execution`: `CallSpec`, `FanoutResult`, `FanoutStatus`,
  `PoolOutcome`, `ProcessJob`, `ProcessWorkerError`,
  `ProcessCancellationError`, `run_call_pool`, and `DEFAULT_CONCURRENCY`.
  `DEFAULT_CONCURRENCY` now belongs to `whetstone.eval.runtime_engine`, and
  `RowDispatchStatus` (in `whetstone.eval.drivers`) replaces `FanoutStatus`.
- Unused `whetstone.coordination.official` certification package (official
  evaluation records, evaluation authority, aggregation, mapping, selection,
  and store); nothing consumed it.

## 0.1.1 - 2026-08-21

### Added

- First published release of the evaluation and optimization toolkit:
  experiment contract, batched evaluation engine, COPRO-wired optimizer
  harness, and platform pipeline extras.
- `register_runtime` accepts a caller `Experiment` and proposer transport so
  non-toy consumers can replace the default toy experiment and dummy transport.
- `RunRequest` lives on `HarnessRunController` so in-process consumers can
  import it without the optional DBOS extra.
