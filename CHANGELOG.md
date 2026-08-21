# Changelog

All notable changes to `whetstone-ai` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.1.2 - 2026-08-21

### Added

- `build_inline_proposal_executor` builds an in-process proposal executor,
  alongside `DbosProposalExecutor`.
- `SearchEvidence` records the eval and reward refs for evaluations an
  optimizer drives inside its own search, bound to the run and step that
  drove them through `optim_run_id` and `optim_step_index`. The harness
  verifies both against the Step Request before persisting the entry on
  `OptimStepResult.search_evidence`, so the binding is harness-verified
  rather than adapter-attested.
- A terminal step may report `seed_retained=True` with no accepted
  candidates, so "the search kept the seed" is representable without a
  substitute candidate. `OptimResult` mirrors the final step. Only a step
  whose output contract sets `terminal_proposal_count` may claim it, and
  only for the run's own seed: the adapter names the retained candidate and
  the harness checks it against the new `OptimRun.initial_candidate_ref`,
  which `OptimStepResult.retained_candidate_ref` then records.
- GEPA persists the reflection responses its search rejected on every step's
  own state under `skipped_mutations`, not only on the terminal effect
  transcript, so a skip on a continuing step survives a process death.
- `OutputContract.terminal_proposal_count` lets one contract state both the
  continuing and terminal accepted-candidate cardinality, for optimizers
  whose step may terminalize on its own schedule.
- Each optimizer owns a step-contract provider, resolved by adapter key,
  that declares its first-step and continuation contracts and parses its own
  launch control. `StepRequestBuilder` and `HarnessRunController` dispatch
  through that registry.

### Changed

- `CanonicalGepaEvalAuthority` calls the `EvalEngine` identity-hash methods,
  so a `RuntimeEvalEngine` binds directly.
- The GEPA data registry keys entries by the engine-resolvable `task_id` and
  carries `task_hash` alongside; the evaluation seam needs no translation.
  Registry schema and loader projection are version 2.
- The GEPA completed-result check verifies the candidate against the outputs
  record's candidate ref instead of an absent top-level `candidate_id`.
- Every GEPA step binds one contract that permits either continuing with no
  proposals or terminalizing with the run terminal cardinality. A COMPLETE
  step must honor the run terminal cardinality rather than be the identical
  contract object.
- `OutputContract.require_distinct_bases` constrains proposed candidates
  rather than accepted ones, so a step may accept two candidates sharing a
  base. It is enforced at Step granularity; the Optimization Result needs no
  separate check.
- A rejected GEPA reflection response is retried once with the rejection fed
  back into the prompt; a second rejection skips that component's mutation
  instead of ending the run. Each rejected attempt records a
  `GepaSkippedMutation` on its own step's state and on the terminal effect
  transcript, with `exhausted=True` marking the attempts that actually
  dropped a mutation. Provider and transport failures still surface
  immediately.
- A GEPA evaluation's `OptimEvalRequest` carries the harness step index as
  `optim_step_index`, matching every other adapter. It previously carried the
  per-step effect ordinal, which `run_one_gepa_iteration` resets each step, so
  two steps replaying the same candidate on the same batch could mint
  byte-identical requests and therefore identical intent and claim keys in
  `EvalEngineService`. `GepaEffectContext` now carries `optim_step_index` and
  is schema version 2, as is `GepaEffectSlot`; `invocation_ordinal` remains
  effect-replay ordering only.
- Optimization Run, Step Request, Step Result, and Optimization Result records
  are schema version 3: `OptimRun` gains `initial_candidate_ref`, and Step
  Result gains `retained_candidate_ref`. The GEPA reflection prompt and
  upstream adapter identity are version 2. `OutputContract`'s
  `terminal_proposal_count` key and `OptimRun`'s new key both change the
  content and identity hashes of every previously stored `OptimRun` record.

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
- `SubprocessGraphRolloutEvalDriver` labels a cancelled row from dr-exec's own
  measured execution span rather than from a locally computed dispatch clock.
  The previous heuristic compared the driver's clock against a deadline armed
  later inside dr-exec's `run_batch`, so near expiry the two clocks disagreed
  and a row could be labelled either way; the span dr-exec publishes on the
  completion is a single-clock measurement that cannot disagree with itself.
- `GraphRolloutEvalDriver` collects the result of a row that finished before
  the operation deadline fired instead of overwriting it with a deadline miss.
  `Future.cancel()` reports False both for a running future and for a finished
  one, so the driver now asks `done()` first and keeps the real row.
- Both drivers reject a negative or NaN `max_wall_seconds` with `ValueError`
  at the call boundary instead of expiring the batch and persisting every row
  as a deadline miss. Positive infinity means no batch deadline, exactly as
  `None` does; `0.0` remains a valid already-elapsed wall. The rule lives in
  `whetstone.eval.drivers.row_common.validated_phase_wall_seconds`, which both
  drivers use, so they cannot drift on which walls are legal.
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
