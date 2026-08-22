# Changelog

All notable changes to `whetstone-ai` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `prepare_miprov2_run` wires MIPROv2 through the in-process harness: a
  step-contract provider registered by adapter key, opening state under
  `MIPROV2_STATE_KEY`, and an explicit `experiment` plus `initial_state`.
  Toy callers use `register_toy_runtime(..., extra_adapters={...})` with
  `build_miprov2_adapter` / `prepare_toy_miprov2_run`. MIPROv2 is not on
  the platform pipeline.
- `Miprov2DemoMode` (`fewshot`, `zeroshot`, `ground_only`) is the persisted
  demo decision. `zeroshot_opt` is derived from it. `ground_only` bootstraps
  demos to ground instruction proposals, keeps the demo dimension out of the
  study, and never attaches a demo set to a candidate; the study transcript
  marks it as a Whetstone deviation.

### Removed

- `whetstone.core.effects`. The package was a rename-level duplicate of the
  generic lease design dr-store now ships as `dr_store.lease.LeaseAuthority`:
  its authority, models, storage layer, and memory, SQLite, and PostgreSQL
  backends are deleted in favor of dr-store's.

### Changed

- MIPROv2 control schema version 6 → 7; GEPA control schema version 1 → 2.
  `num_threads` is removed from both controls (concurrency belongs to the
  eval engine).
- MIPROv2 evaluations go through harness intents and land on
  `resolved_intents`. `search_evidence` stays empty: that field is for
  in-search evals the run never proposes (GEPA).
- Effect leasing runs on `dr_store.lease.LeaseAuthority`.
  `whetstone.core.leasing` is the boundary: `EffectLeaseAuthority` composes
  dr-store's authority and owns only the translation between whetstone's
  identity vocabulary and dr-store's lease vocabulary -- `TypedRef` for
  terminal result refs and whetstone's `TerminalFailure`. Replay policy,
  fencing, takeover, and terminal authority are dr-store's.
  `EffectAuthority` is now `EffectLeaseAuthority` and `AcquireResult` is now
  `EffectAcquireResult`; `EffectRequest`, `EffectLease`, `EffectTerminal`,
  `ReplayPolicy`, `TerminalOutcome`, `StaleLeaseError`, and
  `TerminalConflictError` keep their names. `ToolAdmissionAuthority` is
  unchanged and still composes the lease authority.
- The lease boundary round-trips the original whetstone `TerminalFailure`
  through a reserved JSON-safe envelope, so persist-and-compare consumers
  (`ToolCallStore.complete`, FAILED replay) stay equal after coercion.
  `LeaseAuthoritySchemaMismatchError` is re-exported next to
  `StaleLeaseError` and `TerminalConflictError`.
- Durable Tool admission ASCII-encodes the stored `EffectTerminal`, so
  persist-and-compare still holds for unpaired surrogates on SQLite and
  PostgreSQL admission.
- A transient failure while publishing a terminal through a maintenance
  handle is now retryable. The handle restarts its renewer and stays open,
  where the previous implementation marked terminalization started before
  calling the authority and so poisoned the handle permanently.
- **Deliberate dev-mode cutover:** the owned lease table is now dr-store's
  `dr_store_lease_authority` at dr-store's schema version, replacing
  `whetstone_effect_authority`. Existing whetstone effect databases are
  unreadable by this release. Recorded dev data is historical; there is no
  migration.
- Pin `dr-exec==0.1.13`. A row worker that raises now surfaces the
  exception's type and message (plus a capped traceback) in the
  `RowWorkerError` the subprocess graph-rollout driver raises, so a
  worker-side row failure no longer has to be reproduced in process to be
  named.

## 0.1.3 - 2026-08-21

### Added

- `build_runtime` assembles a `RegisteredRuntime` from explicit
  collaborators (store, engine, adapter registry, lease authority).
  Platform mode requires a ledger engine so fan-in verification cannot
  be silently off. `RegisteredRuntime.close()` forwards to the eval
  engine (and any closeable authority).
- `whetstone.testing.register_toy_runtime` holds the former toy
  defaults (`/tmp` sqlite, `DummyProposerTransport`, toy COPRO).
- `platform/deploy.py` is the shared DBOS/queue/dispatcher assembly used
  by integration tests and the CLI. `PlatformDbosConfig` is constructed
  with explicit `application_version` and `executor_id`.
- `whetstone-optim run`, `status`, and `result` submit a bound launch
  and read the run manifest / `OptimPlatformRunResult`. `run` defaults
  to a live `ProviderProposerTransport` (`--proposer provider`);
  `--proposer fake` keeps the scripted transport for tests. Controller
  identity is pinned by `--owner-id`, or derived from
  `--application-version` + `--executor-id`. The CLI always closes the
  runtime if `build_runtime` succeeded, including when `deploy_platform`
  fails. Effect leases persist on `--store-path` so a restarted CLI can
  replay a completed proposal or eval instead of charging again.

### Changed

- `register_runtime` is gone. Callers pass an explicit adapter registry
  into `build_runtime`. Adding or removing an adapter changes
  controller identity.
- `prepare_copro_run` / `prepare_gepa_run` require `experiment`,
  `render_contract`, and `mutation_field`.

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

- Closed-form tests for `eval/analysis`: bootstrap mean and paired-delta
  intervals, power decomposition and minimum detectable difference, and
  anchor calibration.

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
  `GepaSkippedMutation` on its own step's state (the terminal step's also
  appear on the terminal effect transcript), with `exhausted=True` marking the attempts that actually
  dropped a mutation. Provider and transport failures still surface
  immediately.
- A GEPA evaluation's `OptimEvalRequest` carries the harness step index as
  `optim_step_index`, matching every other adapter. It previously carried the
  per-step effect ordinal, which `run_one_gepa_iteration` resets each step, so
  two steps executing the same candidate on the same batch could mint
  byte-identical requests and therefore identical intent and claim keys in
  `EvalEngineService`. The index is stamped on when an evaluation actually
  executes, not carried in `GepaEffectContext`: the effect context and
  `GepaEffectSlot` stay step-agnostic so that a step can replay the prefix
  earlier steps already paid for, since `run_one_gepa_iteration` re-runs
  `optimize` from the seed each step. `invocation_ordinal` remains
  effect-replay ordering only.
- A GEPA evaluation effect records the harness Intent Resolution it obtained,
  so a step replaying that effect from the durable cache reconstructs the same
  `SearchEvidence` the executing step emitted. A step that crashed after
  recording its effects but before persisting its adapter checkpoint therefore
  retries with its search evidence intact rather than silently incomplete.
  Replayed entries rebind to the step reporting them through
  `SearchEvidence.from_replayed_resolution`.
- The GEPA step output contract derives its terminal cardinality from the run
  terminal contract's COMPLETE cardinality rather than its continuing
  `returned_proposal_count`, so a run that binds the two differently no longer
  rejects every honest completing step.
- The harness checks each `SearchEvidence` entry's refs against the store
  before persisting it: a COMPLETED or FAILED entry must cite an
  eval-result record of the expected schema, and its refs must resolve. A
  dangling or wrong-schema ref is now a contract violation rather than
  harness-verified evidence.
- Optimization Run, Step Request, Step Result, and Optimization Result records
  are schema version 3: `OptimRun` gains `initial_candidate_ref`, and Step
  Result gains `retained_candidate_ref`. The GEPA reflection prompt and
  upstream adapter identity are version 2. `OutputContract`'s
  `terminal_proposal_count` key and `OptimRun`'s new key both change the
  content and identity hashes of every previously stored `OptimRun` record.

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

### Fixed

- `run_anchor_calibration` subsets the engine by the caller's task IDs and
  checks anchor evidence against the subset's task hashes; it previously
  passed task hashes to an ID-keyed lookup and could not run against any
  engine whose task IDs differ from their hashes. `run_baseline_preview` no
  longer pre-converts IDs to hashes before calling it.

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
