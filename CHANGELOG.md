# Changelog

All notable changes to `whetstone-ai` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `EvalRole.HELD_OUT` (`"held_out"`) completes the evaluation-role
  enumeration. The role's persisted spelling is pinned by a golden literal
  test alongside the `EvalEvidence` wire format, since stored evidence
  carries the value.
- `EvalConfigs.held_out` is an optional third `EvalSplit`, derived by its own
  `derive_eval_split` call under the `HELD_OUT` split role. Two-role
  experiments leave it `None` and behave exactly as before;
  `held_out_task_hashes` is now derived from the split. `EvalConfigs` also
  gains `splits()` and `split_for()`, and rejects a split filed under the
  wrong role.
- `assert_split_disjointness(configs)` is the mechanical leakage check: it
  intersects the content-addressed task-hash sets of every present split
  pairwise, raising `HeldOutReferencedError` when held-out reaches another
  split and `SplitOverlapError` otherwise, and returns the union of task
  hashes. `EvalConfigs.__post_init__` runs it, so splits that share a task
  identity cannot be assembled into an `EvalConfigs` — and therefore into an
  `Experiment` — at all; calling it directly is now only for the covered-hash
  union or for checking splits before assembly. Enlarging the held-out split
  leaves the internal and official splits byte-identical, which a test now
  encodes.
- `EvalConfigs` construction also requires every present split to carry the
  experiment's shared `procedure_config_hash` in both the procedure it was
  derived with (`procedure_config`) and the procedure its persisted Eval
  Config records (`eval_config.evaluation_procedure_config_hash`), raising
  the new `SplitProcedureMismatchError` otherwise. The runtime executes the
  experiment's rollout graph but persists each split's own `eval_config_ref`,
  so a split carrying a foreign procedure could publish evidence claiming a
  procedure identity that was never run; `EvalSplit` does not cross-validate
  those two fields, so each is checked on its own terms.
- `BootstrapCI.p_value` carries the two-sided bootstrap p-value
  `2 * min(P(stat* <= 0), P(stat* >= 0))`, computed from the same resample
  vector as the interval and clamped into `[1 / resamples, 1]` so an
  all-one-sided bootstrap cannot report an exact zero. The p-value is
  percentile-based rather than null-centered, so it is the interval read as a
  tail area and carries no evidence beyond it; same-signed deltas always land
  the `1 / resamples` floor, which Step 10 must read as "below this
  bootstrap's resolution" and not as significance. This is documented on
  `BootstrapCI` as a known limitation of the estimator.
- `BootstrapCI.degenerate` marks an interval built from fewer than two paired
  observations (`n < 2`), where every resample is the same point and the
  bootstrap carries no information about sampling uncertainty. A degenerate
  interval reports `p_value = 1.0` rather than the `1 / resamples` floor, so
  a single task can no longer present itself as maximally significant and
  survive the Holm adjustment as a false positive. `n >= 2` resamples
  normally, including vectors with only one nonzero delta.
- `holm_adjust(pvalues)` returns Holm-Bonferroni step-down adjusted p-values
  in the input order, so a family of bootstrap p-values can be corrected
  without recomputing any bootstrap.
- `PowerConfig.significance_alpha` (default `0.05`) and
  `PowerConfig.interaction_floor_fraction` (default `0.0`), plus
  `PowerConfig.mdd_multiplier`.
- `OptimResult.cost` reports what a run actually spent, split into
  `task_model` and `proposer` roles. Each role carries `calls`,
  `input_tokens`, `output_tokens`, `priced_calls`, `unpriced_calls`,
  `cached_calls`, `rows_missing_token_breakdown`, and an optional `usd`.
  `whetstone.optim.cost` owns the wire keys and
  `COST_REPORT_SCHEMA_VERSION`; `tests/test_run_cost_report_golden.py` pins
  the exact literals.
- `calls` counts *billable* provider calls: calls the run actually paid for.
  A call the prompt cache replayed is not billable and is reported in
  `cached_calls` instead, so two optimizers with different cache-hit rates
  stay comparable -- on `calls + cached_calls` for evaluation volume, and on
  `calls` alone for spend. A billable call whose provider reported a price
  but no per-direction token split still counts and still contributes its
  price; `rows_missing_token_breakdown` records that its tokens are missing
  from the token totals.
- What counts as a billable call is decided by whether a provider answered,
  not by whether telemetry came back with the answer. An evaluation row that
  succeeded counts as a call even when the provider reported no usage and no
  price: it is counted as *unpriced*, which withholds `usd` rather than
  letting the remaining priced rows present a partial sum as a run total, and
  it is recorded in `rows_missing_token_breakdown`. A *failed* row counts on
  the same terms when its persisted `provider_error` carries a rejected
  response: the provider generated it and charged for it, and only the
  response classifier turned it down. Only a missing row, or a failed row
  whose failure body shows nothing came back at all, is excluded -- along
  with cache hits. `whetstone.execution.call_support` owns that
  discriminator (`PROVIDER_ERROR_KEY`, `REJECTED_RESPONSE_KEY`,
  `evidences_provider_response`), so both cost roles decide "did the provider
  answer?" from one signal.
- A call that was billed and then *failed* still counts, in both roles. A
  task-model row that failed but carried usage contributes its call, tokens,
  and price while staying failed for scoring; a proposer draft that reached a
  provider and came back empty does the same. A draft that made no provider
  call at all reports no usage and is recorded as nothing, so a scripted
  underfill can no longer appear as a priced zero-dollar call. Nor can a
  transport failure: it mints its logical call id before the request leaves
  and comes back with no usage, no price, and no rejected response, so a
  *failed* draft carrying none of the three is recorded as nothing rather
  than as an unpriced call that withholds the role's `usd` for spend that
  never happened. A failed draft whose response evidence *does* carry a
  rejected response is counted even without telemetry, on the same rule the
  task-model rows follow: `ProviderProposerTransport` now persists the
  failure body from `call_telemetry` alongside its response evidence, which
  is what makes a blank-but-billed proposer response legible. Conversely every
  *successful* draft is a call even without telemetry --
  `CodexCliProposerTransport` now mints a `logical_call_id` in the same shape
  as the provider transport's, so a COPRO run using the Codex CLI proposer
  reports its invocations as identified, unpriced calls with an unknown token
  breakdown instead of reporting zero proposer calls for the whole run. The
  identity covers the *invocation*, not the batch slot: one `draft` call is
  one subprocess run returning every requested body, so a breadth-two COPRO
  request reports one billed Codex execution rather than two. The batch size
  participates in the identity, since a differently-sized batch is a
  different invocation.
- Retries are counted per billed attempt. When the execution policy retries a
  response-level failure, the rejected response was still generated and still
  charged, so `call_telemetry` sums tokens and price across every attempt
  that carried a response instead of reporting only the terminal generation.
  An attempt that failed at the transport carried no response and is not
  billed. Aggregation is all-or-nothing per field: `provider_cost` is
  reported only when *every* billed attempt carried a price, and a
  directional token count only when every billed attempt reported it. A
  partly priced retry therefore makes the row unpriced rather than
  publishing an understated total that reads as complete, and a retry
  missing a token breakdown makes the row a
  `rows_missing_token_breakdown` row rather than presenting one attempt's
  tokens as the call's. The proposer transport projects the same
  `call_telemetry` aggregate onto its drafts, so proposer retries are billed
  identically to task-model retries.
- A GEPA Step that dies on a reflection failure no bounded retry can fix now
  fails through a terminal Adapter Output carrying the `proposer_usage` it
  accumulated, rather than raising it away. The durable effect cache marks
  those paid calls replayed, so a resumed Step would not have recorded them
  either and the spend was lost for good. The same output also carries the
  `search_evidence` for every evaluation the Step drove before reflection
  failed: run cost reaches task-model rows only through a Step Result's
  evidence refs, so a failed Step used to drop its whole task-model spend.
- A GEPA Step that *defers* to platform row fan-out reports its spend on the
  same terms. The `CONTINUE` Adapter Output the deferral path returns now
  carries the `proposer_usage` and `search_evidence` accumulated before the
  deferral, so a platform GEPA run costs what the in-process one does. On
  resume `begin_step` clears the adapters while the durable effect cache
  marks those calls replayed, so spend the `CONTINUE` output dropped was
  never recorded at all.
- GEPA reflection usage carries absent token counts through as `None` rather
  than reading an omitted field as `0`, so a call the provider left without a
  token breakdown becomes a `rows_missing_token_breakdown` row instead of
  inventing a complete zero-token total. A reflection result carrying neither
  usage, nor a price, nor any evidence the provider answered evidences no
  billed call and is no longer recorded. A failed reflection that *does*
  evidence a provider response is counted, even with no telemetry at all:
  that covers a parser rejection, and it covers a blank or malformed
  generation that never reached the parser, which the authority reports as a
  plain failure with `rejected_by_parser=False` and whose only billing signal
  is the `rejected_response` on its persisted response evidence. GEPA asks
  `evidences_provider_response` exactly as the COPRO and MIPROv2 side does,
  so a billed blank reflection is no longer dropped -- which had both
  under-reported GEPA proposer spend and left the role's `usd` looking
  complete, because the unpriced call that would have withheld it was never
  recorded.
- `usd` is reported only when every contributing call carried a
  provider-reported price, so a partial sum is never presented as a run
  total. When any call lacks a price the field is absent and the
  `priced_calls`/`unpriced_calls` split shows what a total would have
  covered. Whetstone owns no pricing table: prices come from dr-providers'
  `CostInfo`, which is populated only when the provider returns one.
- `whetstone.optim.cost_aggregation.aggregate_run_cost` is the single owner
  of the calculation. Both the in-process harness and the platform
  run-completion path reach it through `OptimHarness.terminalize`, so a run
  reports the same spend however it ran; a platform run and an in-process
  run of the same control over the same transport produce an identical
  report, asserted in `tests/integration/test_platform_optim.py`.
  Task-model totals are re-derived from persisted evaluation evidence rather
  than from in-memory counters. Proposer totals are recorded by the
  optimizer's adapter as it drives each call and flushed onto the Step
  Result, since a proposer call has no evaluation row to live on. Both roles
  de-duplicate: an evaluation cited by more than one Step is counted once by
  its evidence ref, and a proposer call reported by more than one Step
  Result is counted once by its `call_id`.
- Task-model token usage, provider-reported price, and prompt-cache status
  are now persisted per evaluation row on `EvalOutputRow` (`prompt_tokens`,
  `completion_tokens`, `provider_cost`, `cache_hit`), which is what makes run
  spend re-derivable from the store. A cache hit replays the original call's
  tokens and price verbatim, so `cache_hit` is what keeps it from being
  billed a second time. `EVAL_OUTPUTS_SCHEMA_VERSION` and
  `EVAL_EVIDENCE_SCHEMA_VERSION` are now 5. `cache_hit` is itself the
  evidence that cached work was served, so a cache-hit row counts in
  `cached_calls` even when the original response carried no usage telemetry
  at all — a provider that omits usage should look cheap, not absent.
- `EvalFailureEvidence` gains an optional `outputs_ref`. When
  `RuntimeEvalEngine.evaluate` raises after the driver returned rows —
  scoring or persistence failing after generation — those rows were already
  paid for, so they are persisted and referenced from the failure evidence,
  and `aggregate_run_cost` reads a failure ref exactly as it reads a success
  ref. A failure inside `driver.run` itself leaves its rows unreachable (the
  driver holds them in memory until it returns), so the ref is absent and
  that spend stays unrecoverable. Persisting the partial rows is best effort
  and never replaces the original exception.
- `ProposerCallUsage.prompt_tokens` and `completion_tokens` are now optional.
  A proposer response that omitted a directional count carries `None` rather
  than a normalized `0`, and a call missing *either* count toward
  `rows_missing_token_breakdown`, so an incomplete proposer token total no
  longer presents itself as complete. One known direction is not a
  breakdown: the absent side is carried into the totals as zero, so a call
  reporting only one direction understates its own tokens and has to be
  flagged for that understatement to be visible.
- Proposer-model usage is recorded uniformly on
  `OptimStepResult.proposer_usage` as `ProposerCallUsage`, reported by COPRO,
  GEPA, and MIPROv2 through `AdapterOutput.proposer_usage` rather than three
  different state layouts. Each entry carries a `call_id` -- GEPA's
  reflection effect request hash, the proposer's logical call id for COPRO
  and MIPROv2, which the scripted `FakeProposerTransport` now mints in the
  same shape so the toy path exercises de-duplication instead of disabling
  it -- so run cost can de-duplicate it the way it de-duplicates an
  evidence ref. GEPA records every reflection attempt it *paid for*,
  including one a bounded retry later recovered from; a reflection the
  durable effect cache replayed is not recorded, because GEPA re-drives its
  whole reflection prefix from that cache on every Step and the Step that
  first drove the call already carries its spend.
  `STEP_RESULT_SCHEMA_VERSION` is now 4.

### Changed

- Pinned dependencies moved in lockstep: dr-exec 0.1.14, dr-store 0.2.6, and
  dr-platform 0.2.7.
- The subprocess rollout driver reads `CancelledOutcome.started` to tell a row
  the batch deadline killed inside a worker from one that never left the queue.
  dr-exec now publishes that flag, so the driver no longer infers the
  distinction from the cancelled row's measured span.
- `analyze_power` reports a **two-sided** minimum detectable difference:
  `(z_{1 - significance_alpha/2} + z_{target_prob}) * sqrt((tau^2 + 2 sigma^2/K) / T)`.
  It previously used `z_{target_prob}` alone, which is a one-sided 80%
  detection threshold rather than a 95% significance MDE and understated the
  detectable effect by a factor of 3.33 at every grid point. The
  pre-registered MDE table is pinned by a numeric golden test.
- The interaction-variance estimate no longer carries an implicit
  `0.1 * within_sample_var` floor. The method-of-moments estimate is already
  truncated at zero, and the floor inflated every MDE by an amount with no
  estimator behind it; a study that wants a floor now sets
  `interaction_floor_fraction` explicitly.
- `run_anchor_calibration` calibrates anchors on any evaluation role instead
  of refusing every non-internal split. `AnchorCalibrationResult` records the
  `eval_role` and `split_role` it measured, and the optional `eval_role`
  argument asserts the engine is bound to the role the caller expects. Reward
  evidence is validated only on the internal role, which is the only role that
  mints it.

### Known limitations

- Run cost can under-report GEPA proposer spend after a crash *inside* a Step.
  Counting a replayed reflection once relies on the original call's Step
  Result existing. If a worker dies after `record_proposal_result` persists a
  paid reflection but before that Step's `OptimStepResult` is stored, the
  resumed Step loads the reflection from the durable effect cache, marks it
  replayed, and suppresses its usage — while no Step Result carries the
  original. A run that completes its Steps normally is unaffected.
- `calls` counts persisted rows, not provider responses. When the execution
  policy retries a response-level failure, every billed attempt aggregates
  into the one row's telemetry, so two billed responses report as one call
  whose tokens and price cover both. The spend is complete; the call count
  and the `priced_calls`/`unpriced_calls` split describe rows rather than
  attempts. This is by design — the row is the unit of durable evidence
  `aggregate_run_cost` reads.
- Task-model spend from tool-mediated evaluations is not aggregated here.
  `EngineToolEvaluator` stores its `EvalEvidence` refs under
  `OptimStepResult.tool_evidence`, which `aggregate_run_cost` does not
  traverse; that wiring lands with the Codex tool work on branch
  `08-22-codex`.

## 0.1.5 - 2026-08-22

### Added

- GEPA is platform-wired: `submit_optim_run` runs a GEPA adapter inline or
  with PLATFORM deferral. Search evals raise `EvalPlatformDeferred` and
  fan-in resumes the same `step_index` so every completed step still
  carries resolvable `search_evidence`.
- `build_gepa_harness_adapter` is the shared production constructor used
  by the CLI and tests. `whetstone-optim run --adapter gepa` reconstructs
  the adapter from a stored launch and honors `--proposer` (provider or
  fake).
- Continuation pools re-supply both the GEPA checkpoint and accumulated
  skipped mutations from the last completed `prior.state_ref`.
- GEPA search-eval candidates must round-trip the run's canonical
  assembler: the run base candidate plus the control `component_names`.
  An intent that does not is rejected with "not assembled from the run
  base and control component_names", and a GEPA step without the run
  seed candidate is rejected with "GEPA step must carry the run seed
  candidate". Known limitation: in PLATFORM mode the eval authority
  persists the `OptimEvalRequest` and binds its intent key before
  deferring, so a rejected candidate spends no budget and executes no
  row but leaves an orphan intent record in the store.

### Changed

- Per-intent `task_hashes` on `OptimEvalRequest` scopes GEPA minibatch
  fan-out through the same engine-narrowing path MIPROv2 uses.
  `load_terminal_optim_result` accepts a seed-retained result with no
  proposals.
- GEPA fan-in is safe to retry, guarded by the head's monotone
  `platform_stage_index` watermark rather than `step_index` or pending
  deferral fields alone. Because GEPA resumes the same `step_index` after
  every deferral, one step can own several episodes, and the pending
  deferral fields that identify an episode are cleared as soon as that
  episode's own fan-in resumes the head. `platform_stage_index` survives
  that clearing and only ever increases, so it orders a replayed fan-in
  against the head in every case. A retry of the fan-in the head last
  resumed — the head standing at exactly that fan-in's resume stage —
  stays idempotent and re-reports the same successor. Any replay the head
  has moved beyond, whether a newer episode has merely resumed or has
  since completed the step, is inert: it leaves the live head, that
  head's pending fan-out and step-keyed deferred intents, and its
  `platform_stage_index` untouched, performs no persist, and enqueues no
  `optim_step` that would duplicate the successor already queued.
- `whetstone-optim run` refuses a launch it cannot honestly evaluate.
  Both the `--adapter gepa` and `--adapter copro` paths rebuild their
  evaluation engine from `ReferenceEvalRuntimeConfig`, whose experiment is
  the built-in toy experiment, and a launch persists only eval-config
  identity hashes -- not the live `rollout_graph` needed to rebuild any
  other experiment. A launch whose bound eval config (GEPA
  `control.metric`, COPRO `control.eval_config_ref`) does not address the
  rebuilt engine's now raises `ToyExperimentOnlyError` naming both refs and
  the limitation, instead of fanning the run out over toy tasks. Known
  limitation: the command runs only launches bound against the toy
  experiment; drive any other experiment through a runtime built with it.

### Fixed

- `build_gepa_harness_adapter` partitions the GEPA data registry into the
  control's `trainset_task_hashes` and `valset_task_hashes` instead of
  passing the registry's train/val union as the trainset with no valset.
  One eval engine serves both splits, so the registry holds their ordered
  union; handing that union to upstream GEPA reflected on validation
  instances and let Pareto selection score training instances, which
  contradicted the split `run_gepa_engine` enforces and corrupted
  selection. The seam now rejects a registry whose splits overlap or do
  not cover it, and a control that binds validation back to the trainset
  still passes upstream's `valset=None` default unchanged.

## 0.1.4 - 2026-08-22

### Added

- `prepare_miprov2_run` wires MIPROv2 through the in-process harness: a
  step-contract provider registered by adapter key, opening state under
  `MIPROV2_STATE_KEY`, and an explicit `experiment` plus `initial_state`.
  Toy callers use `register_toy_runtime(..., extra_adapters={...})` with
  `build_miprov2_adapter` / `prepare_toy_miprov2_run`. MIPROv2 is not on
  the platform pipeline.
- `Miprov2DemoMode` (`fewshot`, `zeroshot`, `ground_only`) is the persisted
  demo decision. `zeroshot_opt` is derived from it. Faithful zeroshot keeps
  control maxima at 0/0 and the demo dimension out of the study, but still
  bootstraps 3/0 demos to ground instruction proposals and then discards
  them (DSPy's zero-shot path). `ground_only` is the Whetstone extension:
  it bootstraps fewshot-sized pools to ground proposals, never attaches a
  demo set to a candidate, and marks the study transcript as a deviation.
  Both non-searching modes share the zeroshot auto-mode trial/instruct
  arm (`num_instruct_candidates = n`, one search variable per component).

### Changed

- MIPROv2 control schema version 6 → 7; GEPA control schema version 1 → 2.
  `num_threads` is removed from both controls (concurrency belongs to the
  eval engine).
- MIPROv2 evaluations go through harness intents and land on
  `resolved_intents`. `search_evidence` stays empty: that field is for
  in-search evals the run never proposes (GEPA).
- MIPROv2 study replay is pinned against a live in-process Optuna 4.8.0
  TPE sampler through 25 fewshot trials and 5 `add_trial` promotions,
  past the random-startup window. Auto-mode instruct/trial counts are
  pinned for fewshot vs the shared zeroshot/ground_only arm.
- MIPROv2 eval bindings reject an engine whose task-model hash differs
  from the request. `prepare_miprov2_run` rejects a mutation-field
  override that disagrees with the control. Completing steps bind the
  run's COMPLETE cardinality via `accepted_count_for`.

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

### Removed

- `whetstone.core.effects`. The package was a rename-level duplicate of the
  generic lease design dr-store now ships as `dr_store.lease.LeaseAuthority`:
  its authority, models, storage layer, and memory, SQLite, and PostgreSQL
  backends are deleted in favor of dr-store's.

### Changed

- `register_runtime` is gone. Callers pass an explicit adapter registry
  into `build_runtime`. Adding or removing an adapter changes
  controller identity.
- `prepare_copro_run` / `prepare_gepa_run` require `experiment`,
  `render_contract`, and `mutation_field`.
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
