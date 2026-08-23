# Changelog

All notable changes to `whetstone-ai` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- The per-task vector and the evaluation-level aggregate now read the same
  rows the same way. `per_task_score` scored every non-present row as 0.0 and
  divided by `num_seeds`, while the aggregate applied the plan's own
  missing-row policy, so the two disagreed off identical evidence: under a
  tolerant policy a task with three present rows at 1.0 and one failed repeat
  reported 0.75 in the vector beside 1.0 in the aggregate.

  `per_task_score` is now the mean over a task's **present** rows, obtained by
  calling the same `aggregate` entry point on the same rows the aggregate
  uses, so there is one definition of a task's score rather than two that can
  drift. A lost repeat is skipped rather than scored zero, and a task with no
  present row is `None` — *unobserved*, not *scored 0.0*.

  Two consequences of the old zero-padding are fixed with it. A fully lost
  task entered anchor calibration (`analyze_power` via `per_task_values`) as a
  measured hard task at 0.0, biasing the anchor gap and every MDD on the
  power surface; calibration now rejects an anchor carrying an unobserved
  task, which its existing full-observation requirement already implied. And
  GEPA no longer coerces an absent score to 0.0: a task with no present row is
  projected as the failed row it already had a path for.

### Changed
- `per_task_count` counts a task's **present** rows. It previously returned
  `len(completed_rows(num_seeds))`, and `completed_rows` pads missing repeats
  *in*, so it always equalled `num_seeds` — leaving any downstream
  row-completeness weighting built on `per_task_counts` inert. It now drops
  below `num_seeds` exactly when a task lost repeats, and reads zero precisely
  where the score is unobserved for want of data.
- `EvalEvidence.schema_version` is **6**: `per_task_values` widens to
  `tuple[float | None, ...]` so consumers can distinguish a task that scored
  zero from one that was never observed. `per_task_counts` keeps its type and
  changes meaning as above. `EvalOutputsRecord` is unaffected and stays at
  version 5.
- `per_task_score` takes the plan's `AggregationConfig`, since the reduction
  is now the plan's own policy rather than a second hardcoded one. Under a
  propagating policy a partially observed task withholds its score (`None`)
  while its count still reports the rows actually observed.
- Evidence validation no longer re-derives the per-task vector as
  `sum(...)/num_seeds`. It checks the claims every mean policy must satisfy: a
  reported value equals the present-row mean exactly, and a value may be
  withheld only for a task that actually lost a row.

## 0.1.12 - 2026-08-23

### Fixed
- GEPA no longer dies with `GEPA evaluation positions must be unique` when the
  trainset size is not a multiple of `reflection_minibatch_size`. Upstream's
  `EpochShuffledBatchSampler` pads each shuffled epoch up to a multiple of the
  minibatch size by repeating its least-frequent ids, so a reflection minibatch
  legitimately carries the same instance twice — and a Whetstone evaluation
  request is position-unique by contract, so the run aborted mid-flight. The
  shapes a protocol actually uses were affected: trainset 4 or 44 with
  minibatch 3 both fail, while 6/3 and 4/2 divide evenly and never did.

  The padding is the pinned algorithm, not an upstream defect, so it is
  reproduced rather than suppressed: neither the sampler nor the uniqueness
  contract changed. `WhetstoneGepaAdapter.evaluate` now evaluates the batch's
  *distinct* instances in one position-unique request and expands the returned
  rows back to the upstream batch shape, so GEPA receives the repeated
  instance's score, output, and trajectory once per occurrence. That repeat is
  load-bearing: upstream compares `sum(scores)` before and after a mutation, so
  a doubled instance must carry double weight on both sides.

  Accounting deliberately splits. Logical metric calls remain upstream's —
  it charges the padded batch length, duplicates included — so a
  `max_metric_calls` budget keeps its upstream meaning. Provider rows are
  billed once per distinct instance, since re-evaluating one instance under a
  fixed candidate is the same evaluation, making a padded run slightly cheaper
  in provider spend without changing search behaviour. The collapse and
  expansion are a pure function of the upstream batch and add no adapter state,
  so determinism, replay, and crash-retry are unaffected and no persisted
  schema changed.

## 0.1.11 - 2026-08-23

### Fixed
- MIPROv2 and GEPA can now evaluate candidates at more than one repeat per
  task. Both refused a multi-repeat evaluation plan outright — MIPROv2 with
  `engine sampling repeats (N) do not match the requested num_seeds (1)`
  because `EvalBindingRequest.num_seeds` defaulted to 1 and no construction
  site set it, and GEPA with `GEPA evaluation engine must use a single-repeat
  plan`. A protocol that pre-registers a repeat count for *every* evaluation,
  in-search ones included, could therefore not run either optimizer at all.
  COPRO was already repeat-transparent and is unchanged in behaviour.

  The repeat count is a property of the bound eval engine's split, which was
  already the authority; the optimizers now carry it rather than asserting it
  away. The score each search consumes is unchanged in kind: it is the
  existing canonical reduction — the per-task mean over repeats
  (`whetstone.eval.drivers.eval_result.per_task_score`, surfaced as
  `EvalEvidence.per_task_values`) — and no second reduction was introduced.
  MIPROv2's row budget (`task_rows`) and its completed-effect ledger now
  count `tasks x repeats` rather than tasks.

### Changed
- `GEPA_RESULT_SCHEMA_VERSION` is `whetstone.gepa_detailed_result/v2`: the
  detailed result now carries `validation_num_seeds`, so a v1 consumer with
  `extra="forbid"` must refuse it rather than read it as v1.
- `Miprov2Control` gains `num_seeds`, the repeats every in-search evaluation
  of the run pays for, and records it in the control's identity payload: a
  control that evaluates each task three times is a materially different
  control from one that evaluates it once. `MIPROV2_CONTROL_SCHEMA_VERSION`
  is now 8.
- The persisted MIPROv2 study contract records `validation_num_seeds`, so an
  audit reads the repeat count off the run record instead of inferring it
  from row counts. `MIPROV2_STUDY_SCHEMA_VERSION` is now 7, and
  `MIPROV2_INTENT_CONTEXT_SCHEMA_VERSION` is now 3 for the matching per-intent
  field.
- MIPROv2 bootstraps a demo from repeat 0's execution trace, chosen
  deterministically so a replay reproduces the same demo. DSPy bootstraps a
  demo from a single sampled trace, so repeats cannot be averaged into one;
  the repeats still inform the demo's recorded score, which is the reward
  over the whole reduced evaluation. Every repeat of a bootstrap's one task
  must still succeed.
- GEPA projects one row per task at any repeat count: the Pareto score stays
  the per-task mean over repeats, and the lowest-`seed_index` repeat that
  *completed* supplies the representative output and trace the reflection
  reads. A task projects a scored row whenever at least one repeat completed,
  carrying that mean plus the count of failed repeats, and a failed row only
  when every repeat failed; a single flaky repeat therefore no longer wedges
  the evaluation. A GEPA metric call remains one candidate-task evaluation,
  matching what `gepa_auto_budget` already counts (valset size plus minibatch
  sizes, in task units), so a `max_metric_calls` budget pinned in metric calls
  keeps its meaning under repeats — a repeated evaluation bills K_REPEAT times
  as many provider rows, while its metric-call count is unchanged. The GEPA
  run record now states the repeat count it resolved to
  (`GepaDetailedResult.validation_num_seeds`, mirroring MIPROv2's
  `validation_num_seeds`), so an audit can diff a run against the envs
  manifest without walking evidence. The evaluation-response projection
  identity now states its repeat reduction instead of pinning `num_seeds`
  to 1.
- A failed MIPROv2 in-search evaluation now debits its whole `tasks x repeats`
  row matrix. `resolve_evaluation_failure` counted tasks only, while the
  canonical replay recomputes `len(task_batch_hashes) * num_seeds`, so at more
  than one repeat folding a non-COMPLETED evaluation wedged the run with
  `completed-effect ledger is not the canonical evidence replay` and
  under-reported rows the evaluation had already paid for.
- The toy run harnesses (`prepare_toy_copro_run`, `prepare_toy_miprov2_run`,
  `prepare_toy_gepa_run`, `prepare_toy_codex_run`) accept `num_seeds`.


- The platform step executor hands the launch's `extra_pools` to the opening
  Step only; later Steps rebuild their state from the prior result, so a
  stale opening state can no longer override it. (They reach only the opening Step.)
- MIPROv2 can now run through the platform step path at all. The platform
  step executor built its opening `extra_pools` from scratch and never
  merged the launch's own, so an optimizer whose opening state is larger
  than its control lost that state on the way to its first Step. MIPROv2
  binds its opening `Miprov2State` at pool key `miprov2_state`, so every
  MIPROv2 run through `execute_optim_step_sync` failed at step 0 with
  "MIPROv2 initial step requires the opening state at pool key
  'miprov2_state'". The in-process controller had always passed
  `launch.extra_pools` through; only the platform path dropped them.

  The executor now merges the launch's pools with its own.
  `platform_stage_index` — the GEPA deferral salt, which only the platform
  path can know — stays executor-owned: a launch binding that key is
  rejected rather than silently winning or losing, since overwriting the
  launch's pool would corrupt bound opening state and overwriting the salt
  would replay a deferred CONTINUE. COPRO, GEPA, and Codex were unaffected,
  as none binds launch pools.

## 0.1.10 - 2026-08-23

### Fixed
- A Codex run no longer reaches the user's `~/.agents` tree. The Codex CLI
  resolves its agent-extension roots — the 0.148 skills loader's
  `~/.agents/skills` among them — from `HOME`, and scans them at startup
  before reading any config; no `skills` config key and no `--disable`
  feature flag suppresses that scan. Each run now points `HOME` at its own
  scratch directory, so the scan finds an empty directory the agent already
  owns. The real home holds dotfiles, credentials, and the trees
  `~/.agents/skills` symlinks into, and the agent's single MCP tool is meant
  to be its only capability. Granting the sandbox profile read access to the
  scanned paths was measured and rejected: the entries symlink into the
  dotfiles repository.

  Under `(deny default)` the un-redirected scan failed with `EPERM` and
  logged `failed to scan skill path ... Operation not permitted (os error
  1)`. The run still succeeded, but the line landed in the stderr tail that
  unrelated failures quote, where it read as their cause.

- A failing Codex Step now names the CLI's own error instead of whatever
  advisory happened to reach stderr. Under `--json` the actionable cause is
  an `{"type": "error"}` item on *stdout*, while stderr carries startup
  noise, so a failure message built from the stderr tail alone could report
  a warning as the cause — which is how a real study-stage failure came to
  be reported as a skills-loader problem. The message now leads with the
  transcript's error items, then the last few JSONL event types (which
  distinguish "never started" from "died mid-stream" when the CLI exits
  with no error item at all), and keeps the stderr tail after them. Both
  the `Codex exited N` path and the `produced no final output artifact`
  path report it, and the transcript detail is bounded the way the stderr
  tail is.

## 0.1.9 - 2026-08-22

### Fixed
- MIPROv2 no longer aborts a durable run with `ValueError: No valid program
  found in param_score_dict` when minibatching is on and every observed
  parameter combination has already been promoted — reachable in practice at
  `num_candidates == 2`, where consecutive full-eval steps exhaust the
  combinations the sampler has proposed. `select_promotion` now falls back to
  the last-ranked combination, matching DSPy's
  `get_program_with_highest_avg_score`, which returns the last-ranked entry in
  the same state instead of failing.

- A Codex Step that concludes the seed is best now completes as
  `seed_retained` whichever way the agent says so. The artifact accepts
  two statements of it — `selected_call_id: null`, and selecting an
  evaluated call whose candidate content is the seed's — but only the
  first was a terminal. The second reached the mutation diff check, which
  refuses a mutation equal to its base, and failed the whole Step under
  `codex_selection_contract`, discarding a Step whose evaluations the run
  had already admitted, paid for, and debited. Observed three times on
  real-Codex runs. The evidence and the `tool_calls` debit are unchanged;
  the cited call id is recorded on the Step state as
  `codex_seed_retained_call_id`. Genuinely invalid selections — an unknown
  call id, a refused or unscored call, a base outside the Step Request —
  still fail as before. The production prompt now also tells the agent to
  return `selected_call_id: null` rather than re-selecting the seed.

## 0.1.8 - 2026-08-22

### Added
- `tests/real_codex/`: a nine-rung ladder that runs the Codex-direct
  optimizer against the real `codex` CLI with a fake task model (config
  acceptance, auth preflight, one hosted-MCP Step, capacity / wall-budget /
  no-call terminalization, multi-evaluation loop, reasoning efforts,
  transcript retention, sandbox denial, bearer-token refusal). Marked
  `real_codex`, deselected by default, opt-in with `WHETSTONE_REAL_CODEX=1`.
  `scripts/check-real-codex.sh` runs it and writes the transcript and rung
  table under `~/drotherm/data/whetstone-ai/real-codex/<timestamp>/`;
  `.github/workflows/real-codex.yml` runs it on demand on a self-hosted
  macOS runner.
- `tests/test_codex_wire_goldens.py` pins the structured-output schema and the
  config keys the runner writes for the real CLI as literals.

### Changed
- A custom `SubprocessCodexRunner(prompt_builder=...)` now receives a
  `CodexPromptContext` instead of the bare `OptimStepRequest`. It carries
  the Step's `model_route` and `base_ref` — the two values the agent can
  derive from nothing it can see — alongside `tool_name`,
  `lease_token_hash`, and `max_tool_calls`, so a builder no longer has to
  rederive them from private runner helpers and risk disagreeing with the
  route the Step's own evaluation server advertises. Breaking: builders
  take one context argument.

### Fixed
- `.github/workflows/real-codex.yml` passes the dispatch `selector` input
  through the environment rather than interpolating it into the `run:`
  script, so a dispatcher cannot inject shell onto the self-hosted macOS
  runner that holds a logged-in Codex session. An empty selector still
  runs the whole ladder.
- The Codex-direct optimizer now produces evaluations against the real CLI.
  Four defects each made every real run yield zero evaluations and were
  unreachable by the scripted fake CLI: the output schema derived from
  Pydantic carried `additionalProperties: true` and was rejected by the
  structured-output validator; `code_mode_host` — which routes MCP tool
  calls in codex 0.148 — was denied, hiding `evaluate_candidate`; the MCP
  approval mode `auto` failed every call under `codex exec`'s `never`
  approval policy (now `approve`); and `model_route` / `base_ref` were
  unguessable, so the agent spent admitted capacity on refused guesses (now
  named in the prompt and pinned as schema `const`).
- Codex web search is disabled through `web_search = "disabled"`; the two
  feature-flag denials it replaced are deprecated no-ops in codex 0.148.
- A failure while building the Codex prompt now terminalizes the Step
  instead of stranding its effect lease. Prompt construction runs under
  the entered MCP host but sat outside the runner's normalized region, so
  an unexpected Step Request shape or a raising `prompt_builder` escaped
  `CodexAdapter.invoke` entirely and left the `NO_REDRIVE` effect
  nonterminal until the lease lapsed.

## 0.1.7 - 2026-08-22

### Added
- The Codex-direct optimizer is wired through the shared contracts:
  `CodexControl`, a `CodexStepContractProvider` registered under the
  `codex` adapter key, `prepare_codex_run` beside the other
  `prepare_*_run` functions, `whetstone-optim run --adapter codex`, and
  `build_toy_codex_control` / `build_toy_codex_adapter` /
  `prepare_toy_codex_run` in `whetstone.testing`.
- Codex is granted exactly one tool: evaluate a candidate on the run's
  internal split and read back the aggregate reward plus per-task
  scores. Every call is admitted against a per-run `ToolCapacity` whose
  size is the control's `max_tool_calls`, which is simultaneously the
  step's `tool_calls` budget, so the admission cap and the Issued Tool
  Call ledger's limit cannot drift. A capacity refusal is advisory --
  the agent is told further calls will be refused, and the wall budget
  is the hard stop.
- The Codex output artifact carries no candidate body. It names the
  `call_id` it selected, and the adapter rebuilds that candidate from
  the call's recorded, content-addressed arguments, so a template that
  was never evaluated through the tool cannot be returned. An artifact
  naming a call that was never issued is a terminal failure, and so is
  one naming a call whose evaluation terminally failed: `COMPLETED` is
  also the terminal state of a failed evaluation, which carries no
  output and no reward, and a candidate that was never successfully
  scored is not a result.
- The ledger is total over *admitted* calls, not reported ones, on every
  path that terminalizes a step. Every failing exit leaves the adapter
  through one path that reconciles first and fails second, so ledger
  totality does not depend on which thing went wrong. It enumerates the
  durable admission entries and re-issues every completed one through the
  guarded handle before it fails. The handle reads the recorded terminal
  instead of evaluating, so this records work already paid for and never
  buys more. An agent therefore cannot hide paid evaluations from the
  Step Result or leave the `tool_calls` budget under-debited -- not by
  omitting them from `evaluated_call_ids`, not by corrupting or omitting
  the artifact's `lease_token_hash`, not by reporting a call id twice or
  naming a selection it never evaluated, and not by exiting nonzero
  without an artifact at all. A run that simply ran out of wall clock
  still surfaces everything it spent.
- A Codex process that fails without a usable artifact -- a nonzero exit,
  an unspawnable process, an unreadable or malformed final message --
  terminalizes the step under `codex_execution_failed` instead of raising
  out of the adapter. The harness runs its effect-lease maintenance only
  once the adapter returns, so an escaping exception left the effect
  non-terminal, and this adapter's `NO_REDRIVE` policy then blocked the
  run from recovering until the lease lapsed.
- A shortfall says which kind it is. An omitted call that `COMPLETED` is
  the agent under-reporting (`codex_unreported_evaluation`); an admitted
  call whetstone's own evaluation server never reached a terminal for is
  a harness failure (`codex_evaluation_interrupted`), named with the
  interrupted call ids. The agent had no result to report in the second
  case, so the two no longer share one accusatory code. This holds on the
  wall-budget stop too, where the kill can strand an in-flight evaluation
  in `ACCEPTED` with its capacity already debited. The admission contract
  has no typed release, so that capacity slot stays consumed -- the run
  really did commit the evaluation -- and the step names the stranded
  calls rather than letting them vanish into the accepted count.
- A durable admitted call whose recorded `template` and `base_ref` cannot
  be read back is a typed failure (`codex_recorded_call_contract`), not a
  silent skip. Reconciliation validates the recorded arguments before it
  issues the call, so a call can no longer reach the Issued Tool Call
  ledger while being omitted from the evidence the step's single shared
  terminal failure is computed over.
- The candidate rebuilt from the selected tool call is assembled the same
  way every other proposal path assembles one: from the base candidate's
  payload with only the run's mutation field replaced. Rebuilding it from
  the mutation field alone dropped every other payload field the base
  carried, so any multi-field candidate failed the mutation diff even on
  a legitimately evaluated selection.
- The Codex preflight probe resolves the default `~/.codex` credentials
  when `CODEX_HOME` is unset, so a user authenticated the ordinary way
  passes it. The probe is constructed with an explicit environment, which
  previously resolved its auth source to nothing and staged no
  credentials into the scratch `CODEX_HOME`. Credentials still reach the
  untrusted agent only as files in its own scratch home, never as
  environment values, and `containment` now owns the accepted auth
  filenames so the preflight's check and the runner's staging cannot
  drift apart.
- Truncated Codex output is never presented as a contiguous stream. When
  a finite output budget retains a head and a tail and drops the middle,
  the join carries an explicit elision marker line -- identified by a
  fixed sentinel token rather than a human-readable prefix -- and
  the isolation block records `stdout_truncated` / `stderr_truncated`
  alongside the dropped byte counts. Concatenating the two fragments bare
  fabricated a line the process never emitted, which the JSONL parser
  then read as a malformed event at a boundary Codex never produced -- or
  as a well-formed event that never happened.
- `ToolCallStore.admitted_entries` joins `accepted_count` across the
  memory, SQLite, and PostgreSQL admission backends: the count says how
  many evaluations a scope paid for, the projection says which calls and
  what durable state each one reached.
- whetstone hosts the MCP evaluation endpoint itself, outside the Codex
  sandbox, and gives the agent only a loopback URL and a bearer token.
  The evaluation server is the sole writer of the whetstone store -- the
  durable ledger, and the admission-capacity rows that cap paid
  evaluations -- so the agent's sandbox profile grants no write access to
  it at all; its scratch directory is the whole writable set. The
  containment profile permits network, so the endpoint is authenticated
  rather than merely bound to loopback.
- A run-scoped lease token (`WS_MCP_RUN_LEASE_TOKEN`) does two separate
  jobs. `WS_MCP_RUN_LEASE_BINDING` binds it to the run's exact Tool
  Config and capacity binding, and the server refuses to start when the
  digest it recomputes disagrees, so a token minted for another run
  brings up nothing. Separately, the adapter refuses an artifact whose
  recorded token hash is not the one it minted, which shows the artifact
  came from a process that received this step's prompt.
- A Codex wall-budget stop terminalizes the step under a typed failure
  instead of raising `subprocess.TimeoutExpired` out of the optimizer,
  which previously escaped the harness's effect-lease maintenance and
  wedged the run until the lease lapsed.
- `codex_auth_preflight` proves a usable Codex session -- binary, auth
  source, and one cheap structured probe. It is required rather than
  optional: `prepare_codex_run` takes no default `preflight`, and the
  CLI's `--adapter codex` path runs the real check before it builds an
  adapter, so a broken session commits no capacity or eval budget.
- `build_codex_executor` is the repository's one production dr-exec
  `ProcessExecutor` construction site.

### Changed
- Run cost reads tool-mediated evaluations. `aggregate_run_cost` now walks
  `OptimStepResult.tool_evidence` alongside `resolved_intents` and
  `search_evidence`, following each Tool Result's
  `evaluation_evidence_refs` to the output rows behind it. The Codex arm
  has no proposer -- the agent proposes -- and drives every evaluation
  through a tool, so it cited all of its spend from a channel run cost did
  not read and an entire Codex run reported `task_model.calls == 0`. The
  three channels are unioned and still de-duplicated by evidence ref, so an
  evaluation reachable through more than one is paid for once.
- `EngineToolEvaluator` was dead on arrival: it referenced `Candidate`,
  `EvalEvidence`, and `EvalFailureEvidence` without importing them, and
  raised `ToolEvaluationError` with a plain string where the constructor
  takes a `TerminalFailure`. Every failure path now constructs a real
  `TerminalFailure` under an owned code, and each is covered by a test.
- `build_runtime` takes an injectable `admission` authority, an
  injectable `tool_executor`, and derives `adapter_replay_policy` from
  the registered adapters instead of hardcoding `DURABLE_WORKFLOW`.
  Codex requires `NO_REDRIVE` and a durable admission authority, because
  its capacity gates an out-of-process MCP server; a `TOOL_USING` run
  additionally needs a tool executor, which `build_runtime` never
  passed.
- The Codex dr-exec job now bounds `payload_output` as well as
  `wall_time`, and truncation under that budget is a reported outcome
  rather than an error. `RegisteredRuntime` exposes the exact
  `tool_store` the harness admits through, and the runtime engine
  exposes its `reward_policy`.
- `SubprocessCodexRunner` calls `Executor.run_blocking`; it previously
  called the coroutine `Executor.run` without awaiting it.
- The task-model API key reaches the evaluation server alone. It was
  previously added to the environment allowlist of the Codex process
  itself, which -- with network allowed and an interpreter on PATH --
  was the credential an agent would need to score candidates outside the
  tool entirely.
- `CodexControl` carries `reasoning_effort`, which now reaches the CLI as
  `-c model_reasoning_effort`. No control field shapes identity without
  shaping execution; the module docstring names the route each one takes,
  since only `codex_binary`, `model`, `reasoning_effort`, and
  `denied_features` reach the argv itself.
- A failed Step Result may now supersede its nested terminal failures
  instead of being required to equal every one of them. Two evaluations
  that failed for different reasons -- the ordinary shape, since
  `EngineToolEvaluator` names the failing call in its own failure --
  previously made the Step Result unconstructable: it raised a raw
  `ValidationError` after the effect lease had already been terminalized,
  so every re-run replayed the same checkpoint and raised again. A
  superseding Step failure must name the exact set of nested codes it
  stands for, under `superseded_failure_codes`, so it still cannot
  silently disagree with its own evidence.
- The Codex MCP host serves on the loopback socket it reserved rather
  than closing it and letting uvicorn re-bind, and takes readiness from
  uvicorn's own started signal rather than a connect probe. A probe
  proved only that *something* accepted on the port, so a process that
  won the re-bind window became the agent's evaluation endpoint and
  received the run's bearer token while uvicorn's bind failure went
  unread. Its startup budget is also now the real time it names: the
  previous wait counted iterations without sleeping, spending a nominal
  30 seconds in about 0.14 of them. The host closes the persistent store
  session its server opened, which the stdio server it replaced used to
  release by exiting.

### Removed
- `CodexControl.max_turns` and `CodexControl.seed`. Both shaped the
  control identity and the recorded hyperparameters while reaching no
  part of the invocation -- `codex exec` exposes neither, and
  `--strict-config` rejects both as unknown configuration fields -- so
  two runs differing only in `max_turns` recorded different identities
  and executed byte-identical commands.
- The stdio MCP server entrypoint, the `whetstone-mcp-eval` console
  script, and the runtime staging that copied the whetstone package into
  every run's scratch directory. The agent no longer spawns the
  evaluation server, so none of it has a caller.
- `CodexOutputArtifact.proposals` and the adapter's proposal-contract
  validation, superseded by ledger-resolved selection.
- The MCP evaluation tool's optional `task_ids` subset variant, and the
  evaluator's engine-narrowing branch. A narrowed engine mints a
  different Eval Config identity, which `EvaluatingToolExecutor` rejects
  as `tool_eval_config_mismatch`, so such a call could never complete.
- The unused `Path`-taking `_parse_output_artifact`.

### Fixed
- An admitted Tool Call whose evaluation the engine rejects now reaches a
  terminal instead of being stranded. `EngineToolEvaluator.validate` runs
  before admission and can only check the call's Eval Config binding and
  model route; a render-contract violation, or an output field the engine
  cannot supply, is discovered inside `evaluate`, when the entry is
  already `ACCEPTED`, its capacity debited, and its effect lease held.
  `EvaluatingToolExecutor` caught only `ToolEvaluationError` around
  `evaluate`, so the `ToolValidationError` raised there propagated out as
  an MCP error, left the entry nonterminal until its lease expired, and
  made reconciliation fail the whole Step as
  `codex_evaluation_interrupted` -- an agent that submitted one bad
  template could not recover by submitting a good one. The rejection now
  persists a terminal `ToolResult` under the new
  `tool_evaluation_rejected` code, on the same path an evaluation failure
  takes, so the ledger stays total and the Step still completes on the
  agent's next valid call.
- Every Tool Call the Codex adapter reconciles must bind one of its Step
  Request's candidates as its base, not only the call the agent selected.
  A `base_ref` is never resolved during evaluation, so a syntactically
  valid ref from another run -- or a forged one -- was scored, and
  `_candidate_from_call` checked the base against the request only for
  the *selected* call. An agent could therefore evaluate a candidate
  outside the run's mutation ancestry, then select a legitimate call, and
  the Step would complete carrying that candidate on its paid Tool
  Evidence. `_admitted_calls` now requires exact ref equality against the
  Step Request's candidates for every reported call, failing under
  `codex_recorded_call_contract` through the terminalizing path.
- `_admitted_calls` validates a reported call's recorded `template` and
  `base_ref` before issuing it, matching `_issue_completed`. It called the
  guarded handle first, so a call with unusable recorded args reached the
  Issued Tool Call ledger and was then left out of the reconciled
  evidence carried to the terminal path -- the ledger-versus-evidence
  split the reconciliation rule exists to prevent -- and the failure was
  attributed to the agent's reporting rather than to the recorded call's
  contract. Such a call now never reaches the ledger, and fails under
  `codex_recorded_call_contract`.
- A Codex output artifact naming another run now terminalizes the step
  under `codex_artifact_run_mismatch` instead of raising past the adapter
  checkpoint. `CodexRunner` is a Protocol, so the adapter cannot assume
  the runner validated the artifact's run; the mismatch check sat outside
  the terminalizing block, and the harness releases the effect lease only
  once the adapter returns an `AdapterOutput`, so a `NO_REDRIVE` run was
  wedged until the lease lapsed. The check now runs on the single
  `_terminalize` path, reconciling the ledger first, so evaluations the
  step already paid for stay reachable and debited.
- Run cost reads the rows a tool-mediated evaluation paid for before it
  failed. `EvaluatingToolExecutor` builds a failed `ToolResult` from the
  evaluator's `TerminalFailure` alone, leaving `evaluation_evidence_refs`
  empty, so an evaluation that produced `EvalFailureEvidence` with an
  `outputs_ref` -- scoring or persistence failing after billed provider
  rows were produced -- cited its evidence only from the failure's
  `details`. `aggregate_run_cost` now follows that typed ref, symmetric
  with the `resolved_intents` path and de-duplicated by ref, so those
  billed rows no longer drop out of Codex task-model spend.
- The truncated-JSONL parser no longer forgives a complete malformed
  record next to the stitch. When retention ended or began exactly on a
  record boundary, a whole malformed line adjacent to the elision marker
  was classified as budget damage on position alone and silently dropped,
  so the persisted `jsonl_events` differed from the retained output. A
  boundary line is now forgiven only when it is demonstrably cut. The
  head side must open a record it never closes. The tail side cannot be
  read off its own shape -- its first retained byte lands mid-token, so
  brace and quote parity are both meaningless there, and a complete
  malformed line such as `not json}` closes without opening exactly as a
  real fragment does -- so it additionally requires the retained head to
  end mid-record, which is the one place the stream demonstrably shows a
  record spanning the elision. A tail line beside a head that ended on a
  clean record boundary now fails the run rather than being deleted and
  reported as retention damage. Genuinely cut fragments are still
  tolerated and counted.
- A Codex run whose stdout exceeded `max_output_bytes` no longer fails on
  its own truncation. The retained stream is a stitched head+tail
  carrying a deliberately non-JSON elision marker, and the budget may cut
  the head's last line and the tail's first line mid-record; the strict
  JSONL parser rejected all three, so a zero-exit run with a valid final
  artifact could not complete. The parser now skips the marker and drops
  the two boundary fragments a stitch can damage, recording the count as
  `jsonl_dropped_partial_lines` in the process evidence. An untruncated
  stream stays strict, and a truncated one still rejects malformed lines
  away from the stitch.
- `prepare_codex_run` attests every engine-derived binding on the
  control, not just the reward policy and Eval Config. A control whose
  `evaluation_execution_policy_hash`, `task_model_identity_hash`, or
  `internal_task_hashes` disagreed with the runtime engine was accepted,
  so evaluations ran against the engine's policy, model route, and task
  split while the persisted optimizer identity claimed the control's --
  silent provenance corruption no downstream reader could detect.
- Every way the Codex runner can fail now terminalizes the step. The
  adapter caught only `CodexStructuredExecutionFailure`, so a zero-exit
  CLI whose artifact failed schema validation, and a dr-exec
  `ExecutorFailure`, both raised the base `OpaqueStepError` past the
  adapter checkpoint. The harness never ran its effect-lease
  maintenance, leaving that `NO_REDRIVE` effect non-terminal instead of
  producing `codex_execution_failed`.
- `whetstone-optim run --adapter codex` carries the launch's mutation
  field and template render contract on the serialized runtime
  configuration, so the out-of-process MCP evaluation server rebuilds the
  harness's engine rather than the toy defaults. Previously a non-default
  mutation field made every tool call fail preflight and a non-default
  render contract could score a different prompt than the harness
  declared. The server refuses to start when the configuration cannot
  supply the field, when it disagrees with the Tool Config's
  `candidate_template_field`, or when it carries no render contract. The
  contract previously defaulted silently to the toy one even though the
  mutation-field check passed, so the server rendered the agent's
  candidate under different rules than the harness scored the baseline
  with and reported the result as comparable.
- MCP host setup and lifecycle failures terminalize the step instead of
  escaping it. `build_server_from_env` raising on a mismatched runtime
  configuration, and `CodexMcpHost.__enter__` raising `CodexMcpHostError`
  for a squatted port, a bind or lifespan failure, or a startup that
  missed its deadline, are not `OpaqueStepError`, so they unwound past
  the adapter checkpoint and left that `NO_REDRIVE` effect non-terminal
  until the lease lapsed. They now fail the step under
  `codex_mcp_host_failed`, which the ledger keeps distinct from an agent
  that ran and failed: the Codex process never started, so nothing was
  paid for. That code covers host startup only. An unforeseen failure
  inside the agent execution still terminalizes -- an escaping exception
  would wedge the same `NO_REDRIVE` run -- but under
  `codex_execution_failed`, because the host is up and the agent may
  have run; reporting it as a host failure claimed the step never
  started and buried the real defect behind a host diagnostic. A
  teardown that fails after a completed run is labelled as such, and
  never displaces an exception raised by the run itself.
- The JSONL parser identifies the elision marker by an exact,
  sentinel-anchored match on the whole line. It previously skipped any
  line merely starting with `[... `, so a genuine Codex line opening
  with a bracketed aside was dropped from the transcript silently, and
  -- because the stitch-boundary search stops at the first
  marker-shaped line -- such a line also misdirected that search, so
  the two lines the output budget really did cut were no longer
  forgiven and a truncated run with a valid final artifact failed.
- Schema and JSONL parse failures keep their isolation evidence. Both
  were raised as a bare `OpaqueStepError` after `_execute_structured`
  had already built the isolation record, so the terminalized step stored
  an empty `codex_isolation` -- no profile, no budgets, and no output
  truncation flags, leaving a reader unable to tell a malformed artifact
  from one the output budget cut in half.

### Known limitations
- Codex process isolation is macOS-only: it requires `sandbox-exec` and
  refuses to run without it rather than falling back to an insecure
  path. A Linux containment profile is separate work. Of the 77 tests in
  `tests/test_codex_*.py`, 13 are Darwin-gated -- the 10 end-to-end tests
  that spawn a sandboxed process, the 2 sandbox-profile tests, and the
  one preflight test that asserts a nonzero probe exit -- so 64 run on
  Linux. Every guarantee whetstone enforces in Python is among them: the
  environment allowlist and run lease binding
  (`tests/test_codex_containment_boundary.py`), ledger totality,
  selection, and shared-failure terminalization
  (`tests/test_codex_adapter_selection.py`), admission
  (`tests/test_codex_admission.py`), timeout terminalization
  (`tests/test_codex_budget_exhaustion.py`), the evaluation endpoint's
  startup and teardown (`tests/test_codex_mcp_host_lifecycle.py`), and
  both golden files.
- dr-exec v1 accepts no finite limit on `process_count` or the resource
  axes, so those are recorded as unbudgeted in the artifact's isolation
  block. The wall budget and the process boundary are the containment.

## 0.1.6 - 2026-08-22

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
  from the token totals. A call reporting only one token direction is
  recorded there too, in both roles: the known side still evidences the
  call, but the absent side enters the totals as zero, so the token totals
  understate it and one direction is not a breakdown.
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

- A GEPA Step now records search evidence only for the evaluations the run
  does not already account for, instead of also re-reporting the prefix it
  replayed from the durable effect cache. Upstream `optimize` re-runs from
  the seed every Step, so the old behaviour made Step *i* carry roughly *i*
  entries and a run's evidence grow quadratically in Steps while paid
  evaluations stayed flat: a measured 556-step run produced 155,956 entries
  for 91 distinct evaluations, a 1.73 GB `runtime.sqlite`, and a 766 MB
  `result.json`. On a 60-step fake-transport run the new rule takes total
  entries from 1,770 to 59 (equal to the distinct evaluations) and
  `result.json` from 3.20 MB to 330 KB. No persisted record shape changed;
  the terminal Pareto-front artifact is per-run and remains complete.
- The rule is *already recorded on an ancestor Step Result*, not *served
  from the effect cache*. `CanonicalGepaAdapterFactory.search_evidence`
  walks this run's durable Step chain back through
  `prior_step_result_ref`, unions the `search_evidence` keys it finds, and
  reports every evaluation the search touched -- replayed or fresh -- that
  the chain does not already carry. The two rules differ exactly where it
  matters: an attempt that crashes after the effect cache durably records an
  evaluation but before its Step Result persists, and a PLATFORM deferral
  episode whose placeholder Step Result is discarded when the same
  `step_index` resumes. In both, the evaluation replays yet sits on no Step
  Result, so filtering on replay alone would have lost it permanently.
  Reconciling against the chain keeps it, and each evaluation is still
  recorded exactly once run-wide.
- `SearchEvidence.from_replayed_resolution` is retained, and is now used
  only for an evaluation the durable Step chain never recorded -- the
  attempt that executed it persisted no Step Result, so the reporting Step
  is the first to account for it. Per-entry harness verification --
  run/step binding, expected schema, and store resolution -- is unchanged,
  so a dropped or forged entry is still rejected.
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

### Fixed

- MIPROv2 bootstrap reads the instruction from the control's declared
  `mutation_field` instead of a hardcoded `"user_prompt_template"` key.
  Every demo mode bootstraps, so any experiment whose mutated field carried
  another name raised `KeyError` on the first bootstrap teacher, before any
  provider spend could produce a result. The toy MIPROv2 fixtures now accept
  a `mutation_field`, and a parametrized harness test drives one bootstrap
  per demo mode under a non-default field name.
- Public entrypoints import cleanly as the first `whetstone` import, so
  import order is no longer load-bearing for consumers. `provider.llm_call`
  reached into the `eval.drivers` package for two metadata codecs, and
  `eval.schema`/`eval.protocol` reached back into `experiment.binding`;
  either edge, taken first, hit a partially initialized module. The shared
  call-metadata wire keys and codecs now live in
  `whetstone.execution.call_metadata`, and `EvalConfigRef` lives in
  `whetstone.eval.config_ref` beside `EvalConfig`, with
  `whetstone.experiment.binding` re-exporting it for existing call sites.

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
