# Platform graph workflow implementation notes

The v1 platform graph workflow runs `PredictionSpecRecord` rows through DBOS
and persists append-only generation and node-attempt outcomes. It supports a
direct single-spec command and a queued batch-submission path.

## Running the platform path

Run one existing prediction spec:

```bash
uv run python -m dr_dspy.platform.worker run-one \
  --database-url "$DATABASE_URL" \
  --prediction-id "<prediction-id>"
```

`run-one` requires a `PredictionSpecRecord` row to exist before it starts.
Create specs through tests, migration/backfill setup, ad-hoc insertion, or the
batch submit path before using the direct runner command.

Start the queue-consuming platform DBOS worker:

```bash
uv run python -m dr_dspy.platform.worker worker \
  --database-url "$DATABASE_URL" \
  --worker-concurrency 1
```

The `worker` command registers and listens to the
`dr-dspy-platform-generation-v1` queue. Queue registration uses
`on_conflict="always_update"` so worker-concurrency changes made through the
CLI are reflected in DBOS queue metadata on restart.

Submit a JSONL file of `PredictionSpecRecord` payloads:

```bash
uv run python -m dr_dspy.platform.worker submit-jsonl \
  --database-url "$DATABASE_URL" \
  --operation-key "<stable-submit-key>" \
  --experiment-name "<experiment-name>" \
  --specs-file specs.jsonl
```

`submit-jsonl` uses a two-pass JSONL path: a lightweight indexing pass reads
only `prediction_id`, `fair_order_key`, and `experiment_name` from each line,
rejects duplicate `prediction_id` values within the submit operation, inserts the
experiment row if needed, persists batch operation/item audit rows, and enqueues
workflows on `dr-dspy-platform-generation-v1`. Submission is resumable by
operation key: existing terminal enqueue outcomes (`enqueued`,
`workflow_already_present`) are skipped, while pending or previously failed
items are retried.

Score one existing v1 generation run:

```bash
uv run python -m dr_dspy.platform.worker score-one \
  --database-url "$DATABASE_URL" \
  --generation-run-id "<generation-run-id>"
```

The default scoring surface persists one append-only
`ScoreAttemptRecord` using scoring profile `humaneval@v1`. That profile owns
the parser profile `humaneval-best-effort@v1`, metrics profile
`humaneval-metrics@v1`, and HumanEval timeout. The CLI exposes scoring profile
id/version options so parser, metric, timeout, or scoring changes create new
score attempts instead of mutating old results. The default HumanEval task
loader reads `evalplus/humanevalplus` split `test` and selects the task by the
stored v1 prediction spec `task_id`. The command prints `insert_status` as
`inserted` or `already_present`; rerunning the same generation/scoring-profile
attempt is idempotent and reports `already_present`.

Score attempts use `status=success` for completed domain scoring, including
zero-score outcomes such as failed tests, empty generations, extraction
failure, unsupported terminal-output shapes, or no top-level functions. They
use `status=error` for infrastructure or workflow failures such as missing
generation rows or task loading failures. The scorer writes extracted-code
metadata, per-test results when evaluation runs, aggregate evaluation counts in
`metrics.custom["evaluation"]`, and versioned text, Python leakage, AST,
compression, and per-stage metrics into JSONB payloads. Node-output metrics
include every output field; non-string values are converted to canonical JSON
text at the platform boundary before metric extraction. It does not update
generation/node-attempt rows, v0 tables, or projections.

The HumanEval task loader uses a process-local cached task map keyed by dataset
name and split. Direct `score-one` behavior is unchanged, while batch/rescore
callers in the same worker process avoid reparsing the full HumanEval dataset
for every generation run.

The submit path separates chunked persistence from queue admission. After
indexing, refs are globally sorted by `(fair_order_key, prediction_id)` and
loaded in `--chunk-size` windows for full validation and persistence. Peak
Python memory stays bounded to O(n) lightweight refs plus O(`--chunk-size`) full
specs during load/persist instead of materializing every spec at once. The
returned submit summary still loads all batch item audit rows (O(n)), which is
acceptable for operator visibility. After all windows are persisted, enqueueing
repeatedly selects pending batch items for the operation
ordered by `(fair_order_key, prediction_id)` in `--chunk-size` pages. This keeps
large JSONL submissions bounded while giving deterministic queue mixing across
the full persisted operation instead of only within the current input window.
The programmatic `Iterable[PredictionSpecRecord]` submit API may still
materialize the full operation for global ordering, which is acceptable for
small in-memory batches. Fair-order keys are part of the `PredictionSpecRecord`
contract, so submit validates the records but does not recompute a separate
scheduling key. If a later window contains an invalid spec, earlier windows may
already have been persisted, but enqueueing does not begin until all persistence
windows finish.

Fairness currently controls submission and queue-admission order, not strict
execution order. With registered worker concurrency above 1, DBOS workers can
start and finish queued workflows out of the fair prefix, so early partial
results may still clump by provider/model. Use worker concurrency 1 when strict
drain order matters more than throughput; a stricter multi-worker fairness
policy would need a later queue or leasing design.

The submit command does not start a queue worker. Its
`--queue-registration-concurrency` option only registers the DBOS queue metadata
that workers will later use.

During submission, `dr_dspy_batch_submit_operations.status` is set to
`enqueuing` (the only in-progress operation status) and its `requested_count` tracks the number of specs observed so
far. The final summary changes the status to `completed`, `partial`, or `error`.
If a submit process crashes mid-enqueue, operation status remains `enqueuing`
and item rows show the exact pending/enqueued/failed state.

Each batch submit item follows an enqueue lifecycle of
`pending → claiming → enqueued | workflow_already_present | failed`. Claiming
is an explicit in-flight status: a worker atomically moves a row from `pending`
to `claiming` with claim metadata before calling DBOS enqueue, then transitions
the row to a terminal enqueue status when the outcome is recorded. Outcome
updates are claim-scoped and raise if they match zero rows. On resume, submit
resets all `claiming` rows for the operation back to `pending` (along with
retrying `failed` rows) before draining the pending queue again. Re-enqueue after
reset is safe because platform generation workflows use deterministic IDs.

The CLI currently reuses the legacy `dr_dspy.harness.dbos` bootstrap helpers to
avoid introducing a second DBOS configuration path while v1 and v0 coexist.
Workflow start-race detection lives in `dr_dspy.platform.dbos_compat` instead.

## Migration status

The v1 platform schema is still pre-deployment. The `20260629_0001` revision has
been edited while the branch is being hardened, including the draft
`dr_dspy_batch_submit_items.status` shape being replaced by separate
`insert_status` and `enqueue_status` columns. Local or Neon databases that
applied an earlier draft v1 migration should reset the v1 platform tables and
rerun Alembic from the current revision set. This branch does not promise an
upgrade path from earlier draft v1 schemas until the v1 migration history is
declared deployed/frozen.

## Clock steps

Generation start and generation completion use distinct DBOS step names. This
avoids depending on DBOS memoization details for repeated calls to a single
clock step. Node-attempt timestamps are captured inside the node execution step,
where the provider call happens. If DBOS exhausts retries before the node step
returns, the workflow converts the step exception into a terminal node error in
a separate DBOS step.

## Workflow start idempotency

Platform generation workflows use deterministic IDs:
`platform-generate-v1:{generation_run_id}` where `generation_run_id` is derived
from `(prediction_id, attempt_index)`.

`_start_prediction_graph_workflow_handle` starts the workflow under
`SetWorkflowID`. If another caller wins the start race, the platform catches
DBOS workflow-conflict errors (via the shared `workflow_start_raced` helper from
`dr_dspy.platform.dbos_compat`) and calls `DBOS.retrieve_workflow(workflow_id)`
to join the existing run.

Sequential operator re-runs of `run-one` for the same `(prediction_id,
attempt_index)` therefore return the existing completed result instead of
surfacing a raw conflict error. Append-only persistence (`ON CONFLICT DO NOTHING`)
keeps replay idempotent inside a single workflow outcome. Idempotency is
first-write-wins: if a replayed step ever produced different values than the
first run, the database would keep the first persisted rows and would not
surface the divergence.

## Node attempt indexes

Both `generation_runs` and `node_attempts` expose an `attempt_index` column, but
they mean different things:

- `generation_runs.attempt_index` indexes whole workflow reruns for one
  prediction. It participates in `stable_generation_run_id(prediction_id,
  attempt_index)`.
- `node_attempts.attempt_index` indexes retries of an individual node inside one
  generation run.

Node-attempt persistence records one terminal outcome for each invoked node in a
generation run. DBOS retries happen inside the node execution step and do not
create separate node-attempt rows. Until explicit node reattempt workflows are
added, each invoked node is persisted with `INITIAL_NODE_ATTEMPT_INDEX` (0).

## Provider config scope

The runtime provider config is reconstructed from the fields currently stored in
`ProviderConfigRef`: provider kind, endpoint kind, model, throttle key, and
request parameters. Custom provider runtime fields such as `base_url`,
`api_key_env`, and capability flags are not spec-owned yet; adding those belongs
in a later provider-config contract change.

## Throttle preflight and backoff

Each provider node resolves its `ProviderConfigRef` in a dedicated DBOS preflight
step before the LM execution step. If the provider has a `throttle_key`, the
preflight step reads the current throttle backoff state; the workflow durably
sleeps until the key is unblocked before calling the LM step. Retryable provider
failures update the backoff state for that throttle key; successful calls clear
it. Backoff is advisory across concurrent workers, but state read/write errors
and provider-resolution errors are treated as workflow failures rather than
silent fallbacks.

The dedicated `dr_dspy_throttle_backoff` table is a deliberate coordination
choice. DBOS queueing handles durable workflow execution, but it does not model
per-provider-key `blocked_until` and `consecutive_failures` state shared by
independent workflows. The table is therefore the app-owned cross-worker throttle
coordination point, while DBOS remains responsible for workflow durability and
queue dispatch.

## Review closure

The following review items were assessed and closed without further code changes
unless noted above:

- **Partial persist on validation:** intentional chunked commit before enqueue.
- **Fairness vs execution:** fair submit/enqueue order only; not strict multi-worker
  execution order.
- **Throttle table in step 9:** intentional app-owned coordination for backoff.
- **Append-only vs mutable batch audit:** generation/node/score outcomes are
  append-only; batch submit rows are mutable operational audit.
- **Profile/version ambiguity:** deferred to step 10 (HumanEval scoring).
- **Pure/domain boundary:** no refactor needed.
- **Legacy v0 / PlainPromptAdapter:** no new ChatAdapter coupling on the platform
  path.
- **Raw vs parsed artifacts:** out of scope for this phase.
- **Throttle clear after success:** a failed backoff clear on an otherwise
  successful LM call is swallowed so the generation outcome is not lost; stale
  `blocked_until` may linger until the next retryable failure or a later clear.
  Preflight reads and failure writes still propagate DB errors.

## Follow-up notes

- Replace prompt metadata keys such as `user_prompt_template`, `system_prompt`,
  and `provider_config_id` with typed graph/spec fields once the graph contract
  is ready for another breaking change.
- Move database engine/pool ownership into the platform worker runtime instead
  of creating short-lived SQLAlchemy engines inside each DBOS step.
- Move DBOS bootstrap ownership out of `dr_dspy.harness.dbos` and into a shared
  runtime module.
- Add a supported spec-construction path for v1 runs, either as a CLI helper or
  a standard integration-test fixture.
- Extend the persisted provider config contract before allowing experiments to
  vary provider runtime details such as `base_url`, `api_key_env`, or capability
  flags from specs.
- Add DBOS queue-worker submit/resume E2E coverage once enqueue-to-worker
  integration fixtures are standardized.

## Integration-test status

Integration tests live under `tests/integration/` and are opt-in via
`@pytest.mark.integration`. See [TESTING.md](../TESTING.md) for commands,
fixtures, and the tier model:

- **Tier 1:** Postgres round-trip for `load_prediction_spec_step` and
  `persist_generation_result_step`, plus submit/resume helpers:
  `update_operation_summary` terminal completion for all
  `workflow_already_present` items, `prepare_enqueue_retries` claiming reset,
  and `submit_prediction_specs` with real summary accounting (mocked enqueue
  only).
- **Tier 2–3:** End-to-end `run_prediction_graph_workflow_once` under DBOS with
  mocked LM (happy path, retry-exhaustion error fallback with
  `node_step_error_result_step` and preserved node-attempt timestamps, upstream
  `BLOCKED` runs, error-path idempotent replay, duplicate-start recovery,
  persist idempotency, and throttle preflight reading Postgres before the LM
  step).
- **Tier 3.5:** Frozen v0 sample rows reshaped through
  `src/dr_dspy/migration/v0_reshape.py` (outcome import and spec pass-through).

The default unit suite still covers pure graph orchestration, node execution,
record conversion, idempotent persistence SQL shape, queue registration,
submit/resume item selection, partial enqueue failure handling, throttle state
statement construction, and worker import without Postgres or DBOS. Full DBOS
queue-worker submit/resume E2E remains follow-up work.
