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

`submit-jsonl` streams JSONL parsing into the submit path, validates each window
of specs against the requested experiment, inserts the experiment row if needed,
persists batch operation/item audit rows, and enqueues workflows on
`dr-dspy-platform-generation-v1`. Submission is resumable by operation key:
existing completed/enqueued batch items are skipped, while pending or failed
items are retried.

The submit path uses approximate windowed fairness. It reads at most
`--chunk-size` specs into memory, validates them, orders that window by the
stored fair-order key, persists the window, enqueues its workflows, and then
continues to the next window. Fair-order keys are part of the
`PredictionSpecRecord` contract, so submit validates the records but does not
recompute a separate scheduling key. This supports large JSONL submissions
without globally materializing and sorting every spec, but it does not provide a
globally fair order across windows. If a later window contains an invalid spec,
earlier windows may already have been persisted and enqueued.

Fairness currently controls submission and queue-admission order, not strict
execution order. With `--queue-worker-concurrency` above 1, DBOS workers can
start and finish queued workflows out of the fair prefix, so early partial
results may still clump by provider/model. Use worker concurrency 1 when strict
drain order matters more than throughput; a stricter multi-worker fairness
policy would need a later queue or leasing design.

During submission, `dr_dspy_batch_submit_operations.status` is set to
`enqueuing` and its `requested_count` tracks the number of specs observed so
far. The final summary changes the status to `completed`, `partial`, or `error`.
If a submit process crashes mid-enqueue, operation status remains `enqueuing`
and item rows show the exact pending/enqueued/failed state.

The CLI currently reuses the legacy `dr_dspy.harness.dbos` bootstrap helpers to
avoid introducing a second DBOS configuration path while v1 and v0 coexist.

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
`dr_dspy.harness.dbos`) and calls `DBOS.retrieve_workflow(workflow_id)` to join
the existing run.

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

Each provider node resolves its `ProviderConfigRef` before the LM call. If the
provider has a `throttle_key`, a DBOS preflight step reads the current throttle
backoff state and durably sleeps until the key is unblocked. Retryable provider
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
- Add Postgres/DBOS integration coverage for submit/resume, throttle
  upsert/read, and workflow-level preflight behavior once the project has a
  standard live-fixture setup.

## Integration-test status

Integration tests live under `tests/integration/` and are opt-in via
`@pytest.mark.integration`. See [TESTING.md](../TESTING.md) for commands,
fixtures, and the tier model:

- **Tier 1:** Postgres round-trip for `load_prediction_spec_step` and
  `persist_generation_result_step`.
- **Tier 2–3:** End-to-end `run_prediction_graph_workflow_once` under DBOS with
  mocked LM (happy path, retry-exhaustion error fallback with
  `node_step_error_result_step` and preserved node-attempt timestamps, upstream
  `BLOCKED` runs, error-path idempotent replay, duplicate-start recovery, and
  persist idempotency).
- **Tier 3.5:** Frozen v0 sample rows reshaped through
  `src/dr_dspy/migration/v0_reshape.py` (outcome import and spec pass-through).

The default unit suite still covers pure graph orchestration, node execution,
record conversion, idempotent persistence SQL shape, queue registration,
submit/resume item selection, partial enqueue failure handling, throttle state
statement construction, and worker import without Postgres or DBOS. Live
Postgres/DBOS coverage for submit/resume, throttle upsert/read, and
workflow-level preflight behavior remains follow-up work.
