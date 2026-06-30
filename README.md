# dr-dspy

Graph-based HumanEval evaluation platform workbench. The current migration path
is toward graph-shaped generation specs, explicit LM/prompt boundaries, and
append-only terminal outcomes. The old direct and enc-dec DBOS workers remain in
the tree as legacy v0 data-generation surfaces until migration validation is
complete.

## Package layout

- `humaneval/` — task parsing, code extraction, scoring, compression metrics
- `lm/boundary.py` — forward prompt/provider request and response boundary
- `lm/runner.py`, `lm/signatures.py`, `lm/openrouter.py` — legacy DSPy compatibility for v0 workflows
- `lm/utils.py` — shared JSON/text helpers used by the forward boundary
- `lm/logging.py` — legacy-adjacent DSPy LM telemetry mixins (recordability at log time)
- `graph/` — pure graph execution and graph-spec hashing
- `records/` — Pydantic domain contracts, stable ids, and fair-order keys
- `db/schema.py` — SQLAlchemy Core table definitions for v1 eval records
- `db/io.py` — typed row builders, row parsers, and insert/select helpers
- `db/migrations/` — Alembic migrations for the v1 schema
- `platform/` — v1 DBOS graph workflow, plain-prompt node execution, and append-only generation/node persistence
- `eval_failures/` — worker failure taxonomy, retry policy, recording/generation boundaries
- `serialization.py` — JSON-safe encoding for telemetry and DB payloads
- `harness/` — legacy v0 DBOS workflows, batch operations, repair, worker monitoring
- `experiments/` — legacy v0 HumanEval direct and enc-dec DBOS backends

## Design notes

- [Graph-based eval platform design](docs/append-only-eval-records-design.md)
  captures the planned migration toward graph-shaped generation specs,
  append-only outcomes, explicit prompt/LM boundaries, rescoring, metrics, and
  Unitbench-facing projections.
- [Platform graph workflow implementation notes](docs/platform-graph-workflow-implementation.md)
  describe the current v1 workflow entrypoint, DBOS timing boundaries,
  node-attempt indexing semantics, provider-config scope, integration-test
  status, and follow-up work.
- [TESTING.md](TESTING.md) documents unit vs integration tests, the tier
  model, shared fixtures, CI scripts, and conventions for adding coverage.
- [Repo split and naming plan](docs/repo-split-and-naming-plan.md) describes
  the standalone `whetstone-ai` repository and deferred `dr_dspy` → `whetstone`
  rename.

## Testing

See [TESTING.md](TESTING.md) for how to run unit vs integration tests, the tier
model, shared fixtures, and conventions for adding new coverage.

## V1 graph workflow

The v1 execution path runs `PredictionSpecRecord` rows through the pure graph
runner, calls the LM provider boundary through DBOS steps, and persists
append-only generation/node outcomes. It supports both direct single-spec
execution and queued batch submission.

Run one existing prediction spec:

```bash
uv run python -m dr_dspy.platform.worker run-one \
  --database-url "$DATABASE_URL" \
  --prediction-id "<prediction-id>"
```

This command assumes the `PredictionSpecRecord` already exists in the
database. This phase does not include a spec-creation CLI; specs must be
inserted by a test fixture, migration/backfill path, or ad-hoc setup before
`run-one` can execute them.

Start the platform DBOS generation worker:

```bash
uv run python -m dr_dspy.platform.worker worker \
  --database-url "$DATABASE_URL" \
  --worker-concurrency 1
```

The `worker` command registers and listens to
`dr-dspy-platform-generation-v1`. Queue registration uses the configured
`--worker-concurrency` and updates the DBOS queue record on restart so operator
concurrency changes are picked up reliably.

Submit a JSONL file of `PredictionSpecRecord` payloads:

```bash
uv run python -m dr_dspy.platform.worker submit-jsonl \
  --database-url "$DATABASE_URL" \
  --operation-key "<stable-submit-key>" \
  --experiment-name "<experiment-name>" \
  --specs-file specs.jsonl
```

`submit-jsonl` indexes specs in a lightweight first pass, globally orders by
fair-order key, loads and persists `--chunk-size` windows, then enqueues
generation workflows on `dr-dspy-platform-generation-v1` in matching pages.
Re-running the same `--operation-key` resumes from durable batch items:
terminal enqueue outcomes (`enqueued`, `workflow_already_present`) are skipped
and pending or failed items are retried.

Queued graph workflow execution runs a DBOS throttle preflight step before each
LM node call. Rate-limited and transient provider failures update the throttle
state; later workflows with the same key durably sleep before calling the
provider.

Scoring, Unitbench-facing projections, and v0 migration/backfill remain
deferred.

The legacy v0 direct and enc-dec workflows write mutable prediction rows that
mix requested specs, workflow status, generation artifacts, scores, and repair
state. Those rows remain source data for migration/backfill, but new
implementation work should not build domain contracts, graph workflows,
rescoring, or reporting on top of v0 repair/status/reporting flows.

## Database migrations

The v1 eval schema lives under `db/` and is applied with Alembic from the
repository root.

Connection config uses the same `DATABASE_URL` env var as the legacy v0
workers. When unset, Alembic falls back to peer-auth
`postgresql+psycopg:///dr_dspy` (your OS Postgres role, database `dr_dspy`).
Copy `.env.example` to `.env` and adjust the URL if your local role or database
name differs.

```bash
# Apply all migrations
uv run alembic upgrade head

# Inspect current revision
uv run alembic current

# Render SQL without connecting (offline mode)
uv run alembic upgrade head --sql
```

Alembic reads `DATABASE_URL` in `db/migrations/env.py` and normalizes
`postgresql://` URLs to the project's `postgresql+psycopg://` driver form.
The `sqlalchemy.url` value in `alembic.ini` is only a fallback when
`DATABASE_URL` is not set.

## Failure handling (`eval_failures`)

Eval worker step failures are classified, summarized for DB/logs, and persisted
with structured metadata. This package is **not** a global exception registry.

### Module roles

| Module | Responsibility |
|--------|----------------|
| `serialization.py` | Typed `SerializationError` hierarchy for unencodable values |
| `eval_failures/recording.py` | `ensure_recordable` / `recordable_jsonb` bridge → `RecordingFailureError` |
| `eval_failures/generation.py` | `require_generation_text`, enc-dec/direct validators |
| `eval_failures/exceptions.py` | `EvalFailureError` hierarchy with `failure_class` |
| `eval_failures/policy.py` | Third-party heuristics, `summarize_exception`, `should_retry_step` |

### Recording boundary

All storable JSON/JSONB values pass through `ensure_recordable` or
`recordable_jsonb`. Unencodable LM telemetry or persistence payloads raise
`RecordingFailureError` (permanent, no step retry) instead of being silently
dropped or stored as empty objects. Call sites include LM logging, predictor
metadata, legacy v0 experiment DB writes, and legacy v0 harness batch
operations.

### Generation boundary

Typed generation failures (`EmptyGenerationError`, `PredictionParseError`) are
raised from `eval_failures.generation.require_generation_text` on the forward
path and from legacy `lm.runner.run_predictor` for v0 DSPy experiment workflows.
Humaneval job builders call
`validate_encdec_generation` / `validate_direct_generation` before constructing
`GenerationResult`, so empty or unparseable LM output becomes
`generation_error` rather than a false `generated` row.

### Worker workflow pattern

Legacy v0 experiment DBOS workflows catch step exceptions, call
`summarize_exception`, and record errors via `failure_summary_payload`.
Retryable failures (`transient`, `rate_limited`) return recoverable status
strings; permanent failures do not step-retry.

Scoring test failures are domain semantics: a wrong answer records `score=0`
with `scoring_status='scored'`. That is not a worker failure.
