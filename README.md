# dr-dspy

Graph-based HumanEval evaluation platform workbench. Generation and scoring run
through graph-shaped specs, explicit LM/prompt boundaries, and append-only
terminal outcomes under `dr_dspy.platform`.

Legacy v0 runtime code (mutable prediction-table workflows, repair CLIs, DSPy
`Predict` experiment backends) has been **removed**. Frozen v0 row reshape logic
remains under `migration/` for backfill — see
[`docs/v0-migration-completion-checklist.md`](docs/v0-migration-completion-checklist.md).

## Package layout

- `humaneval/` — task parsing, code extraction, scoring, compression metrics
- `lm/boundary.py` — forward prompt/provider request and response boundary
- `lm/utils.py` — shared JSON/text helpers used by the forward boundary
- `graph/` — pure graph execution and graph-spec hashing
- `records/` — Pydantic domain contracts, stable ids, and fair-order keys
- `db/schema.py` — SQLAlchemy Core table definitions for v1 eval records
- `db/io.py` — typed row builders, row parsers, and insert/select helpers
- `db/migrations/` — Alembic migrations for the v1 schema
- `platform/` — v1 DBOS graph workflow, plain-prompt node execution, append-only persistence, CLI entrypoints
- `migration/` — v0 row → v1 record reshape (backfill only; delete after migration validated)
- `eval_failures/` — worker failure taxonomy, retry policy, recording/generation boundaries
- `serialization.py` — JSON-safe encoding for telemetry and DB payloads

## Design notes

- [Graph-based eval platform design](docs/append-only-eval-records-design.md)
  captures the planned migration toward graph-shaped generation specs,
  append-only outcomes, explicit prompt/LM boundaries, rescoring, metrics, and
  Unitbench-facing projections.
- [Platform graph workflow implementation notes](docs/platform-graph-workflow-implementation.md)
  describe the current v1 workflow entrypoint, DBOS timing boundaries,
  node-attempt indexing semantics, provider-config scope, integration-test
  status, and follow-up work.
- [v0 migration completion checklist](docs/v0-migration-completion-checklist.md)
  documents backfill retention and post-migration cleanup.
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

Scoring workflows and batch rescoring are available via `score-one` and
`rescore`. Unitbench-facing projections and the full v0 backfill job remain
deferred — see [`docs/v0-migration-completion-checklist.md`](docs/v0-migration-completion-checklist.md).

## Database migrations

The v1 eval schema lives under `db/` and is applied with Alembic from the
repository root. Migration history is **frozen** at head revision
`20260630_0005` (nine revisions from `20260629_0001`). See
[`docs/v1-schema-migrations.md`](docs/v1-schema-migrations.md) for the
revision changelog, post-freeze policy, and **reset-not-upgrade** procedure
for databases that applied draft schemas during hardening.

Connection config uses the `DATABASE_URL` env var. When unset, Alembic falls
back to peer-auth `postgresql+psycopg:///dr_dspy` (your OS Postgres role,
database `dr_dspy`). Copy `.env.example` to `.env` and adjust the URL if your
local role or database name differs.

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
dropped or stored as empty objects.

### Generation boundary

Typed generation failures (`EmptyGenerationError`, `PredictionParseError`) are
raised from `eval_failures.generation.require_generation_text` on the forward
LM boundary path.

### Worker workflow pattern

Platform DBOS workflows catch step exceptions, call `summarize_exception`, and
persist structured failure metadata on terminal outcome rows. Retryable failures
(`transient`, `rate_limited`) may step-retry per policy; permanent failures do
not.

Scoring test failures are domain semantics: a wrong answer records a failed test
outcome in score-attempt metrics. That is not a worker failure.
