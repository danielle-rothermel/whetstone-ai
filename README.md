# whetstone-ai

Experiment orchestration for HumanEval compression studies. Whetstone builds
graph-shaped prediction specs, runs generation and scoring through DBOS, stores
append-only results in Postgres, and includes a compact COPRO-style DSPy prompt
optimizer.

This repo's role: experiment orchestration: DBOS workflows, Postgres persistence, DSPy optimization.
Neighbors: dr-serialize, dr-providers, dr-graph, dr-platform, dr-code, unitbench.
Dependency direction: consumes dr-code's evaluator library; unitbench reads our Postgres.

## Package Layout

- `records/` - Pydantic domain contracts, stable IDs, fair-order keys, and
  payload-size guards.
- `db/schema.py` - SQLAlchemy Core tables for experiment, prediction,
  generation, node-attempt, scoring, and batch-submit records.
- `db/io.py` - typed row builders, row parsers, JSONB validation, and
  insert/select helpers.
- `platform/` - DBOS runtime setup, graph generation workflows, scoring
  workflows, batch submission, queue registration, and CLI entrypoints.
- `lm/` - prompt/provider request and response boundaries.
- `analysis/` - read-only frame loading, plots, reports, and sample inspection
  helpers.
- `optimization/` - COPRO-style encoder prompt search and artifact writers.
- `eval_failures/` - worker failure taxonomy, retry policy, and recordability
  checks.
- `dspy_serialization.py` - JSON-safe DSPy value handling for telemetry and DB
  payloads.

## Documentation

- [TESTING.md](TESTING.md) - test tiers, CI scripts, fixtures, and conventions.
- [Running the COPRO optimizer](docs/running_copro.md) - prompt-optimization
  commands and output artifacts.
- [Limit enforcement points](docs/ref/limit_enforcement_points.md) - payload
  caps and JSONB safety checks.
- [docs/testing_logs.md](docs/testing_logs.md) - operator notes appended by
  optimizer runs.
- [CHANGELOG.md](CHANGELOG.md) - dated project history.

Plans, priorities, and backlog live in Linear, not repository docs.

## Setup

Install dependencies with `uv`:

```bash
uv sync --group dev
```

`DATABASE_URL` points at the application Postgres database. When unset, the
platform uses `postgresql+psycopg:///dr_dspy`. Bare `postgresql://` URLs are
normalized to `postgresql+psycopg://` by the platform CLIs and Alembic env.

Apply the schema from the repository root:

```bash
uv run alembic upgrade head
uv run alembic current
```

## Testing

See [TESTING.md](TESTING.md) for details.

```bash
./scripts/ci/lint.sh
./scripts/ci/unit.sh
DATABASE_URL=postgresql+psycopg:///dr_dspy ./scripts/ci/integration.sh
```

## Platform Workflow

Build prediction specs from an experiment JSON config:

```bash
uv run python -m whetstone.platform.worker build-specs \
  --config-file configs/experiments/humaneval_encdec_smoke.json \
  --configs-root configs \
  --output specs.jsonl
```

Config fragments live under `configs/`: `models/`, `splits/`, and
`experiments/`. Fragment paths in experiment JSON are relative to
`--configs-root` unless an absolute path is provided.

Run one existing prediction spec:

```bash
uv run python -m whetstone.platform.worker run-one \
  --database-url "$DATABASE_URL" \
  --prediction-id "<prediction-id>"
```

Start the generation worker:

```bash
uv run python -m whetstone.platform.worker worker \
  --database-url "$DATABASE_URL" \
  --worker-concurrency 1
```

Submit a JSONL file of `PredictionSpecRecord` payloads:

```bash
uv run python -m whetstone.platform.worker submit-jsonl \
  --database-url "$DATABASE_URL" \
  --operation-key "<stable-submit-key>" \
  --experiment-name "<experiment-name>" \
  --specs-file specs.jsonl
```

`submit-jsonl` indexes specs, orders them by fair-order key, persists
`--chunk-size` windows, and enqueues generation workflows in matching pages.
Re-running the same `--operation-key` resumes from durable batch-item state.

Score a generation run:

```bash
uv run python -m whetstone.platform.worker score-one \
  --database-url "$DATABASE_URL" \
  --generation-run-id "<generation-run-id>"
```

Batch scoring is available through `rescore`. Successful and partial generation
runs with terminal output are scoreable; runs without terminal output are
rejected by the scoring target loader.

## Analysis

Read-only scripts query enc-dec experiment rows in Postgres and write tabular
artifacts, figures, and HTML run logs. Tabular outputs go to
`artifacts/{script_name}/{timestamp}_{stem}.csv|md` (gitignored). Figures and
run logs go to `figs/{script_name}/`.

```bash
uv run python scripts/analysis/q1_model_candidates.py \
  --experiment-name "<experiment-name>"

uv run python scripts/analysis/q2_compression_range.py \
  --experiment-name "<experiment-name>"

uv run python scripts/analysis/q3_repeat_stability.py \
  --experiment-name "<experiment-name>"

uv run python scripts/analysis/q4_task_variation.py \
  --experiment-name "<experiment-name>"
```

Inspect one run as a horizontal-scroll HTML report plus a JSON debug bundle:

```bash
uv run python scripts/analysis/sample_run_inspector.py \
  --experiment-name "<experiment-name>" \
  --sample-index 0
```

## Failure Handling

All storable JSON/JSONB values pass through `ensure_recordable` or
`recordable_jsonb`. Unencodable LM telemetry or persistence payloads raise
`RecordingFailureError` instead of being silently dropped or stored as empty
objects.

Platform DBOS workflows catch step exceptions, call `summarize_exception`, and
persist structured failure metadata on terminal outcome rows. Retryable failures
may step-retry per policy; permanent failures do not.
