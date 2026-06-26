# dr-dspy

This package holds reusable helpers and experiment scripts for DSPy work in
this workspace. The repo is intentionally split between readable experiment
entrypoints in `scripts/` and stable infrastructure in `src/dr_dspy/`.

## Experiments

### 1. HumanEval Bootstrap v0

Script:
[`scripts/humaneval_dspy_harness_bootstrap_v0.py`](scripts/humaneval_dspy_harness_bootstrap_v0.py)

This experiment runs a DSPy `BootstrapFewShot` pass over HumanEval Plus. It:

- builds a shuffled HumanEval train/dev split;
- asks an LM to emit a single Python function for each prompt;
- evaluates generated code in a subprocess sandbox;
- logs run, flow, module, adapter, LM, and metric events;
- saves the compiled DSPy program artifact to
  `logs/compiled_humaneval.json` by default.

The script keeps the experiment-defining choices local: dataset, signature,
metric, optimizer, model setup, run flow, and CLI flags. Shared mechanics are
imported from `src/dr_dspy/`.

Real HumanEval runs call OpenRouter directly through `LoggingOpenRouterLM`,
not through LiteLLM. The default model is `openai/gpt-5-nano`, and the script
sets OpenRouter reasoning to the lowest supported configuration for that model:
`{"effort": "minimal", "exclude": false}`. It also caps completions with
`max_completion_tokens=1000` as a safety limit. Override the model with
`--model`; set `OPENROUTER_API_KEY` in the environment or package-local `.env`
before running the real script.

For a deterministic smoke run, use the parallel mock script:
[`scripts/mocks/humaneval_dspy_harness_bootstrap_v0_mock.py`](scripts/mocks/humaneval_dspy_harness_bootstrap_v0_mock.py).
It imports the real `run_humaneval_bootstrap_flow` but supplies a tiny mock
dataset and `LoggingCallableLM`, so it exercises the same harness without
calling a real model.

### 2. HumanEval Eval-Only DBOS v0

Script:
[`scripts/humaneval_dspy_eval_only_dbos_v0.py`](scripts/humaneval_dspy_eval_only_dbos_v0.py)

This experiment is a durable eval-only model sweep over HumanEval Plus. It:

- builds a seeded HumanEval sample slice;
- expands model x temperature x repetition jobs;
- queues high-parallel OpenRouter generation through DBOS;
- queues lower-parallel sandbox scoring through DBOS;
- stores resumable prediction and score state in Postgres tables;
- reports price/performance summaries by experiment name.

The script is intentionally Postgres-only. It uses `DATABASE_URL` for the
application tables and `DBOS_SYSTEM_DATABASE_URL` for DBOS system tables; when
`DBOS_SYSTEM_DATABASE_URL` is unset, it uses `DATABASE_URL` for both.

Generation and scoring are separate DBOS queues:

```text
dr_dspy_humaneval_generation  # default worker_concurrency=200
dr_dspy_humaneval_scoring     # default worker_concurrency=32
```

Run submit/status/analyze as short-lived commands, and run workers as separate
long-lived processes. For example:

```sh
uv run python scripts/humaneval_dspy_eval_only_dbos_v0.py init-db

uv run python scripts/humaneval_dspy_eval_only_dbos_v0.py submit \
  --experiment-name smoke-db-queue \
  --sample-count 2 \
  --mock-generation

uv run python scripts/humaneval_dspy_eval_only_dbos_v0.py worker \
  --queue generation \
  --experiment-name smoke-db-queue

uv run python scripts/humaneval_dspy_eval_only_dbos_v0.py enqueue-scores \
  --experiment-name smoke-db-queue

uv run python scripts/humaneval_dspy_eval_only_dbos_v0.py worker \
  --queue scoring \
  --experiment-name smoke-db-queue

uv run python scripts/humaneval_dspy_eval_only_dbos_v0.py analyze \
  --experiment-name smoke-db-queue
```

Use `status` for a compact database summary while or after workers run:

```sh
uv run python scripts/humaneval_dspy_eval_only_dbos_v0.py status \
  --experiment-name smoke-db-queue
```

Use `repair` when app rows and DBOS workflow state drift apart, or when failed
generation/scoring rows need fresh workflow IDs. It is a dry run by default and
reports stranded generation rows, retryable generation errors, pending scoring
work, stranded scoring rows, and retryable scoring errors:

```sh
uv run python scripts/humaneval_dspy_eval_only_dbos_v0.py repair \
  --experiment-name smoke-db-queue
```

Apply the repair after checking the dry-run counts:

```sh
uv run python scripts/humaneval_dspy_eval_only_dbos_v0.py repair \
  --experiment-name smoke-db-queue \
  --apply
```

`repair --apply` uses one repair token for the run. It reconciles stranded app
statuses, enqueues generation retries as
`generate-retry:<repair-token>:<prediction-id>`, enqueues missing scoring work,
and enqueues scoring retries as `score-retry:<repair-token>:<prediction-id>`.
Run the relevant worker after applying repair; for scoring-only repairs:

```sh
uv run python scripts/humaneval_dspy_eval_only_dbos_v0.py worker \
  --experiment-name smoke-db-queue \
  --queue scoring
```

Workers print compact queue state changes to stdout: when selected queues have
active work, and when they become empty and wait for more jobs. Detailed
per-job logs go to
`logs/<experiment-name>-<hash>/<timestamp>-<queue>-pid<PID>.log` by default;
pass `--log-file` to choose an exact file, or `--no-monitor` to disable the
stdout monitor.

`temperature-probe` is the only command that intentionally makes immediate
OpenRouter calls. It refuses to run unless `--confirm-live` is passed.

## Repository Shape

`scripts/` contains experiment entrypoints. A script should make the exact
dataset, optimizer, adapter, metric, run flow, and artifact choices easy to
inspect in one place.

`scripts/mocks/` contains deterministic mock runners for experiments. Mock
scripts are allowed to import the real flow function from the experiment script,
but they own fake datasets, fake solvers, and smoke-test-specific setup.

`src/dr_dspy/` contains behavior expected to remain stable across experiments:

- `code_eval.py`: generated-code extraction and subprocess evaluation.
- `dspy_event_log.py`: DSPy callback telemetry.
- `dspy_programs.py`: reusable DSPy execution helpers.
- `event_log.py`: SQLite/Postgres event writers and writer construction.
- `flow.py`: flow context tracking for event logs.
- `lm_logging.py`: logging LM wrappers.
- `openrouter_lm.py`: direct OpenRouter chat-completions LM wrapper.
- `run_metadata.py`: run metadata capture and sanitization.
- `runtime.py`: shared script runtime setup.
- `serialization.py`: DSPy-aware JSON-safe serialization.

`tests/` covers library behavior, the mock harness path, and the DBOS eval-only
planning/generation/scoring/analysis helpers. The tests are not part of the
Ruff/Ty target by default; they remain executable with pytest.

## Design Decisions

Default to a script first. Move code into `src/dr_dspy/` only when it is likely
to be reused unchanged by multiple experiments and centralizing it reduces setup
bugs.

Keep experiment-defining decisions in the script. The library should not hide
which dataset, signature, optimizer, metric, model, or artifact path makes an
experiment what it is.

Prefer clean boundaries over compatibility shims. This package is early enough
that breaking changes are acceptable when they make the structure clearer.

Use Postgres as the default event store and the only store for DBOS-backed
sweeps. `DATABASE_URL` is the standard configuration key, and scripts load the
package-local `.env` file before writer construction. SQLite remains available
for the older bootstrap event writer via `--event-store sqlite`.

Keep mock infrastructure parallel to, not inside, experiment scripts. The main
script should stay readable as the real experiment; the mock script should prove
that the same flow can run with prepared train/dev data and a prepared LM.

## Local Setup

Create a package-local `.env` from the example:

```sh
cp .env.example .env
```

For real OpenRouter-backed experiments, add:

```sh
OPENROUTER_API_KEY=...
```

The default local database URL is:

```sh
postgresql:///dr_dspy
```

Create the local database if needed:

```sh
createdb dr_dspy
```

Run package checks from `dr-dspy/`:

```sh
uv run ruff check src scripts
uv run ty check
uv run pytest tests
```

See [`TESTING.md`](TESTING.md) for the mock harness smoke command and success
criteria.
