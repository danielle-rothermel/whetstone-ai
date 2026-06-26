# Testing

Run the package checks from `dr-dspy/`:

```sh
uv run ruff check src scripts
uv run ty check
uv run pytest tests
```

Run the mock HumanEval bootstrap smoke harness:

```sh
uv run python scripts/mocks/humaneval_dspy_harness_bootstrap_v0_mock.py \
  --compiled-path /tmp/dr-dspy-smoke.json
```

The smoke harness defaults to Postgres and reads `DATABASE_URL`; the repo-local
`.env` sets it to `postgresql:///dr_dspy` when it is unset. The smoke harness
should exit with status `0`, write the compiled JSON file, write events to
Postgres, and print `baseline:  100.000` and `optimized: 100.000`.

Check the DBOS eval-only harness CLI without making live model calls:

```sh
uv run python scripts/humaneval_dspy_eval_only_dbos_v0.py --help
uv run python scripts/humaneval_dspy_eval_only_dbos_v0.py temperature-probe
```

`temperature-probe` should exit with status `2` unless `--confirm-live` is
passed. Do not pass `--confirm-live` in automated checks.

Plan a small temperature sweep without writing rows or enqueueing workflows:

```sh
uv run python scripts/humaneval_dspy_eval_only_dbos_v0.py temperature-sweep \
  --experiment-name local-dry-run
```

For a local DBOS/Postgres smoke, run these commands from `dr-dspy/` with
`DATABASE_URL` set to a local Postgres database:

```sh
uv run python scripts/humaneval_dspy_eval_only_dbos_v0.py init-db

uv run python scripts/humaneval_dspy_eval_only_dbos_v0.py submit \
  --experiment-name local-mock-dbos-smoke \
  --sample-count 2 \
  --mock-generation

uv run python scripts/humaneval_dspy_eval_only_dbos_v0.py worker \
  --queue generation
```

Stop the generation worker after jobs complete, then enqueue and run scoring:

```sh
uv run python scripts/humaneval_dspy_eval_only_dbos_v0.py enqueue-scores \
  --experiment-name local-mock-dbos-smoke

uv run python scripts/humaneval_dspy_eval_only_dbos_v0.py worker \
  --queue scoring

uv run python scripts/humaneval_dspy_eval_only_dbos_v0.py status \
  --experiment-name local-mock-dbos-smoke

uv run python scripts/humaneval_dspy_eval_only_dbos_v0.py analyze \
  --experiment-name local-mock-dbos-smoke
```
