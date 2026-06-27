# Testing

Run the package checks from `dr-dspy/`:

```sh
uv run ruff check src scripts
uv run ty check
uv run pytest tests
```

Check the direct DBOS eval CLI without making live model calls:

```sh
uv run python scripts/humaneval_dspy_eval_only_dbos_v0.py --help
uv run python scripts/humaneval_dspy_eval_only_dbos_v0.py submit \
  --experiment-name direct-dry-run \
  --sample-count 2 \
  --dry-run
```

Check the encoder-decoder DBOS eval CLI:

```sh
uv run python scripts/humaneval_dspy_eval_only_encdec_dbos_v0.py --help
uv run python scripts/humaneval_dspy_eval_only_encdec_dbos_v0.py submit \
  --experiment-name encdec-dry-run \
  --sample-count 2 \
  --dry-run
```

For a local DBOS/Postgres direct-eval smoke, run these commands from `dr-dspy/`
with `DATABASE_URL` set to a local Postgres database. Live generation also
requires `OPENROUTER_API_KEY`.

```sh
uv run python scripts/humaneval_dspy_eval_only_dbos_v0.py init-db

uv run python scripts/humaneval_dspy_eval_only_dbos_v0.py submit \
  --experiment-name local-dbos-smoke \
  --sample-count 2

uv run python scripts/humaneval_dspy_eval_only_dbos_v0.py worker \
  --queue both \
  --experiment-name local-dbos-smoke

uv run python scripts/humaneval_dspy_eval_only_dbos_v0.py status \
  --experiment-name local-dbos-smoke

uv run python scripts/humaneval_dspy_eval_only_dbos_v0.py analyze \
  --experiment-name local-dbos-smoke
```

For a local encoder-decoder smoke, use the same environment requirements:

```sh
uv run python scripts/humaneval_dspy_eval_only_encdec_dbos_v0.py init-db

uv run python scripts/humaneval_dspy_eval_only_encdec_dbos_v0.py submit \
  --experiment-name local-encdec-smoke \
  --sample-count 2

uv run python scripts/humaneval_dspy_eval_only_encdec_dbos_v0.py worker \
  --queue both \
  --experiment-name local-encdec-smoke

uv run python scripts/humaneval_dspy_eval_only_encdec_dbos_v0.py status \
  --experiment-name local-encdec-smoke

uv run python scripts/humaneval_dspy_eval_only_encdec_dbos_v0.py analyze \
  --experiment-name local-encdec-smoke
```

The worker should print compact queue counts to stdout. Stop it manually after
the generation and scoring queues drain.
