# dr-dspy

This package holds reusable helpers and experiment entrypoints for DSPy work in
this workspace. The repo is intentionally split between executable experiment
definitions in `scripts/` and stable experiment/runtime infrastructure in
`src/dr_dspy/`.

## Experiments

### HumanEval Eval-Only DBOS v0

Script:
[`scripts/humaneval_dspy_eval_only_dbos_v0.py`](scripts/humaneval_dspy_eval_only_dbos_v0.py)

This is a durable direct-decoder sweep over HumanEval Plus. It:

- builds a seeded HumanEval sample slice;
- expands model x temperature x repetition jobs;
- queues OpenRouter generation through DBOS;
- queues sandbox scoring through DBOS;
- scores generated code with the shared HumanEval evaluator;
- stores resumable prediction and score state in Postgres tables;
- reports price/performance summaries by experiment name.

The evaluator does not require the generated function name to match the
benchmark `entry_point`. It extracts candidate code, finds top-level functions,
and passes each one to the HumanEval `check(candidate)` cases until one function
passes all cases.

### HumanEval Encoder-Decoder Eval DBOS v0

Script:
[`scripts/humaneval_dspy_eval_only_encdec_dbos_v0.py`](scripts/humaneval_dspy_eval_only_encdec_dbos_v0.py)

This is the two-stage evaluation tool for large encoder-decoder runs:

```text
ground-truth code -> encoder model -> encoded description
encoded description -> decoder model -> decoded code
```

The script stores each run in one wide Postgres row, including:

- the stripped ground-truth code used as encoder input;
- encoder and decoder model configs;
- encoded description and decoded generation;
- encoder/decoder usage, metadata, and cost;
- decoded-code extraction, validation, and HumanEval score;
- evaluator diagnostics such as tested function names and case status counts;
- raw, zlib, gzip, bz2, lzma, and zstd compression metrics for the encoded
  description compared with stripped ground-truth code length.

Model configs are paired. Pass `--model-pairs-json` as a JSON list or an
`@path/to/file.json` containing entries shaped like:

```json
[
  {
    "encoder": {"model": "openai/gpt-5.1-codex-mini", "reasoning": {}},
    "decoder": {"model": "openai/gpt-5.1-codex-mini", "reasoning": {}}
  }
]
```

Both DBOS scripts use `DATABASE_URL` for application tables and
`DBOS_SYSTEM_DATABASE_URL` for DBOS system tables. When
`DBOS_SYSTEM_DATABASE_URL` is unset, they use `DATABASE_URL` for both.

For non-dry-run submissions, `submit` stores a deterministic submission record,
launches a durable DBOS dispatcher workflow, and tails dispatcher progress while
jobs are inserted and generation workflows are enqueued in batches. Workers can
start before the full Cartesian sweep has been submitted. Submit and worker
detail logs are written under `src/logs/<experiment>-<hash>/`.

Common direct eval flow:

```sh
uv run python scripts/humaneval_dspy_eval_only_dbos_v0.py init-db

uv run python scripts/humaneval_dspy_eval_only_dbos_v0.py submit \
  --experiment-name direct-smoke \
  --sample-count 2

uv run python scripts/humaneval_dspy_eval_only_dbos_v0.py worker \
  --queue both \
  --experiment-name direct-smoke

uv run python scripts/humaneval_dspy_eval_only_dbos_v0.py analyze \
  --experiment-name direct-smoke
```

Common encoder-decoder eval flow:

```sh
uv run python scripts/humaneval_dspy_eval_only_encdec_dbos_v0.py init-db

uv run python scripts/humaneval_dspy_eval_only_encdec_dbos_v0.py submit \
  --experiment-name encdec-smoke \
  --sample-count 2

uv run python scripts/humaneval_dspy_eval_only_encdec_dbos_v0.py worker \
  --queue both \
  --experiment-name encdec-smoke

uv run python scripts/humaneval_dspy_eval_only_encdec_dbos_v0.py analyze \
  --experiment-name encdec-smoke
```

Use `status` for database counts, `enqueue-scores` to backfill score workflows
for generated rows, and `repair` to reconcile stranded DBOS workflows and
re-enqueue failed generation/scoring work with fresh workflow IDs.

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
```
