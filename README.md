# dr-dspy

This package holds reusable helpers and experiment scripts for DSPy work in
this workspace. The repo is intentionally split between readable experiment
entrypoints in `scripts/` and stable infrastructure in `src/dr_dspy/`.

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
  --sample-count 2 \
  --apply

uv run python scripts/humaneval_dspy_eval_only_encdec_dbos_v0.py worker \
  --queue both \
  --experiment-name encdec-smoke

uv run python scripts/humaneval_dspy_eval_only_encdec_dbos_v0.py analyze \
  --experiment-name encdec-smoke
```

Use `status` for database counts and `repair` to re-enqueue pending or failed
generation/scoring work with fresh workflow IDs.

## Repository Shape

`scripts/` contains experiment entrypoints. A script should make the exact
dataset, signature, model, queue, persistence, and CLI choices easy to inspect
in one place.

`src/dr_dspy/` contains behavior expected to remain stable across experiments:

- `analysis.py`: shared numeric summaries and report formatting helpers.
- `code_eval.py`: legacy generated-code subprocess evaluation.
- `code_extraction.py`: generated-code extraction and syntax validation.
- `compression.py`: encoded-description compression metrics.
- `dbos_runtime.py`: shared DBOS/Postgres runtime, queue, workflow, and
  connection-pool helpers.
- `dspy_runner.py`: shared logged LM construction and DSPy predictor execution.
- `human_eval_sampling.py`: HumanEval Plus loading, parsing, and seeded
  sampling.
- `humaneval_direct_dbos.py`: direct-decoder HumanEval DBOS/Postgres adapter.
- `humaneval_encdec_dbos.py`: encoder-decoder HumanEval DBOS/Postgres adapter.
- `human_eval.py`: HumanEval task parsing and name-independent evaluation.
- `lm_logging.py`: logging LM wrappers.
- `lm_utils.py`: model config and LM response helpers.
- `openrouter_lm.py`: direct OpenRouter chat-completions LM wrapper.
- `parsed_code.py`: AST-backed code parsing and comment/docstring stripping.
- `parsed_tests.py`: HumanEval `check(candidate)` case parsing.
- `runtime.py`: shared script runtime setup.
- `scoring.py`: reusable generated-code scoring over `HumanEvalTask`.
- `serialization.py`: DSPy-aware JSON-safe serialization.
- `signatures.py`: reusable signature field model.
- `worker_monitor.py`: shared two-phase generation/scoring worker monitor.

`tests/` covers reusable library behavior and the direct DBOS eval planning,
generation, scoring, and analysis helpers.

## Design Decisions

Default to a script first. Move code into `src/dr_dspy/` only when it is likely
to be reused unchanged by multiple experiments and centralizing it reduces setup
bugs.

Keep experiment-defining decisions in the script. The library should not hide
which dataset, signature, optimizer, metric, model, or artifact path makes an
experiment what it is.

The HumanEval ground truth for encoder/compression experiments is
`prompt + canonical_solution` with comments and docstrings stripped. This keeps
the encoder input aligned with executable solution behavior rather than
benchmark prose.

Prefer clean boundaries over compatibility shims. This package is early enough
that breaking changes are acceptable when they make the structure clearer.

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

See [`TESTING.md`](TESTING.md) for smoke commands and success criteria.
