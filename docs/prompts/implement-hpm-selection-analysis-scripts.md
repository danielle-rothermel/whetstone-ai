# Prompt: implement HPM selection analysis scripts

You are working in `/Users/daniellerothermel/drotherm/repos/whetstone-ai` on
branch `today_exp`.

First read:

- `AGENTS.md`
- `docs/testing_logs.md`
- `docs/v0-migration-completion-checklist.md`
- `docs/remaining-implementation-intentions.md`
- `src/dr_dspy/db/schema.py`
- `src/dr_dspy/db/io.py`
- `src/dr_dspy/platform/scoring.py`
- `src/dr_dspy/humaneval/metrics.py`

The current push is about choosing today's HPMs quickly after enc-dec migration
and rescoring. Build only the analysis scripts needed to answer the immediate
model/compression/repeat/task-selection questions. Do not build a dashboard,
projection system, Unitbench export, notebook workflow, migration code, or
general reporting framework.

## Goal

Add small script-per-question analysis commands that query the v1 Postgres
tables, build pandas data views, write summary artifacts, and save one or two
plots per question.

Use this layout:

```text
src/dr_dspy/analysis/
  __init__.py
  db.py
  frames.py
  plotting.py
scripts/analysis/
  q1_model_candidates.py
  q2_compression_range.py
  q3_repeat_stability.py
  q4_task_variation.py
```

Keep shared helpers minimal. If a helper is only used by one script, leave it in
that script.

## Dependency policy

Check `pyproject.toml` first. If `pandas` and `matplotlib` are not direct
dependencies, add only those needed dependencies with `uv add pandas
matplotlib`. Do not add seaborn, plotly, altair, notebook-only dependencies, or
statistical packages unless absolutely necessary.

Use Typer for the scripts. Use `uv run python` for local execution.

## Data source

Query v1 append-only tables, not v0 legacy tables:

- `dr_dspy_prediction_specs`
- `dr_dspy_generation_runs`
- `dr_dspy_node_attempts`
- `dr_dspy_score_attempts`

The scripts should accept:

- `--experiment-name` repeatable, or one value if repeatable is too much for
  today
- `--database-url`
- `--env-file`
- `--output-dir`
- `--limit` for debugging if easy

Normalize database URLs the same way platform CLIs do. Load `.env` by default
using existing platform helpers if possible.

The base dataframe should include at least:

- `experiment_name`
- `prediction_id`
- `generation_run_id`
- `score_attempt_id`
- `task_id`
- `model`
- `provider_kind`
- `endpoint_kind`
- `graph_layout`
- `generation_status`
- `score_status`
- `score`
- `generated_code_outcome`
- `repetition_seed`
- `dimensions`
- normalized compression axis:
  - prefer `compression_target` when present
  - fall back to `budget_ratio` when present
  - expose the normalized column as `compression_target`
- `encoder_model` and `decoder_model` when present in dimensions or provider
  configs
- provider costs if available from node attempts
- simple text/compression metrics when available in score metrics

Support both new v1 smoke/sweep rows and migrated v0 enc-dec rows. Migrated rows
may use dimensions like `encoder_model`, `decoder_model`, and `budget_ratio`;
new rows may use `compression_target`.

## Question scripts

### Q1: model candidates

File: `scripts/analysis/q1_model_candidates.py`

Answer:

- Which models are reliable enough to include?
- Which models have reasonable pass rate?
- Which models look expensive or pathological?

Output:

- `q1_model_candidates.csv`: one row per model/provider/compression target if
  useful, plus aggregate model rows if easy.
- `q1_model_candidates.md`: ranked shortlist with caveats.
- `q1_pass_rate_by_model.png`: pass rate with counts labeled or included in
  subtitle.
- Optional `q1_generation_score_health.png`: generation success/error and score
  success/error by model.

Metrics:

- total specs/runs
- generation success rate
- score success rate
- pass rate over score-success rows
- scoreable count
- avg provider cost if available
- count of failures/errors

### Q2: compression range

File: `scripts/analysis/q2_compression_range.py`

Answer:

- What compression target range should today's sweep use?
- Where does performance begin to collapse?
- Which targets have enough coverage to trust?

Output:

- `q2_compression_range.csv`
- `q2_compression_range.md`
- `q2_pass_rate_vs_compression.png`
- Optional `q2_coverage_vs_compression.png`

Metrics:

- pass rate by `compression_target`
- pass rate by model and `compression_target`
- scoreable count
- generation/score error count
- average realized compression metric if available from score metrics

### Q3: repeat/task stability

File: `scripts/analysis/q3_repeat_stability.py`

Answer:

- How many repeats per `(task_id, model, compression_target)` look necessary
  for a rough performance estimate?
- What total `N * D` gives a stable enough optimization signal?

Keep this pragmatic. Use simple bootstrap/subsample estimates from available
rows; do not add scipy/statsmodels. If data is too sparse, say so in the `.md`
summary and output the coverage table anyway.

Output:

- `q3_repeat_stability.csv`
- `q3_repeat_stability.md`
- `q3_stability_by_sample_size.png`

Metrics:

- observed variance by grouping
- bootstrap/subsample mean pass-rate interval width by sample size
- coverage counts by model/task/compression

### Q4: task variation

File: `scripts/analysis/q4_task_variation.py`

Answer:

- Which HumanEval task IDs produce useful optimization signal?
- Which tasks are always right or always wrong and should be downweighted or
  avoided for train/dev selection?

Output:

- `q4_task_variation.csv`
- `q4_task_variation.md`
- `q4_task_signal_rank.png`
- Optional `q4_task_pass_rate_distribution.png`

Metrics:

- task-level pass rate
- number of score-success attempts
- number of distinct models
- number of distinct compression targets
- variance or entropy-like score
- flags:
  - always pass
  - always fail
  - sparse
  - useful signal

## Shared helper expectations

Keep helper modules small:

- `db.py`: database URL/env loading and SQLAlchemy engine creation.
- `frames.py`: one or two functions to load the base dataframe and normalize
  JSON dimensions/metrics.
- `plotting.py`: tiny save helpers and consistent light-mode matplotlib style.

Do not put business decisions in SQL if pandas grouping is clearer. SQL should
primarily fetch the joined raw material.

Plots should be light-mode, readable, and saved as PNGs. Avoid dark themes.

## CLI behavior

Each script should be runnable like:

```bash
uv run python scripts/analysis/q1_model_candidates.py \
  --experiment-name v0_encdec_backfill_20260630 \
  --output-dir artifacts/analysis/v0_encdec_backfill_20260630
```

For scripts that can consume multiple experiments, support repeated
`--experiment-name` if easy:

```bash
uv run python scripts/analysis/q1_model_candidates.py \
  --experiment-name v0_encdec_backfill_20260630 \
  --experiment-name humaneval_encdec_smoke_v1 \
  --output-dir artifacts/analysis/model_selection_20260630
```

Each script should print the output file paths and a short summary table to the
terminal.

## Testing

Add focused tests for helper logic, not brittle plot pixel tests.

Minimum useful coverage:

- compression-axis normalization chooses `compression_target` and falls back to
  `budget_ratio`
- aggregation functions handle missing score attempts
- task-variation flags classify always-pass, always-fail, sparse, and useful
  signal examples
- repeat-stability helper handles sparse data without crashing

Run targeted tests:

```bash
uv run pytest tests/test_analysis_frames.py tests/test_analysis_scripts.py
```

If you choose different test file names, run those exact tests and report them.

## Documentation / testing log

Add a short entry to `docs/testing_logs.md` after running at least one script
against real or freshly migrated data. Include:

- command run
- experiment name(s)
- output directory
- row counts loaded
- generated artifact paths
- any caveat about sparse/incomplete data

Do not over-document this in README today.

## Constraints

- Enc-dec analysis only.
- Read from v1 tables only.
- Do not mutate database rows.
- Do not create projections.
- Do not add a dashboard or web UI.
- Do not add notebook/marimo work.
- Do not run full experiments.
- Keep scripts simple and independently runnable.
- Prefer clear CSV/Markdown/PNG artifacts over clever abstractions.

## Handoff

When done, report:

- files changed
- dependencies added, if any
- scripts implemented
- tests run
- one example command per script
- artifact directory from the live/fixture run
- caveats about missing coverage or sparse data
