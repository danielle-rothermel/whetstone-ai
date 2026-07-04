# Inspecting enc-dec v0 backfill rows

How to select and inspect rows migrated from `dr_dspy_encdec_eval_predictions` into v1 append-only tables via `backfill-v0-encdec`.

Backfilled rows are **not** in a separate table. They live in the normal v1 tables with provenance markers set during reshape/backfill.

**Related:** [`docs/ref/limit_enforcement_points.md`](ref/limit_enforcement_points.md), [`docs/testing_logs.md`](testing_logs.md), [`src/dr_dspy/migration/v0_encdec_backfill.py`](../src/dr_dspy/migration/v0_encdec_backfill.py)

---

## Provenance markers

| Marker | Table | Field | Notes |
|--------|-------|-------|-------|
| `v0_source` | `dr_dspy_generation_runs` | `summary->metadata->v0_source` | **Most reliable** for run-level filters; includes `v0_prediction_id`, `v0_layout`, `v0_generation_status`, `v0_scoring_status` |
| `legacy-v0-migration` | `dr_dspy_prediction_specs` | `fair_order_seed` | Set by `reshape_v0_*`; native v1 submits use experiment config seeds |
| `encdec` | `dr_dspy_prediction_specs` | `graph_layout` | Enc-dec backfill only (direct backfill would use `direct`) |
| `v0_encdec_backfill` | `dr_dspy_experiments` | `config_metadata->source` | Set when backfill creates the experiment row; idempotent re-runs may not update an existing row |

For full backfill (no `--target-experiment-name`), legacy experiment names are preserved (e.g. `encdec-budget-full-v0`, `encdec-smoke`). Smoke runs use isolated names like `v0_encdec_backfill_smoke_20260630_r2`.

---

## Best single filter: `v0_source` on generation runs

Every backfilled run stores provenance in `summary.metadata.v0_source`:

```sql
-- All backfilled generation runs (enc-dec or direct)
SELECT
  gr.generation_run_id,
  gr.prediction_id,
  gr.status,
  gr.summary->'metadata'->'v0_source'->>'v0_prediction_id' AS v0_prediction_id,
  gr.summary->'metadata'->'v0_source'->>'v0_layout' AS v0_layout,
  gr.summary->'metadata'->'v0_source'->>'v0_generation_status' AS v0_generation_status
FROM dr_dspy_generation_runs gr
WHERE gr.summary->'metadata' ? 'v0_source'
ORDER BY gr.generation_run_id;
```

Enc-dec only:

```sql
WHERE gr.summary->'metadata'->'v0_source'->>'v0_layout' = 'encdec'
```

---

## Spec-level filter: `fair_order_seed`

```sql
SELECT ps.*
FROM dr_dspy_prediction_specs ps
WHERE ps.fair_order_seed = 'legacy-v0-migration'
  AND ps.graph_layout = 'encdec';
```

Specs joined to runs:

```sql
SELECT
  ps.experiment_name,
  ps.task_id,
  ps.prediction_id,
  gr.generation_run_id,
  gr.status AS run_status,
  gr.summary->'metadata'->'v0_source'->>'v0_prediction_id' AS v0_prediction_id
FROM dr_dspy_prediction_specs ps
JOIN dr_dspy_generation_runs gr ON gr.prediction_id = ps.prediction_id
WHERE ps.fair_order_seed = 'legacy-v0-migration'
  AND ps.graph_layout = 'encdec'
  AND gr.summary->'metadata' ? 'v0_source';
```

---

## Smoke samples only

When backfill used `--target-experiment-name`:

```sql
SELECT ps.*
FROM dr_dspy_prediction_specs ps
WHERE ps.experiment_name LIKE 'v0_encdec_backfill_smoke_%';
```

---

## Full backfill (legacy experiment names)

After full backfill without `--target-experiment-name`:

```sql
SELECT ps.*
FROM dr_dspy_prediction_specs ps
WHERE ps.experiment_name IN ('encdec-budget-full-v0', 'encdec-smoke')
  AND ps.fair_order_seed = 'legacy-v0-migration';
```

Discover distinct legacy experiment names in v0:

```sql
SELECT DISTINCT experiment_name
FROM dr_dspy_encdec_eval_predictions
WHERE generation_status IN ('generated', 'generation_error')
ORDER BY experiment_name;
```

Experiments touched by the backfill CLI:

```sql
SELECT experiment_name, config_metadata
FROM dr_dspy_experiments
WHERE config_metadata->>'source' = 'v0_encdec_backfill';
```

---

## Node attempts and score attempts

```sql
-- Node attempts for backfilled runs
SELECT na.*
FROM dr_dspy_node_attempts na
JOIN dr_dspy_generation_runs gr ON gr.generation_run_id = na.generation_run_id
WHERE gr.summary->'metadata' ? 'v0_source';

-- Score attempts (after rescore) for backfilled runs
SELECT sa.*
FROM dr_dspy_score_attempts sa
JOIN dr_dspy_generation_runs gr ON gr.generation_run_id = sa.generation_run_id
WHERE gr.summary->'metadata' ? 'v0_source';
```

Pass-rate spot check on rescored backfill:

```sql
SELECT
  ps.experiment_name,
  COUNT(*) AS score_attempts,
  COUNT(*) FILTER (
    WHERE (sa.metrics->'custom'->'evaluation'->>'passed')::boolean IS TRUE
  ) AS evaluation_passed
FROM dr_dspy_score_attempts sa
JOIN dr_dspy_generation_runs gr ON gr.generation_run_id = sa.generation_run_id
JOIN dr_dspy_prediction_specs ps ON ps.prediction_id = gr.prediction_id
WHERE gr.summary->'metadata' ? 'v0_source'
GROUP BY ps.experiment_name
ORDER BY ps.experiment_name;
```

---

## Cross-check against v0 source table

Join migrated runs back to the legacy row:

```sql
SELECT
  v0.prediction_id AS v0_pk,
  v0.experiment_name AS v0_experiment,
  v0.generation_status AS v0_gen_status,
  gr.generation_run_id,
  ps.prediction_id AS v1_prediction_id,
  ps.experiment_name AS v1_experiment
FROM dr_dspy_encdec_eval_predictions v0
JOIN dr_dspy_generation_runs gr
  ON gr.summary->'metadata'->'v0_source'->>'v0_prediction_id' = v0.prediction_id
JOIN dr_dspy_prediction_specs ps ON ps.prediction_id = gr.prediction_id
WHERE v0.generation_status IN ('generated', 'generation_error');
```

---

## Quick counts

```sql
SELECT COUNT(*) AS backfilled_runs
FROM dr_dspy_generation_runs
WHERE summary->'metadata' ? 'v0_source';

SELECT COUNT(*) AS backfilled_encdec_specs
FROM dr_dspy_prediction_specs
WHERE fair_order_seed = 'legacy-v0-migration'
  AND graph_layout = 'encdec';
```

After a full enc-dec backfill, expect these counts to match terminal v0 enc-dec rows (dry-run `selected_v0_rows`; ~54k on the live DB as of 2026-06-30).

---

## Practical recommendation

| Goal | Filter |
|------|--------|
| Run-level inspection / joins | `summary->'metadata' ? 'v0_source'` on `dr_dspy_generation_runs` |
| Spec-level inspection | `fair_order_seed = 'legacy-v0-migration'` AND `graph_layout = 'encdec'` |
| Smoke-only subset | `experiment_name LIKE 'v0_encdec_backfill_smoke_%'` |
| v0 ↔ v1 reconciliation | Join on `v0_source.v0_prediction_id` = `dr_dspy_encdec_eval_predictions.prediction_id` |

Prefer **`v0_source` + `fair_order_seed`** over experiment `config_metadata` alone — experiment rows may pre-exist from other work and backfill inserts are idempotent (`ON CONFLICT DO NOTHING`).
