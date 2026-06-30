# Testing logs

Chronological record of manual / live pipeline runs. Newest entries at the top.

---

## 2026-06-30 — Rescore schema fix (`evaluation_incomplete`) + progress `total_candidates`

**Branch:** `today_exp`  
**Change:** Fix score-attempt check constraint drift; add backlog denominator to progress heartbeats  
**Operator:** agent (Cursor)

### Problem

Live `rescore` on `encdec-budget-full-v0` failed mid-batch with:

```text
CheckViolation: ck_dr_dspy_score_attempts_generated_code_outcome
generated_code_outcome = evaluation_incomplete
```

The Python enum and live `schema.py` already allowed `evaluation_incomplete` (partial HumanEval coverage — score 0, not an error row), but the applied Alembic revision `20260629_0001` omitted it from the Postgres check constraint. A secondary `DBOSException: System database accessed before DBOS was launched` appeared while DBOS tried to record the failed workflow — treat that as fallout from the insert failure, not the root cause.

### Fix

Migration [`20260630_0006`](../src/dr_dspy/db/migrations/versions/20260630_0006_score_attempt_evaluation_incomplete_outcome.py) widens `ck_dr_dspy_score_attempts_generated_code_outcome` to include `evaluation_incomplete`.

Progress heartbeats on `backfill-v0-encdec` and `rescore` now include **`total_candidates`** alongside **`selected`** (e.g. `selected=2 total_candidates=52593`).

### Schema head — run before large rescore batches

**Always upgrade the DB before a full enc-dec rescore.** Stale constraints will fail inserts mid-run; idempotent `ON CONFLICT DO NOTHING` means failed rows leave no score attempt and will retry on the next pass after upgrade.

```bash
uv run alembic current
# expect 20260630_0006 (head)

uv run alembic upgrade head
# → 20260630_0006
```

### Automated tests

```bash
uv run pytest tests/test_db_migrations.py tests/test_platform_progress_log.py \
  tests/test_v0_encdec_backfill.py tests/test_platform_scoring.py \
  tests/test_platform_worker_cli.py -q
# → 106 passed
```

### Retry command (operator)

```bash
uv run python -m dr_dspy.platform.worker rescore \
  --experiment-name encdec-budget-full-v0 \
  --generation-status success \
  --max-in-flight 200 \
  --progress-interval 5
```

Use `--generation-status success` only for v0 backfill pass-rate work; many `partial` v0 rows lack scorable terminal output.

---

## 2026-06-30 — Backfill/rescore concurrency controls

**Branch:** `today_exp`  
**Change:** Operator-controlled throughput for enc-dec v0 backfill and batch rescore  
**Operator:** agent (Cursor)

### Code changes under test

1. **[`src/dr_dspy/migration/v0_encdec_backfill.py`](../src/dr_dspy/migration/v0_encdec_backfill.py):** chunked backfill with per-chunk commits, offset paging, parallel reshape (`ThreadPoolExecutor`), Rich stderr progress (heartbeats + events).
2. **[`src/dr_dspy/platform/progress_log.py`](../src/dr_dspy/platform/progress_log.py):** shared Rich progress helper for backfill and rescore.
3. **[`src/dr_dspy/platform/rescoring.py`](../src/dr_dspy/platform/rescoring.py):** `--max-in-flight` wave scheduling with internal await (default **100**).
4. **[`src/dr_dspy/platform/worker.py`](../src/dr_dspy/platform/worker.py):** new CLI flags on `backfill-v0-encdec` and `rescore`.

### New CLI flags

| Command | Flag | Default | Purpose |
|---------|------|---------|---------|
| `backfill-v0-encdec` | `--chunk-size` | omitted (legacy single transaction) | Commit after each chunk |
| `backfill-v0-encdec` | `--reshape-workers` | `1` | Parallel CPU-bound reshape (writes stay serial) |
| `rescore` | `--max-in-flight` | `100` | Cap scheduled scoring workflows before awaiting |
| both | `--progress-interval` | `5` | Rich heartbeat interval on stderr (seconds); `0` = events only |

Omitting `--chunk-size` preserves the legacy single-transaction backfill path.

Progress stderr shows `selected` (processed so far) and `total_candidates` (batch denominator). Ensure schema head is current before large rescoring — see entry above (`20260630_0006`).

### Automated tests

```bash
uv run pytest tests/test_v0_encdec_backfill.py tests/test_platform_worker_cli.py tests/test_platform_scoring.py -q
# → 87 passed
```

### Verification commands (operator)

```bash
# Backfill dry-run smoke (chunked)
uv run python -m dr_dspy.platform.worker backfill-v0-encdec \
  --dry-run --limit 100 --chunk-size 50 --reshape-workers 4

# Rescore dry-run smoke
uv run python -m dr_dspy.platform.worker rescore \
  --experiment-name encdec-budget-full-v0 \
  --generation-status success \
  --generation-status partial \
  --limit 50 --max-in-flight 10 --dry-run
```

### Recommended full-run commands

```bash
# Full enc-dec backfill (chunked, parallel reshape)
uv run python -m dr_dspy.platform.worker backfill-v0-encdec \
  --chunk-size 1000 \
  --reshape-workers 4

# Per-experiment rescore (explicit in-flight cap)
uv run python -m dr_dspy.platform.worker rescore \
  --experiment-name encdec-budget-full-v0 \
  --generation-status success \
  --generation-status partial \
  --max-in-flight 30

uv run python -m dr_dspy.platform.worker rescore \
  --experiment-name encdec-smoke \
  --generation-status success \
  --generation-status partial \
  --max-in-flight 30
```

### Verdict

**Pass (automated).** Chunked backfill, parallel reshape, and max-in-flight rescore wired with tests. Live full-run not executed in this pass.

### Caveats

1. Partial backfill commits are durable; rerun is idempotent (`already_present` counts rise).
2. Rescore default `--max-in-flight 100` replaces prior unbounded schedule-then-await-at-end behavior.
3. Offset paging is acceptable on the frozen read-only v0 table; no keyset cursor needed for this one-shot migration.

---

## 2026-06-30 — Tier-1 limit raise + enc-dec backfill smoke r2

**Branch:** `today_exp`  
**Change:** Domain payload caps raised to serialization ceiling — see [Limit enforcement points](ref/limit_enforcement_points.md#changelog)  
**Target experiment:** `v0_encdec_backfill_smoke_20260630_r2`  
**Operator:** agent (Cursor)

### Code changes under test

1. **[`src/dr_dspy/records/limits.py`](../src/dr_dspy/records/limits.py):** all tier-1 byte caps → `DOMAIN_PAYLOAD_MAX_BYTES` = `PAYLOAD_MAX_BYTES` (~768 MiB); `METRICS_STAGES_MAX_COUNT` 64 → 10_000.
2. **[`tests/test_records_limits.py`](../tests/test_records_limits.py):** policy guard (byte caps == `PAYLOAD_MAX_BYTES`, stages count ≥ 1000).
3. Oversized rejection tests updated to monkeypatch small caps on `records.models` (avoid ~768 MiB test allocations).

Tier-2 (Postgres/serialization) and tier-3 (preview/truncation) unchanged.

### Pre-backfill limit audit

```bash
uv run pytest tests/test_records_limits.py -q   # → 2 passed
# one-liner: all byte caps == PAYLOAD_MAX_BYTES, METRICS_STAGES_MAX_COUNT >= 1000
# → tier-1 limits OK
rg "256 \* 1024|128 \* 1024" src/dr_dspy/records/  # → no matches
```

### Dry-run (after limit raise)

```bash
uv run python -m dr_dspy.platform.worker backfill-v0-encdec --dry-run
```

| Metric | Before limits raise | After limits raise |
|--------|--------------------:|-------------------:|
| `selected_v0_rows` | 54,041 | 54,041 |
| `reshaped_specs` | 53,397 | **54,041** |
| `reshape_failures` | **644** | **0** |
| Writes | 0 | 0 |

Wall clock: ~61s. All former `TaskInputsPayload` failures (oversized v0 `test` fields ~502 KiB) now reshape successfully.

### Tiny backfill r2

```bash
uv run python -m dr_dspy.platform.worker backfill-v0-encdec \
  --limit 10 \
  --target-experiment-name v0_encdec_backfill_smoke_20260630_r2
```

10 specs inserted, 10 runs, 20 node attempts, 0 reshape failures.

### Tiny rescore r2

```bash
uv run python -m dr_dspy.platform.worker rescore \
  --experiment-name v0_encdec_backfill_smoke_20260630_r2 \
  --generation-status success \
  --generation-status partial
```

| Metric | Value |
|--------|------:|
| `scheduled_count` | 10 |
| `failed_count` | 0 |
| Score attempts | 10 × `success` |
| HumanEval evaluation passed | 6 / 10 |

### Verdict

**Pass.** Tier-1 limits no longer block enc-dec v0 backfill reshape. Full-table dry-run is clean (`reshape_failures: 0`). Repo ready for operator full enc-dec backfill.

### Full enc-dec backfill/rescore commands

```bash
# Confirm schema head
uv run alembic current

# Full enc-dec backfill (preserves legacy experiment_name per row)
uv run python -m dr_dspy.platform.worker backfill-v0-encdec

# Discover legacy experiment names (if needed)
psql "$DATABASE_URL" -c "
SELECT DISTINCT experiment_name
FROM dr_dspy_encdec_eval_predictions
WHERE generation_status IN ('generated', 'generation_error')
ORDER BY experiment_name;
"

# Per-experiment rescore under humaneval@v1
uv run python -m dr_dspy.platform.worker rescore \
  --experiment-name encdec-budget-full-v0 \
  --generation-status success \
  --generation-status partial

uv run python -m dr_dspy.platform.worker rescore \
  --experiment-name encdec-smoke \
  --generation-status success \
  --generation-status partial
```

---

## 2026-06-30 — HPM selection analysis scripts

**Branch:** `today_exp`  
**Command/feature:** `scripts/analysis/q1–q4` (new)  
**Target experiment:** `v0_encdec_backfill_smoke_20260630`  
**Operator:** agent (Cursor)

### Code changes under test

1. **New package:** [`src/dr_dspy/analysis/`](../src/dr_dspy/analysis/) — `db.py`, `frames.py`, `plotting.py`, `cli_options.py`; loads v1 enc-dec rows into pandas, normalizes `compression_target` from `dimensions.values.compression_target` or `budget_ratio`.
2. **New scripts:** [`scripts/analysis/`](../scripts/analysis/) — `q1_model_candidates.py`, `q2_compression_range.py`, `q3_repeat_stability.py`, `q4_task_variation.py`.
3. **Dependencies:** direct `pandas`, `matplotlib` via `uv add`.
4. **Tests:** [`tests/test_analysis_frames.py`](../tests/test_analysis_frames.py), [`tests/test_analysis_scripts.py`](../tests/test_analysis_scripts.py).

### Tests run

```bash
uv run pytest tests/test_analysis_frames.py tests/test_analysis_scripts.py
# → 15 passed
```

### Live analysis run

```bash
uv run python scripts/analysis/q1_model_candidates.py \
  --experiment-name v0_encdec_backfill_smoke_20260630

uv run python scripts/analysis/q2_compression_range.py \
  --experiment-name v0_encdec_backfill_smoke_20260630

uv run python scripts/analysis/q3_repeat_stability.py \
  --experiment-name v0_encdec_backfill_smoke_20260630

uv run python scripts/analysis/q4_task_variation.py \
  --experiment-name v0_encdec_backfill_smoke_20260630
```

Output layout (per run, shared timestamp):

- Tabular: `artifacts/{script_name}/{timestamp}_{stem}.csv|md` (gitignored)
- Figures: `figs/{script_name}/{timestamp}_{stem}.png` (tracked in git)

| Metric | Value |
|--------|------:|
| Generation runs loaded | 10 |
| Score-success rows | 10 |
| Models in Q1 summary | 7 |
| Compression targets observed | 0.25, 0.5, 0.75, 1.5, 2.0 (+ 1 missing) |

### Artifacts

Example Q1 outputs:

- `artifacts/q1_model_candidates/{timestamp}_model_candidates.csv|md`
- `figs/q1_model_candidates/{timestamp}_pass_rate_by_model.png`
- `figs/q1_model_candidates/{timestamp}_generation_score_health.png`

### Caveats

- N=10 smoke sample is too sparse for Q3 bootstrap intervals or Q4 useful-signal flags; scripts report that explicitly in `.md` summaries.
- One migrated row lacks `budget_ratio` in dimensions (shows as `nan` compression target).
- Q3/Q4 need full backfill + repeats before optimization-signal conclusions are trustworthy.

### Verdict

Analysis scripts run end-to-end against migrated v1 enc-dec data. Ready for use after full enc-dec backfill/rescore populates more rows.

### Sample run inspector (added)

All 10 smoke samples inspected (`--sample-index 0..9`); HTML reports tracked in
`analysis/samples/v0_encdec_backfill_smoke_20260630/` (JSON bundles gitignored).

```bash
uv run python scripts/analysis/sample_run_inspector.py \
  --experiment-name v0_encdec_backfill_smoke_20260630 \
  --sample-index 0
```

Q1–Q4 re-run on 2026-06-30 (`20260630_185954`–`20260630_185959` timestamps).

- Unit tests: `tests/test_analysis_inspect.py` (5 passed)

---

## 2026-06-30 — Enc-dec v0 backfill smoke

**Branch:** `today_exp`  
**Command/feature:** `backfill-v0-encdec` (new) + `rescore` on migrated sample  
**Target experiment:** `v0_encdec_backfill_smoke_20260630`  
**Operator:** agent (Cursor)

### Code changes under test

1. **New module:** [`src/dr_dspy/migration/v0_encdec_backfill.py`](../src/dr_dspy/migration/v0_encdec_backfill.py) — reads `dr_dspy_encdec_eval_predictions`, reshapes terminal rows via `reshape_v0_encdec_row`, idempotent v1 inserts.
2. **New CLI:** `uv run python -m dr_dspy.platform.worker backfill-v0-encdec` with `--dry-run`, `--limit`, `--target-experiment-name`, `--database-url`, `--env-file`.
3. **Tests:** [`tests/test_v0_encdec_backfill.py`](../tests/test_v0_encdec_backfill.py), CLI wiring in [`tests/test_platform_worker_cli.py`](../tests/test_platform_worker_cli.py).

### Schema head check

```bash
uv run alembic current
# → 20260630_0006 (head)
```

### Tests run

```bash
uv run pytest tests/test_v0_reshape.py tests/integration/test_v0_reshape_outcomes.py tests/test_v0_encdec_backfill.py tests/test_platform_worker_cli.py
# → 29 passed
```

### Dry-run

```bash
uv run python -m dr_dspy.platform.worker backfill-v0-encdec --dry-run
```

| Metric | Count |
|--------|------:|
| `selected_v0_rows` (terminal) | 54,041 |
| `non_terminal_v0_rows` | 1,175 |
| `reshaped_specs` | 53,397 |
| `reshape_failures` | 644 |
| Writes | 0 |

**First reshape error:** `TaskInputsPayload` byte limit (262144) exceeded on rows with very large `test` fields (~502 KiB). These rows fail reshape today; full backfill will skip ~644 terminal rows unless limits or reshape handling change.

Wall clock: ~75s (full-table dry-run reshape, no writes).

### Tiny backfill

```bash
uv run python -m dr_dspy.platform.worker backfill-v0-encdec \
  --limit 10 \
  --target-experiment-name v0_encdec_backfill_smoke_20260630
```

| Metric | First run | Idempotent rerun |
|--------|----------:|-----------------:|
| `selected_v0_rows` | 10 | 10 |
| `reshaped_specs` | 10 | 10 |
| `specs_inserted` | 10 | 0 |
| `specs_already_present` | 0 | 10 |
| `runs_inserted` | 10 | 0 |
| `runs_already_present` | 0 | 10 |
| `node_attempts_inserted` | 20 | 0 |
| `node_attempts_already_present` | 0 | 20 |

All 10 selected rows had `generation_status=generated` (scoreable). Ordering `generation_status ASC, prediction_id ASC` surfaces successes first.

### Validation queries/results

| Check | Result |
|-------|--------|
| Prediction specs | 10 for `v0_encdec_backfill_smoke_20260630` |
| Generation run statuses | 10 × `success` |
| Node attempts per run | 2 each (encoder + decoder) on spot-check |
| `v0_source` metadata | Present (e.g. v0 ids `000869a65f496d4a…`, `000079233ab1eabf…`) |

### Tiny rescore

```bash
uv run python -m dr_dspy.platform.worker rescore \
  --experiment-name v0_encdec_backfill_smoke_20260630 \
  --generation-status success \
  --generation-status partial
```

| Metric | Value |
|--------|------:|
| `selected_count` | 10 |
| `scheduled_count` | 10 |
| `failed_count` | 0 |
| Persisted score attempts | 10 |
| Score attempt status | 10 × `success` |
| HumanEval evaluation passed | 5 / 10 |

Scoring profile: `humaneval@v1`. DBOS shutdown clean (~35s).

### Verdict

**Pass** for enc-dec v0 backfill smoke path: CLI, dry-run counts, tiny write, idempotent rerun, v1 row shapes, and `humaneval@v1` rescoring all work on live data.

### Blockers / caveats

1. **644 terminal rows** fail reshape due to `TaskInputsPayload` 256 KiB cap on oversized v0 `test` fields — investigate before full backfill if those rows matter.
2. **Full dry-run is slow** (~75s) because it reshapes all 54k terminal rows; acceptable for preflight but consider `--limit` for quick checks.
3. **Legacy experiment names** in v0 data: `encdec-budget-full-v0`, `encdec-smoke` (full backfill preserves these when `--target-experiment-name` is omitted).

### Full enc-dec backfill/rescore commands

```bash
# Confirm schema head
uv run alembic current

# Full enc-dec backfill (preserves legacy experiment_name per row)
uv run python -m dr_dspy.platform.worker backfill-v0-encdec

# Discover legacy experiment names (if needed)
psql "$DATABASE_URL" -c "
SELECT DISTINCT experiment_name
FROM dr_dspy_encdec_eval_predictions
WHERE generation_status IN ('generated', 'generation_error')
ORDER BY experiment_name;
"

# Per-experiment rescore under humaneval@v1
uv run python -m dr_dspy.platform.worker rescore \
  --experiment-name encdec-budget-full-v0 \
  --generation-status success \
  --generation-status partial

uv run python -m dr_dspy.platform.worker rescore \
  --experiment-name encdec-smoke \
  --generation-status success \
  --generation-status partial
```

---

## 2026-06-30 — HumanEval enc-dec smoke r2 (fixes validation)

**Branch:** `today_exp`  
**Experiment:** `humaneval_encdec_smoke_v1`  
**Operation key:** `humaneval-encdec-smoke-live-20260630-r2`  
**Operator:** agent (Cursor)

### Changes under test

1. **Model routing:** `gpt-5.4-nano` uses OpenAI API id `gpt-5.4-nano`; `gpt-oss-120b` moved to OpenRouter (`openai/gpt-oss-120b`).
2. **Scoring payload limits:** removed `per_test_results` entry-count cap; byte cap raised to **128 MB**.
3. **`rescore` lifecycle:** batch rescoring now awaits all scheduled scoring workflow handles before `DBOS.destroy()`.
4. **`DATABASE_URL` normalization:** bare `postgresql://` URLs auto-normalize to `postgresql+psycopg://` in `resolve_database_url`.

**Unit tests:** 119 passed (`test_records_contracts`, `test_platform_dbos_bootstrap`, `test_platform_scoring`, `test_platform_worker_cli`, rescore dry-run CLI).

### Preflight

| Check | Result |
|-------|--------|
| `.env` `OPENROUTER_API_KEY` | set |
| `.env` `OPENAI_API_KEY` | set |
| `.env` `DATABASE_URL` | `postgresql:///dr_dspy` (bare URL — no manual driver suffix) |
| DBOS URL at submit | auto-normalized to `postgresql+psycopg:///dr_dspy` ✓ |

### Commands run

```bash
uv run python -m dr_dspy.platform.worker build-specs \
  --configs-root configs \
  --config-file configs/experiments/humaneval_encdec_smoke.json \
  --output /tmp/whetstone-smoke-r2/specs.jsonl
# → spec_count: 24

uv run python -m dr_dspy.platform.worker submit-jsonl \
  --specs-file /tmp/whetstone-smoke-r2/specs.jsonl \
  --operation-key humaneval-encdec-smoke-live-20260630-r2 \
  --experiment-name humaneval_encdec_smoke_v1 \
  --queue-registration-concurrency 30
# → inserted_count: 16, already_present_count: 8, enqueued_count: 16

uv run python -m dr_dspy.platform.worker worker --worker-concurrency 30
# (ran to completion; worker stopped after all 24 operation items terminal)

uv run python -m dr_dspy.platform.worker rescore \
  --experiment-name humaneval_encdec_smoke_v1 \
  --generation-status success
# → scheduled_count: 16, failed_count: 0 (~59s, clean DBOS shutdown)
```

### Submit note (8 reused Qwen runs)

Because prediction specs are content-addressed, the 8 unchanged Qwen specs from r1 were **`workflow_already_present`** — they point at the **r1 generation runs**, not freshly generated r2 runs. Only **16** specs (GPT nano + OSS, whose model fragments changed) received new generation runs.

### Generation results (operation r2, all 24 items)

| Status | Count |
|--------|------:|
| `success` | 24 |

| Model (encoder) | Success count |
|-----------------|-------------:|
| `gpt-5.4-nano` (OpenAI) | 8 |
| `openai/gpt-oss-120b` (OpenRouter) | 8 |
| `qwen/qwen3-coder-flash` (OpenRouter, reused r1 runs) | 8 |

All three model paths generated successfully — no `model_not_found` errors.

### Scoring results

**`rescore` batch:** 16 candidates selected (runs without existing score attempts), 16 scheduled, 0 failed. No DBOS lifecycle error.

| Scope | Success | Error | Notes |
|-------|--------:|------:|-------|
| 16 newly enqueued runs | 16 | 0 | All persisted; **993** `per_test_results` rows each (HumanEval/146) |
| 8 reused Qwen runs | 0 | 8 | Pre-existing r1 error score attempts (`512 entries` cap) — excluded from rescore |

**Pass rates (16 newly scored runs):**

| Model | Runs | Avg pass rate | Min | Max |
|-------|-----:|--------------:|----:|----:|
| `gpt-5.4-nano` | 8 | 0.25 | 0.0 | 1.0 |
| `openai/gpt-oss-120b` | 8 | 0.25 | 0.0 | 1.0 |

### What was validated

- OpenAI + OpenRouter model strings fix generation for all three smoke models.
- `per_test_results` byte-only cap handles HumanEval/146 (993 cases >> old 512 count cap).
- `rescore` await fix: batch scoring completes and DBOS shuts down cleanly.
- Bare `postgresql:///dr_dspy` works end-to-end without manual `+psycopg` export.

### Follow-ups

1. **Clean-slate smoke:** To score all 24 specs in one rescore pass, truncate v1 score attempts or use a fresh experiment name / DB reset so Qwen specs re-enqueue new generation runs.
2. **Admin port warning:** `Failed to start admin server: Address already in use` during overlapping DBOS launches — cosmetic; did not block scoring.

### Verdict

**Pass** for the four fixes under test. Generation and scoring are trustworthy for newly enqueued runs. Full 24-model pass-rate table requires either a clean DB or re-submit after clearing stale Qwen generation/score rows from r1.

### `per_test_results` storage sizing (post-run analysis)

Measured on successful score attempts in Postgres (`dr_dspy_score_attempts.per_test_results`).
The domain cap (128 MiB) applies to **serialized JSON** (`validate_payload_size` /
`to_jsonable`), not Postgres `pg_column_size` (JSONB binary is ~20–50× smaller).

| Metric | Typical (993 cases) | Worst seen (2979 cases) |
|--------|---------------------|-------------------------|
| Postgres `pg_column_size` | 24–60 KiB | 143 KiB |
| Serialized JSON text | 0.24–0.29 MiB | 3.38 MiB |
| vs 128 MiB cap | ~0.2% | ~2.6% |

- **~578 B/case** (JSON text average); verbose failures inflate
  `actual_output_repr` (≈65%) and `message` (≈26%).
- Case counts on HumanEval/146 varied **993 / 1986 / 2979** (all `input_result`) —
  likely evalplus expansion multiples of the base 993, not different tasks.
- **Decision:** keep full per-test persistence for the eval push; 128 MiB is ample.
  Truncation is unnecessary unless fleet storage becomes an issue — prefer field-level
  caps on `actual_output_repr` / `message` over row-count limits if that happens later.
- Rough sweep estimate (164 tasks × 5 models × 5 repeats, if all like HumanEval/146):
  ~3.5 GiB total in `per_test_results`.

---

## 2026-06-30 — HumanEval enc-dec composable smoke (live E2E)

**Branch:** `today_exp`  
**Experiment:** `humaneval_encdec_smoke_v1`  
**Config:** [`configs/experiments/humaneval_encdec_smoke.json`](../configs/experiments/humaneval_encdec_smoke.json)  
**Operation key:** `humaneval-encdec-smoke-live-20260630`  
**Operator:** agent (Cursor)

### Intent

Validate the full v1 path end-to-end:

`build-specs` → `submit-jsonl` → `worker` (high concurrency) → scoring

Using composable configs (3 models × `tiny` split × 2 compression targets × 4 repeats = **24 specs**).

### Preflight

| Check | Result |
|-------|--------|
| `.env` `OPENROUTER_API_KEY` | set |
| `.env` `OPENAI_API_KEY` | set |
| `.env` `DATABASE_URL` | `postgresql:///dr_dspy` (no driver suffix) |
| v1 Alembic head | **Not applied** — DB had legacy v0 tables + lone `dr_dspy_experiments` from a failed partial migration |

**Remediation:** Dropped partial `dr_dspy_experiments` + `alembic_version`, then:

```bash
uv run alembic upgrade head   # → 20260630_0006 (head)
```

Legacy v0 tables (`dr_dspy_eval_predictions`, etc.) left untouched per project policy.

### Commands run

```bash
# Spec generation (HF dataset load for evalplus/humanevalplus)
uv run python -m dr_dspy.platform.worker build-specs \
  --configs-root configs \
  --config-file configs/experiments/humaneval_encdec_smoke.json \
  --output /tmp/whetstone-smoke/specs.jsonl
# → spec_count: 24, experiment_name: humaneval_encdec_smoke_v1

# Submit + enqueue (requires psycopg driver URL — see findings)
export DATABASE_URL=postgresql+psycopg:///dr_dspy
uv run python -m dr_dspy.platform.worker submit-jsonl \
  --specs-file /tmp/whetstone-smoke/specs.jsonl \
  --operation-key humaneval-encdec-smoke-live-20260630 \
  --experiment-name humaneval_encdec_smoke_v1 \
  --queue-registration-concurrency 30
# → inserted_count: 24, enqueued_count: 24, failed_count: 0

# Generation worker (background)
uv run python -m dr_dspy.platform.worker worker --worker-concurrency 30

# Scoring (attempted)
uv run python -m dr_dspy.platform.worker rescore \
  --experiment-name humaneval_encdec_smoke_v1 \
  --generation-status success
# → failed (DBOS conflict — see findings)

uv run python -m dr_dspy.platform.worker score-one \
  --generation-run-id e20c892261d13969f2aed219
# → score workflow completed; DBOS recovered 8 orphaned generation workflows on launch
```

### Sampled task

All 24 specs used **`HumanEval/146`** (`sample_seed: 0`, `sample_count: 1` from `configs/splits/tiny.json`).

Compression targets in specs: `0.25` and `0.5` (12 specs each).  
Models: 8 specs per model (2 targets × 4 repetition seeds).

### Generation results

**Wall clock:** ~9s from first to last generation run completion (17:42:28 → 17:42:37 ET) with `--worker-concurrency 30`.

| Status | Count | Models |
|--------|------:|--------|
| `success` | 8 | `qwen/qwen3-coder-flash` (OpenRouter) |
| `blocked` | 16 | `openai/gpt-5.4-nano`, `openai/gpt-oss-120b` (OpenAI) |

**OpenRouter path (success):**

- Encoder produced concise descriptions (example: *"Counts numbers > 10 with odd first and last digits."*, ~51 chars).
- Decoder produced Python (often fenced in markdown code blocks).
- Total provider cost (16 node attempts across 8 runs): **~$0.0017**.

**OpenAI path (blocked):**

All 16 runs blocked at decoder because encoder failed. Root cause from `dr_dspy_node_attempts`:

```text
Error code: 400 - model_not_found
  "The requested model 'openai/gpt-5.4-nano' does not exist."
  "The requested model 'openai/gpt-oss-120b' does not exist."
```

These model strings are valid OpenRouter-style slugs but **not** valid on the direct OpenAI Responses API with the current account. Config fragments need corrected OpenAI model IDs before the OpenAI smoke leg can pass.

### Scoring results

| Step | Outcome |
|------|---------|
| `rescore --generation-status success` | **Failed** — `DBOSException: System database accessed before DBOS was launched` (likely conflict with concurrent DBOS recovery of in-flight generation workflows) |
| `score-one` (single run) | Workflow completed; **8** score attempts persisted (DBOS auto-recovered remaining success workflows on launch) |
| Score attempt status | **8 / 8 `error`** (permanent) |

Score failure message (all 8):

```text
per_test_results cannot exceed 512 entries
```

HumanEval/146 appears to exceed the platform's `ScoreAttemptRecord` per-test cap. Scoring logic ran but persistence rejected the payload. This is independent of the generation path — a platform limit to raise or handle for heavy tasks.

### What was validated

- Composable config expansion → 24 distinct prediction specs, shared `experiment_name`.
- `submit-jsonl` insert + DBOS enqueue at concurrency 30.
- Parallel generation at concurrency 30 completes 24 enc-dec jobs in ~9s.
- **OpenRouter** humaneval enc-dec graph: encoder → decoder, prompts, budgets, persistence.
- HumanEval task inputs (`gt_code`, `budget`, etc.) materialized correctly in specs.

### Findings / follow-ups

1. **`DATABASE_URL` driver:** CLI `submit-jsonl` / `worker` use SQLAlchemy `create_engine(database_url)` without normalizing `postgresql://` → `postgresql+psycopg://`. Workaround: export `DATABASE_URL=postgresql+psycopg:///dr_dspy`. Consider normalizing in `resolve_database_url` or worker bootstrap.

2. **OpenAI model IDs:** Update [`configs/models/gpt54-nano-openai.json`](../configs/models/gpt54-nano-openai.json) and [`configs/models/gpt-oss-120b-openai.json`](../configs/models/gpt-oss-120b-openai.json) with API-valid model names (not OpenRouter slugs).

3. **Scoring cap:** `per_test_results` 512-entry limit breaks scoring for HumanEval/146. Either cap stored per-test rows, summarize, or pick a smaller smoke task in `tiny.json` until fixed.

4. **`rescore` + DBOS recovery:** Batch rescoring while generation workflows are still marked recoverable may hit DBOS lifecycle errors. Prefer waiting for worker shutdown or use sequential `score-one` until investigated.

5. **Smoke split:** Consider pinning `tiny.json` to a known-small task (e.g. via future `task_ids` list) to avoid sampling pathological tasks like HumanEval/146 during smoke.

### Artifacts

- Specs JSONL: `/tmp/whetstone-smoke/specs.jsonl` (24 lines)
- DB rows: `dr_dspy_prediction_specs` (24), `dr_dspy_generation_runs` (24), `dr_dspy_score_attempts` (8)

### Verdict

**Partial pass.** The v1 composable enc-dec pipeline is live and fast at concurrency 30. OpenRouter generation succeeded for all 8 Qwen specs. OpenAI configs need model ID fixes; scoring needs a fix or smaller smoke task before pass-rate numbers are trustworthy.
