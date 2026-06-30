# Testing logs

Chronological record of manual / live pipeline runs. Newest entries at the top.

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
uv run alembic upgrade head   # → 20260630_0005 (head)
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
