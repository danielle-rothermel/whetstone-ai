# Limit enforcement points

Reference map of payload-size and related limits in the codebase. Tier-1 domain caps are aligned with the serialization ceiling (~768 MiB) for the June 30 eval push — catastrophe guards only, not experiment policy.

---

## Three tiers

| Tier | Purpose | Where defined | Typical role |
|------|---------|---------------|--------------|
| **1 — Domain payload caps** | Catastrophe guards on v1 Pydantic models (aligned with tier 2) | [`src/dr_dspy/records/limits.py`](../../src/dr_dspy/records/limits.py) | Fail fast at record construction before Postgres insert |
| **2 — Postgres / serialization ceiling** | Last-resort guard before JSONB insert | [`src/dr_dspy/serialization.py`](../../src/dr_dspy/serialization.py) | Catch pathological blobs (~768 MiB) |
| **3 — Truncation / preview** | Error messages and debug output only | [`src/dr_dspy/serialization.py`](../../src/dr_dspy/serialization.py), [`src/dr_dspy/lm/boundary.py`](../../src/dr_dspy/lm/boundary.py) | Not persistence validators |

---

## Tier 1 — Domain payload caps

### Constants ([`limits.py`](../../src/dr_dspy/records/limits.py))

| Constant | Current value | Applies to |
|----------|--------------:|------------|
| `DOMAIN_PAYLOAD_MAX_BYTES` | ~768 MiB (`PAYLOAD_MAX_BYTES`) | Alias for all tier-1 byte caps |
| `TASK_INPUTS_MAX_BYTES` | ~768 MiB | `TaskInputsPayload.values` (`prompt`, `test`, `entry_point`, …) |
| `GRAPH_SNAPSHOT_MAX_BYTES` | ~768 MiB | `GraphSnapshotPayload` (serialized graph) |
| `BATCH_SUBMIT_SPEC_MAX_BYTES` | ~768 MiB | `BatchSubmitOperationRecord.spec` |
| `PROVIDER_TELEMETRY_MAX_BYTES` | ~768 MiB | `UsageCostPayload.usage_metadata`, `ResponseMetadataPayload.response_metadata` |
| `NODE_OUTPUT_MAX_BYTES` | ~768 MiB | `NodeOutputPayload` (`values` + `metadata`) |
| `METRICS_MAX_BYTES` | ~768 MiB | `ScoreAttemptRecord.metrics` |
| `PER_TEST_RESULTS_MAX_BYTES` | ~768 MiB | `ScoreAttemptRecord.per_test_results` |
| `METRICS_STAGES_MAX_COUNT` | 10_000 | `ScoreAttemptRecord.metrics.stages` (entry count, not bytes) |

### Enforcement helper

- **`validate_payload_size(value, *, max_bytes, label)`** in [`limits.py`](../../src/dr_dspy/records/limits.py)
- Calls **`to_jsonable(value, max_bytes=...)`** from [`serialization.py`](../../src/dr_dspy/serialization.py)
- Raises **`ValueError`** with a label prefix on oversize payloads

### Model validators ([`models.py`](../../src/dr_dspy/records/models.py))

All domain caps are enforced via `@model_validator(mode="after")` on Pydantic models:

| Model | Field / scope | Constant |
|-------|---------------|----------|
| `TaskInputsPayload` | `values` | `TASK_INPUTS_MAX_BYTES` |
| `GraphSnapshotPayload` | full snapshot JSON | `GRAPH_SNAPSHOT_MAX_BYTES` |
| `UsageCostPayload` | `usage_metadata` | `PROVIDER_TELEMETRY_MAX_BYTES` |
| `ResponseMetadataPayload` | `response_metadata` | `PROVIDER_TELEMETRY_MAX_BYTES` |
| `NodeOutputPayload` | `values` + `metadata` | `NODE_OUTPUT_MAX_BYTES` |
| `ScoreAttemptRecord` | `per_test_results` | `PER_TEST_RESULTS_MAX_BYTES` |
| `ScoreAttemptRecord` | `metrics.stages` length | `METRICS_STAGES_MAX_COUNT` |
| `ScoreAttemptRecord` | `metrics` | `METRICS_MAX_BYTES` |
| `BatchSubmitOperationRecord` | `spec` | `BATCH_SUBMIT_SPEC_MAX_BYTES` |

**Summary:** 7 byte caps + 1 count cap, defined in **one file**, enforced at **8 validator sites** in **one model file**.

### Production construction sites for task inputs

Oversized task inputs fail when `TaskInputsPayload` is constructed:

| Path | File | Notes |
|------|------|-------|
| v0 backfill / reshape | [`src/dr_dspy/migration/v0_reshape.py`](../../src/dr_dspy/migration/v0_reshape.py) | Maps v0 `prompt`, `test`, `entry_point` into `values` |
| New v1 specs | [`src/dr_dspy/platform/spec_builder.py`](../../src/dr_dspy/platform/spec_builder.py) | HumanEval task materialization |

### Related count limit (same constant, different site)

| Site | File | Rule |
|------|------|------|
| Score metrics assembly | [`src/dr_dspy/platform/scoring.py`](../../src/dr_dspy/platform/scoring.py) | Node output metric sources capped at `METRICS_STAGES_MAX_COUNT - 1` before building `MetricsPayload` |

---

## Tier 2 — Postgres / serialization ceiling

Defined in [`serialization.py`](../../src/dr_dspy/serialization.py):

| Constant | Value | Role |
|----------|------:|------|
| `POSTGRES_JSONB_MAX_BYTES` | 1 GiB | PostgreSQL jsonb per-value reference |
| `PAYLOAD_MAX_BYTES` | ~768 MiB | Default max for `to_jsonable` / `ensure_recordable` (1 GiB minus 25% binary overhead headroom) |
| `MAX_JSONABLE_DEPTH` | 100 | JSON nesting depth guard |

### Enforcement

| Mechanism | File | When |
|-----------|------|------|
| `to_jsonable(..., max_bytes=PAYLOAD_MAX_BYTES)` | `serialization.py` | Default serialization path |
| `ensure_recordable(value)` | [`eval_failures/recording.py`](../../src/dr_dspy/eval_failures/recording.py) | Failure metadata and generic recordability |
| `_validate_jsonb_fields(row, *fields)` | [`db/io.py`](../../src/dr_dspy/db/io.py) | Every JSONB column on insert |

**Insert paths that call `_validate_jsonb_fields`:**

- `experiment_row` → `config_metadata`
- `prediction_spec_row` → `task_snapshot`, `graph_snapshot`, `dimensions`, `provider_configs`
- `generation_run_row` → `summary`
- `node_attempt_row` → `provider_config`, `output`, `usage_cost`, `response_metadata`, `failure`
- `score_attempt_row` → `extracted_code`, `metrics`, `per_test_results`, `failure`
- `batch_submit_operation_row` → `spec`, `metadata`
- `batch_submit_item_row` → `enqueue_metadata`, `failure`

Domain validators (tier 1) run **first** when building Pydantic records. Tier 2 runs again at DB write if the row reaches `db/io` without having been validated earlier.

---

## Tier 3 — Truncation / preview (non-persistence)

| Constant | Value | File | Use |
|----------|------:|------|-----|
| `MESSAGE_PREVIEW` | 512 chars | `serialization.py` | Short error/debug previews |
| `DEBUG_DETAIL_LIMIT` | 256 KiB | `serialization.py` | Debug detail truncation |
| `ENCODED_PREVIEW_SLICE` | 8192 | `serialization.py` | Head/tail of encoded blobs in errors |
| (inline) | 512 chars | `lm/boundary.py` | `response_preview` in provider logging |

These do **not** block record persistence.

---

## Historical notes

- **Enc-dec backfill smoke (2026-06-30, initial):** Full dry-run reshaped 53,397 / 54,041 terminal v0 rows; **644 failed** because v0 `test` fields (~502 KiB) exceeded the old 256 KiB `TASK_INPUTS_MAX_BYTES`. Fixed by tier-1 raise — see changelog below and [`docs/testing_logs.md`](../testing_logs.md).
- **`per_test_results`:** Previously had a 512-entry count cap; removed for eval push. Tier-1 byte cap now matches serialization ceiling.

---

## Changelog

### 2026-06-30 — Tier-1 raised to serialization ceiling (eval push)

- All domain byte caps (`TASK_INPUTS_*`, `GRAPH_SNAPSHOT_*`, `BATCH_SUBMIT_*`, `PROVIDER_TELEMETRY_*`, `NODE_OUTPUT_*`, `METRICS_*`, `PER_TEST_RESULTS_*`) raised from 128 KiB–128 MiB → `DOMAIN_PAYLOAD_MAX_BYTES` = `PAYLOAD_MAX_BYTES` (~768 MiB).
- `METRICS_STAGES_MAX_COUNT` raised from 64 → 10_000.
- Motivation: enc-dec v0 backfill dry-run had 644 reshape failures on ~502 KiB v0 `test` fields.
- Tier-2/3 unchanged.
- Policy guard: [`tests/test_records_limits.py`](../../tests/test_records_limits.py).

---

## Quick reference — enforcement site counts

| Category | Definition sites | Enforcement sites |
|----------|-----------------:|------------------:|
| Domain byte/count caps (tier 1) | 1 (`limits.py`) | 8 validators in `models.py` + 1 in `scoring.py` |
| Postgres ceiling (tier 2) | 1 (`serialization.py`) | `ensure_recordable` + 8 JSONB insert paths in `db/io.py` |
| Preview/truncation (tier 3) | 1–2 files | Serialization / logging only |
