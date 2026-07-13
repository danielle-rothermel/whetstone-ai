# Running the v6 COPRO encoder-prompt optimizer

`whetstone-copro` optimizes the HumanEval encoder instruction pair without
bypassing Platform lifecycle, acceptance, or publication contracts. It uses
deterministic manual proposals; provider execution happens only in the normal
Whetstone worker.

## Lifecycle

Each depth is a separate immutable Experiment. The coordinator performs these
steps in order:

1. Build all candidate × task × repeat × compression-target Prediction specs.
   Every cell for one prompt candidate carries the same validated
   `dimensions.values.candidate_id`.
2. Submit the Generation Operation and wait through the typed Platform
   inspector.
3. Freeze scoreable Generation outcomes into a Scoring Manifest, submit the
   Scoring Operation, and wait through the same typed inspector.
4. Evaluate and promote strict Experiment acceptance.
5. Explicitly export the Whetstone Analysis and Detail Bundles.
6. Pin the committed Analysis Bundle and rank candidates from that immutable
   snapshot. Pin loss fails with `PINNED_BUNDLE_GONE`; the iteration never
   drifts to the latest pointer.

The coordinator never reads analysis from operational Postgres and workers
never export implicitly.

## Prerequisites

Live execution requires:

- `DATABASE_URL` and, when separate, `DBOS_SYSTEM_DATABASE_URL`;
- the API-key environment variable required by the selected model fragment;
- `WHETSTONE_BUNDLE_INTEGRITY_KEY_ID`;
- `WHETSTONE_BUNDLE_INTEGRITY_PRIVATE_KEY_PATH`; and
- `WHETSTONE_BUNDLE_INTEGRITY_PUBLIC_KEY_RING`.

Do not print these values. Apply the fresh schema before a run.

Start the current worker command in a separate terminal:

```bash
uv run python -m whetstone.platform.worker serve --worker-concurrency 4
```

## Zero-spend preflight

Dry-run builds the exact candidates and specs and atomically checkpoints the
operator artifacts. It does not resolve database/integrity configuration,
submit Operations, wait, export, read bundles, or call a provider.

```bash
uv run whetstone-copro \
  --model-config configs/models/gpt54-nano-openai.json \
  --split configs/splits/tiny.json \
  --compression-target 0.5 \
  --breadth 2 \
  --depth 1 \
  --repeats 1 \
  --output-dir artifacts/optimization/copro-preflight \
  --dry-run
```

## Live run

```bash
uv run whetstone-copro \
  --model-config configs/models/gpt54-nano-openai.json \
  --split configs/splits/tiny.json \
  --compression-target 0.5 \
  --breadth 3 \
  --depth 2 \
  --repeats 1 \
  --analysis-destination artifacts/optimization/copro/analysis.duckdb \
  --detail-destination artifacts/optimization/copro/detail.duckdb \
  --output-dir artifacts/optimization/copro
```

Use a stable `--run-id` to replay the same deterministic Operations and local
artifacts. Do not reuse a run ID with different inputs.

## Durable outputs

The coordinator atomically replaces these files after every completed depth:

- `run.json`: lifecycle operation keys and pinned bundle coordinates;
- `candidates.jsonl`: candidate instruction identities;
- `attempts.csv`: aggregate score and error counts; and
- `best_prompt.json`: current global best candidate and attempt.

Candidate ranking uses pass rate, scoreable count, combined Generation/Scoring
errors, instruction length, then candidate ID. Counts are aggregated across
all tasks and repeats sharing the published candidate identity.

## Verification

```bash
uv run pytest tests/test_optimization_copro.py -q
./scripts/ci/lint.sh
./scripts/ci/unit.sh
```
