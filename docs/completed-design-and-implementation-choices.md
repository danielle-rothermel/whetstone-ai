# Completed design and implementation choices

**Current status:** Architecture reference only. During the June 30 eval push,
use [`../AGENTS.md`](../AGENTS.md) for active priorities and do not turn this
reference into new platform-polish work unless it directly blocks backfill,
rescoring, model selection, the enc-dec budget sweep, or the minimal
COPRO-style experiment loop.

**Canonical companion:** [`remaining-implementation-intentions.md`](remaining-implementation-intentions.md) (backlog and deferred work).

**Date:** 2026-06-30  
**Purpose:** Consolidated record of settled architecture, product, and implementation decisions with no remaining ambiguity.

---

## Platform architecture

### Separation of concerns

Three distinct ownership domains:

| Domain | Owns | Does not own |
|---|---|---|
| **DBOS** | Durable workflow execution, retries, recovery, in-flight state | Scientific facts, analysis selection |
| **Append-only app tables** | Requested specs, terminal generation outcomes, terminal scoring outcomes | Live workflow lifecycle mirroring |
| **Projection tables/views** | Selected generation + score outcomes for analysis | Source of truth for what happened |

Terminal errors (provider, parser, HumanEval, uncategorized) are persisted as append-only outcome rows because they affect experimental coverage and cost. Transient states (`queued`, `started`, `retrying`, `deduplicated`) stay in DBOS only.

### Graph-based prediction identity

- Direct and enc-dec are both **node layouts in one generation graph model**.
- A **prediction spec** describes requested work for one task, graph, and repetition — not mutable workflow status.
- **Graph spec is part of prediction identity**, with stable graph/dimensions hashing.
- **Node ids** are first-class artifact addresses.
- **Optimizers** (especially COPRO) operate over graph variants, particularly encoder instructions.

Prediction spec includes: experiment identity, task id/inputs, graph spec (nodes, configs, instructions), dimensions digest, provider axis identity (`provider_kind`, `endpoint_kind`, `model`, `throttle_key`, optional `config_id`), and stable prediction id from axes plus repetition seed.

### Append-only outcome model

**Node attempts** record terminal outcomes only:

- `success`: output + usage, cost, response metadata
- `error`: failure class/type/message + structured metadata

**Blocked** is pure-graph-runner bookkeeping for nodes not invoked due to upstream errors. It is **not** persisted as a node-attempt row. Persist the upstream error; a generation-run summary may note the terminal node was blocked.

**Score attempts** are append-only, keyed by prediction id, generation artifact/run id, scoring profile, and parser version. Rescoring inserts new rows; it never rewrites history.

**Generation runs** group one full graph execution. `generation_run_id` is stable from `(prediction_id, attempt_index)`.

### Node attempt index semantics

- `generation_runs.attempt_index` — whole workflow reruns for one prediction.
- `node_attempts.attempt_index` — retries of an individual node inside one run.

DBOS retries happen inside the node execution step and do not create separate node-attempt rows. Each invoked node persists with `INITIAL_NODE_ATTEMPT_INDEX` (0) until explicit node-reattempt workflows exist.

### Batch submission

Batch submission is a first-class scaling primitive for tens/hundreds of thousands of specs.

A submit operation:

- Generates specs in streaming or chunked batches
- Computes deterministic prediction ids and fair-order keys
- Bulk inserts requested specs and batch items
- Enqueues DBOS work in chunks
- Is idempotent and resumable by operation key plus spec identity
- Reports intended, inserted, already-present, enqueued, already-scheduled, and failed counts

Batch records answer **"what work did we request and enqueue?"** — not **"what workflow is currently running?"**

Batch summary counts derive from terminal batch-item rows:

- `inserted_count` / `already_present_count` — spec dedupe
- `enqueued_count` / `already_scheduled_count` / `failed_count` — enqueue outcomes (must account for every item)

Batch items use separate `insert_status` (`inserted` | `already_present`) and `enqueue_status` columns.

Batch submit audit rows are **mutable operational audit**; generation, node, and score outcomes remain append-only.

### Fair queue ordering

Work is **not** enqueued in raw cross-product order. Each spec gets a deterministic **fair-order key** mixed across provider/endpoint, model, graph/layout, task id, repetition seed, and temperature/config variant.

The prediction-spec contract stores both `fair_order_seed` (ordering control, not identity) and derived `fair_order_key`. Inserts validate the key by recomputing from persisted fields.

**Fairness controls submission and queue-admission order**, not strict multi-worker execution order. With worker concurrency > 1, DBOS may finish workflows out of fair-prefix order. Use concurrency 1 when strict drain order matters.

### Throttle-key-aware backoff

Each LLM node/provider config exposes a **throttle key** (default: `provider:endpoint:model`).

Rate-limit and transient provider errors back off **only the affected throttle key**.

The **`dr_dspy_throttle_backoff` table** is the deliberate app-owned cross-worker coordination point for `blocked_until` and `consecutive_failures`. DBOS handles workflow durability and queue dispatch; it does not model per-key backoff state shared across independent workflows.

If retries exhaust, terminal node failure persists as a normal append-only error outcome.

Preflight reads throttle state before LM calls; successful calls clear backoff. A failed backoff clear on an otherwise successful LM call is swallowed so the generation outcome is not lost (stale `blocked_until` may linger until next retryable failure or later clear).

### Projection for analysis

Projection is **explicitly mutable** — a selected view, not source of truth:

- Append-only tables → "what happened?"
- Projection → "what should analysis use?"
- DBOS → "what workflow is currently running?"

---

## API and module boundaries

### Clean reusable components

- HumanEval task loading, sampling, test parsing, execution, scoring primitives
- Code extraction, parser profiles, source validation
- Text/code/compression metric extraction
- Serialization and recordability helpers
- Named exceptions and error classification
- LM request construction, provider response parsing, plain prompt adapter
- **Pure graph execution**

### Pure graph runner contract

Answers: *given a graph and a node executor, what happened?*

Owns: graph validation, topological order, input resolution, injected `run_node`, per-node outputs/errors, typed `GraphRunResult`.

Does **not** know: databases, DBOS, HumanEval, scoring, projections, batch submission, provider retry state, experiment reporting.

```python
result = execute_graph(
    graph=graph_spec,
    inputs=task_inputs,
    run_node=llm_node_runner,
)
```

### Platform-specific boundary

Answers: *which graph should run, when, where is the result stored, what should analysis use?*

Owns: experiment config → specs, provider config selection, node runner injection, DBOS submission, append-only persistence, scoring terminal artifacts, projection updates.

Also platform-specific: DBOS workflows/queues, batch submit records, throttle coordination, migration/backfill, Unitbench/Neon publishing, score-attempt persistence, rescoring orchestration.

Job submission isolates pure pieces (spec generation, fair-order keys, chunking, idempotency keys, batch summaries) where practical; DB insert and DBOS enqueue stay platform-coupled.

---

## Prototype carry-forward (eval-platform-v1 worktree)

### Adopt

- `GraphSpec`, `NodeSpec`, `NodeConfig`, `FieldSpec`
- Stable graph/dimensions hashing in prediction identity
- Node ids as artifact addresses
- Deterministic topological execution
- Input bindings (`task.prompt`, `encoder.output`, downstream node refs)
- Instruction mutation helpers for optimizer/COPRO
- Per-node artifacts: output, usage, cost, response metadata
- Encoder budget as node config/extra metadata

### Adapt

- Graph executor uses **plain prompt path**, not DSPy ChatAdapter formatting
- Stricter enums for roles, field types, node ops, outcome states, provider kinds, workflow/layout types
- Storage split into append-only specs, node outcomes, score outcomes, analysis projections
- Batch helpers for requested-work submission and summary accounting — not DBOS lifecycle mirroring
- Experiment records describe axes/configuration, not mutable execution progress

### Reject

- Raw SQL DDL strings as schema source
- One mutable predictions row mixing spec, workflow state, artifact, score, projection
- `generation_status` / `scoring_status` as workflow lifecycle source of truth
- DSPy signature/adaptor formatting as primary prompt control for new experiments

---

## v0 surfaces

v0 runtime code is **removed** from the repository (`experiments/`, `harness/`,
`lm/runner.py`, v0 CLIs, and related DSPy `Predict` paths). Legacy Postgres
tables may remain as read-only backup until backfill completes.

v1 replacement is thinner:

- Config/spec files define experiments, graph layouts, model/provider axes, scoring profiles
- Typer commands call explicit domain operations
- DBOS handles durable workflow execution
- App tables record terminal generation, error, scoring, and metrics outcomes
- Projections and Unitbench exports read selected outcomes

**Repair flow is not ported.** Transient recovery belongs to DBOS. Terminal coverage-affecting failures are normal outcome rows.

Migration is a **staged cutover**, not long-term dual-write:

1. Freeze v0 for new experimental writes
2. Build new schema and node-based execution on current branch
3. Backfill v0 direct/enc-dec rows
4. Validate counts, artifacts, costs, legacy scores, projections
5. Rescore migrated artifacts as new score attempts
6. Use new path for next COPRO experiments
7. Keep v0 tables as backup until trusted

`.worktrees/eval-platform-v1` is proof-of-concept reference only — not the branch to continue on.

---

## Prompt and LM boundaries

### PlainPromptAdapter (new generation path)

- Optional system message
- Caller-built user message unchanged
- Raw LM response as configured output field
- No field markers, fallback adapters, or hidden formatting

Prompt builders own encoder/decoder content — keeps optimizer target explicit.

Platform path uses plain prompt path; **no new ChatAdapter coupling** on the platform path.

### Provider config

Experiments support OpenRouter and direct OpenAI-style endpoints via provider-configured LM boundary.

Provider config captures: provider kind, endpoint kind, base URL, API key env var, model, temperature/reasoning support, token limit param name, extra request body mapping, throttle key override.

Graph specs reference provider/model configs without leaking endpoint details into prompt builders, scoring, or storage.

Runtime provider config is reconstructed from `ProviderConfigRef`: provider kind, endpoint kind, model, throttle key, request parameters. Custom runtime fields (`base_url`, `api_key_env`, capability flags) are **not spec-owned yet** — deferred contract extension.

Direct OpenAI support exists via typed configs, request builders, response parsers, and fake-client dispatch for chat completions and Responses-style endpoints.

---

## Scoring and metrics

### Scoring profiles (implemented defaults)

Default scoring surface: **`humaneval@v1`**

- Parser: **`humaneval-best-effort@v1`** (JSON/code-object unwrap, recoverable extraction)
- Field-marker parser: **`humaneval-field-marker@v1`** (instruction-adherence, not full ChatAdapter mirror)
- Metrics profile: **`humaneval-metrics@v1`**
- HumanEval dataset default: **`evalplus/humanevalplus`** split **`test`** (`DEFAULT_SCORE_DATASET_NAME` / `DEFAULT_SCORE_DATASET_SPLIT` in `records/hashing.py` — single source of truth)

Score attempts unique per: generation run, scoring profile, parser profile, score attempt index, HumanEval dataset name/split.

Score attempt statuses:

- `success` — completed domain scoring including zero-score outcomes (failed tests, empty generations, extraction failure, etc.)
- `error` — infrastructure/workflow failures (missing generation rows, task loading failures)

Scoring does not mutate generation/node-attempt rows, v0 tables, or projections during score/rescore.

### Rescoring flow (design)

1. Select terminal generation artifacts (including legacy imports)
2. Score with named `scoring_profile` and `parser_version`
3. Insert one score-attempt outcome per prediction/artifact/profile
4. Update analysis projection **only after** score-attempt batch validation

Parser/scoring profile includes: strict field-marker extraction, best-effort extraction, JSON `{"code": ...}` unwrap, HumanEval code cleaning, rejection of empty strings and dict/list literals, intentional unwrap of DSPy `Code` objects.

Score-attempt rows store profile/parser as `(id, version)` references plus typed score/metric payloads.

### Metrics persistence

Persist full HumanEval per-case results and aggregate counts.

Persist per-stage text metrics for every raw node output, encoder outputs, decoder raw generations, and extracted code.

Text metrics include: character/byte/line counts, nonempty lines, word counts, average word length, punctuation/symbol counts.

Encoder outputs get **Python leakage metrics** (keywords, `def`/`return`/`import`, fenced blocks, indentation, operators, task-specific names).

Extracted code gets AST/code metrics when parsing succeeds; raw text metrics + parse failure when not.

Terminal and node-output metrics use original payloads converted to text at platform boundary; extraction metrics use parser result. Non-string payloads go through recordability boundary and canonical JSON.

`GeneratedCodeOutcome` on primitive HumanEval score results supports later score-attempt persistence of pass/fail/extraction/no-top-level-function reasons.

Metrics payloads enforce domain-tier stage-count and byte caps at record validation time.

Pre-live hardening uses deterministic HumanEval task/test shape metrics and stdlib `ast` code-shape summaries inside existing metrics JSONB — keep `humaneval-metrics@v1` for that pass.

External exploratory tooling (textdescriptives/spaCy, MinHash, tree-sitter, radon/lizard, parquet, DuckDB) belongs to a **later analysis pipeline**, not current scoring profiles.

---

## Schema and database tooling

- **Schema authority:** Python eval platform (dr-dspy / future whetstone)
- **Stack:** SQLAlchemy Core + Alembic + Pydantic `BaseModel` contracts + generated TypeScript types for Unitbench
- **Not default:** SQLAlchemy ORM (append-only experimental facts, not mutable app objects)
- **Module split:** `db/schema.py`, `db/migrations/`, `records/`, `db/io.py`
- **Unitbench:** read-only consumer of published projections; types generated from Python contracts or introspection — not hand-maintained in parallel
- **No second migration authority:** Drizzle maybe later as read-side query builder; Prisma rejected (second app data model)
- **Ownership:** dr-dspy owns schema/migrations/writes/backfills/rescoring; Neon stores published copy; Unitbench reads projections

---

## Platform workflow implementation (shipped behavior)

### Entrypoints

| Command | Purpose |
|---|---|
| `run-one --prediction-id` | Run one existing spec (spec must pre-exist) |
| `worker` | Queue consumer on `dr-dspy-platform-generation-v1` |
| `submit-jsonl` | Two-pass chunked JSONL submit with resumable operation key |
| `score-one --generation-run-id` | Score one generation run |
| `rescore --experiment-name` | Batch schedule scoring workflows (default: successful runs, `humaneval@v1`) |

Workflow IDs: `platform-generate-v1:{generation_run_id}`, `platform-score-v1:{score_attempt_id}`.

### DBOS workflow design

- Generation start and completion use **distinct clock step names**
- Node-attempt timestamps captured inside node execution step (where provider call happens)
- Step retry exhaustion → terminal node error in separate DBOS step
- Deterministic workflow IDs with start-race recovery via `workflow_start_raced` / `DBOS.retrieve_workflow`
- Append-only persistence uses `ON CONFLICT DO NOTHING` — first-write-wins idempotency

### Submit path behavior

- Lightweight indexing pass → global sort by `(fair_order_key, prediction_id)` → chunked persist → chunked enqueue
- Peak memory O(n) refs + O(chunk-size) full specs
- Enqueue lifecycle: `pending → claiming → enqueued | workflow_already_present | failed`
- Crash mid-enqueue leaves operation `enqueuing`; resume resets `claiming` → `pending` and retries `failed`
- Submit does not start a worker; `--queue-registration-concurrency` only registers queue metadata
- Rescore dry-run opens DB only; scheduling run launches DBOS runtime for workflow submission only

### Batch rescoring

Selector joins generation runs to prediction specs, orders by `(fair_order_key, prediction_id, generation_run_id)`, anti-joins existing score attempts.

Orphan detection: `workflow_in_flight`, `workflow_orphan`, `recovered` (via `DBOS.retrieve_workflow(...).get_result()`).

Default `--recover-orphans` enabled. Does not write projections or app-owned scoring lifecycle state.

HumanEval task map cached process-locally by dataset name + split.

### Integration test tiers (defined)

- **Tier 1:** Postgres round-trip for load/persist steps, submit/resume helpers, scoring step idempotency
- **Tier 2–3:** End-to-end workflows under DBOS with mocked LM; scoring replay, memoization, orphan recovery
- **Tier 3.5:** Frozen v0 sample rows through `migration/v0_reshape.py`
- **Tier 4:** JSONL submit → DBOS enqueue → queue consumer → generation → scoring (mock LM + HumanEval loader only)

Default unit suite covers pure graph orchestration, record conversion, idempotent SQL, queue registration, submit/resume selection, throttle statement construction without Postgres/DBOS.

### Review items closed

- Partial persist before enqueue — intentional chunked commit
- Fairness vs execution — submit/enqueue order only
- Throttle table — intentional app-owned coordination
- Append-only vs mutable batch audit — as designed
- Pure/domain boundary — `recordable_text` at recordability boundary; platform reuses `domain_score.raw_generation` for terminal metrics
- Dataset defaults — centralized in `records/hashing.py`
- Throttle clear failure swallowing — preserves generation outcome

Platform CLI bootstraps DBOS through `dr_dspy.platform.dbos_bootstrap`; workflow start-race handling lives in `dr_dspy.platform.dbos_compat`.

---

## HumanEval domain modules

`dr_dspy/humaneval/` carries forward as domain layer with review goals:

- Split runtime AST from persistable parsed-code summaries
- Keep discriminated test-case shape; review `Any` fields and stable case ids
- Keep deterministic sampling as dataset selection/spec preparation
- Compression as part of broader versioned metrics profile
- Preserve task overrides and test parsing unless concrete issue found

Core primitives expose persistable summaries for parsed code, parsed tests, and per-case evaluation results.

Subprocess runner validates per-case output; partial runner output preserved (not whole-batch error) — current behavior until per-test persistence semantics finalized.

---

## Repository identity and structure

### Split decision

Extract from nested `dr-dspy/` inside `stanfordnlp/dspy` fork into standalone personal-org repo. DSPy becomes pinned PyPI dependency (`dspy==3.3.0b1`), not vendored substrate. Fork relationship is overhead with no upstream intent.

### Name

**whetstone** — sharpening metaphor; broad enough for future methods.

| Surface | Choice |
|---|---|
| GitHub repo | `<personal-org>/whetstone-ai` |
| PyPI distribution | `whetstone-ai` |
| Import package | `whetstone` (after rename) |

GitHub remains actual home for now. Name reservations on Cursor Origin, Codeberg, Tangled are defensive holds — not migration plans.

### Intended package shape (post-rename)

```
whetstone/
├── humaneval/     # approved scope
├── graph/         # pure graph execution (plain-prompt platform path)
├── platform/      # append-only facts, DBOS, CLI entrypoints
└── ...            # future siblings reserved, not built
```

**Broad repo identity, narrow current content.** Future approaches (KG, RL, agent-sandbox) are README future directions — not scaffolding.

Rename `dr_dspy` → `whetstone` is a **separate commit after extraction settles**.

---

## Migration status (schema)

v1 platform schema migration history is **frozen** at revision `20260630_0005` (nine linear revisions from `20260629_0001`). Existing revision files must not be rewritten; future schema changes add forward revisions only. Databases that applied unlisted draft v1 schemas during hardening must **reset v1 platform tables and replay** Alembic from base — there is no supported upgrade path from draft shapes. See [`v1-schema-migrations.md`](v1-schema-migrations.md).
