> **RETIRES AT D3 MERGE** — this documents the in-flight composable migration (draft PR #4) and becomes historical when it merges. Tracked in Linear S17/DEV-21.

# Platform Extraction: High-Level Design

Status: draft — high-level plan only. Sections will be filled in as design
discussion continues.

This doc covers the **platform library** extraction: the durable
experiment-running machinery currently embedded in `dr_dspy.platform` (plus
the generic halves of `records/batch_submit.py` and related modules). It is
one of five planned extractions (hashing/serialization, graph runner,
LM boundary + failure taxonomy, codegen-eval, platform); the platform is the
largest and lands last, after the failure taxonomy it depends on is stable in
its own package.

## What the library is

Conventions on top of DBOS for running large sweeps of durable work:

- Stable identity for work items (canonical-hash IDs from declared axes).
- Idempotent, resumable batch submission (operation keys, claim/lease
  enqueue lifecycle, dedup enqueue with race detection).
- Throttling/backoff keyed by operator-visible throttle keys.
- Fair ordering across sweep axes.
- Progress, attempt-history, and run-health observability.
- Artifact offload and rebuildable analysis projections over append-only
  outcome rows.

## What the library is not

An explicit anti-goals list is a first-class part of the design.
*(Lineage: dr-queues shipped a good "what dr-queues is not" list but then
grew a workflow engine anyway — the drift into step sequencing is what
killed it once DBOS-shaped tools were found to exist.)*

The library must never own:

- **Orchestration.** No step definitions, step sequencing, handler
  registries, job state machines, or retry scheduling. DBOS owns workflow
  execution and recovery. Workflows and steps stay app-side; the library
  accepts callables and typed items, never step definitions.
- **Domain semantics.** No LM calls, prompts, scoring, benchmarks, or
  model configs.
- **Speculative backends.** No swappable storage/transport interfaces until
  a second real user exists. *(Lineage: dr-llm gen-3 projection branches —
  NATS + Zarr shards + SQLite sidecar + hypergraph metadata, all built ahead
  of demand; the author's own verdict was "good idea, over-engineered.")*
- **Untyped payload slots.** Work items and results cross the boundary as
  caller-supplied typed models, not `dict[str, Any]` grab-bags.
  *(Lineage: dr-queues' `JobEnvelope.payload/step_outputs` dicts forced
  dr-bottleneck to build adapter models over them; dr-llm's opaque JSONB
  `request_json`/`response_json` blobs were the root cause of the "awful
  analysis" era.)*

## Public API altitude

The public surface is a small facade, not the internal machinery:

1. Declare typed axes (id + metadata per member).
2. Build work items from axes with stable hashed IDs — declarative,
   idempotent seeding that can be re-run and reconciled.
3. `submit(operation_key, items)` — durable, resumable batch submission.
4. `run(...)` — worker entrypoint with concurrency config.
5. `progress()` / attempt history / run health — queryable observability.
6. Read-side helpers for analysis (projection tables / DataFrames).

*(Lineage: nl_latents is the measurement of where the previous boundary
failed — it imported ~20 low-level `dr_llm.pool` symbols and still had to
hand-roll an axis-metadata catalog table, a filtered round-robin claim
backend, a config-drift repair pass, and a bespoke seeder. Anything a
consumer had to re-implement locally is spec for this facade. Declarative
seeding specifically fixes dr-llm gen-2's roughest edge: cross-product
inserts with stringly payload contracts, advisory-lock `sample_idx`
allocation, and silent `ON CONFLICT DO NOTHING` under-seeding.)*

The app side should not wrap this facade in passthrough re-exports; if a
shadow API accretes, the library altitude is wrong. *(Lineage:
dr-bottleneck's nearly-empty wrappers over dr-queues.)*

## What gets extracted from dr-dspy

The generic layer of `platform/`, behind a `SubmittableItem`-style protocol
(`item_id`, `order_key`, `group_key` — today's `prediction_id`,
`fair_order_key`, `experiment_name`) and an injectable enqueue target:

- `progress_log.py`, `dbos_compat.py` — already fully generic.
- `dbos_bootstrap.py` — URL/config resolution.
- `backoff.py` — throttle math plus a library-owned throttle table. The
  retryable-class set keys on the canonical `FailureClass` imported from
  dr-providers (resolved cross-doc question — see `overall.md`).
- `jsonl_specs.py` — JSONL byte-offset indexing/windowing.
- `fairness.py` — order-key sort + windowing (key-extractor injected).
- `queue_worker.py` — dedup enqueue with race detection.
- The claim/lease loop in `submission.py` (PENDING → CLAIMING →
  ENQUEUED/FAILED with CAS claims and stale-claim recovery).
- The batch status state machine in `records/batch_submit.py`.

## What stays in dr-dspy

- `records/` identity contract — `records/hashing.py` is a frozen wire
  format (versioned ID axes), not a utility. The library provides the
  hashing *mechanism*; the app owns the axis names and their stability.
- `db/` schema and migrations for generation/node/score attempts.
- `platform/persistence.py`, `graph_workflow.py`, `node_execution.py`,
  scoring workflows, `spec_builder.py` (the app-specific instantiation of
  the seeding facade), and the worker CLI.

Table ownership splits along the same line: the library owns and migrates
its own tables (throttle/backoff, batch operations/items, projections);
the app owns its outcome tables. *(Lineage: dr-queues/dr-bottleneck's
separate DB namespaces was one of the things that split got right.)*

## Features to add during extraction

Each is justified by a prior failure, and each is built in its minimal
form:

- **Artifact offload.** Content-addressed blob refs (sha256 key, size,
  content-type) with pointers stored in rows; one local-dir backend now, S3
  only when actually needed. Directly relieves the Postgres JSONB size
  guards in `serialization.py`. *(Lineage: dr-llm gen-3
  `artifact_projection` — keep `PayloadRef`/content-addressing/verify-on-
  read; drop shards, sidecar index, and staging dance.)*
- **Rebuildable analysis projections.** Flat, typed analysis tables (or
  DataFrames) derived from append-only outcome rows, keyed by
  `projection_version`, rebuilt from scratch rather than migrated.
  Plotting/reporting stays app-side; the *projection pattern* is platform.
  *(Lineage: dr-llm gen-2's blob-parsing 1,500-line notebooks are the
  recurring failure this prevents; gen-3's metadata projection is the good
  idea, minus the entity/assertion/role hypergraph.)*
- **Fairness: tags and holds.** Keep submission-time fair ordering
  (order keys); add target tags (`provider=...`) and operator holds with
  relative expiry (`+30m`) on the same table as automatic throttle/backoff
  state. Avoid claim-time round-robin, which fights DBOS's queue ownership.
  *(Lineage: dr-queues' target tags/holds control plane; dr-llm gen-2's
  round-robin claiming shows the demand, not the mechanism.)*
- **Attempt/progress observability as contract.** Queryable attempt
  history, lease/claim visibility, and run-health derivation are part of
  the library's public surface, not an app afterthought. *(Lineage:
  dr-llm gen-2's inspectable leases table; dr-queues' `JobAttempt` audit
  log and `RunHealth`.)*

## Deferred (recorded, not built)

- **Reader-side claims.** Exactly-once consumption of completed results
  per consumer run, with top-up generation — needed eventually for
  multi-stage pipelines (encoder → decoder lineage). Do not build until a
  second real consumer exists; do not design the schema in a way that
  precludes it. *(Lineage: dr-llm gen-2's `sampling/` layer.)*

## Validation

Before freezing the API, sketch nl_latents' seed → run → read loop written
against the facade. The API is right when that sketch is small and imports
no internals.

Second consumer sketch: **optimizer population evaluation**. An outer-loop
optimizer (COPRO today; GEPA/RL later) manufactures a population of graph
specs and needs "submit population under one operation key → await batch
outcomes → read scores" as a first-class loop against the same facade —
optimizers are ordinary durable code that searches over spec data (see
`graph_runner.md`). `optimization/copro.py` is the existing prototype of
this shape.

## Open sections (to fill in)

- Package name and repo layout.
- Exact protocol definitions (`SubmittableItem`, enqueue seam).
- Library-owned schema and migration story.
- Projection API sketch.
- Artifact store API sketch.
- Cutover plan from `dr_dspy.platform`.
