> **RETIRES AT D3 MERGE** — this documents the in-flight composable migration (draft PR #4) and becomes historical when it merges. Tracked in Linear S17/DEV-21.

# Platform Extraction: High-Level Design

Status: extracted (Stage 6 complete, 2026-07-04). The library lives at
`danielle-rothermel/dr-platform`; whetstone consumes it via
`platform/platform_db.py` (frozen physical naming + lineage adoption),
`platform/submission.py` (the app composition), and `queue_worker.py`.
The former open sections are filled in at the bottom; consumer sketches
live in `sketches/` and were re-checked against the shipped facade
(6d).

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

*(Done — `sketches/nl_latents_loop.py` and
`sketches/optimizer_population_eval.py`. Both import only the
`dr_platform` facade; each hand-rolled piece of the lineage systems
maps to exactly one facade call. The one facade addition they forced
is `await_operation` — both consumers otherwise re-implement
work-completion polling.)*

*(6d re-check against the shipped library: the altitude held — no
passthrough wrappers accreted. Drift folded back into the sketches:
facade calls take an explicit `schema` handle (`PlatformSchema`) and
`group_key`; the seed hook runs inside the registration transaction
and returns inserted ids; no adapter class is needed for whetstone
(`PredictionSpecRecord` satisfies `SubmittableItem` directly). The
optimizer loop is no longer hypothetical: whetstone's
`evaluate_specs_queue` now runs submit_batch → await_operation →
rescore in production code.)*

Second consumer sketch: **optimizer population evaluation**. An outer-loop
optimizer (COPRO today; GEPA/RL later) manufactures a population of graph
specs and needs "submit population under one operation key → await batch
outcomes → read scores" as a first-class loop against the same facade —
optimizers are ordinary durable code that searches over spec data (see
`graph_runner.md`). `optimization/copro.py` is the existing prototype of
this shape.

## Package name and repo layout (resolved)

**`dr-platform`** (package `dr_platform`), private repo
`danielle-rothermel/dr-platform`, scaffolded like the siblings (uv, src
layout, py≥3.12, strict ruff + ty + pytest + pre-commit, py.typed).

Dependencies: `pydantic`, `sqlalchemy`, `dbos`, `alembic` (the library
ships migrations its adopters run), `dr-serialize` (canonical digests
for item/claim IDs and deterministic jitter), `dr-providers`
(`FailureClass` only — the taxonomy home per overall.md). `pandas` is an
optional extra (`dr-platform[frames]`): the projection core returns
typed rows; the DataFrame helper needs the extra.

Module map (whetstone source → library module):

| dr_platform module | extracted from | notes |
|---|---|---|
| `items.py` | (new) | `SubmittableItem`, `ItemIdentity`, `stable_item_id` |
| `fairness.py` | `platform/fairness.py` | key extractor = protocol fields |
| `jsonl.py` | `platform/jsonl_specs.py` | field names parameterized |
| `submission.py` | `platform/submission.py` | claim/lease loop, `submit_batch` |
| `enqueue.py` | `platform/queue_worker.py` | dedup enqueue + race detection, de-domained |
| `backoff.py` | `platform/backoff.py` | + tags/holds columns |
| `progress.py` | `platform/progress_log.py` | verbatim (already generic) |
| `observability.py` | (new + `dbos_compat` status vocab) | `load_operation_progress`, attempt history, `await_operation` |
| `projections.py` | (new; pattern from `analysis/frames.py`) | rebuildable versioned projections |
| `artifacts.py` | (new; PayloadRef lineage) | content-addressed local-dir store |
| `dbos_config.py` | `dbos_bootstrap.py` + `dbos_compat.py` | URL/config resolution; the single `dbos._error` shim |
| `db/` | `db/schema.py` (platform tables) | prefix-parameterized schema + own Alembic lineage |
| `batch_status.py` | `records/batch_submit.py` | status/count state machine (pure) |

Stays in whetstone: `graph_workflow.py`, `node_execution.py`,
`persistence.py`, scoring workflows, `spec_builder.py`, worker CLI,
`records/` (all of it — the batch operation/item *record models* move
into the library since they describe library-owned tables; the frozen
ID-axis functions in `records/hashing.py` stay), `db/` outcome schema
and its migrations, `analysis/` plotting/reporting.

## Protocol definitions (resolved)

```python
@runtime_checkable
class SubmittableItem(Protocol):
    """What the library needs to know about a work item — nothing else.
    Whetstone adapts PredictionSpecRecord: prediction_id /
    fair_order_key / experiment_name."""

    @property
    def item_id(self) -> str: ...     # stable identity (PK-ish)
    @property
    def order_key(self) -> str: ...   # fair-ordering sort key
    @property
    def group_key(self) -> str: ...   # sweep/experiment grouping
```

**Identity compatibility (`ItemIdentity`).** Persisted digest recipes
currently hash JSON payloads whose *key names* are domain words:
`batch_submit_item_id = sha256_json_digest({"operation_key": ...,
"prediction_id": ...}, length=32)`. Those IDs are primary keys in
existing rows and an ID axis under the frozen-contracts rule, so the
facade takes an identity config:

```python
class ItemIdentity(BaseModel):
    item_key_label: str = "item_id"   # whetstone passes "prediction_id"
    id_length: int = 32
```

The claim-token recipe (`{"operation_key", <label>, "claimed_at"}`)
parameterizes the same way. New adopters use the neutral default;
whetstone reproduces today's bytes exactly.

**Enqueue seam.** The library never sees a workflow function; it sees a
callable that starts (or finds) durable work for one item and reports
what happened:

```python
class EnqueueOutcome(BaseModel):
    workflow_id: str
    enqueued: bool                      # False -> already scheduled
    metadata: dict[str, Any] = {}       # e.g. generation_run_id

type EnqueueItem = Callable[[str], EnqueueOutcome]   # item_id ->
```

The library separately exposes the mechanism apps build enqueue targets
from (today's queue_worker internals, de-domained):
`dedup_enqueue(queue_name, workflow_id, workflow, *args) ->
EnqueueOutcome` wrapping `SetWorkflowID` + `SetEnqueueOptions(
deduplication_id=...)` + `DBOS.get_workflow_status` pre-check +
`WORKFLOW_START_RACE_ERRORS` handling. Whetstone's enqueue target stays
app-side (it derives `generation_run_id` and names the frozen queue
`dr-dspy-platform-generation-v1`; queue registration and names remain
app-owned frozen strings).

**Seed hook.** Today `prepare_submission_records` inserts domain rows
(experiment, prediction specs) in the same transaction as operation and
item registration. The library keeps the atomicity without the domain
knowledge:

```python
def submit_batch(
    engine, *,
    operation_key: str,
    items: Sequence[SubmittableItem],
    enqueue: EnqueueItem,
    seed: Callable[[Connection, Sequence[SubmittableItem]], None] | None = None,
    identity: ItemIdentity = ItemIdentity(),
    chunk_size: int = 500,
    metadata: dict[str, Any] | None = None,
) -> BatchSubmitResult: ...
```

`seed` runs inside the registration transaction per window; whetstone
passes its experiment/spec inserts. The claim/lease state machine
(PENDING → CLAIMING → ENQUEUED / WORKFLOW_ALREADY_PRESENT / FAILED,
CAS claims on `enqueue_status='pending'`, claim-token CAS on outcome
commit, stale-claim and failed-item resets before each pass) moves
verbatim. Enqueue failures are recorded as a library-owned
`EnqueueFailure` model (`error_type`, `message`, `failure_class:
FailureClass | None`, `metadata`) — same JSONB shape as today's
`FailureMetadataPayload` so existing rows read back cleanly.

**Backoff/throttle.** `record_throttle_failure` drops its dependency on
whetstone's `FailureSummary`: it takes `failure_class: FailureClass`,
`error_type: str`, `message: str | None`, `metadata` explicitly.
Retryable set stays `{TRANSIENT, RATE_LIMITED}` keyed on the canonical
`FailureClass`. Tags and holds land on the same table (below):
`set_hold(connection, *, throttle_key, duration | until, reason)`,
`clear_hold`, `set_tags`, `list_throttle_state(connection, *,
tag_filter=None)`; `delay_until_unblocked_seconds` becomes
`max(blocked_until, hold_until)` so operator holds and automatic
backoff compose.

**Await primitive.** Both consumer sketches need "wait until this
operation's work is done" and hand-roll it today (copro's
`wait_for_generation_runs`; nl_latents' bespoke polling). Workflow IDs
are deterministic and recorded in `enqueue_metadata`, so the library
can watch DBOS workflow statuses with no domain knowledge:
`await_operation(engine, *, operation_key, poll_interval_seconds,
timeout_seconds)` polls the recorded workflow IDs until none are in
`{ENQUEUED, PENDING, DELAYED}`, returning a status breakdown; timeout
raises with the breakdown attached.

## Library-owned schema and migration story (resolved)

Two constraints jointly force the design: the library owns its tables
with its own Alembic lineage (this doc / prompt), and whetstone's
existing physical names (`dr_dspy_batch_submit_operations`,
`dr_dspy_batch_submit_items`, `dr_dspy_throttle_backoff`) are frozen
byte-for-byte. Therefore:

- **Prefix-parameterized schema.** `PlatformSchema(prefix="dr_platform")`
  builds the SQLAlchemy `MetaData` with names
  `{prefix}_batch_submit_operations`, `{prefix}_batch_submit_items`,
  `{prefix}_throttle_backoff`, `{prefix}_projections` (registry),
  plus per-projection tables `{prefix}_projection_{name}`. Whetstone
  configures `prefix="dr_dspy"` — physical names unchanged.
- **Own Alembic lineage, stamped baseline for whetstone.** dr-platform
  ships `alembic/` with version table `{prefix}_platform_alembic_version`
  and revision 0001 = create-platform-tables (reading the prefix from
  Alembic `-x prefix=...` / env config). Fresh adopters run
  `upgrade head`. Whetstone's tables already exist from its frozen
  historical migrations, so whetstone **stamps** revision 0001 instead
  of running it. Whetstone's own lineage keeps its history byte-frozen
  and simply never touches platform tables again; all future platform
  DDL (e.g. the holds/tags columns) arrives as dr-platform revisions.
- **New columns at extraction** (dr-platform revision 0002, run — not
  stamped — by whetstone too): `hold_until timestamptz NULL`,
  `hold_reason text NULL`, `tags jsonb NOT NULL DEFAULT '{}'` on the
  throttle table; the `{prefix}_projections` registry table
  (`projection_name`, `projection_version`, `built_at`, `row_count`,
  PK `(projection_name, projection_version)`).

Risk recorded: two Alembic version tables in one database is standard
but unusual here; the guard is that the whetstone lineage's platform-
table history is additive-frozen and dr-platform 0001 is a no-op-if-
stamped baseline. The integration tier (which migrates a scratch DB)
must exercise both paths: fresh `upgrade head` and stamp-then-0002.

## Projection API sketch (resolved)

The *pattern* is platform; queries and plotting stay app-side.

```python
class ProjectionSpec[RowT: BaseModel](BaseModel):
    name: str                    # e.g. "copro_candidate_scores"
    version: str                 # projection_version; bump -> rebuild
    row_model: type[RowT]        # typed rows, never dict grab-bags
    build: Callable[[Connection], Iterable[RowT]]   # app-side query

def rebuild_projection(engine, spec) -> ProjectionBuildResult
    # delete rows for (name, version), re-insert from build(),
    # upsert the registry row; never migrated, always rebuilt.
def load_projection_rows(engine, spec, *, group_key=None) -> list[RowT]
def load_projection_frame(engine, spec, *, group_key=None) -> DataFrame
    # requires dr-platform[frames]
```

Projection tables are `{prefix}_projection_{name}` with columns derived
from `row_model` (scalar pydantic fields → native columns via a small
fixed mapping: str/int/float/bool/datetime; everything else JSONB),
plus `projection_version`. A `group_key` column is materialized when
the row model declares one, giving cheap per-experiment reads.
Whetstone's `dr_dspy_prediction_projection` table predates this API and
stays app-owned as-is; whetstone adopts the library pattern for new
projections rather than migrating that table (conservative choice).

## Artifact store API sketch (resolved)

```python
class ArtifactRef(BaseModel):
    sha256: str
    size_bytes: int
    content_type: str

class ArtifactStore(Protocol):
    def put_bytes(self, data: bytes, *, content_type: str = "application/octet-stream") -> ArtifactRef: ...
    def get_bytes(self, ref: ArtifactRef | str) -> bytes: ...   # verify-on-read
    def exists(self, sha256: str) -> bool: ...

class LocalDirArtifactStore:                 # the only backend now
    def __init__(self, root: Path | str) -> None: ...
    # layout: <root>/<sha[:2]>/<sha[2:4]>/<sha>; atomic write via
    # tmpfile + rename; get_bytes recomputes the digest and raises
    # ArtifactIntegrityError on mismatch.
```

Rows keep pointers (the three `ArtifactRef` fields) wherever the app
wants them; the library does not own an artifacts table in v1 (the
content-addressed directory is self-describing; a registry table can
arrive with the S3 backend when demand exists).

## Cutover plan from `whetstone.platform` (resolved)

Sequenced like every other stage — library lands green first, then one
whetstone cutover commit per surface, full suite + integration tier +
goldens after each:

1. **6a — repo + pure kernel.** Scaffold dr-platform; move the pure
   modules (`progress.py`, `batch_status.py`, `fairness.py`,
   `jsonl.py`, `dbos_config.py`, `items.py`) with their tests,
   de-domained (protocol fields / parameterized field names). No
   whetstone change yet.
2. **6b — schema + stateful modules.** `PlatformSchema`, Alembic
   lineage (0001 baseline + 0002 holds/tags/projections),
   `backoff.py`, `submission.py`, `enqueue.py`, `observability.py`
   (`await_operation`), `projections.py`, `artifacts.py`; suite green
   against a scratch Postgres (fresh-upgrade path).
3. **6c — whetstone cutover.** Path dep; `SubmittableItem` adapter on
   `PredictionSpecRecord` (property aliases — records models are
   frozen); seed hook wraps today's experiment/spec inserts; enqueue
   target stays app-side over `dedup_enqueue`; backoff call sites in
   `graph_workflow.py` pass explicit failure fields; batch
   operation/item record models re-home to the library (whetstone
   re-exports nothing — imports move); whetstone stamps dr-platform
   0001 and runs 0002; delete the extracted modules; worker CLI and
   `rescoring.py` re-wire onto the facade. Frozen strings verified
   unchanged: queue/workflow/step names, all `dr_dspy_*` table names,
   `enqueue_metadata` keys (`enqueue_claim_id`, `claimed_at`,
   `workflow_id`, `generation_run_id`), `batch_submit_item_id` digest
   bytes (via `ItemIdentity(item_key_label="prediction_id")`).
4. **6d — consumer validation.** copro's `evaluate_specs_queue` +
   `wait_for_generation_runs` collapse onto `submit_batch` +
   `await_operation`; the integration tier exercises stamp-then-0002
   migration; the two sketches are re-checked against the real facade
   (they must not have drifted).

Escalation points (stop and write up rather than improvise): any
mismatch in `batch_submit_item_id` bytes; any Alembic state where
stamp/upgrade would touch a frozen whetstone revision; any place the
whetstone adapter wants a passthrough wrapper (altitude smell per the
house rules).
