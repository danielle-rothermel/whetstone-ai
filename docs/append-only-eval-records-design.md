# Graph-based eval platform design

**Status:** decision note  
**Date:** 2026-06-29  
**Scope:** the next dr-dspy eval platform schema, execution, scoring, and
analysis design

This note captures the decisions made after auditing current direct/enc-dec
results, comparing them with the `eval-platform-v1` worktree design, and
clarifying the migration path for the next COPRO-oriented experiments.

## Summary

We want the next platform to keep the good parts of the v1 worktree:

- direct and enc-dec are both node layouts in one generation graph model
- the graph spec is part of prediction identity
- node outputs are addressable by node id
- optimizers operate over graph variants, especially encoder instructions

But we do **not** want to keep the current table shape where result data and DBOS
execution state are mixed in one mutable prediction row. That design made
rescoring hard to reason about because "score this artifact under a new parser"
looked like "mutate workflow state again."

The revised direction is:

- DBOS owns workflow durability and in-flight execution state.
- dr-dspy app tables store requested specs, immutable attempt outcomes, and
  analysis projections.
- Successful outputs and terminal errors are both persisted as append-only
  outcome rows.
- We do not mirror live DBOS lifecycle status in app tables.
- Batch submission remains a first-class scaling primitive, but it records
  requested work and enqueue outcomes rather than workflow lifecycle state.
- Prompt formatting, provider capabilities, parsing, scoring, and metrics are
  explicit versioned contracts owned by dr-dspy.

## Migration strategy

This migration should happen on the current branch. The
`.worktrees/eval-platform-v1` implementation is a proof of concept and design
reference, not the branch we continue building on.

The old direct and enc-dec tables remain the source data until migration is
verified. They should not be treated as the active execution path for future
experiments once this work begins.

The cutover plan is:

1. Freeze v0 for new experimental writes.
2. Build the new graph/spec/outcome/projection schema and node-based execution
   path on the current branch.
3. Backfill existing v0 direct and enc-dec rows into the new append-only model.
4. Validate migrated counts, artifacts, costs, legacy scores, and projections.
5. Run rescoring as new score attempts over the migrated terminal artifacts.
6. Use the new path for the next COPRO experiments.
7. Keep v0 tables as backup until the migrated data and new reports are trusted.

This is a staged cutover, not a long-term dual-write design. The goal is to stop
investing in v0 execution while keeping v0 data intact until the new schema has
proven that it preserves the old facts and supports the new analysis.

## Implementation sequence

Implement the migration as stacked PRs. The first PR should land this design
doc. After that, start by extracting and testing standalone primitives before
building the new schema and workflows.

The intended sequence is:

1. **Design doc:** commit this platform design as the shared north star.
2. **Core primitives:** stabilize HumanEval loading, sampling, test execution,
   scoring primitives, code extraction, parsing helpers, compression helpers,
   serialization helpers, named exceptions, error classification, and any
   deterministic ordering helpers that are already clean enough to keep.
3. **LM and prompt boundary:** add or solidify the plain prompt adapter,
   OpenRouter caller, OpenAI caller, request construction, response parsing,
   exact-message tests, fake-client tests, and usage/cost metadata extraction
   where available.
4. **Pure graph execution core:** add the reusable graph runner with no
   database, DBOS, HumanEval, scoring, or projection knowledge.
5. **Archive v0 surfaces:** after useful primitives are extracted, move old
   orchestration-heavy surfaces out of the intended path or mark them clearly
   as legacy. This includes old CLIs, manifests, repair flow, old reporting,
   and v0 `experiments/` implementation details.
6. **Domain contracts:** add graph specs, provider configs, node outcomes,
   score outcomes, metrics/profile models, stable ids, and fair-order keys.
7. **Schema and migrations:** add SQLAlchemy Core tables and Alembic migrations
   for specs, node outcomes, score outcomes, projections, batch submit records,
   and any operational throttle state that proves necessary.
8. **Platform graph workflow:** wrap the pure graph runner in DBOS workflows and
   append-only node outcome persistence.
9. **Batch submission, fairness, and backoff:** add chunked idempotent submit,
   deterministic queue mixing, and throttle-key-aware retry/backoff behavior.
10. **HumanEval scoring and metrics:** add improved extraction, parser/scoring
    profiles, per-test persistence, text/code metrics, Python leakage metrics,
    and score-attempt insertion.
11. **Migration and validation:** backfill v0 direct/enc-dec data into the new
    append-only shape and validate counts, artifacts, costs, legacy scores, and
    projections.
12. **Rescoring:** add the workflow/CLI for rescoring migrated artifacts and
    moving projections after validation.
13. **Unitbench/export:** publish stable projections/views to Neon and align or
    generate TypeScript types for the viewer.

The reason for starting with primitives is to force the reusable pieces to have
their own tests and contracts before they sit under DBOS workflows and database
records. Otherwise, utility code that should remain swappable will tend to
inherit v0 workflow assumptions or new platform-specific persistence details.

### Implementation Notes

For the core-primitives stage, the HumanEval parser/scorer now exposes
persistable summaries for parsed code, parsed tests, and per-case evaluation
results. These summaries are primitive contracts only; v0 experiment write
paths still persist the legacy score columns.

`GeneratedCodeOutcome` is available on the primitive HumanEval score result so
later score-attempt rows can store why a generated answer passed, failed,
failed extraction, or had no top-level function. Persisting that field belongs
to the score-attempt schema/scoring-profile stage, not the v0 table cleanup.

Subprocess runner output is validated per returned case, but this stage keeps
the existing behavior that partial runner output is preserved instead of being
treated as a whole-batch runner error. Stricter cardinality requirements should
be decided with the per-test persistence and score-attempt semantics.

For the LM and prompt boundary stage, the plain no-hidden-formatting prompt path
is available and tested at the LM library boundary only. The current v0 direct
and enc-dec workflows still call `dspy.Predict` and therefore still use DSPy
prompt formatting. The later graph-runner stage should adopt caller-built
messages through the plain prompt path rather than rewriting the v0 experiments
in this PR.

## API boundary strategy

Some components should be intentionally clean, reusable APIs. Others can remain
closer to the dr-dspy platform because they coordinate persistence,
orchestration, and reporting.

Design these as clean reusable components:

- HumanEval task loading, sampling, test parsing, test execution, and scoring
  primitives
- code extraction, parser profiles, and source validation
- text/code/compression metric extraction
- serialization and recordability helpers
- named exception and error-classification primitives
- LM request construction, provider response parsing, and plain prompt adapter
- pure graph execution

The pure graph runner should answer one question:

```text
given a graph and a node executor, what happened?
```

It should own graph validation, topological order, input resolution, calling an
injected `run_node`, collecting per-node outputs/errors, and returning a typed
`GraphRunResult`. It should not know about databases, DBOS, HumanEval, scoring,
projections, batch submission, provider retry state, or experiment reporting.

It should have an API shaped like:

```python
result = execute_graph(
    graph=graph_spec,
    inputs=task_inputs,
    run_node=llm_node_runner,
)
```

The platform workflow should answer the surrounding questions:

```text
which graph should run, when should it run, where is the result stored, and
what should analysis use?
```

It owns converting experiment configs into specs, selecting provider configs,
creating the injected node runner, submitting DBOS work, persisting append-only
outcomes, scoring terminal artifacts, and updating projections.

Handle these pieces with a more platform-specific boundary:

- DBOS workflow definitions and queues
- append-only persistence and projection updates
- batch submit records and enqueue summaries
- operational throttle/backoff coordination
- migration/backfill jobs
- Unitbench/Neon publishing

For job submission, isolate the pure pieces where practical: spec generation,
fair-order key generation, chunking, idempotency keys, and batch summaries.
Database insertion and DBOS enqueueing can be platform-specific because they are
inherently coupled to storage and orchestration.

For scoring, keep the scorer itself pure and reusable, but let score-attempt
persistence, projection movement, and rescoring batch orchestration remain
platform-specific.

This boundary choice is a deliberate compromise. We want the pieces most likely
to be reused in other projects to stay decoupled, but we do not want to turn the
whole eval platform into a generic workflow framework before the next
experiments run.

## Prototype carry-forward

The `.worktrees/eval-platform-v1` implementation is useful as a proof of
concept, but it should not be ported wholesale.

Adopt these pieces:

- `GraphSpec`, `NodeSpec`, `NodeConfig`, and `FieldSpec` as the core language
  for generation flows
- stable graph/dimensions hashing as part of prediction identity
- node ids as first-class artifact addresses
- deterministic topological graph execution
- input bindings such as `task.prompt`, `encoder.output`, and downstream node
  references
- instruction mutation helpers for optimizer/COPRO use
- per-node artifacts containing output, usage, cost, and response metadata
- encoder budget behavior represented as node config/extra metadata

Adapt these pieces:

- the graph executor should use the new plain prompt path rather than DSPy's
  ChatAdapter prompt formatting
- spec models should use stricter enums for roles, field types, node ops,
  outcome states, provider kinds, and workflow/layout types where values are
  closed project concepts
- the v1 storage concepts should be split into append-only specs, node
  outcomes, score outcomes, and analysis projections
- batch-operation helpers may be reused for requested-work submission and
  summary accounting, but not for DBOS lifecycle mirroring
- experiment records should describe axes/configuration, not mutable execution
  progress

Reject these pieces:

- raw SQL DDL strings as the schema source
- one mutable predictions row that mixes spec, workflow state, generation
  artifact, score, and selected projection
- `generation_status` / `scoring_status` columns as the source of truth for
  workflow lifecycle
- DSPy signature/adaptor formatting as the primary prompt-control mechanism
  for new experimental generation

## v0 surfaces not carried forward by default

The current CLI commands, manifests, repair flow, old reporting surfaces, and
implementation details inside `dr_dspy/experiments/` should be treated as v0
scaffolding. Some of the configuration ideas and operational lessons are worth
documenting, but the code paths themselves were shaped around mutable
prediction rows and status repair.

The v1 replacement should be thinner:

- config/spec files define experiments, graph layouts, model/provider axes, and
  scoring profiles
- Typer commands call explicit domain operations
- DBOS handles durable workflow execution, retries, and recovery
- app tables record terminal generation, error, scoring, and metrics outcomes
- projections and Unitbench exports read selected outcomes

The repair flow should not be ported as a general mechanism. In v1, transient
workflow recovery belongs to DBOS. Terminal failures that affect experimental
coverage are persisted as normal outcome rows.

## Decisions

### Use graph-shaped prediction specs

Keep the v1 worktree's core graph model. A prediction spec represents intended
work for one task, graph, and repetition. It should include:

- experiment identity
- task id and task inputs
- graph spec, including node configs and instructions
- dimensions digest / stable prediction id

The prediction spec is not a mutable workflow-status row. It describes what was
requested.

### Store node attempts as append-only outcomes

Node execution should write append-only rows for terminal outcomes:

- `success`: node output plus usage, cost, response metadata
- `error`: failure class/type/message and structured failure metadata

This captures the experimental fact that a node either produced an artifact or
failed in a way that affects coverage/cost/outcomes.

It does not store transient states such as queued, started, retrying, or
deduplicated. Those belong to DBOS.

### Keep batch submission for scale

Batch submission remains a core design requirement because some experiments
submit tens or hundreds of thousands of specs. Queue fairness must not turn into
slow one-by-one workflow submission.

A submit operation should:

- generate specs in streaming or chunked batches
- compute deterministic prediction ids and fair-order keys per spec
- bulk insert requested specs and batch items
- enqueue DBOS work in chunks
- be idempotent and resumable by operation key plus spec identity
- report intended, inserted, already-present, enqueued, and failed counts

Batch records answer "what work did we request and enqueue?" They do not answer
"what workflow is currently running?" DBOS owns the latter.

### Mix queued work deterministically

Work should not be enqueued in raw cross-product order. That order can hammer a
single low-rate-limit model for a long time and make partial runs misleading.

Each spec should receive a deterministic fair-order key derived from the
experiment seed and stable spec identity, mixed across axes such as:

- provider and endpoint
- model
- graph/layout
- task id
- repetition seed
- temperature/config variant

This gives reproducible queue order while interleaving models and configs. It
also keeps early partial results more representative.

### Add config-key-aware rate-limit backoff

Provider/model rate limits have been a repeated orchestration pain point, so v1
should model them directly. Each LLM node/provider config should expose a
throttle key, defaulting to something like:

```text
provider:endpoint:model
```

Rate-limit and transient provider errors should be classified by the named
exception system and used to back off only the affected throttle key. A
rate-limited OpenRouter model should not block unrelated OpenAI or other-model
work.

The active backoff state is operational state, not scientific output state. It
can live in DBOS-managed scheduling state or a small operational throttle table
if DBOS needs an app-visible coordination point. If retries are exhausted, the
terminal node failure is persisted as a normal append-only error outcome.

### Store score attempts as append-only outcomes

Scoring and rescoring should write append-only rows keyed by:

- prediction id
- terminal generation artifact or generation run id
- scoring profile
- parser version

Score attempts also store terminal outcomes:

- `success`: score, extracted code, extraction method, compile/eval/compression
  metrics
- `error`: parser/scoring/evaluation failure metadata

This makes rescoring a normal insert of a new scoring outcome, not a rewrite of
history.

### Keep a projection for analysis

We still need a simple query surface for model selection, optimizer reads, and
reports. Add or maintain a projection that points at the selected generation and
score outcomes for each prediction spec.

The projection is allowed to be mutable because it is explicitly a selected
view, not the source of truth.

The invariant is:

- append-only tables answer "what happened?"
- projection tables/views answer "what should analysis use?"
- DBOS answers "what workflow is currently running?"

### Persist terminal errors, but not live workflow state

We should store end-state errors as append-only rows because they affect the
outcome of an experiment. That includes provider failures, parser failures,
HumanEval execution failures, and other terminal errors we are not yet good at
categorizing.

We should not store mutable app-level copies of DBOS state such as started,
queued, retry pending, or in-flight attempt counts. Duplicating those states has
already made the repo harder to reason about.

## Rationale

The current v0 direct and enc-dec prediction tables combine three concerns:

- scientific outputs: generations, extracted code, score, metrics
- workflow control: generation/scoring statuses, failure classes, repair state
- query projection: the current score to use for analysis

The v1 worktree improves the generation model by moving direct and enc-dec into
a unified graph-shaped `dr_dspy_predictions` table with JSONB dimensions and
artifacts. However, the implemented table still mixes mutable workflow state
with artifacts and metrics.

The new design separates those concerns. DBOS is already responsible for durable
workflow execution, retries, and recovery. The app database should not replicate
that state. It should record the terminal facts that matter for evaluation and
analysis.

This also fixes the rescoring problem. When parser/scoring logic changes, we add
a new scoring profile and insert new score attempts for existing terminal
generation artifacts. The projection can then be moved to the new profile after
validation.

## Rescoring implications

The immediate rescoring flow should:

1. Select terminal generation artifacts, including legacy imports if needed.
2. Score them with a named `scoring_profile` and `parser_version`.
3. Insert one score-attempt outcome per prediction/artifact/profile.
4. Update the analysis projection only after the score-attempt batch is
   validated.

The parser/scoring profile we discussed should include:

- strict ChatAdapter extraction, for instruction-adherence measurement
- best-effort extraction, for recoverable-code measurement
- JSON `{"code": ...}` unwrap before code cleaning
- existing HumanEval code cleaning
- rejection of empty strings and dict/list literals as valid extracted code
- intentional unwrap of DSPy `Code` objects instead of relying on `str(value)`

## Prompt/control implications

New generation runs should stop relying on DSPy's ChatAdapter prompt formatting
for this experiment path. Add a minimal `PlainPromptAdapter` that:

- accepts an optional system message
- sends a caller-built user message unchanged
- returns the raw LM response as the configured output field
- does not add field markers, fallback adapters, or hidden formatting

Prompt builders, not the adapter, should own the encoder and decoder prompt
content. This keeps the future optimizer's target explicit.

## Provider and LM boundary implications

The current OpenRouter-specific LM wrapper should become a provider-configured
LM boundary. Experiments need to support both OpenRouter and direct OpenAI-style
endpoints because provider behavior differs, including temperature support,
reasoning parameters, token-limit names, and endpoint style.

Provider config should capture:

- provider kind, such as `openrouter` or `openai`
- endpoint kind, such as chat completions or responses
- base URL and API key environment variable
- model name
- temperature support
- reasoning support and request shape
- token limit parameter name
- provider-specific extra request body mapping
- throttle key override when the default provider/endpoint/model key is not
  specific enough

Graph specs should reference these provider/model configs without requiring
prompt builders, scoring, or storage code to know endpoint-specific request
details.

## HumanEval domain modules

The `dr_dspy/humaneval/` package is the domain layer that should mostly carry
forward, but it needs a review before schema freeze.

The review goals are:

- split runtime AST objects from persistable parsed-code summaries in
  `parsed_code.py`
- keep the discriminated test-case shape in `parsed_tests.py`, while reviewing
  `Any` fields and stable case ids for persisted per-test results
- keep `sampling.py` as deterministic dataset selection/spec preparation, with
  naming or placement clarified if useful
- keep `compression.py` as a small metrics component, but make compression one
  part of a broader versioned metrics profile
- preserve the current task overrides and test parsing behavior unless the
  review finds a concrete schema or correctness issue

## Metrics and analysis implications

Scoring/rescoring should persist detailed evaluation and generation metrics, not
only pass/fail aggregates. These metrics should be versioned with the scoring or
metrics profile so future fixes create new attempts rather than rewriting old
facts.

Persist full HumanEval per-case results, including:

- task id, case id, function name, status, and message
- input representation
- expected output representation
- actual output representation when available
- aggregate counts derived from the per-case rows or payload

Persist per-stage text metrics for:

- every raw node output
- encoder outputs
- decoder raw generations, even when extraction fails
- extracted code, when available

The text metrics should include at least character counts, byte counts, line
counts, nonempty line counts, word counts, average word length, and punctuation
or symbol counts.

For encoder outputs, add Python leakage metrics whose purpose is to detect
descriptions that secretly pass code-like content. These metrics should count
Python keywords and code-like markers such as `def`, `return`, `import`, fenced
code blocks, indentation/code-like lines, operators, punctuation density, and
task-specific names when available.

For extracted code, add AST/code metrics when parsing succeeds, such as
top-level functions, classes, imports, AST node counts, and simple statement or
branch counts. If parsing fails, still persist raw text metrics and the parse
failure.

## Schema and database tooling

Use the Python eval platform as the schema authority. The durable workflows,
generation writes, scoring writes, migration/backfill scripts, and rescoring
jobs all live in dr-dspy, so the canonical database schema should live there
too.

The preferred stack is:

- SQLAlchemy Core for table definitions and query construction
- Alembic for reviewed database migrations
- Pydantic `BaseModel` contracts for domain records and JSONB payloads
- generated TypeScript types for the Unitbench/read-side app

This replaces ad hoc DDL and persistence SQL strings with explicit schema
objects while keeping the relational model visible. SQLAlchemy ORM models are
not the default fit here because these records are append-only experimental
facts and projections, not long-lived mutable application objects.

The practical module split should be:

- `dr_dspy/db/schema.py`: SQLAlchemy `MetaData` and `Table` definitions
- `dr_dspy/db/migrations/`: Alembic migration files generated from, then
  reviewed against, the table definitions
- `dr_dspy/records/`: Pydantic models for prediction specs, node outcomes,
  score outcomes, projections, and JSON payloads
- `dr_dspy/db/io.py`: typed insert/select helpers around the schema objects

Unitbench should remain a consumer of published projections, not a second owner
of the schema. For now, keep its Neon access simple and read-only: server-side
`SELECT` queries against stable projection tables or views are acceptable. The
important improvement is that its row and payload types should be generated from
the canonical Python contracts or from database introspection, rather than
hand-maintained in parallel.

Do not maintain separate Python and TypeScript migration authorities. Drizzle
can be considered later as a read-side query builder if Unitbench's queries
become painful, but it should be generated from or introspected against the
canonical schema. Prisma is not a good fit for this repo because it would push
toward a second application data model and migration system.

The intended ownership boundary is:

- dr-dspy owns schema definitions, migrations, writes, backfills, and rescoring
- Neon stores the published local/remote copy of the data
- Unitbench reads stable projections/views and renders them
- generated TypeScript types keep the viewer aligned with the Python-owned
  contracts

## Open questions

- Exact table names and primary keys for specs, node attempts, score attempts,
  and projections.
- Whether node attempts should be grouped by a `generation_run_id` for one full
  graph execution.
- How much legacy v0/v1 data should be imported into the new append-only shape
  versus rescored in place for short-term model selection.
- Whether projection should be a physical table, a view, or a DuckDB/read-side
  projection first.
