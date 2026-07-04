# Composable Extraction — Overall Plan

Status: living overview tying the per-library design docs together. Update
as docs land and decisions get made.

Whetstone is the composition of a few reusable concepts colocated for
experiment speed. The plan extracts them into five libraries so this repo
keeps only its domain: the identity contract, the outcome schema, the
workflows, and the analysis scripts (~6–7k of the current ~21.5k lines).

## Per-library design docs

- [`serialize.md`](serialize.md) — canonical hashing + JSON-safe
  serialization (dr-serialize)
- [`graph_runner.md`](graph_runner.md) — hashable treatment specs + pure
  interpreter
- [`llm_provider.md`](llm_provider.md) — LM query kernel (dr-providers v0.2)
- [`platform.md`](platform.md) — durable sweep machinery on DBOS
- [`code_parse_test.md`](code_parse_test.md) — code parsing + test execution
  (dr-code nucleus)

## Target shape

```
dr-serialize (hashing + serialization)        [new, tiny — serialize.md]
      ↑                    ↑
   dr-graph           dr-providers v0.2       [graph_runner.md]
  (treatment specs)   (LM kernel + FailureClass)
      ↑                    ↑
      |               platform library        [new — platform.md]
      |                    ↑
      +--------------------+
                whetstone-ai (the app)
                     ↑
              dr-code (parsing + test exec)   [existing repo, pruned]
                (no deps on anything above)
```

- **dr-serialize** — canonical hashing and the JSON-safe serialization
  engine with pluggable type handlers and configurable limits. The only
  package everything else touches. DSPy handlers and Postgres sizing
  become app-side registration/config.
- **dr-providers v0.2** — the LM query kernel per `llm_provider.md`. Also
  the canonical home of the `FailureClass` taxonomy and failure records.
- **dr-graph** — a hashable treatment-description language plus pure
  interpreter; deliberately not a workflow engine. v1 = DAG + subgraph
  composition + completed-nodes resume hook, open-string ops/field types,
  structural failure protocol. Growth (conditional routing, bounded
  iteration, dynamic map) is admitted only while specs stay finite,
  deterministic to replay, and canonically hashable — the requirement
  imposed by outer-loop optimizers (GEPA/RL) that mutate and hash specs
  as genomes.
- **platform library** — DBOS conventions per `platform.md`: stable item
  identity, durable claim/lease batch submission, throttle/backoff,
  fairness with tags/holds, attempt observability, artifact offload,
  rebuildable projections. Largest extraction; lands last.
- **dr-code** — strings-in/outcomes-out parsing and sandboxed test
  execution per `code_parse_test.md`. HumanEval first benchmark module,
  Unitbench second. Deliberately depends on none of the packages above
  (its gen-3 predecessor died from baked-in queue/provider deps).

**Whetstone keeps:** `records/` (frozen ID axes — a versioned wire format,
not a utility), `db/` outcome schema and migrations, generation/scoring
workflows, `node_execution.py`, `persistence.py`, `spec_builder.py`, the
worker CLI, thin adapters to each library, domain failure exceptions and
enc-dec validators, and the analysis scripts (the *projection pattern*
moves to the platform; plotting/reporting stay app-side).

## Sequencing

Each step ends with a whetstone cutover — an import swap plus a thin
adapter, callers updated in the same change, no compatibility shims. The
app shrinks incrementally; no big-bang migration.

1. **dr-serialize** — days of work, unblocks everything, zero risk.
2. **dr-code nucleus + golden profile port** — fully parallel to the rest
   (no upstream deps); sequenced internally in `code_parse_test.md`
   (prune → golden tests under existing v1 profile IDs → corpus harness →
   improvements as profile v2).
3. **dr-providers v0.2** — the failure taxonomy must stabilize here before
   the platform starts. Independent urgency: Gemini support for inference
   credits is the nearest hard deadline.
4. **dr-graph** — small; byte-stable digest reproduction is the one
   hard obligation.
5. **platform library** — last: biggest surface, depends on the taxonomy,
   and benefits from every earlier cutover having already thinned
   `dr_dspy.platform`.

## House rules

All three existing docs converged on the same principles from different
lineages; they apply to every extraction:

1. **Frozen contracts are the migration mechanism.** Wire payloads,
   `parser_profile_id`/`scoring_profile_id`, and record ID axes play the
   same role: extract byte-identical under existing versions; land
   improvements as new opt-in versions. Old experiments stay comparable.
2. **Data over class hierarchies.** Provider config records, failure
   records, typed work items, parser profiles — not per-provider classes,
   exception taxonomies as API, `dict` payload slots, or config
   indirection.
3. **Anti-goals are first-class design.** Every predecessor died of scope
   creep; each doc names its poison explicitly (platform: no
   orchestration, no speculative backends; providers: no streaming, tools,
   or sessions yet; dr-code: no universal Task abstraction).
4. **Corpus-backed regression before improvement.** Audit corpus
   (providers), corruption corpus + golden tests (dr-code). Measurements
   settle design debates; improvements only land against a pinned
   baseline.
5. **Validate API altitude against a real consumer sketch.** Platform:
   nl_latents' seed → run → read loop, and optimizer population
   evaluation (COPRO/GEPA). Providers: whetstone's thin adapter. If the
   sketch needs library internals or grows passthrough wrappers, the
   boundary is wrong.
6. **Data for what is searched over and compared; code for what acts.**
   Graph specs (treatment/genome) are data; optimizers and workflows are
   durable code (DBOS). The same rule appears at every layer: provider
   knobs vs retry policy, treatment specs vs execution, genomes vs
   optimizers. Hash what you'd hold constant in a comparison; log what
   you'd change without forking history. Spec-language growth is admitted
   only while specs stay finite, replay-deterministic, and canonically
   hashable.

## Resolved cross-doc decisions

- **Failure taxonomy home.** dr-providers owns the `FailureClass` enum and
  the failure record; the platform imports them (it already depends on
  dr-providers); the graph runner interoperates via a locally-defined
  structural `Protocol` and depends on nothing. dr-serialize stays
  taxonomy-free.
- **Artifact offload ↔ serialization limits.** dr-serialize exposes limits
  as config (Postgres preset, not constants) precisely so the platform's
  artifact offload can change inline-payload ceilings. Pointers added in
  both docs.
- **Graph vs DBOS layering.** One workflow engine (DBOS); graphs are
  hashable treatment data, not a second execution vocabulary. Unification
  across manual and optimizer use happens at the platform facade, not in
  the spec language. See house rule 6 and `graph_runner.md`.

## Remaining gaps

- **`dr_dspy` → `whetstone` rename.** Decided: this is migration stage
  one, before any import swaps. Scope: Python package and repo naming
  only — persisted string constants (DBOS queue/workflow/step names,
  profile IDs, Alembic revision IDs, table names) stay unchanged. (Plan
  referenced in `docs/remaining-implementation-intentions.md`.)
- **Open questions carried in the per-library docs.** Conformance
  severity defaults and the shared JSON contract home (providers);
  public API shape, stability policy, cross-language canonical-JSON spec
  (serialize); builder API, additive digest policy, composition
  representation (dr-graph); libcst vs `ast.unparse`, corpus acceptance
  thresholds, Unitbench requirements (dr-code); package name, protocol
  definitions, schema ownership, projection/artifact API sketches,
  cutover plan (platform).
