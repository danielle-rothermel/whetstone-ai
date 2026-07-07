> **RETIRES AT D3 MERGE** — this documents the in-flight composable migration (draft PR #4) and becomes historical when it merges. Tracked in Linear S17/DEV-21.

# Graph Runner Extraction

Status: draft — high-level plan; sections will be expanded as design
discussion continues.

Goal: extract `dr_dspy/graph/` (models, execution, hashing) into a
standalone library — repo/package **`dr-graph`** (decided) — a **hashable
treatment-description language plus a pure interpreter**. Not a workflow
engine.

## The rule this library exists to serve

**Whatever is searched over must be data; whatever does the searching is
code.** Graphs are the searched-over layer — experiment conditions today,
optimizer genomes (GEPA / RL over prompt-and-structure) tomorrow. An
optimizer must mutate, canonically hash, dedupe, diff, and persist lineages
of the inner loop; none of that is possible against hard-coded workflows.
Optimizers and platform workflows are ordinary durable code (DBOS) that
read and write these specs.

*(Lineage: `optimization/copro.py` is the existence proof — an outer loop
written as plain code manufacturing graph specs and submitting them through
the platform. The industry poles: LangGraph made graphs programmable and
gave up canonical hashability; DBOS/Temporal made programs durable so
workflow DSLs aren't needed. This design keeps both poles and refuses to
merge them.)*

## What the library is

- A spec vocabulary for finite, hashable computation structures:
  `GraphSpec` (nodes + terminal node), `NodeSpec`/`NodeConfig` (typed
  input/output fields, input bindings, parameters, open metadata), and the
  binding-ref grammar (`task.prompt`, `encoder.description`).
- Structural validation: acyclicity, ref legality, binding targets,
  terminal-node existence, topological ordering — trust established before
  any money is spent executing.
- Pure, sequential, deterministic execution: `execute_graph(graph, inputs,
  run_node)` resolves each node's inputs from external inputs + upstream
  outputs and calls the **injected** `run_node` callback. Failed nodes mark
  downstream dependents blocked; independent branches continue; the result
  aggregates per-node outcomes into a graph status with terminal output.
- Canonical spec digest (`graph_digest`) — the identity of the treatment.

Durability composes in from outside: whetstone's `run_node` wraps each node
call in DBOS steps (throttle preflight → durable sleep → LM call), so crash
recovery resumes at node granularity without re-buying completed nodes.
The runner's deterministic walk is what makes DBOS replay line up — purity
is a contract, not an aesthetic.

## What the library is not (anti-goals)

- **Not a workflow engine.** No durability, retries, scheduling, or
  persistence — the callback owns all of that. *(Lineage: dr-queues shipped
  an anti-goals list, grew a workflow engine anyway, and died.)*
- **Not an LM framework.** No prompts, providers, or model configs. App
  conventions ride in open node `metadata`, parsed by typed app-side views
  (whetstone's `NodePromptSpec` pattern is the recommended idiom).
- **Not a workflow DSL.** No unbounded control flow, no persistent state —
  see the admission test below. A spec language rich enough to express an
  optimizer is a programming language interpreted inside DBOS, recreating
  what DBOS exists to eliminate.
- **Sequential execution is a feature.** Deterministic order serves DBOS
  replay and reproducibility. Parallelism across nodes or graphs belongs to
  the caller (the platform already runs whole graphs concurrently).

## v1 scope and generalizations at extraction

Extract `graph/models.py`, `graph/execution.py`, `graph/hashing.py` nearly
as-is, with these changes:

- **Open strings for `op` and field types.** Nothing dispatches on them at
  runtime (the callback does); they are spec content that the digest must
  distinguish. Closed enums would make every new node kind a library
  release.
- **Parameterized reserved input namespace** (default `"task"`). Digest-
  affecting, so decided at extraction time or never.
- **`ClassifiedFailure` as a structural `Protocol`** (`failure_class`,
  `error_type`, `metadata`, `underlying`), replacing the current duck-typed
  introspection. Defined locally — structural typing needs no shared home;
  canonical failure-class values live in dr-providers, with no dependency
  from this package.
- **Neutral `node()` / `graph()` builder helpers** (bindings + output field
  + metadata) so consumers stop rewriting spec-assembly boilerplate.
  Prompt-aware builders (`spec_builder.py`) stay app-side.
- **Subgraph composition** — promoted into v1 scope: optimizers mutate and
  reuse subgraphs across candidates, and nested treatment structure is
  still treatment data.
- **Completed-nodes resume hook**: optional `completed` mapping of node id
  → prior output; those nodes are skipped and their outputs participate in
  binding resolution. Enables cross-attempt reuse (fail at node 10 of 20 →
  new attempt re-buys nothing before node 10). Reuse *policy* — which prior
  outputs are safe, provenance of "chimera" runs assembled across attempts
  — is platform/app-side, recorded per node.

## Growth path and the admission test

Designed-for growth, driven by the target inner loop (multi-branching
pathways that converge): **conditional routing** (static alternative edge
sets with runtime gates), **bounded iteration** (repeat a subgraph ≤ N
times with an exit condition), **dynamic map/fan-out** (subgraph over a
runtime list, bounded).

Admission test — a feature enters the spec language only if specs remain:

1. **finite** (a bounded structure, unrollable),
2. **deterministic to replay** (DBOS step sequences line up), and
3. **canonically hashable** (optimizers can dedupe and compare).

The outer loop pins this line: the moment inner-loop structure can't be
hashed, the optimizer loses its genome representation and experiment
identity collapses. Unbounded loops, persistent cross-run state, and
learning updates stay code.

## Identity obligations

- Extraction reproduces current digests **byte-for-byte** — `graph_digest`
  is a frozen axis of `prediction_id` (same discipline as dr-code's profile
  IDs and dr-providers' wire payloads).
- The digest covers the full spec including `metadata`: prompt templates
  are treatment variables, so changing a prompt is a new condition. Stated
  deliberately, not incidentally.
- New spec-language features must not change digests of existing specs
  (additive canonicalization policy — open section).

## Whetstone impact

Import swap for `dr_dspy/graph`. `spec_builder.py`, `prompts.py`, and the
DBOS `run_node` wrapping in `graph_workflow.py` stay app-side unchanged.

## Consumers

Whetstone generation workflows (today), COPRO (existing spec-manufacturing
outer loop), GEPA/RL optimizers (target), and the platform's
population-evaluation use case (see `platform.md`).

## Open sections (to fill in)

- Neutral builder API surface.
- Additive canonicalization / digest policy for growth features.
- Subgraph composition representation (inline vs referenced-and-digested).
- Conditional-routing and bounded-iteration spec shapes.
