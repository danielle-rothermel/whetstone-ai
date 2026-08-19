# whetstone-ai

Generic toolkit for **evaluating and optimizing** LLM prompts and programs.

Whetstone sits above the **dr-*** libraries (graphs, providers, store, serialize,
exec) and below domain-specific environments. It owns the reusable experiment
contract, batched evaluation engine, optimizer harness, and evidence/analysis
plumbing — not task datasets, domain scoring rules, or application UI.

**In scope here:** evaluation at scale, optimization (COPRO / MIPROv2 / GEPA),
and stepping through runs to inspect behavior.

**Out of scope here:** particular benchmarks or envs (those live in separate
packages or repos), one-off experiment scripts, and product-facing runners.

## Core capabilities

1. **Evaluation** — batched, efficient sweeps over candidates and tasks;
   configurable splits, graph rollouts, concurrency, and durable evidence.
   Bundled reference driver: `GraphRolloutEvalDriver`
   (`eval/drivers/graph_rollout.py`) — parallel in-process graph rollouts
   with injected `EvalProcedureRunner`.
2. **Evaluation analysis** — bootstrap confidence intervals, power analysis, and
   anchor calibration over persisted evaluation evidence (`eval/analysis/`).
3. **Optimization** — shared harness and adapters that propose candidates and
   drive evaluation intents in a loop.
4. **Sandbox & interpretation** — dry-run previews and toy-graph helpers to step
   through optimizer behavior before spending full eval budget
   (`whetstone-sandbox`).

```text
Evaluation  →  Evaluation analysis
     ↓
Optimization  →  Sandbox / interpretation
```

## dr-* libraries

| Package | Role in whetstone |
|---------|-------------------|
| **dr-graph** | Rollout graphs: LLM-call → eval nodes, executed per task row |
| **dr-providers** | Provider call configs, transport, and invocation evidence |
| **dr-store** | Content-addressed persistence for candidates, evidence, and step records |
| **dr-serialize** | Strict JSON and canonical identity hashing |
| **dr-exec** | Budgeted subprocess execution (e.g. Codex optimizer steps) |

## Stable seams

- **Experiment** — generation graph, initial/ceiling candidates, eval configs, reward policy
- **EvaluationEngine** — validates and evaluates a candidate; returns typed evidence refs
- **OptimizerAdapter** — COPRO / MIPROv2 / GEPA plug into a shared optimization harness
- **Graph rollouts** — `experiment/graph/` builds standard two-node graphs; drivers execute them per row

## Sandbox

```bash
uv run whetstone-sandbox copro --task-prompt "Say hello"
uv run whetstone-sandbox graph --run
```

Requires Python 3.13+. Optional extras: `dbos`, `postgres`, `platform`.

## Platform integration tests

Tier 2 tests exercise the dr-platform harness against Postgres + DBOS:

```bash
uv sync --extra platform
createdb whetstone_platform_test   # once, if needed
uv run pytest -m integration tests/integration/
```

Set `WHETSTONE_TEST_DATABASE_URL` when not using the default
`postgresql+psycopg:///whetstone_platform_test`. Locally, tests skip when
Postgres is unavailable; in CI they fail hard. Default `uv run pytest` excludes
integration tests via the pytest marker.
