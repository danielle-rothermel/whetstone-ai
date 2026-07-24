# Whetstone

Whetstone evaluates prompt candidates through typed, content-addressed
configuration and evidence records. Environment sampling, evaluation,
optimization, provider execution, and reporting have explicit ownership
boundaries.

## Canonical runner

`whetstone.runner.optimization_run` advances Identity, COPRO, MIPROv2, GEPA,
and Codex through the shared optimization harness and their algorithm-specific
adapters. `whetstone.runner.cell` selects the first optimizer-ordered proposal
from the durable `OptimizationResult`, runs official evaluation, computes
task-resampled confidence intervals, and projects a validated cell record.

Each run uses a file-backed SQLite object store. A cell artifact records the
typed `OptimizationResult` reference separately from its readable trace path.
Step requests, results, candidates, dispositions, tool/evaluation evidence,
and prompt evidence remain addressable through canonical durable references.

The CLI accepts an explicit typed cell factory:

```sh
whetstone-validate cell --factory package.module:build_cell
whetstone-validate cell --factory package.module:build_dry_cell --dry
whetstone-validate status --root artifacts/run
```

Dry cells use the same runner, harness, adapters, persistence, and reporting
seams as live cells; their factories inject scripted provider, proposer, or
Codex process boundaries.

## Validation

```sh
./scripts/ci/lint.sh
./scripts/ci/unit.sh
./scripts/ci/dbos.sh
```

The unit shard uses four xdist workers with `loadgroup`. Tests sharing the
PostgreSQL/DBOS schema run in the serial shard. Each shard is also a separate
CI matrix job.
