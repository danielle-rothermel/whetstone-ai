---
name: danielle-diff-check
description: Use before finalizing code changes, refactors, reviews with suggested patches, commits, or pull requests. Scans the final diff for unintended rewrites, unrelated cleanup, behavior changes, weak verification, Python convention drift, and other issues Danielle wants caught before handoff.
---

# Danielle Diff Check

## Final Diff Check

Before finishing, scan the final diff for issues that should be fixed or called out.

### Scope and intent

- Broad rewrites where a local change would have worked.
- Drive-by dead-code removal, cleanup, formatting, or refactors unrelated to the request.
- Unrelated changes to behavior, observability, diagnostics, interfaces, or user workflow.
- Compatibility shims, aliases, adapters, migration layers, or dual paths where a clean break would be clearer.

### Structure and contracts

- Speculative abstractions, base classes, extension points, factories, or configuration layers.
- Mixed responsibilities across domain logic, adapters, persistence, UI, transport, and configuration.
- Public contracts that expose persistence, framework, raw row, or external API shapes.
- Privacy signaled mainly through `_` class-name prefixes instead of exports or module boundaries.
- Replaceable infrastructure hard-coded deep inside domain logic.
- Duplicate or near-duplicate functions with unclear job boundaries.
- Comments that narrate code instead of explaining constraints.

### Domain data

- Inline domain literals, duplicated strings, or magic values that should be constants.
- String `Literal[...]` annotations that should be `StrEnum`.
- Boolean flags that hide modes, strategies, or behavior variants.
- Raw external values passed into business logic instead of parsed at the boundary.
- New or modified code without explicit types, or public API types that hide absence, failure, optional values, or dynamic data.
- Base classes or compositional base models used only for field, helper, config, or wiring reuse.
- New parsing logic hidden inside dense expressions.

### Errors and resilience

- Discarded-return validation side effects.
- Swallowed errors, log-and-reraise catches, genericized exceptions, or unchained translated errors.
- Retries for permanent failures, unbounded retries, invisible retry behavior, or external calls without explicit timeouts.
- Defaults or degraded behavior used as silent recovery from missing data, failed operations, or correctness failures.
- Batch operations that abort unnecessarily, silently drop failed items, or lack structured success/failure summaries.

### Verification

- Missing, brittle, broad, happy-path-only, or implementation-locking verification.
- Hidden or uncontrolled nondeterminism from time, randomness, generated IDs, concurrency, ordering, external state, or eventual consistency.
- Scripts that do substantial work without progress logging or stable verification evidence.
- Demo scripts without clear self-validation or corresponding `TESTING.md` success criteria.
- Direct `json` imports or calls.
