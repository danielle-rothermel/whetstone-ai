# Planning Docs

Versioned planning efforts live under `docs/planning/<effort>/`.

## Layout

Each effort has a `README.md` index and numbered version directories. A version contains `plan.md` and, when reviewed, a `reviews/` packet with the exact prompts, findings, and unified feedback for that plan.

Do not create an effort packet unless an actual planning effort exists.

## Lifecycle

- **draft** — mutable during investigation
- **in-review** — frozen while review artifacts accumulate
- **reviewed** — immutable after unified feedback is complete
- **superseded** — immutable after a successor exists

Create a successor by copying the reviewed plan into the next version and applying accepted feedback there. Never edit a frozen version.

## Artifact ownership

- The issue tracker owns live questions and detailed decisions.
- Version packets are immutable plan-and-review snapshots.
- `CONTEXT.md` and `docs/adr/` remain the canonical domain language and architectural decisions.
- Reports, prototypes, and handoffs are temporary; capture their durable conclusions in a ticket, active draft, glossary, or ADR.

Review prompts must name the exact plan version and code revision or date they evaluate.
