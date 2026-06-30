# Repository instructions

## Current Priority: June 30 eval push

The active goal is to get trustworthy evaluation numbers quickly. Bias every
change toward producing those numbers, not toward finishing the platform
backlog.

Prioritize, in order:

1. Backfill v0 direct and enc-dec rows into v1 append-only records, validate
   enough to trust them, and rescore with `humaneval@v1`.
2. Choose the final model set for today's experiments.
3. Run an enc-dec budget sweep baseline with the chosen models, using roughly
   N=3-5 repetitions for a solid first evaluation point.
4. Get the simplest useful COPRO-style loop running so the presentation has
   numbers.

During this push:

- Keep the original v0 Postgres tables around as read-only backup unless the
  user explicitly asks to drop or rename them.
- Make only code changes required for backfill, scoring, model selection, the
  budget sweep, or the minimal COPRO experiment loop.
- Do not work on Unitbench/export, Neon publishing, generated TypeScript types,
  projection movement, first-class scoring/profile tables, provider-contract
  cleanup, engine-pooling cleanup, repo extraction, or the `dr_dspy` to
  `whetstone` rename unless they directly block the active goal.
- If you notice cleanup, hardening, or future platform work that should happen
  after this push, record it in
  `docs/remaining-implementation-intentions.md` instead of doing it now.
- Treat `docs/remaining-implementation-intentions.md` as the future-work and
  deprioritized-backlog document during this push, not as the active task list.
