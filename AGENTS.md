# Repository instructions

## Current Priority: June 30 eval push

The active goal is to get trustworthy evaluation numbers quickly. Bias every
change toward producing those numbers, not toward finishing the platform
backlog.

Prioritize, in order:

1. Prove tiny new v1 enc-dec jobs work, then backfill v0 enc-dec rows into v1
   append-only records, validate enough to trust them, and rescore with
   `humaneval@v1`.
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
- Scope smoke tests, migration, and rescoring to enc-dec only unless the user
  explicitly expands the push to direct runs.
- Do not work on Unitbench/export, Neon publishing, generated TypeScript types,
  projection movement, first-class scoring/profile tables, provider-contract
  cleanup, engine-pooling cleanup, repo extraction, or the `dr_dspy` to
  `whetstone` rename unless they directly block the active goal.
- If you notice cleanup, hardening, or future platform work that should happen
  after this push, record it in
  `docs/remaining-implementation-intentions.md` instead of doing it now.
- Treat `docs/remaining-implementation-intentions.md` as the future-work and
  deprioritized-backlog document during this push, not as the active task list.

## Today's Enc-Dec Experiment Shape

Use one v1 enc-dec graph shape for today's smoke tests and sweeps.

- Encoder and decoder use the same model for a given run.
- Sweep over selected models and compression targets.
- Fix temperature at `0`.
- Use OpenAI endpoints for OpenAI models and OpenRouter for non-OpenAI models.
- Treat the compression target as a ratio used to derive `BUDGET`, the maximum
  encoder description length in characters, from the cleaned HumanEval ground
  truth code length.
- The encoder user prompt template is fixed:

````text
{{INSTRUCTIONS_START}}
Use at most {{BUDGET}} characters.

```python
{{GT_CODE}}
```
{{INSTRUCTIONS_END}}
````

- The worst baseline prompt variables are:
  - `INSTRUCTIONS_START="Provide a concise description of the following code."`
  - `INSTRUCTIONS_END=""`
  - `ENCODER_SYSTEM_PROMPT=""`
- Manual prompt-optimization baselines should vary only
  `INSTRUCTIONS_START`, `INSTRUCTIONS_END`, and `ENCODER_SYSTEM_PROMPT`.
- Keep the decoder system prompt and user prompt fixed across runs. The decoder
  user prompt should be equivalent to:

```text
Write functional code in Python according to the following description.
Output only the final answer, without any descriptions or surrounding
characters.

{{ENCODED_DESC}}
```

- The outputs needed for today's decision-making are HumanEval pass rates,
  simple descriptive generation statistics, and compression levels.

## Decisions To Make From Historical Data

Before launching the full sweep, make brief data checks to choose:

- The model slate: one Gemini, one GPT nano-class model, one GPT-OSS model, one
  strong open-weight model, and one weak or cheap open-weight model. Prefer
  models that were reliable in migrated enc-dec results, perform reasonably,
  and are not price outliers. Exclude models with aggressive rate limits or
  random failures.
- The compression-target range, using the previous enc-dec compression sweep as
  directional evidence. Expect differences because today's path controls prompt
  formatting directly instead of relying on DSPy reformatting.
- The repeat and task sizing needed for a stable-enough optimization signal:
  repeats per `(task_id, model, compression_target)` and total `N * D` per
  mode.
- The HumanEval task subset: prefer tasks with useful variation, especially
  tasks that are often wrong but sometimes succeed. Avoid task sets dominated
  by always-right or always-wrong items.
