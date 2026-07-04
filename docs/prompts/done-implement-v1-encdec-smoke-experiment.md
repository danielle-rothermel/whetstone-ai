# Prompt: implement tiny v1 enc-dec smoke experiment support

**Status (2026-06-30):** Addressed on branch `today_exp` — humaneval enc-dec spec
builder, composable configs under `configs/`, live smoke r1/r2 documented in
[`docs/testing_logs.md`](../testing_logs.md). Follow-on platform fixes (model
strings, `per_test_results` byte cap, `rescore` await, `DATABASE_URL`
normalization) landed in the r2 smoke pass.

You are working in `/Users/daniellerothermel/drotherm/repos/whetstone-ai` on
branch `today_exp`.

First read:

- `AGENTS.md`
- `docs/remaining-implementation-intentions.md`
- `docs/v0-migration-completion-checklist.md`

The current push is explicitly about getting evaluation numbers quickly. Do not
do platform cleanup, projection work, Unitbench/export work, migration deletion,
repo rename work, or direct-mode work. Implement only what is needed to support
a tiny new v1 enc-dec job with the experiment shape below.

## Goal

Make the existing v1 platform able to build and run a tiny enc-dec HumanEval
experiment whose encoder/decoder prompt shape matches today's target design.
This is a pre-migration smoke path: after your change, another agent will run a
tiny real v1 enc-dec job and review it before any v0 migration/backfill.

## Required experiment shape

Scope is enc-dec only.

- Encoder and decoder use the same model for a given run.
- Sweep over selected models and compression targets.
- Temperature is fixed at `0`.
- OpenAI models should use the OpenAI endpoint; non-OpenAI models should use
  OpenRouter.
- Compression target is a ratio used to derive `budget_chars` from the cleaned
  HumanEval ground-truth code length.
- The encoder prompt variables are:
  - `instructions_start`
  - `budget`
  - `gt_code`
  - `instructions_end`
- The baseline values are:
  - `instructions_start`: `Provide a concise description of the following code.`
  - `instructions_end`: empty string
  - encoder system prompt: empty string
- The fixed encoder user prompt should render as:

````text
{instructions_start}
Use at most {budget} characters.

```python
{gt_code}
```
{instructions_end}
````

- The fixed decoder user prompt should render as:

```text
Write functional code in Python according to the following description.
Output only the final answer, without any descriptions or surrounding
characters.

{encoded_desc}
```

Use Python `str.format_map` placeholders internally, because the current prompt
runtime formats node prompts that way. Do not introduce a second templating
system unless it is truly unavoidable.

## Current code facts to verify before editing

Relevant files:

- `src/dr_dspy/platform/spec_builder.py`
- `src/dr_dspy/platform/prompts.py`
- `src/dr_dspy/platform/graph_workflow.py`
- `src/dr_dspy/platform/node_execution.py`
- `src/dr_dspy/humaneval/task.py`
- `src/dr_dspy/humaneval/metrics.py`
- `tests/test_platform_spec_builder.py`
- `tests/test_platform_prompts.py`
- `tests/test_platform_graph_workflow.py`

Known state:

- `build-specs`, `submit-jsonl`, `worker`, and `rescore` already exist.
- The graph runtime only passes `spec.task.inputs.values` into graph execution.
- Prompt formatting currently sees only node-bound inputs.
- Current enc-dec spec builder hardcodes encoder prompt `Describe {prompt}` and
  decoder prompt `Write code from {description}`.
- Current task inputs are `prompt`, `test`, and `entry_point`; ground-truth code
  is metadata only.
- Same-model encoder/decoder provider configs are valid if both have
  `config_id`s, for example `encoder` and `decoder`.
- Scoring already records HumanEval pass/fail outcomes, node-output stages,
  character counts, and compression metrics.

## Blocking pieces to implement

Implement the smallest coherent change that supports the smoke experiment.

1. Add a today-specific enc-dec spec construction path.
   - Prefer extending `ExperimentSpecConfig` and `iter_experiment_specs` in
     `spec_builder.py` only as much as needed.
   - Keep the existing minimal fixture behavior working.
   - Do not rewrite the whole spec builder.

2. Materialize encoder prompt inputs into `TaskInputsPayload`.
   - Add `gt_code`, `budget`, `instructions_start`, and `instructions_end` to
     task inputs for the new enc-dec experiment shape.
   - Prefer cleaned ground-truth code from
     `HumanEvalTask.ground_truth_code_without_comments`; fall back to
     `HumanEvalTask.ground_truth_code` only if cleaned code is unavailable.
   - Compute `budget` from the compression target ratio and the selected GT code
     character length. Use a small clearly named helper.
   - Preserve `prompt`, `test`, and `entry_point` because scoring and other
     code still depend on them.

3. Build the target encoder/decoder graph.
   - Encoder node should bind the four task fields above and output
     `description`.
   - Decoder node should bind `encoded_desc` from `encoder.description` and
     output `code`.
   - Encoder and decoder should keep `provider_config_id` values `encoder` and
     `decoder`.
   - Encoder/decoder system prompts should be fixed or empty as described
     above.

4. Support same-model enc-dec model configs for smoke/sweep use.
   - It is acceptable for the initial JSON config to list both provider entries
     explicitly with the same model and `temperature: 0`.
   - If adding model-list expansion is simpler and well-contained, keep it
     narrow. Do not build a general experiment DSL today.

5. Add a tiny smoke fixture config if useful.
   - Put it under `tests/fixtures/experiment_configs/`.
   - It should be one enc-dec model, one compression target, one task, one
     repetition, temperature `0`.

## Verification required

Add focused tests. Avoid broad brittle tests.

Required minimum:

- A spec-builder unit test that confirms the new enc-dec spec:
  - has layout `encdec`
  - has encoder and decoder provider configs
  - can use the same model for encoder and decoder
  - has temperature `0`
  - includes `gt_code`, `budget`, `instructions_start`, and `instructions_end`
    task inputs
  - binds encoder prompt inputs to those task fields
  - binds decoder prompt input to `encoder.description`
- A prompt-formatting or mocked graph test that proves the target templates bind
  and run without missing-input errors.

Run targeted tests, at least:

```bash
uv run pytest tests/test_platform_spec_builder.py tests/test_platform_prompts.py
```

If you touch graph execution or workflow behavior, also run the most relevant
mocked graph tests:

```bash
uv run pytest tests/test_platform_graph_workflow.py
```

Do not run a live provider smoke test and do not run migration/backfill. Stop
after code and targeted tests are ready.

## Constraints

- Keep changes narrowly scoped.
- Prefer Pydantic models over dataclasses.
- Use `uv run python` / `uv run pytest`.
- Do not add dependencies.
- Do not delete v0 migration code or v0 tables.
- Do not add projection movement, Unitbench export, or long-term cleanup.
- Do not implement direct-mode support for this experiment shape unless it is
  incidental to a tiny shared helper and does not expand scope.

## Expected handoff

When done, report:

- Files changed.
- Exact experiment config shape now supported.
- Tests run and their result.
- Any remaining blocker before a real tiny v1 enc-dec smoke job can be run.

