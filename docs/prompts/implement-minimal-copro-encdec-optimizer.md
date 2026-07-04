# Prompt: implement minimal COPRO-style enc-dec optimizer

You are working in `/Users/daniellerothermel/drotherm/repos/whetstone-ai` on
branch `today_exp`.

First read:

- `AGENTS.md`
- `docs/testing_logs.md`
- `docs/remaining-implementation-intentions.md`
- `docs/completed-design-and-implementation-choices.md`
- `configs/experiments/humaneval_encdec_smoke.json`
- `src/dr_dspy/platform/spec_builder.py`
- `src/dr_dspy/platform/worker.py`
- `src/dr_dspy/platform/submission.py`
- `src/dr_dspy/platform/rescoring.py`
- `src/dr_dspy/analysis/frames.py`
- DSPy's installed COPRO reference:
  `.venv/lib/python3.13/site-packages/dspy/teleprompt/copro_optimizer.py`

The current goal is to have a working COPRO-style loop on the current v1
enc-dec platform for today's presentation. This is not a full optimizer
framework. Build the smallest useful implementation that can propose prompt
variants, evaluate them through v1 generation/scoring, and report the best
candidate.

## Conceptual target

DSPy COPRO does this:

1. Start with the current instruction and output prefix.
2. Ask a prompt model to propose `breadth - 1` alternative
   instruction/prefix pairs.
3. Include the original instruction/prefix as a candidate.
4. Evaluate each candidate on a train/dev set.
5. Keep the best candidate.
6. At the next depth, ask the prompt model for new candidates using previous
   attempts and validation scores as context.
7. Return the best program and candidate history.

Our equivalent is:

- Treat the "program" as a v1 HumanEval enc-dec experiment shape.
- Optimize only encoder prompt variables:
  - `instructions_start`
  - `instructions_end`
- Keep fixed:
  - encoder user prompt template
  - decoder prompt
  - encoder system prompt, unless existing config support makes it trivial
  - model routing
  - compression-target budget logic
  - temperature `0`

Do not reintroduce DSPy into the runtime path. Use DSPy only as a reference for
the algorithm.

## Minimal deliverable

Add a small operator script plus focused helper module:

```text
src/dr_dspy/optimization/
  __init__.py
  copro.py
scripts/optimization/
  run_copro_encdec.py
```

The script should be runnable like:

```bash
uv run python scripts/optimization/run_copro_encdec.py \
  --model-config configs/models/gpt54-nano-openai.json \
  --split configs/splits/tiny.json \
  --compression-target 0.5 \
  --breadth 3 \
  --depth 2 \
  --repeats 1 \
  --output-dir artifacts/optimization/copro_smoke
```

It is acceptable if the first live run is tiny. The important claim is that the
loop is integrated with the v1 platform end to end.

## Candidate model

Use Pydantic models, for example:

- `CoproCandidate`
  - `candidate_id`
  - `depth`
  - `parent_candidate_id`
  - `instructions_start`
  - `instructions_end`
  - `proposal_source`
  - `instructions_digest`
- `CoproAttempt`
  - candidate fields
  - experiment name
  - scoreable count
  - pass count
  - pass rate
  - generation/score error counts
- `CoproRunConfig`
  - model config path
  - split path
  - compression targets
  - breadth/depth/repeats
  - prompt model/provider config
  - output dir

Keep these models in `src/dr_dspy/optimization/copro.py`.

## Candidate proposal

Implement two proposal modes:

1. `--proposal-mode manual`
   - No LLM optimizer call.
   - Generate a small fixed set of candidate instruction pairs including the
     baseline.
   - This guarantees a working local smoke path even if prompt-model calls fail.

2. `--proposal-mode lm`
   - Use the existing provider boundary to call a prompt model.
   - Ask for strict JSON containing candidate objects with
     `instructions_start` and `instructions_end`.
   - Include prior attempts and scores after depth 0, following DSPy COPRO's
     "attempted instructions plus resulting scores" idea.

Default to `manual` if that is safer for immediate demo reliability. If `lm` is
implemented, keep parsing strict and fail clearly on malformed JSON.

Suggested initial proposal prompt:

```text
You are optimizing encoder instructions for a code-compression experiment.
The encoder sees Python ground-truth code and must describe it within a fixed
character budget. The decoder will write Python code from the description.

Return JSON only:
{"candidates":[{"instructions_start":"...","instructions_end":"..."}]}

Baseline instructions_start:
{baseline_start}

Baseline instructions_end:
{baseline_end}

Prior attempts, ordered by score:
{attempt_history}
```

## Evaluation path

For each candidate/depth:

1. Build v1 prediction specs using the existing HumanEval enc-dec shape.
2. Put the candidate values into `humaneval_encdec.instructions_start` and
   `humaneval_encdec.instructions_end`.
3. Add optimizer metadata to dimensions so rows are traceable:
   - `optimizer`: `"copro_minimal"`
   - `copro_run_id`
   - `candidate_id`
   - `candidate_depth`
   - `parent_candidate_id`
   - `instructions_digest`
   - `compression_target`
4. Use existing platform insertion/submission/generation/scoring paths.
5. Aggregate score attempts with existing analysis helpers.

Prefer function calls over shelling out when the existing code is easy to use.
Shelling out to existing CLIs is acceptable if it is much faster and more
reliable for today; make the command log explicit either way.

The output experiment names should be deterministic and isolated, for example:

```text
copro_minimal_{run_id}_d{depth}_c{candidate_id}
```

or one shared experiment name with candidate metadata in dimensions. Choose the
simpler approach that the analysis helpers can read reliably.

## Scoring / selection

Candidate score for today:

```text
pass_rate over score-success rows
```

Tie-breakers:

1. larger scoreable count
2. fewer generation/score errors
3. shorter combined instructions

If data is too sparse, still select a best candidate and report the caveat.

## Outputs

Write artifacts under `--output-dir`:

- `candidates.jsonl`
- `attempts.csv`
- `best_prompt.json`
- `summary.md`
- `commands.log` if shell commands are used

The summary should include:

- run config
- candidate table
- best candidate
- caveats
- exact command used

Also append a short entry to `docs/testing_logs.md` after a tiny run:

- command run
- breadth/depth/repeats
- model/split/compression target
- candidates evaluated
- best candidate/pass rate
- artifact paths
- verdict/caveats

## CLI options

Support at least:

- `--model-config`
- `--split`
- `--compression-target` repeatable or one value
- `--breadth`
- `--depth`
- `--repeats`
- `--proposal-mode manual|lm`
- `--prompt-model` for LM proposal mode
- `--prompt-provider-kind`
- `--prompt-endpoint-kind`
- `--output-dir`
- `--database-url`
- `--env-file`
- `--generation-worker-concurrency` or document that an external worker must
  already be running
- `--rescore-max-in-flight` if the current rescore implementation supports it;
  otherwise omit and document the current limitation
- `--dry-run`

Use Typer for the script.

## Verification

Add focused tests for the helper layer:

- baseline/manual candidates include the original prompt
- candidate IDs/digests are stable
- attempt history renders previous candidates and scores
- candidate metadata is added to dimensions
- best-candidate selection handles ties deterministically
- malformed LM proposal JSON fails clearly if LM mode is implemented

Run targeted tests:

```bash
uv run pytest tests/test_optimization_copro.py -q
```

If you touch spec-builder or platform submission/scoring code, run the relevant
existing tests too.

Run a tiny smoke command before handoff. It can be as small as:

```bash
uv run python scripts/optimization/run_copro_encdec.py \
  --model-config configs/models/gpt54-nano-openai.json \
  --split configs/splits/tiny.json \
  --compression-target 0.5 \
  --breadth 2 \
  --depth 1 \
  --repeats 1 \
  --proposal-mode manual \
  --output-dir artifacts/optimization/copro_smoke
```

If the live run cannot complete because another long DBOS job is using the same
database or runtime, run unit tests and document the exact blocker in the
testing log.

## Constraints

- Enc-dec only.
- Do not change stable IDs.
- Do not change scoring semantics.
- Do not change DB schema.
- Do not build projection movement.
- Do not build a general optimizer framework.
- Do not add notebook or dashboard work.
- Do not reintroduce DSPy as a runtime dependency.
- Keep changes small and presentation-oriented.

## Handoff

When done, report:

- files changed
- command implemented
- whether manual mode works
- whether LM proposal mode works
- tests run
- tiny smoke result or blocker
- artifact directory
- the sentence we can truthfully put on a slide about the working COPRO loop
