# Implementation prompt: archive v0 surfaces

You are implementing design step 5 from `docs/append-only-eval-records-design.md`,
after core primitives, the LM/prompt boundary, and the pure graph execution core
are in place. Read these docs before editing code:

- `docs/append-only-eval-records-design.md` (step 5 and surrounding context)
- `docs/generation-experiment-design.md`

Also review the current v0 orchestration surfaces under:

- `src/dr_dspy/experiments/`
- `src/dr_dspy/harness/`
- `scripts/humaneval_dspy_eval_only_dbos_v0.py`
- `scripts/humaneval_dspy_eval_only_encdec_dbos_v0.py`
- `scripts/reclassify_encdec_generation_http_errors.py`
- `src/dr_dspy/runtime.py`
- `src/dr_dspy/lm/runner.py`, `src/dr_dspy/lm/signatures.py`, `src/dr_dspy/lm/openrouter.py`, `src/dr_dspy/lm/logging.py`

## Goal

Mark the old v0 orchestration stack clearly as legacy so new platform work does
not treat mutable prediction-row workflows, repair/status flows, or DSPy
predictor paths as the forward architecture.

This PR should answer:

```text
can a new contributor or agent tell which modules are legacy v0 orchestration,
which modules are forward graph/platform primitives, and which import boundaries
pure code must not cross — without changing live v0 behavior?
```

It should not implement schema/migrations, graph DBOS workflows, v0
migration/backfill, projection movement, Unitbench export, fair-order key
redesign, or physical relocation into a `legacy/` package unless explicitly
requested.

## Design constraints

Preserve live v0 behavior:

- Do not change `harness/ordering.stable_order_key` join semantics. v0 submit
  shuffles and repair `ORDER BY md5(...)` depend on the existing format.
- Do not change `runtime.configure_multiprocessing()` fork/spawn policy for v0
  Typer CLIs.
- Do not rewrite experiment workflows, repair logic, or prediction-row writes.
- Do not delete v0 tables or stop v0 CLIs from running.

Label and guard instead:

- Add legacy docstrings to orchestration-heavy modules and v0 scripts.
- Update README and superseded design docs so the package layout communicates
  intent.
- Split LM signaling: `lm/boundary.py` is forward; DSPy runner/signature/
  OpenRouter/logging modules are legacy-adjacent compatibility surfaces.
- Add import-boundary tests so pure modules (`graph/`, `humaneval/`,
  `eval_failures/`, `serialization.py`, `lm/boundary.py`) cannot import DBOS,
  SQLAlchemy, schema/platform packages, or legacy orchestration modules.
- Keep graph-core protections against early schema imports (`alembic`,
  `sqlalchemy`, `dr_dspy.db`, `dr_dspy.records`) in the boundary test.

## Scope

### In scope

1. Legacy module docstrings across `experiments/` and `harness/`.
2. Legacy docstrings on v0 Typer scripts and v0-adjacent LM/runtime modules.
3. README reframing: forward path vs legacy v0 data-generation surfaces.
4. Superseded banner in `docs/generation-experiment-design.md`.
5. `tests/test_platform_boundaries.py` with recursive `rglob` coverage.
6. Legacy contract tests that lock unchanged v0 ordering and multiprocessing
   behavior.
7. Tests proving legacy modules remain importable after labeling.
8. Commit this phase prompt so future reruns have an in-repo definition of
   done.

### Out of scope

- Moving code into `dr_dspy.legacy`
- Freezing v0 writes with runtime deprecation warnings
- Rewriting v0 experiments to use `lm/boundary.py`
- New domain contracts, fair-order helpers, or append-only persistence
- Changing historical v0 data or migration tooling beyond doc labels

## Module labeling expectations

At minimum, these surfaces should read as legacy or legacy-adjacent in module
docstrings and README prose:

- `experiments/*`
- `harness/*`
- `scripts/*_v0.py` and repair CLIs tied to mutable prediction rows
- `runtime.py` (v0 Typer entrypoint helpers)
- `lm/runner.py`, `lm/signatures.py`, `lm/openrouter.py`, `lm/logging.py`
- `lm/__init__.py` should explain the boundary vs compatibility split

Forward modules to keep unmarked as legacy:

- `graph/*`
- `humaneval/*` primitives
- `eval_failures/*` (except documented persistence-boundary exceptions)
- `serialization.py`
- `lm/boundary.py`
- `lm/utils.py` shared helpers consumed by the forward boundary

## Tests

Add or extend tests that prove archive contracts rather than only today's import
graph:

1. `tests/test_platform_boundaries.py`
   - pure modules must not import forbidden platform/runtime/legacy surfaces
   - pure modules covered recursively (`rglob`)
   - DSPy imports forbidden in DSPy-free pure paths (`graph/`, `humaneval/`,
     `eval_failures/`, `lm/boundary.py`) except documented serialization cases

2. `tests/test_harness_ordering.py`
   - lock legacy join-based `stable_order_key` format used by v0 submit/repair

3. `tests/test_runtime_multiprocessing.py`
   - lock fork-on-non-Windows / spawn-on-Windows policy for v0 script entrypoints

4. `tests/test_archive_contracts.py`
   - legacy orchestration modules remain importable
   - key legacy modules and README sections stay consistently labeled

Keep existing harness regression tests (for example enqueue-race coverage) and
label them as legacy contract tests when touched.

## Definition of done

- README and design docs distinguish forward graph/LM primitives from legacy v0
  orchestration without implying v0 repair/status flows are the build target.
- Legacy docstrings exist on orchestration modules, v0 scripts, runtime, and
  v0-adjacent LM modules listed above.
- Pure import-boundary test passes and uses recursive discovery.
- Legacy ordering and multiprocessing behavior have explicit contract tests.
- Legacy modules import cleanly in CI.
- No behavioral changes to v0 ordering, multiprocessing, workflow logic, or DB
  writes beyond documentation and guardrails.
- This prompt is committed under `docs/prompts/archive-v0-surfaces.md`.

## PR notes

Call out deferred follow-ups explicitly:

- optional physical move to `dr_dspy.legacy`
- optional runtime deprecation warnings on v0 CLIs
- v0 write freeze before migration cutover
- new fair-order helper for design step 9 (do not reuse legacy
  `stable_order_key`)
