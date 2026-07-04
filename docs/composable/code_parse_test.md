# Code Parsing + Testing Extraction

Status: draft — high-level plan only; sections will be expanded as design
discussion continues.

Goal: extract code extraction/parsing and generated-code test execution out
of `dr_dspy/humaneval/` into a standalone, provider-free, orchestration-free
library, merging the best pieces of the earlier efforts and landing the
parsing improvements identified since.

## Lineage

Four prior efforts inform this plan:

- `../nl-code` (gen 1) — multi-dataset harness. Proved that a universal
  Task/TestSuite abstraction fails (down-casting everywhere); built the
  strongest execution assets (Docker runner, ground-truth verification at
  dataset load, versioned dataset caching).
- `../code-eval` (gen 2) — extraction engine. The load-bearing ~530 lines
  (multi-strategy fan-out, repair scheduling, keep-every-candidate with
  deterministic ranking) plus a 4,100-sample synthetic corruption corpus;
  over-engineered scaffolding around it is not carried forward.
- `../dr-code` (gen 3) — packaging shape. Clean nucleus
  (`models/`, `parsing/`, `testing/`, `datasets/`, `analysis/`) with zero
  queue/provider imports; best test executor (fork sandbox). Died from
  baked-in `dr-queues`/`dr-providers` deps — both now superseded by
  whetstone's own platform/LM layers.
- `whetstone-ai` (current) — production semantics. Versioned parser/scoring
  profiles feeding frozen score-attempt IDs; JSON and field-marker
  extraction; the scoring contract experiments depend on.

## High-level design

### Home and packaging

The library lives in `dr-code`, pruned to its dependency-clean nucleus:
delete `pipeline/` (dr-queues) and `generation/` (dr-providers), absorb the
load-bearing core of `code-eval` directly rather than pinning the frozen
`code-eval==0.2.0` package. Strings in, outcomes out — no LM calls, no
queueing, no DB. Whetstone swaps its `humaneval/` package for the dependency.

*Lineage: dr-code's layout proved the boundary; its dependency baggage
proved the constraint.*

### Parsing pipeline: three stages, not one flat list

Restructure extraction into container parsers ("what envelope is this
response in?": JSON code field, Markdown document, bare text), candidate
repairs ("make this block parsable": dedent, fence strip, truncation,
import inference), and post-parse transforms ("clean a valid AST": drop
`if __name__`, trim trailing junk). Containers fan out, repairs multiply
candidates, validators filter, deterministic ranking selects — the
keep-every-candidate machinery is retained.

*Lineage: code-eval's fan-out/repair/rank engine, reorganized; whetstone's
JSON + field-marker extractors merged in (code-eval lacked both).*

### Markdown via a CommonMark library

Replace hand-rolled fence/blockquote/list regexes with a spec-compliant
parser (markdown-it-py) as the Markdown *container* strategy. Guard: never
first in the ladder — bare Python fed to a markdown parser gets mangled
(`# comment` is a heading). Hypothesis to validate on the corpus: most
prose+code responses are well-formed Markdown, so several bespoke
strategies may collapse into "markdown container + direct parse."

*Lineage: both code-eval and whetstone approximate Markdown with regexes;
code-eval's unterminated-fence special case is CommonMark-spec behavior.*

### Post-parse transforms on the AST

Semantic cleanups (`_drop_if_name`, `_drop_after_last_return`) move from
pre-parse string manipulation to AST-level transforms once a candidate
parses (stdlib `ast` suffices for correctness). Open decision: `ast.unparse`
(normalizes formatting, strips comments) vs libcst (lossless) — it silently
shifts compression-metric values, so it must be chosen deliberately.

*Lineage: whetstone's current string-splitting versions are the bug
evidence; boundary-parsing discipline applied to code text.*

### Test execution: fork sandbox + infra-vs-test contract

Replace the plain `sys.executable` subprocess runner with dr-code's
fork-based sandbox (resource limits, timeout+SIGKILL, env scrubbing).
Preserve the infra-failure vs test-failure distinction as a first-class
outcome contract — the one design all three prior generations converged on
independently. Make per-benchmark supported test shapes explicit
(`inputs_results` vs `inputs_ref_func`) instead of silently skipping.

*Lineage: nl-code's `CodeExecutionInfrastructureError` contract; dr-code's
executor and `infra_error` reclassification; the silent test-shape skip is
a three-generation-old hole to close.*

### Assertion failure messages

The runner's bare `assert actual == expected` yields empty failure
messages, compensated by re-executing the candidate to capture actuals.
Fix at the root: `assertion()` raises with a formatted
expected/got message captured at the moment of failure; delete the
re-execution path (removes a nondeterminism hazard). Diagnostics-only —
pass/fail semantics unchanged; note it in the profile changelog.

*Lineage: identified in whetstone's `task.py` runner script during this
design review.*

### Versioned profiles as the migration mechanism

`parser_profile_id` / `scoring_profile_id` feed frozen score-attempt
identity hashes. The extraction must reproduce current behavior
byte-for-byte under the existing v1 profile IDs; every improvement above
lands as a new profile version. Old experiments stay comparable, new ones
opt in.

*Lineage: whetstone's frozen ID contract; also the disciplined answer to
code-eval's shipped-but-disabled config indirection.*

### Regression harness

Port code-eval's synthetic corruption corpus (plus the real pool samples)
as the library's regression suite, alongside golden tests pinning v1
profile output. Corpus-first ordering turns the container-taxonomy and
markdown-library questions into measurements instead of design debates.

*Lineage: code-eval's corpus and recovery-rate property tests.*

### Benchmarks own their shapes

No universal Task/TestSuite base class. Each benchmark is its own module
owning its task model, test parsing, and runner; shared surface is narrow
outcome protocols. HumanEval is the first module; Unitbench lands as the
second.

*Lineage: nl-code's central failure — the thin shared Task forced
`isinstance` down-casts and never unified test-suite types.*

## Sequencing

1. Prune dr-code to the nucleus; port whetstone profiles with golden tests
   (exact-behavior, existing profile IDs).
2. Port the code-eval corpus as the regression harness.
3. Run the markdown-library experiment against the corpus; settle the
   container taxonomy.
4. Ship the improved pipeline (three-stage, markdown container, AST
   transforms, assertion messages) as profile v2; swap the fork sandbox in
   as an independent change.

## Open sections (to fill in)

- Exact public API surface and outcome models
- libcst vs `ast.unparse` decision for the metrics path
- Corpus experiment design and acceptance thresholds
- Whetstone cutover plan (import swap, test migration)
- Unitbench module requirements
