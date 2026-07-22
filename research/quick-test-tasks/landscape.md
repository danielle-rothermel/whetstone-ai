# Quick-Test Task Landscape

Date: 2026-07-21 · whetstone-ai

A review of the whole design space searched for the **quick-test task**: the one
cheap, single-call, exact-scored task family that validates the prompt-optimizer
implementations (Eval identity, COPRO, MIPROv2, GEPA, Codex CLI agent) and the
shared execution contract before the expensive HumanEval+ code-compression
experiment.

**Authoritative rubric:** [`quick-test-rubric.html`](../../design/quick-test-rubric.html) — read
first. In brief: one Generation node + one Eval node; strict per-example 0/1 exact
match by a simple deterministic function (no sandbox, no partial credit inside an
example); bounded MC / few-word outputs; default prompts **well below a KNOWN
ceiling** with the gap closing **INCREMENTALLY** (not one-shot guessable);
difficulty that decomposes into **independent latent rules**; a synthetic/generatable
pool with exact ground truth (hundreds of instances, contamination-resistant);
diagnostic failures for a reflection LM; few-shot demos that measurably help; and
prompt-quality differences resolvable on 10-20 task evals at temperature 0. A
lower-priority tie-breaker rewards tasks sitting in active 2025-2026 research so the
quick test could grow into its own direction.

## Inputs and pointers

- Idea space: [`brainstorm.md`](brainstorm.md) — 76 brainstormed ideas, 8 themes (A-H).
- Merged pool: [`candidates-merged.md`](candidates-merged.md) — 76 brainstorm + 108
  literature candidates → 24 merged candidates (9 clusters dropped).
- Tie-breaker scan: [`trends-2025-2026.md`](trends-2025-2026.md) — 9 active research areas.
- Prompt-optimizer methods: [`research-prompt-opt-literature.md`](research-prompt-opt-literature.md).

Per-domain deep-dive research docs (detail behind each cluster):

- [`research-synthetic-rule-induction-and-composition.md`](research-synthetic-rule-induction-and-composition.md)
- [`research-constructed-notation-arithmetic-and-nons.md`](research-constructed-notation-arithmetic-and-nons.md)
- [`research-formal-grammars-term-rewriting-and-logic.md`](research-formal-grammars-term-rewriting-and-logic.md)
- [`research-deterministic-simulation-state-machines-.md`](research-deterministic-simulation-state-machines-.md)
- [`research-constructed-language-translation-and-lay.md`](research-constructed-language-translation-and-lay.md)
- [`research-classification-under-nonstandard-remappe.md`](research-classification-under-nonstandard-remappe.md)
- [`research-structured-extraction-and-canonicalizati.md`](research-structured-extraction-and-canonicalizati.md)
- [`research-instruction-following-and-constrained-ge.md`](research-instruction-following-and-constrained-ge.md)

---

## The domains searched

Two independent search passes were merged. The **brainstorm** pass generated 76
ideas across model lenses (`claude-opus-4-8[1m]`, `kimi-k2p7`, `deepseek-v4-flash`,
`minimax-m3`, `gpt-5.x-codex`, and first-principles design). The **literature** pass
contributed 108 candidates anchored in 2024-2026 benchmarks. Both were filtered
through the same two hard rubric constraints — strict per-example 0/1 exact match
(no partial credit inside an example) and bounded MC / few-word output, single call,
no sandbox — which forced every "per-field partial scoring" idea to be rewritten as a
single-field / single-token query, with incrementality relocated to **rule strata
across the task pool** rather than partial credit within an instance.

The space that resulted spans eight domains, each a recognizable slice of the
synthetic-verifiable-task landscape:

1. **Constructed notation / invented arithmetic** (Theme A) — invented glyphs, bases,
   operators, units; numeric or short-string output under nonstandard conventions.
2. **Constructed language / morphology / translation** (Theme B) — artificial
   grammars mapping marked tokens to a canonical output template.
3. **Ciphers / composable reversible transforms** (Theme C) — layered invented
   transforms decoded to a closed vocabulary.
4. **Formal grammars, rewrite systems, logic/deduction** (Theme D) — well-formedness
   judgments, term rewriting to normal form, truth evaluation, implication compliance.
5. **Deterministic simulation & state machines** (Theme E) — invented machines, games,
   protocols, automata run forward to a bounded answer.
6. **Classification under remapped label semantics** (Theme F) — opaque codebooks and
   override policies that defeat the pretrained label prior.
7. **Structured extraction / canonicalization / serialization** (Theme G) — house-style
   normalizers producing exact canonical strings; the largest cluster and the closest
   analog to the project's own dr-serialize canonical-JSON work.
8. **Instruction following / constrained generation** (Theme H) — simultaneous
   constraints or an authority policy over directives.

## The clusters of task types found

After merging near-duplicates, the 24 surviving candidates organize into a handful of
structural clusters that cut across the eight domains:

- **Invented-notation evaluators** (c01 glyph base arithmetic, c02 mixed-radix units,
  c03 keyed operators, c10 interlocking calendars/clocks) — parse a per-instance legend,
  then evaluate under nonstandard place/precedence/modulus rules.
- **Abstract symbol-system execution** (c04 term rewriting, c05 FSM/protocol/pointer
  state tracking, c17 invented-gate circuits, c19 grid-world, c20 micro-game turns) —
  run a latent machine to a bounded derived fact.
- **Constraint / violation diagnosis** (c06 artificial-grammar violation class, c16
  Wason implication compliance, c18 synthetic deduction, c22 stacked verifiable
  constraints) — classify well-formedness or compliance against conjoined rules.
- **Remapped-label & policy classification** (c13 decision-list triage, c14 instruction-
  hierarchy priority ledger, c15 opaque-codebook labels, c21 relational nonce-graph QA) —
  emit an MC letter or closed-vocab token under an opaque codebook or precedence policy.
- **Canonicalization / extraction under invented conventions** (c11 house-convention
  canonicalizer, c12 sigil-schema extraction, c24 composite-key sorting) — produce an
  exact canonical string under bespoke house rules.
- **Rule-induction from I/O** (c23 hidden-rule string-transform ladder, c08 Rosetta MC
  puzzles, c09 cipher-chain decode, c07 case-marked conlang translation) — infer a hidden
  transform from demos and apply it, making the MIPROv2 demo axis load-bearing.

## Full ranked table of all scored candidates

Score is the summed rubric fit (14 criteria, 0-2 each, max 28). Tie-breaker (0-2,
separate from the rubric) reflects 2025-2026 research activity. Verified finalists
carry the highest scores; verdicts appear in the notes column where a lens flagged a
repairable weakness (none fatal).

| ID | Name | Domain | Score | TB | One-line note |
|----|------|--------|:----:|:--:|---------------|
| c05 | Deterministic Execution & State Tracking (FSM/protocol/pointer) | state-machine simulation | 28 | 2 | Most diagnostic cluster; both lenses refuted-but-repairable — conventional rules are one-shot guessable and the no-CoT vs long-trace dilemma threatens the known ceiling. |
| c06 | Artificial-Grammar Violation Classification | formal-language membership (MC) | 28 | 2 | Clean MC strata, 25% guess floor; needs non-mnemonic rule instantiations and label recomputation to keep the ladder honest. |
| c11 | House-Convention Canonicalizer | serialization/normalization | 28 | 2 | Mirrors dr-serialize; drop the check-digit/global-sort strata a small model can't execute, keep tables wider than any demo batch. |
| c12 | Sigil-Schema Field Extraction & Dialect Routing | structured extraction | 28 | 2 | Strictly one field per instance; drop the arithmetic checksum stratum, force same-type distractors, randomize the implied-decimal place. |
| c14 | Instruction-Hierarchy Priority Ledger | authority/conflict resolution | 28 | 2 | Nonce authority tags defeat system>user priors; fix one hidden policy across splits, use anti-convention flags to block delegated induction. |
| c15 | Opaque-Codebook Remapped-Label Classification | classification under remapped labels | 28 | 2 | Flipped-label mechanism gives a real floor; needs two-sided model gating and paraphrase families so one trace batch can't leak the whole table. |
| **c21** | **Relational Micro-World with Invented Vocabulary** | relational/kinship QA over nonce graphs | **28** | 2 | **Top finalist. Invented codebook + seeded relation properties, both lenses survive (minor); needs unique-answer rejection sampling and stratified minibatches.** |
| c01 | Invented-Glyph Positional Base Arithmetic | constructed notation | 27 | 2 | Only c4 is soft (positional eval is a known skill); commit to secret per-stratum conventions and cap demo coverage. |
| c03 | Keyed Custom-Operator Expression Evaluation | invented operators/precedence | 27 | 2 | Opaque per-instance rule keys + forced boundary cases keep the wrap/associativity rungs resolvable; foil-bank rejection sampling recommended. |
| c07 | Case-Marked Micro-Conlang Translation | constructed-language morphology | 27 | 2 | Natural MIPROv2 demo axis; but rules are given in-gloss or predicted by linguistic priors — must MTOB-ify (withhold affix semantics) to restore the ladder. |
| c10 | Interlocking-Cycle Calendar & Clock Arithmetic | constructed cyclic systems | 27 | 1 | Independent coprime moduli isolate which cycle rule broke; solid, less research-current. |
| c13 | Precedence-Ordered Decision-List Classification | policy/triage classification | 27 | 2 | Archetypal independent-rules-with-override task; first-match decision list, feature-combinatorial pool. |
| c16 | Wason Implication-Compliance Classifier | conditional-rule compliance | 27 | 2 | Distinctive model-side residual: vacuous-truth & directionality errors persist even under a perfect instruction. |
| c17 | Invented-Gate Circuit Evaluation | boolean circuits/connectives | 27 | 2 | Randomized truth tables make the demo axis load-bearing — no fixed instruction encodes them. |
| c18 | Depth-Controlled Synthetic Deduction (T/F/Unknown) | synthetic logic/multi-hop | 27 | 2 | RuleTaker/PrOntoQA-style depth bins give an incremental ladder; proof engine names the failed step. |
| c19 | Grid-World State Prediction | deterministic spatial simulation | 27 | 2 | One derived fact (not a path) removes many-valid-answers ambiguity; grid-nav familiarity is the mild risk. |
| c20 | Invented Micro-Game Turn Resolution | invented game rules | 27 | 2 | Strong real-card-game priors guarantee a low floor; injected illegal moves exercise the failure path. |
| c24 | Composite-Key Sorting & Canonical Ordering | ordering under invented conventions | 27 | 2 | Randomized sort keys per batch; each tie-breaker leaves a distinct diff signature. BBH evidences prompt-sensitive headroom. |
| c02 | Fictional Mixed-Radix Unit Conversion | invented measurement systems | 26 | 1 | NUMCoT-documented conversion brittleness; overlaps c01/c10, edged out on research currency. |
| c04 | Term-Rewriting to Normal Form | abstract rewriting systems | 26 | 1 | Cleanest anti-one-shot design (abstract productions have nothing to guess); risk is being too hard, and char-arithmetic temp-0 noise. |
| c08 | Rosetta-to-Match-Up MC Linguistic Puzzles | linguistic-puzzle rule induction (MC) | 26 | 2 | Distractors each differ by one latent rule → built-in incremental ladder; MC framing tames scoring. |
| c09 | Composable Cipher-Chain Decode to Closed Vocab | invented composable ciphers | 26 | 2 | Structural (not shift-arithmetic) ops + closed-vocab / MC variant to dodge temp-0 char noise. |
| c22 | Stacked Verifiable-Constraint Micro-Generation | instruction following/constrained gen | 26 | 2 | COLLIE-style checker library; strict all-pass 0/1; conditional Selection constraints must be discovered. |
| c23 | Hidden-Rule String-Transform Induction Ladder | compositional rule induction from I/O | 26 | 2 | Demos load-bearing by construction; InductionBench warns to curate difficulty so easiest strata are reachable. |

## Which regions were strong or weak against the rubric, and why

**Strong regions.**

- **Invented-notation evaluators & abstract symbol-system execution** (c01, c03, c04,
  c05, c10, c17). These excel on the criteria that matter most: difficulty decomposes
  cleanly into independent latent rules (criterion 10), failures localize to a single
  un-applied rule (criterion 11), and per-instance randomized tables/legends are
  contamination-proof by construction (criterion 8). Their shared weakness is criterion
  4: when the underlying skill (positional arithmetic, FSM simulation, standard protocol
  semantics) is pretraining-common, a competent one-shot prompt can approach the ceiling
  unless the conventions are made **arbitrary and secret** and stratified so no fixed
  prompt covers the pool.
- **Remapped-label & policy classification** (c13, c14, c15, c21). Opaque codebooks and
  nonce authority policies genuinely defeat the pretrained prior, giving a real low floor
  with an inspectable rule-by-rule ladder and a 25%/guess-pinned baseline. **c21**
  (relational nonce-graph QA) is the strongest overall: it stacks an invented closed
  vocabulary on top of seeded nonstandard relation properties, so both the codebook and
  the traversal rules must be learned, and failures cleanly separate codebook misses from
  traversal misses. Both adversarial lenses left it standing with only minor,
  pinning-level fixes.
- **Constraint / violation diagnosis** (c06, c16, c18). MC output makes scoring trivially
  deterministic; injected single-class violations map each failure to a known rule. c16
  adds a rare bonus — genuine model-side residual (vacuous-truth and directionality errors
  survive a perfect instruction), so the ceiling isn't instantly reachable.

**Weak regions.**

- **Ciphers / char-arithmetic transforms** (Theme C, c09). Character-level shift
  arithmetic is token-noise-prone, threatening temp-0 resolvability (criterion 5). Only
  the structural-op + closed-vocabulary / MC variant survives, and even then it sits at
  26.
- **Constructed-language translation** (Theme B, c07). The load-bearing rules are either
  handed to the model in the per-instance gloss or predicted by universal linguistic
  priors (case marking, topic particles), so a single competent paragraph can one-shot
  much of the gap. It scores 27 on structure but the optimization-path lens refuted it as
  designed — it needs an MTOB-style redesign that withholds affix semantics.
- **Free-generation instruction-following** (Theme H, c22). Attractive for research
  currency (IFEval/IFBench lineage) but scoring wants a deterministic checker library and
  strict all-pass 0/1; the conditional/"discover the hidden constraint" framing is what
  keeps it from being one-shot enumerable. Lands at 26.
- **Anything with partial-credit instincts** (much of Theme G's original multi-field
  form). The no-partial-credit rule forced these into single-field queries; the ones that
  survived (c11, c12, c24) did so by relocating incrementality to pool strata, but several
  multi-field extractors dropped out in the merge.

The recurring failure mode across every region is **criterion 4 collapse**: whenever the
latent rules are conventional, fully stated in-instance, or enumerable in one paragraph,
the gap becomes one-shot guessable and every optimizer looks identical. The winning move,
seen in c21/c15/c14/c11, is to make the decisive knowledge **arbitrary seeded information**
that cannot be recalled or guessed — only discovered from demos and failure traces —
while keeping a designer-authored ceiling prompt that provably reaches ~100%.

## Notable candidates just below the finalist cut

The seven 28-scorers (c05, c06, c11, c12, c14, c15, c21) are the verified finalists. Just
below them, at 27, sit several genuinely strong tasks that lost only on a single soft
criterion or on research currency:

- **c17 Invented-Gate Circuit Evaluation** — arguably the purest embodiment of "demos are
  load-bearing" (randomized truth tables cannot be encoded by any fixed instruction), but
  its single-token boolean output gives slightly coarser strata than the 28-tier tasks.
- **c13 Precedence-Ordered Decision-List Classification** — the archetypal
  independent-rules-with-override task; edged out mainly because its rule families are less
  contamination-novel than nonce-graph or opaque-codebook designs.
- **c16 Wason Implication-Compliance Classifier** — uniquely retains model-side residual
  headroom, but the binary/MC output and vacuous-truth subtlety make its floor harder to
  pin precisely.
- **c18 Depth-Controlled Synthetic Deduction** — the cleanest incremental ladder via proof
  depth bins, held back only by the heavier reasoning load per instance.
- **c04 Term-Rewriting to Normal Form** (26) — the strongest anti-one-shot design in the
  whole pool (abstract productions have literally nothing to guess), but flagged for being
  potentially too hard and for char-arithmetic temp-0 noise on longer normal forms.
- **c08 Rosetta MC linguistic puzzles** (26) — a built-in incremental ladder (each
  distractor differs by exactly one latent rule) with clean MC scoring; a strong
  research-current dark horse that the exact-match tier simply outscored.

## Bottom line

The space is richest and most rubric-aligned in the **invented-convention** family —
tasks where a seeded generator hides arbitrary rules and an invented codebook behind a
single bounded question. The seven finalists all live there. Among them, **c21 (Relational
Micro-World with Invented Vocabulary)** is the standout: it is the only candidate whose
two independent sources of headroom (an unguessable closed vocabulary *and* seeded
nonstandard relation properties) both survive adversarial review with only minor pinning
fixes, while c05/c06/c11/c12/c14/c15 each carry a repairable (non-fatal) design change
before pinning. See the per-domain research docs linked above for the mechanism detail,
floor/ceiling evidence, and generator sketches behind each cluster.
