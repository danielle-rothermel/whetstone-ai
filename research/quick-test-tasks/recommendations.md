# Quick-Test Task Recommendations

whetstone-ai · 2026-07-21

Recommendations for the single cheap task family that validates the optimizer implementations (COPRO, MIPROv2, GEPA, Codex CLI agent) end to end before the HumanEval+ compression experiment. Authoritative criteria: [quick-test-rubric.html](../../design/quick-test-rubric.html). Inputs: the merged candidate list ([candidates-merged.md](candidates-merged.md)), rubric scorecards, and two adversarial skeptic passes per finalist (optimization-path lens and measurement lens).

**Bottom line: adopt c21 (Relational Micro-World with Invented Vocabulary) as the quick test.** It is the only finalist that survived both adversarial lenses with only minor-severity caveats, and every caveat is a pinning/verification obligation inside artifacts the adaptation plan already builds. Backup: c03 (Keyed Custom-Operator Expression Evaluation), the only other candidate both skeptics left unrefuted. Three more candidates (c14, c11, c12) are viable with real but bounded redesign work.

## A note on ordering (where this doc departs from the mechanical ranking)

The mechanical sort is by rubric total (28 or 27 of 28), which puts c05 (FSM/state tracking) in the top tier. **This doc demotes c05 out of the recommended set entirely, and promotes c03 (total 27) to #2, because the skeptic verdicts justify overriding raw scores:**

- **c05** was refuted by *both* skeptic lenses at major severity. Its named latent rules (AUTH-before-DATA, undefined-transition fallback, monotone sequence numbers) are textbook protocol/interpreter conventions a competent prompt engineer states one-shot, and its difficulty knob (trace length) is caught in a dilemma: long traces make the ceiling unreachable in a single no-scratchpad call (its own citations — [arXiv:2503.02854](https://arxiv.org/abs/2503.02854), [arXiv:2509.09677](https://arxiv.org/abs/2509.09677) — document this), while short traces plus allowed CoT leave a 2-3 rung ladder that one demo bootstrap collapses. The fixes exist but amount to redesigning it into something much closer to c21/c03 (arbitrary counter-intuitive conventions). The 28/28 scorecard did not price this in.
- **c15** (opaque-codebook classification) and **c01** (invented-glyph base arithmetic) were also refuted by both lenses (major): c15's rule *family* is famous flipped-label literature, so a "here are the rule types, induce the bindings from demos" meta-prompt one-shots it; c01 is trapped between explicit specs (one-shot) and implicit quirks (no writable ~100% reference prompt), and its encode direction has non-unique ground truth. Both are excluded.
- **c03** scored 27 (a 1 on criterion 4) but is the only candidate besides c21 where *neither* skeptic refuted the design — both lenses concluded the incremental ladder survives, contingent on fixes that are concrete and cheap (opaque convention tokens, foil-bank rejection sampling). A candidate whose weakest criterion survived adversarial attack outranks 28-scorers whose measurement story did not.

Remaining order within the top five weighs: verdict health first, then tie-breaker strength, then adaptation cost.

---

## 1. c21 — Relational Micro-World with Invented Vocabulary  (ADOPT)

**Domain:** relational/kinship QA over nonce graphs. **Rubric total: 28/28. Verdicts: optimization-path not refuted (minor), measurement not refuted (minor)** — the only finalist with no major-severity finding.

### What the task is

A pinned generator builds a small graph of nonce entities connected by nonce relation symbols. Each relation symbol carries *seeded, nonstandard* properties fixed per generator seed: which relations are symmetric, which are transitive, an inverse mapping, a two-relation composition table, explicit-negation precedence, and closed-world handling. Answers use an invented closed vocabulary (a gender-neutral "nieph", a "bond-" prefix for in-law-style links, a relation-distance cutoff). An instance is a few facts plus one query; output is one/two words from the closed vocabulary or a 4-way MC letter (true / false / both / unknown), scored 0/1 exact match against a graph-traversal oracle.

**Concrete example instance** (one seed's conventions: `tovv` is transitive and non-symmetric; two `tovv` hops map to the codeword `nieph`; `brell` is symmetric; closed-world default for `tovv` is *unknown*):

> Facts:
> - Marn tovv Belu.
> - Belu tovv Sagre.
> - NOT (Sagre brell Marn).
>
> Q1: What is Sagre to Marn? Answer with one word from the task vocabulary.
> **Gold: `nieph`** (two tovv-hops). A naive prompt answers "grandchild" → exact-match 0.
>
> Q2: Does Belu tovv Marn? Answer A (true) / B (false) / C (both) / D (unknown).
> **Gold: `D`** (tovv is not symmetric under this seed; closed-world default for tovv is unknown). A naive prompt guessing symmetry answers A → 0.

### Rubric scorecard highlights and weak spots

Straight 2s on all fourteen criteria. Standouts:

- **Criterion 4 (low floor, incremental gap) is structural, not tuned.** The hidden rulebook is *arbitrary seeded information* — no zero-shot prompt, however competent, can contain the per-seed property assignments or the invented codebook. With only a few facts per instance, the rules are under-determined from any single instance, so "infer the rules yourself" prompts cannot shortcut the ladder.
- **Criterion 10 (independent latent rules):** ~9 strata (six relation-property rules plus three vocabulary rules), each worth roughly 10-15 points of pool score.
- **Criterion 11 (diagnostic failures):** codebook misses (model emitted a standard English kin term) are cleanly distinguishable from traversal misses, and within traversal the violated rule is identifiable from expected-vs-got.
- **Weak spots (from the verdicts, none refuting):** (a) a best-effort "uninterpreted-symbols, closed-world literal logician" prompt could land at 40-60% if hidden-rule strata are under-weighted; (b) some rules (composition table, distance cutoff) may be practically non-inducible from 0/1 feedback, stalling *all* optimizers at the same sub-ceiling; (c) multiple-valid-answer instances and an incoherent "both" label are latent generator bugs; (d) uniform random 10-task minibatches have a ~21% chance of missing a given stratum entirely.

### Surviving skeptic caveats → how the plan addresses them

| Caveat | Mitigation (now part of the adaptation plan) |
|---|---|
| Literal-logician baseline could sit mid-high | Pin that engineered prompt as an explicit measured baseline; weight ≥60-70% of instances to strata requiring a nonstandard seeded property or codebook term so it scores ≤~50% |
| Non-inducible rules could falsify the success criterion for reasons unrelated to optimizer quality | Per-rule discoverability check before pinning: feed a reflection LM k failure traces from one stratum; shrink/drop rules it cannot state (keep composition table tiny, distance cutoff small) |
| In-context rule induction via large demo sets | Cap few-shot demo counts; adversarially test an "induce the hidden conventions from the examples" prompt on the pinned model |
| Non-unique answers / incoherent "both" | Rejection-sample instances whose rule-closure admits >1 valid closed-vocab answer (run the oracle over the whole vocabulary, reject ties); define "both" as arising only for relations whose seeded profile lacks the negation-precedence property |
| Stratum-starved minibatches | Stratified minibatch composition: ≥1-2 tasks per latent-rule stratum in every 10-20 task internal eval |
| Silent ceiling shortfall | Empirically verify the reference prompt at ~100% and the naive floor on the pinned model at temp 0 *before* freezing the manifest, tuning hop depth down until the ceiling holds |

### Concrete adaptation plan

1. Build the two pinned artifacts: a graph sampler (nonce entities + per-seed relation property profiles) and a traversal oracle that computes exact answers under those assignments. Pin generator version, dataset manifest, disjoint minibatch/subset/official split identities.
2. Cap hop depth so the full-rule reference prompt reaches ~100%; balance the four MC outcomes and closed-vocab answer distribution so constant-guess baselines are pinned low (25% MC floor).
3. Tag every instance with its governing latent-rule stratum for failure attribution and stratified aggregation (average 0/1 within, then across strata, per the rubric's cross-strata callout).
4. Store the reference ~100% prompt, the naive baseline, and the literal-logician baseline; pre-register a falsifiable criterion (e.g., recover ≥90% of the gap within 8 iterations) *after* the discoverability check confirms every retained rule is inducible.
5. Normalize answers (lowercase, strip whitespace, canonicalize hyphenated `bond-` terms); unparseable/off-vocabulary/empty outputs are decisive 0s (exercises criterion 13).
6. For GEPA/two-objective coverage, add mean output UTF-8 bytes as the distinct second Objective with a fixed length-budget axis, per the rubric's mandatory callout.

### Tie-breaker case

Strong (2/2). Sits squarely in the active in-context rule-learning / synthetic-formal-language ICL and rule-induction lines (cipher-ICL, GINC/RegBench, MIR-Bench, FalsifyBench, LINGOLY/Linguini linguistics-olympiad family) and is directly adjacent to PrOntoQA's fictional-symbol methodology (ICLR 2023 — no arXiv ID in the ranked data). The invented-vocabulary + seeded-nonstandard-property design is exactly the contamination-proof, decomposable, few-shot-sensitive probe those 2025-2026 lines care about; it could grow into a standalone optimizer-comparison benchmark. Only gap: no flagship live leaderboard to anchor to.

---

## 2. c03 — Keyed Custom-Operator Expression Evaluation  (BACKUP)

**Domain:** invented operators / nonstandard precedence arithmetic. **Rubric total: 27/28 (criterion 4 = 1). Verdicts: optimization-path not refuted (major), measurement not refuted (major)** — the only other candidate both skeptics left standing.

### What the task is

Each instance opens with a per-instance symbol table defining invented infix operators as closed forms (e.g., `(a·k + b) mod m`), decoy symbols, an opaque mode token, and a short chained expression. Output is one integer (or a 4-way MC letter over candidate results), scored 0/1 exact match against a ~10-line reference evaluator. Latent, family-constant conventions: modular wrap into residues 1..m (not 0..m-1), the semantics of the opaque mode token (precedence/associativity), threshold post-processing, and a recurrence stratum with an index term.

**Concrete example instance** (family-latent conventions: `⊞` binds tighter than `◇`; results wrap into 1..m):

> Definitions: `x ◇ y = (3x + 2y) mod 6` · `x ⊞ y = |x − y| + 1` · `x ⨂ y = x` (decoy, never used) · mode: `K3`
> Evaluate: `2 ◇ 4 ⊞ 3`
>
> **Gold: `4`** — house precedence: `4 ⊞ 3 = 2`, then `2 ◇ 2 = 10 mod 6 = 4`.
> Naive left-to-right/PEMDAS reading: `2 ◇ 4 = 14 mod 6 = 2`, then `2 ⊞ 3 = 2` → answers `2` → 0.

### Rubric scorecard highlights and weak spots

- **Highlights:** textbook single-call + exact-integer-match graph (criteria 1-3, 14); per-instance random parameters make it the most straightforwardly contamination-proof candidate (criterion 8); the reference evaluator gives an exact, checkable ceiling (criterion 9); diagnostic errors — off-by-one at the residue boundary flags the wrap rule, order errors flag associativity, magnitude errors flag precedence (criterion 11).
- **Weak spot: criterion 4 scored 1.** A generic "ignore PEMDAS, read the table literally, obey the stated flags, output only the integer" prompt captures everything the instance states legibly. The candidate's headroom lives *entirely* in conventions the instance does not state or actively mis-states. Both skeptics confirmed this is fixable but load-bearing.
- **Quantified measurement hole (found by simulation in the verdict):** the wrap rule as originally framed is nearly invisible — naive 0..m-1 and house 1..m answers *coincide* on 85-97% of unforced instances (they differ only when an intermediate hits ≡0 mod m), so learning it would be worth ~2 points on a 15-task eval, far below the rubric's ≥10-point resolvability bar.

### Surviving skeptic caveats → how the plan addresses them

- **Legible flags = one-shot rungs.** Fix: all per-instance rule toggles (precedence/associativity, threshold, recurrence) must be *opaque tokens* (`mode: K3`) whose semantics are family-global latents never stated in any instance; surface `mod m` notation stays deliberately misleading vs. the 1..m ground truth.
- **Invisible wrap-rule effect size.** Fix: foil-bank rejection sampling — evaluate every candidate instance under wrong-wrap, wrong-associativity, PEMDAS, no-threshold, and no-index foils; accept only instances where each active stratum's foil answer differs from truth (~m extra evaluator calls per instance, negligible). MC distractors are drawn from the deduplicated foil outputs.
- **Demo leak.** Fix: scope few-shot demos by stratum so no single demo set reveals more than one hidden convention.
- **COPRO scoping (structural caveat, applies to every strong candidate):** a score-only instruction rephraser cannot produce per-run hidden conventions, so COPRO validates plumbing and generic-instruction gains, not deep hyperparameter comparison — but the anti-PEMDAS/table-following rung is deliberately kept reachable by blind instruction search so a broken COPRO is still distinguishable from a stumped one.

### Concrete adaptation plan

1. Seeded generator + tiny reference evaluator jointly define ground truth; pin seed ranges for disjoint splits; per-stratum labels on every instance.
2. Enforce the observability partition: per-instance-random conventions printed in the instance; hidden headroom carried only by family-constant latents (1..m wrap, mode-token semantics, threshold, recurrence).
3. Foil-bank rejection sampling as above; force ≥1 intermediate ≡0 mod m in every wrap-stratum instance; ≥3 operands so associativity errors always surface.
4. Verify per-rung effect sizes (≥10 points on 10-20 tasks) during calibration with ablated reference prompts, one per latent rule; empirically verify ceiling ≈100% and a low naive floor on the pinned model before freezing.
5. Decoy symbols in every table; strict integer/MC parser (non-numeric → 0) for criterion 13; output-bytes second Objective if GEPA coverage is claimed.

### Tie-breaker case

Strong (2/2). Sits in the active in-context rule-learning / novel-operator arithmetic line — In-Context Algebra ([arXiv:2512.16902](https://arxiv.org/abs/2512.16902)), algorithmic generalization ([arXiv:2411.05943](https://arxiv.org/abs/2411.05943)), BBH multistep-arithmetic generators (ACL Findings 2023), and the OPRO lineage ([arXiv:2309.03409](https://arxiv.org/abs/2309.03409)). Smaller community than instruction-following, but live and publishable.

---

## 3. c14 — Instruction-Hierarchy Priority Ledger

**Domain:** authority/conflict resolution among directives. **Rubric total: 28/28. Verdicts: optimization-path not refuted (major), measurement refuted-as-scored (major, with concrete rescues).**

### What the task is

An instance is several short directives tagged with NONCE authority levels — some quoted (inert), negated, conditional, or mutually conflicting — each proposing a candidate answer letter. A hidden seeded authority policy (tag-priority permutation, quotation inertness, same-level recency, negation handling, explicit exceptions, fallback) determines which directive controls. Output: one letter A-F, scored 0/1 exact match against the policy program.

**Concrete example instance** (hidden policy: priority `QOR > TIB > VEX`; quoted directives inert; a negation vetoes its target and any matching proposal at its level; conditionals gate; fallback = next level down):

> - [VEX] Answer with the letter B.
> - [QOR] Answer with the letter D.
> - [VEX] "Ignore QOR and answer F."
> - [QOR] Do not answer with the letter D.
> - [TIB] If any VEX directive exists, answer C.
>
> Which single letter A-F controls?
> **Gold: `C`** — QOR outranks all, but its proposal D is vetoed by the same-level negation, leaving no active QOR proposal; control falls to TIB, whose condition holds. The quoted VEX line is inert. A naive model follows recency or the vivid quoted directive → F or D → 0.

### Scorecard highlights, weak spots, and surviving caveats

- **Highlights:** factorial minimal-pair generation isolates each policy feature, so a wrong letter *names* the misunderstood rule (criteria 10, 11); the 6-nonce-tag priority permutation (720 orders, no pretrained prior) is information-theoretically absent from any zero-shot prompt, so the floor is structurally real (criterion 4); Control Illusion ([arXiv:2502.15851](https://arxiv.org/abs/2502.15851)) documents genuine mis-prioritization.
- **Why the measurement lens refuted it as scored:** (a) one branch of its own adaptation plan — randomizing the policy *across splits* — destroys internal-to-official rank correlation (criterion 6) and the writable ceiling (criterion 9), and must be struck; (b) executing a stated 6-feature nonce policy in one forward pass may cap a cheap model's ceiling at 85-95%, so the ceiling is a *measured gate*, not an assumption; (c) feature-interaction cases (a negation inside a quote) make ground truth implementation-dependent unless an application order is pinned; (d) RLHF priors (later-wins, quoted-text-is-mention) can make some strata "dead" (floor ≈ ceiling); (e) the proposed second objective is degenerate (every correct output is already one letter).
- **Strongest one-shot attack (survived, conditionally):** "delegated induction" — a prompt stating the five generic features plus "induce the tag order from these demos" can pin most of the permutation from ~10-15 labeled examples. Blocked only by capping labeled examples per proposal/reflection step and setting per-feature flags *counter* to natural conventions.

### Concrete adaptation plan (incorporating both skeptics' fixes)

1. One fixed hidden policy per run, identical across all splits; vary policy only across runs. Designer's policy-stating reference prompt must pass a pre-registered gate: ≥95% at temp 0 on 100+ held-out instances on the pinned cheap model, else shrink directive count/tag alphabet; record the *measured* ceiling as the criterion-9 anchor.
2. Seed anti-prior flag settings (quotation not always inert, primacy sometimes beating recency); measure per-stratum floors at calibration and drop/reweight dead strata.
3. Pin the policy program's feature-application order (inertness → negation → exceptions → priority → recency → fallback) in the versioned schema; exclude cross-feature interactions from single-feature strata or give them an explicit interaction stratum.
4. Cap demos/reflection minibatch sizes so pairwise-order evidence per step underdetermines the permutation; make the calibration adversary the *delegated-induction* prompt, not a naive one.
5. Deterministic normalization (strip, casefold, must equal one char A-F, else 0). Fix the degenerate second objective by scoring correctness on the final line while measuring bytes over the full output, so reasoning-before-answer trades accuracy against bytes.

### Tie-breaker case

The strongest tie-breaker in the pool. Instruction hierarchy is top-tier, safety-relevant, actively benchmarked 2025-2026 territory: IHEval ([arXiv:2502.08745](https://arxiv.org/abs/2502.08745)), The Instruction Hierarchy ([arXiv:2404.13208](https://arxiv.org/abs/2404.13208)), Control Illusion ([arXiv:2502.15851](https://arxiv.org/abs/2502.15851)). A synthetic nonce-tag priority ledger with factorial minimal pairs would be a novel, contamination-proof contribution with genuine standalone-benchmark potential. If the tie-breaker were weighted equal to the rubric, this candidate would rank #2.

---

## 4. c11 — House-Convention Canonicalizer

**Domain:** serialization/normalization under invented house rules. **Rubric total: 28/28. Verdicts: optimization-path not refuted (major), measurement refuted-as-specified (major, rescued by a rule-set change).**

### What the task is

One deterministic normalization pipeline (canonical JSON or slug preferred) governed by 5-6 invented house rules that deliberately deviate from every public standard. Instance = a messy value; output = the canonical string; scored 0/1 whole-string exact match against a ~40-line reference function. Each rule owns a stratum, so pool accuracy climbs rule by rule and expected/got diffs name the violated rule.

**Concrete example instance** (slug family; house rules: lowercase; token separator is `~` not `-`; drop words in an invented stopword list {the, of, an}; ordinals map through an invented table (`2nd` → `twyx`); keep at most 4 tokens, truncating from the end):

> Input title: `The Quick   Brown Fox (2nd Edition)!`
> **Gold: `quick~brown~fox~twyx`**
> Naive slugify emits `the-quick-brown-fox-2nd-edition` → 0. Each wrong rule (kept "the", used `-`, missed `twyx`, no truncation) is visible in the diff.

### Scorecard highlights, weak spots, and surviving caveats

- **Highlights:** infinite contamination-proof pool; the reference function *is* the ceiling definition; a single wrong rule fails the whole example, honoring the no-partial-credit rule; directly mirrors the project's dr-serialize canonical-JSON theme, so intuitions transfer to the main experiment.
- **Why the measurement lens refuted it as specified:** the merged rule set included computation-heavy rules — a custom check digit, length-then-codepoint sorts over long key lists, bespoke float formatting — that a small model cannot reliably execute in one constrained forward pass even with the rules stated verbatim. That makes the true ceiling 70-90% and unknowable in advance (criterion 9 fails), with the residual slips landing exactly where temp-0 provider nondeterminism flips tokens. Also: the output-bytes second objective is degenerate (correct canonical strings have fixed length), and raw whole-completion matching conflates format compliance with rule knowledge.
- **Optimization-path caveats (survived):** the plan's own step 7 ("demo sets covering *all* invented month abbreviations, *all* check-digit cases") manufactures its strongest one-shot attack — full-coverage demos turn lookup rules into copy-from-context. Table-width vs. minibatch-width is never pinned, so one GEPA reflection could transcribe every rule visible in a batch.

### Concrete adaptation plan (incorporating both skeptics' fixes)

1. Restrict rules to single-pass-executable transformations: table lookups (invented month/stopword/ordinal tables), character-class rewrites, fixed replacements, whitespace collapsing, truncation policy, sorts of ≤3 short keys. Drop the arithmetic check digit and long-list sorts (or reduce the check digit to a fully enumerable ≤3-digit lookup). Keep rule count ≤6.
2. Enforce table-width > coverage: table-valued rules get more entries (12-20) than any demo set or GEPA minibatch (cap minibatch ~5; cap demo sets below full table coverage) so rules must accrete over iterations. Strike "cover all cases" from demo curation.
3. Generator is adversarial per instance: every instance actively distinguishes house rule from public standard (reject inputs where they coincide); global formatting rules form an explicit first-rung stratum; ≥2 tasks per stratum in every internal subset.
4. Pre-freeze certification: reference prompt ≥95% exact match with 3 agreeing temp-0 repeats on the pinned official split; derive floor/gap targets from measured numbers. Freeze exactly one trivial normalization (strip surrounding whitespace/code fences) in the Eval Config.
5. Run the "infer every rule from the examples" one-shot with the largest permitted demo set; treat >50% as a rule-mix bug. Document that the output-bytes objective is degenerate here — this candidate validates single-objective optimization plus two-objective *plumbing* only, per the rubric's mandatory callout (a real limitation vs. c21/c14).

### Tie-breaker case

Strong (2/2): rides the verifiable/constraint-based instruction-following surge and text normalization specifically — PolyNorm ([arXiv:2511.03080](https://arxiv.org/abs/2511.03080)), StructEval ([arXiv:2505.20139](https://arxiv.org/abs/2505.20139)), JSONSchemaBench ([arXiv:2501.10868](https://arxiv.org/abs/2501.10868)), with RFC 8785 JCS as a task donor. Derivative of an active wave rather than opening one, but the dr-serialize alignment is a project-specific bonus no other candidate has.

---

## 5. c12 — Sigil-Schema Field Extraction and Dialect Routing

**Domain:** structured extraction under invented markups. **Rubric total: 28/28. Verdicts: optimization-path not refuted (major), measurement refuted-as-specified (major, rescued by stratum redesign).**

### What the task is

Synthetic records in invented markups — sigil-tagged fields, fixed-width columns, custom-escaped delimiters, or dialects signaled by a header token — where each instance asks for ONE field returned in normalized house form. Output is a single short token, 0/1 exact match against the generator's own record. Latent rules: sigil→field map, per-dialect conventions, padding/order conventions, escape handling, per-field normalization (randomized implied-decimal position), enum precedence.

**Concrete example instance** (dialect `VEL`: `ᚠ` marks unit price with implied decimal at 3 places this run; `ᚷ` marks a date; a second money-shaped field `ᚢ` is a distractor):

> Record: `⟦hdr:VEL⟧ ᚠ7203 ᚢ1150 ᚷ0925`
> Query: unit price, in house form.
> **Gold: `7.203`** — sigil map picks `ᚠ` over the same-type distractor `ᚢ`; this run's implied-decimal position is 3.
> Naive extraction answers `7203` or `72.03` → 0.

### Scorecard highlights, weak spots, and surviving caveats

- **Highlights:** per-run randomized sigils/widths/delimiters/vocabularies are contamination-proof and genuinely resist one-shot prompts; strictly one field per instance keeps scoring binary; the spec-dump reference prompt defines the ceiling.
- **Why the measurement lens refuted it as specified:** the checksum VALID/INVALID stratum breaks three criteria at once — 50% guess floor, zero-bit binary failures a reflection LM cannot learn from, and an arithmetic-execution ceiling a cheap model cannot reach; exact character offsets in fixed-width strata fight BPE tokenization; cents→decimal at a fixed position is world knowledge, not latent headroom.
- **Optimization-path caveats (survived):** a semantic type-matching prompt ("find the field matching the queried type") solves the sigil stratum for free unless every record carries ≥2 same-type distractors; COPRO's score-only proposer cannot reach per-run randomized rules (scope it to plumbing validation, as with c03/c21).

### Concrete adaptation plan (incorporating both skeptics' fixes)

1. Drop or replace the checksum stratum with a non-arithmetic decision rule (e.g., output the field whose stated check char mismatches a stated reference — comparison/lookup, not modular arithmetic).
2. Redesign fixed-width strata around field order plus an invented padding glyph (short ≤4-char widths, distinctive pad runs) rather than character counting.
3. Mandate ≥2 same-type distractor fields per record; randomize the implied-decimal position per run (2/3/4 places) so normalization is an inducible rule, not a Stripe convention.
4. Pre-flight gate: spec-dump reference prompt ≥98% per stratum on the pinned model/seed before a stratum enters the pool; record per-stratum reference/floor scores in the manifest so gap-recovery is falsifiable per stratum.
5. Stratified splits with even dialect/normalization mixes; canonical single-token output surface with a published exact-match normalizer; pinned generator + seed metadata.

### Tie-breaker case

Solid but not flagship (2/2 with caveats): EDC ([arXiv:2404.03868](https://arxiv.org/abs/2404.03868)) canonicalization, StructText ([arXiv:2507.21340](https://arxiv.org/abs/2507.21340)) as a reverse generator, plus an LLMStructBench-style "prompting beats model size" finding whose provenance is unverified — do not cite it in anything public without checking.

---

## Near-misses and exclusions

- **c06 (Artificial-Grammar Violation Classification, 28/28):** same verdict profile as c11/c12 (opt-path survives with major caveats, measurement refuted-then-rescued). Excluded from the top five only because its rescue moves headroom from execution difficulty to hidden rule tables — at which point it converges on what c21 and c03 already do more cleanly — and its convention-exploiting-prompt floor (~50-60% if instantiations are mnemonic) needs a red-team gate the others don't. A fine sixth choice; sources: MLRegTest (JMLR 2024), FLaRe ([arXiv:2411.07107](https://arxiv.org/abs/2411.07107)).
- **c05, c15, c01:** excluded — both skeptic lenses refuted each (see the ordering note at top). c07 (Case-Marked Micro-Conlang) was refuted on the optimization path outright: its gloss table hands the latent morphology to the model in-context, and its "hard rule" (case beats word order) is the highest-prior rule in linguistics; the rescue is a structural redesign (MTOB-style withheld glosses) that converts it into a different candidate.

---

## How to decide among these

All five survivors share one scoping fact worth stating once: **any task that is genuinely not one-shot guessable (criterion 4) and contamination-proof (criterion 8) puts the decisive knowledge outside COPRO's score-only instruction rephrasing.** On every candidate above, COPRO validates contract plumbing and generic-instruction gains; deep hyperparameter ranking comes from MIPROv2/GEPA/agent. Plan for that regardless of choice.

The actual trade-offs:

- **Lowest risk, most rungs → c21.** Only candidate with no major-severity finding; ~9 independent strata gives the finest-grained optimizer/hyperparameter resolution; all remaining work is verification gates, not redesign. Cost: the graph oracle + rejection sampler is the most intricate generator in the set, and the discoverability check adds a pre-pinning step. **Choose this unless generator effort is the binding constraint.**
- **Simplest build → c03.** A ~10-line evaluator and an expression sampler; the foil-bank fix is mechanical. Cost: fewer rungs (~4-5 family-constant latents), so hyperparameter resolution is coarser, and criterion 4 depends entirely on keeping mode-token semantics opaque — one legibility slip collapses a rung. **Choose this if you want the quick test running this week.**
- **Best research upside → c14.** Top-tier, safety-relevant instruction-hierarchy line; would make the strongest standalone publication and the only candidate with a *non-degenerate* two-objective story after its fix. Cost: heaviest calibration burden (measured ceiling gate, dead-strata audit, anti-prior seeding, demo-leak throttling), and its ladder is the most vulnerable to a strong reflection LM naming several features in one step.
- **Best transfer to the main experiment → c11.** Canonical-JSON house rules directly rehearse dr-serialize intuitions. Cost: its second objective is provably degenerate (it must *declare* single-objective-plus-plumbing coverage per the rubric callout), and it needs the rule-set diet plus table-width policing before its ceiling is trustworthy.
- **Best failure-taxonomy workout → c12.** Dialect routing + format traps naturally exercise parse-failure paths and the retry machinery. Cost: needed the most stratum surgery (checksum removal, fixed-width redesign), and its per-instance single-token failures are the least individually diagnostic for GEPA — diagnosis lives at the batch level.

**Recommended path:** build c21; while its generator is under construction, stand up c03 in a day or two as the plumbing smoke test (its evaluator is trivial), then retire c03 or keep it as a second pinned family for cross-checking optimizer rankings. If the team later wants the quick test to grow into its own research direction, c14 is the one to invest the extra calibration effort in.
