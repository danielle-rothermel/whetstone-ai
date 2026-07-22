# Quick-Test Candidate List — Merged Brainstorm + Literature

Date: 2026-07-21
Inputs: 76 brainstormed ideas + 108 literature candidates.
Rubric: `design/quick-test-rubric.html` (authoritative). Two rubric constraints shaped every merge:
(a) scoring is strict per-example 0/1 exact match — **no partial credit inside an example**, so all
"per-field partial scoring" brainstorm variants were rewritten as single-field / single-token queries
with incrementality coming from **rule strata across the task pool**; (b) outputs bounded to MC or a
few words, single LLM call, no sandbox.

Result: **24 merged candidates**, 9 merged clusters dropped.

---

## Kept candidates

### c01 · Invented-Glyph Positional Base Arithmetic — origin: both
**Domain:** constructed notation / nonstandard numeral systems
**Merged from (brainstorm):** Glyphscript Arithmetic Decode; Nonstandard Positional Base Evaluation; Balanced mixed-radix / invented positional number system; Roman-numeral / base-N transcoder (base-mode); EncodedArith.
**Merged from (literature):** Base Arithmetic counterfactual (Wu et al. 2023); non-decimal sub-tasks (Yuan et al. 2023); balanced-ternary family; length-generalization arithmetic suites; Teaching Algorithmic Reasoning via ICL (2022); NumericBench (generator donor); cryptarithmetic (hard-tail stratum only).
**Summary:** Instance = one or two numerals in an invented glyph alphabet plus an inline legend (glyph→digit map) and a base spec that varies per stratum (base 9/11, mixed-radix, balanced/signed digits, MSB- vs LSB-first, null-glyph position skip). Output = a single decimal integer or fixed-width glyph string; exact match vs a ~10-line positional evaluator; seeded per-instance random legends. Latent rules: legend parse, place weights, signed-digit handling, endianness, null glyph, carry variant. Literature documents the low default floor and the named "decimal round-trip" failure mode (diagnostic for reflection).
**Adaptation:** Build the seeded generator; avoid bases 8/16 (pretraining-frequent); randomize which quirks are active per stratum so no fixed one-shot prompt covers the pool; constrain output to the bare numeral.

### c02 · Fictional Mixed-Radix Unit Conversion — origin: both
**Domain:** invented measurement systems / mixed-radix arithmetic
**Merged from (brainstorm):** Notation Unit-System Conversion; Symbolic unit / dimension algebra; Unit-Convention Numeric QA.
**Merged from (literature):** NUMCoT (2024); GSM-Symbolic / GSM-Plus (template-generation methodology only).
**Summary:** Instance = a quantity in invented units with an inline randomized conversion legend ("1 mor = 7 kel, 1 kel = 12 tiv") plus a conversion request; output = a single integer in the smallest unit or a canonical mixed-radix string ("4 mor 0 kel 2 tiv"), exact match vs trivial mixed-radix reduction. Latent rules: legend parse, per-boundary factors, carry/normalization (no unit exceeds its base), truncation vs rounding keyword, one signed-digit unit stratum, unit-simplification rewrites. Per-instance randomized ratios/names are contamination-proof and force the prompt to encode the general procedure, not numbers. NUMCoT documents genuine conversion-ratio brittleness (low floor).
**Adaptation:** Strip NL narrative to a one-line question; invent all units (no real-world ratios); stratify quirks (rounding, signed unit) so score climbs rule by rule.

### c03 · Keyed Custom-Operator Expression Evaluation — origin: both
**Domain:** invented operators / nonstandard precedence arithmetic
**Merged from (brainstorm):** Custom binary operator (# operator); Keyed-DSL Arithmetic; Custom-precedence arithmetic mini-grammar; Custom Arithmetic Precedence (CAP); Modular polynomial / linear-recurrence next-term.
**Merged from (literature):** Novel-operator / keyed-DSL in-context arithmetic family; In-Context Algebra (2025); BBH multistep-arithmetic generators (OPRO/EvoPrompt substrate).
**Summary:** Instance opens with a per-instance symbol table defining invented infix operators (closed forms like (a·k+b) mod m), a precedence/associativity flag, decoy symbols, and a short chained expression; output = one integer, exact match vs a tiny reference evaluator. Latent rules: table lookup, precedence tier per operator, associativity direction, modular wrap into residues 1..m, threshold post-processing, recurrence stratum with index term. Naive prompts assume PEMDAS and standard semantics; the ceiling prompt states the general read-table-then-evaluate procedure. Per-instance random parameters give contamination-proofing and a known ~100% ceiling.
**Adaptation:** Flip precedence per instance; keep expressions deep enough (3+ operands) that associativity errors surface; add MC-over-four-results variant for the cheapest format.

### c04 · Term-Rewriting to Normal Form — origin: both
**Domain:** abstract rewriting systems / formal grammars
**Merged from (brainstorm):** Symbolic Rewrite-System Normal Form; Expression rewriting under invented rewrite rules; Grammar-of-rewrites (RS3).
**Merged from (literature):** Abstract Term-Rewriting to Normal Form (Terese; confluence-strategy work 2024); L-system derivation (parallel-rewriting stratum, bounded feature).
**Summary:** Instance = 3–6 productions over invented symbols + a start string of 8–14 symbols, with the system confluent+terminating (or a fixed leftmost-innermost strategy) so the normal form is unique; output = the short normal-form string, or a bounded derived feature (final length / symbol at position k) for the L-system stratum, exact match vs a reference rewriter run to fixpoint. Latent rules: each production, redex-selection strategy, iterate-to-fixpoint (not one pass), empty-production deletion, guard conditions, parallel-vs-sequential application. Failures localize to a single un-applied production or a halted iteration — ideal reflection signal. Seeded random rule sets are contamination-proof.
**Adaptation:** Build generator + oracle; enforce determinism via confluence checks or fixed strategy; cap L-system iterations so outputs stay bounded.

### c05 · Deterministic Execution & State Tracking (FSM / protocol / pointer) — origin: both
**Domain:** state-machine simulation and protocol execution
**Merged from (brainstorm):** Glyph Automaton; Protocol Handshake Validator; Opaque Device State; Priority-Rule Scheduler Resolver; Turing-Tape Rewrite (bounded tag-system); Reference-Chained Value Deref.
**Merged from (literature):** TMBench (2025); Scheduler/protocol-validator FSM family (state-tracking mechanistic work 2025); Long-Horizon Execution retrieve-and-sum (2026).
**Summary:** A seeded generator emits a small deterministic machine — FSM transition table + event trace, a message protocol with injected single-rule violations, a device-command or tag-system sequence, or a key→value store with pointer chains — and asks ONE bounded question: final state label, index of first violating message, final device state letter, or fully dereferenced value/final sum. Output = single token, exact match vs the reference machine. Latent rules: per-transition entries, undefined-transition fallback, ordering constraints (AUTH-before-DATA), monotone sequence fields, cycle→sentinel, undefined→default, max-depth cap, append-vs-delete ordering. Injected violations map each failure to a known rule, making this the most diagnostic cluster for GEPA reflection. Randomized tables/traces are contamination-proof.
**Adaptation:** Truncate all outputs to one derived token (never a trace); tune step counts so a default prompt lands mid-range; stratify by which rule the injected violation breaks.

### c06 · Artificial-Grammar Violation Classification — origin: both
**Domain:** formal-language membership / constraint diagnosis (MC)
**Merged from (brainstorm):** Artificial Grammar Well-Formedness Judgment; Symbolic Sequence Categorization (SSC); Nested-Bracket Path Resolver.
**Merged from (literature):** MLRegTest (JMLR 2024) / FLaRe; Dyck membership & next-legal-bracket (transformer-expressivity line).
**Summary:** Generator samples strings from a randomized formal grammar (feature-agreement pairs, nesting-depth limits, forbidden adjacencies, balanced multi-type delimiters) and either keeps them well-formed or injects exactly one violation of a known class; model outputs an MC letter: A well-formed / B agreement / C nesting / D adjacency, with an earliest-violation-by-scan-order tie-break. Exact letter match; GT is the injected class. Each constraint the prompt captures catches one violation class — clean pool-level incremental strata — and the MC label directly names the violated rule (maximally diagnostic). Variants: next-legal-closing-bracket, depth-transparent bracket-kind path queries. Seeded grammars are contamination-proof; 25% guess floor.
**Adaptation:** Balance labels across classes; compose 4–5 constraint types incl. one attribute-dependent depth limit so one-shot prompts catch 2–3 of 4 classes.

### c07 · Case-Marked Micro-Conlang Translation — origin: both
**Domain:** constructed-language morphology / translation to fixed template
**Merged from (brainstorm):** Case-Marked Micro-Language Translation.
**Merged from (literature):** Synthetic Case-Marked Micro-Language family; MTOB paradigm (2024); ConLangs metalinguistic probing generator (2025); SIGMORPHON-style synthetic inflection.
**Summary:** A generator builds an agglutinative micro-language per run: random roots plus affixes marking case, number, tense, plus a topic-fronting particle; instance = one conlang sentence + a tiny gloss table; output = an English paraphrase in a rigid 3–6-word template (fixed articles, fixed order), exact match after trivial normalization. Latent rules: case suffix determines subject/object regardless of position (the load-bearing rule naive SVO prompts miss), number/tense affixes, article insertion, fronting reorder, plus an inflection stratum with allomorphy. GT = deterministic template filler. Randomized morphology per run is contamination-proof; MTOB/LingOly document large headroom for grammar-from-context learning.
**Adaptation:** Replace chrF/METEOR with exact template match; keep affix inventory rotating; few-shot glossed pairs are the natural MIPROv2 demo axis.

### c08 · Rosetta→Match-Up MC Linguistic Puzzles — origin: both
**Domain:** linguistic-puzzle rule induction (multiple choice)
**Merged from (brainstorm):** Invented Tonal/Diacritic Orthography Disambiguation.
**Merged from (literature):** LingOly (NeurIPS 2024); From Rosetta to Match-Up (2026); Synthetic Rosetta/Match-Up generator family; rare-script decipherment (2025).
**Summary:** Seeded conlang grammar; instance = N glossed example pairs plus one held-out query presented as 4-way multiple choice (which pairing/meaning is correct), including a diacritic/tone homograph-disambiguation variant with a marker-precedence hierarchy (tone resolves homograph, context particle overrides tone). Distractors are constructed so each differs from the truth by exactly one latent rule (word order, agreement, affix meaning, diacritic override), so every captured rule eliminates one distractor — a built-in incremental ladder with diagnostic wrong answers. Output = single letter, exact match. Contamination-proof by regeneration; converts translation difficulty into clean MC scoring.
**Adaptation:** The distractor-isolates-one-rule construction is the load-bearing engineering; validate distractor quality on a pilot batch.

### c09 · Composable Cipher-Chain Decode (closed vocabulary) — origin: both
**Domain:** invented composable ciphers / reversible transforms
**Merged from (brainstorm):** Composable Cipher Chain Resolution; Cipher-With-Invented-Rules Decoder.
**Merged from (literature):** CipherBank (2025); CipherBench novel-cipher methodology; ICL Ciphers / Rashid (2025–2026); closed-vocab cipher-chain family; char-level caveat cluster (StringLLM/CUTE) as a design constraint.
**Summary:** Plaintext drawn from a fixed closed word list ("north gate open") is encoded by an ordered stack of 2–3 named STRUCTURAL reversible ops (keyed block reversal, invented digraph swap, positional skip — no shift arithmetic, which the char-level literature shows is token-noise-prone) with a per-instance key; instance = ciphertext + op stack + key; output = the plaintext words, exact match against the closed vocabulary. Latent rules: each op's inverse, the apply-inverses-in-REVERSE-order meta-rule, key-derived block sizes. Layer count is the difficulty dial; seeded keys/ops are contamination-proof. Failures are diagnostic per layer.
**Adaptation:** Keep ops structural not arithmetic; closed vocab (or MC over candidate plaintexts) to protect temp-0 resolvability; validate the noise floor before adoption.

### c10 · Interlocking-Cycle Calendar / Clock Arithmetic — origin: brainstorm
**Domain:** constructed cyclic systems / modular conventions
**Merged from (brainstorm):** Constructed Calendar/Cycle Date Arithmetic; Nonstandard modular clock arithmetic; Invented congruence classification; Modular Digit Oracle (sanity-stratum only).
**Merged from (literature):** — (modular counterfactual evidence covered under c01 sources).
**Summary:** An invented calendar of 2–3 coprime named cycles (Tzolk'in-style but randomized), plus a clock variant with positions labeled 1..m, forward-only subtraction, and sticky-move rules; instance = current tuple/position + "advance N"; output = canonical tuple string or single position, exact match vs per-cycle modular arithmetic. Latent rules: independent moduli (no shared carry), 1-indexed labels, a skip/leap rule on one cycle, sticky/ignored moves, canonical formatting. Randomized cycle lengths and names per run are contamination-proof; wrong-cycle errors isolate which modulus rule broke. Quirks are stratified so no single instruction sentence covers the whole pool.
**Adaptation:** Add 2–3 interacting quirks (offset cycle, every-other-day cycle) to keep the modular-arithmetic prior from collapsing the gap one-shot.

### c11 · House-Convention Canonicalizer — origin: both
**Domain:** serialization/normalization under invented house rules
**Merged from (brainstorm):** Canonical JSON Serializer; Identifier case-and-affix normalizer; Phone/number canonicalizer; Whitespace/quote linter fixup; Date/time fictional locale; Multi-rule redaction/masking; Slug/URL canonicalizer; CSV/record reserializer; Whitespace-and-indent block canonicalizer; Signed-Checksum Field Validator.
**Merged from (literature):** RFC 8785 JCS as generatable task; House-style normalizer family (slugify/phone/date/checksum, PolyNorm 2025); StructEval non-renderable slice; JSONSchemaBench (schema-pool donor).
**Summary:** Pick ONE deterministic normalization pipeline (canonical JSON, slug, identifier, or fictional-locale date/phone) governed by 5–8 invented house rules that deliberately deviate from every public standard: length-then-codepoint key sort, bespoke float/escape policy, invented month abbreviations and grouping tables, custom check digit. Instance = a messy value; output = the canonical string; exact whole-string match vs a ~40-line reference function (satisfying no-partial-credit — a single wrong rule fails the example). Each rule owns a stratum of instances so pool accuracy climbs rule by rule and expected/got diffs name the violated rule. Random inputs give an infinite contamination-proof pool; invented conventions are not one-shot guessable. Directly mirrors the project's dr-serialize canonical-JSON theme, so results transfer conceptually.
**Adaptation:** Choose one family (slug or canonical-JSON strongest); keep values small so canonical output stays under ~120 chars; output bytes doubles as the vestigial second objective.

### c12 · Sigil-Schema Field Extraction & Dialect Routing — origin: both
**Domain:** structured extraction under invented markups
**Merged from (brainstorm):** Glyph-Ledger Field Extraction; Positional Fixed-Width Decoder; Escaped-Delimiter CSV; Multi-Schema Router Extraction; Enum-Canonicalization With Hierarchy; Unit-and-Sign Normalizer; Protocol Bitmask.
**Merged from (literature):** EDC canonicalize step (EMNLP 2024); StructText reversed as generator (2025); DTBench/ExtractBench/LLMStructBench (provenance-unverified, motivation only).
**Summary:** Synthetic records in invented markups — sigil-tagged fields, fixed-width columns with implied decimals, custom-escaped delimiters, or one of several dialects signaled by a header token — where each instance asks for ONE field returned in normalized house form (integer-cents→decimal, dotted enum code via most-specific-wins, checksum VALID/INVALID). Output = a single short token; exact match vs the generator's own record; strictly one field per instance so scoring stays 0/1 with no intra-example partial credit. Latent rules: sigil→field map, per-dialect parsing conventions, column widths, escape handling, per-field normalization, enum precedence — each stratified across the pool. Randomized sigils/widths/delimiters/enum vocabularies per run are contamination-proof and genuinely resist one-shot guessing.
**Adaptation:** Rewrite all multi-field pipe-output brainstorm variants as single-field queries; ensure dialect mix is balanced so per-dialect rule capture shows as pool-level steps.

### c13 · Precedence-Ordered Decision-List Classification — origin: both
**Domain:** policy/triage classification with override rules
**Merged from (brainstorm):** Traffic-Light Priority Classifier; Priority Vote Aggregator; Temporal-Window Event Labeler; Priority-Conflict Record Reconciler; Orthogonal-Feature Labeling.
**Merged from (literature):** Certifiably optimal rule lists (first-match semantics); LLMTabBench (2026); RuleArena synthetic rule engine (ACL 2025); In-Context Boolean Concept Learning (2024).
**Summary:** Generator samples a short ordered decision list (first-match semantics) over 4–6 categorical/numeric features plus override rules — security=true forces P0, senior veto, regional cutoff shift, paying-never-SUPPRESS floor — with an explicit precedence chain; instance = one record (or vote set / event), output = one label from a ~5-way set, exact match vs executing the list. Each rule owns a distinct case slice, so partial rule coverage yields partial pool score and failures are diagnostic ("all security=true cases wrong" = that rule missed). Feature-combinatorial seeded generation is contamination-proof; ≥5 interacting rules with ordering resist one-shot specification. Boolean-concept complexity is a principled difficulty dial.
**Adaptation:** Build the rule-list sampler; relabel classes with opaque symbols to defeat priors; balance strata so 10–20-task internal evals resolve prompt differences.

### c14 · Instruction-Hierarchy Priority Ledger — origin: both
**Domain:** authority/conflict resolution among directives
**Merged from (brainstorm):** Priority Ledger.
**Merged from (literature):** IHEval (2025); The Instruction Hierarchy (2024); Control Illusion (2025); CFBench requirement-prioritization.
**Summary:** Instance = several short directives tagged with NONCE authority levels — some quoted (inert), negated, conditional, or mutually conflicting — each proposing a candidate answer token; a hidden seeded authority policy (tag-priority order, quotation inertness, same-level recency, negation handling, explicit exceptions, fallback) determines which directive controls. Output = a single letter A–F, exact match vs the policy program. Factorial minimal-pair generation isolates each policy feature, so a wrong answer names the misunderstood rule. Nonce tags whose hierarchy differs from familiar system>user conventions defeat pretrained priors; Control Illusion documents that models genuinely mis-prioritize, giving a real non-one-shot gap. Fully synthetic and contamination-proof.
**Adaptation:** Verify the gap is incremental (not binary) on the chosen cheap model; keep 5+ independently seeded policy features.

### c15 · Opaque-Codebook Remapped-Label Classification — origin: both
**Domain:** classification under remapped label semantics
**Merged from (brainstorm):** Contrarian Sentiment Codebook; Cipher-Category Tagger.
**Merged from (literature):** Flipped-label ICL (Wei et al. 2023); Symbol Tuning / SUL-ICL; Semantic Anchors (2025); In-Context Fixation (2026); Demonstration Shortcut (2024); contrarian-codebook synthetic family; Ethos/Liar/SST-suite (format template only).
**Summary:** Template-generated short texts (reviews / event sentences) are labeled with opaque codes (ALPHA..DELTA, or compositional tags like `#mix.cal`) by a deterministic feature→code function with 4+ interacting rules: base polarity map, sarcasm inversion, grudging-praise trigger ("finally", "after N tries"), neutral-fact carve-out, weekday suffix, and a precedence order among them. Output = one code token, exact match. Default prompts follow the pretrained sentiment prior and score near floor (the flipped-label literature's documented mechanism); each rule conveyed lifts its stratum. Seeded slot-filling generation is contamination-proof.
**Adaptation:** Use 4+ rules with precedence (a bare flip is one-shot guessable); per the Semantic Anchors caution, confirm the chosen cheap model can override priors at all before adopting.

### c16 · Wason Implication-Compliance Classifier — origin: both
**Domain:** conditional-rule compliance (binary/MC)
**Merged from (brainstorm):** Wason-Style Rule Compliance Classifier.
**Merged from (literature):** Wason Selection Task LLM adaptations (EACL 2026; deductive-competence 2023).
**Summary:** Records of 4–6 visible attributes are checked against 4–5 conjoined implication rules ("if shape=circle then edge=solid"); a record is VIOLATION iff some rule's antecedent holds and its consequent fails (vacuous cases are OK). Output = OK/VIOLATION, or the violated-rule letter for richer diagnostics and a lower guess floor; exact match vs a trivial boolean. Latent rules: each implication, vacuous-truth handling, directionality (not biconditional), distractor attributes. Distinctive property: models systematically err on vacuous truth and directionality even when rules are fully stated, so a "perfect" instruction does not instantly hit ceiling — unusual, desirable model-side residual. Attribute-combinatorial seeded generation is contamination-proof.
**Adaptation:** Prefer the which-rule-violated MC output over bare binary to escape the 50% guess floor; balance vacuous/violating cases across strata.

### c17 · Invented-Gate Circuit Evaluation — origin: both
**Domain:** boolean-circuit / invented-connective evaluation
**Merged from (brainstorm):** Marble-Track / Logic-Gate Signal Propagator; Micro-Logic Notation Truth Evaluation.
**Merged from (literature):** Circuit / logic-gate DAG synthetic family (CIRCUIT 2025; propositional circuit-analysis 2024).
**Summary:** A random small DAG of 4–6 gate types whose truth tables are INVENTED and randomized per run; a formula-syntax variant adds invented connective glyphs, nonstandard precedence, undefined-atom propagation, and ill-formedness detection. Instance gives primary inputs and asks the value at a named node (0/1) or an MC verdict (true/false/undefined/ill-formed); output = a single token, exact match vs evaluating the DAG/formula. Each gate/connective table is an independent, cleanly demo-learnable latent rule — no fixed instruction can encode randomized tables without demos, making the MIPROv2 demo axis load-bearing. Wrong-gate errors produce diagnostic patterns. Seeded random semantics are contamination-proof; boolean arithmetic gives strong temp-0 determinism.
**Adaptation:** Use 4–6 gate types so table capture is incremental; query nodes whose paths cover different gates per stratum.

### c18 · Depth-Controlled Synthetic Deduction (True/False/Unknown) — origin: both
**Domain:** synthetic logic / multi-hop deduction
**Merged from (brainstorm):** Constraint-Satisfaction Seating Puzzle (unique-solution stratum).
**Merged from (literature):** RuleTaker (2020); ProofWriter (2021); FLD/FLD* (ICML 2023, NeurIPS 2024); PrOntoQA(-OOD) (2023); LogicNLI (2021); RuleBERT (soft-rule stratum); SATBench (2025, hard tail); FOLIO (style reference only).
**Summary:** Regenerate RuleTaker/FLD/PrOntoQA-style theories: sampled facts + if-then rules over fictional predicates ("every wumpus is a yumpus") with controlled proof depth (D0–D5), distractor rules, and closed- vs open-world semantics; instance = theory + query; output = True/False/Unknown, exact match. GT comes from the generator's proof engine, which also names the failed inference step — strong reflection diagnostics. Latent rules: chaining depth, negation-as-failure vs explicit negation, open-world Unknown, exceptions, quantifiers; depth bins give the incremental ladder and a known ceiling. A unique-solution constraint-puzzle stratum (seating arrangement queried for one fact, name or YES/NO output) covers the CSP flavor. Fresh symbols per run are contamination-proof.
**Adaptation:** Drop all proof-chain generation (label only); regenerate rather than reuse published sets; strip NL templating toward compact symbolic rendering to bound tokens.

### c19 · Grid-World State Prediction — origin: both
**Domain:** deterministic spatial micro-simulation
**Merged from (brainstorm):** Grid Robot Deterministic Navigator.
**Merged from (literature):** GridRoute (2025); GRASP (2024); LLM-BabyBench Predict split (2025); bAbI entity/state tracking (regenerated).
**Summary:** A small seeded ASCII grid (≈6x6) with walls, a robot pose, and a command string including invented commands (J=jump-2, W=wrap); the model predicts ONE derived fact — final coordinate, heading, or carrying-flag — never a path, which removes many-valid-answers ambiguity. Output = a single token ("3,5" or "E"), exact match vs a tiny simulator. Latent rules: relative turn semantics, wall blocking vs edge wrap/clamp, invented command semantics (jump-through-wall behavior), origin/indexing convention — varied across strata so a single fixed convention statement cannot cover the pool. Randomized grids are contamination-proof; off-by-one-at-a-wall failures are cleanly diagnostic.
**Adaptation:** State-prediction framing only; add 2–3 invented commands whose semantics are best pinned by demos; keep grids small ASCII.

### c20 · Invented Micro-Game Turn Resolution — origin: both
**Domain:** invented game rules / deterministic resolution
**Merged from (brainstorm):** Invented Card Trick-Taking Scorer; Dice/Token Resource Micro-Game Resolver.
**Merged from (literature):** TextArena-style invented-game family; Game Reasoning Arena (2025); ZeroSumEval (2025).
**Summary:** A seeded generator invents a tiny deterministic game per run: nonstandard rank order (7-high), trump/override cards, resource exchange rates, caps with overflow discard, and a counterintuitive action-resolution priority; instance = a game state plus a trick or move sequence; output = winning card id, a final token count, or WIN/LOSE — one token, exact match vs the reference resolver. Each rule flips a distinct instance subset, so rule capture is incremental and failures are diagnostic ("used real-world ace-high" is legible to a reflection LM). Strong priors toward real card games guarantee a low default floor; invented rules per run are contamination-proof and few-shot worked turns measurably teach rank order and priority.
**Adaptation:** Highest design effort in the list: rules must be validated unambiguous with a reference ~100% prompt before use; inject illegal moves for the failure-path criterion.

### c21 · Relational Micro-World with Invented Vocabulary — origin: brainstorm
**Domain:** relational/kinship QA over nonce graphs
**Merged from (brainstorm):** Kinship Convention QA; Relational Micro-World.
**Merged from (literature):** — (adjacent to PrOntoQA's fictional-symbol approach, covered separately in c18).
**Summary:** A generator builds a small graph of nonce entities and relations where relation symbols carry seeded nonstandard properties (symmetry, transitivity, inverse mapping, two-relation composition, explicit-negation precedence, closed-world handling) and answers use an invented closed vocabulary (gender-neutral 'nieph', 'bond-' prefix for in-law links, a relation-distance cutoff). Instance = a few facts + one query; output = one/two words from the closed vocabulary or a 4-way MC letter (true/false/both/unknown), exact match vs graph traversal under the seeded property assignments. Failures cleanly separate codebook misses (model used a standard English term) from traversal misses. Seeded names and property assignments are contamination-proof; demos teach the invented vocabulary directly.
**Adaptation:** Keep hop depth low enough that a full-rule prompt reaches ~100%; balance unknown cases.

### c22 · Stacked Verifiable-Constraint Micro-Generation — origin: both
**Domain:** instruction following / constrained generation
**Merged from (brainstorm):** Constrained Micro-Description.
**Merged from (literature):** IFEval (2023) checker library; COLLIE (2024) constraint grammar; VFF (2025); IFBench + IF-RLVR OOD constraints (2025); FollowBench levels; InFoBench/DRFR decomposition; ComplexBench Selection operator; CFBench taxonomy; LIFEBench length; RewardBench 2 Precise-IF atoms; GEPA IFBench results.
**Summary:** Instance = a trivial micro-task plus 3–5 composed constraints drawn from a deterministic checker library (exact word count, forbidden letter, required keyword, casing, end-with token), including conditional Selection constraints ("if the input has property P obey rule A else rule B") that must be DISCOVERED, not just copied. Output = a few words; the example scores 1 iff ALL checkers pass (0/1, no partial credit), with per-checker verdicts logged as diagnostic facts for reflection. Constraint atoms and base tasks are generated synthetically (COLLIE-style grammar), giving an infinite contamination-proof pool; constraint count/level is the difficulty dial and IFBench-style OOD atoms resist one-shot enumeration. LIFEBench/VFF document low small-model floors.
**Adaptation:** Restrict to short-output-checkable atoms; use strict all-pass scoring with stratified constraint combos so pool score climbs as prompts internalize more atom types.

### c23 · Hidden-Rule String-Transform Induction Ladder — origin: both
**Domain:** compositional rule induction from I/O examples
**Merged from (brainstorm):** Compositional String Decode (CSD).
**Merged from (literature):** InductionBench (2025, both survey entries); Instruction Induction suite (ACL 2023 / APE); List Functions (2021) + MIR-Bench (2025); symbolic hidden-rule mapping family; PCFG SET (2020); SCAN/MiniSCAN + COGS/SLOG (regenerated, bounded); Re-ARC/1D-ARC (linearized stratum); WILT/HERO'S JOURNEY/ARISE (generators only).
**Summary:** The designer defines K independent transformation rules (casing keyed to token length, positional swaps, subregular ISL/OSL string edits, per-primitive command→action mappings, 1D-ARC-style short-array transforms); each seeded instance applies a rule subset to an input. The model sees few-shot I/O demos plus one query and outputs the transformed short string — or a 4-way MC among candidates each wrong in exactly one rule. Exact match vs the reference transformer; prompt quality equals how many rules it states or induces, giving a smooth incremental ladder with a known ceiling (state all rules) and floor (naive prompt); demos are load-bearing (MIPROv2 axis alive by construction). Randomized rule subsets and invented vocabularies are contamination-proof. InductionBench's finding that frontier models fail even simple classes warns to curate complexity so the easiest strata are provably reachable.
**Adaptation:** Curate rule complexity against the reachability risk; bound outputs to short strings or MC; use the wrong-in-exactly-one-rule distractor trick for maximal diagnosis.

### c24 · Composite-Key Sorting & Canonical Ordering — origin: both
**Domain:** ordering/formatting under invented conventions
**Merged from (brainstorm):** Invented-rule sorting of symbolic tokens; CaseOrder; MultiRuleSort; Permutation composition with nonstandard conventions.
**Merged from (literature):** BBH word_sorting / symbolic-subset generators (OPRO/EvoPrompt substrate).
**Summary:** Instance = 3–6 tokens or (letter,number) pairs — or, in one stratum, 3–4 small permutations in cycle notation with a stated composition-direction convention — governed by a nonstandard composite key: sort by second letter, ties by length descending, vowels reranked after consonants, per-element transforms (double if even, triple if odd), canonical cycle form (smallest element leads). Output = one comma-joined line or canonical string, exact match vs a reference sorter/composer. Latent rules: primary key, each tie-breaker, rerank rule, element transform, formatting/separator conventions — each visible as a distinct diff signature. Sort keys and directions are randomized per batch so the prompt must convey them rather than guess; seeded token lists are contamination-proof and BBH evidences large prompt-sensitive headroom on this format.
**Adaptation:** Randomize key/direction per batch; ask for the full ordering so every key level is exercised; keep lists short so the output stays one bounded line.

---

## Dropped clusters (9)

| # | Cluster | Constituents | Why dropped |
|---|---------|--------------|-------------|
| D1 | Inverted multiple-choice convention | brainstorm "Inverted Multiple-Choice"; lit "Option-label / MC format-remapping sensitivity" | Both sources flag HIGH one-shot risk — a single sentence ("answer the alphabetically-first incorrect option") closes the gap, failing criterion 4. Remapped-label ground covered by c15; usable at most as a plumbing check. |
| D2 | Interval-Overlap Field Selection | brainstorm item | Only 2–3 boundary/tie conventions, enumerable in one prompt; boundary-convention flavor absorbed by c12/c13 strata. |
| D3 | TokenCounter (distinct vowel/consonant counting) | brainstorm item | Letter counting is tokenization-bound; StringLLM/CUTE literature shows char-count tasks are noisy at temp 0, directly threatening criterion 5. |
| D4 | CRUXEval-style code-output prediction | lit CRUXEval/CRUXEval-X | Published set heavily contaminated; regeneration needs a code DSL and unconstrained return types fight exact-match bounding. Execution flavor covered by c05, rule-induction flavor by c23. |
| D5 | Grounded grid compositional (gSCAN/ReaSCAN) | lit items | Grid rendering + action-sequence outputs are the most expensive to bound; own reviewer judged it "not worth the adaptation cost." Grid covered by c19, compositional splits by c23. |
| D6 | DSPy/GEPA reference benchmarks | lit GSM8K, HotpotQA, HoVer, PUPA | Retrieval pipelines, multi-node graphs, soft/composite metrics, contamination — fail criteria 1–2 outright. Retained only as published evidence that COPRO/MIPROv2/GEPA close incremental gaps. |
| D7 | Contaminated classification templates | lit Ethos, Liar, SST-2/Subj/TREC/AG-News suite | Single latent construct, contaminated, subjective labels; their cheap single-call MC format is inherited by c13/c15 synthetic reframes. |
| D8 | FoFo format-following | lit item | LLM-as-judge scoring fails criterion 2 (deterministic exact check); format-idea content absorbed into c11/c22. |
| D9 | Char-level string-task cluster | lit StringLLM, CUTE, letter-counting, Divide-and-Conquer | Not a task: kept as the red-flag design caveat that justifies closed-vocab/MC framing in c09 and the D3 drop. |

Reference-only literature entries folded into kept candidates' source lists rather than counted as clusters: GSM-Symbolic/GSM-Plus (c02 methodology), NumericBench (c01), JSONSchemaBench (c11), StructText/DTBench/ExtractBench/LLMStructBench (c12, provenance caveats noted), FOLIO (c18 style reference), cryptarithmetic (c01 hard tail), L-systems (c04), Long-Horizon Execution (c05), MTOB/ConLangs/SIGMORPHON (c07), rare-script decipherment (c08).
