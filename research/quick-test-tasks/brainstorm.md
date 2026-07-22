# Quick-Test Task Brainstorm

Consolidated brainstorm of candidate quick-test tasks for validating the
prompt-optimization harness (Eval identity, COPRO, MIPROv2, GEPA, Codex CLI
agent) before the HumanEval+ code-compression experiment.

Authoritative rubric: `design/quick-test-rubric.html`. In short, an ideal task
has: one LLM call + exact deterministic scoring (no sandbox); bounded MC / few-word
output; default prompts well below a KNOWN ceiling with an INCREMENTAL gap;
difficulty that decomposes into independent latent rules; a synthetic/generatable
pool with exact ground truth (hundreds of instances, contamination-resistant);
diagnostic failures for a reflection LM; few-shot demos that measurably help; and
prompt-quality differences resolvable on 10-20 task evals at temperature 0.

Below, every one of the 76 brainstormed ideas is preserved, lightly grouped by
theme. Each entry notes its source lens/model. Duplicates and near-duplicates
across the independent brainstorm passes are flagged inline — these repeats are a
useful signal that a theme is robust.

Source-model legend (as tagged in the raw brainstorm):
`claude-opus-4-8[1m]`, `kimi-k2p7`, `deepseek-v4-flash`, `minimax-m3`,
`gpt-5.x-codex`, and several "first-principles design" entries (untagged model).

---

## Theme A — Constructed notation / invented arithmetic & number systems

These invent glyphs, bases, operators, or units and ask for a numeric or short-string
result under nonstandard conventions. Strong on independent latent rules (glyph maps,
precedence, radix, carry, modulus) and contamination resistance via per-instance
randomization. Core one-shot risk: arithmetic is a known skill, so ceiling prompts
can be compact — mitigated by stacking quirks and randomizing which quirks fire.

- **Glyphscript Arithmetic Decode** (`claude-opus-4-8[1m]`) — invented numeral+operator
  system; infer glyph->digit map, operator meanings, nonstandard precedence,
  associativity, modular wrap, reset-glyph edge case. Output: single integer.
  One-shot risk MODERATE (mitigate by randomizing the glyph->digit map per instance).
- **Notation Unit-System Conversion** (`claude-opus-4-8[1m]`) — made-up units with
  non-decimal factors, mixed-radix write-out, carry/borrow, canonical normalization,
  truncation/rounding, a balanced/signed-digit unit. Output: integer or mixed-radix string.
- **Nonstandard Positional Base Evaluation** (`claude-opus-4-8[1m]`) — invented positional
  numerals, non-uniform/negative/balanced base, digit legend, endianness varied per
  instance, null glyph that skips a position. Output: single integer (possibly negative).
  Flagged higher one-shot risk in its lens (positional eval is a known skill).
- **Constructed Calendar/Cycle Date Arithmetic** (`claude-opus-4-8[1m]`) — interlocking
  coprime cycles (Tzolk'in/Haab-style, randomized); independent modular advance per cycle,
  legend parse, a leap/skip rule, canonical tuple formatting. Output: tuple string.
- **Custom binary operator evaluation (layered # operator)** (untagged / first-principles) —
  `a # b` defined by a stack of 4-6 rules keyed on operand properties (parity branch,
  modular residues 1..m, prime bonus, threshold subtract, left-associativity, identity).
  Output: single integer. *Near-duplicate of Glyphscript's operator-semantics core.*
- **Nonstandard modular clock arithmetic** (untagged) — clock labeled 1..m, forward-only
  subtraction, sticky moves ignored under a condition, direction-flip on even sizes.
  Output: integer in 1..m. One-shot risk MODERATE-HIGH (needs 3+ orthogonal conventions).
  *Overlaps Constructed Calendar (modular advance) themes.*
- **Balanced mixed-radix / invented positional number system** (untagged) — nonstandard/
  signed digit set (balanced ternary, factorial base, aliased alphabet), carry threshold,
  MSD ordering, zero representation. Output: short string. *Duplicate of Nonstandard
  Positional Base Evaluation and Notation Unit-System (mixed-radix) above.*
- **Custom-precedence arithmetic mini-grammar** (untagged) — redefined precedence/associativity
  over a small operator set + custom unary op; evaluate unparenthesized expression under the
  table. Output: single integer. One-shot risk MODERATE-HIGH (precedence tables are compact).
- **Modular polynomial / linear-recurrence next-term under invented rules** (untagged) —
  recurrence over a nonstandard modulus with aliased residues, additive index term, 0/1-based
  indexing. Output: single residue/aliased symbol.
- **Custom Arithmetic Precedence (CAP)** (`kimi-k2p7`) — MC over 4 integer options; custom
  precedence, operator semantics (+ means max/mod, * means subtract), wrap/clamp rule,
  parentheses-ignored-or-override. Output: A/B/C/D. *MC variant of Custom-precedence grammar.*
- **EncodedArith** (`deepseek-v4-flash`) — add two ints, left-pad sum to 3 digits, encode each
  digit via a per-instance digit->letter key, space-join. Output: 3 space-separated tokens.
- **Keyed-DSL Arithmetic** (`minimax-m3`) — per-instance symbol table maps 6-10 tokens to ops,
  then a chained expression; precedence flipped per instance, decoy symbols, mid-expression
  state mutations, final post-processing (abs/mod/sign). Output: single integer.
  *Strongest anti-one-shot member of this theme (state-mutating ops resist verbalization).*
- **Micro-Logic Notation Truth Evaluation** (`claude-opus-4-8[1m]`) — invented connective
  glyphs incl. a nonstandard gate, nonstandard precedence, atom-truth legend, undefined
  propagation, ill-formedness detection. Output: A/B/C/D (true/false/undefined/ill-formed).
  *Bridges to Theme D (logic); listed here for the invented-notation evaluator core.*

## Theme B — Constructed language / morphology / translation

Artificial grammars mapping marked tokens to a canonical output template. Independent
affix rules; default prompts assume English SVO. One-shot risk rises if the affix
inventory is small and stable.

- **Case-Marked Micro-Language Translation** (`claude-opus-4-8[1m]`) — agglutinative suffixes
  mark role/number/tense; case suffix determines subject/object regardless of position;
  topic-fronting particle reorders output. Output: fixed-template 3-6 word English phrase.
  One-shot risk MODERATE-HIGH.
- **Invented Tonal/Diacritic Orthography Disambiguation** (`claude-opus-4-8[1m]`) — base glyph +
  stacked diacritics; tone resolves homographs, context particle overrides tone, precedence
  among markers. Output: A/B/C/D meaning. *Shares homograph/precedence structure with the
  enum/codebook tasks in Theme F.*

## Theme C — Ciphers, encodings & composable reversible transforms

Layered invented transforms decoded back to a closed vocabulary. Clean per-layer
diagnostics; main caveat is char-arithmetic noise threatening temp-0 resolvability
(criterion 5), partly mitigated by closed-vocab nearest-match outputs.

- **Composable Cipher Chain Resolution** (`claude-opus-4-8[1m]`) — ordered stack of named
  reversible ops with a per-instance key; decode applies inverses in REVERSE order; keyed
  block sizes, invented digraph swaps. Output: few closed-vocabulary words.
- **Cipher-With-Invented-Rules Decoder** (untagged / first-principles) — per-position shift
  schedule + substitution twist + positional skip; decode to a closed nonsense-word list.
  Output: short plaintext token. Flagged higher-noise / one-shot risk MEDIUM-HIGH.
  *Near-duplicate of Composable Cipher Chain (composed reversible layers).*

## Theme D — Formal grammars, rewrite systems & logic/deduction

Abstract symbol systems: well-formedness judgments, term rewriting to normal form,
truth evaluation, implication compliance. Some of the strongest anti-one-shot designs
(abstract productions have nothing to guess); risk instead is being too hard to learn.

- **Artificial Grammar Well-Formedness Judgment** (`claude-opus-4-8[1m]`) — verdict MC over
  agreement / nesting-depth / adjacency / balanced-delimiter violations; report earliest by
  scan order. Output: A/B/C/D. Injected single-class violations make failures maximally diagnostic.
- **Symbolic Rewrite-System Normal Form** (`claude-opus-4-8[1m]`) — confluent rewrite rules,
  leftmost-outermost strategy, reduce to fixpoint, overlap disambiguation, guarded rule.
  Output: short normal-form string.
- **Expression rewriting under invented rewrite rules (normal form)** (untagged) — small
  term-rewriting system (`ab->c`, etc.), leftmost-innermost, fixpoint, empty-string deletions,
  rule priority. Output: short string or `NORMAL`. *Duplicate of Symbolic Rewrite-System Normal
  Form.*
- **Grammar-of-rewrites: apply an ordered rule set to a token string** (untagged /
  first-principles) — abstract productions over a tiny alphabet, leftmost-innermost, iterate to
  fixed point, empty-string handling. Output: reduced string, ~1-8 chars. Rated VERY LOW / lowest
  one-shot risk in the set. *Third instance of the rewrite-to-normal-form idea.*
- **Turing-Tape Rewrite (bounded tag-system)** (untagged / first-principles) — tag-system with
  per-leading-symbol append + front-delete, append-vs-delete ordering, halt condition, step cap.
  Output: final string or integer length. Flagged char-arithmetic-noisy. *Rewrite-system family.*
- **Micro-Logic Notation Truth Evaluation** (`claude-opus-4-8[1m]`) — *also relevant here;*
  primary listing under Theme A.
- **Invented congruence / 'nonstandard equals' classification** (untagged) — equivalence
  relation over integers (digit-sum-mod-k AND last-digit-parity, etc.); MC YES/NO. Output:
  YES/NO or A/B/C/D. One-shot risk HIGH; author flags it as the cheap format/failure-path
  validator rather than the primary incrementality test.
- **Wason-Style Rule Compliance Classifier** (untagged / first-principles) — implication rules
  over attributes; vacuous-truth handling, directionality (not biconditional), conjunction,
  distractor attributes. Output: OK / VIOLATION. Notable: even a perfect instruction leaves
  model-side residual (implication errors), so one-shot risk is MEDIUM-LOW.
- **Relational Micro-World** (`gpt-5.x-codex`) — nonce entities/relations; learn symmetry,
  transitivity, inverse, two-relation composition, negation precedence, closed-world handling.
  Output: A/B/C/D (true/false/both/unknown). RuleTaker/ProofWriter-adjacent.
- **Artificial Grammar / Well-Formedness** overlaps: the SSC entry in Theme F also tests
  conjunctive symbolic feature rules.

## Theme E — Deterministic simulation & state machines (micro-games, protocols, automata)

Run an invented machine/game/graph forward to a bounded answer. Rich independent-rule
decomposition (per-transition, per-gate, per-move) and injected-violation diagnostics.
Familiar templates (grids, schedulers) raise one-shot risk; invented commands/tables lower it.

- **Glyph Automaton (deterministic finite-state tape machine)** (untagged / first-principles) —
  arbitrary transition table (latent), undefined-transition fallback, reset symbols, accept
  condition. Output: state label / ACCEPT / REJECT. One-shot risk LOW (table randomized per run).
- **Protocol Handshake Validator** (untagged / first-principles) — state-machine protocol over
  message logs; ordering, mutual-exclusion/counts, field monotonicity, post-CLOSE prohibitions,
  required-response pairing. Output: VALID or first-violation index. Injected per-rule violations.
- **Protocol Bitmask** (`gpt-5.x-codex`) — synthetic packet; return an error code under a seeded
  protocol (field order, case norm, required-field logic, checksum, mode constraint, forbidden
  pair, arbitrary error-bit assignments). Output: integer 0-127. XOR-diagnosable. *Overlaps
  Protocol Handshake Validator and Signed-Checksum Validator.*
- **Grid Robot Deterministic Navigator** (untagged / first-principles) — commands F/L/R + invented
  (jump-2, wrap); obstacle blocking, edge wrap-vs-block, origin convention. Output: coordinate or
  heading. One-shot risk MEDIUM (grid nav is familiar).
- **Dice/Token Resource Micro-Game Resolver** (untagged / first-principles) — invented exchange
  rates, resource caps/overflow, action priority/resolution order, payment-before-gain, win
  threshold. Output: integer count or WIN/LOSE.
- **Invented Card Trick-Taking Scorer** (untagged / first-principles) — invented suits/ranks,
  trump precedence, follow-suit eligibility, non-standard rank order, override cards, tie-break,
  combo bonus. Output: card id or integer score. Strong prior toward real card games = headroom.
- **Priority-Rule Scheduler Resolver** (untagged / first-principles) — invented scheduling
  discipline (preemption, tie-breaks, aging/priority-boost, idle handling, inclusive/exclusive
  time). Output: job id or integer time.
- **Marble-Track / Logic-Gate Signal Propagator** (untagged / first-principles) — DAG of invented
  2-input gates (arbitrary truth tables), topological eval, unconnected-input default, multi-fanin
  rule. Output: bit or small integer count. One-shot risk LOW-MEDIUM (tables must come from demos);
  makes MIPROv2's demo axis load-bearing.
- **Opaque Device State** (`gpt-5.x-codex`) — <=6 nonce commands over 3 binary lamps; per-command
  effect, toggle vs assignment, parity-conditioned command, adjacent-command cancellation, reset
  scope, opaque state->label map. Output: A-H. *Near-duplicate of Grid Robot / Glyph Automaton
  simulation core with an opaque output mapping.*
- **Permutation composition with nonstandard notation and convention** (untagged) — cycle notation,
  composition direction convention, 1-indexing, canonical cycle form. Output: cycle string or
  integer image. *Group-theory sim; overlaps automata/rewrite execution demands.*
- **Constraint-Satisfaction Seating Puzzle (single-answer variant)** (untagged / first-principles) —
  unique-solution line/circle placement; immediate-vs-loose adjacency, left/right directionality,
  circular wrap, negation, exclusivity. Output: name or YES/NO. Zebra-puzzle-adjacent.

## Theme F — Classification under nonstandard / remapped label semantics

Opaque codebooks and override policies defeat the pretrained label prior. Excellent
independent-rule / precedence decomposition and clean MC or short-label scoring. Main
risk: a diligent engineer can enumerate a small codebook in one paragraph — mitigated by
4+ interacting rules with precedence, per-instance keys, and compositional labels.

- **Contrarian Sentiment Codebook** (untagged / first-principles) — opaque codes ALPHA..DELTA;
  base map, sarcasm inversion, grudging-triggers, neutral-fact carve-out, precedence. Output:
  single code token. One-shot risk MEDIUM-HIGH.
- **Cipher-Category Tagger (symbol semantics remap)** (untagged / first-principles) — compositional
  tag (base #up/#dn/#mix + optional .cal suffix); polarity map, co-occurrence -> #mix, weekday
  suffix, cancellation, exact formatting. Output: composed tag string.
- **Inverted Multiple-Choice (answer-the-wrong-one convention)** (untagged / first-principles) —
  report a letter that is NOT the correct option under a selection convention. Output: A-D.
  One-shot risk HIGH; author flags it as a floor/plumbing check, not the main incremental task.
- **Priority Vote Aggregator (weighted-label reconciliation)** (untagged / first-principles) —
  role-weighted votes, senior veto, guest-ignore-unless-alone, conservative tie-break, precedence.
  Output: ACCEPT/REVISE/REJECT. *Overlaps Priority-Conflict Record Reconciler (Theme G) and
  Priority Ledger.*
- **Traffic-Light Priority Classifier (multi-rule decision procedure)** (untagged /
  first-principles) — incident flags -> P0..P3/SUPPRESS via overrides (security forces P0, region
  downgrade, age upgrade, paying-never-suppress, precedence). Output: 5-way label. Archetypal
  independent-rules task.
- **Temporal-Window Event Labeler (calendar convention QA)** (untagged / first-principles) —
  routing label from date/time/type; cutoff -> FREEZE, Friday rule, regional cutoff shift, hotfix
  bypass, weekend HOLD, boundary inclusivity. Output: routing label.
- **Unit-Convention Numeric QA (nonstandard measurement rules)** (untagged / first-principles) —
  trivial arithmetic reported under stacked conventions (unit divisor, rounding direction, numeral
  base) selected by keywords with precedence. Output: number/numeral string. *Bridges to Theme A.*
- **Kinship Convention QA (nonstandard family tree)** (untagged / first-principles) — invented
  gender-neutral relation terms, distance cutoff, in-law 'bond-' prefix, directionality. Output:
  1-2 words from a closed invented vocabulary.
- **Modular Digit Oracle (base-and-offset)** (untagged / first-principles) — hidden arithmetic
  convention over a digit string (position selection, per-class sign, modulus, permutation table).
  Output: single digit. One-shot risk HIGH (closed-form saturates fast); a floor sanity task.
  *Bridges to Theme A.*
- **Symbolic Sequence Categorization (SSC)** (`kimi-k2p7`) — classify an 8-12 char sequence into
  one of four categories defined by a conjunction of features (target bigram, count parity, prefix
  pattern, run-length threshold). Output: A/B/C/D.
- **TokenCounter** (`deepseek-v4-flash`) — output `V|C|W` = distinct vowels, distinct consonants,
  total words; the counterintuitive "distinct" nuance is the trap. Output: pipe-delimited ints.
- **MultiRuleSort** (`deepseek-v4-flash`) — sort (letter,number) pairs by number then alpha, then
  double-if-even/triple-if-odd, comma-join. Output: comma-separated ints. *Overlaps CaseOrder /
  Invented-rule sorting (Theme G).*
- **Orthogonal-Feature Labeling** (`minimax-m3`) — 4-5 categorical attributes + few-shot demos;
  label from 2-3 hidden instance-specific rules with override conditions and tie-breaks. Output:
  short string. Tie-breaker: 2024-2026 in-context rule-induction research.
- **Compositional String Decode (CSD)** (`kimi-k2p7`) — pick which of 4 candidates results from a
  hidden fixed protocol (op order, casing rule, delimiter rule, truncation). Output: A/B/C/D.
  *Overlaps Theme C ciphers and Theme G string transforms, but MC-scored as classification.*

## Theme G — Structured extraction / canonicalization / serialization to a schema

House-style normalizers and extractors: exact-string canonical output under arbitrary
conventions. Directly mirrors the project's own dr-serialize canonical-JSON theme.
Independent per-field/per-rule ladders; contamination-proofed by inventing the house style.
Watch all-or-nothing cascades (use per-field partial scoring or few fields).

- **Canonical JSON Serializer (house style)** (untagged / first-principles) — nonstandard key
  order, float normalization, whitespace policy, escape policy, arrays-not-sorted trap, last-wins
  dedup. Output: canonical string. One-shot risk MEDIUM (deviate from RFC-8785/JCS deliberately).
- **Identifier case-and-affix normalizer** (untagged) — camelCase target, acronym collapse,
  verb-prefix strip, digit boundaries, separator collapse, stopword lowercasing. Output: one token.
- **Phone/number canonicalizer with regional latent rules** (untagged) — fictional numbering plan;
  strip separators, dial-code table, invented grouping, extension rewrite, trunk stripping. Output:
  formatted string. One-shot risk LOW (invented plan).
- **Whitespace/quote/style linter fixup (single-line)** (untagged) — comma spacing, paren padding,
  quote-preference with exception, operator spacing, semicolon spacing, comment-gap rule. Output:
  one line.
- **Date/time reserialization to a fictional locale format** (untagged) — DMY with `.` separators,
  invented month abbreviations, `HHhMM` dropping seconds, pad-day-not-hour, locale suffix, offset
  table. Output: formatted datetime string. *Overlaps Temporal-Window (Theme F) on date parsing.*
- **Roman-numeral / base-N transcoder with house quirks** (untagged) — Roman with IIII-not-IV
  exception + overline thousands, plus invented base-4 alphabet mode. Output: numeral string.
  *Overlaps Theme A number-system tasks.*
- **Multi-rule text redaction/masking canonical form** (untagged) — per-field masks (account keep
  last 4 with `#`, PIN all `*`, name -> initials), preserve labels, leave unknown fields untouched.
  Output: masked line.
- **Slug/URL canonicalizer with independent transforms** (untagged) — lowercase+hyphenate, diacritic
  fold, symbol substitution table, stopword removal, section prefix, truncate at word boundary,
  collapse hyphens. Output: slug string. Long incremental ladder.
- **CSV/record reserializer with quoting and column rules** (untagged) — trim outer/preserve inner,
  RFC-style quoting with doubling, boolean coercion table on designated columns, empty-field render
  trap. Output: CSV row. *Overlaps Escaped-Delimiter CSV (Theme D-ish parsing).*
- **Whitespace-and-indent block canonicalizer (mixed tabs/spaces)** (untagged) — tab=2 spaces, snap
  to even multiple preserving depth, strip trailing ws, collapse blank lines, exact join sentinel.
  Output: joined multi-line string.
- **Glyph-Ledger Field Extraction** (untagged / first-principles) — sigil-tagged receipt; sigil->
  field map, vendor uppercase, amount = int/100, ref trailing-numeric, status synonym->enum. Output:
  pipe-joined fields.
- **Nested-Bracket Path Resolver** (untagged) — mixed bracket styles denote node kinds; some kinds
  depth-transparent; sibling index origin; whitespace significance. Output: node label.
  *Parsing sibling of the rewrite/tree themes.*
- **Priority-Conflict Record Reconciler** (untagged) — conflicting source-tagged records; per-field
  source-priority, missing-value semantics, tie-break. Output: pipe-joined fields. One-shot risk LOW
  (per-field tables are arbitrary). *Overlaps Priority Vote Aggregator / Priority Ledger.*
- **Unit-and-Sign Normalizer** (untagged) — mixed conventions (F->C, dB->linear, hex->dec) at fixed
  precision. Output: pipe-joined numerics. *Overlaps Theme A conversions.*
- **Positional Fixed-Width Decoder** (untagged) — no-delimiter record; column widths, implied
  decimal, trim/pad, field ordering. Output: pipe-joined fields. Cascade caveat noted (use per-field
  scoring or 2-3 columns).
- **Escaped-Delimiter CSV Under Custom Quoting** (untagged) — 2-char delimiter, backslash escaping,
  escaped-delimiter literal, trim policy; return Nth field. Output: field string.
- **Enum-Canonicalization With Hierarchy** (untagged) — free text -> dotted 2-level code; synonym
  clusters, hierarchy, most-specific-wins, fallback. Output: dotted enum code. One-shot risk LOW.
  *Overlaps Cipher-Category Tagger / Contrarian Sentiment codebook structure.*
- **Interval-Overlap Field Selection** (untagged) — tagged intervals + point query; endpoint
  inclusivity, overlap tie-break, format normalization, out-of-range fallback. Output: tag token.
- **Reference-Chained Value Deref** (untagged) — key-value store with `@k` pointers; transitive
  resolution, cycle->sentinel, undefined->default, max-depth cap. Output: number or sentinel.
- **Multi-Schema Router Extraction** (untagged) — header token routes among dialects (JSON-ish / KV /
  positional); extract field b regardless. Output: field value. One-shot risk LOW (per-dialect rules).
- **Signed-Checksum Field Validator** (untagged) — custom check-digit; segmentation, letter->value
  map, weights, modulus, valid threshold. Output: pipe fields + VALID/INVALID. Checksum resists
  demo-only learning — good for differentiating instruction-carrying optimizers. *Overlaps Protocol
  Bitmask.*
- **Invented-rule sorting / canonical ordering of symbolic tokens** (untagged) — composite sort key
  (primary + tie-breaks + vowel/consonant reranking + stability). Output: ordered list or token.
  *Overlaps MultiRuleSort / CaseOrder.*
- **Symbolic unit / dimension algebra under invented conversion rules** (untagged) — numeric
  conversion + symbolic unit-simplification rewrites with priority, canonical-range normalization,
  formatting. Output: `number unit` string. *Bridges Theme A (units) and Theme D (rewriting).*
- **CaseOrder** (`deepseek-v4-flash`) — sort words by length, reverse each, title-case, comma-join.
  Output: single line. *Overlaps Invented-rule sorting / MultiRuleSort.*

## Theme H — Instruction following / constrained generation (IFEval-style)

Simultaneous constraints or an authority policy over directives. Note: some require a
deterministic constraint-checker or canonical reference rather than pure free-gen scoring.

- **Constrained Micro-Description** (`minimax-m3`) — 4-7 word description satisfying 5-7 hidden
  per-instance constraints (required/forbidden word, exact count, first-letter, allowed alphabet,
  casing, digit-at-position, punctuation). Output: 4-7 word phrase; needs exact-match reference or
  a per-constraint checker. Tie-breaker: IFEval / FollowBench.
- **Priority Ledger** (`gpt-5.x-codex`) — several tagged directives (quoted/negated/conditional/
  conflicting); identify the controlling directive under a seeded authority policy (tag-priority,
  quoted-text-inert, recency, negation, conditional activation, exception priority, fallback).
  Output: A-F. Tie-breaker: instruction-hierarchy research. *Overlaps Priority Vote Aggregator /
  Priority-Conflict Reconciler.*

---

## Cross-cutting observations

- **Most robust themes** (appear repeatedly across independent lenses): rewrite-to-normal-form
  (Theme D, 3-4 instances), deterministic simulation/state machines (Theme E), invented-notation
  arithmetic (Theme A), remapped-label classification (Theme F), and canonicalization/extraction
  (Theme G, the largest cluster and the closest analog to the project's own dr-serialize work).
- **Priority/authority reconciliation** recurs across Themes F/G/H (Priority Vote Aggregator,
  Priority-Conflict Record Reconciler, Priority Ledger, Traffic-Light Classifier) — a strong
  independent-rules-with-precedence pattern.
- **Best anti-one-shot designs** (rules must be induced from demos, so MIPROv2's demo axis is
  load-bearing): abstract rewrite systems, invented gate/transition tables (Marble-Track, Glyph
  Automaton), per-dialect routers, per-field priority tables, arbitrary checksums.
- **Weakest on incrementality / saturate fast** (better as floor/plumbing/format validators):
  Inverted Multiple-Choice, Modular Digit Oracle, Invented congruence YES/NO.
- **Temp-0 resolvability caveats** (char-arithmetic noise; validate criterion 5 before adopting):
  Cipher-With-Invented-Rules Decoder, Turing-Tape Rewrite, and cipher chains generally — prefer
  closed-vocabulary or output-length variants.
- **Vestigial second metric** (rubric callout): output length in characters as the compression
  analog, with a length-budget knob as the variance source, applies cheaply to most text-output
  tasks here (Themes A/D/E/G especially).
