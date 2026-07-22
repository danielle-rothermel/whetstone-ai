# Repo Review: json-schema-faker/json-schema-faker

- **URL:** https://github.com/json-schema-faker/json-schema-faker
- **Review date:** 2026-07-21
- **License:** MIT (LICENSE file present; `package.json` `"license": "MIT"`). Copyright 2024 Alvaro Cabrera.
- **Serving candidate:** c11 House-Convention Canonicalizer
- **Version reviewed:** v0.6.2 (TypeScript/Bun rewrite), cloned `--depth 1` from `main`.
- **One-line verdict:** Excellent, cleanly seedable deterministic JSON *instance generator* with a runnable test suite — but it is generation-only with NO canonicalizer/oracle, so for c11 the actual task substance (house rules + ground truth) is new code we author, not config glue. Provisional 2.

---

## Active path

Entry -> sampling -> serialization (there is no ground-truth/oracle stage in this repo):

- `src/index.ts:23` `generate(schema, options)` — creates PRNG, ref registry, format registry, builds `GenerateContext`, calls `walk`.
- `src/index.ts:129` `generateJson(...)` — wraps `generate` + `JSON.stringify` (this is the serialization we'd use).
- `src/random.ts:1-51` — `mulberry32` PRNG + `createRandom(seed)` instance (`next/int/bool/pick/shuffle/fork`).
- `src/schema-walker.ts:202` `walk` / `:243` `walkSchemaBody` — the type dispatch engine; every branch draws through `ctx.random`.
- `src/generators/*.ts` — per-type samplers: `object.ts` (635 LOC, the heaviest), `array.ts`, `string.ts`, `number.ts`, `integer.ts`, `boolean.ts`, `null.ts`, `enum-const.ts`, `composition.ts`.
- Supporting: `ref-resolver.ts`, `merge.ts`, `formats/*`, `pattern/regex-gen.ts`, `utils/*`.

**Active-path LOC (approx):** ~1,000 LOC for the core spine (`index` 269 + `random` 51 + `schema-walker` 459 + `types` 227) plus generators — `object.ts` alone is 635. A realistic active path for simple object/array/string/number schemas is ~1,500-2,500 LOC. The engine is broad because it implements a large slice of JSON Schema 2020-12; a benchmark using a narrow schema subset would exercise a fraction of it.

## Seed plumbing (THREADED — clean)

- Seed enters at `src/index.ts:36`: `const random = createRandom(options?.seed ?? 1)`.
- Stored on the context: `src/index.ts:53` `random,` inside `ctx`, then passed to `walk(schema_, ctx)` at `:88`.
- Every random draw in the walker and generators goes through `ctx.random.*` (e.g. `schema-walker.ts:264,330,391,415`; `object.ts:188,407,433,459`; `string.ts:100,111`). No module-global `Math.random`, no `np.random.seed`-style import-time seeding.
- PRNG is a per-call **instance** (`random.ts:13`), not a shared singleton — concurrent `generate` calls do not interfere.
- `createGenerator`/`createJsonGenerator` (`index.ts:98,137`) advance the seed by an internal `callCount` (`seed + callCount++`) — re-seedable and reproducible across a batch.
- Verified empirically: same seed -> byte-identical JSON; different seed -> different JSON (see Tests).

**Grep for nondeterminism red flags:** no `Math.random`, no `crypto`/`randomUUID` in `src/`. Object/property emission order follows `Object.entries(schema.properties)` (insertion order, deterministic in V8). No set/dict-iteration nondeterminism found on the output path.

## Oracle independence — THE KEY GAP

- **There is no oracle in this repo.** It is a one-directional generator: schema -> random valid instance. The generated instance *is* the output; there is nothing that independently computes a "correct answer" to check against.
- For c11 (a *canonicalizer* — normalize a messy input to a single canonical string under invented house rules), the ground truth is the canonical form. That logic does not exist here and cannot be tautologically borrowed from the generator.
- Consequence: jsf can serve as the **input generator** (produce diverse, re-seedable pre-canonicalization JSON documents / strata). The **canonicalizer + oracle** — the 5-6 invented house rules applied deterministically to produce the exact ground-truth string — is entirely new code we write outside the repo. That new code is the substance of the task, not "scoring glue."

## Tests (runnable, green)

- `bun test` -> **519 pass / 0 fail**, 2006 assertions, ~870ms. Runs out of the box after `bun install` (Bun 1.3.14 present).
- `tests/unit/random.test.ts` explicitly asserts determinism: same seed -> identical 100-length sequence (`:5-11`); different seeds diverge (`:13-19`); `int` range, `pick`, `shuffle` permutation, `fork` independence.
- Large integration corpus: `tests/integration/*` + `tests/schema-faker-tests/**` (hundreds of `issue-*.json` fixtures) validate generated output against schemas (via `ajv` in `tests/helpers/validate.ts`).
- Empirical end-to-end check (my scratch script, since deleted): `generateJson(schema,{seed:123})` twice -> identical; `seed:124` -> different; bounds (`maxItems`, `minLength/maxLength`) respected.

## Global state / hidden coupling / dead code

- `src/index.ts:15` `globalFormatRegistry` — module-global, mutated by `registerFormat` (`:110`). Only matters if we register custom named formats globally; per-call `options.formats` (`index.ts:41`) bypasses it. Not a determinism risk for our use unless we opt in.
- `src/extensions.ts` — global extension registry via `define`/`reset` (`index.ts:117,121`). Same caveat: only engaged if we use `jsf.define`.
- `src/browser.ts:10` `let _options` (module-global) and `:27` `seed ?? Date.now()` — **browser playground only**, not on the library `generate()` path. Avoid `browser.ts`.
- `src/generators/string.ts:107` `maxDt = ctx.maxDateTime ? ... : new Date()` — **date/date-time format generation uses wall-clock time as the upper bound when `maxDateTime` is unset.** This is a genuine nondeterminism source for `format: "date"`/`"date-time"` fields. Mitigation is trivial: always pass `minDateTime`+`maxDateTime`, or avoid date formats in benchmark schemas.
- `pattern`-based string generation (`pattern/regex-gen.ts`) and `patternProperties` key synthesis (`object.ts:524`) are heuristic and can fail to match complex regexes — a correctness sharp edge for exotic schemas, not for the simple schemas a canonicalizer benchmark would use.

## Adaptation-diff sketch

What we reuse unchanged:
- `generate` / `generateJson` (`src/index.ts`) as the **seeded input generator**. Pass explicit `seed` + `optionalsProbability`/`alwaysFakeOptionals` to control strata; avoid date formats (or pin `minDateTime`/`maxDateTime`) to sidestep the one wall-clock issue.

What we write NEW (outside the repo, ~200-350 LOC):
1. **Schema/config per latent-rule stratum** (~30-60 LOC JSON): schemas whose instances exercise each house rule (key casing, ordering, whitespace, numeric formatting, slug rules, nonce vocab, etc.).
2. **The canonicalizer + oracle** (~150-250 LOC, the core): implement the 5-6 invented house rules as a deterministic normalization pipeline over the generated JSON, emitting the exact canonical string. This is BOTH the reference the model must reproduce AND the ground truth.
3. **Invented/nonce vocab injection** (~20-40 LOC): if rules key off nonce tokens, either add them via schema `enum`/`const` or a custom `options.formats` generator.
4. **Scoring glue** (~30-50 LOC): single LLM call, whole-string exact match (0/1) vs. the oracle string, strata bookkeeping.

Files changed **inside the repo:** effectively none required (use it as an npm/Bun dependency). If we vendor it, zero edits to `src/` are needed for the generation half.

## Red flags

- **NO oracle / generation-only.** The task's ground-truth (canonical output) is not present and must be authored; our diff is therefore *task logic + glue*, not *config + glue*. This is why it is not a 3.
- **Wall-clock nondeterminism in date formats** (`string.ts:107`): must pin `maxDateTime` or avoid date formats for reproducibility.
- **Module-global format/extension registries** (`index.ts:15`, `extensions.ts`): benign unless used; avoid global `registerFormat`/`define` across parallel runs.
- **Heuristic regex/pattern generation** can silently produce non-matching or degenerate strings for complex patterns — keep benchmark schemas simple.
- `browser.ts` seeds from `Date.now()` — do not use that entry point.

## Fit summary vs. provisional-I1 anchors

- Maintained: yes (v0.6.2, 2026). License: yes (MIT). Actively used: yes.
- "Our diff is config + scoring glue": **NO** — the canonicalizer/oracle is substantial new logic. Active-path code: well-written, typed, deterministic, tested; likely to work as claimed (verified).
- One anchor-3 criterion fails (diff is not just config/glue; repo supplies only the input-generator half). -> **Provisional I1 = 2** (genuinely reusable as the re-seedable input generator, but the oracle must be built).

---

## Run verification (2026-07-21)

**Verdict: CONFIRMED runnable and re-seedable.** The generator works out of the box, is byte-deterministic on a fixed seed, varies on a new seed, and produces schema-valid instances. No dependency fight — `node_modules/` already present in the clone, `bun` 1.3.14 on PATH via mise.

### Environment
- No venv/pip needed — this is a Bun/TypeScript project (not Python). `bun` 1.3.14, `node` and `npm` all resolved via mise.
- `node_modules/` already present in the clone; `bun` runs `src/*.ts` directly (no build step required).

### Public API used
- `generate(schema, options): Promise<unknown>` and `generateJson(schema, options): Promise<string>` from `src/index.ts` (both `async`).
- Seed passed as `options.seed` (`src/index.ts:36` `createRandom(options?.seed ?? 1)`). Used `alwaysFakeOptionals: true` so optional props are always emitted for a stable comparison.

### Scratch script (created at repo root, run, then deleted)
`bun run verify_run.ts` with a hand-checkable object schema (integer/string/array/number/boolean/enum with explicit bounds). Exact script removed after the run; the load-bearing calls were:
```ts
const outA1 = await generateJson(schema, { seed: 42,   alwaysFakeOptionals: true });
const outA2 = await generateJson(schema, { seed: 42,   alwaysFakeOptionals: true });
const outB1 = await generateJson(schema, { seed: 4242, alwaysFakeOptionals: true });
```

### Results
1. **Same seed (42) x2 -> byte-identical:** `A1===A2` = **true**. Both produced `{"id":160,"name":"0PkGq","age":47,"tags":["Dp2UtmFQL","","ZdKbq",""],"score":0.18568900716491044,"active":false,"color":"green"}`.
2. **Different seed (4242) -> differs:** `A1!==B1` = **true**. Seed 4242 -> `{"id":155,"name":"5FPr","age":30,...,"score":0.125...,"active":true,...}`.
3. **Ground-truth hand-check on 3 instances (seeds 42, 4242, 777):** all **PASS**. Every constraint verified programmatically: `id` in [100,200] & integer; `name` length in [3,8]; `age` in [18,65]; `tags` length in [2,4] & all strings; `score` in [0,1]; `active` boolean; `color` in enum; `additionalProperties:false` respected (no extra keys).
   - seed 42:  `id=160, name="0PkGq"(5), age=47, tags=4, score=0.186, active=false, color=green` -> PASS
   - seed 4242: `id=155, name="5FPr"(4), age=30, tags=4, score=0.125, active=true, color=green` -> PASS
   - seed 777:  `id=169, name="lhN"(3), age=36, tags=4, score=0.062, active=false, color=green` -> PASS
   - Note: `tags` array items had no `minLength`, so empty-string items (`""`) appear and are schema-valid — not a bug.

### Corroboration
- Repo's own determinism unit test: `bun test tests/unit/random.test.ts` -> **7 pass / 0 fail**, 403 assertions, ~15ms. Explicitly asserts same-seed identical sequences and different-seed divergence.

### Blockers
- None. Timebox not hit; ran on first attempt.
