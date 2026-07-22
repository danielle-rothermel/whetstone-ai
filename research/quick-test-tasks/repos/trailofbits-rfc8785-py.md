# Repo Review: trailofbits/rfc8785.py

- **URL:** https://github.com/trailofbits/rfc8785.py
- **Review date:** undefined
- **License:** Apache-2.0 (`LICENSE`, confirmed header line 1-3)
- **Serving candidate:** c11 House-Convention Canonicalizer
- **One-line verdict:** A clean, reliable RFC 8785 JSON canonicalizer that gives us a ready-made deterministic *oracle* for a canonical-JSON task, but ships **zero** generator/sampler/seed code — we must build the entire instance-generation half ourselves, so it is a reusable oracle component, not a drop-in re-seedable benchmark.

## Active path

The repo has no generator entry point. The only "active path" relevant to us is the
canonicalization oracle:

- `src/rfc8785/__init__.py:18` re-exports `dump`, `dumps` and the error types.
- `src/rfc8785/_impl.py:177` `dumps(obj) -> bytes` wraps `dump(obj, sink)` (`_impl.py:188`) over a `BytesIO`.
- `dump` dispatches by type: null/bool/int (`_impl.py:193-205`), str via `_serialize_str` (`_impl.py:79`), float via `_serialize_float` (`_impl.py:97`), list/tuple (`_impl.py:214`), dict with UTF-16BE key sort (`_impl.py:227-252`).

**Active-path LOC:** ~200 (all of `_impl.py`, single file; `__init__.py` is ~10 loc of re-export).

## Checklist findings

### Seed plumbing
**Absent — there is no generation code at all.** `grep` for `random`, `numpy`, `seed`
returns nothing in `src/`. The library takes a caller-provided Python object and emits
canonical bytes deterministically. Determinism of the *oracle* is total (pure function of
input; dict order normalized by `sorted(...key=utf-16be)` at `_impl.py:238`). But there is
no seed to thread because there is nothing to sample. We supply 100% of the seedable
generator ourselves.

### Oracle independence
**Strong, if we generate inputs by an independent path.** `dumps` computes the canonical
form directly from the RFC 8785 rules, not by remembering how an input was built. As long
as OUR generator produces ordinary Python dicts/lists/scalars (e.g. shuffled key order,
mixed number forms) and we call `rfc8785.dumps` to get ground truth, the oracle is logically
independent of the generation procedure. No tautology risk from the repo itself.

Caveat: the repo implements *public* RFC 8785, not our "5-6 invented house rules." For c11
we need deviations from the standard. We would either (a) post-process `dumps` output, or
(b) fork the ~200 loc and edit the specific rules (escape table `_impl.py:31-41`, key sort
`_impl.py:238`, float formatting `_impl.py:97-174`, int domain `_impl.py:23`). That is real
logic surgery inside the oracle, not just config glue.

### Tests
`test/test_impl.py` (175 loc) and `test/test_init.py` plus vector assets in
`test/assets/{input,output,outhex}` (6 named vectors: arrays, french, structures, unicode,
values, weird). Tests are runnable pytest: float stringification table (`test_impl.py:17-60`),
integer domain (`:85`), invalid UTF-8 (`:95`), enum handling (`:108-169`), non-string keys
(`:172`). A 100M-case ES6 float file is optional and skipped if absent (`conftest.py:24`,
`test_impl.py:64-65`). Tests cover the oracle well; **nothing tests a generator** because
none exists. CI workflows present (`tests.yml`, `float-tests.yml`, `lint.yml`).

### Global state / hidden coupling / dead code
- Module-level `_ESCAPE_DCT` built at import via a loop (`_impl.py:40-41`): read-only after
  construction, not a determinism hazard.
- No global mutable state, no set/dict-iteration nondeterminism on the active path (dict
  iteration is explicitly re-sorted at `_impl.py:238`).
- No dead code on the path; the file is tight and single-purpose.
- `# TODO: Optimize this?` (`_impl.py:182`) is cosmetic.

## Adaptation-diff sketch

Because the repo is oracle-only, the diff is mostly **new external code we write**, not edits
to the repo:

1. **New file `gen.py` (outside repo, ~120-180 loc):** seedable instance generator. A
   `numpy.random.Generator` / `random.Random(seed)` threaded explicitly, producing nested
   Python structures with latent-rule strata (key-order shuffles, number-form variants,
   unicode/escape edge cases, invented nonce vocab for keys/values). This is the entire
   sampling half and does not exist yet.
2. **House-rule layer (~40-100 loc):** either post-process `rfc8785.dumps()` output to apply
   5-6 invented deviations, OR vendor `_impl.py` (~200 loc) and edit `_ESCAPE_DCT` (`:31-41`),
   the key-sort comparator (`:238`), float format (`:97-174`), int bounds (`:23`). Post-
   processing is cleaner but some rules (e.g. altered sort order) force a fork.
3. **Scoring glue (~30 loc):** single LLM call, `0/1` exact whole-string match of model output
   vs `house_canonicalize(instance)`.
4. **Config/strata wiring (~20 loc).**

Estimated total new code: **~220-350 loc**, of which essentially none is a modification of the
existing repo (we either import `dumps` unchanged or fork 200 loc). This is buildable from what
I read, but the generator + strata are entirely on us.

## Red flags

- **No generator, no sampler, no seed anywhere in the repo** — the re-seedable instance
  generation half of the requirement is 100% missing. This is the dominant flag.
- **Implements the public standard, not "invented house rules"** — c11 explicitly wants
  deviations from public standards; delivering that requires either post-processing or forking
  and editing the oracle's rule logic.
- Minor: float serialization is adapted from a third-party reference impl (`_impl.py:26-29`,
  `:119-120`); well-tested here but worth knowing if we fork it.
