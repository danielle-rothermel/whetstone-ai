# Repo Review: open-thought/reasoning-gym

- URL: https://github.com/open-thought/reasoning-gym
- Review date: 2026-07-21
- License: Apache-2.0 (`LICENSE` present; `NOTICE.txt` present) — permissive, no obstacle
- Serving candidate id: c10 Interlocking-Cycle Calendar & Clock Arithmetic
- One-line verdict: Clean, deterministic, well-tested seeded-generator infra with two calendar/clock tasks (`calendar_arithmetic`, `time_intervals`), but both are strictly REAL-GREGORIAN via Python `datetime`/`calendar` — the invented interlocking-cycle / coprime-modulus semantics of c10 are NOT in-repo and must be authored, so this is reuse-infra-write-task, not config+glue.

## Provisional I1: 2

Maintained, Apache-2.0, actively used (NeurIPS 2025 Spotlight, PyPI, 2026-04-17 last push, 1463 stars), active-path code is well-written and verified deterministic per-seed with a genuine oracle. Held to 2 (not 3): neither shipped calendar/clock task implements invented calendars, coprime interlocking cycles, or seeded conventions. c10 needs a NEW `ProceduralDataset` subclass (modular cycle arithmetic + oracle + strict 0/1 scoring). The infra (`ProceduralDataset`, `Random(seed+idx)` pattern, factory, curriculum) is reused as-is; the task logic is net-new. That exceeds "config + scoring glue."

## Active path

Entry -> sampling -> oracle -> serialization, read end to end for both closest tasks:

- `reasoning_gym/factory.py:51` `create_dataset(name, **kwargs)` -> builds config dataclass, calls `config.validate()` (`factory.py:70`), instantiates dataset.
- `reasoning_gym/dataset.py:10` `ProceduralDataset` base: `__init__` (`dataset.py:13`) sets `self.seed = seed if not None else Random().randint(0, 2**32)` (`dataset.py:20`); `__getitem__` abstract (`dataset.py:49`); default `score_answer` exact+substring partial credit (`dataset.py:63-72`).
- calendar_arithmetic: `reasoning_gym/arithmetic/calendar_arithmetic.py` — `__getitem__` (`:128`) does `rng = random.Random(self.seed + idx)` (`:129`), `rng.choice(self.tasks)` (`:130`) dispatches to 7 handlers (`:116-124`). Each handler samples dates and computes ground truth via Python `datetime`/`calendar` (e.g. `_weekday_offset` `:144-181`, `target_date = start_date + timedelta(days=offset_days)`; `target_date.strftime("%A")`). Serialization dict `:138-142`.
- time_intervals: `reasoning_gym/arithmetic/time_intervals.py` — `__getitem__` (`:86`) `rng = random.Random(self.seed + idx)` (`:88`), `_generate_times` (`:153`) samples real datetimes, difference computed via `end_dt - start_dt` (`:110`) then formatted (`:113-133`).
- Registration at import bottom of each file: `register_dataset(...)` (`calendar_arithmetic.py:531`, `time_intervals.py:353`) populating module-global `DATASETS`/`CURRICULA` (`factory.py:14-15`).

Active-path LOC (approx): ~148 (`dataset.py`) + ~119 (`factory.py`) + 532 (`calendar_arithmetic.py`) OR 354 (`time_intervals.py`). One calendar candidate task exercises roughly 300-680 LOC end to end.

## Checklist findings

### Seed plumbing: THREADED, clean
- Seed enters via config dataclass field `seed` (`calendar_arithmetic.py:61`, `time_intervals.py:28`), passed to base `__init__` (`calendar_arithmetic.py:114` / `time_intervals.py:84` -> `dataset.py:20`).
- Every random call site uses a fresh per-item generator: `rng = random.Random(self.seed + idx)` (`calendar_arithmetic.py:129`, `time_intervals.py:88`). All `rng.randint/choice/sample` calls take this local `rng`; helper methods `_random_date_for_year` (`calendar_arithmetic.py:421`) and `_random_date_between` (`:428`) receive `rng` as an argument. `Weekday.random(rng)` (`:30`) also threaded. No hidden global `random` consumption on the path.
- Reseeding wrapper `ReseedingDataset._create_chunk` derives `new_config.seed = (self.dataset.seed + chunk_num) % 2**32` (`dataset.py:112-121`) — deterministic and re-seedable.
- Verified live: same seed -> byte-identical items; different seed -> different items (both tasks).

No named red flags on the calendar/clock path (no module-global `random.seed`, no `np.random`, no set/dict-iteration sampling). `grep` for `np.random` in `reasoning_gym/arithmetic/` returns nothing.

### Oracle independence: INDEPENDENT of the LLM, but shares stdlib routine with the prompt
- calendar_arithmetic ground truth is derived from the sampled date via Python `datetime`/`calendar` (`_weekday_offset` `:157-158`, `_count_days` `:322-326`, `_is_leap_year` uses `calendar.isleap` `:406`). The answer is computed from ground-truth dates, not by inverting model output. Genuine oracle.
- Caveat: for weekday tasks the prompt's shown weekday and the answer both come from `strftime("%A")` on the same `datetime` — a bug in stdlib would corrupt both symmetrically (standard for generator+oracle-in-one-file). Not a second independent implementation. Hand-check below re-derived independently and matched.
- time_intervals oracle uses `end_dt - start_dt` timedelta (`:110`), independent of the string formatting shown; but its scoring is lossy (see below).
- Hand-check (3 `weekday_offset` instances, independently recomputed `start_date + timedelta(offset)` then `strftime("%A")`): all matched `answer`, `score_answer==1.0`.

### Scoring: PARTIAL-CREDIT by default — MISMATCH with our 0/1 requirement
- `CalendarArithmeticDataset.score_answer` (`calendar_arithmetic.py:439-494`) awards graded credit: weekday tasks give 0.1 for any valid-but-wrong weekday (`:459-460`), 0.05 for title-cased (`:462-463`); numeric tasks return `exp(-5 * relative_error)` continuous reward (`:483-484`). Only `is_leap_year` is truly 0/1 (`:489-492`). Verified in test at `test_calendar_arithmetic.py:119-134`.
- `TimeIntervalsDataset.score_answer` (`time_intervals.py:252-328`) is also continuous: `1.0 - diff/max_diff` partial credit.
- Base default `score_answer` (`dataset.py:63-72`) also gives substring partial credit.
- For c10's strict single-token 0/1 exact match we MUST override `score_answer` in our subclass (~5-10 LOC). Do not reuse these.

### Tests: PRESENT and RUNNABLE (verified)
- `tests/test_calendar_arithmetic.py` (218 lines, 9 tests): config validation (`:44`), determinism `ds1[i]==ds2[i]` (`:51-58`), item structure, per-task correctness, graded scoring assertions (`:119-134`).
- `tests/test_time_intervals.py` (131 lines, 6 tests).
- Ran in a fresh `uv` venv (`uv pip install -e .` + pytest): all 15 tests across both files PASS.

### Global state / hidden coupling / dead code
- Module-global registry dicts `DATASETS`, `CURRICULA` (`factory.py:14-15`) populated at import; keyed lookups only, no iteration-order dependence on our path; double-register raises (`factory.py:37`).
- `Weekday.__getitem__` is defined as a `@classmethod` (`calendar_arithmetic.py:33-35`) which is an unusual/likely-buggy override of the Enum metaclass indexer, but it is not exercised on the active path (code uses `Weekday(index+1)` and `list(cls)[...]`). Off-path oddity, note only.
- No dead code on the active path. No corpus-file dependency for these two tasks (unlike word_sorting).
- Import emits harmless third-party `cellpylib` `SyntaxWarning: "is" with str literal`; off our path.

### There is NO invented / interlocking-cycle calendar or clock
- Both tasks are strictly REAL-GREGORIAN: they call Python `datetime`, `date`, `timedelta`, `calendar.monthrange`, `calendar.isleap`, `strftime` (`calendar_arithmetic.py:1-6,157,406,424`; `time_intervals.py:3,110`). Fixed 7-day week (`Weekday` enum `:16-23`), 12 real months, real leap rules.
- `grep -rin "coprime|interlock|modulus|invented"` finds only unrelated hits (fraction GCD, corpus wordlists). No coprime cycle lengths, no invented calendar conventions, no per-instance seeded rule-set. The c10 latent-rule strata (cycle lengths, offset conventions) do not exist as config.

## Adaptation-diff sketch

We do NOT edit repo files; consume reasoning-gym as a PyPI/vendored dependency and add glue in OUR tree.

- c10 core (new generator, ~120-200 LOC new): subclass `ProceduralDataset` in our repo. Config carries invented cycle lengths (sample pairwise-coprime cycle lengths for interlocking calendar+clock, e.g. week-length W, month-length M, hour-cycle H with gcd==1), an offset/naming convention, and the latent-rule stratum id. `__getitem__` copies the `rng = Random(seed+idx)` pattern (`calendar_arithmetic.py:129`); sample a start position and an offset; ORACLE = pure modular arithmetic over the coprime cycles (`(start + offset) % cycle`, CRT reconstruction) — a fresh independent implementation, NOT stdlib `datetime`. Render the single-token answer under the seeded convention.
- Strict scoring (~5-10 LOC): override `score_answer` for 0/1 exact match; bypass the lenient calendar/time/base scorers.
- Optional curriculum (~20-40 LOC): a `BaseCurriculum` subclass exposing cycle-length ranges / convention families as strata (mirrors `CalendarArithmeticCurriculum` `:497-528`).
- Shared glue (outside repo, ~30-50 LOC): thin wrapper mapping our config -> instantiation, single LLM call, 0/1 exact-match scoring.

Estimated new code: ~180-300 LOC in our tree; 0 lines changed in reasoning-gym. We reuse the infra (`ProceduralDataset`, seed plumbing, factory/registration, curriculum scaffolding) and the `Random(seed+idx)` idiom, but AUTHOR the interlocking-cycle generator + oracle from scratch. This is why I1=2 not 3.

## Red flags summary
- No invented / interlocking-cycle / coprime-modulus calendar-clock task exists; both shipped tasks are real-Gregorian via stdlib `datetime`/`calendar`. c10 semantics are net-new authoring. (Primary reason for I1=2.)
- Default `score_answer` on both calendar_arithmetic (`:439-494`) and time_intervals (`:252-328`) is PARTIAL/continuous credit, not 0/1; base default gives substring credit (`dataset.py:70-71`). Must override for strict exact match.
- Calendar weekday oracle shares stdlib `strftime` with the shown prompt weekday (single implementation, no independent cross-check) — standard but noted.
- `Weekday.__getitem__` classmethod override (`:33-35`) is off-path but suspicious; do not rely on `Weekday[i]` indexing.
- time_intervals scoring is lossy (`1.0 - diff/max_diff`) and parses formatted strings — do not reuse as an exact oracle.

## Run verification (2026-07-21)

Environment (lightweight; deps sympy, pytz, python-dateutil, etc.):

```
cd /tmp/qt-repo-review/reasoning-gym
uv venv --python 3.12 .venv
VIRTUAL_ENV=$PWD/.venv uv pip install -e .    # reasoning-gym dev build installed OK
```

### Determinism + reseed (SHA-256 over first 5 items)

`create_dataset(name, seed=..., size=5)`; seed A=42 twice, seed B=123 once:

```
calendar_arithmetic  A c07837e7847a  A2 c07837e7847a  identical=True  B a746aafe0279  differs=True
time_intervals       A 24a0266bfdf4  A2 24a0266bfdf4  identical=True  B 1528477b0e6e  differs=True
```

Same seed -> byte-identical (deterministic). Different seed -> different (re-seeds). PASS both.

### Ground-truth hand-check (3 weekday_offset instances, independent oracle)

Independently recomputed `date.fromisoformat(start) + timedelta(offset_days)` then `strftime("%A")`; matched `answer` and `score_answer==1.0`:

```
[0] 2022-03-13  -84 -> Sunday    oracle=Sunday    OK  score=1.0
[1] 2022-06-13  -17 -> Friday    oracle=Friday    OK  score=1.0
[2] 2022-10-12  -35 -> Wednesday oracle=Wednesday OK  score=1.0
```

### Tests
`pytest tests/test_calendar_arithmetic.py tests/test_time_intervals.py` -> 15 passed.

Verdict: infra RAN, is deterministic per-seed, re-seeds on change, and Gregorian ground truth is correct — but the interlocking-cycle / invented-calendar task c10 demands is not shipped and must be written on top of the (reusable) infra.
