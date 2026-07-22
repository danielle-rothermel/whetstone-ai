# Repo Review: belindal/state-tracking

- **URL:** https://github.com/belindal/state-tracking
- **Review date:** undefined
- **License:** MIT (`LICENSE`, Copyright (c) 2025 Belinda Zou Li) — permissive.
- **Serving candidate ids:** c05 (Deterministic Execution & State Tracking — FSM/protocol/pointer)
- **One-line verdict:** Clean, tiny, self-consistent permutation-composition generator + exact oracle; reusable as config+scoring glue, but seed is not plumbed (trivial to inject). Provisional I1 = 2.

## Active path

Entry point for our use is the generator only (`permutation_task.py`); training/eval/interpret code is off-path for a single-LLM-call benchmark.

- **Generator entry:** `permutation_task.py:226-256` (`__main__`) -> `PermutationTask.simulate` (`permutation_task.py:137-198`).
- **Sampling:** `simulate` loop `:167-183` calls `choose_random_action` (`:100-108` -> `random.choice`) then `update_state` (`:128-135`).
- **Ground-truth computation:** `PermutationState.apply_action` (`:30-42`): `new_perm = tuple(self.permutation[action[j]-1] for j in range(len(self.permutation)))`. State sequence recorded per step at `:175`.
- **Serialization:** `:190-196` writes `{"story": <space-joined action strings>, "state_seq": [perm tuples]}` to `story_{idx}.json` under `train/` or `test/`.
- **Parity helper (optional oracle stratum):** `compute_parity` `:201-223` (inversion count mod 2), independent of generation.

Active-path LOC: ~180 (the whole of `permutation_task.py`; alternative topic-model generator `make_topic_training_data.py` is ~125 LOC and off our path unless we want nonce/latent-topic vocab).

## Checklist findings

- **Active-path LOC:** ~180 (single file `permutation_task.py`).
- **Seed plumbing (RED FLAG):** No `random.seed`/`np.random.seed` anywhere in `permutation_task.py`; no `--seed` CLI arg (`:227-233`). Random call sites are module-global `random.choice` in `choose_random_action` (`:108`) and unseeded `np.random.random()` for the train/test split (`:186`). Consequence: default runs are NOT reproducible. Verified externally that a single `random.seed(s)`+`np.random.seed(s)` before `simulate` fully determinizes output (standalone reimplementation of `apply_action` + sampling produced byte-identical rollouts across two seeded runs). Seed is threadable but currently ABSENT.
- **Oracle independence:** Ground truth is computed by forward simulation (`apply_action`, `:41`) as the story is built — it is the same deterministic transition the model must predict, computed by plain index composition, not by inverting/tautological lookup. `compute_parity` (`:201`) is a fully independent second oracle. Verified composition by hand: applying `(2,3,1)` to identity `(1,2,3)` -> `(2,3,1)` -> `(3,1,2)` (correct group action). Oracle is trustworthy.
- **Tests covering generator/oracle:** NONE. `test_dataset.py` only exercises HF tokenization of an already-generated dataset (needs `transformers`+`gpt2` download); it does not test the generator or oracle. No pytest suite. Generator correctness is unverified by repo tests (I verified it manually instead).
- **Global state / hidden coupling / dead code:** `PermutationTask` carries mutable `current_state` (`reset`/`update_state`) — stateful object, fine single-threaded, unsafe for parallel reuse. `_init_actions` == `_init_states` (all N! permutations are actions). `get_valid_actions` and the `state_to_nl`/`nl_to_state` dicts (`:60-64`) are built but unused by `simulate` (used by `eval.py`/topic model) — mild dead weight on our path, not harmful. Dict/set usage is deterministic (insertion-ordered; `stories_set` only dedupes). No import-time global mutation.
- **Serialization gotcha:** duplicate-story skip (`:178-179`) means fewer files than `num_stories` and non-contiguous `story_idx`; only relevant at tiny state-space / long dedup pressure. `num_items` limited to {3,5} (`:228`). README arg names (`--num_items`, `--data_dir`, `--story_length`) use underscores but argparse defines hyphenated `--num-items` etc. (`:228-232`) — README examples will error as written (minor).

## Adaptation-diff sketch

We do NOT need training/eval/interpret. Reuse `permutation_task.py` as a library.

- **New glue file (outside repo), ~60-90 LOC:** import `PermutationTask`, `PermutationState`, `compute_parity`. For each instance: `random.seed(seed)`/`np.random.seed(seed)`, build task, roll out K actions via `choose_random_action`/`update_state`, capture final state (and/or parity) as the exact 0/1 oracle. Render prompt (action string sequence) + question ("final permutation?"), score with exact string match against `current_state.to_string()`. Single LLM call per instance handled by our harness.
- **Latent-rule strata:** vary `num_items` (S3 vs S5), rollout length (trace-length strata), and parity-vs-full-state query type — all available with zero repo edits.
- **Invented/nonce vocab:** map the digit action/state alphabet through a per-instance nonce token table in our glue (repo emits digits; substitution is a dict in our code). Optionally reuse `make_topic_training_data.py`'s vocab-mapping idea, but simpler to do in glue.
- **Repo edits proper:** effectively 0 required if we call `PermutationTask` as a library and seed before use. If we prefer in-repo determinism: add `--seed` arg + `random.seed`/`np.random.seed` at `simulate` entry (~4 lines at `:154`/`__main__`).

Total new code: one ~60-90 LOC glue module + config. No modification to the oracle. This meets "our diff is config + scoring glue."

## Red flags (named)

1. **Unthreaded/absent seed** — no seed param or `.seed()` call on the active path; default output non-reproducible until we inject a seed externally.
2. **No tests for generator/oracle** — only a tokenization smoke test; correctness rests on our manual verification (done, passed).
3. **Stateful mutable task object** — not reentrant/parallel-safe; instantiate fresh per instance.
4. **README/argparse arg-name mismatch** (`--num_items` vs `--num-items`) — cosmetic but examples fail as written.
5. **Dedup drops instances** — `story_idx` non-contiguous, count < requested at small state spaces.

## Why I1 = 2 (not 3)

Maintained/permissive/2025-active/well-written/oracle-reliable all hold, and our diff is genuinely config+glue. It falls short of 3 on ONE anchor: seed is not plumbed on the active path (named red flag), so "works reproducibly as claimed" requires us to add seeding rather than it being provided. Genuinely reusable -> 2.
