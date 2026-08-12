# ED1 calibration staging discipline

Vocabulary: **MDD** — minimum detectable difference. The power analysis
(`whetstone.evaluation.analysis.power.analyze_power`) is closed-form only: it
returns the variance decomposition, the target gap
(`alpha` x certified headroom), and the MDD surface over the
`(n_tasks, num_samples)` grid. There is no simulator, no recommender, and no
automated verdict; reading the surface is a human decision.

## Staging discipline for calibration samples (K_CAL)

1. **Start at K_CAL = 4** samples per task for the calibration anchors.
2. **Inspect the variance components** from the run's
   `VarianceDecomposition`: `within_sample_var` vs. `between_task_var` vs.
   `interaction_var`, and the resulting MDD surface against the target gap.
3. **Double K_CAL only if the components are obviously unstable.**
   "Obviously unstable" is a human reading of the decomposition —
   deliberately not an automated trigger, threshold knob, or sequential
   test. If a doubling decision needs an argument, it is not obvious;
   stay put.
4. **Hard cap: 16** samples per task (`DEFAULT_SAMPLE_CAP`). The closed-form
   surface stops there because within-task sample noise shrinks as
   `2 * within / num_samples` while the task-x-candidate interaction floor
   does not shrink at all — past the cap, more samples buy noise reduction
   that the interaction floor makes irrelevant. Spend additional budget on
   tasks (`n_tasks`), not samples.

The staged sequence is therefore 4 → 8 → 16, with each doubling a recorded
human decision, never an automatic escalation.
