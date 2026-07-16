# Whetstone

Whetstone evaluates graph-shaped HumanEval workloads through the Whetstone
generation and scoring Operations. Domain outcomes are append-only; platform
execution is provided by `dr-platform`.

Run the local checks with:

```sh
./scripts/ci/unit.sh
./scripts/ci/lint.sh
DATABASE_URL=postgresql+psycopg:///dr_dspy ./scripts/ci/integration.sh
```

Publication has two explicit, independently promoted surfaces:

- the six-table Analysis Bundle: experiments, predictions, generation runs,
  score attempts, sweep metrics, and failure metrics;
- the root-cascaded Detail Bundle, including application-snapshot platform
  attempts.

Consumers pin a bundle before reading it.
