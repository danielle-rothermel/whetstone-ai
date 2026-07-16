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
