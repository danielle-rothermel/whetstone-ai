# Testing

Run unit tests with `./scripts/ci/unit.sh` and lint/type checks with
`./scripts/ci/lint.sh`. Integration tests use a disposable PostgreSQL schema:

```sh
DATABASE_URL=postgresql+psycopg:///dr_dspy ./scripts/ci/integration.sh
```

Before an experiment, run the publication-focused tests and verify that an
Analysis Bundle pin resolves all six members. Tests never require provider
credentials; external live-store validation is an isolated opt-in gate.
