# Platform v6 refreshed baseline

The implementation stack is rooted on canonical Whetstone commit `23254e87b12dd16c173a396cd09326abe0708a1d`. The reviewed planning head `e8f3c60dea1d7470305d2f7aa6aecc81de7cf77a` contributes documentation and vocabulary only.

The canonical snapshot-loader work is retained as validated content provenance, with these final-hard-cut corrections:

- snapshot bytes and validated header/version fields define identity; paths are locators only;
- injected rows require an explicit matching snapshot identity;
- snapshot digest participates in scoring axes, selection, recipes, Score Attempt identity, relationships, acceptance, inspection, and publication;
- workflows durably carry stable IDs rather than database URLs or snapshot paths;
- execution resolves and verifies one expected content snapshot once; and
- the fresh final schema replaces the additive migration and fabricated unknown backfill.

The W0-W8 matrices must include same-bytes/different-paths, different-bytes/same-name, corrupt/header mismatch, mutation after registration, unavailable locator replay, mixed-snapshot acceptance rejection, and published pin stability.
