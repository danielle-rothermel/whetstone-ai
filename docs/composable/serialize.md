# Hashing + Serialization Extraction (dr-serialize)

Status: draft — high-level plan; sections will be expanded as design
discussion continues.

Goal: extract `dr_dspy/hashing.py` and `dr_dspy/serialization.py` into the
foundation package the other four libraries and the app depend on. Small,
dependency-free, stable.

## What the library is

- **Canonical JSON**: sorted-key, compact, NaN-rejecting serialization,
  and truncatable SHA-256 digests over JSON-able values (`canonical_json`,
  `sha256_json_digest`).
- **JSON-safe conversion engine**: best-effort `to_jsonable` /
  `to_metadata_dict` with an ordered, pluggable handler chain and
  depth/size guards.
- **Typed error taxonomy**: the `SerializationError` hierarchy with rich
  diagnostics (path-to-offending-value, previews).

## Lineage

- `hashing.py` is 100% generic today — zero internal imports; extract
  verbatim.
- `serialization.py` is a generic engine with two app accretions to strip
  at extraction:
  - **DSPy handlers** (`Example`, `BaseLM`, `Signature` summaries) — become
    handlers whetstone registers at import time via a public registration
    API.
  - **Postgres JSONB sizing constants** (`POSTGRES_JSONB_MAX_BYTES`, depth
    limits) — become a `SerializationLimits` config; Postgres ships as *a*
    preset, not *the* truth.
- `SANITIZE_KEYS` (api keys, auth headers) is LLM-credential domain — moves
  to dr-providers with the boundary code that uses it.

## Decisions

- **Handler registration is public API.** Ordered chain; the library ships
  stdlib + Pydantic handlers only. Consumers register their own (whetstone:
  DSPy; others as needed).
- **Limits are explicit config**, injected at the call site or carried by a
  configured encoder. Named presets provided (`POSTGRES_JSONB`). This must
  stay config, not constants: the platform's artifact offload
  (`platform.md`) changes what inline-payload ceilings consumers need.
- **Digest truncation lengths are caller-owned.** Identity contracts
  (whetstone's `records/hashing.py` axis names and lengths) are app-side
  frozen wire formats; this library provides only the mechanism.
- **The failure vocabulary does not live here.** `FailureClass` and the
  failure record live in dr-providers (see `llm_provider.md`); consumers
  that must not depend on dr-providers (graph runner) define structural
  `Protocol`s locally. Keeps this package scope-pure and dependency-free.

## What stays app-side

- `records/hashing.py` — frozen identity axes (a versioned wire format,
  not a utility).
- `eval_failures/recording.py` — the psycopg `Jsonb` /
  `FailureMetadataPayload` persistence bridge.
- DSPy handler registration.

## Anti-goals

- No storage, DB coupling, or compression.
- **Not a general utils package.** Scope test for any candidate addition:
  needed by at least two of the other four packages, and about canonical
  serialization or digests. A broader "foundations" scope invites
  dumping-ground drift — hence the narrow name.

## Naming

`dr-serialize` (decided): the scope is canonical serialization and digests
over serialized forms, and the name should keep it that way.

## Open sections (to fill in)

- Exact public API (module layout; free functions vs a configured encoder
  object).
- Version/stability policy — everything depends on this package, so
  breaking changes are the most expensive in the family.
- Whether `canonical_json`'s guarantees need a written cross-language spec
  (ties to the shared JSON contract question in `llm_provider.md` — a TS
  twin must hash identically).
