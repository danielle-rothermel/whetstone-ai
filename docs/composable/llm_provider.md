> **RETIRES AT D3 MERGE** — this documents the in-flight composable migration (draft PR #4) and becomes historical when it merges. Tracked in Linear S17/DEV-21.

# LLM Provider Library — Design Plan

Status: draft, high-level plan only. Sections will be expanded as design
discussion continues.

Goal: extract whetstone's LM provider boundary into a reusable library so
this repo keeps only its domain (records, schema, workflows) and consumes
providers as a dependency.

## Lineage

Four prior provider implementations inform this design:

- **dr-llm** (`../dr-llm/src/dr_llm/llm/providers/`) — full framework: 9
  providers, per-model control validation, headless CLI transports, usage
  and cost accounting. Worked, but the per-model control/legality rulebook
  rotted and made it unmaintainable.
- **dr-providers** (`../dr-providers/`) — minimal published package:
  OpenRouter-only, raw httpx, pure-data frozen request models, audit corpus
  with ground-truth parses. The right skeleton, missing features.
- **whetstone `lm/boundary.py`** (this repo) — operational fit: two endpoint
  kinds (chat + responses), payload-shape enums instead of legality rules,
  five-class failure taxonomy feeding DBOS retry/throttle, no in-client
  retry.
- **llmflow** (`../llmflow/`) — TypeScript event-sourced session runtime:
  providers emit typed event streams, errors as data records, per-provider
  capability declaration, fixture provider as a testing peer.

The stable kernel is the intersection of all four: request, response,
usage, warning, failure record, provider config, transport. Everything
that appeared in only one codebase (orchestrators, event sourcing,
sessions, tools) stays out of the library.

## Decisions

### Target: dr-providers v0.2, not a new package

Extend the existing published package with whetstone's boundary features
as a breaking release, consistent with its 0.1.x cleanup pattern.
*(from: dr-providers)*

### Python library, language-neutral contract

The kernel stays a Python library — whetstone is the paying consumer and
runs sync inside DBOS steps. The Pydantic models plus the audit corpus
(real payloads with expected parses) are the spec; a TS twin or thin HTTP
facade are later deployment modes, added when a real consumer exists.
"Own the endpoint" is a runtime property, deferred with the session
runtime. *(from: litellm's library-core/proxy-optional architecture;
llmflow's service layer)*

### Providers as config records, not classes

Provider differences are data: base URL, API key env var, endpoint kind
(chat / responses), reasoning request shape, token-limit parameter name.
No per-provider class hierarchies or registries until a genuinely
different wire shape lands (e.g. native Gemini `generateContent`).
*(from: dr-llm's collapse under per-provider class scaffolding; whetstone's
`ReasoningRequestShape` / `TokenLimitParameter` enums)*

Provider order: OpenRouter, OpenAI, then Gemini via Google's
OpenAI-compatible endpoint as a third config preset. Native Gemini API
becomes the first real second wire shape only if compat-endpoint gaps
(thinking budgets) require it. *(resolved: credits are AI Studio /
Gemini API key — a config preset, no new auth mode)*

### Raw httpx transport, no provider SDKs

The OpenAI SDK in the middle is why whetstone's failure classification
needs a hardcoded exception-name table. Raw httpx collapses
classification to transport exceptions + HTTP status codes in one place.
*(from: dr-providers, dr-llm `response_validation.py`)*

### Classify, don't retry

The library classifies failures precisely and never sleeps; callers own
retry policy. Whetstone's DBOS throttle needs this; an in-client retry is
invisible to it. Transport retry stays available but opt-in for
script/CLI use. *(from: whetstone platform; dr-providers' in-client retry
as the counterexample)*

### Failures as records, exceptions as carriers

A Pydantic failure record (failure class, code, message, retryable,
metadata) is the primary artifact; exceptions carry it. Adopt whetstone's
five-class `FailureClass` taxonomy — `RATE_LIMITED` stays distinct because
it drives throttle backoff. *(from: convergence of whetstone
`FailureSummary` and llmflow `LlmError`)*

**dr-providers is the canonical home of `FailureClass` and the failure
record** (resolved cross-doc question — see `overall.md`). The platform
library imports them (it already depends on this package); the graph
runner interoperates via a locally-defined structural `Protocol`
(`failure_class` / `error_type` / `metadata`) without depending on this
package. The `SANITIZE_KEYS` credential-redaction list moves here from
whetstone's serialization module (see `serialize.md`).

### Validation: structure before send, conformance after receive

Never predict per-model legality — that is the rulebook that rotted.

1. **Structural** — frozen models, `extra="forbid"`, no silent defaults:
   every knob is `None` (never serialized) or explicit. *(from:
   dr-providers)*
2. **Capabilities** — per-provider (not per-model) `supported_controls`
   declaration; loud pre-send error when a request sets a knob the
   provider config cannot transport. *(from: llmflow `ProviderInfo`)*
3. **Payload recording** — `build_payload()` is public and pure; the exact
   wire payload rides on the response so callers can persist it. *(from:
   the silent-defaults incident behind dr-llm's validation layers)*
4. **Conformance** — post-response checks against observed evidence
   (reasoning requested but zero reasoning tokens, ignored token caps,
   model substitution). Violations are warnings with severity; the caller
   decides what is fatal. *(from: dr-llm `ReasoningWarning` + llmflow
   `LlmWarning`; the gpt-nano silent-temperature incident)*
5. **Canary probes** — for knobs with no response echo (temperature): paid
   one-time probes per (provider, model), results committed to the audit
   corpus as measured capability records. *(from: dr-providers' audit
   corpus discipline)*

### Response as materialized parts

`LlmResponse` is composed of typed parts — text, usage, warnings, finish
reason, cost, provider metadata — so a future streaming mode can emit the
same parts incrementally without a breaking redesign. Port `TokenUsage`
(incl. reasoning-token extraction) and `CostInfo` from dr-llm. Streaming
itself is deferred. *(from: llmflow materializers; dr-llm `core/usage.py`)*

### Thread-aware, not stateful

The kernel stays stateless single-shot, but the nouns are thread-ready:
requests take a full transcript; responses carry an optional
**continuation handle** (Responses API `previous_response_id`, CLI session
IDs) for provider-native threads. Sessions, event logs, tool loops, and
approvals belong to a future separate runtime package that composes this
kernel with injected storage — durability substrates diverged in every
prior project (DBOS, Electric, RabbitMQ) and must not be baked in.
*(from: llmflow's providers/runtime package split; whetstone's DBOS
runtime)*

### Testing: fixture provider + audit corpus

Ship a `FixtureProvider` implementing the real interface (configurable
text/error/usage behavior) as public API, and grow the audit corpus with
whetstone's real OpenAI/OpenRouter response shapes. Parser changes are
regression-checked against corpus ground truth. *(from: llmflow
`FixtureProvider`; dr-providers audit corpus)*

## Whetstone-side impact

`lm/boundary.py` shrinks to a thin adapter: node config → `LlmRequest`,
`LlmResponse` → `ProviderResult`/records, stable `node_attempt_id` passed
as the idempotency key. Domain exceptions (`PredictionParseError`,
`Stranded*`), `require_generation_text`, and enc-dec field validators stay
here. `eval_failures/policy.py` drops its openai/httpx heuristic table and
keeps only psycopg/DBOS heuristics (which later move with the durable
batch-runner extraction).

## Deferred / out of scope

- Streaming and SSE (interface shaped to allow it later)
- Tool loops and approval gates (session runtime concern)
- Session/event-log runtime package (separate future design)
- Service endpoint / HTTP facade (deployment mode, when a web consumer exists)
- TypeScript twin (when a TS consumer needs in-process depth)

## Open questions

- Exact home and shape of the shared JSON contract / corpus so a future
  TS implementation can consume it.
- Conformance severity policy: which violations default to warnings vs
  errors at the library level. **Resolved at extraction (Stage 4b,
  2026-07-04), smallest conservative choice:** every conformance check
  (reasoning-not-observed, token-limit-exceeded, model-substitution)
  defaults to severity WARNING; nothing is fatal at the library level —
  the caller decides. Escalation policies can land later without a
  breaking change because severity is data on the warning record.
