"""Whetstone semantic provider layer over dr-providers transport.

This package owns the Whetstone-specific provider *semantics* that sit atop the
reusable dr-providers transport boundary:

* **Generation acceptance and the closed semantic failure taxonomy**
  (:mod:`whetstone.provider.classification`) — deterministic classification of
  every Provider Transport Outcome into a Generation or a
  Provider Semantic Failure.
* **Provider Execution Policy** (:mod:`whetstone.provider.policy`) — composes
  one Provider Transport Policy reference with bounded attempts, per-class
  retry eligibility, and deterministic backoff, duplicating no transport field.
* **Provider Call Attempt / Provider Call Result**
  (:mod:`whetstone.provider.attempt`) — the serializable logical-attempt
  wrapper and terminal semantic Result.
* **A pure, DBOS-free attempt-loop driver** (:mod:`whetstone.provider.driver`)
  — bounded, deterministic, with injectable transport, clock, and sleep hooks.
  The DBOS-durable executor lands in the next stage and wraps this loop.
"""

from __future__ import annotations

from whetstone.provider.attempt import (
    PROVIDER_CALL_ATTEMPT_SCHEMA,
    PROVIDER_CALL_ATTEMPT_SCHEMA_VERSION,
    PROVIDER_CALL_RESULT_SCHEMA,
    PROVIDER_CALL_RESULT_SCHEMA_VERSION,
    ProviderCallAttempt,
    ProviderCallResult,
)
from whetstone.provider.classification import (
    Generation,
    ProviderSemanticFailure,
    SemanticFailureClass,
    accept_generation,
    classify_outcome,
    is_blank,
)
from whetstone.provider.driver import TransportCall, run_provider_call
from whetstone.provider.policy import (
    PROVIDER_EXECUTION_POLICY_SCHEMA,
    PROVIDER_EXECUTION_POLICY_SCHEMA_VERSION,
    BackoffSchedule,
    ProviderExecutionPolicy,
    default_retry_eligibility,
)

__all__ = [
    "PROVIDER_CALL_ATTEMPT_SCHEMA",
    "PROVIDER_CALL_ATTEMPT_SCHEMA_VERSION",
    "PROVIDER_CALL_RESULT_SCHEMA",
    "PROVIDER_CALL_RESULT_SCHEMA_VERSION",
    "PROVIDER_EXECUTION_POLICY_SCHEMA",
    "PROVIDER_EXECUTION_POLICY_SCHEMA_VERSION",
    "BackoffSchedule",
    "Generation",
    "ProviderCallAttempt",
    "ProviderCallResult",
    "ProviderExecutionPolicy",
    "ProviderSemanticFailure",
    "SemanticFailureClass",
    "TransportCall",
    "accept_generation",
    "classify_outcome",
    "default_retry_eligibility",
    "is_blank",
    "run_provider_call",
]
