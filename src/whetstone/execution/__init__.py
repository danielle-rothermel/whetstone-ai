"""Execution primitives shared by the pilot and cell evaluation phases.

Leaf utilities with no dependency on :mod:`whetstone.envs` or the heavy
:mod:`whetstone.runner` cell/pilot modules, so both can import them without an
import cycle:

* :mod:`whetstone.execution.fanout` -- the bounded-concurrency worker pool with
  deterministic keyed assembly, a per-call runner guard, shared rate-limit
  backpressure, and a whole-run deadline.
* :mod:`whetstone.execution.partials` -- the append-only ``.partial.jsonl``
  per-call log the phases write incrementally and read on resume.
* :mod:`whetstone.execution.call_support` -- pure inspection helpers over a
  terminal Provider Call Result (failure code, rate-limit detection, guard
  deadline).
"""

from __future__ import annotations
