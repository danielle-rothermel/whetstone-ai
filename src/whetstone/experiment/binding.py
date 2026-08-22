"""Experiment-facing re-export of the Eval Config reference type.

The definitions live in `whetstone.eval.config_ref`, in the eval layer
beside `EvalConfig`. They are re-exported here because the binding of an
Eval Config to an experiment is an experiment-layer concern, and these
names are the established import site for the optim and experiment call
sites.
"""

from __future__ import annotations

from whetstone.eval.config_ref import (
    EVAL_CONFIG_RECORD_SCHEMA,
    EvalConfigRef,
    eval_config_reference,
)

__all__ = [
    "EVAL_CONFIG_RECORD_SCHEMA",
    "EvalConfigRef",
    "eval_config_reference",
]
