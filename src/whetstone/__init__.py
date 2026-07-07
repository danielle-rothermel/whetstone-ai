"""Whetstone package initialization.

Registers whetstone's DSPy serialization handlers with dr-serialize so
any code path that serializes telemetry sees DSPy values handled.
"""

from whetstone.dspy_serialization import register_dspy_handlers

register_dspy_handlers()
