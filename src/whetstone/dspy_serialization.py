"""DSPy-aware serialization handlers registered with dr-serialize.

``register_dspy_handlers`` runs at whetstone package import (see
``whetstone/__init__.py``); the handlers recognize DSPy values when the
``dspy`` package is installed and fall through otherwise.
"""

from __future__ import annotations

from typing import Any, ClassVar

from dr_serialize import (
    JsonableHandle,
    JsonPath,
    SerializationError,
    ValueTransformError,
    convert_value,
    detail_repr,
    preview_repr,
    register_handler,
)


def _dspy_module() -> Any:
    import dspy

    return dspy


class SignatureSummaryError(ValueTransformError):
    message_prefix: ClassVar[str] = "signature summary failed"


class ExampleSerializationError(ValueTransformError):
    message_prefix: ClassVar[str] = "dspy.Example transform failed"


def _signature_summary(
    sig_cls: type[Any],
    path: JsonPath,
) -> dict[str, Any]:
    """Summarize a Signature class for logging."""
    try:
        fields_summary = [
            (
                name,
                str(field.annotation),
                (field.json_schema_extra or {}).get("__dspy_field_type")
                if isinstance(field.json_schema_extra, dict)
                else None,
            )
            for name, field in sig_cls.fields.items()
        ]
    except Exception as error:
        raise SignatureSummaryError(
            path=path,
            underlying=error,
            value_preview=preview_repr(sig_cls),
            detail=detail_repr(sig_cls),
        ) from error
    return {
        "signature": getattr(sig_cls, "signature", repr(sig_cls)),
        "instructions": getattr(sig_cls, "instructions", ""),
        "fields": fields_summary,
    }


def jsonable_dspy_example(
    x: Any, depth: int, path: JsonPath
) -> JsonableHandle:
    try:
        dspy = _dspy_module()
    except ImportError:
        return False, None
    if isinstance(x, dspy.Example):
        try:
            return True, convert_value(x.toDict(), depth + 1, path)
        except SerializationError:
            raise
        except Exception as error:
            raise ExampleSerializationError(
                path=path,
                underlying=error,
                value_preview=preview_repr(x),
                detail=detail_repr(x),
            ) from error
    return False, None


def jsonable_dspy_signature_type(
    x: Any, depth: int, path: JsonPath
) -> JsonableHandle:
    del depth
    if not isinstance(x, type):
        return False, None
    try:
        dspy = _dspy_module()
        if issubclass(x, dspy.Signature):
            return True, _signature_summary(x, path)
    except ImportError:
        pass
    except TypeError:
        pass
    return False, None


def jsonable_dspy_lm(x: Any, depth: int, path: JsonPath) -> JsonableHandle:
    del depth, path
    try:
        dspy = _dspy_module()
    except ImportError:
        return False, None
    if isinstance(x, dspy.BaseLM):
        # Lazy: package __init__ registers these handlers, and importing
        # whetstone.lm there would break the graph/lm isolation contract
        # (see tests/test_graph_imports.py).
        from whetstone.lm.utils import sanitize_lm_kwargs

        return True, {
            "_kind": "BaseLM",
            "class": f"{type(x).__module__}.{type(x).__name__}",
            "model": getattr(x, "model", None),
            "kwargs": sanitize_lm_kwargs(getattr(x, "kwargs", {})),
        }
    return False, None


def register_dspy_handlers() -> None:
    """Idempotently register whetstone's DSPy handlers with dr-serialize."""
    register_handler(jsonable_dspy_example)
    register_handler(jsonable_dspy_signature_type)
    register_handler(jsonable_dspy_lm)
