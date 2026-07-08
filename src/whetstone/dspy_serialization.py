"""DSPy-aware serialization handlers for dr-serialize.

The handlers recognize DSPy values when the ``dspy`` package is installed and
fall through otherwise. Install them by using ``dspy_serializer()`` or passing
``DSPY_HANDLERS`` to a ``Serializer``.
"""

from __future__ import annotations

from typing import Any, ClassVar

from dr_serialize import (
    ConversionContext,
    JsonableHandle,
    JsonableHandler,
    JsonPath,
    SerializationError,
    Serializer,
    ValueTransformError,
    detail_repr,
    postgres_jsonb_limits,
    preview_repr,
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
    x: Any, ctx: ConversionContext
) -> JsonableHandle:
    try:
        dspy = _dspy_module()
    except ImportError:
        return False, None
    if isinstance(x, dspy.Example):
        try:
            return True, ctx.convert(x.toDict())
        except SerializationError:
            raise
        except Exception as error:
            raise ExampleSerializationError(
                path=ctx.path,
                underlying=error,
                value_preview=preview_repr(x),
                detail=detail_repr(x),
            ) from error
    return False, None


def jsonable_dspy_signature_type(
    x: Any, ctx: ConversionContext
) -> JsonableHandle:
    if not isinstance(x, type):
        return False, None
    try:
        dspy = _dspy_module()
        if issubclass(x, dspy.Signature):
            return True, _signature_summary(x, ctx.path)
    except ImportError:
        pass
    except TypeError:
        pass
    return False, None


def jsonable_dspy_lm(x: Any, ctx: ConversionContext) -> JsonableHandle:
    del ctx
    try:
        dspy = _dspy_module()
    except ImportError:
        return False, None
    if isinstance(x, dspy.BaseLM):
        # Lazy: importing whetstone.lm at module load would break the
        # graph/lm isolation contract (see tests/test_graph_imports.py).
        from whetstone.lm.utils import sanitize_lm_kwargs

        return True, {
            "_kind": "BaseLM",
            "class": f"{type(x).__module__}.{type(x).__name__}",
            "model": getattr(x, "model", None),
            "kwargs": sanitize_lm_kwargs(getattr(x, "kwargs", {})),
        }
    return False, None


DSPY_HANDLERS: tuple[JsonableHandler, ...] = (
    jsonable_dspy_example,
    jsonable_dspy_signature_type,
    jsonable_dspy_lm,
)


def dspy_serializer(max_bytes: int | None = None) -> Serializer:
    limits = (
        postgres_jsonb_limits()
        if max_bytes is None
        else postgres_jsonb_limits(max_bytes)
    )
    return Serializer(limits=limits, handlers=DSPY_HANDLERS)
