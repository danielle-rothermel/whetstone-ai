from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Mapping
from types import MappingProxyType
from typing import Any, ClassVar, Self, TypeVar

from dr_serialize import (
    StrictJsonError,
    build_identity_document,
    canonical_json,
    identity_document_hash,
    validate_strict_json,
)
from dr_store import ObjectReference
from pydantic import BaseModel, ConfigDict, Field
from pydantic_core import CoreSchema, core_schema

__all__ = [
    "ContentHash",
    "FiniteFloat",
    "IdentityHash",
    "IdentityRef",
    "ImmutableJsonObject",
    "ImmutableJsonValue",
    "NonEmptyId",
    "NonNegativeInt",
    "OpaqueKey",
    "TerminalFailure",
    "TypedRef",
    "assert_materialized_ref_matches",
    "canonical_json_equal",
    "compute_identity_hash",
    "compute_prefixed_identity_key",
    "freeze_json_object",
    "identity_ref_from_config_variable",
    "reject_non_json",
    "require_full_hash",
    "store_config_resolver",
    "typed_ref_for_record",
]

_HEX = frozenset("0123456789abcdef")


def require_full_hash(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    if len(value) != 64 or any(char not in _HEX for char in value):
        raise ValueError(
            f"{field} must be a full 64-char lowercase SHA-256 hash, "
            f"got {value!r}"
        )
    return value


class _ValidatedString(str):
    _field_name: ClassVar[str]

    def __new__(cls, value: str) -> Self:
        if not isinstance(value, str):
            raise TypeError(f"{cls._field_name} must be a string")
        cls._validate(value)
        return str.__new__(cls, value)

    @classmethod
    def _validate(cls, value: str) -> None:
        raise NotImplementedError

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, _handler: Any
    ) -> CoreSchema:
        return core_schema.no_info_after_validator_function(
            cls,
            core_schema.str_schema(strict=True),
            serialization=core_schema.to_string_ser_schema(),
        )


class IdentityHash(_ValidatedString):
    _field_name = "identity hash"

    @classmethod
    def _validate(cls, value: str) -> None:
        require_full_hash(value, field=cls._field_name)


class ContentHash(_ValidatedString):
    _field_name = "content hash"

    @classmethod
    def _validate(cls, value: str) -> None:
        require_full_hash(value, field=cls._field_name)


class NonEmptyId(_ValidatedString):
    _field_name = "ID"

    @classmethod
    def _validate(cls, value: str) -> None:
        if not value:
            raise ValueError(f"{cls._field_name} must be non-empty")


class OpaqueKey(_ValidatedString):
    _field_name = "opaque key"

    @classmethod
    def _validate(cls, value: str) -> None:
        if not value:
            raise ValueError(f"{cls._field_name} must be non-empty")


class NonNegativeInt(int):
    def __new__(cls, value: int) -> Self:
        if type(value) is not int:
            raise TypeError("nonnegative integer must be a strict integer")
        if value < 0:
            raise ValueError("nonnegative integer cannot be negative")
        return int.__new__(cls, value)

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, _handler: Any
    ) -> CoreSchema:
        return core_schema.no_info_after_validator_function(
            cls,
            core_schema.int_schema(strict=True, ge=0),
        )


class FiniteFloat(float):
    def __new__(cls, value: int | float) -> Self:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("finite float must be a number, not a boolean")
        converted = float(value)
        if not math.isfinite(converted):
            raise ValueError("finite float must be finite")
        if converted == 0.0:
            converted = 0.0
        return float.__new__(cls, converted)

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, _handler: Any
    ) -> CoreSchema:
        return core_schema.no_info_after_validator_function(
            cls,
            core_schema.float_schema(strict=True, allow_inf_nan=False),
            serialization=core_schema.plain_serializer_function_ser_schema(
                float,
                return_schema=core_schema.float_schema(),
            ),
        )


type JsonScalar = bool | int | float | str | None
type ImmutableJsonValue = JsonScalar | Mapping[str, Any] | tuple[Any, ...]


def _freeze_validated_json(value: Any) -> ImmutableJsonValue:
    if isinstance(value, dict):
        return ImmutableJsonObject._from_validated(value)
    if isinstance(value, list):
        return tuple(_freeze_validated_json(item) for item in value)
    return value


def _json_value(value: ImmutableJsonValue) -> Any:
    if isinstance(value, ImmutableJsonObject):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value


class ImmutableJsonObject(Mapping[str, ImmutableJsonValue]):
    __slots__ = ("_items", "_lookup")

    _items: tuple[tuple[str, ImmutableJsonValue], ...]
    _lookup: Mapping[str, ImmutableJsonValue]

    def __init__(self, value: dict[str, Any]) -> None:
        try:
            validated = validate_strict_json(value)
        except StrictJsonError as exc:
            raise ValueError(
                "value must be strict finite JSON with string object keys: "
                f"{exc}"
            ) from exc
        if not isinstance(validated, dict):
            raise TypeError("immutable JSON object requires a JSON object")
        frozen = self._from_validated(validated)
        object.__setattr__(self, "_items", frozen._items)
        object.__setattr__(self, "_lookup", frozen._lookup)

    @classmethod
    def _from_validated(cls, value: dict[str, Any]) -> ImmutableJsonObject:
        instance = object.__new__(cls)
        items = tuple(
            (key, _freeze_validated_json(item)) for key, item in value.items()
        )
        object.__setattr__(instance, "_items", items)
        object.__setattr__(
            instance,
            "_lookup",
            MappingProxyType(dict(items)),
        )
        return instance

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("immutable JSON objects cannot be modified")

    def __getitem__(self, key: str) -> ImmutableJsonValue:
        return self._lookup[key]

    def __iter__(self) -> Iterator[str]:
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({dict(self)!r})"

    def __reduce__(
        self,
    ) -> tuple[type[ImmutableJsonObject], tuple[dict[str, Any]]]:
        return (type(self), (self.to_json(),))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, (dict, ImmutableJsonObject)):
            return False
        try:
            return canonical_json_equal(self, other)
        except (StrictJsonError, TypeError, ValueError):
            return False

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, _handler: Any
    ) -> CoreSchema:
        def validate(value: Any) -> ImmutableJsonObject:
            if isinstance(value, cls):
                return value
            if type(value) is not dict:
                raise TypeError("immutable JSON object requires a JSON object")
            return cls(value)

        return core_schema.no_info_plain_validator_function(
            validate,
            json_schema_input_schema=core_schema.dict_schema(
                keys_schema=core_schema.str_schema(strict=True)
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda value: _json_value(value),
                return_schema=core_schema.dict_schema(),
            ),
        )

    def to_json(self) -> dict[str, Any]:
        return _json_value(self)


def freeze_json_object(
    value: dict[str, Any] | ImmutableJsonObject,
    *,
    field: str,
) -> ImmutableJsonObject:
    if isinstance(value, ImmutableJsonObject):
        return value
    try:
        return ImmutableJsonObject(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field}: {exc}") from exc


def reject_non_json(value: Any, *, field: str) -> None:

    def unwrapped(item: Any) -> Any:
        if isinstance(item, ImmutableJsonObject):
            return {key: unwrapped(entry) for key, entry in item.items()}
        if isinstance(item, dict):
            return {key: unwrapped(entry) for key, entry in item.items()}
        if isinstance(item, (list, tuple)):
            return [unwrapped(entry) for entry in item]
        return item

    try:
        canonical_json(unwrapped(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field}: {exc}") from exc


def canonical_json_equal(left: Any, right: Any) -> bool:

    def normalized(value: Any) -> Any:
        if isinstance(value, ImmutableJsonObject):
            return value.to_json()
        if isinstance(value, tuple):
            return [normalized(item) for item in value]
        return value

    left_json = validate_strict_json(normalized(left))
    right_json = validate_strict_json(normalized(right))
    return canonical_json(left_json) == canonical_json(right_json)


def compute_identity_hash(
    *, schema: str, schema_version: int, payload: Any
) -> IdentityHash:
    document = build_identity_document(
        schema=schema, schema_version=schema_version, payload=payload
    )
    return IdentityHash(identity_document_hash(document))


def compute_prefixed_identity_key(
    *, schema: str, schema_version: int, prefix: str, payload: Any
) -> OpaqueKey:
    digest = compute_identity_hash(
        schema=schema, schema_version=schema_version, payload=payload
    )
    return OpaqueKey(f"{prefix}{digest}")


class TypedRef(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_name: NonEmptyId
    content_hash: ContentHash

    @property
    def reference(self) -> ObjectReference:
        return ObjectReference(
            schema=self.schema_name, content_hash=self.content_hash
        )


class IdentityRef(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    record_ref: TypedRef
    record_hash: IdentityHash


class TerminalFailure(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    code: NonEmptyId
    message: NonEmptyId
    details: ImmutableJsonObject = Field(
        default_factory=lambda: ImmutableJsonObject({})
    )


def typed_ref_for_record(schema: str, record: Any) -> TypedRef:
    reference = ObjectReference.for_record(schema, record)
    return TypedRef(
        schema_name=reference.schema, content_hash=reference.content_hash
    )


def assert_materialized_ref_matches(
    *,
    record: BaseModel | Mapping[str, Any],
    ref: IdentityRef,
    schema: str,
) -> None:
    if isinstance(record, BaseModel):
        content = record.model_dump(mode="json")
        identity_hash = getattr(record, "identity_hash", None)
    else:
        content = dict(record)
        identity_hash = content.get("identity_hash")
    materialized_ref = typed_ref_for_record(schema, content)
    if materialized_ref != ref.record_ref:
        raise ValueError(
            f"resolved {schema!r} record reference does not match IdentityRef"
        )
    valid_hashes = {str(materialized_ref.content_hash)}
    if identity_hash is not None:
        valid_hashes.add(str(identity_hash))
    if str(ref.record_hash) not in valid_hashes:
        raise ValueError(
            f"resolved {schema!r} record hash does not match IdentityRef"
        )


def identity_ref_from_config_variable(variable: Mapping[str, Any]) -> IdentityRef:
    schema_name = variable.get("schema_name")
    identity_hash = variable.get("identity_hash")
    if not isinstance(schema_name, str) or not schema_name.strip():
        raise ValueError("config reference schema_name must be non-empty")
    resolved_hash = require_full_hash(
        identity_hash, field="config reference identity_hash"
    )
    return IdentityRef(
        record_ref=TypedRef(
            schema_name=schema_name,
            content_hash=ContentHash(resolved_hash),
        ),
        record_hash=IdentityHash(resolved_hash),
    )


_T = TypeVar("_T", bound=BaseModel)


def store_config_resolver(
    store: Any,
    schema: str,
    model: type[_T],
) -> Callable[[IdentityRef], _T]:
    def resolve(ref: IdentityRef) -> _T:
        if ref.record_ref.schema_name != schema:
            raise ValueError(
                f"expected config schema {schema!r}, "
                f"got {ref.record_ref.schema_name!r}"
            )
        content = store.get(ref.record_ref.reference)
        if not isinstance(content, dict):
            raise ValueError("stored config content must be a JSON object")
        instance = model.model_validate(content)
        assert_materialized_ref_matches(record=instance, ref=ref, schema=schema)
        return instance

    return resolve
