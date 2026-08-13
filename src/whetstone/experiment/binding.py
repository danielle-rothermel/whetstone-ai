from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator

from whetstone.core.identity import (
    IdentityHash,
    TypedRef,
    typed_ref_for_record,
)
from whetstone.evaluation import SCHEMA_EVAL_CONFIG, EvalConfig

EVAL_CONFIG_RECORD_SCHEMA = SCHEMA_EVAL_CONFIG

__all__ = [
    "EVAL_CONFIG_RECORD_SCHEMA",
    "EvalConfigRef",
    "eval_config_reference",
]


class EvalConfigRef(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    record: EvalConfig
    record_ref: TypedRef
    config_hash: IdentityHash

    @model_validator(mode="after")
    def _validate(self) -> EvalConfigRef:
        if self.config_hash != self.record.config_hash:
            raise ValueError(
                "Eval Config identity_hash must match the exact typed record"
            )
        expected = typed_ref_for_record(
            EVAL_CONFIG_RECORD_SCHEMA, self.record.model_dump(mode="json")
        )
        if self.record_ref != expected:
            raise ValueError(
                "Eval Config record_ref must address the exact typed record"
            )
        return self


def eval_config_reference(eval_config: EvalConfig) -> EvalConfigRef:
    return EvalConfigRef(
        record=eval_config,
        record_ref=typed_ref_for_record(
            EVAL_CONFIG_RECORD_SCHEMA, eval_config.model_dump(mode="json")
        ),
        config_hash=eval_config.config_hash,
    )
