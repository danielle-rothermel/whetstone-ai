from __future__ import annotations

from whetstone.coordination.official import (
    SelectedRecordMapping,
    SelectedRecordMappingEntry,
)
from whetstone.core.identity import TypedRef
from whetstone.evaluation import DefinitionRef, EvalConfig
from whetstone.experiment.binding import (
    EvalConfigRef,
    eval_config_reference,
)

GRAPH_A = "a" * 64
GRAPH_B = "b" * 64
EVAL_HASH = "c" * 64


def eval_config_ref() -> EvalConfigRef:
    return eval_config_reference(
        EvalConfig(
            definition_ref=DefinitionRef(
                definition_id="authority-test",
                version="1",
                schema_name="whetstone.eval.definition",
                identity_hash="a" * 64,
            ),
            sampling_config_hash="b" * 64,
            evaluation_procedure_config_hash="d" * 64,
            aggregation_config_hash="e" * 64,
            config_hash=EVAL_HASH,
        )
    )


def full_hash(char: str) -> str:
    return char * 64


def content_ref(schema: str, char: str) -> TypedRef:
    return TypedRef(schema_name=schema, content_hash=full_hash(char))


def record_ref(char: str) -> TypedRef:
    return content_ref("whetstone.materialization_record", char)


def aggregate_ref(char: str) -> TypedRef:
    return content_ref("whetstone.aggregate", char)


def result_ref(char: str) -> TypedRef:
    return content_ref("whetstone.generation_result", char)


def oer_ref(char: str) -> TypedRef:
    return content_ref("whetstone.official_evaluation_record", char)


def mapping_entry(
    *,
    record_char: str,
    graph_hash: str,
    planned_keys: tuple[str, ...],
    result_keys: tuple[str, ...],
    aggregate_char: str,
) -> SelectedRecordMappingEntry:
    return SelectedRecordMappingEntry(
        record_ref=record_ref(record_char),
        graph_hash=graph_hash,
        planned_key_set=planned_keys,
        result_key_set=result_keys,
        aggregate_ref=aggregate_ref(aggregate_char),
    )


def single_entry_mapping(
    *,
    planned_keys: tuple[str, ...],
    result_keys: tuple[str, ...] | None = None,
) -> SelectedRecordMapping:
    return SelectedRecordMapping(
        entries=(
            mapping_entry(
                record_char="1",
                graph_hash=GRAPH_A,
                planned_keys=planned_keys,
                result_keys=(
                    planned_keys if result_keys is None else result_keys
                ),
                aggregate_char="9",
            ),
        )
    )
