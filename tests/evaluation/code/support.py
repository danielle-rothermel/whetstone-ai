"""Shared builders for the code-eval adapter tests."""

from __future__ import annotations

from dr_code.eval import OperatorLineage
from dr_providers import ProviderTransportResponse

from whetstone.provider.classification import Generation

FULL_HASH = "0" * 64


def transport_response(*, text: str) -> ProviderTransportResponse:
    return ProviderTransportResponse(
        text=text,
        raw_body={"choices": [{"message": {"content": text}}]},
        response_id="resp-1",
        model="test-model",
        finish_reason="stop",
    )


def generation(*, text: str) -> Generation:
    return Generation(text=text, response=transport_response(text=text))


def operator_lineage() -> OperatorLineage:
    return OperatorLineage(
        evaluation_procedure_config_hash=FULL_HASH,
        operator="compressed_length",
        operator_version="1",
        step=None,
        step_version=None,
    )
