from __future__ import annotations

from typing import Any

from dr_dspy.lm.boundary import EndpointKind, ProviderKind
from dr_dspy.records.models import ProviderConfigRef


def provider_identity_key(
    ref: ProviderConfigRef,
) -> tuple[ProviderKind, EndpointKind, str, str]:
    return (
        ref.provider_kind,
        ref.endpoint_kind,
        ref.model,
        ref.throttle_key,
    )


def provider_configs_have_ambiguous_identity(
    configs: tuple[ProviderConfigRef, ...],
) -> bool:
    keys = [provider_identity_key(config) for config in configs]
    return len(keys) != len(set(keys))


def validate_provider_configs_identity(
    configs: tuple[ProviderConfigRef, ...],
) -> None:
    if not provider_configs_have_ambiguous_identity(configs):
        return
    for config in configs:
        if config.config_id is None:
            raise ValueError(
                "config_id is required on provider_configs entries when "
                "multiple configs share provider_kind, endpoint_kind, "
                "model, and throttle_key"
            )


def provider_snapshot_matches_axis(
    snapshot: dict[str, Any] | ProviderConfigRef,
    *,
    provider_kind: str,
    endpoint_kind: str,
    model: str,
    throttle_key: str,
    config_id: str | None,
) -> bool:
    if isinstance(snapshot, ProviderConfigRef):
        snapshot_config_id = snapshot.config_id
        snapshot_kind = snapshot.provider_kind.value
        snapshot_endpoint = snapshot.endpoint_kind.value
        snapshot_model = snapshot.model
        snapshot_throttle = snapshot.throttle_key
    else:
        snapshot_config_id = snapshot.get("config_id")
        snapshot_kind = snapshot.get("provider_kind")
        snapshot_endpoint = snapshot.get("endpoint_kind")
        snapshot_model = snapshot.get("model")
        snapshot_throttle = snapshot.get("throttle_key")
    return (
        snapshot_kind == provider_kind
        and snapshot_endpoint == endpoint_kind
        and snapshot_model == model
        and snapshot_throttle == throttle_key
        and snapshot_config_id == config_id
    )


def find_provider_config_ref(
    configs: tuple[ProviderConfigRef, ...],
    *,
    provider_kind: str,
    endpoint_kind: str,
    model: str,
    throttle_key: str,
    config_id: str | None,
) -> ProviderConfigRef:
    matches = [
        config
        for config in configs
        if provider_snapshot_matches_axis(
            config,
            provider_kind=provider_kind,
            endpoint_kind=endpoint_kind,
            model=model,
            throttle_key=throttle_key,
            config_id=config_id,
        )
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(
            "denormalized provider columns must match "
            "provider_configs snapshot"
        )
    raise ValueError(
        "denormalized provider columns match multiple provider_configs entries"
    )
